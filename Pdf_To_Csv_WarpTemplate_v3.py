import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import pandas as pd
import os
import re
import logging
import unicodedata
from pathlib import Path
from pytesseract import Output

# --- CONFIGURATION (v4: フェイルセーフ強化版) ---
CONFIG = {
    # 外部ツールパス
    'POPPLER_PATH': r'C:\Program Files\poppler-25.11.0\Library\bin',
    'TESSERACT_CMD': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
   
    'DPI': 300,
    'TEMPLATE_PDF': r'C:\gemini\Template.pdf',
    'ENABLE_WARP': True,
   
    # --- ワープ設定 ---
    'ECC_MIN_ACCEPT': 0.35, # これ以下ならリサイズのみ
    
    # デバッグ
    'DEBUG_DIR': 'debug',
    
    # 【アンカー設定】
    'ANCHOR_BASE_RECT': (374, 759, 277, 42),
    'ANCHOR_SEARCH_MARGIN': 50, # 探索範囲を少し拡大
   
    # 【行ピッチ】
    'ROW_PITCH_Y': 137.6,
    'MAX_ROWS': 10, # 念のため多めに

    # 【項目別ROI設定】
    'ROI_ITEMS': {
        # === 上段 ===
        'REV':       {'rect': (840, 2, 30, 40),   'type': 'rev'}, 
        # === 中段 ===
        '注番':      {'rect': (-124, 47, 120, 42), 'type': 'order_no'},
        '部品番号':  {'rect': (2, 47, 267, 42),    'type': 'text'},
        '附属書番号': {'rect': (487, 47, 229, 42),  'type': 'appendix'},
        '供給元':    {'rect': (963, 47, 562, 42),  'type': 'supplier'},
        '数量':      {'rect': (1670, 48, 90, 40), 'type': 'integer'},  
        '加工費':    {'rect': (2123, 48, 369, 41), 'type': 'number'},
        # === 下段 ===
        '名称':      {'rect': (2, 94, 474, 42),    'type': 'text'},
        '納期':      {'rect': (1595, 94, 200, 41), 'type': 'date'},    
        '工番・分番': {'rect': (1805, 94, 311, 42), 'type': 'job_no'},    
        '材料費':    {'rect': (2123, 94, 369, 42), 'type': 'number'},      
    },
     
    # OCR設定
    # ★変更: アンカー用は制限を緩める（PSM 11: Sparse Text, ホワイトリストなし）
    'TESS_CONFIG_ANCHOR': r'--psm 11', 

    'TESS_CONFIG_NUM':    r'--psm 6 -c tessedit_char_whitelist="0123456789.,"',
    'TESS_CONFIG_INT':    r'--oem 1 --psm 6 -l eng -c tessedit_char_whitelist=0123456789',
    'TESS_CONFIG_TEXT':   r'--psm 6',
    'TESS_CONFIG_REV':    r'--psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
    'TESS_CONFIG_ALPHANUM': r'--psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"',
    'TESS_CONFIG_JPN':    r'--psm 7 -l jpn',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pytesseract.pytesseract.tesseract_cmd = CONFIG['TESSERACT_CMD']

# --- 前処理関数群 ---
def preprocess_image(cv_image):
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Deskew
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < a < 45: angles.append(a)
        if angles: angle = np.median(angles)
   
    if abs(angle) > 0.1:
        logging.info(f"Deskewing angle: {angle:.2f}")
        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv_image = cv2.warpAffine(cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
       
    return cv_image, gray

def preprocess_for_ocr(roi_img):
    roi_large = cv2.resize(roi_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 文字を太らせる
    kernel = np.ones((2, 2), np.uint8)
    roi_bin = cv2.erode(roi_bin, kernel, iterations=1) 
    roi_padded = cv2.copyMakeBorder(roi_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return roi_padded

# --- Warp関連 ---
def _load_pdf_first_page_as_cv(pdf_path):
    images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
    cv_img = np.array(images[0])
    return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

def _build_line_mask_from_template(template_gray):
    bin_inv = cv2.adaptiveThreshold(template_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
    lines = cv2.bitwise_or(cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel), cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel))
    mask = cv2.dilate(lines, np.ones((7, 7), np.uint8), iterations=1)
    return (mask > 0).astype(np.uint8) * 255

def warp_page_to_template(target_color, target_gray, template_gray, template_mask):
    th, tw = template_gray.shape[:2]
    tgt_gray  = cv2.resize(target_gray, (tw, th), interpolation=cv2.INTER_LINEAR)
    
    scale = 0.35
    sw, sh = int(tw * scale), int(th * scale)
    tmpl_s = cv2.resize(template_gray, (sw, sh))
    tgt_s  = cv2.resize(tgt_gray, (sw, sh))
    mask_s = cv2.resize(template_mask, (sw, sh), interpolation=cv2.INTER_NEAREST) if template_mask is not None else None

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-4)
    
    try:
        _, warp_matrix = cv2.findTransformECC(tmpl_s, tgt_s, warp_matrix, cv2.MOTION_AFFINE, criteria, inputMask=mask_s, gaussFiltSize=3)
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale
        aligned_gray = cv2.warpAffine(tgt_gray, warp_matrix, (tw, th), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        aligned_color = cv2.warpAffine(cv2.resize(target_color, (tw, th)), warp_matrix, (tw, th), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return aligned_color, aligned_gray, 1.0, "affine"
    except Exception:
        return cv2.resize(target_color, (tw, th)), tgt_gray, 0.0, "fallback"

# --- アンカー探索 ---
def find_anchor_and_offset(gray_img, rect):
    x, y, w, h = rect
    img_h, img_w = gray_img.shape
    
    margin = CONFIG.get('ANCHOR_SEARCH_MARGIN', 40)
    search_y = max(0, y - margin)
    search_h = h + (margin * 2)
    if search_y + search_h > img_h: search_h = img_h - search_y
    
    roi = gray_img[search_y : search_y + search_h, x : x + w]
    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # image_to_data で詳細取得
    d = pytesseract.image_to_data(roi_bin, config=CONFIG['TESS_CONFIG_ANCHOR'], output_type=Output.DICT)
    
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        text = d['text'][i].strip().upper()
        # 判定条件: F7かALLかF?などが含まれていればOK
        if len(text) > 1 and (('F7' in text) or ('ALL' in text) or ('F?' in text) or ('FALL' in text)):
            text_y = d['top'][i]
            text_h = d['height'][i]
            
            # 期待値とのズレを計算
            expected_center_y = margin + (h / 2)
            actual_center_y = text_y + (text_h / 2)
            offset_y = actual_center_y - expected_center_y
            
            return True, int(offset_y), text
            
    return False, 0, ""

def clean_text(text, type_):
    text = text.strip()
    if type_ == 'number':
        text = re.sub(r'[^\d\.,\-]', '', text)
    elif type_ == 'integer':
        text = re.sub(r'[^\d\-]', '', text)
    elif type_ == 'date':
        text = text.replace(' ', '')
        match = re.search(r'(\d{2})[/-](\d{1,2})[/-](\d{1,2})', text)
        if match:
            yy, mm, dd = match.groups()
            text = f"20{yy}/{mm.zfill(2)}/{dd.zfill(2)}"
    elif type_ == 'rev':
        match = re.search(r'[A-Z]', text)
        text = match.group(0) if match else ""
    elif type_ == 'order_no':
        text = text.replace('I', '1').replace('O', '0').replace('Z', '2')
        clean = re.sub(r'[^A-Z0-9]', '', text)
        text = clean[-5:] if len(clean) > 5 else clean
    elif type_ == 'job_no':
        text = text.replace(" ", "").replace("O", "0")
    elif type_ == 'appendix':
        text = re.sub(r'[^A-Z0-9\-]', '', text)
    elif type_ == 'supplier':
        text = unicodedata.normalize('NFKC', text)
        text = text.replace(' ', '')
    else:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
    return text

def process_pdf(pdf_path, template_pdf=None):
    output_data = []
    debug_dir = Path(CONFIG.get('DEBUG_DIR', 'debug'))
    debug_dir.mkdir(exist_ok=True)
    
    # テンプレート読み込み
    template_gray = None
    template_mask = None
    if template_pdf is None: template_pdf = CONFIG.get('TEMPLATE_PDF')
    
    if CONFIG.get('ENABLE_WARP', True) and template_pdf and os.path.exists(template_pdf):
        try:
            templ_cv = _load_pdf_first_page_as_cv(template_pdf)
            templ_cv, template_gray = preprocess_image(templ_cv)
            template_mask = _build_line_mask_from_template(template_gray)
            logging.info(f"Loaded template: {template_pdf}")
        except Exception as e:
            logging.warning(f"Template load failed: {e}")

    try:
        images = convert_from_path(pdf_path, dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
    except Exception as e:
        logging.error(f"PDF Conversion Failed: {e}")
        return []

    for page_idx, img_pil in enumerate(images):
        logging.info(f"Processing Page {page_idx + 1}")
        cv_img = np.array(img_pil)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        color_img, gray_img = preprocess_image(cv_img) # 簡易前処理
        
        # ECC Warp
        if template_gray is not None:
            logging.info('Starting ECC warp...')
            aligned_color, aligned_gray, score, mode = warp_page_to_template(color_img, gray_img, template_gray, template_mask)
            logging.info(f"Alignment: mode={mode} score={score:.4f}")
            if score >= float(CONFIG.get('ECC_MIN_ACCEPT', 0.35)):
                color_img, gray_img = aligned_color, aligned_gray
            else:
                logging.warning("Alignment score low. Fallback to resize.")
                th, tw = template_gray.shape[:2]
                color_img = cv2.resize(color_img, (tw, th))
                gray_img = cv2.resize(gray_img, (tw, th))
        
        debug_img = color_img.copy()
        ax, ay, aw, ah = CONFIG['ANCHOR_BASE_RECT']
        pitch = CONFIG['ROW_PITCH_Y']
        
        # ページ内ループ用のオフセット保持変数
        current_page_offset = 0
       
        for row_idx in range(CONFIG['MAX_ROWS']):
            base_ay = int(ay + (row_idx * pitch))
            
            # アンカー探索
            is_anchor, offset_y, anchor_text = find_anchor_and_offset(gray_img, (ax, base_ay, aw, ah))
            
            # デバッグ描画（探索エリア）
            margin = CONFIG.get('ANCHOR_SEARCH_MARGIN', 40)
            cv2.rectangle(debug_img, (ax, base_ay - margin), (ax+aw, base_ay+ah+margin), (0, 255, 255), 1)

            # === ロジック変更: アンカーが見つからなくても、データがあれば続行する ===
            if is_anchor:
                current_page_offset = offset_y # 新しいオフセットで更新
                logging.info(f"Row {row_idx+1}: Found Anchor '{anchor_text}' Offset={offset_y:+d}")
                color = (0, 255, 0)
            else:
                # 見つからない場合は「前回のオフセット」を使って強制続行してみる
                logging.warning(f"Row {row_idx+1}: Anchor NOT found. Using last offset={current_page_offset:+d} and checking content...")
                offset_y = current_page_offset
                color = (0, 0, 255) # 赤枠
            
            corrected_ay = base_ay + offset_y
            cv2.rectangle(debug_img, (ax, corrected_ay), (ax+aw, corrected_ay+ah), color, 2)

            # データ取得
            row_data = {}
            row_data['機種'] = "F7 ALL"

            ENHANCED_ITEMS = ['部品番号', '附属書番号', '注番', '数量', '工番・分番', '材料費', '加工費','納期']
            
            # 部品番号の中身を確認するための一時変数
            part_no_val = ""

            for key, setting in CONFIG['ROI_ITEMS'].items():
                dx, dy, w, h = setting['rect']
                rx = ax + dx
                ry = corrected_ay + dy # 補正座標を使用
               
                if ry+h > gray_img.shape[0] or rx+w > gray_img.shape[1]: continue

                roi = gray_img[ry:ry+h, rx:rx+w]
               
                if key in ENHANCED_ITEMS:
                    processed_roi = preprocess_for_ocr(roi)
                else:
                    _, processed_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ocr_conf = CONFIG['TESS_CONFIG_TEXT']
                if setting['type'] == 'number':    ocr_conf = CONFIG['TESS_CONFIG_NUM']
                elif setting['type'] == 'integer': ocr_conf = CONFIG['TESS_CONFIG_INT']
                elif setting['type'] == 'rev':     ocr_conf = CONFIG['TESS_CONFIG_REV']
                elif setting['type'] in ['order_no', 'appendix', 'job_no']: ocr_conf = CONFIG['TESS_CONFIG_ALPHANUM']
                elif setting['type'] == 'supplier': ocr_conf = CONFIG['TESS_CONFIG_JPN']
               
                raw_text = pytesseract.image_to_string(processed_roi, config=ocr_conf)
                clean_val = clean_text(raw_text, setting['type'])
                row_data[key] = clean_val
                
                if key == '部品番号':
                    part_no_val = clean_val
               
                cv2.rectangle(debug_img, (rx, ry), (rx+w, ry+h), (255, 0, 0), 1)
            
            # === 終了判定 ===
            # アンカーも見つからず、かつ部品番号も空っぽなら、本当に表が終わったとみなす
            if not is_anchor and part_no_val == "":
                logging.info(f"Row {row_idx+1}: End of table detected (Empty part number). Breaking.")
                break
            
            output_data.append(row_data)

        cv2.imwrite(str(debug_dir / f"debug_p{page_idx+1}.jpg"), debug_img)

    return output_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_pdf", nargs="?", default="target.pdf")
    parser.add_argument("--template", default=None)
    parser.add_argument("--out", default="output_ocr.csv")
    args = parser.parse_args()

    if not os.path.exists(args.target_pdf):
        print(f"Error: {args.target_pdf} not found.")
        raise SystemExit(1)

    results = process_pdf(args.target_pdf, template_pdf=args.template)
    if results:
        df = pd.DataFrame(results)
        if 'REV' not in df.columns and 'REV.' in df.columns:
            df.rename(columns={'REV.': 'REV'}, inplace=True)

        target_columns = [
            '機種', 'REV', '受注額', '部品番号', '附属書番号', '供給元',
            '注番', '名称', '数量', '納期', '工番・分番', '材料費', '加工費'
        ]
        for col in target_columns:
            if col not in df.columns: df[col] = ""

        df[target_columns].to_csv(args.out, index=False, encoding="utf-8-sig", errors="ignore")
        print(f"Done. Saved to {args.out}")
    else:
        print("No data extracted.")