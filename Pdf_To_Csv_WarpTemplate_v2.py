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

# --- CONFIGURATION (成功版ベース + 最新ROI) ---
CONFIG = {
    # 外部ツールパス
    'POPPLER_PATH': r'C:\Program Files\poppler-25.11.0\Library\bin',
    'TESSERACT_CMD': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
   
    # 画像化設定
    'DPI': 300,
   
   
    # テンプレートPDF（単一レイアウト前提：この座標系に正規化します）
    # ★あなたの環境に合わせて変更してください
    'TEMPLATE_PDF': r'C:\gemini\Template.pdf',
    'ENABLE_WARP': True,
   
    # --- ワープ（罫線ベースECC） ---
    'ECC_TRY_HOMOGRAPHY_IF_BELOW': 0.45,
    'ECC_MIN_ACCEPT': 0.35,
    'ECC_MAX_ITER': 2000,
    'ECC_EPS': 1e-6,
    'MASK_ADAPTIVE_BLOCK': 51,
    'MASK_ADAPTIVE_C': 15,
    'MASK_LINE_KERNEL': 80,
    'MASK_DILATE': 7,
   
    # デバッグ
    'DEBUG_DIR': 'debug',
    'SAVE_WARPED_DEBUG': True,
    # 【アンカー設定】(成功版の設定を採用)
    'ANCHOR_BASE_RECT': (374, 759, 277, 42),
   
    # 【行ピッチ】 (成功版の値を採用)
    'ROW_PITCH_Y': 137.6,
    'MAX_ROWS': 8,

    # 【項目別ROI設定】 (最新の微調整済み座標を適用)
    'ROI_ITEMS': {
        # === 上段 (Y=0) ===
        'REV':       {'rect': (840, 2, 30, 40),   'type': 'rev'}, # 幅を少し狭めて罫線回避

        # === 中段 (Y=+44) ===
        '注番':      {'rect': (-124, 47, 120, 42), 'type': 'order_no'}, # 左に見切れ対策
        '部品番号':  {'rect': (2, 47, 267, 42),    'type': 'text'},
        '附属書番号': {'rect': (487, 47, 229, 42),  'type': 'appendix'},
        '供給元':    {'rect': (963, 47, 562, 42),  'type': 'supplier'},
        '数量':      {'rect': (1670, 48, 90, 40), 'type': 'integer'},  
        '加工費':    {'rect': (2123, 48, 369, 41), 'type': 'number'},

        # === 下段 (Y=+88) ===
        '名称':      {'rect': (2, 94, 474, 42),    'type': 'text'},
        '納期':      {'rect': (1595, 94, 200, 41), 'type': 'date'},    
        '工番・分番': {'rect': (1805, 94, 311, 42), 'type': 'job_no'},    
        '材料費':    {'rect': (2123, 94, 369, 42), 'type': 'number'},      
    },
     
    # OCR設定 (成功版の設定 + 日本語対応を追加)
    'TESS_CONFIG_ANCHOR': r'--psm 7 -c tessedit_char_whitelist="F7ALL0123456789"',
    'TESS_CONFIG_NUM':    r'--psm 6 -c tessedit_char_whitelist="0123456789.,"',
    
    # 変更後: numeric_mode を削除し、PSMを6 (ブロック) に変更
    'TESS_CONFIG_INT': (
        r'--oem 1 --psm 6 -l eng '
        r'-c tessedit_char_whitelist=0123456789'
    ),

    'TESS_CONFIG_TEXT':   r'--psm 6',
    'TESS_CONFIG_REV':    r'--psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
    'TESS_CONFIG_ALPHANUM': r'--psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"',
    'TESS_CONFIG_JPN':    r'--psm 7 -l jpn',
}

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pytesseract.pytesseract.tesseract_cmd = CONFIG['TESSERACT_CMD']

def preprocess_image(cv_image):
    """
    成功版のロジックを採用: 二値化 + Deskew (Hough変換)
    """
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
   
    # 二値化 (Otsu)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   
    # Deskew (Hough変換による傾き補正)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
   
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < a < 45:
                angles.append(a)
       
        if angles:
            angle = np.median(angles)
   
    # 傾き補正実行
    if abs(angle) > 0.1:
        logging.info(f"Deskewing angle: {angle:.2f}")
        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv_image = cv2.warpAffine(cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
       
    return cv_image, gray

def preprocess_for_ocr(roi_img):
    """
    【追加機能】切り抜いたROI専用の前処理
    拡大 + 余白追加でOCR精度を高める
    """
    # 1. 拡大 (2倍) - 小さい文字の潰れ防止
    # INTER_CUBIC は文字の滑らかさを保つのに適しています
    roi_large = cv2.resize(roi_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
   
    # 2. 二値化 (Otsu)
    _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    #   === ★追加: 膨張処理 (文字を太くする) ===
    # 2x2のカーネルで1回膨張させ、細い「1」を太く強調する
    kernel = np.ones((2, 2), np.uint8)
    roi_bin = cv2.erode(roi_bin, kernel, iterations=1) 
    # 注: OpenCVでは白黒が逆（白地に黒文字）の場合、erodeで黒文字が太くなります。
    # Tesseractは通常「白地に黒文字」を好みますが、
    # 二値化の結果が「黒地に白文字」になっている場合は dilate を使ってください。
    # ※ otsuの結果次第ですが、一般的に紙スキャンなら「白地に黒文字」なので erode で太ります。
    # ======================================  

    # 3. 余白追加 (Padding) - 画像の端にある文字を読みやすくする
    # 上下左右に10ピクセルの白フチを追加
    roi_padded = cv2.copyMakeBorder(
        roi_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
   
    return roi_padded


# ============================================================
# テンプレートへワープして座標系を揃える（罫線ベースECC）
# ============================================================
def _load_pdf_first_page_as_cv(pdf_path):
    images = convert_from_path(
        pdf_path,
        first_page=1,
        last_page=1,
        dpi=CONFIG['DPI'],
        poppler_path=CONFIG['POPPLER_PATH'],
    )
    if not images:
        raise RuntimeError(f"Template PDF has no pages: {pdf_path}")
    cv_img = np.array(images[0])
    return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

def _build_line_mask_from_template(template_gray):
    """
    文字の差分ではなく「罫線（レイアウト）」に寄せて合わせるためのマスクを作る
    """
    block = int(CONFIG.get('MASK_ADAPTIVE_BLOCK', 51))
    if block % 2 == 0:
        block += 1
    c = int(CONFIG.get('MASK_ADAPTIVE_C', 15))

    bin_inv = cv2.adaptiveThreshold(
        template_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, c
    )

    k = int(CONFIG.get('MASK_LINE_KERNEL', 80))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))

    h_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)

    lines = cv2.bitwise_or(h_lines, v_lines)

    d = int(CONFIG.get('MASK_DILATE', 7))
    kernel = np.ones((d, d), np.uint8)
    mask = cv2.dilate(lines, kernel, iterations=1)
    mask = (mask > 0).astype(np.uint8) * 255
    return mask

def _normalize_for_ecc(gray):
    # ECCはfloat32で濃淡パターンを合わせるので、軽くノイズを抑えて正規化
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    g = cv2.equalizeHist(g)
    return (g.astype(np.float32) / 255.0)

def _resize_to_hw(img, th, tw):
    h, w = img.shape[:2]
    if (h, w) == (th, tw):
        return img
    interp = cv2.INTER_AREA if (w > tw or h > th) else cv2.INTER_LINEAR
    return cv2.resize(img, (tw, th), interpolation=interp)

def warp_page_to_template(target_color, target_gray, template_gray, template_mask):
    """
    ページごとのズレ・歪み・サイズ差をテンプレートへワープして補正する
    戻り値: aligned_color, aligned_gray, score, mode
    """
    th, tw = template_gray.shape[:2]

    # サイズが微妙に違う時は、いったんテンプレートサイズへ正規化
    tgt_color = _resize_to_hw(target_color, th, tw)
    tgt_gray  = _resize_to_hw(target_gray, th, tw)

    tmpl_f = _normalize_for_ecc(template_gray)
    tgt_f  = _normalize_for_ecc(tgt_gray)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(CONFIG.get('ECC_MAX_ITER', 2000)),
        float(CONFIG.get('ECC_EPS', 1e-6))
    )

    # まずAFFINE（平行移動・回転・拡大縮小・せん断）
    warp_aff = np.eye(2, 3, dtype=np.float32)
    try:
        cc_aff, warp_aff = cv2.findTransformECC(
            tmpl_f, tgt_f, warp_aff,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=template_mask,
            gaussFiltSize=5
        )
        aligned_gray = cv2.warpAffine(
            tgt_gray, warp_aff, (tw, th),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )
        aligned_color = cv2.warpAffine(
            tgt_color, warp_aff, (tw, th),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )

        # 台形歪みまで取りたい場合はHOMOGRAPHYを追加で試す
        if cc_aff < float(CONFIG.get('ECC_TRY_HOMOGRAPHY_IF_BELOW', 0.45)):
            warp_h = np.eye(3, 3, dtype=np.float32)
            try:
                cc_h, warp_h = cv2.findTransformECC(
                    tmpl_f, tgt_f, warp_h,
                    cv2.MOTION_HOMOGRAPHY,
                    criteria,
                    inputMask=template_mask,
                    gaussFiltSize=5
                )
                aligned_gray_h = cv2.warpPerspective(
                    tgt_gray, warp_h, (tw, th),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE
                )
                aligned_color_h = cv2.warpPerspective(
                    tgt_color, warp_h, (tw, th),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE
                )
                if cc_h > cc_aff:
                    return aligned_color_h, aligned_gray_h, float(cc_h), "homography"
            except cv2.error:
                pass

        return aligned_color, aligned_gray, float(cc_aff), "affine"
    except cv2.error:
        # ECCが失敗したら「サイズ正規化のみ」にフォールバック
        return tgt_color, tgt_gray, 0.0, "fallback"

def check_anchor(gray_img, rect):
    x, y, w, h = rect
    img_h, img_w = gray_img.shape
    if x < 0 or y < 0 or x+w > img_w or y+h > img_h:
        return False, ""

    roi = gray_img[y:y+h, x:x+w]
    # アンカー判定は従来通り
    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   
    text = pytesseract.image_to_string(roi_bin, config=CONFIG['TESS_CONFIG_ANCHOR'])
    text = text.strip().upper()
   
    # 判定緩和 ('FALL'等も許容)
    is_match = ('F7' in text) or ('F?' in text) or ('F1' in text) or ('FALL' in text)
   
    return is_match, text

def find_correct_orientation(cv_image):
    """成功版の回転ロジックを維持"""
    ax, ay, aw, ah = CONFIG['ANCHOR_BASE_RECT']
   
    for angle in [0, 90, 180, 270]:
        temp_img = cv_image.copy()
        if angle != 0:
            if angle == 90: temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180: temp_img = cv2.rotate(temp_img, cv2.ROTATE_180)
            elif angle == 270: temp_img = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
       
        processed_color, processed_gray = preprocess_image(temp_img)
        match, text = check_anchor(processed_gray, (ax, ay, aw, ah))
        logging.info(f"Angle {angle}: Anchor Text='{text}', Match={match}")
       
        if match:
            return processed_color, processed_gray
           
    logging.warning("Anchor not found in any orientation. Using default (0).")
    return preprocess_image(cv_image)

def clean_text(text, type_):
    """最新の整形ロジックを適用"""
    text = text.strip()
   
    if type_ == 'number':
        text = re.sub(r'[^\d\.,\-]', '', text)
    elif type_ == 'integer':
        # ★追加: 整数専用クリーニング (カンマ・ピリオドも除去)
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
        # I->1, O->0 補正
        text = text.replace('I', '1').replace('O', '0').replace('Z', '2')
        clean = re.sub(r'[^A-Z0-9]', '', text)
        if len(clean) > 5: text = clean[-5:]
        else: text = clean
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
    
    # --- テンプレート読み込み（単一レイアウト前提） ---
    template_gray = None
    template_mask = None
    if template_pdf is None:
        template_pdf = CONFIG.get('TEMPLATE_PDF')
    if CONFIG.get('ENABLE_WARP', True) and template_pdf and os.path.exists(template_pdf):
        try:
            templ_cv = _load_pdf_first_page_as_cv(template_pdf)
            templ_cv, template_gray = preprocess_image(templ_cv)
            template_mask = _build_line_mask_from_template(template_gray)
            logging.info(f"Loaded template: {template_pdf}  size={template_gray.shape[1]}x{template_gray.shape[0]}")
        except Exception as e:
            logging.warning(f"Template load failed -> disable warp. reason={e}")
            template_gray = None
            template_mask = None
    else:
        if CONFIG.get('ENABLE_WARP', True):
            logging.warning("Template not found or warp disabled. Running without warp.")
   
    try:
        images = convert_from_path(pdf_path, dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
    except Exception as e:
        logging.error(f"PDF Conversion Failed: {e}")
        return []

    for page_idx, img_pil in enumerate(images):
        logging.info(f"Processing Page {page_idx + 1}")
        cv_img = np.array(img_pil)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
       
        # 向き補正・前処理
        color_img, gray_img = find_correct_orientation(cv_img)
        
        # --- テンプレートへワープして座標系を揃える ---
        if template_gray is not None and template_mask is not None:
            aligned_color, aligned_gray, score, mode = warp_page_to_template(color_img, gray_img, template_gray, template_mask)
            logging.info(f"Alignment: mode={mode} score={score:.4f}")
            if score < float(CONFIG.get('ECC_MIN_ACCEPT', 0.35)):
                logging.warning(f"Alignment score low ({score:.4f}). Use non-warp (resize only).")
                th, tw = template_gray.shape[:2]
                color_img = _resize_to_hw(color_img, th, tw)
                gray_img = _resize_to_hw(gray_img, th, tw)
            else:
                color_img, gray_img = aligned_color, aligned_gray
        
        debug_img = color_img.copy()
       
        ax, ay, aw, ah = CONFIG['ANCHOR_BASE_RECT']
        pitch = CONFIG['ROW_PITCH_Y']
       
        for row_idx in range(CONFIG['MAX_ROWS']):
            curr_ax = ax
            curr_ay = int(ay + (row_idx * pitch))
           
            is_anchor, anchor_text = check_anchor(gray_img, (curr_ax, curr_ay, aw, ah))
           
            color = (0, 255, 0) if is_anchor else (0, 0, 255)
            cv2.rectangle(debug_img, (curr_ax, curr_ay), (curr_ax+aw, curr_ay+ah), color, 2)
           
            if not is_anchor:
                # 【修正】 アンカーが見つからない場合、このページの処理を終了 (break)
                logging.info(f"Row {row_idx+1}: Anchor mismatch. Stop processing this page.")
                break
            
            row_data = {}
            # 機種名補正
            model_name = anchor_text.replace('?', '7').replace('1', '7') if anchor_text else "F7 ALL"
            if "FALL" in model_name: model_name = "F7 ALL"
            row_data['機種'] = model_name

            # 強化前処理を適用する項目のリスト
            ENHANCED_ITEMS = ['部品番号', '附属書番号', '注番', '数量', '工番・分番', '材料費', '加工費','納期']

            for key, setting in CONFIG['ROI_ITEMS'].items():
                dx, dy, w, h = setting['rect']
                rx = curr_ax + dx
                ry = curr_ay + dy
               
                if ry+h > gray_img.shape[0] or rx+w > gray_img.shape[1]: continue

                roi = gray_img[ry:ry+h, rx:rx+w]
               
                # 指定項目だけ preprocess_for_ocr を適用
                if key in ENHANCED_ITEMS:
                    processed_roi = preprocess_for_ocr(roi)
                else:
                    # それ以外は単純な二値化
                    _, processed_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ocr_conf = CONFIG['TESS_CONFIG_TEXT']
                if setting['type'] == 'number':   ocr_conf = CONFIG['TESS_CONFIG_NUM']
                elif setting['type'] == 'integer': ocr_conf = CONFIG['TESS_CONFIG_INT'] # ★追加
                elif setting['type'] == 'rev':    ocr_conf = CONFIG['TESS_CONFIG_REV']
                elif setting['type'] in ['order_no', 'appendix', 'job_no']: ocr_conf = CONFIG['TESS_CONFIG_ALPHANUM']
                elif setting['type'] == 'supplier': ocr_conf = CONFIG['TESS_CONFIG_JPN']
               
                raw_text = pytesseract.image_to_string(processed_roi, config=ocr_conf)
                clean_val = clean_text(raw_text, setting['type'])
                row_data[key] = clean_val
               
                cv2.rectangle(debug_img, (rx, ry), (rx+w, ry+h), (255, 0, 0), 1)
           
            output_data.append(row_data)
            logging.info(f" Extracted Row {row_idx+1}")

        cv2.imwrite(str(debug_dir / f"debug_p{page_idx+1}.jpg"), debug_img)

    return output_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("target_pdf", nargs="?", default="target.pdf", help="Input PDF (default: target.pdf)")
    parser.add_argument("--template", default=None, help="Template PDF path (default: CONFIG['TEMPLATE_PDF'])")
    parser.add_argument("--out", default="output_ocr.csv", help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.target_pdf):
        print(f"Error: {args.target_pdf} not found.")
        raise SystemExit(1)

    results = process_pdf(args.target_pdf, template_pdf=args.template)
    if results:
        df = pd.DataFrame(results)
        if 'REV.' in df.columns:
            df.rename(columns={'REV.': 'REV'}, inplace=True)

        # F7エクセルの列定義に合わせる
        target_columns = [
            '機種', 'REV', '受注額', '部品番号', '附属書番号', '供給元',
            '注番', '名称', '数量', '納期', '工番・分番', '材料費', '加工費'
        ]
        for col in target_columns:
            if col not in df.columns:
                df[col] = ""

        df[target_columns].to_csv(args.out, index=False, encoding="utf-8-sig", errors="ignore")
        print(f"Done. Saved to {args.out}")
    else:
        print("No data extracted.")
