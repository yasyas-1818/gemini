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

# --- CONFIGURATION ---
CONFIG = {
    # 外部ツールパス
    'POPPLER_PATH': r'C:\Program Files\poppler-25.11.0\Library\bin',
    'TESSERACT_CMD': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    
    # ★位置合わせ用テンプレート
    'TEMPLATE_PDF': 'Template.pdf', 
   
    # 画像化設定
    'DPI': 300,
   
    # 【アンカー設定】(既存設定)
    'ANCHOR_BASE_RECT': (374, 759, 277, 42),
    'ROW_PITCH_Y': 137.6,
    'MAX_ROWS': 8,

    # 【項目別ROI設定】 (既存設定)
    'ROI_ITEMS': {
        'REV':       {'rect': (840, 2, 30, 40),   'type': 'rev'}, 
        '注番':      {'rect': (-124, 47, 120, 42), 'type': 'order_no'}, 
        '部品番号':  {'rect': (2, 47, 267, 42),    'type': 'text'},
        '附属書番号': {'rect': (487, 47, 229, 42),  'type': 'appendix'},
        '供給元':    {'rect': (963, 47, 562, 42),  'type': 'supplier'},
        '数量':      {'rect': (1670, 48, 90, 40), 'type': 'integer'},  
        '加工費':    {'rect': (2123, 49, 370, 40), 'type': 'number'},
        '名称':      {'rect': (2, 94, 474, 42),    'type': 'text'},
        '納期':      {'rect': (1595, 94, 200, 41), 'type': 'date'},    
        '工番・分番': {'rect': (1805, 94, 311, 42), 'type': 'job_no'},    
        '材料費':    {'rect': (2123, 95, 369, 40), 'type': 'number'},      
    },
     
    # OCR設定
    'TESS_CONFIG_ANCHOR': r'--psm 7 -c tessedit_char_whitelist="F7ALL0123456789"',
    'TESS_CONFIG_NUM':    r'--psm 6 -c tessedit_char_whitelist="0123456789.,"',
    'TESS_CONFIG_INT':    r'--oem 1 --psm 6 -l eng -c tessedit_char_whitelist=0123456789',
    'TESS_CONFIG_TEXT':   r'--psm 6',
    'TESS_CONFIG_REV':    r'--psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
    'TESS_CONFIG_ALPHANUM': r'--psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"',
    'TESS_CONFIG_JPN':    r'--psm 7 -l jpn',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pytesseract.pytesseract.tesseract_cmd = CONFIG['TESSERACT_CMD']

# ==========================================
# ★高速化版: 回転対応の位置合わせ関数
# ==========================================
def align_image_robust(target_img_rgb, template_gray):
    """
    高速化版: 画像を縮小して特徴点マッチングを行い、計算結果を元画像に適用する。
    これにより処理時間を数秒→0.5秒程度に短縮する。
    """
    h_orig, w_orig = template_gray.shape
    
    # --- 1. 高速化のための縮小処理 (スケール 0.25倍 = 面積比1/16) ---
    scale = 0.25 
    w_small = int(w_orig * scale)
    h_small = int(h_orig * scale)
    
    # テンプレートとターゲットを縮小
    template_small = cv2.resize(template_gray, (w_small, h_small))
    target_gray_orig = cv2.cvtColor(target_img_rgb, cv2.COLOR_BGR2GRAY)
    target_small = cv2.resize(target_gray_orig, (w_small, h_small))

    # --- 2. 特徴点検出 (縮小画像で実行) ---
    detector = cv2.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(template_small, None)
    kp2, des2 = detector.detectAndCompute(target_small, None)

    if des1 is None or des2 is None:
        logging.warning("  -> Not enough features found. Skipping alignment.")
        return target_img_rgb

    # --- 3. マッチング ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 上位15%を採用
    good_matches = matches[:int(len(matches) * 0.15)]
    
    if len(good_matches) < 4:
        logging.warning("  -> Not enough good matches. Skipping alignment.")
        return target_img_rgb

    # --- 4. 変換行列(ホモグラフィ)の計算 (縮小画像基準) ---
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M_small, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if M_small is None:
        logging.warning("  -> Homography calculation failed.")
        return target_img_rgb

    # --- 5. 行列のスケール補正 (小さい画像用Mを、大きい画像用に変換) ---
    # 行列計算: H_large = inv(S) * H_small * S
    # ここで S は「大きい座標」を「小さい座標」にするスケーリング行列
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    S_inv = np.array([[1.0/scale, 0, 0], [0, 1.0/scale, 0], [0, 0, 1]])
    
    # M_large = S_inv @ M_small @ S
    M_large = S_inv.dot(M_small).dot(S)

    # --- 6. 元の高解像度画像を変形 ---
    aligned_img = cv2.warpPerspective(target_img_rgb, M_large, (w_orig, h_orig))
    
    logging.info(f"  -> Robust alignment applied (Fast Mode). Matches: {len(good_matches)}")
    
    return aligned_img
# --- テンプレート準備関数 ---
def prepare_template_from_file():
    if not os.path.exists(CONFIG['TEMPLATE_PDF']):
        logging.error(f"Template PDF not found: {CONFIG['TEMPLATE_PDF']}")
        return None
    
    logging.info("Loading Template PDF...")
    try:
        images = convert_from_path(CONFIG['TEMPLATE_PDF'], dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
        if not images: return None
        
        # テンプレート全体を使用（特徴点マッチングには情報量が多いほうが良いため）
        # ただし、可変領域（日付など）の影響を避けるため、マスク済みのTemplate.pdfの使用を推奨
        template_cv = np.array(images[0])
        template_gray = cv2.cvtColor(template_cv, cv2.COLOR_RGB2GRAY)
        return template_gray
    except Exception as e:
        logging.error(f"Failed to load template: {e}")
        return None

# --- OCR用ヘルパー関数群 (変更なし) ---
def clean_text(text, type_):
    text = text.strip()
    if type_ == 'number': text = re.sub(r'[^\d\.,\-]', '', text)
    elif type_ == 'integer': text = re.sub(r'[^\d\-]', '', text)
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
        if len(clean) > 5: text = clean[-5:]
        else: text = clean
    elif type_ == 'job_no': text = text.replace(" ", "").replace("O", "0")
    elif type_ == 'appendix': text = re.sub(r'[^A-Z0-9\-]', '', text)
    elif type_ == 'supplier':
        text = unicodedata.normalize('NFKC', text)
        text = text.replace(' ', '')
    else:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_for_ocr(roi_img):
    roi_large = cv2.resize(roi_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    roi_bin = cv2.erode(roi_bin, kernel, iterations=1) 
    roi_padded = cv2.copyMakeBorder(roi_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return roi_padded

def check_anchor(gray_img, rect):
    x, y, w, h = rect
    img_h, img_w = gray_img.shape
    if x < 0 or y < 0 or x+w > img_w or y+h > img_h: return False, ""
    roi = gray_img[y:y+h, x:x+w]
    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(roi_bin, config=CONFIG['TESS_CONFIG_ANCHOR'])
    text = text.strip().upper()
    is_match = ('F7' in text) or ('F?' in text) or ('F1' in text) or ('FALL' in text)
    return is_match, text

# --- メイン処理 ---
def process_pdf(pdf_path):
    output_data = []
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
   
    # 1. テンプレート画像を読み込み (Template.pdf)
    template_gray = prepare_template_from_file()
    if template_gray is None:
        print("Error: Could not load template. Aborting.")
        return []

    try:
        images = convert_from_path(pdf_path, dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
    except Exception as e:
        logging.error(f"PDF Conversion Failed: {e}")
        return []

    for page_idx, img_pil in enumerate(images):
        logging.info(f"Processing Page {page_idx + 1}")
        cv_img = np.array(img_pil)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
       
        # ★ここで回転対応の補正を実行
        # Targetのページを、Template.pdfに合わせて変形(傾き補正+移動)させる
        cv_img = align_image_robust(cv_img, template_gray)
        
        # OCR用のグレースケール画像作成
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        debug_img = cv_img.copy() # デバッグ用画像

        # 以降は既存のロジック (補正済み画像に対して座標を指定)
        ax, ay, aw, ah = CONFIG['ANCHOR_BASE_RECT']
        pitch = CONFIG['ROW_PITCH_Y']
       
        for row_idx in range(CONFIG['MAX_ROWS']):
            curr_ax = ax
            curr_ay = int(ay + (row_idx * pitch))
           
            is_anchor, anchor_text = check_anchor(gray_img, (curr_ax, curr_ay, aw, ah))
           
            color = (0, 255, 0) if is_anchor else (0, 0, 255)
            cv2.rectangle(debug_img, (curr_ax, curr_ay), (curr_ax+aw, curr_ay+ah), color, 2)
           
            if not is_anchor:
                # アンカーが見つからない場合はそのページ終了
                # (補正が成功していれば、データがある限り見つかるはず)
                logging.info(f"Row {row_idx+1}: Anchor mismatch (End of data or bad align).")
                break
            
            row_data = {}
            model_name = anchor_text.replace('?', '7').replace('1', '7') if anchor_text else "F7 ALL"
            if "FALL" in model_name: model_name = "F7 ALL"
            row_data['機種'] = model_name

            ENHANCED_ITEMS = ['部品番号', '附属書番号', '注番', '数量', '工番・分番', '材料費', '加工費','納期']

            for key, setting in CONFIG['ROI_ITEMS'].items():
                dx, dy, w, h = setting['rect']
                rx = curr_ax + dx
                ry = curr_ay + dy
               
                if ry+h > gray_img.shape[0] or rx+w > gray_img.shape[1]: continue

                roi = gray_img[ry:ry+h, rx:rx+w]
                if key in ENHANCED_ITEMS: processed_roi = preprocess_for_ocr(roi)
                else: _, processed_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ocr_conf = CONFIG['TESS_CONFIG_TEXT']
                if setting['type'] == 'number':   ocr_conf = CONFIG['TESS_CONFIG_NUM']
                elif setting['type'] == 'integer': ocr_conf = CONFIG['TESS_CONFIG_INT']
                elif setting['type'] == 'rev':    ocr_conf = CONFIG['TESS_CONFIG_REV']
                elif setting['type'] in ['order_no', 'appendix', 'job_no']: ocr_conf = CONFIG['TESS_CONFIG_ALPHANUM']
                elif setting['type'] == 'supplier': ocr_conf = CONFIG['TESS_CONFIG_JPN']
               
                raw_text = pytesseract.image_to_string(processed_roi, config=ocr_conf)
                clean_val = clean_text(raw_text, setting['type'])
                row_data[key] = clean_val
               
                cv2.rectangle(debug_img, (rx, ry), (rx+w, ry+h), (255, 0, 0), 1)
           
            output_data.append(row_data)
            logging.info(f" Extracted Row {row_idx+1}")

        # 結果確認用画像の保存
        cv2.imwrite(str(debug_dir / f"debug_p{page_idx+1}_corrected.jpg"), debug_img)

    return output_data

if __name__ == "__main__":
    target_pdf = "target.pdf"
   
    if not os.path.exists(target_pdf):
        print(f"Error: {target_pdf} not found.")
    else:
        results = process_pdf(target_pdf)
        if results:
            df = pd.DataFrame(results)
            if 'REV.' in df.columns: df.rename(columns={'REV.': 'REV'}, inplace=True)
           
            target_columns = [
                '機種', 'REV', '受注額', '部品番号', '附属書番号', '供給元',
                '注番', '名称', '数量', '納期', '工番・分番', '材料費', '加工費'
            ]
            for col in target_columns:
                if col not in df.columns: df[col] = ""
           
            output_csv = "output_ocr.csv"
            df[target_columns].to_csv(output_csv, index=False, encoding="utf-8-sig", errors="ignore")
            print(f"Done. Saved to {output_csv}")
        else:
            print("No data extracted.")