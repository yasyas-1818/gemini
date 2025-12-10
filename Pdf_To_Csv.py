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
    'POPPLER_PATH': r'C:\gemini\poppler\Library\bin',
    'TESSERACT_CMD': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
   
    # 画像化設定
    'DPI': 300,
   
    # 【アンカー設定】(成功版の設定を採用)
    'ANCHOR_BASE_RECT': (372, 753, 277, 41),
   
    # 【行ピッチ】 (成功版の値を採用)
    'ROW_PITCH_Y': 137.6,
    'MAX_ROWS': 8,

    # 【項目別ROI設定】 (最新の微調整済み座標を適用)
    'ROI_ITEMS': {
        # === 上段 (Y=0) ===
        'REV':       {'rect': (840, 0, 30, 38),   'type': 'rev'}, # 幅を少し狭めて罫線回避

        # === 中段 (Y=+44) ===
        '注番':      {'rect': (-125, 47, 120, 40), 'type': 'order_no'}, # 左に見切れ対策
        '部品番号':  {'rect': (1, 47, 267, 40),    'type': 'text'},
        '附属書番号': {'rect': (485, 46, 229, 40),  'type': 'appendix'},
        '供給元':    {'rect': (962, 46, 562, 40),  'type': 'supplier'},
        '数量':      {'rect': (1670, 47, 90, 37), 'type': 'integer'},  
        '加工費':    {'rect': (2120, 46, 369, 40), 'type': 'number'},

        # === 下段 (Y=+88) ===
        '名称':      {'rect': (0, 92, 474, 40),    'type': 'text'},
        '納期':      {'rect': (1590, 92, 200, 40), 'type': 'date'},    
        '工番・分番': {'rect': (1803, 92, 311, 40), 'type': 'job_no'},    
        '材料費':    {'rect': (2120, 92, 369, 40), 'type': 'number'},      
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

def process_pdf(pdf_path):
    output_data = []
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
   
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
                # ログのみ出力して続行
                logging.warning(f"Row {row_idx+1}: Anchor mismatch. Proceeding...")
           
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
    target_pdf = "target.pdf"
   
    if not os.path.exists(target_pdf):
        print(f"Error: {target_pdf} not found.")
    else:
        results = process_pdf(target_pdf)
        if results:
            df = pd.DataFrame(results)
            if 'REV.' in df.columns: df.rename(columns={'REV.': 'REV'}, inplace=True)
           
            # F7エクセルの列定義に合わせる
            target_columns = [
                '機種', 'REV', '受注額', '部品番号', '附属書番号', '供給元',
                '注番', '名称', '数量', '納期', '工番・分番', '材料費', '加工費'
            ]
            for col in target_columns:
                if col not in df.columns: df[col] = ""
           
            output_csv = "output_ocr.csv"
            df[target_columns].to_csv(output_csv, index=False, encoding="cp932", errors="ignore")
            print(f"Done. Saved to {output_csv}")
        else:
            print("No data extracted.")