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
    # 外部ツールパス (環境に合わせて変更してください)
    'POPPLER_PATH': r'C:\Program Files\poppler-25.11.0\Library\bin',
    'TESSERACT_CMD': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
   
    'DPI': 300,
   
    # テンプレートファイル名 (可変部分を白塗りしたPDF推奨)
    'TEMPLATE_PDF': 'toMatch.pdf',  # 修正: きれいなテンプレートファイル名を指定

    # 【アンカー設定】(toMatch.pdfに合わせた設定)
    'ANCHOR_BASE_RECT': (372, 753, 277, 41),
    'ROW_PITCH_Y': 137.6,
    'MAX_ROWS': 8,

    # 【項目別ROI設定】
    'ROI_ITEMS': {
        'REV':       {'rect': (840, 0, 30, 38),   'type': 'rev'},
        '注番':      {'rect': (-125, 47, 120, 40), 'type': 'order_no'},
        '部品番号':  {'rect': (1, 47, 267, 40),    'type': 'text'},
        '附属書番号': {'rect': (485, 46, 229, 40),  'type': 'appendix'},
        '供給元':    {'rect': (962, 46, 562, 40),  'type': 'supplier'},
        '数量':      {'rect': (1670, 47, 90, 37), 'type': 'integer'},  
        '加工費':    {'rect': (2120, 46, 369, 40), 'type': 'number'},
        '名称':      {'rect': (0, 92, 474, 40),    'type': 'text'},
        '納期':      {'rect': (1590, 92, 200, 40), 'type': 'date'},    
        '工番・分番': {'rect': (1803, 92, 311, 40), 'type': 'job_no'},    
        '材料費':    {'rect': (2120, 92, 369, 40), 'type': 'number'},      
    },
     
    # OCR設定
    'TESS_CONFIG_ANCHOR': r'--psm 7 -c tessedit_char_whitelist="F7ALL0123456789"',
    'TESS_CONFIG_NUM':    r'--oem 1 --psm 7 -l eng -c tessedit_char_whitelist="0123456789,."',
    'TESS_CONFIG_NUM_ALT': r'--oem 1 --psm 8 -l eng -c tessedit_char_whitelist="0123456789,."',
    'TESS_CONFIG_INT':    r'--oem 1 --psm 6 -l eng -c tessedit_char_whitelist=0123456789',
    'TESS_CONFIG_TEXT':   r'--psm 6',
    'TESS_CONFIG_REV':    r'--psm 10 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"',
    'TESS_CONFIG_ALPHANUM': r'--psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"',
    'TESS_CONFIG_JPN':    r'--psm 7 -l jpn',
}

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pytesseract.pytesseract.tesseract_cmd = CONFIG['TESSERACT_CMD']

def preprocess_image(cv_image):
    """基本の前処理（Deskew含む）"""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 傾き検出 (Hough変換)
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
    
    # 傾き補正
    if abs(angle) > 0.1:
        # logging.info(f"Deskewing angle: {angle:.2f}") # ログがうるさければコメントアウト
        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv_image = cv2.warpAffine(cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
    return cv_image, gray

def _to_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edge image for robust template matching."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blur, 50, 150)

def _get_template_patches(w: int, h: int):
    """Return a list of (name, (x, y, w, h)) in template coordinates.
    Patches are chosen to avoid variable fields (dates / order numbers) as much as possible.
    """
    patches = []
    # Center title area (e.g., 注文書 / 確定価格注文)
    patches.append((
        "title",
        (int(w * 0.30), int(h * 0.02), int(w * 0.40), int(h * 0.14))
    ))
    # Left-center label area (stable Japanese labels)
    patches.append((
        "labels",
        (int(w * 0.05), int(h * 0.12), int(w * 0.45), int(h * 0.12))
    ))
    return patches

def align_image_to_template(img_target, img_template):
    """Template-matching based translation alignment.

    Fixes:
    1) Avoid full-width ROI (cannot detect X shift when ROI width == target width).
    2) Avoid crashing on 1px size differences by using smaller patches and safe guards.
    3) Compute shift using the patch's original template coordinates (x0,y0), not assuming (0,0).
    """
    h_templ, w_templ = img_template.shape[:2]
    h_tgt, w_tgt = img_target.shape[:2]
    logging.info(f"Alignment Start: Template({w_templ}x{h_templ}) vs Target({w_tgt}x{h_tgt})")

    # If size differs significantly, normalize by resizing (keeps alignment logic simple).
    # For 1px-ish differences we keep original to avoid unnecessary distortion.
    work_target = img_target
    if abs(w_tgt - w_templ) > 5 or abs(h_tgt - h_templ) > 5:
        work_target = cv2.resize(img_target, (w_templ, h_templ), interpolation=cv2.INTER_AREA)
        h_tgt, w_tgt = work_target.shape[:2]
        logging.info(f"Target resized for alignment: ({w_tgt}x{h_tgt})")

    gray_target = cv2.cvtColor(work_target, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    # Search area: top part of the target (where header/title lives)
    search_h = min(int(h_tgt * 0.40), int(h_templ * 0.45), h_tgt)
    search = gray_target[0:search_h, :]
    search_edges = _to_edges(search)

    best = None  # (score, found_x, found_y, patch_x0, patch_y0, name)
    for name, (x0, y0, pw, ph) in _get_template_patches(w_templ, h_templ):
        # Clip patch inside template bounds
        x0c = max(0, min(x0, w_templ - 1))
        y0c = max(0, min(y0, h_templ - 1))
        pwc = max(20, min(pw, w_templ - x0c))
        phc = max(20, min(ph, h_templ - y0c))

        patch = gray_template[y0c:y0c+phc, x0c:x0c+pwc]
        patch_edges = _to_edges(patch)

        # Guard: patch must be smaller than search
        if patch_edges.shape[0] >= search_edges.shape[0] or patch_edges.shape[1] >= search_edges.shape[1]:
            continue

        res = cv2.matchTemplate(search_edges, patch_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        fx, fy = max_loc

        if (best is None) or (max_val > best[0]):
            best = (max_val, fx, fy, x0c, y0c, name)

    if best is None:
        logging.warning("Alignment failed (no valid patch). Returning resized/original image without shift.")
        aligned_img = cv2.resize(work_target, (w_templ, h_templ), interpolation=cv2.INTER_AREA) if work_target.shape[1] != w_templ or work_target.shape[0] != h_templ else work_target
        aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        return aligned_img, aligned_gray

    score, found_x, found_y, patch_x0, patch_y0, patch_name = best
    # Compute translation so that the found patch position matches its template position.
    shift_x = patch_x0 - found_x
    shift_y = patch_y0 - found_y
    logging.info(
        f"Alignment Patch='{patch_name}' score={score:.4f} found=({found_x},{found_y}) templ=({patch_x0},{patch_y0}) -> dx={shift_x}, dy={shift_y}"
    )

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_img = cv2.warpAffine(work_target, M, (w_templ, h_templ), borderValue=(255, 255, 255))
    aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
    return aligned_img, aligned_gray

def preprocess_for_ocr(roi_img):
    """ROI専用前処理（文字・記号系の汎用）

    ※材料費/加工費は別関数 preprocess_amount_roi を使う（罫線が強く、太らせ処理が逆効果になりやすい）
    """
    # 1) 拡大
    roi_large = cv2.resize(roi_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 2) 二値化 (Otsu)
    _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 3) 文字を少し太らせる（細い「1」対策）
    #    ※表の罫線セル（材料費/加工費）には使わない
    kernel = np.ones((2, 2), np.uint8)
    roi_bin = cv2.erode(roi_bin, kernel, iterations=1)

    # 4) 余白追加
    roi_padded = cv2.copyMakeBorder(roi_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return roi_padded

def preprocess_amount_roi(gray_roi):
    """Preprocess for amount cells (材料費/加工費): remove table lines + OCR friendly binarization."""
    roi = cv2.resize(gray_roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold tends to work better on scanned cells
    roi_bin = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )

    inv = 255 - roi_bin  # for line extraction

    # Extract horizontal/vertical lines (kernel sizes relative to ROI size)
    h_len = max(12, inv.shape[1] // 12)
    v_len = max(12, inv.shape[0] // 2)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    lines = cv2.bitwise_or(h_lines, v_lines)

    inv_clean = cv2.subtract(inv, lines)
    cleaned = 255 - inv_clean  # back to black-on-white

    cleaned = cv2.copyMakeBorder(cleaned, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return cleaned

def check_anchor(gray_img, rect):
    """
    【修正】成功版(Pdf_To_Csv.py)と同じロジックに戻す
    アライメント済みなので、拡大やパディングなどの過剰な加工は不要。
    """
    x, y, w, h = rect
    img_h, img_w = gray_img.shape
    
    # 画像外参照ガード
    if x < 0 or y < 0 or x+w > img_w or y+h > img_h:
        return False, ""

    roi = gray_img[y:y+h, x:x+w]
    
    # シンプルな二値化のみ (成功版と同じ)
    _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   
    text = pytesseract.image_to_string(roi_bin, config=CONFIG['TESS_CONFIG_ANCHOR'])
    text = text.strip().upper()
   
    is_match = ('F7' in text) or ('F?' in text) or ('F1' in text) or ('FALL' in text)
   
    return is_match, text

def find_correct_orientation(cv_image, template_img):
    """Detect orientation (0/90/180/270) using patch-based template matching.

    Fix:
    - Avoid full-width ROI matching (it becomes very weak and is sensitive to small size diffs).
    """
    if template_img is None:
        return preprocess_image(cv_image)

    h_templ, w_templ = template_img.shape[:2]
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    # Prepare template patches (edges)
    templ_patches = []
    for name, (x0, y0, pw, ph) in _get_template_patches(w_templ, h_templ):
        x0c = max(0, min(x0, w_templ - 1))
        y0c = max(0, min(y0, h_templ - 1))
        pwc = max(20, min(pw, w_templ - x0c))
        phc = max(20, min(ph, h_templ - y0c))
        patch = gray_template[y0c:y0c+phc, x0c:x0c+pwc]
        templ_patches.append((name, _to_edges(patch)))

    best_score = -1.0
    best_img = cv_image

    for angle in [0, 90, 180, 270]:
        temp_img = cv_image
        if angle != 0:
            if angle == 90:
                temp_img = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                temp_img = cv2.rotate(cv_image, cv2.ROTATE_180)
            elif angle == 270:
                temp_img = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        search_h = min(int(gray.shape[0] * 0.40), int(h_templ * 0.45), gray.shape[0])
        search = gray[0:search_h, :]
        search_edges = _to_edges(search)

        angle_best = -1.0
        for name, patch_edges in templ_patches:
            if patch_edges.shape[0] >= search_edges.shape[0] or patch_edges.shape[1] >= search_edges.shape[1]:
                continue
            res = cv2.matchTemplate(search_edges, patch_edges, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            angle_best = max(angle_best, max_val)

        # logging.info(f"Angle {angle}: match score={angle_best:.4f}")
        if angle_best > best_score:
            best_score = angle_best
            best_img = temp_img

    if best_score > 0.20:
        logging.info(f"Orientation Match Score: {best_score:.4f}")
        return preprocess_image(best_img)

    logging.warning("Orientation detection failed (Low Score). Using default.")
    return preprocess_image(cv_image)

def clean_text(text, type_):
    """テキスト整形"""
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
        text = unicodedata.normalize('NFKC', text).replace(' ', '')
    else:
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
    return text

def load_template_image():
    """テンプレートPDFの1ページ目を読み込んで画像化する"""
    template_path = CONFIG['TEMPLATE_PDF']
    if not os.path.exists(template_path):
        logging.error(f"Template file '{template_path}' not found.")
        return None
    
    try:
        logging.info(f"Loading template from {template_path}...")
        images = convert_from_path(template_path, first_page=1, last_page=1, dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
        if not images:
            return None
        
        # OpenCV形式に変換
        templ_cv = np.array(images[0])
        templ_cv = cv2.cvtColor(templ_cv, cv2.COLOR_RGB2BGR)
        
        # テンプレート自体も、座標定義時と同じ状態(Deskew済み)にするのが望ましい
        # ここでは自己参照を防ぐため、単純なDeskew（preprocess_image）のみ呼ぶ
        templ_deskewed, _ = preprocess_image(templ_cv)
        return templ_deskewed

    except Exception as e:
        logging.error(f"Failed to load template: {e}")
        return None

def process_pdf(target_pdf):
    output_data = []
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
   
    # 1. テンプレート画像の準備
    template_img = load_template_image()
    if template_img is None:
        logging.error("Cannot proceed without template.")
        return []

    try:
        images = convert_from_path(target_pdf, dpi=CONFIG['DPI'], poppler_path=CONFIG['POPPLER_PATH'])
    except Exception as e:
        logging.error(f"PDF Conversion Failed: {e}")
        return []

    for page_idx, img_pil in enumerate(images):
        logging.info(f"Processing Page {page_idx + 1}")
        cv_img = np.array(img_pil)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
       
        # 2. ターゲット画像の一次補正（回転・Deskew）
        #    【変更点】テンプレート画像を渡して、マッチングによる強力な向き検知を行う
        target_color, _ = find_correct_orientation(cv_img, template_img)
        
        # 3. テンプレートに対する位置合わせ（Image Registration）実行！
        #    これで画像がテンプレートと完全に重なる位置に変形される
        aligned_color, aligned_gray = align_image_to_template(target_color, template_img)
        
        debug_img = aligned_color.copy()
        
        ax, ay, aw, ah = CONFIG['ANCHOR_BASE_RECT']
        pitch = CONFIG['ROW_PITCH_Y']
       
        # 4. 位置合わせ済みの画像に対してOCR実行 (座標は固定でOK)
        for row_idx in range(CONFIG['MAX_ROWS']):
            curr_ax = ax
            curr_ay = int(ay + (row_idx * pitch))
           
            # アンカー確認は「補正後」の画像に対して行う
            is_anchor, anchor_text = check_anchor(aligned_gray, (curr_ax, curr_ay, aw, ah))
           
            color = (0, 255, 0) if is_anchor else (0, 0, 255)
            cv2.rectangle(debug_img, (curr_ax, curr_ay), (curr_ax+aw, curr_ay+ah), color, 2)
           
            if not is_anchor:
                # ズレ補正済みなので、アンカーがない＝本当にデータがない と判断してbreak
                break
            
            row_data = {}
            model_name = anchor_text.replace('?', '7').replace('1', '7') if anchor_text else "F7 ALL"
            if "FALL" in model_name: model_name = "F7 ALL"
            row_data['機種'] = model_name

            ENHANCED_ITEMS = ['部品番号', '附属書番号', '注番', '数量', '工番・分番', '納期']
            AMOUNT_ITEMS = ['材料費', '加工費']

            for key, setting in CONFIG['ROI_ITEMS'].items():
                dx, dy, w, h = setting['rect']
                rx = curr_ax + dx
                ry = curr_ay + dy
               
                if ry+h > aligned_gray.shape[0] or rx+w > aligned_gray.shape[1]: continue

                roi = aligned_gray[ry:ry+h, rx:rx+w]
               
                if key in AMOUNT_ITEMS:
                    processed_roi = preprocess_amount_roi(roi)
                elif key in ENHANCED_ITEMS:
                    processed_roi = preprocess_for_ocr(roi)
                else:
                    _, processed_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ocr_conf = CONFIG['TESS_CONFIG_TEXT']
                if setting['type'] == 'number':   ocr_conf = CONFIG['TESS_CONFIG_NUM']
                elif setting['type'] == 'integer': ocr_conf = CONFIG['TESS_CONFIG_INT']
                elif setting['type'] == 'rev':    ocr_conf = CONFIG['TESS_CONFIG_REV']
                elif setting['type'] in ['order_no', 'appendix', 'job_no']: ocr_conf = CONFIG['TESS_CONFIG_ALPHANUM']
                elif setting['type'] == 'supplier': ocr_conf = CONFIG['TESS_CONFIG_JPN']
               
                raw_text = pytesseract.image_to_string(processed_roi, config=ocr_conf)
                # Fallback for amount cells: try another PSM if digits are not found
                if setting['type'] == 'number' and not re.search(r'\d', raw_text):
                    raw_text = pytesseract.image_to_string(processed_roi, config=CONFIG.get('TESS_CONFIG_NUM_ALT', ocr_conf))
                clean_val = clean_text(raw_text, setting['type'])
                row_data[key] = clean_val
               
                cv2.rectangle(debug_img, (rx, ry), (rx+w, ry+h), (255, 0, 0), 1)
           
            output_data.append(row_data)
            logging.info(f" Extracted Row {row_idx+1}")

        cv2.imwrite(str(debug_dir / f"debug_p{page_idx+1}.jpg"), debug_img)

    return output_data

if __name__ == "__main__":
    target_pdf = "target.pdf" # 処理したいPDFファイル名
    
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