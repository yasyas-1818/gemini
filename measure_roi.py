import cv2
import numpy as np
from pdf2image import convert_from_path
import sys

# --- 設定 ---
POPPLER_PATH = r'C:\gemini\poppler\Library\bin' # 環境に合わせて変更してください
PDF_PATH = 'target.pdf'
DPI = 300

# 画面表示用の縮小率（画像がデカすぎるので）
DISPLAY_HEIGHT = 900 

# グローバル変数
g_scale = 1.0
g_anchor = None # (x, y, w, h) 原寸座標
g_start_point = None
g_img_original = None
g_img_display = None

def get_real_coords(disp_x, disp_y):
    """画面上の座標を、原寸(300DPI)の座標に変換"""
    return int(disp_x * g_scale), int(disp_y * g_scale)

def mouse_callback(event, x, y, flags, param):
    global g_start_point, g_anchor, g_img_display

    real_x, real_y = get_real_coords(x, y)

    # 左クリック押下：ドラッグ開始
    if event == cv2.EVENT_LBUTTONDOWN:
        g_start_point = (x, y)

    # 左クリック離す：確定
    elif event == cv2.EVENT_LBUTTONUP:
        if g_start_point is None:
            return

        start_x, start_y = g_start_point
        end_x, end_y = x, y
        
        # 矩形計算（画面用）
        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        x2, y2 = max(start_x, end_x), max(start_y, end_y)
        w, h = x2 - x1, y2 - y1

        # 矩形計算（原寸用）
        rx1, ry1 = get_real_coords(x1, y1)
        rx2, ry2 = get_real_coords(x2, y2)
        rw, rh = rx2 - rx1, ry2 - ry1

        # クリックだけ（ドラッグじゃない）なら無視
        if w < 5 or h < 5:
            g_start_point = None
            return

        # 描画（青枠）
        cv2.rectangle(g_img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("Measure Tool", g_img_display)
        
        # --- ロジック ---
        print("-" * 40)
        
        # モード1: アンカー未設定なら、これがアンカーになる
        if g_anchor is None:
            g_anchor = (rx1, ry1, rw, rh)
            print(f"★ アンカー設定完了: {g_anchor}")
            print(f"Configへの記述 -> 'ANCHOR_BASE_RECT': {g_anchor},")
            print("\n【次の手順】各項目（数量、金額など）をドラッグして囲んでください。")
            # アンカーは緑枠で上書き
            cv2.rectangle(g_img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Measure Tool", g_img_display)

        # モード2: アンカー設定済みなら、相対座標を計算してコード生成
        else:
            ax, ay, aw, ah = g_anchor
            
            # 相対座標 (dx, dy)
            dx = rx1 - ax
            dy = ry1 - ay # 行ズレは考慮せず、単純なY差分
            
            # Y座標から「段」を推測 (行ピッチ132px前提での簡易判定)
            # 0付近=上段, 40~60付近=中段, 80~100付近=下段
            row_hint = "上段(Y=0)"
            if 30 < dy < 70: row_hint = "中段(Y=+44付近)"
            elif 70 < dy < 120: row_hint = "下段(Y=+88付近)"
            
            # コンソール用出力（コピペ用フォーマット）
            print(f"計測: 幅={rw}, 高さ={rh} | アンカーからの距離: x={dx}, y={dy}")
            print(f"推定位置: {row_hint}")
            print("↓ コピー用コード ↓")
            print(f"'項目名': {{'rect': ({dx}, {dy}, {rw}, {rh}), 'type': 'text/number'}},")

        g_start_point = None

def main():
    global g_scale, g_img_original, g_img_display

    print("PDFを読み込んでいます...")
    try:
        images = convert_from_path(PDF_PATH, dpi=DPI, poppler_path=POPPLER_PATH)
        g_img_original = np.array(images[0])
        g_img_original = cv2.cvtColor(g_img_original, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"エラー: {e}")
        print("Popplerのパスやファイル名を確認してください。")
        return

    # 表示用に縮小
    h, w = g_img_original.shape[:2]
    g_scale = h / DISPLAY_HEIGHT
    disp_w = int(w / g_scale)
    g_img_display = cv2.resize(g_img_original, (disp_w, DISPLAY_HEIGHT))

    print("\n=== 最強測定ツール 操作方法 ===")
    print("1. 画面が開いたら、「F7 ALL」の文字をドラッグして囲んでください（アンカー設定）。")
    print("2. 次に、「数量」や「金額」などの項目をドラッグして囲んでください。")
    print("3. コンソールに「コピペ用コード」が表示されるので、それをConfigに貼るだけです。")
    print("-------------------------------------------------------")

    cv2.namedWindow("Measure Tool")
    cv2.setMouseCallback("Measure Tool", mouse_callback)

    while True:
        cv2.imshow("Measure Tool", g_img_display)
        key = cv2.waitKey(1)
        if key == 27: # ESCで終了
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()