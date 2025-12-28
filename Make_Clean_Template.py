import fitz  # PyMuPDF
import os

# ==========================================
# 設定エリア: 消したい場所（可変テキスト）の座標
# ==========================================
# 座標系は PDFのポイント単位
# 左上が (0,0) です。
# 調整方法: DEBUG_MODE = True で実行し、赤い箱の位置を見て数値を調整してください。
MASK_AREAS = [
    # 形式: (x0, y0, x1, y1)  -> (左, 上, 右, 下)
    
    # 例1: 右上の日付・ページ番号エリア (大まかな目安)
    (563, 5, 585, 20),
    
    # 例2: 左上の注文番号などが変わる場合
    (550, 105, 570, 200),

    (553, 690, 570, 780),
]

# Trueにすると「赤い枠」で表示します（場所確認用）。
# Falseにすると「真っ白」で塗りつぶします（本番用）。
DEBUG_MODE = False 

def create_clean_template(input_pdf, output_pdf):
    if not os.path.exists(input_pdf):
        print(f"エラー: {input_pdf} が見つかりません。")
        return

    # PDFを開く
    doc = fitz.open(input_pdf)
    
    # 1ページ目のみ取得するために新しいドキュメントを作成
    out_doc = fitz.open()
    out_doc.insert_pdf(doc, from_page=0, to_page=0)
    page = out_doc[0]

    # マスク処理
    for rect_coords in MASK_AREAS:
        rect = fitz.Rect(rect_coords)
        
        # 新しい図形(Shape)オブジェクトを作成
        shape = page.new_shape()
        shape.draw_rect(rect)
        
        if DEBUG_MODE:
            # デバッグ用: 赤色 (枠線あり、塗りつぶしなし)
            shape.finish(color=(1, 0, 0), width=2)
            shape.commit() # 描画を確定
            
            # テキスト挿入 (旧 draw_text -> 新 insert_text)
            page.insert_text((rect.x0, rect.y0+10), "MASK", color=(1,0,0), fontsize=8)
        else:
            # 本番用: 白色で塗りつぶし (枠線なし)
            # fill=(1,1,1) -> 白, color=(1,1,1) -> 枠も白
            shape.finish(color=(1, 1, 1), fill=(1, 1, 1), stroke_opacity=0, fill_opacity=1)
            shape.commit() # 描画を確定

    # 保存
    out_doc.save(output_pdf)
    out_doc.close()
    
    mode_str = "【確認用(赤枠)】" if DEBUG_MODE else "【本番用(白塗り)】"
    print(f"成功: {output_pdf} を作成しました。{mode_str}")
    if DEBUG_MODE:
        print("作成されたPDFを開いて、赤枠が消したい文字をカバーしているか確認してください。")
        print("OKなら、スクリプト内の 'DEBUG_MODE = False' に書き換えて再度実行してください。")

if __name__ == "__main__":
    TARGET_FILE = "target.pdf"
    TEMPLATE_FILE = "Template.pdf"
    
    create_clean_template(TARGET_FILE, TEMPLATE_FILE)