import fitz  # PyMuPDF
import os

def create_grid_pdf(input_pdf, output_pdf):
    if not os.path.exists(input_pdf):
        print(f"エラー: {input_pdf} が見つかりません。")
        return

    doc = fitz.open(input_pdf)
    # 1ページ目だけ取得
    out_doc = fitz.open()
    out_doc.insert_pdf(doc, from_page=0, to_page=0)
    page = out_doc[0]
    
    # ページの幅と高さを取得
    width = page.rect.width
    height = page.rect.height
    
    print(f"ページサイズ: 幅={width}, 高さ={height}")
    print("グリッドを描画中...")

    # シェイプオブジェクト作成
    shape = page.new_shape()
    
    # --- グリッド線と座標を描く ---
    # 50ポイント刻みで線を描く
    step = 50
    
    # 縦線 (X軸)
    for x in range(0, int(width), step):
        shape.draw_line((x, 0), (x, height))
        
    # 横線 (Y軸)
    for y in range(0, int(height), step):
        shape.draw_line((0, y), (width, y))
    
    # 薄い青色で描画
    shape.finish(color=(0, 0, 1), width=0.5, stroke_opacity=0.5)
    shape.commit()

    # --- 座標テキストを書き込む ---
    # 文字が見やすいように赤色で
    for x in range(0, int(width), step):
        for y in range(0, int(height), step):
            # 交点に (x, y) を書く
            page.insert_text((x + 2, y + 10), f"{x},{y}", fontsize=6, color=(1, 0, 0))

    out_doc.save(output_pdf)
    out_doc.close()
    print(f"成功: {output_pdf} を作成しました。")
    print("このファイルを開いて、消したい文字の上に書いてある数値(x,y)を読み取ってください。")

if __name__ == "__main__":
    create_grid_pdf("target.pdf", "debug_grid.pdf")