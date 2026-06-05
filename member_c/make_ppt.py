"""生成成員 C 的報告簡報 member_C_report.pptx（16:9，含圖表）。

內容對應 REPORT.md。圖檔取自 outputs/ 與 per_user_pipeline/outputs/。
"""
from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

import c_common as cc

OUT_MAIN = cc.OUT_DIR
OUT_PU = cc.OUT_DIR.parent / "per_user_pipeline" / "outputs"

NAVY = RGBColor(0x2C, 0x3E, 0x50)
ACCENT = RGBColor(0x8B, 0x1A, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
FONT = "Microsoft JhengHei"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _set_font(tf, size, color=NAVY, bold=False):
    for p in tf.paragraphs:
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold


def add_title_slide(title, subtitle):
    s = prs.slides.add_slide(BLANK)
    box = s.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.7), Inches(1.6))
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = title
    _set_font(tf, 34, NAVY, True)
    b2 = s.shapes.add_textbox(Inches(0.8), Inches(4.1), Inches(11.7), Inches(1.2))
    tf2 = b2.text_frame
    tf2.word_wrap = True
    tf2.text = subtitle
    _set_font(tf2, 18, GREY, False)
    # 底色條
    bar = s.shapes.add_textbox(Inches(0.8), Inches(2.25), Inches(4.5), Inches(0.1))
    return s


def _title_bar(s, title):
    box = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = title
    _set_font(tf, 26, ACCENT, True)


def add_bullets_slide(title, bullets):
    s = prs.slides.add_slide(BLANK)
    _title_bar(s, title)
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(12.0), Inches(5.5))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (txt, lvl) in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "    – ") + txt
        p.space_after = Pt(8)
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(20 if lvl == 0 else 17)
            r.font.color.rgb = NAVY if lvl == 0 else GREY
            r.font.bold = False
    return s


def add_image_slide(title, image_path, caption=None, bullets=None):
    s = prs.slides.add_slide(BLANK)
    _title_bar(s, title)
    img = Path(image_path)
    if not img.exists():
        add = s.shapes.add_textbox(Inches(1), Inches(3), Inches(10), Inches(1))
        add.text_frame.text = f"[缺圖] {img.name}"
        return s
    # 圖置中，寬度上限 11.5"
    from PIL import Image
    w, h = Image.open(img).size
    max_w, max_h = Inches(11.6), Inches(4.7)
    ratio = min(max_w / (w / 96 * 914400 / 914400 * 96), 1)  # placeholder
    disp_w = Inches(11.0)
    disp_h = Inches(11.0 * h / w)
    if disp_h > Inches(4.9):
        disp_h = Inches(4.9)
        disp_w = Inches(4.9 * w / h)
    left = Inches((13.333 - disp_w.inches) / 2)
    top = Inches(1.45)
    s.shapes.add_picture(str(img), left, top, height=disp_h)
    if caption:
        cap = s.shapes.add_textbox(Inches(0.6), Inches(6.7), Inches(12.1), Inches(0.6))
        cap.text_frame.word_wrap = True
        cap.text_frame.text = caption
        _set_font(cap.text_frame, 14, GREY, False)
    if bullets:
        bb = s.shapes.add_textbox(Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.95))
        tf = bb.text_frame
        tf.word_wrap = True
        for i, t in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = "• " + t
            for r in p.runs:
                r.font.name = FONT
                r.font.size = Pt(14)
                r.font.color.rgb = NAVY
    return s


# ============ 投影片內容 ============
add_title_slide(
    "Task 2 Layer 3：商品語意分群與個人增量分析",
    "電子發票消費行為診斷 ｜ 成員 C\n以 BERT embedding 建立商品分類，拆解「這個月多花的錢：變貴 vs 買更多」")

add_bullets_slide("1. 任務定位與資料", [
    ("任務：商品語意分群 → 重複購買/單價追蹤 → 個人增量分析", 0),
    ("回答：這個月比平常多花的錢，花在哪些類型商品、是變貴還是買更多", 0),
    ("輸入（成員 A）：2613 筆明細、3 人、2025-09～2026-05、6 類消費標籤", 0),
    ("BERT [CLS] embedding 2613×768，逐列對齊；直接複用、不重訓", 1),
    ("關鍵設計：分群中間向量供下游通膨分析使用", 0),
])

add_image_slide("2. 系統架構", OUT_MAIN / "system_architecture.png",
                caption="A 清洗+分類 → B 時序預測+異常 → C 商品分群+通膨拆解（本報告）")

add_bullets_slide("3. 商品語意分群 — 主方法與結果", [
    ("方法：BERT embedding → UMAP(5D, cosine) → HDBSCAN", 0),
    ("群數 ~110（自動決定，免設 K）、noise 6.9%", 0),
    ("purity 0.99 / homogeneity 0.95 / silhouette 0.88", 0),
    ("分群品質佳：美式咖啡群（中/大/特大）、牛肉麵店餐點群、手搖飲群", 1),
])

add_image_slide("3.1 分群方法對照（為何選 UMAP+HDBSCAN）", OUT_MAIN / "exp_clustering.png",
                bullets=["UMAP 把 noise 從 32% 砍到 7%、silhouette 最高 0.88",
                         "BERT emb 直接 KMeans(K=6) 還原 6 類：ARI 0.97 → embedding 品質高"])

add_image_slide("3.2 與 Baseline 對照（驗證方法價值）", OUT_MAIN / "exp_slide_baselines.png",
                bullets=["解決『語意辨識不清』：覆蓋率 Regex 44% → Ours 93%",
                         "解決『雜訊干擾』：noise TF-IDF 36% → Ours 7%"])

add_image_slide("4. 為何三人一起分群（pooled）", OUT_PU / "exp_per_user_clustering.png",
                bullets=["分群=建公共商品字典(pooled)；診斷=讀個人帳本(分人)",
                         "分開分群：小資料量 user 2 粒度崩塌(59→8 群)、失去跨人可比 → 否決"])

add_bullets_slide("5. 顆粒度：兩層式商品分類", [
    ("細群 ~110 對增量分析太細 → 合併細群中心向量成『超類型』", 0),
    ("K=15 太粗（134 款揉一團，無法命名）", 1),
    ("K=25 採用：飲料/咖啡/正餐/拉麵分開，bar 數適中、好講故事", 1),
    ("K=35 類別更純但略瑣碎", 1),
    ("最終兩層：精準層（細群，追單一商品）＋ 故事層（超類型 K=25）", 0),
])

add_image_slide("6. 主結果：個人增量分析（精準層）", OUT_MAIN / "step3c_user1.png",
                bullets=["這個月 vs 平常月均，按類型拆成『變貴(價)』與『買更多(量)』",
                         "右圖：各類型單價成長歷史（清燉牛肉 +11%、波士頓 −6%）"])

add_image_slide("6.1 主結果：超類型故事層（K=25）", OUT_MAIN / "step3d_K25_user1.png",
                bullets=["以大類呈現『這個月多花在哪』，headline 更好講"])

add_bullets_slide("6.2 兩個核心發現", [
    ("① 個人通膨是『品項分化』的，不是齊漲", 0),
    ("user 1：清燉牛肉 +11%、沙拉飯 +9%，但波士頓 −6%、紅茶持平", 1),
    ("② 多花的錢常來自『買更多』而非『變貴』", 0),
    ("user 1『清燉牛肉類』本月多花 1003 元 = 買更多 860 + 變貴 143", 1),
    ("（平常每月 2.25 份 → 本月 6 份；單價 229→253）", 1),
])

add_bullets_slide("7. 限制", [
    ("資料偏食：飲食佔 ~82%，結論主要適用高頻飲食類", 0),
    ("發票覆蓋率：現金/訂閱/轉帳無發票，範圍限可由發票觀察的消費", 0),
    ("低頻品項（<10 次）無法納入商品群（良性邊界）", 0),
    ("超類型目前用前 3 名品項自動命名，未來可用 LLM 命名成『手搖飲類』", 0),
])

add_bullets_slide("8. 結論", [
    ("高品質商品語意分群（noise 7%、purity 0.99），有完整方法對照與 baseline 驗證", 0),
    ("兩層商品分類體系（細群追價 + 超類型講故事），並以實驗否決分開分群", 0),
    ("完成個人消費增量分析，拆解『變貴 vs 買更多』，揭示個人通膨的品項分化特性", 0),
])

out = OUT_MAIN.parent / "member_C_report.pptx"
prs.save(str(out))
print(f"已輸出 {out}（共 {len(prs.slides.__iter__.__self__._sldIdLst)} 頁）")
