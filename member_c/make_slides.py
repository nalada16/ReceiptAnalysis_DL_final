# -*- coding: utf-8 -*-
"""依 report_outline.md 生成簡報 member_C_slides.pptx（16:9）。

讀 CSV 取真實數字、嵌入對應圖表。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

import c_common as cc

OUT = cc.OUT_DIR
PF = OUT.parent / "price_feature_pipeline" / "outputs"
PU = OUT.parent / "per_user_pipeline" / "outputs"
FONT = "Microsoft JhengHei"
NAVY = RGBColor(0x2C, 0x3E, 0x50)
ACCENT = RGBColor(0x8B, 0x1A, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
TEALC = RGBColor(0x1E, 0x4A, 0x40)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5


def _f(run, size, color=NAVY, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def title_bar(s, text):
    box = s.shapes.add_textbox(Inches(0.45), Inches(0.28), Inches(12.4), Inches(0.85))
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = text
    _f(tf.paragraphs[0].runs[0], 25, ACCENT, True)
    # 底線
    ln = s.shapes.add_shape(1, Inches(0.5), Inches(1.18), Inches(3.0), Pt(2.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACCENT
    ln.line.fill.background()


def new(title=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        title_bar(s, title)
    return s


def bullets(s, items, top=1.45, left=0.7, width=12.0, sizes=(19, 16)):
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.6))
    tf = box.text_frame
    tf.word_wrap = True
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "    – ") + txt
        p.space_after = Pt(7)
        _f(p.runs[0], sizes[0] if lvl == 0 else sizes[1], NAVY if lvl == 0 else GREY)
    return box


def image(s, path, left, top, max_w, max_h, caption=None):
    path = Path(path)
    if not path.exists():
        tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(max_w), Inches(0.5))
        tb.text_frame.text = f"[缺圖 {path.name}]"
        return
    w, h = Image.open(path).size
    r = min(max_w / w, max_h / h)
    dw, dh = w * r, h * r
    pl = left + (max_w - dw) / 2
    s.shapes.add_picture(str(path), Inches(pl), Inches(top), height=Inches(dh))
    if caption:
        cb = s.shapes.add_textbox(Inches(left), Inches(top + dh + 0.02), Inches(max_w), Inches(0.35))
        cb.text_frame.word_wrap = True
        cb.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        r2 = cb.text_frame.paragraphs[0].add_run(); r2.text = caption
        _f(r2, 11, GREY); r2.font.italic = True


def table(s, headers, rows, left, top, width, col_w=None, fsize=12, hsize=12.5):
    nr, nc = len(rows) + 1, len(headers)
    tb = s.shapes.add_table(nr, nc, Inches(left), Inches(top), Inches(width), Inches(0.4 * nr)).table
    if col_w:
        for i, w in enumerate(col_w):
            tb.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        c = tb.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = ACCENT
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        rr = p.add_run(); rr.text = str(h); _f(rr, hsize, RGBColor(255, 255, 255), True)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = tb.cell(i, j)
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF6, 0xF1, 0xF1) if i % 2 else RGBColor(0xFF, 0xFF, 0xFF)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = c.text_frame.paragraphs[0]
            rr = p.add_run(); rr.text = "" if v is None else str(v); _f(rr, fsize, NAVY)
    return tb


def note(s, text, top=6.9):
    b = s.shapes.add_textbox(Inches(0.6), Inches(top), Inches(12.1), Inches(0.5))
    b.text_frame.word_wrap = True
    r = b.text_frame.paragraphs[0].add_run(); r.text = text
    _f(r, 12.5, GREY); r.font.italic = True


# ============================================================
# 封面
# ============================================================
s = new()
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.5), Inches(11.7), Inches(1.6))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run(); r.text = "個人化消費增量分析"
_f(r, 40, ACCENT, True)
sb = s.shapes.add_textbox(Inches(0.8), Inches(4.1), Inches(11.7), Inches(1.0))
sb.text_frame.word_wrap = True
r = sb.text_frame.paragraphs[0].add_run()
r.text = "商品語意分群 · 兩層式商品分類 · 個人消費增量（通膨）診斷　|　成員 C"
_f(r, 18, GREY)

# p1
s = new("基本介紹")
bullets(s, [
    ("核心目的：解釋「這個月多花的錢，花在哪裡」", 0),
    ("透過分群演算法找出長期固定購買的同類型商品，再做時間序列分析", 0),
    ("Input：任務一經 BERT embedding 後的發票資料（2613 筆、3 人、9 個月）", 0),
    ("Output：商品分群結果 + 個人消費增量分析報告", 0),
])

# p2 架構
s = new("系統 / 實驗架構（三階段）")
image(s, OUT / "layer3_architecture.png", 3.5, 1.25, 6.3, 6.0)
bullets(s, [
    ("第一階段（精準層）", 0),
    ("BERT → UMAP → HDBSCAN → ~110 細群", 1),
    ("第二階段（故事層）", 0),
    ("去 noise/不純群 → AHC 合併 → 25 超類型", 1),
    ("第三階段", 0),
    ("個人化增量分析（變貴/買更多）", 1),
], top=2.2, left=0.6, width=3.0, sizes=(16, 13))

# p3 第一階段
s = new("第一階段：商品語意分群（精準層）")
bullets(s, [
    ("目的：產出乾淨、語意一致、可追單一商品單價的細群", 0),
    ("Input：所有人資料一起做分群（pooled）　|　Output：~110 群", 0),
    ("Evaluation：silhouette 0.88、purity 0.99、noise 6.9%", 0),
    ("UMAP：保留局部語意、壓低維度（noise 砍 ~5 倍）；HDBSCAN：自動決定群數、標記 noise", 0),
], top=1.4, sizes=(17, 14))
table(s, ["細群", "群內主要品項", "解讀"], [
    ["大冰美式 等3款", "大冰美式 / 特大冰美式 / 中冰美式", "美式咖啡（不同容量）"],
    ["皮蛋乾麵 等15款", "皮蛋乾麵 / 燙青菜 / 蛤蠣湯", "麵店主食+小菜"],
    ["母傳清燉牛肉 等6款", "清燉牛肉 / 紅燒牛肉 / 紅燒三寶", "牛肉麵店全餐點"],
], 1.0, 4.4, 11.3, col_w=[2.6, 5.2, 3.5], fsize=13)

# p4 pooled
s = new("為何「全部人一起分群」（pooled）")
bullets(s, [
    ("分群＝建一本共用商品字典（商品身份與「誰買」無關），三人共用同一套定義才能跨人對齊", 0),
    ("實測「三人分開分群」對小資料量者崩潰：粒度崩塌（user 2：59→8 群）、失去跨人可比（100%→0%）", 0),
], top=1.35, sizes=(16, 14))
image(s, PU / "pu_step3c_user2.png", 0.6, 2.7, 6.0, 3.4, "分開分群：user 2 只剩 3–4 類，全是瓶裝水/文具")
image(s, OUT / "step3d_K25_user2.png", 6.8, 2.7, 6.0, 3.4, "一起分群：速食/茶飲/零食…正常多元")
note(s, "分群是「建字典」、診斷才「分人」——pooled 不會把個人消費混在一起算。", top=6.4)

# p5 baseline
s = new("與其他分群演算法 / baseline 比較")
rows = [
    ["Regex 關鍵字（規則下限）", "0.44", "56%", "0.26"],
    ["TF-IDF + HDBSCAN（傳統 ML）", "0.64", "36%", "0.71"],
    ["BERT + K-Means（消融）", "1.00", "0%", "0.83"],
    ["BERTopic（Ours）", "0.93", "7%", "0.88"],
]
table(s, ["方法", "覆蓋率", "noise", "silhouette"], rows, 0.7, 1.6, 7.4,
      col_w=[3.7, 1.2, 1.2, 1.3], fsize=13)
bullets(s, [
    ("Ours 同時解決：", 0),
    ("語意辨識不清（覆蓋率 44%→93%）", 1),
    ("雜訊干擾（noise 36%→7%）", 1),
    ("BERT+KMeans 的 0% noise 是假象：強迫每點進群、無離群機制", 0),
], top=4.5, sizes=(16, 14))

# p5b 價格 feature
s = new("我們也試過：加價格 feature（負面結果）")
image(s, PF / "pf_clustering_comparison.png", 0.6, 1.4, 7.2, 4.6)
bullets(s, [
    ("想法：BERT 輸入含金額區間但訊號被稀釋，試在 UMAP 後顯式加價格特徵", 0),
    ("結果：純 UMAP 最佳", 0),
    ("noise 5.6% vs 加價格 7.5–10.5%", 1),
    ("silhouette 0.87 vs 0.81–0.83", 1),
    ("原因：BERT 已隱含價格訊號，顯式加入＝冗餘+噪音", 0),
    ("「試過、數字說不行」的誠實負面結果", 0),
], top=1.5, left=8.0, width=5.0, sizes=(14, 12))

# p6 第二層
s = new("為何需要第二層（合併成超類型）")
bullets(s, [
    ("問題：~110 細群對「多花在哪」太細，錢被切太碎、講不出重點", 0),
    ("做法：去掉 noise + 不純細群（單價落差 p90/p10>6，如代收/購物袋雜物群），再把細群中心向量合併成 25 群", 0),
    ("為何第二層用 Agglomerative Hierarchical Clustering（非 HDBSCAN）：", 0),
    ("可直接指定群數 K（HDBSCAN 不能精準控 K，我們要剛好 25 個大類）", 1),
    ("只對 ~108 個細群中心分群，點少、算力低", 1),
    ("「一個細群一票」合併，不受購買頻率灌水", 1),
    ("K 的選擇：比較 K=15/25/35，K=15 太粗、K=35 略瑣碎 → 採 K=25", 0),
], top=1.4, sizes=(17, 14))

# p7 25 超類型
s = new("第二層分群結果（25 超類型，節選）")
try:
    sm = pd.read_csv(OUT / "supertype_K25_members.csv").sort_values("n_rows", ascending=False)

    def top3(x):
        return " / ".join(p.split("(")[0] for p in str(x).split(" | ")[:3])
    rows = [[r["custom_name"], int(r["n_distinct_items"]), top3(r["top_items"])]
            for _, r in sm.head(9).iterrows()]
except Exception:
    rows = [["（讀取失敗）", "", ""]]
table(s, ["超類型（人工命名）", "品項數", "代表品項"], rows, 1.2, 1.55, 11.0,
      col_w=[2.6, 1.2, 7.2], fsize=13)
note(s, "細群（精準層）保留可追單品；超類型（故事層）給「多花在哪大類」的 headline，可下鑽。"
        "完整見 supertype_K25_members.csv", top=6.7)

# p8 方法
s = new("第三階段：個人化增量分析 — 方法")
bullets(s, [
    ("定義：每人「這個月」=最新月(2026-05)；「平常」=之前所有月的月均", 0),
    ("兩因子拆解公式：", 0),
    ("變貴 = (本月單價 − 平常單價) × 本月數量", 1),
    ("買更多 = (本月數量 − 平常月均數量) × 平常單價", 1),
    ("增量 ΔS = 變貴 + 買更多 = 這個月 − 一個典型月 的花費差", 1),
    ("門檻（只分析值得比的類型）：基期≥2 月 且（月均量≥10 或 月均花費≥100）", 0),
    ("舉例：珍奶「50→70 元」是變貴；「一月 5 杯→10 杯」是買更多", 0),
], top=1.4, sizes=(17, 14))

# p9 三人結果
s = new("三人「這個月多花 / 少花在哪」")
table(s, ["user", "本月", "最大貢獻類型", "故事"], [
    ["0", "多花 945", "正餐主食 +1162", "均價 115→256，含一筆 1199 元韓式烤肉聚餐"],
    ["1", "少花 720", "正餐主食 +651", "正餐量價齊升，但手搖/咖啡大幅少買抵消"],
    ["2", "多花 285", "速食小食 +302", "瓶裝茶飲漲 67%、速食買更多"],
], 0.7, 1.5, 12.0, col_w=[0.8, 1.6, 2.6, 7.0], fsize=12.5)
image(s, OUT / "step3d_K25_user0.png", 0.6, 3.5, 4.0, 3.6, "user 0")
image(s, OUT / "step3d_K25_user1.png", 4.66, 3.5, 4.0, 3.6, "user 1")
image(s, OUT / "step3d_K25_user2.png", 8.72, 3.5, 4.0, 3.6, "user 2")

# p10 核心發現
s = new("核心發現 + 有趣案例")
bullets(s, [
    ("① 個人通膨是「品項分化」的，不是齊漲：user 1 正餐 +19%、但手搖甜點 −7%、咖啡 −5%", 0),
    ("② 「變貴」要分辨真漲價 vs 偶發高價（重點案例）", 0),
    ("user 0 正餐主食均價暴衝 115→256（+123%）", 1),
    ("下鑽品項：平常 99 元雞肉堡 + 一筆 1199 元韓式烤肉 2 人套餐（聚餐）", 1),
    ("→ 非系統性漲價，而是一次性高價聚餐；展示可下鑽到品項層級驗證", 1),
    ("③ 部分是「買更多」非「變貴」：user 2 速食 6.3→11 份，增量幾乎全來自量", 0),
], top=1.4, sizes=(16, 14))

# p11 延伸洞察
s = new("延伸洞察：跨全時序（不只看本月）")
image(s, OUT / "insight_monthly_spend.png", 0.6, 1.5, 7.0, 4.6)
bullets(s, [
    ("寒假效應：user 0、2 在 2026-01 暴跌（680 / 39 元），user 1 反而最高", 0),
    ("長期習慣：user 1 瓶裝茶飲翻倍（8→16）、停車減半（22→11）；user 0 速食戒掉（24→3）", 0),
    ("單月暴衝：user 0 速食 9/10 月 3.3×、user 1 運動健身 1 月 2.9×", 0),
    ("→ 「異常落在哪個類型」正是成員 B 異常偵測(Layer 2)的可解釋線索，串接整個系統", 0),
], top=1.5, left=7.9, width=5.1, sizes=(13.5, 12))

# p12 結論
s = new("結論")
bullets(s, [
    ("以 BERT + UMAP + HDBSCAN 建立高品質商品語意分群（noise 7%、純度 0.99），有完整方法與 baseline 驗證", 0),
    ("設計兩層商品分類體系（細群追價 + 超類型講故事），並以實驗否決「分開分群」與「加價格 feature」", 0),
    ("完成個人消費增量分析（超類型 K=25），拆解「變貴 vs 買更多」，揭示個人通膨的品項分化特性", 0),
    ("跨時序延伸洞察（寒假季節性、習慣長期成長/衰退、單月暴衝），為成員 B 時序預測與異常偵測提供可解釋線索", 0),
], top=1.5, sizes=(17, 14))

# backup
s = new("Backup / Q&A")
bullets(s, [
    ("指標怎麼算：purity（每群多數類別佔比）、NMI（分群與標籤互資訊，懲罰群數過多）、silhouette（群內緊群間分）", 0),
    ("為何 pooled 不會洩漏個人消費：分群只看品名/店名語意，診斷才帶 user_id 分人算", 0),
    ("第二層 AHC vs KMeans：AHC 確定性（KMeans 換 seed 結果就變，seed 間 ARI 0.47）、附 dendrogram", 0),
    ("資料限制：飲食佔 82%、發票覆蓋率（現金/訂閱無發票）、低頻品項無法納入", 0),
], top=1.5, sizes=(16, 14))

out = OUT.parent / "member_C_slides.pptx"
prs.save(str(out))
print(f"已輸出 {out}（{len(prs.slides._sldIdLst)} 頁）")
