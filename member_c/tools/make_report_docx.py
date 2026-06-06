# -*- coding: utf-8 -*-
"""生成成員 C 的完整書面報告 Word 檔（member_C_report.docx）。

包含：各部分目的、input/output、演算法選擇原因、evaluation、實驗比較、結果分析，
以及 25 個超類型的分群結果（含品項）、大量表格與圖表。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

import c_common as cc

OUT = cc.OUT_DIR
PF = cc.OUT_DIR.parent / "price_feature_pipeline" / "outputs"
PU = cc.OUT_DIR.parent / "per_user_pipeline" / "outputs"
FONT = "Microsoft JhengHei"

doc = Document()

# 預設字型（含中文 east-asia）
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def _set_cjk(run):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), FONT)


def H(text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    _set_cjk(r)
    if level == 1:
        r.font.color.rgb = RGBColor(0x8B, 0x1A, 0x2B)
    return h


def P(text, bold=False, italic=False, size=10.5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    _set_cjk(r)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    _set_cjk(r)
    return p


def img(path, width=6.3, caption=None):
    path = Path(path)
    if not path.exists():
        P(f"[缺圖：{path.name}]", italic=True)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        r.italic = True
        r.font.size = Pt(9)
        _set_cjk(r)


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(h))
        r.bold = True
        r.font.size = Pt(9.5)
        _set_cjk(r)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            r.font.size = Pt(9)
            _set_cjk(r)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


def df_table(df, cols, headers=None, widths=None, fmt=None):
    headers = headers or cols
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if fmt and c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
            row.append(v)
        rows.append(row)
    return table(headers, rows, widths)


# ============================================================
# 封面
# ============================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("電子發票消費行為診斷\nTask 2 Layer 3：商品語意分群與個人消費增量分析")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(0x8B, 0x1A, 0x2B)
_set_cjk(r)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("成員 C 書面報告　|　商品語意分群 · 兩層式商品分類 · 個人增量（通膨）分析")
r.font.size = Pt(11)
_set_cjk(r)
doc.add_paragraph()

# ============================================================
# 0. 摘要
# ============================================================
H("0. 摘要", 1)
P("本報告負責電子發票分析系統的 Task 2 Layer 3：將成員 A 的 BERT 分類結果，進一步以語意"
  "分群成「商品類型」，再追蹤每個類型沿時間的單價與數量變化，回答「這個月比平常多花的錢，"
  "花在哪些類型、是變貴還是買更多」。核心成果：")
bullet("以 BERT embedding + UMAP + HDBSCAN 建立約 110 個高品質商品語意群（noise 6.9%、純度 0.99）。")
bullet("設計兩階段分群（細群 → 合併成 25 個超類型），兼顧「精準追單一商品」與「好講故事的大類」。")
bullet("完成個人增量分析，揭示個人通膨為「品項分化」而非齊漲，並能定位到具體商品類型與品項。")
bullet("以 7 組對照實驗驗證方法選擇（分群演算法、baseline、pooled vs 分人、顆粒度、價格 feature）。")

# ============================================================
# 1. 資料與系統定位
# ============================================================
H("1. 資料與系統定位", 1)
H("1.1 目的", 2)
P("把離散的發票品項，轉成「可比較、可追蹤」的商品類型，作為個人消費增量（通膨）診斷的基礎。")
H("1.2 Input / Output", 2)
table(["項目", "內容"], [
    ["Input：明細", "all_user_6_label.csv：2613 筆、3 位使用者、2025-09～2026-05、6 類消費標籤"],
    ["Input：語意向量", "receipt_embeddings.npy：2613×768 BERT [CLS] embedding（成員 A，逐列對齊）"],
    ["Output：分群", "細群（~110）+ 超類型（K=25），每筆明細的類型標籤"],
    ["Output：分析", "每人每類型的單價/數量變化、增量拆解、跨時序洞察"],
], widths=[1.6, 4.7])
P("資料分布：飲食 82%、交通 6%、購物 5%、娛樂 3%、教育 3%、醫療 1%（偏食，結論主要適用高頻飲食類）。")
img(OUT / "system_architecture.png", 6.3,
    "圖 1：系統架構。成員 A 清洗+分類 → 成員 B 時序預測+異常 → 成員 C（本報告）商品分群+通膨拆解。")

# ============================================================
# 2. 商品語意分群
# ============================================================
H("2. 商品語意分群", 1)
H("2.1 目的", 2)
P("回答「哪些發票明細其實是同一種商品」，把雜亂品名歸併成語意一致的商品群（product identity resolution）。")

H("2.2 演算法選擇原因", 2)
table(["元件", "選擇", "原因"], [
    ["語意表示", "複用成員 A 的 BERT embedding", "Transformer 預訓練+微調的語意向量品質高；不重訓＝遷移學習，省算力"],
    ["降維", "UMAP（5D, cosine）", "保留局部語意結構、壓低維度讓密度估計穩定；實測把 noise 砍 ~5 倍"],
    ["分群", "HDBSCAN", "密度式、自動決定群數、能標記 noise（離群品項不汙染群）；不需預設 K"],
    ["框架", "BERTopic", "整合 UMAP+HDBSCAN 並提供主題詞；本案因中文短品名改用最高頻品名標籤"],
], widths=[1.1, 2.0, 3.2])
P("深度學習定位：核心為 BERT（Transformer）。本層將其中間層 [CLS] 向量作為商品語意特徵，"
  "屬遷移學習／表示學習的下游應用——以預訓練深度模型提特徵，交給輕量非監督演算法做結構發現。", italic=True)

H("2.3 Evaluation（評估指標與計算方式）", 2)
P("商品分群為非監督任務，無直接準確率，故以成員 A 的 6 類消費標籤為外部基準，並配合內部指標：")
table(["指標", "類型", "計算方式（摘要）", "判讀"], [
    ["Purity", "外部", "每群取多數真實類別點數加總 ÷ 總點數", "越高越好（但群越多越易高，需配群數）"],
    ["NMI", "外部", "分群與標籤的正規化互資訊 I(C;T)/mean(H(C),H(T))", "懲罰群數遠多於類別數；細分群天生偏低"],
    ["Silhouette", "內部", "(b−a)/max(a,b)，a=群內平均距、b=最近他群平均距", "越高=群內緊、群間分；跨空間不可直接比"],
    ["noise_ratio", "結構", "HDBSCAN 標為 −1 的比例", "越低越好（但極稀有品項變 noise 屬良性）"],
], widths=[1.1, 0.7, 3.0, 1.5])

H("2.4 實驗比較①：分群方法對照（3 降維 × 3 分群）", 2)
P("固定兩種粒度公平比較：細粒度 K=108（與 HDBSCAN 同級）、粗粒度 K=6（對齊 6 類）。")
try:
    ec = pd.read_csv(OUT / "exp_clustering.csv")
    show = ec[ec["config"].isin([
        "raw768+HDBSCAN", "PCA30+HDBSCAN", "UMAP5+HDBSCAN",
        "UMAP5+KMeans(K=108)", "raw768+KMeans(6)", "UMAP5+KMeans(6)"])]
    df_table(show, ["config", "n_clusters", "noise_ratio", "purity", "NMI", "ARI", "silhouette"],
             ["組合", "群數", "noise", "purity", "NMI", "ARI", "silhouette"])
except Exception as e:
    P(f"[讀取 exp_clustering.csv 失敗：{e}]")
img(OUT / "exp_clustering.png", 6.3, "圖 2：12 組分群方法對照（完整數據見 exp_clustering.csv）。")
P("結果分析：")
bullet("UMAP5+HDBSCAN 在商品群任務最佳：silhouette 0.88（最高）、noise 6.9%（raw/PCA 上 HDBSCAN noise 高達 32–34%）。")
bullet("BERT embedding 直接 KMeans(K=6) 即能還原 6 類消費（ARI 0.97、NMI 0.92）→ 證明 embedding 品質高。")
bullet("UMAP 是雙面刃：利於細商品群、卻傷害粗 6 類分群（K=6 時 ARI 掉到 0.45）。")

H("2.5 實驗比較②：與 Baseline 對照", 2)
P("以相同外部基準比較規則法、傳統 ML、消融與本方法。")
try:
    eb = pd.read_csv(OUT / "exp_slide_baselines.csv")
    df_table(eb, ["method", "n_clusters", "noise_ratio", "coverage", "purity", "silhouette"],
             ["方法", "群數", "noise", "覆蓋率", "purity", "silhouette"])
except Exception as e:
    P(f"[讀取 exp_slide_baselines.csv 失敗：{e}]")
img(OUT / "exp_slide_baselines.png", 6.3, "圖 3：Baseline 對照。")
bullet("解決『語意辨識不清』：覆蓋率 Regex 44% → Ours 93%（能合併大/中/特大冰美式等變體）。")
bullet("解決『雜訊干擾』：noise TF-IDF 36% → Ours 7%。")
bullet("BERT+K-Means 的 0% noise 是假象——強迫每點進群、無離群機制。")

H("2.6 實驗比較③：價格 feature concat（負面結果）", 2)
P("動機：BERT 輸入雖含金額區間但訊號被稀釋。測試在 UMAP 後顯式 concat 標準化價格特徵（權重 w）。"
  "w² /(5+w²)=價格 variance 佔比。")
try:
    pf = pd.read_csv(PF / "pf_clustering_comparison.csv")
    df_table(pf, ["weight", "price_var_share", "n_clusters", "noise_ratio", "silhouette", "purity"],
             ["weight", "價格var佔比", "群數", "noise", "silhouette", "purity"])
except Exception as e:
    P(f"[讀取 pf_clustering_comparison.csv 失敗：{e}]")
img(PF / "pf_clustering_comparison.png", 6.3, "圖 4：加入價格 feature 對分群品質的影響。")
bullet("純 UMAP（w=0）在 noise（5.6%）與 silhouette（0.87）皆最佳；加價格反而 noise↑、silhouette↓。")
bullet("原因：BERT embedding 已隱含價格訊號（預測金額區間準確率 70%），顯式加入＝冗餘＋離散化噪音。")
bullet("決策：主流程不加額外價格 feature。（獨立實驗於 price_feature_pipeline/）")

img(OUT / "step1_umap_scatter.png", 5.5, "圖 5：商品分群在 2D 空間的散佈（群分得很開，灰 × 為 noise）。")

# ============================================================
# 3. 兩層式商品分類（超類型）
# ============================================================
H("3. 兩層式商品分類（超類型）", 1)
H("3.1 目的與方法（雙階段分群）", 2)
P("細群（~110）對「多花在哪」太細，故再合併成大類。採兩階段分群：")
P("第一層 HDBSCAN（密度分群＋濾雜訊）：得乾淨、可追單一商品的細群（精準層）。")
P("第二層 對「細群中心向量」做 Agglomerative(Ward) 合併成 K 個超類型（故事層）：")
bullet("把每個細群所有品項 embedding 取平均 → 768 維中心向量；~108 個細群變成 ~108 個點。")
bullet("對這些中心點做 Ward 階層合併到剩 K 群。用 Agglomerative 而非 HDBSCAN 因為：可直接指定 K、且只對 ~108 點分群（算力低）。")
P("與「直接分 K 群」的差異：兩階段先濾 noise、細群為原子（只併不拆，精準層完整保留）、"
  "以「一個細群一票」合併（不受購買頻率灌水），且同時保有兩層可下鑽。")

H("3.2 演算法選擇與 K 的決定", 2)
P("比較 K=15/25/35：K=15 過粗（最大群含 134 款無法命名）、K=35 略瑣碎；採 K=25（飲料/咖啡/正餐/拉麵分開、bar 數適中）。")

H("3.3 超類型分群結果（K=25，含品項）", 2)
P("下表為 25 個超類型的完整結果：人工命名、品項數、明細筆數、單價中位數、代表品項。"
  "（每超類型完整品項見 supertype_K25_members.csv）")
try:
    sm = pd.read_csv(OUT / "supertype_K25_members.csv").sort_values("n_rows", ascending=False)

    def top_items(s, n=8):
        return "、".join(p.split("(")[0] for p in str(s).split(" | ")[:n])
    rows = []
    for _, r in sm.iterrows():
        rows.append([r["custom_name"], int(r["n_distinct_items"]), int(r["n_rows"]),
                     f'{r["unit_price_p50"]:.0f}', top_items(r["top_items"])])
    table(["超類型（人工命名）", "品項數", "筆數", "單價中位", "代表品項（前8）"],
          rows, widths=[1.3, 0.6, 0.55, 0.65, 3.2])
except Exception as e:
    P(f"[讀取 supertype_K25_members.csv 失敗：{e}]")

# ============================================================
# 4. 個人消費增量分析（主結果）
# ============================================================
H("4. 個人消費增量分析（超類型 K=25）", 1)
H("4.1 目的、Input / Output", 2)
P("目的：解釋「這個月」比「平常」多花的錢，落在哪些超類型，並拆解為「變貴」與「買更多」。")
table(["項目", "內容"], [
    ["Input", "明細 + 超類型標籤；每人「這個月」=最新月(2026-05)，「平常」=之前所有月的月均"],
    ["Output", "每人每超類型的 P0/P1/Q0/Q1、變貴(價效應)、買更多(量效應)、增量 ΔS；圖表"],
], widths=[1.2, 5.1])
H("4.2 方法（拆解公式與門檻）", 2)
P("變貴 price_effect = (P1−P0)·Q1　；　買更多 qty_effect = (Q1−Q0)·P0　；　ΔS = 變貴 + 買更多")
P("「值得分析」門檻（per-user、每超類型）：基期≥2月 且（月均數量≥10 或 月均花費≥100 元）"
  "——量大或花錢多擇一即納入，過濾「買太少、比漲價沒意義」的類別。")

H("4.3 結果與分析（逐人）", 2)
try:
    bk = pd.read_csv(OUT / "step3d_increment_breakdown.csv")
    user_note = {
        0: "本月多花 945 元，主因正餐主食均價 115→256（含一筆 1199 元韓式烤肉聚餐，屬偶發高價）。",
        1: "本月少花 720 元；正餐主食量價齊升(+651)，但手搖/咖啡類大幅少買抵消。",
        2: "本月多花 285 元；瓶裝茶飲大漲(+67%)、速食小食買更多。",
    }
    for u in sorted(bk["user_id"].unique()):
        H(f"user {u}", 3)
        P(user_note.get(u, ""))
        d = bk[bk.user_id == u].reindex(
            bk[bk.user_id == u]["delta_spend"].abs().sort_values(ascending=False).index).head(6)
        df_table(d, ["supertype", "P0", "P1", "Q0_per_month", "Q1_this_month", "delta_spend", "變貴_price", "買更多_qty"],
                 ["超類型", "平常單價", "本月單價", "平常/月量", "本月量", "增量", "變貴", "買更多"])
        img(OUT / f"step3d_K25_user{u}.png", 6.3, f"圖：user {u} 增量拆解（左）與主要類型單價歷史（右）。")
except Exception as e:
    P(f"[讀取 step3d_increment_breakdown.csv 失敗：{e}]")

P("完整版：每人「所有通過門檻超類型」的單價歷史（>7 個自動分兩張子圖）。以 user 1 為例：")
img(OUT / "history_user1.png", 6.0, "圖：user 1 全部 14 個超類型的單價成長歷史。")

# ============================================================
# 5. 延伸洞察
# ============================================================
H("5. 延伸洞察（跨全時序）", 1)
P("把超類型沿 9 個月攤開，發現「單月異常」「長期趨勢」等更有趣的型態。")
H("5.1 寒假效應（季節性）", 2)
bullet("user 0、user 2 在 2026-01 消費暴跌（680 元 / 39 元，vs 平常數千）→ 寒假離校。")
bullet("user 1 反而 1 月最高（11996）→ 生活型態相反。啟示：時序預測需考慮校園行事曆季節性。")
img(OUT / "insight_monthly_spend.png", 6.3, "圖：每人每月總消費（user 0/2 寒假暴跌）。")
H("5.2 長期習慣趨勢", 2)
table(["類型", "user", "變化", "解讀"], [
    ["瓶裝茶飲（量）", "1", "8 → 16 杯/月（翻倍）", "喝瓶裝茶習慣持續增強"],
    ["停車費（次）", "1", "22 → 11 次/月（減半）", "整學期開車/停車越來越少"],
    ["速食小食（量）", "0", "24 → 3 份/月", "學期初狂吃，之後幾乎戒掉"],
], widths=[1.2, 0.5, 1.8, 2.8])
H("5.3 單價長期走勢 & 單月暴衝", 2)
bullet("user 0 便宜類別長期走低：超商零食 −46%、優酪乳 −40%；user 2 瓶裝茶飲 +144%（品質升級型通膨）。")
bullet("單月暴衝：user 0 速食 9/10 月 3.3×、user 1 運動健身 1 月 2.9×（期末紓壓？）→ 正是成員 B 異常偵測(Layer 2)的目標。")
img(OUT / "insight_user0_upgrade.png", 6.3, "圖：user 0 單價長期走勢。")

# ============================================================
# 6. 分群設計討論（實驗）
# ============================================================
H("6. 分群設計討論：為何三人一起分群（pooled）", 1)
P("分群＝建一本公共「商品字典」（商品身份與誰買無關），故三人 pooled；個人診斷才分人。"
  "實測「三人分開分群」對小資料量 user 2 不可靠：")
bullet("粒度崩塌：user 2 單獨分群只分出 8 群（pooled 限定時 59 群）。")
bullet("失去跨人可比：被 ≥2 人購買的商品，pooled 100% 落同群、per-user 0%。")
bullet("小 user 增量無意義：user 2 分開後只剩 3 個可分析類型、全是瓶裝水/文具。")
img(PU / "exp_per_user_clustering.png", 6.3, "圖：per-user vs pooled 分群品質對照。")

# ============================================================
# 7. 限制與結論
# ============================================================
H("7. 限制", 1)
bullet("資料偏食：飲食佔 82%，結論主要適用高頻飲食類。")
bullet("發票覆蓋率：現金/訂閱/轉帳無發票，範圍限可由發票觀察的消費。")
bullet("低頻品項（<10 次）無法納入商品群（良性邊界）。")
bullet("超類型命名為人工（可重現但帶主觀），未來可用 LLM 自動命名。")
bullet("「變貴」需分辨真漲價 vs 偶發高價（如 user 0 正餐 +123% 來自單筆 1199 元聚餐），解讀時應回查品項。")

H("8. 結論", 1)
bullet("以 BERT + UMAP + HDBSCAN 建立高品質商品語意分群（noise 7%、純度 0.99），有完整方法與 baseline 驗證。")
bullet("設計兩層商品分類體系（細群追價 + 超類型講故事），並以實驗否決分開分群與加價格 feature。")
bullet("完成個人消費增量分析，拆解「變貴 vs 買更多」，揭示個人通膨的品項分化特性。")
bullet("跨時序延伸洞察（寒假季節性、習慣長期成長/衰退、單月暴衝）為時序預測與異常偵測提供可解釋線索。")

out_path = OUT.parent / "member_C_report.docx"
doc.save(str(out_path))
print(f"已輸出 {out_path}")
print(f"段落數: {len(doc.paragraphs)}, 表格數: {len(doc.tables)}")
