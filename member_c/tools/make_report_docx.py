# -*- coding: utf-8 -*-
"""生成成員 C（Task 2 Layer 3）書面報告 Word 檔（member_C_report.docx）。

結構（精簡版）：
  1. 任務總覽：三階段（分群 → 合併 → 個人化分析）
  2. 第一階段：商品語意分群（詳細）
  3. 第二階段：超類型合併（詳細）
  4. Evaluation 與成果
  5. 個人化消費增量分析 + insight
  6. 總結與限制
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import sys
sys.path.insert(0, r"D:\ReceiptAnalysis_DL_final\member_c")
import c_common as cc

OUT = cc.OUT_DIR
PF = OUT.parent / "price_feature_pipeline" / "outputs"
PU = OUT.parent / "per_user_pipeline" / "outputs"
FONT = "Microsoft JhengHei"

doc = Document()
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def _cjk(run):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rpr.rFonts.set(qn("w:eastAsia"), FONT)


def H(text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    _cjk(r)
    if level == 1:
        r.font.color.rgb = RGBColor(0x8B, 0x1A, 0x2B)
    return h


def P(text, bold=False, italic=False, size=10.5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    _cjk(r)
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    _cjk(r)
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
        _cjk(r)


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
        _cjk(r)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            r.font.size = Pt(9)
            _cjk(r)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()
    return t


# ============================================================
# 封面
# ============================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Task 2 Layer 3：商品語意分群與個人化消費增量分析")
r.bold = True
r.font.size = Pt(19)
r.font.color.rgb = RGBColor(0x8B, 0x1A, 0x2B)
_cjk(r)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("成員 C 書面報告")
r.font.size = Pt(12)
_cjk(r)
doc.add_paragraph()

# ============================================================
# 1. 任務總覽
# ============================================================
H("1. 任務總覽", 1)
P("本任務承接成員 A 的發票分類結果，回答一個更深入的問題：「這個月比平常多花的錢，花在哪些商品類型，"
  "是因為變貴還是買更多？」整體分為三個階段：")
table(["階段", "任務", "產出"], [
    ["第一階段", "商品語意分群", "~110 個語意一致的細群（精準層）"],
    ["第二階段", "合併分群（超類型）", "25 個可命名的大類（故事層）"],
    ["第三階段", "個人化消費增量分析", "每人每類型的「變貴 vs 買更多」拆解"],
], widths=[1.0, 1.9, 3.4])
P("設計理念：成員 A 的 6 大類粒度太粗（飲食佔 82%，所有消費幾乎都擠在「飲食」），無法回答「哪一種商品"
  "變貴」。因此我們在大類底下重新做語意分群，把商品切得更細、更符合資料真實分布，作為個人化分析的基礎。")
img(OUT / "layer3_architecture.png", 4.5, "圖 1：三階段架構總覽。")

# ============================================================
# 2. 第一階段：商品語意分群
# ============================================================
H("2. 第一階段：商品語意分群", 1)
P("目的：把雜亂的發票品名，自動歸併成語意一致的商品群——讓同一群裡的品項真的是「同一種東西」，"
  "後續才能有意義地追蹤單價與數量。")
H("2.1 方法", 2)
P("直接複用成員 A 訓練好的 BERT 模型輸出的 768 維語意向量（遷移學習，不重新訓練），流程如下：")
table(["步驟", "做法", "原因"], [
    ["BERT Embedding", "取每筆明細的 768 維 [CLS] 向量（複用 Task 1）",
     "Transformer 預訓練語意向量品質高，省算力"],
    ["UMAP 降維", "768 維 → 5 維（cosine）",
     "高維下點距離趨於相等、密度估計失準；降維保留語意鄰近結構"],
    ["HDBSCAN 分群", "密度分群（min_cluster_size=10）",
     "自動決定群數、標記 noise（離群品項不汙染群）、不需預設 K"],
], widths=[1.2, 2.5, 2.6])
P("最終得到 ~110 個細群，每群是語意一致的商品類型（如「美式咖啡群」含中/大/特大冰美式）。")

# ============================================================
# 3. 第二階段：超類型合併
# ============================================================
H("3. 第二階段：超類型合併", 1)
P("目的：~110 個細群對「錢花在哪」太細，消費被切得太碎。將細群合併成 25 個可命名、好解讀的大類。")
H("3.1 方法", 2)
bullet("去除 noise：移除第一階段 HDBSCAN 標記為 −1 的離群品項。")
bullet("AHC 合併：對每個細群的「中心向量」（~108 個點）做 Agglomerative Hierarchical Clustering（Ward）合併成 K 群。")
bullet("人工命名：依群內品項語意，為 25 個超類型命名（如正餐主食、手搖飲料、瓶裝茶飲）。")
H("3.2 為何第二階段改用 AHC（而非 HDBSCAN）", 2)
bullet("可直接指定群數 K：我們要「剛好 25 個大類」，HDBSCAN 無法精準控 K。")
bullet("小樣本下更穩定：只對 ~108 個中心點分群，HDBSCAN 的密度估計在這麼少的點上不可靠，AHC 直接算距離合併更適合。")
bullet("確定性 + 不受頻率灌水：Ward 合併結果固定不受隨機種子影響；以「一個細群一票」合併，高頻商品不會主導大類。")
P("與「一開始就直接分 25 群」相比，兩階段先濾 noise、細群為原子（只併不拆，精準層完整保留），"
  "且同時保有「細群追單一商品」與「超類型講故事」兩層，可下鑽。")

# ============================================================
# 4. Evaluation 與成果
# ============================================================
H("4. Evaluation 與成果", 1)
H("4.1 評估指標", 2)
P("商品分群為非監督任務，以成員 A 的 6 類標籤為外部基準（purity、NMI），並配合內部指標（silhouette）"
  "與結構指標（noise）。purity/NMI 跨方法一致；silhouette 為內部指標、不同空間下僅供參考。")
H("4.2 與其他方法對照", 2)
table(["方法", "覆蓋率", "noise", "purity", "NMI", "silhouette"], [
    ["Regex 關鍵字（規則下限）", "0.44", "56%", "0.977", "0.436", "0.26"],
    ["TF-IDF + HDBSCAN（傳統 ML）", "0.64", "36%", "0.986", "0.237", "0.71"],
    ["BERT + UMAP + K-Means（消融）", "1.00", "0%", "0.991", "0.262", "0.83"],
    ["BERT + UMAP + HDBSCAN（採用）", "0.93", "7%", "0.991", "0.271", "0.88"],
], widths=[2.4, 0.85, 0.7, 0.8, 0.7, 1.0])
bullet("採用方法同時解決「語意辨識不清」（覆蓋率 44%→93%）與「雜訊干擾」（noise 36%→7%）。")
bullet("BERT+K-Means 的 0% noise 是假象——強迫每點進群、無離群機制。")
img(OUT / "exp_slide_baselines.png", 6.0, "圖 2：分群方法 baseline 對照。")
H("4.3 設計驗證（否決的方案）", 2)
bullet("分開分群（per-user）：user 2 群數從 59 崩到 8（粒度崩塌）、失去跨人可比 → 採三人一起分群（pooled）。")
bullet("加價格 feature：BERT 已隱含價格訊號，UMAP 後顯式加入反而 noise↑、silhouette↓ → 純 UMAP 最佳。")
H("4.4 分群成果（25 超類型節選）", 2)
try:
    sm = pd.read_csv(OUT / "supertype_K25_members.csv").sort_values("n_rows", ascending=False)

    def top_items(s, n=6):
        return "、".join(p.split("(")[0] for p in str(s).split(" | ")[:n])
    rows = []
    for _, r in sm.head(10).iterrows():
        rows.append([r["custom_name"], int(r["n_distinct_items"]), int(r["n_rows"]),
                     f'{r["unit_price_p50"]:.0f}', top_items(r["top_items"])])
    table(["超類型", "品項數", "筆數", "單價中位", "代表品項"],
          rows, widths=[1.2, 0.6, 0.55, 0.65, 3.2])
except Exception as e:
    P(f"[讀取 supertype_K25_members.csv 失敗：{e}]")

# ============================================================
# 5. 第三階段：個人化消費增量分析
# ============================================================
H("5. 第三階段：個人化消費增量分析", 1)
H("5.1 方法", 2)
P("每人「這個月」=最新月（2026-05）、「平常」=之前各月月均。對每個超類型做兩因子拆解：")
P("　變貴 = (本月單價 − 平常單價) × 本月數量　；　買更多 = (本月數量 − 平常月均數量) × 平常單價")
P("　增量 ΔS = 變貴 + 買更多 = 這個月比一個典型月在此類型上的花費差")
P("分析門檻（per-user、每超類型）：基期 ≥ 2 月 且（月均量 ≥ 10 或 月均花費 ≥ 100），過濾買太少、比漲價沒意義的類別。")
H("5.2 增量分析結果", 2)
table(["user", "本月", "最大貢獻超類型", "解讀"], [
    ["0", "多花 945", "正餐主食 +1162", "均價 115→256，含一筆 1199 元韓式烤肉聚餐 → 一次性高價，非系統性漲價"],
    ["1", "少花 720", "正餐主食 +651", "正餐量價齊升，但手搖/咖啡大幅少買抵消 → 整體少花"],
    ["2", "多花 285", "速食小食 +302", "瓶裝茶飲漲 67%、速食「買更多」（6.3→11 份）"],
], widths=[0.5, 0.9, 1.5, 3.4])
img(OUT / "increment_all_users.png", 6.4,
    "圖 3：三位 user 增量拆解（紅=變貴、藍=買更多、黑菱形=總增量 ΔS）。")
H("5.3 核心發現", 2)
bullet("正餐主食是最大增量來源（user 0/1 皆是最大貢獻超類型）。")
bullet("增量由「變貴」與「買更多」共同驅動，兩因子拆解能明確區分成因。")
bullet("個人通膨是「品項分化」的，不是齊漲：user 1 正餐 +19%、手搖甜點 −7%、手搖麵食 −19%。")
bullet("「變貴」須分辨真漲價 vs 偶發高價：user 0 正餐 +123% 來自單筆聚餐，可下鑽品項層級回查。")
bullet("部分類型是「買更多」非「變貴」：user 2 速食增量幾乎全來自量。")

# ============================================================
# 6. 延伸洞察
# ============================================================
H("6. 延伸洞察（跨全時序）", 1)
P("把超類型沿 9 個月攤開，能看到單月增量看不到的長期型態：")
bullet("季節性：user 0、2 在 2026-01（寒假）消費暴跌（user 2 僅 39 元），user 1 相反 → 校園行事曆季節性。")
bullet("結構性遷移：user 1 手搖甜點 1 月衝頂後下滑、正餐持續走高，兩線交叉 = 消費重心由「飲料零食」轉「正餐」。")
bullet("長期習慣：user 0 速食量從 24 斷崖式掉到個位數後不再回升（戒掉）；user 1 瓶裝茶飲量翻倍。")
bullet("單月暴衝：正是成員 B 異常偵測的目標，本層提供「異常落在哪個商品類型」的可解釋線索。")
img(OUT / "timeseries_all_users.png", 6.4,
    "圖 4：三位 user × 三指標（總花費 / 購買數量 / 平均單價）時序。")

# ============================================================
# 7. 總結與限制
# ============================================================
H("7. 總結與限制", 1)
P("總結：", bold=True)
bullet("以 BERT + UMAP + HDBSCAN 建立高品質商品語意分群（noise 7%、純度 0.99），有完整方法與 baseline 驗證。")
bullet("設計兩層商品分類體系（細群追價 + 超類型講故事），並以實驗否決分開分群與加價格 feature。")
bullet("完成個人消費增量分析，拆解「變貴 vs 買更多」，揭示個人通膨的品項分化特性。")
bullet("跨時序洞察為成員 B 的時序預測與異常偵測提供可解釋的商品類型線索。")
P("限制：", bold=True)
bullet("資料規模小（3 人、2613 筆）且飲食佔 82%，結論主要適用高頻飲食類；雖以兩階段分群將粒度切得更合理，但更多元資料會更好。")
bullet("發票覆蓋不全（缺現金/轉帳/訂閱）；超類型命名為人工（可重現但帶主觀）；以單月當「本月」對單筆大額較敏感。")

out_path = OUT.parent / "member_C_report.docx"
doc.save(str(out_path))
print(f"已輸出 {out_path}")
print(f"段落數: {len(doc.paragraphs)}, 表格數: {len(doc.tables)}")
