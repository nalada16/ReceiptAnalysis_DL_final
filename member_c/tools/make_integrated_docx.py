# -*- coding: utf-8 -*-
"""生成整份專案的完整書面報告 Word 檔（INTEGRATED_REPORT.docx）。

依據 INTEGRATED_REPORT.md，整合三位成員的成果：
  成員 A（語意分類）→ 成員 B（時序預測+異常）→ 成員 C（商品分群+增量診斷）
並嵌入整體架構圖與成員 C 的關鍵圖表。
"""
from __future__ import annotations
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import sys
sys.path.insert(0, r"D:\ReceiptAnalysis_DL_final\member_c")
import c_common as cc

OUT = cc.OUT_DIR
ROOT = OUT.parent.parent
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
r = title.add_run("電子發票個人化消費行為分析\n完整研究報告")
r.bold = True
r.font.size = Pt(22)
r.font.color.rgb = RGBColor(0x8B, 0x1A, 0x2B)
_cjk(r)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("成員 A 語意分類　·　成員 B 時序預測與異常偵測　·　成員 C 商品分群與增量診斷")
r.font.size = Pt(11)
_cjk(r)
doc.add_paragraph()

P("專案定位：在「真實電子發票資料有限、消費行為受外部事件影響、每人消費頻率不同」的條件下，"
  "建立一套可重跑、可比較、可解釋的個人化消費行為診斷系統。三人各司一個分析層次，"
  "串成「分類 → 預測/異常 → 增量歸因」的完整 pipeline。", italic=True)

# ============================================================
# 0. 系統總覽
# ============================================================
H("0. 系統總覽", 1)
table(["層次", "負責人", "任務", "核心方法", "主要產出"], [
    ["Task 1", "成員 A", "發票明細語意分類", "BERT fine-tuning（CKIP 繁中）", "6 類消費標籤 + 768 維向量"],
    ["Layer 1-2", "成員 B", "日常預測 + 事件異常", "LSTM / MLP / ARIMA + Autoencoder", "短期趨勢預測 + 異常分數"],
    ["Layer 3", "成員 C", "商品分群 + 增量診斷", "BERT + UMAP + HDBSCAN + AHC + 兩因子拆解", "商品類型體系 + 增量歸因"],
], widths=[0.9, 0.7, 1.6, 2.3, 1.7])
P("資料流：成員 A 輸出「已分類明細 + BERT 向量」→ 成員 B 用分類後金額做每日時序、"
  "成員 C 用 BERT 向量做商品分群 → 兩者互為可解釋線索（C 的「異常落在哪個商品類型」可解釋 B 的異常日）。")
img(OUT / "system_architecture_full.png", 6.2,
    "圖 0：整體專案系統架構。Task 1 分類與向量化 → Task B 時序/異常、Task C 分群/增量 並行 → 整合診斷輸出。")

# ============================================================
# 1. 資料來源
# ============================================================
H("1. 資料來源與共同基礎", 1)
table(["項目", "內容"], [
    ["資料來源", "財政部電子發票整合服務平台匯出的個人消費 CSV（組員與親友自願提供）"],
    ["使用者", "3 位（編號 0 / 1 / 2）"],
    ["期間", "2025-09 ~ 2026-05（約 9 個月）"],
    ["明細筆數", "~2613 筆（金額 ≤ 0 的折扣/退款已移除）"],
    ["隱私處理", "移除發票號碼、載具條碼、買賣方統編、完整地址"],
], widths=[1.3, 5.0])
P("消費類別分布（高度不平衡）：")
table(["類別", "筆數", "佔比"], [
    ["飲食", "2,153", "82.4%"],
    ["交通", "164", "6.3%"],
    ["購物", "131", "5.0%"],
    ["娛樂", "72", "2.8%"],
    ["教育", "69", "2.6%"],
    ["醫療健康", "24", "0.9%"],
], widths=[1.4, 1.2, 1.2])
P("飲食佔 82% 是日常消費的自然分布，但對評估指標選擇（Macro F1）與下游分析（飲食為主）皆有直接影響。", italic=True)

# ============================================================
# 2. Task 1（成員 A）
# ============================================================
H("2. Task 1（成員 A）：發票明細語意分類", 1)
H("2.1 任務與核心難點", 2)
P("將每筆發票明細歸到 6 類（飲食/交通/購物/娛樂/教育/醫療健康）。核心難點：發票品名資訊量極低"
  "——「商品」「餐飲」「Food Court」等空洞品名在真實發票中很常見。關鍵設計決策是把店家名稱與"
  "金額區間一起送進模型。")
H("2.2 模型設計", 2)
bullet("主模型：ckiplab/bert-base-chinese（繁中 BERT），輸入 [CLS] 品名 [SEP] 店家 [SEP] 金額區間 [SEP]，取 [CLS] 接 linear head。")
bullet("金額離散化：5 區間（VERY_LOW / LOW / MID / HIGH / VERY_HIGH），切點 50 / 150 / 500 / 1500 元。")
bullet("訓練：lr 2e-5、batch 16、5 epochs、max_len 64；以 GroupShuffleSplit（品名+店家為 group key）切 80/20，避免資料洩漏。")
bullet("標注：半自動（LLM 批次預標 + 人工審核），對「品名+店家」去重後只標一次以控成本。")
H("2.3 評估結果", 2)
table(["模型", "Accuracy", "Macro F1", "Weighted F1"], [
    ["BERT（品名+店家+金額）主模型", "0.98", "0.92", "0.98"],
    ["BERT（僅品名）", "0.95", "0.82", "0.95"],
    ["TF-IDF + MLP", "0.81", "0.44", "0.81"],
    ["TF-IDF + Logistic Regression", "0.68", "0.40", "0.74"],
], widths=[3.0, 1.1, 1.1, 1.1])
P("關鍵結論：")
bullet("BERT 語言表徵有決定性優勢：傳統 TF-IDF 系列在少數類別 Macro F1 僅 ~0.40，BERT 僅品名版就達 0.82。")
bullet("店家 + 金額是必要而非輔助：Macro F1 從 0.82 提升到 0.92，10 個百分點集中在交通、購物、教育等品名語意最模糊的類別。")

# ============================================================
# 3. Task B（成員 B）
# ============================================================
H("3. Task 2 / Layer 1-2（成員 B）：消費預測與異常偵測", 1)
H("3.1 問題重定義", 2)
P("真實消費不是同質時間序列：日常支出有規律，但事件型支出（醫療、旅遊、聚餐、代買）無法只靠"
  "過去 30 天預知。因此拆成兩個互補層次：Layer 1 routine spending forecasting（日常趨勢預測）、"
  "Layer 2 event-driven anomaly diagnosis（事件異常診斷）。")
H("3.2 資料工程", 2)
P("將分類後明細彙整成每日消費矩陣，補齊缺失日期為 0 並保留 has_transaction mask（缺紀錄≠沒消費）。"
  "加入時間特徵，以 sliding window「過去 30 天 → 未來 7 天」產生監督樣本，依時間序切 70/15/15。"
  "不同 user 資料密度差異大（user 1 有消費比例 96.3%、user 2 僅 60.1%），故以 user_id 分開建模。")
H("3.3 預測模型與「不能只看 MAE」", 2)
P("比較 MovingAverage7 / WeekdayAverage / ARIMA(1,0,1) / MLP / LSTM_no_time / LSTM_time。"
  "關鍵方法論貢獻：新增 model behavior diagnostics——計算 prediction_std / actual_std，"
  "若 ratio < 0.10 標記為 collapsed_prediction（預測線退化成水平均值線），避免把「MAE 低但幾乎不動」"
  "的退化模型誤判為最佳。")
table(["user", "有消費比例", "採用模型", "Overall MAE", "Normal MAE", "High-spend MAE", "選擇理由"], [
    ["0", "69.8%", "LSTM_time", "189.8", "77.4", "1573.8", "序列模型未退化、整體 MAE 最低"],
    ["1", "96.3%", "MLP", "427.9", "213.5", "1922.5", "LSTM 雖 MAE 低但被標 collapsed，MLP 較穩"],
    ["2", "60.1%", "MLP", "333.7", "140.7", "1683.1", "資料稀疏，window 整體形狀比逐日遞迴更有用"],
], widths=[0.5, 0.9, 1.0, 0.9, 0.8, 0.95, 1.6])
P("深度模型不一定全面勝出：user 0 適合 LSTM、user 1 與 2 適合 MLP，支撐 per-user model suitability。", italic=True)
H("3.4 Routine / Event 分層與異常偵測", 2)
bullet("所有 user 的 Normal-day MAE 遠低於 High-spend-day MAE（如 user 0：77 vs 1574），由數據證明日常型與事件型不該視為同一任務。")
bullet("LSTM Autoencoder（hidden 48、latent 24）以訓練期 reconstruction error 設個人化門檻（p95），解決固定金額門檻對不同消費水準不公平。")
bullet("Threshold sensitivity（p90/p95/p97）：p90 敏感、p97 保守、p95 折衷，讓異常偵測從任意門檻變成可討論的設計選擇。")
bullet("合成資料驗證：注入已知 anomaly，LSTM Autoencoder recall = 1.00、precision 隨 profile 變化，印證「異常偵測是提醒回查，不是絕對判定」。")

# ============================================================
# 4. Task C（成員 C）
# ============================================================
H("4. Task 2 / Layer 3（成員 C）：商品分群與消費增量診斷", 1)
P("一句話：把成員 A 的分類結果，用 BERT 語意把「品項」聚成「商品類型」，再追蹤每個類型沿時間的"
  "單價與數量變化，回答——這個月比平常多花的錢，花在哪些類型、是「變貴」還是「買更多」。", italic=True)
img(OUT / "layer3_architecture.png", 4.3,
    "圖 4-0：成員 C 三階段架構（語意分群 → 超類型合併 → 個人化增量分析）。")

H("4.1 三階段方法", 2)
P("階段一｜商品語意分群（精準層）：重用成員 A 的 768 維 BERT 向量（遷移學習，不重訓）"
  "→ UMAP(5D, cosine) → HDBSCAN(min_cluster_size=10)，得 ~110 個語意一致細群，自動標記 noise。")
P("階段二｜超類型合併（故事層）：對細群中心向量（~108 個點）做 Agglomerative Hierarchical "
  "Clustering（Ward）合併成 25 個可命名超類型。用 AHC 而非 HDBSCAN：可直接指定 K、點少算力低、"
  "結果具確定性，且「一個細群一票」不受購買頻率灌水。")
P("階段三｜個人化消費增量分析：每人「這個月」=最新月、「平常」=之前各月月均，做兩因子拆解：")
P("　　變貴 price_effect = (P1−P0)·Q1　；　買更多 qty_effect = (Q1−Q0)·P0　；　ΔS = 變貴 + 買更多")
P("分析門檻（per-user、每超類型）：基期 ≥ 2 月 且（月均量 ≥ 10 或 月均花費 ≥ 100）。")

H("4.2 分群品質驗證", 2)
P("降維 × 分群對照（以 6 類標籤為外部基準）：")
table(["組合", "noise", "silhouette", "說明"], [
    ["raw768 + HDBSCAN", "32%", "0.70", "高維密度估計失準"],
    ["PCA30 + HDBSCAN", "34%", "0.69", "線性降維保不住語意流形"],
    ["UMAP5 + HDBSCAN（採用）", "7%", "0.88", "noise 砍 ~5 倍、品質最高"],
], widths=[2.2, 0.9, 1.1, 2.1])
P("與 baseline 對照（皆為實際執行數據）：")
table(["方法", "覆蓋率", "noise", "purity", "NMI", "silhouette"], [
    ["Regex 關鍵字（規則下限）", "0.44", "56%", "0.977", "0.436", "0.26"],
    ["TF-IDF + HDBSCAN（傳統 ML）", "0.64", "36%", "0.986", "0.237", "0.71"],
    ["BERT + UMAP + K-Means（消融）", "1.00", "0%", "0.991", "0.262", "0.83"],
    ["BERT + UMAP + HDBSCAN（採用）", "0.93", "7%", "0.991", "0.271", "0.88"],
], widths=[2.4, 0.85, 0.7, 0.8, 0.7, 1.0])
P("注意：purity、NMI 為跨方法一致的外部指標；silhouette 為內部指標，在不同特徵空間下僅供參考。"
  "三項指標共同支持 UMAP + HDBSCAN 在 noise 與群質量上的優勢。", italic=True)
img(OUT / "exp_slide_baselines.png", 6.0, "圖 4-1：分群方法 baseline 對照。")
P("否決的兩條路線（誠實的負面結果）：")
bullet("分開分群（per-user）：user 2 群數從 59 崩到 8（粒度崩塌），失去跨人可比性 → 採 pooled。")
bullet("加價格 feature：BERT 已隱含價格訊號，UMAP 後顯式 concat 反而使 noise 上升（5.6%→10.5%）、silhouette 下降（0.87→0.81）→ 純 UMAP 最佳。")

H("4.3 增量分析主結果", 2)
table(["user", "本月", "最大貢獻超類型", "解讀"], [
    ["0", "多花 945", "正餐主食 +1162", "均價 115→256（含一筆 1199 元韓式烤肉聚餐）→ 一次性高價，非系統性漲價"],
    ["1", "少花 720", "正餐主食 +651", "正餐量價齊升，但手搖/咖啡大幅少買抵消 → 整體少花"],
    ["2", "多花 285", "速食小食 +302", "瓶裝茶飲漲 67%、速食「買更多」（6.3→11 份）"],
], widths=[0.5, 0.9, 1.5, 3.4])
img(OUT / "increment_all_users.png", 6.4,
    "圖 4-2：三位 user 增量拆解（紅=變貴、藍=買更多、黑菱形=總增量 ΔS）。")
P("核心發現：")
bullet("正餐主食是最大增量來源（user 0/1 皆是最大貢獻超類型）。")
bullet("增量由「變貴」與「買更多」共同驅動，兩因子拆解能明確區分。")
bullet("個人通膨是「品項分化」的，不是齊漲：user 1 正餐 +19%、但手搖甜點 −7%、手搖麵食 −19%。")
bullet("「變貴」須分辨真漲價 vs 偶發高價：user 0 正餐 +123% 來自單筆聚餐，可下鑽品項層級回查驗證。")
bullet("部分類型是「買更多」非「變貴」：user 2 速食增量幾乎全來自量。")

H("4.4 跨時序延伸洞察", 2)
bullet("季節性：user 0、2 在 2026-01（寒假）消費暴跌（user 2 僅 39 元），user 1 相反 → 校園行事曆季節性。")
bullet("結構性遷移：user 1 手搖甜點 1 月衝頂後下滑、正餐持續走高，兩線交叉 = 消費重心由「飲料零食」轉「正餐」。")
bullet("長期習慣：user 0 速食量從 24 斷崖式掉到個位數後不再回升（戒掉）；user 1 瓶裝茶飲量翻倍。")
bullet("單月暴衝：可作為成員 B 異常偵測的「異常落在哪個商品類型」的可解釋線索。")
img(OUT / "timeseries_all_users.png", 6.4,
    "圖 4-3：三位 user × 三指標（總花費 / 購買數量 / 平均單價）時序。")

# ============================================================
# 5. 三層整合
# ============================================================
H("5. 三層整合：系統如何串接", 1)
bullet("A → B：成員 B 用成員 A 的分類金額做每日矩陣與 routine/event 分層。")
bullet("A → C：成員 C 直接複用成員 A 的 BERT 中間層向量做下游分群（transfer / representation learning），不重新訓練，是深度學習表示重用的具體實踐。")
bullet("B ↔ C：成員 B 標出「哪一天異常」，成員 C 解釋「那筆異常落在哪個商品超類型、是變貴還是買更多」，兩者互補形成「偵測 + 歸因」的完整診斷。")

# ============================================================
# 6. 限制
# ============================================================
H("6. 整體限制", 1)
P("共通限制：", bold=True)
bullet("樣本數有限：僅 3 位使用者，不能宣稱泛化到所有人。")
bullet("時間長度有限：約 9 個月，難學到年度季節性。")
bullet("發票覆蓋不全：現金、轉帳、小攤、訂閱服務可能無紀錄。")
bullet("資料偏食：飲食佔 82%，結論主要適用高頻飲食類。")
P("各層特有：", bold=True)
bullet("A：少數類別（醫療 24 筆）樣本過少，Recall 偏低；標注仰賴 LLM + 人工。")
bullet("B：高額事件缺外部特徵（聚餐/醫療/旅遊不在序列中）；異常無真實標籤，只能說「值得回查」。")
bullet("C：超類型命名為人工（可重現但主觀）；以單月當「本月」對單筆大額敏感；過濾門檻為經驗值。")

# ============================================================
# 7. 結論
# ============================================================
H("7. 結論", 1)
P("本專案完成一套可重跑、可比較、可解釋的個人化電子發票消費診斷系統，創新不在演算法本身，"
  "而在問題定義、資料處理、模型比較與決策解釋的整合：")
bullet("成員 A：以 BERT（品名+店家+金額）達 Macro F1 0.92，證明多欄位語意整合對低資訊量品名的決定性價值。")
bullet("成員 B：把消費拆成 routine 預測 + event 異常兩層，並以 model degradation check 與合成資料驗證避免「水平線假最佳」與「無 ground truth」兩個陷阱。")
bullet("成員 C：用 BERT + UMAP + HDBSCAN + AHC 建立兩層商品類型體系，以兩因子拆解揭示「個人通膨的品項分化」特性，並以實驗誠實否決分開分群與加價格 feature。")
P("三層接力把「小樣本、隱私敏感、資料不完整」的個人電子發票，轉換成「分類 → 預測/異常 → 增量歸因」"
  "的完整行為診斷流程；但誠實地說，它能在有限資料下估計趨勢、提示異常、歸因增量，"
  "不能宣稱完整且精準預測一個人的所有消費習慣。")

out_path = ROOT / "INTEGRATED_REPORT.docx"
doc.save(str(out_path))
print(f"已輸出 {out_path}")
print(f"段落數: {len(doc.paragraphs)}, 表格數: {len(doc.tables)}")
