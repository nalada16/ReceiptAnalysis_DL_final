# 成員 C — 商品語意分群、個人通膨拆解與整合

> 對應企劃書第八章「成員 C：語意分群、通膨拆解與整合報告」。
> 本資料夾把成員 A 的分類結果（`all_user_6_label.csv` + `receipt_embeddings.npy`）
> 轉成「商品語意群 → 重複購買群單價時序 → 量效應/價效應拆解」，
> 並產出報告所需的表格與圖。

---

## 0. 在這之前：我接到什麼、輸出什麼

| 上游（成員 A 已交付） | 說明 |
|---|---|
| `../all_user_6_label.csv` | 2613 筆明細，3 位使用者，2025-09～2026-05，6 類標籤 |
| `../receipt_embeddings.npy` | **(2613, 768)** BERT `[CLS]` embedding，**逐列對齊 CSV** |

> ⚠️ 我**不需要重跑 BERT**，直接用現成 embedding。`c_common.load_data()` 會 `assert`
> 兩者列數一致，若成員 A 重新匯出資料但沒同步 embedding 會立刻報錯。

| 我的交付（→ `outputs/`） | 對應企劃書 |
|---|---|
| 商品語意分群結果 | 「商品分群結果」 |
| 重複購買群單價時序 | 「單價時序追蹤」 |
| 量效應/價效應拆解案例 | 「通膨拆解案例」 |
| 系統架構圖 + 整合報告 | 「完整系統架構圖、最終報告」（步驟 4，需成員 B 結果） |

---

## 1. 環境（uv）

本專案用 **uv** 管理，不要用 pip。根目錄已有 `pyproject.toml` / `uv.lock`。

```powershell
# 在專案根目錄 D:\ReceiptAnalysis_DL_final
uv sync                       # 依 lock 還原環境
# 新增套件範例（會自動更新 pyproject + lock）
uv add <package>
```

已安裝關鍵套件：`bertopic`、`umap-learn`、`hdbscan`、`jieba`、`scikit-learn`、`pandas`、`matplotlib`。

執行腳本一律用 `uv run`：

```powershell
cd member_c
uv run python step1_clustering.py
uv run python step2_unit_price.py
uv run python step3_inflation.py
```

> Windows 主控台是 cp950，印中文會 `UnicodeEncodeError`。`c_common.py` 已在載入時
> 把 stdout/stderr 切成 UTF-8；所有 CSV 也以 `utf-8-sig` 輸出（Excel 開不亂碼）。

---

## 2. 檔案結構

```
member_c/
├── c_common.py                  # 共用：載資料、欄名常數、單價計算、分群門檻
├── step1_clustering.py          # BERTopic + UMAP + HDBSCAN 商品語意分群（主流程）
├── step1b_cluster_experiments.py# 分群方法對照實驗（3 降維 × 3 分群）
├── step1c_slide_baselines.py    # 簡報 Layer3 四列對照（Regex / TF-IDF+HDBSCAN / BERT+KMeans / Ours）
├── step2_unit_price.py          # 重複購買群篩選 + 每月加權單價時序
├── step3_inflation.py           # 量效應 / 價效應拆解（個人通膨，主流程）
├── step3b_decomp_experiments.py # 拆解替代方案對照（分群鍵、兩項/三項）
├── README_memberC.md            # 本檔（流程 + 完整替代方案）
├── EXPERIMENTS.md               # 實驗紀錄與數字對照（給報告引用）★
└── outputs/                     # 所有產出（csv + png）
```

關鍵共用設定（`c_common.py`）：

```python
MIN_COUNT  = 5   # 重複購買群至少出現 5 次（企劃書 5~10）
MIN_MONTHS = 2   # 至少跨 2 個月才算可比較單價
```

---

## 3. Pipeline 三步驟

### 步驟 1 — 商品語意分群（`step1_clustering.py`）

**目的**：把 2613 筆明細依語意自動分群，找出「同一種商品/同一店家品項」的群。

**做法**：`現成 embedding → UMAP 降維(cosine, 5 維) → HDBSCAN 分群 → BERTopic c-TF-IDF 主題詞`。
BERTopic 以 `embedding_model=None` 餵入現成向量，**執行時不下載任何模型**。

**目前結果**：約 **110 群、noise ≈ 7%**。分群品質佳，例如：

| 群 | 內容（最高頻品名） |
|---|---|
| 美式咖啡群 | 大冰美式 / 特大冰美式 / 中冰美式（同飲品不同容量） |
| 雞肉堡群 | 嫩切雞肉堡6吋 / 厚切嫩牛堡6吋 / 鮮嫩雞柳堡6吋 |
| 飯糰群 | 經典肉鬆飯糰 / 雞肉飯飯糰 / 茶葉蛋 |

**輸出**：`step1_clusters.csv`（每筆明細的 `topic`）、`step1_topics.csv`（每群 `top_items` 標籤）、`step1_umap_scatter.png`。

> 中文 c-TF-IDF 關鍵詞對「短品名」效果差（常抽到空字串/數字），所以**額外用「每群最高頻
> 品名」當人類可讀標籤**（`top_items` 欄）。這比 c-TF-IDF 關鍵詞更適合商品群命名。

### 步驟 2 — 重複購買群篩選 + 單價時序（`step2_unit_price.py`）

**目的**：挑出可比較、夠頻繁的商品群，追蹤其每月單價。

**分群鍵兩種模式**（`--mode`）：

- `exact`（預設）：同 `(品名, 店名)` 視為同商品。**通膨拆解的主力**，最可靠。
- `cluster`：用步驟 1 的語意群（把拼寫變體合併），當對照與探索。需先跑完 step1。

**篩選**：出現 `>= MIN_COUNT` 次且跨 `>= MIN_MONTHS` 個月。目前 exact 模式得 **95 個群**。

**單價定義**：每群每月 `加權平均單價 = Σ金額 / Σ數量`（不是單價欄的算術平均，避免一次買多件被扭曲）。

**輸出**：`step2_repeat_groups.csv`、`step2_unit_price_ts.csv`、`step2_unit_price_top.png`。

```powershell
uv run python step2_unit_price.py            # exact, 分使用者
uv run python step2_unit_price.py --mode cluster --pooled   # 語意群, 全體合併
```

### 步驟 3 — 量效應 / 價效應拆解（`step3_inflation.py`）

**目的**：回答企劃書核心問題「支出增加，是買更多還是變貴？」

對每群比較基期(P0,Q0)與當期(P1,Q1)：

```
Delta        = P1·Q1 − P0·Q0        （= 當期金額 − 基期金額）
price_effect = (P1 − P0)·Q1          （價：相似商品變貴/便宜）
qty_effect   = (Q1 − Q0)·P0          （量：買更多/更少）
```

> 數學上 `price_effect + qty_effect ≡ Delta`，所以 `residual` 應為 0（已驗證）。
> 這是把交互項併入價效應的「精確兩項拆解」。若想分出交互項，見下方替代方案。

**目前結果（各群取自身最早/最晚月為基期/當期）**：

| user | total_delta | price_effect | qty_effect | n_groups |
|---|---:|---:|---:|---:|
| 0 | 272 | −129 | 401 | 25 |
| 1 | −794 | 1252 | −2046 | 63 |
| 2 | −705 | 365 | −1070 | 7 |

漂亮的單一案例（清燉牛肉麵）：`P0=208 → P1=266.5`（漲價）、`Q 1→4`（買更多），
`delta=+858 = 價234 + 量624`。列印/停車費則是單價不變的純量效應。

**輸出**：`step3_decomposition.csv`、`step3_user_summary.csv`、`step3_waterfall.png`。

```powershell
uv run python step3_inflation.py                     # 各群自動基期/當期
uv run python step3_inflation.py --base 2025-10 --cur 2026-04   # 固定月對月
```

---

## 4. 替代方案（演算法選型 Plan B）

> 期末報告的價值在於「比較與選擇」。以下列出每個環節若主方法效果不好可以換什麼，
> 以及**判斷標準**與**怎麼在本程式碼切換**。
> **其中 4.1/4.2（降維、分群）與 4.5（分群鍵）已實際跑成對照實驗，數字見 [`EXPERIMENTS.md`](EXPERIMENTS.md)。**
>
> ```powershell
> uv run python step1b_cluster_experiments.py   # 3 降維 × 3 分群 對照
> uv run python step3b_decomp_experiments.py    # exact vs cluster、兩項 vs 三項
> ```

### 4.1 降維（目前：UMAP）

| 何時換 | 替代 | 怎麼做 |
|---|---|---|
| UMAP 群界線糊、隨機性大 | **PCA**（線性、可重現、快） | 把 `step1` 的 `UMAP(...)` 換成 `sklearn.decomposition.PCA(n_components=30)`；HDBSCAN 改用 `metric="euclidean"` 不變 |
| 想保留全域結構 | **t-SNE**（僅 2D 視覺化，不建議拿來分群） | 只用在 `plot_scatter` |
| embedding 已很乾淨 | **不降維**，直接對 768 維跑 HDBSCAN（用 `metric="euclidean"` 或先 L2 normalize 用 cosine） | 拿掉 UMAP 步驟 |

調參：UMAP `n_neighbors` 調小(5~10)→更看局部、群更碎；調大(30+)→群更大更糊。

### 4.2 分群（目前：HDBSCAN）

| 何時換 | 替代 | 理由 / 怎麼做 |
|---|---|---|
| noise 比例過高（>20%）、太多商品落入 −1 | **KMeans** | 強制每點都進群、無 noise；缺點要先給 K。`sklearn.cluster.KMeans(n_clusters=K)`，K 用 silhouette 掃 |
| 群數需要可控、想看階層 | **Agglomerative（Ward）** | `sklearn.cluster.AgglomerativeClustering`，可剪不同高度看粗細 |
| 想要軟分群（一品項可屬多群） | **GMM** | `sklearn.mixture.GaussianMixture`，輸出機率 |
| HDBSCAN 太碎 | 調 `min_cluster_size`↑（群更大）、`cluster_selection_method="leaf"`→更細 / `"eom"`→更粗 | 已參數化在 step1 頂部 |

> 評估分群好壞：**silhouette score**、**每群是否語意一致（人工抽查 top_items）**、
> **noise 比例**。報告可放「HDBSCAN vs KMeans」的對照表。

### 4.3 主題詞 / 群標籤（目前：c-TF-IDF + jieba，效果差 → 改用最高頻品名）

| 替代 | 怎麼做 |
|---|---|
| **最高頻 `item_clean`**（目前採用） | 已內建 `top_items` 欄，對短品名最實用 |
| **KeyBERT / MMR** | BERTopic 的 `representation_model=KeyBERTInspired()`，需 embedding 較長文字才有效 |
| **LLM 命名** | 把每群 top_items 丟 LLM 產一句話群名（成本低，群數才 ~110） |

### 4.4 通膨拆解（目前：精確兩項拆解）

| 替代 | 公式 | 何時用 |
|---|---|---|
| **三項拆解（含交互）** | `Delta = Q0·ΔP + P0·ΔQ + ΔP·ΔQ` | 想把「又漲又多買」的交互效果獨立出來報告 |
| **Laspeyres / Paasche 指數** | 物價指數 `Σ P1Q0 / Σ P0Q0` | 想算「整體個人 CPI」式的單一通膨率 |
| **對數拆解（LMDI）** | 能源/經濟常用，可加總、無殘差 | 群數多、要嚴謹可加總時 |

| 基期/當期選法 | 取捨 |
|---|---|
| 各群自身最早/最晚月（目前預設） | 用最多資料，但各群期間不同，跨群加總語意較弱 |
| 固定月對月（`--base/--cur`） | 故事乾淨「X 月 vs Y 月」，但只能用兩月都出現的群 |
| 前半 vs 後半 | 折衷，樣本較穩 |

### 4.5 分群鍵（exact vs cluster）

- `exact` 最可靠，**通膨拆解建議用它**。
- `cluster` 能把「珍奶 / 大珍奶 / 珍珠奶茶」合併，覆蓋率更高，但語意群可能混入不同單價商品 →
  拆解時可能高估價效應。報告可兩者並陳，當作 error analysis。

---

## 5. 資料限制（報告必須寫）

- **嚴重偏食**：飲食佔 ~82%，重複購買群幾乎都是飲料/食品/超商，**通膨拆解結論僅適用於高頻
  飲食類**，不能宣稱涵蓋使用者全部消費。
- **發票覆蓋率**：現金小攤、訂閱、轉帳沒有發票，分析範圍限「可由電子發票觀察到的消費」。
- **使用者異質**：user 2 從 2025-10 才有資料、筆數少（428），單獨結論信賴度較低。
- **上游分類誤差**：步驟 1 的群品質取決於 A 的 embedding；報告可做 oracle label vs BERT label
  的 error propagation 對照（與企劃書 6.2 呼應）。

---

## 6. 步驟 4 — 整合（待辦，依賴成員 B）

目前資料夾**還沒有成員 B 的每日消費時序 / 預測 / 異常結果**。整合報告需要向 B 取得：

- 每日消費矩陣（`X_t = [飲食,交通,購物,娛樂,醫療健康,教育]` + `has_transaction` mask）
- LSTM 預測誤差表、LSTM-AE 異常案例

拿到後我負責：

1. 畫**完整系統架構圖**（A 清洗+分類 → B 時序預測+異常 → C 分群+通膨）。
2. 彙整三人實驗表（Task1 分類、Task2 預測、異常、通膨拆解）成一份報告與簡報。
3. 串接：用步驟 1 的群 embedding 平均，作為成員 B LSTM 的語意輔助特徵（企劃書 5.1）。

---

## 7. 一鍵重跑

```powershell
cd D:\ReceiptAnalysis_DL_final
uv sync
cd member_c
uv run python step1_clustering.py
uv run python step2_unit_price.py
uv run python step3_inflation.py
# 產物全在 member_c/outputs/
```
