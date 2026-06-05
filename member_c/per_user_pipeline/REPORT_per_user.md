# per-user 對照管線 — 完整分析報告

> 對照實驗：把三人**各自分群**、再**各自做增量分析**，與主流程（pooled 一起分群）對照，
> 檢驗「分開分群」是否可行。結論：**對大資料量使用者可行，對小資料量使用者（user 2）崩潰**。
> 所有產出在 `per_user_pipeline/outputs/`。

---

## 1. 方法

| 步驟 | 程式 | 做法 |
|---|---|---|
| 分開分群 | `pu_step1_clustering.py` | 每人**只用自己的** embedding 跑 UMAP+HDBSCAN（參數同主流程：min_cluster_size=10） |
| 分開增量 | `pu_step3c_increment.py` | 用各自的群當「類型」，重用 step3c 邏輯（清理+一致性+基期≥2月且量/價門檻）做增量分析 |
| 分群品質對照 | `step1d_per_user_clustering.py` | per-user vs pooled 的 noise/coverage/silhouette/群數 |

---

## 2. 分開分群結果

| user | 筆數 | per-user 群數 | noise |
|---|---:|---:|---:|
| user 0 | 680 | 33 | 9.4% |
| user 1 | 1505 | 62 | 5.0% |
| **user 2** | 428 | **8** | 2.8% |

對照「pooled 限定該 user」的群數：user 0 **73→33**、user 1 **89→62**、user 2 **59→8**。

![per-user vs pooled 分群品質](outputs/exp_per_user_clustering.png)

> 反直覺：user 2 單獨跑 noise 反而下降（10%→3%），但這是**粒度崩塌**的假象——
> 群數從 59 暴跌到 8，把不同商品擠成 8 個粗團塊，homogeneity/silhouette 同步下降。

---

## 3. 分開增量分析結果

> 門檻：per-user 每類型須 **基期≥2月 且（月均量≥10 或 月均花費≥100）**。

| user | 可分析類型數 | 本月多花的錢 | 變貴貢獻 | 買更多貢獻 |
|---|---:|---:|---:|---:|
| user 0 | 10 | −1138 | −592 | −546 |
| user 1 | 39 | −1587 | −12 | −1575 |
| **user 2** | **3** | −302 | −33 | −269 |

### user 0（圖：`outputs/pu_step3c_user0.png`）

![user0](outputs/pu_step3c_user0.png)

| 類型 | 平常單價→本月 | 平常/月→本月 | 增量 |
|---|---|---|---:|
| 餐-十塊雞 等24款 | 311→94 | 3.25→2 份 | −824 |
| (a)北海鱈魚香絲(大) 等16款 | 97→50 | 2.5→2 | −141 |
| 桂格燕麥290ml 等6款 | 41→41 | 2.75→0 | −113 |

→ user 0 本月主要少花在「十塊雞」（單價腰斬+買更少）。

### user 1（圖：`outputs/pu_step3c_user1.png`）

![user1](outputs/pu_step3c_user1.png)

| 類型 | 平常單價→本月 | 平常/月→本月 | 增量 |
|---|---|---|---:|
| 母傳原湯清燉牛肉 等5款 | 229→253 | 2.25→6 份 | **+1003** |
| 黑蒜豚骨拉麵 等9款 | — | 2.62→0 | −478 |
| 韓式泡菜拉麵 等2款 | — | 1.75→0 | −434 |

→ user 1 與主流程結論一致：本月主要多花在清燉牛肉（買更多為主），但本月沒吃拉麵。

### user 2（圖：`outputs/pu_step3c_user2.png`）★ 問題所在

![user2](outputs/pu_step3c_user2.png)

| 類型 | 平常單價→本月 | 增量 |
|---|---|---:|
| 多喝水鹼性竹炭 等2款 | 48.9→48.9 | −155 |
| 加新入袋 a4 單線簿 等19款 | 13.5→10 | −105 |
| 味丹多喝水 等11款 | 38.4→28.7 | −42 |

→ **user 2 只剩 3 個可分析類型，且全是瓶裝水/文具**，完全代表不了真實消費。增量分析形同失效。

---

## 4. 與主流程（pooled）對照

| user | pooled 可分析類型(step3c) | per-user 可分析類型 |
|---|---:|---:|
| user 0 | 6 | 10 |
| user 1 | 33 | 39 |
| **user 2** | **4** | **3** |

跨人破碎度：被 ≥2 人購買的商品種類 13 種，pooled 下 100% 落同群（可比），per-user 下 0%（定義上不可能同群）。

---

## 5. 結論：為何主流程不採 per-user

> 可分析「數量」差距在新門檻下變小（不再是舊門檻的 8→4 砍半），所以否決理由以**穩固證據**為主：

1. **粒度崩塌**：user 2 單獨分群只分出 8 群（pooled 限定時 59 群），細商品群解析不出來。
2. **失去跨人可比**：被 ≥2 人購買的商品，pooled 100% 落同群、per-user 0%。
3. **小 user 增量無意義**：user 2 分開後只剩 3 個可分析類型、全是瓶裝水/文具。
4. **大資料量者沒有明顯好處**：user 0、1 數量與 pooled 相當，沒有理由為此犧牲 user 2。

→ **主流程採 pooled 分群（建公共商品字典）+ 分人增量分析（step3c/step3d）**。本對照管線作為「試過分開、對小 user 不可靠」的實證保留。

---

## 6. 檔案

| 檔案 | 內容 |
|---|---|
| `outputs/pu_clusters.csv` | 三人各自分群結果（每列 `pu_topic`） |
| `outputs/pu_step3c_increment_breakdown.csv` | per-user 增量逐類型明細 |
| `outputs/pu_step3c_user_summary.csv` | per-user 逐人增量摘要 |
| `outputs/pu_step3c_user{0,1,2}.png` | per-user 增量圖 |
| `outputs/exp_per_user_clustering.png` / `.csv` | 分群品質對照 |
| `outputs/exp_per_user_fragmentation.csv` | 跨人破碎度 |
