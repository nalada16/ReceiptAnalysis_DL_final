# per-user 對照管線（三人分開分群 → 分開增量分析）

> 這是一個**獨立對照實驗**，與 `member_c/` 主流程（pooled，三人一起分群）完全分開。
> 目的：實測「三人分開分群、再分開跑增量分析」會不會出問題。
> 所有產出都在本資料夾的 `outputs/`，不與主流程混在一起。

## 檔案

| 檔案 | 作用 |
|---|---|
| `pu_common.py` | 把上層 member_c 接進 path，並把輸出導向本資料夾 outputs/ |
| `pu_step1_clustering.py` | 三人**各自**跑 UMAP+HDBSCAN（參數同主流程），輸出 `pu_clusters.csv` |
| `pu_step3c_increment.py` | 用各自的 per-user 群當「類型」，重用主流程 step3c 邏輯跑增量分析 |
| `step1d_per_user_clustering.py` | per-user vs pooled 的分群**品質**對照（noise/coverage/silhouette/群數） |
| `outputs/` | 全部產出（csv + png） |

## 怎麼跑

```powershell
cd member_c\per_user_pipeline
uv run python pu_step1_clustering.py      # 三人各自分群
uv run python pu_step3c_increment.py      # 三人各自增量分析
uv run python step1d_per_user_clustering.py   # 分群品質對照（pooled vs per-user）
```

## 實測結論：技術上可跑，但小資料量使用者會「崩」

**可分析類型數（清理+一致性+每類型≥10筆後）：pooled 主流程 vs per-user**

| user | 筆數 | pooled 可分析類型 | per-user 可分析類型 |
|---|---:|---:|---:|
| user 0 | 680 | 19 | 25 |
| user 1 | 1505 | 54 | 53 |
| **user 2** | 428 | **8** | **4** |

- **大資料量者（user 0、1）**：分開跑沒問題，可分析類型數相當（user 0 甚至略多）。
- **小資料量者（user 2）**：分開跑後可分析類型**砍半（8→4）**，且 4 個全是瓶裝水/文具
  （多喝水、泰山、味丹、a4 簿），**完全無法代表他的真實消費** → 增量分析失去意義。

**為什麼**：per-user 分群因資料不足會**粒度崩塌**（user 2 單獨跑僅 8 群，見 `step1d`），
細商品群解析不出來；經清理與 ≥10 筆門檻後幾乎不剩東西。pooled 讓 user 2 借助
A/B 的共享商品結構，才解析得出較多可比類型。

**附帶代價**（見 `step1d` 與主流程 EXPERIMENTS.md）：per-user 還會失去跨人商品可比性，
且整體 homogeneity/silhouette 下降。

## 最終取捨

→ **主流程維持 pooled 分群**（建公共商品字典），增量分析在各自帳本上分人計算（step3c）。
per-user 全分開的版本對小資料量使用者不可靠，僅作為對照證據保留於此資料夾。
