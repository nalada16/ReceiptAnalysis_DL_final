# 人工命名流程（自己填 cluster 名字）

> 分群是自動的，但群的「名字」可以由你手動填，讓圖表標籤從
> 「金萱雙q/四季春…(54款)」變成你寫的「手搖飲」。流程設計成**填一次、之後一直沿用**。

---

## 三步驟

### 步驟 1：產生命名表
```powershell
uv run python step1_clustering.py        # 先有分群結果
uv run python step3d_supertypes.py       # （可選）要命名超類型才需先跑
uv run python make_naming_template.py    # 產生命名表
```
會在 `member_c/` 根目錄產生：

| 命名表 | 命名對象 | 給哪張圖用 |
|---|---|---|
| `cluster_names.csv` | 細群（~110） | `step3c_user{u}.png` |
| `supertype_names_K15/25/35.csv` | 超類型 | `step3d_K{K}_user{u}.png` |

### 步驟 2：用 Excel 填名字
打開 `cluster_names.csv`，**只改 `custom_name` 欄**，其他欄是給你判斷用的參考：

| topic | custom_name ←**你填這欄** | auto_label（參考） | n_rows | top_members（群內品項） |
|---|---|---|---|---|
| 4 | `美式咖啡` | 大冰美式 等3款 | 39 | 大冰美式:35 \| 特大冰美式:2 \| 中冰美式:2 |
| 7 | `麵店餐點` | 皮蛋乾麵 等15款 | 37 | 皮蛋乾麵:10 \| 燙青菜:5 \| 蛤蠣湯:4 |
| 2 | （留空＝用自動標籤） | 嫩切雞肉堡6吋 等7款 | 51 | … |

- 看 `top_members`（群內有哪些品項）就能判斷該叫什麼名字。
- **留空的群會自動用 `auto_label`**，不必每個都填。
- 存檔（Excel 存成 CSV UTF-8 即可）。

### 步驟 3：重跑出圖
```powershell
uv run python step3c_personal_inflation.py   # 細群命名 → 套用
uv run python step3d_supertypes.py           # 超類型命名 → 套用
```
圖表標籤就會變成你填的名字（終端會印「套用人工命名 N 個」）。

---

## 流程圖

```
step1_clustering ─► step1_clusters.csv
                          │
   make_naming_template ──┴─► cluster_names.csv / supertype_names_K{K}.csv
                          │       （custom_name 欄留空）
              你在 Excel 填 ▼
                          │
   step3c / step3d ───────┴─► 出圖：有填用你的名字、沒填用 auto_label
```

---

## 設計重點

- **填一次、永久沿用**：`make_naming_template.py` 是 **merge-safe** 的——重跑時會**保留你已填的
  名字**，只補新出現的群。所以即使之後重新分群、或新增資料，你舊的命名不會被洗掉。
- **不擋流程**：沒填名字也能正常出圖（用自動標籤），命名是純加分。
- **細群 / 超類型分開命名**：兩層各有命名表；超類型命名表依 K 各一份。
- 程式：命名表讀寫邏輯在 `naming.py`；產生器在 `make_naming_template.py`；
  套用點在 `step3c`（細群）與 `step3d`（超類型）。
