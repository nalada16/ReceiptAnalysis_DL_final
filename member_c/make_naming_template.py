"""產生/更新人工命名表（merge-safe，不會洗掉你已填的名字）。

用法：
  1. 先跑 step1_clustering.py（→ step1_clusters.csv）
  2. （可選）跑 step3d_supertypes.py（→ supertypes_K{K}.csv）
  3. 跑本檔 → 產生 cluster_names.csv 與 supertype_names_K{K}.csv
  4. 用 Excel 打開、在 custom_name 欄填名字（如「美式咖啡」），存檔
  5. 重跑 step3c / step3d 出圖，就會用你填的名字

欄位：custom_name（你填）、auto_label（自動標籤，供參考）、n_rows、top_members…
"""
from __future__ import annotations

import pandas as pd

import c_common as cc
import naming


def build_fine_template():
    """細群命名表（key=topic）。"""
    clus = pd.read_csv(cc.OUT_DIR / "step1_clusters.csv")
    clus = clus[clus["topic"] != -1]
    rows = []
    for t, sub in clus.groupby("topic"):
        vc = sub[cc.COL_ITEM].value_counts()
        auto = (f"{vc.index[0]} 等{sub[cc.COL_ITEM].nunique()}款"
                if sub[cc.COL_ITEM].nunique() > 1 else f"{vc.index[0]}（單品）")
        rows.append({
            "topic": int(t),
            "auto_label": auto,
            "n_rows": len(sub),
            "n_distinct_items": int(sub[cc.COL_ITEM].nunique()),
            "top_members": " | ".join(f"{i}:{c}" for i, c in vc.head(6).items()),
        })
    df = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    out = naming.merge_template(naming.FINE_NAMES, df, "topic")
    filled = (out["custom_name"].str.strip() != "").sum()
    print(f"細群命名表 → {naming.FINE_NAMES.name}：{len(out)} 群，已填 {filled}")


def build_super_template(K: int):
    """超類型命名表（key=super），來源 supertypes_K{K}.csv。"""
    path = cc.OUT_DIR / f"supertypes_K{K}.csv"
    if not path.exists():
        return
    tax = pd.read_csv(path)
    df = tax.rename(columns={"label": "auto_label", "member_fine_types": "top_members"})[
        ["super", "auto_label", "n_rows", "n_distinct_items", "top_members"]]
    out = naming.merge_template(naming.super_names_path(K), df, "super")
    filled = (out["custom_name"].str.strip() != "").sum()
    print(f"超類型命名表 K={K} → {naming.super_names_path(K).name}：{len(out)} 類，已填 {filled}")


def main():
    build_fine_template()
    for K in (15, 25, 35):
        build_super_template(K)
    print("\n→ 用 Excel 打開命名表，在 custom_name 欄填名字後，重跑 step3c / step3d 即可。")


if __name__ == "__main__":
    main()
