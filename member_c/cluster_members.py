"""產生「商品類型群 → 成員品項」完整對照表，供報告/答辯查閱。

輸出：outputs/step3c_cluster_members.csv
欄位：topic / type_label / n_distinct_items / n_rows / users /
      unit_price_p10,p50,p90 / coherent(一致性是否通過) / members(品項:次數 …)
"""
from __future__ import annotations

import pandas as pd

import c_common as cc
from step3c_personal_inflation import GENERIC_KEYWORDS, MAX_PRICE_RATIO


def main():
    df, _ = cc.load_data(require_embeddings=False)
    clus = pd.read_csv(cc.OUT_DIR / "step1_clusters.csv")
    df["topic"] = clus["topic"].values
    df = df[df["topic"] != -1].copy()

    item_low = df[cc.COL_ITEM].astype(str).str.lower()
    df["is_generic"] = item_low.apply(lambda s: any(k in s for k in GENERIC_KEYWORDS))

    rows = []
    for t, sub in df.groupby("topic"):
        vc = sub[cc.COL_ITEM].value_counts()
        p10, p50, p90 = (sub[cc.COL_UNIT].quantile(q) for q in (0.10, 0.50, 0.90))
        ratio = (p90 / p10) if p10 > 0 else float("inf")
        rows.append({
            "topic": t,
            "type_label": (f"{vc.index[0]} 等{sub[cc.COL_ITEM].nunique()}款"
                           if sub[cc.COL_ITEM].nunique() > 1 else f"{vc.index[0]}（單品）"),
            "n_distinct_items": sub[cc.COL_ITEM].nunique(),
            "n_rows": len(sub),
            "users": ",".join(map(str, sorted(sub[cc.COL_USER].unique()))),
            "unit_price_p10": round(p10, 1),
            "unit_price_p50": round(p50, 1),
            "unit_price_p90": round(p90, 1),
            "price_ratio_p90_p10": round(ratio, 1),
            "generic_share": round(sub["is_generic"].mean(), 2),
            "coherent_ok": ratio <= MAX_PRICE_RATIO,
            "members": " | ".join(f"{i}:{c}" for i, c in vc.items()),
        })
    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    path = cc.OUT_DIR / "step3c_cluster_members.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"共 {len(out)} 個類型群 → {path}")
    print(f"通過一致性的群: {out['coherent_ok'].sum()}；單品群: {(out.n_distinct_items==1).sum()}")
    print("\n=== 成員最多的 8 群 ===")
    print(out.head(8)[["topic", "type_label", "n_distinct_items", "n_rows",
                       "unit_price_p50", "coherent_ok"]].to_string(index=False))


if __name__ == "__main__":
    main()
