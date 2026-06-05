"""產生超類型(K=25)的完整品項對照表。

輸出：outputs/supertype_K25_members.csv
欄位：super / custom_name / n_distinct_items / n_rows / unit_price_p50
      / top_items（前20品項:次數）/ all_items（全部）
"""
from __future__ import annotations
import pandas as pd
import c_common as cc
import step3d_supertypes as s3d
import naming

def main():
    df, emb = s3d.load_clean_with_emb()
    s = s3d.assign_super(df, emb, 25)

    # 套用命名
    names = naming.load_names(naming.super_names_path(25), "super")

    rows = []
    for super_id in sorted(s["super"].unique()):
        sub = s[s["super"] == super_id]
        vc = sub[cc.COL_ITEM].value_counts()
        cname = names.get(super_id, sub["type_label"].iloc[0])
        rows.append({
            "super": super_id,
            "custom_name": cname,
            "n_distinct_items": sub[cc.COL_ITEM].nunique(),
            "n_rows": len(sub),
            "n_users": sub[cc.COL_USER].nunique(),
            "unit_price_p10": round(sub[cc.COL_UNIT].quantile(0.1), 1),
            "unit_price_p50": round(sub[cc.COL_UNIT].median(), 1),
            "unit_price_p90": round(sub[cc.COL_UNIT].quantile(0.9), 1),
            "top_items": " | ".join(f"{i}({c})" for i, c in vc.head(20).items()),
            "all_items": " | ".join(f"{i}({c})" for i, c in vc.items()),
        })

    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    path = cc.OUT_DIR / "supertype_K25_members.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"已輸出 {path}（{len(out)} 個超類型）")
    print(out[["super", "custom_name", "n_rows", "n_distinct_items", "unit_price_p50"]].to_string(index=False))

if __name__ == "__main__":
    main()
