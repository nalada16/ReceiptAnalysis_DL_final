"""輸出增量分析的詳細數據（超類型 K=25），方便查「某 user 某類型有哪些品項、價格」。

輸出兩個檔（outputs/）：
  step3d_increment_breakdown.csv  ── K=25 圖背後的數字：每 user×超類型的 P0/P1/Q0/Q1/變貴/買更多
  increment_detail_K25.csv        ── 逐品項明細：每 user×超類型×品項 的 平常 vs 本月 數量/金額/單價

查法（Excel 篩選 user_id=0）：
  - 想看 user0 每個超類型的總增量 → 看 step3d_increment_breakdown.csv
  - 想看 user0 某超類型裡買了哪些品項、各多少錢 → 看 increment_detail_K25.csv
"""
from __future__ import annotations
import pandas as pd

import c_common as cc
import step3d_supertypes as s3d
import step3c_personal_inflation as s3c
import naming


def main():
    df, emb = s3d.load_clean_with_emb()
    s = s3d.assign_super(df, emb, 25)
    names = naming.load_names(naming.super_names_path(25), "super")

    # ── 1) 超類型層 per-user 數字（圖背後的數據）──
    brk_rows = []
    for u in sorted(s[cc.COL_USER].unique()):
        sub = s[s[cc.COL_USER] == u]
        dec, cur, n_base = s3c.decompose_user(sub)
        if dec.empty:
            continue
        dec = dec.copy()
        dec.insert(0, "user_id", u)
        dec["supertype"] = dec["topic"].map(lambda t: names.get(t, str(t)))
        brk_rows.append(dec)
    brk = pd.concat(brk_rows, ignore_index=True)
    brk = brk.rename(columns={"topic": "super"})[
        ["user_id", "super", "supertype", "P0", "P1", "Q0_per_month", "Q1_this_month",
         "baseline_spend", "current_spend", "delta_spend", "變貴_price", "買更多_qty"]]
    brk.to_csv(cc.OUT_DIR / "step3d_increment_breakdown.csv", index=False, encoding="utf-8-sig")

    # 通過門檻的 (user, super) 集合，明細只列這些（與圖一致）
    passed = set(zip(brk["user_id"], brk["super"]))

    # ── 2) 逐品項明細 ──
    rows = []
    for u in sorted(s[cc.COL_USER].unique()):
        sub = s[s[cc.COL_USER] == u]
        months = sorted(sub["month"].unique())
        cur, base_months = months[-1], months[:-1]
        base = sub[sub["month"].isin(base_months)]
        curm = sub[sub["month"] == cur]
        for sup in sorted(sub["super"].unique()):
            if (u, sup) not in passed:
                continue
            g = sub[sub["super"] == sup]
            for item in g[cc.COL_ITEM].unique():
                b = base[(base["super"] == sup) & (base[cc.COL_ITEM] == item)]
                c = curm[(curm["super"] == sup) & (curm[cc.COL_ITEM] == item)]
                bq, ba = b[cc.COL_QTY].sum(), b[cc.COL_AMT].sum()
                cq, ca = c[cc.COL_QTY].sum(), c[cc.COL_AMT].sum()
                rows.append({
                    "user_id": u, "super": sup,
                    "supertype": names.get(sup, str(sup)),
                    "item": item,
                    "total_qty": (g[cc.COL_ITEM] == item).sum(),
                    "base_qty": bq, "base_spend": round(ba, 1),
                    "base_unit_price": round(ba / bq, 1) if bq else None,
                    "cur_qty": cq, "cur_spend": round(ca, 1),
                    "cur_unit_price": round(ca / cq, 1) if cq else None,
                })
    detail = pd.DataFrame(rows).sort_values(["user_id", "super", "base_spend"],
                                            ascending=[True, True, False])
    detail.to_csv(cc.OUT_DIR / "increment_detail_K25.csv", index=False, encoding="utf-8-sig")

    print(f"已輸出：")
    print(f"  step3d_increment_breakdown.csv  ({len(brk)} 列：user×超類型 數字)")
    print(f"  increment_detail_K25.csv        ({len(detail)} 列：user×超類型×品項 明細)")
    print(f"\n範例：user 0 的超類型增量")
    print(brk[brk.user_id == 0][["supertype", "P0", "P1", "delta_spend"]].to_string(index=False))


if __name__ == "__main__":
    main()
