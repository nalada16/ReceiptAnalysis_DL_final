"""per-user 步驟3c：用「各自分群」的類型，分開跑增量分析。

與主流程 step3c 完全相同的清理與拆解邏輯（直接重用 step3c 的函式），
差別只在「類型」改用 per-user 分群結果（pu_clusters.csv 的 pu_topic）。

目的：檢驗「三人分開分群再分開增量分析」會不會出問題
（預期：小資料量 user 因群粗、可分析類型過少）。

輸出（per_user_pipeline/outputs/）：
  pu_step3c_increment_breakdown.csv
  pu_step3c_user_summary.csv
  pu_step3c_user{u}.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pu_common as pu
import c_common as cc
import step3c_personal_inflation as s3   # 重用主流程的清理/拆解/繪圖邏輯


def build_typed_peruser():
    """載入明細，類型 = per-user 分群；套用與 step3c 相同的清理。"""
    df, _ = cc.load_data(require_embeddings=False)
    clus = pd.read_csv(pu.OUT / "pu_clusters.csv")
    df["topic"] = clus["pu_topic"].values
    df = df[df["topic"] != "noise"].copy()

    # 1) 剔除通用佔位品名（與 step3c 同一份清單）
    item_low = df[cc.COL_ITEM].astype(str).str.lower()
    mask = item_low.apply(lambda s: any(k in s for k in s3.GENERIC_KEYWORDS))
    n0 = len(df); df = df[~mask].copy()
    print(f"剔除通用佔位品名 {n0 - len(df)} 筆")

    # 2) 類型一致性過濾（群內單價 p90/p10）
    def coherent(s):
        p10, p90 = s.quantile(0.10), s.quantile(0.90)
        return (p90 / p10) <= s3.MAX_PRICE_RATIO if p10 > 0 else False
    keep = df.groupby("topic")[cc.COL_UNIT].apply(coherent)
    df = df[df["topic"].isin(keep[keep].index)].copy()
    print(f"保留一致性 OK 的 per-user 類型 {df['topic'].nunique()} 個")

    # 3) 類型標籤 = 群內最高頻品名 +「等N款」，凸顯這是 cluster(類型)而非單品
    top_item = df.groupby("topic")[cc.COL_ITEM].agg(lambda s: s.value_counts().index[0])
    n_items = df.groupby("topic")[cc.COL_ITEM].nunique()
    df["type_label"] = df["topic"].map(
        lambda t: f"{top_item[t]} 等{n_items[t]}款" if n_items[t] > 1 else f"{top_item[t]}（單品）")
    return df


def main():
    df = build_typed_peruser()
    all_break, summary = [], []
    for u in sorted(df[cc.COL_USER].unique()):
        sub = df[df[cc.COL_USER] == u]
        dec, cur, n_base = s3.decompose_user(sub)   # 重用主流程拆解（含 MIN_TYPE_ROWS=10）
        if dec.empty:
            print(f"user {u}: 清理+門檻後『無可分析類型』")
            summary.append({"user_id": u, "current_month": cur,
                            "baseline_months": n_base, "可分析類型數": 0,
                            "本月多花的錢": np.nan, "變貴貢獻": np.nan, "買更多貢獻": np.nan})
            continue
        dec.insert(0, "user_id", u)
        total = dec["delta_spend"].sum()
        summary.append({
            "user_id": u, "current_month": cur, "baseline_months": n_base,
            "可分析類型數": len(dec),
            "本月多花的錢": round(total, 1),
            "變貴貢獻": round(dec["變貴_price"].sum(), 1),
            "買更多貢獻": round(dec["買更多_qty"].sum(), 1),
        })
        all_break.append(dec)
        s3.plot_user(u, dec, sub, cur, n_base, total, pu.OUT / f"pu_step3c_user{u}.png")

    if all_break:
        pd.concat(all_break, ignore_index=True).to_csv(
            pu.OUT / "pu_step3c_increment_breakdown.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(summary)
    summ.to_csv(pu.OUT / "pu_step3c_user_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== per-user 增量分析摘要（vs 主流程 pooled 對照）===")
    print(summ.to_string(index=False))
    print(f"\n已輸出至 {pu.OUT}")


if __name__ == "__main__":
    main()
