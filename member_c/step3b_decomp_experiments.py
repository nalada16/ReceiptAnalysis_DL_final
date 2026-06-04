"""步驟 3b：通膨拆解的替代方案對照（給報告的「嘗試紀錄」）。

通膨拆解本身沒有「準確率」（兩項拆解 residual 恆為 0），但不同的「設計選擇」
會得到不同結論。這裡比較兩個替代分支對結果的影響：

  A. 分群鍵：exact（同品名店名） vs cluster（step1 語意群）
     語意群會把拼寫變體合併、覆蓋率更高，但可能混入不同單價商品 → 影響價/量效應估計。

  B. 拆解公式：兩項精確 vs 三項含交互
     三項：Delta = Q0·ΔP(量) + P0... 實際上 = P0·ΔQ + Q0·ΔP + ΔP·ΔQ(交互)
     兩項把交互併入價效應；三項把交互獨立出來。

輸出：outputs/exp_decomposition.csv（彙總對照）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import c_common as cc
from step2_unit_price import attach_groups, filter_repeat_groups
from step3_inflation import pick_periods, pq


def decompose_both(df, group_col, repeat, by_user):
    """同時算兩項與三項拆解，回傳每群明細。"""
    rows = []
    keys = [cc.COL_USER, group_col] if by_user else [group_col]
    for _, r in repeat.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for k in keys:
            mask &= (df[k] == r[k]).values
        sub = df[mask]
        b, c = pick_periods(sub)
        p0, p1 = pq(sub, b), pq(sub, c)
        if p0 is None or p1 is None or b == c:
            continue
        P0, Q0, _ = p0
        P1, Q1, _ = p1
        delta = P1 * Q1 - P0 * Q0
        rows.append({
            "delta": delta,
            # 兩項精確
            "price2": (P1 - P0) * Q1,
            "qty2": (Q1 - Q0) * P0,
            # 三項含交互
            "price3": Q0 * (P1 - P0),
            "qty3": P0 * (Q1 - Q0),
            "interaction3": (P1 - P0) * (Q1 - Q0),
        })
    return pd.DataFrame(rows)


def summarize(dec, label):
    tot = dec["delta"].sum()
    return {
        "config": label,
        "n_groups": len(dec),
        "total_delta": round(tot, 1),
        # 兩項
        "price_2term": round(dec["price2"].sum(), 1),
        "qty_2term": round(dec["qty2"].sum(), 1),
        # 三項
        "price_3term": round(dec["price3"].sum(), 1),
        "qty_3term": round(dec["qty3"].sum(), 1),
        "interaction": round(dec["interaction3"].sum(), 1),
        # 拆解殘差（兩項應為 0；三項應為 0）
        "resid_2term": round(tot - dec["price2"].sum() - dec["qty2"].sum(), 2),
        "resid_3term": round(
            tot - dec["price3"].sum() - dec["qty3"].sum() - dec["interaction3"].sum(), 2),
    }


def main():
    df0, _ = cc.load_data(require_embeddings=False)
    results = []
    for mode in ["exact", "cluster"]:
        try:
            df, gcol = attach_groups(df0.copy(), mode)
        except SystemExit as e:
            print(f"[跳過 mode={mode}] {e}")
            continue
        repeat = filter_repeat_groups(df, gcol, by_user=True)
        dec = decompose_both(df, gcol, repeat, by_user=True)
        results.append(summarize(dec, f"group={mode}"))
        print(f"  mode={mode}: {len(dec)} 群可拆解")

    out = pd.DataFrame(results)
    out.to_csv(cc.OUT_DIR / "exp_decomposition.csv", index=False, encoding="utf-8-sig")
    print("\n=== 通膨拆解替代方案對照（全使用者合計）===")
    print(out.to_string(index=False))
    print("\n已輸出 exp_decomposition.csv")


if __name__ == "__main__":
    main()
