"""步驟 3：個人通膨拆解（量效應 vs 價效應）。

對每個重複購買商品群，比較「基期」與「當期」：
    P0,Q0 = 基期平均單價、總數量      P1,Q1 = 當期平均單價、總數量
    Delta        = P1*Q1 - P0*Q0      （= 當期金額 - 基期金額）
    price_effect = (P1 - P0) * Q1      （價的效應：相似商品變貴/變便宜）
    qty_effect   = (Q1 - Q0) * P0      （量的效應：買更多/更少）
這組拆解在數學上 price_effect + qty_effect == Delta（無交互殘差），
residual 應接近 0；若不為 0 多半是浮點誤差或群內品項不一致。

輸入：all_user_6_label.csv（分群方式同步驟 2）
輸出：
  outputs/step3_decomposition.csv   每群的量/價效應明細
  outputs/step3_user_summary.csv    每位使用者的總量效應/總價效應/通膨佔比
  outputs/step3_waterfall.png       價效應 vs 量效應長條圖
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import c_common as cc
from step2_unit_price import attach_groups, filter_repeat_groups


def pick_periods(sub: pd.DataFrame, base=None, cur=None):
    """決定基期與當期月份。預設取該群最早/最晚有資料的月份。"""
    months = sorted(sub["month"].unique())
    b = pd.Period(base, "M") if base else months[0]
    c = pd.Period(cur, "M") if cur else months[-1]
    return b, c


def pq(sub: pd.DataFrame, month):
    """回傳該群在某月的 (P=加權均價, Q=總數量, amount=總金額)。沒資料回 None。"""
    m = sub[sub["month"] == month]
    if m.empty:
        return None
    qty = m[cc.COL_QTY].sum()
    amt = m[cc.COL_AMT].sum()
    return amt / qty, qty, amt


def decompose(df: pd.DataFrame, group_col: str, repeat: pd.DataFrame, by_user: bool,
              base=None, cur=None) -> pd.DataFrame:
    rows = []
    keys = [cc.COL_USER, group_col] if by_user else [group_col]
    for _, r in repeat.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for k in keys:
            mask &= (df[k] == r[k]).values
        sub = df[mask]
        b, c = pick_periods(sub, base, cur)
        p0 = pq(sub, b)
        p1 = pq(sub, c)
        if p0 is None or p1 is None or b == c:
            continue  # 基期或當期缺資料、或只有一個月，跳過
        P0, Q0, A0 = p0
        P1, Q1, A1 = p1
        delta = P1 * Q1 - P0 * Q0
        price_effect = (P1 - P0) * Q1
        qty_effect = (Q1 - Q0) * P0
        rows.append({
            **{k: r[k] for k in keys},
            "base_month": str(b), "cur_month": str(c),
            "P0": round(P0, 2), "Q0": Q0, "P1": round(P1, 2), "Q1": Q1,
            "base_spend": round(A0, 2), "cur_spend": round(A1, 2),
            "delta": round(delta, 2),
            "price_effect": round(price_effect, 2),
            "qty_effect": round(qty_effect, 2),
            "residual": round(delta - price_effect - qty_effect, 4),
        })
    return pd.DataFrame(rows)


def user_summary(dec: pd.DataFrame, by_user: bool) -> pd.DataFrame:
    if by_user:
        groups = [(u, sub) for u, sub in dec.groupby(cc.COL_USER)]
    else:
        groups = [("all", dec)]
    out = []
    for key, d in groups:
        total_delta = d["delta"].sum()
        price = d["price_effect"].sum()
        qty = d["qty_effect"].sum()
        row = {
            "total_delta": round(total_delta, 2),
            "price_effect": round(price, 2),
            "qty_effect": round(qty, 2),
            "price_share": round(price / total_delta, 3) if total_delta else np.nan,
            "qty_share": round(qty / total_delta, 3) if total_delta else np.nan,
            "n_groups": len(d),
        }
        if by_user:
            row[cc.COL_USER] = key
        out.append(row)
    return pd.DataFrame(out)


def plot_waterfall(summary: pd.DataFrame, by_user: bool, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    if by_user and cc.COL_USER in summary:
        labels = [f"user {u}" for u in summary[cc.COL_USER]]
        x = np.arange(len(summary))
        ax.bar(x - 0.2, summary["price_effect"], width=0.4, label="price effect")
        ax.bar(x + 0.2, summary["qty_effect"], width=0.4, label="quantity effect")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    else:
        ax.bar(["price effect", "quantity effect"],
               [summary["price_effect"].iloc[0], summary["qty_effect"].iloc[0]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("NT$ contribution to spend change")
    ax.set_title("Spend change decomposition: price vs quantity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact", "cluster"], default="exact")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--base", default=None, help="基期月份 YYYY-MM；預設各群最早月")
    ap.add_argument("--cur", default=None, help="當期月份 YYYY-MM；預設各群最晚月")
    args = ap.parse_args()
    by_user = not args.pooled

    df, _ = cc.load_data(require_embeddings=False)
    df, group_col = attach_groups(df, args.mode)
    repeat = filter_repeat_groups(df, group_col, by_user)

    dec = decompose(df, group_col, repeat, by_user, args.base, args.cur)
    if dec.empty:
        raise SystemExit("沒有可拆解的商品群（基期/當期資料不足）。")
    summary = user_summary(dec, by_user)

    dec.to_csv(cc.OUT_DIR / "step3_decomposition.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(cc.OUT_DIR / "step3_user_summary.csv", index=False, encoding="utf-8-sig")
    plot_waterfall(summary, by_user, cc.OUT_DIR / "step3_waterfall.png")

    print("=== 使用者通膨拆解摘要 ===")
    print(summary.to_string(index=False))
    print("\n已輸出 step3_decomposition.csv / step3_user_summary.csv / step3_waterfall.png")


if __name__ == "__main__":
    main()
