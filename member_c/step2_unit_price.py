"""步驟 2：重複購買商品群篩選 + 單價時序追蹤。

輸入：all_user_6_label.csv（可選 step1_clusters.csv 的語意群）
輸出：
  outputs/step2_repeat_groups.csv      通過門檻的商品群清單
  outputs/step2_unit_price_ts.csv      每群每月加權平均單價
  outputs/step2_unit_price_top.png     幾個高頻群的單價時序折線

分群方式有兩種，預設用 exact（最可靠）：
  exact   ── 同一 (品名, 店名) 視為同商品。通膨拆解的主用途。
  cluster ── step1 的語意群。把同商品的拼寫變體聚在一起，當對照/探索。
"""
from __future__ import annotations

import argparse
import pandas as pd

import c_common as cc


def attach_groups(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    """依 mode 在 df 上建立 'group' 欄，回傳 (df, group_col)。"""
    if mode == "exact":
        df = df.copy()
        df["group"] = cc.group_key_exact(df)
        return df, "group"
    if mode == "cluster":
        path = cc.OUT_DIR / "step1_clusters.csv"
        if not path.exists():
            raise SystemExit("找不到 step1_clusters.csv，請先跑 step1_clustering.py")
        clus = pd.read_csv(path)
        df = df.copy()
        df["group"] = clus["topic"].values
        df = df[df["group"] != -1]  # 丟掉 HDBSCAN noise
        df["group"] = "topic_" + df["group"].astype(str)
        return df, "group"
    raise ValueError(f"unknown mode: {mode}")


def filter_repeat_groups(df: pd.DataFrame, group_col: str, by_user: bool) -> pd.DataFrame:
    """挑出「出現 >= MIN_COUNT 次且跨 >= MIN_MONTHS 個月」的商品群。"""
    keys = [cc.COL_USER, group_col] if by_user else [group_col]
    stat = df.groupby(keys, observed=True).agg(
        n=(cc.COL_AMT, "size"),
        n_month=("month", "nunique"),
        first_month=("month", "min"),
        last_month=("month", "max"),
    ).reset_index()
    keep = stat[(stat["n"] >= cc.MIN_COUNT) & (stat["n_month"] >= cc.MIN_MONTHS)]
    return keep.sort_values("n", ascending=False)


def plot_top(ts: pd.DataFrame, repeat: pd.DataFrame, group_col: str, by_user: bool, path, top_k=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 嘗試套用中文字型，避免方框；找不到就維持預設（圖仍可用）
    for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break

    top = repeat.head(top_k)
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, r in top.iterrows():
        mask = ts[group_col] == r[group_col]
        if by_user:
            mask &= ts[cc.COL_USER] == r[cc.COL_USER]
        sub = ts[mask].sort_values("month")
        x = sub["month"].astype(str)
        label = str(r[group_col])[:18] + (f" (u{r[cc.COL_USER]})" if by_user else "")
        ax.plot(x, sub["unit_price"], marker="o", label=label)
    ax.set_xlabel("month")
    ax.set_ylabel("weighted avg unit price")
    ax.set_title(f"Top-{top_k} repeat-group unit price over time")
    ax.legend(fontsize=8, loc="best")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exact", "cluster"], default="exact")
    ap.add_argument("--pooled", action="store_true", help="不分使用者，全體合併")
    args = ap.parse_args()
    by_user = not args.pooled

    df, _ = cc.load_data(require_embeddings=False)
    df, group_col = attach_groups(df, args.mode)

    repeat = filter_repeat_groups(df, group_col, by_user)
    print(f"[mode={args.mode}, by_user={by_user}] 通過門檻的商品群：{len(repeat)} 個")

    # 只對通過門檻的群算單價時序
    df_keep = df.merge(repeat[[c for c in ([cc.COL_USER, group_col] if by_user else [group_col])]],
                       on=[cc.COL_USER, group_col] if by_user else [group_col])
    ts = cc.monthly_unit_price(df_keep, group_col, by_user=by_user)

    repeat.to_csv(cc.OUT_DIR / "step2_repeat_groups.csv", index=False, encoding="utf-8-sig")
    ts.to_csv(cc.OUT_DIR / "step2_unit_price_ts.csv", index=False, encoding="utf-8-sig")
    plot_top(ts, repeat, group_col, by_user, cc.OUT_DIR / "step2_unit_price_top.png")
    print("已輸出 step2_repeat_groups.csv / step2_unit_price_ts.csv / step2_unit_price_top.png")


if __name__ == "__main__":
    main()
