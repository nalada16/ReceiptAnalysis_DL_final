"""為每位 user 畫「所有通過門檻的超類型」的單價成長歷史折線圖。

超類型數 > MAX_PER_CHART 時自動分成兩張子圖（依本月花費排序，前半/後半）。
輸出：outputs/history_user{u}.png
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

import c_common as cc
import step3d_supertypes as s3d
import step3c_personal_inflation as s3c
import naming

MAX_PER_CHART = 7   # 一張圖最多幾條線，超過就分兩張子圖

for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        plt.rcParams["axes.unicode_minus"] = False
        break


def monthly_unit_price(sub, supers):
    s = sub[sub["super"].isin(supers)]
    g = s.groupby(["super", "month"]).agg(spend=(cc.COL_AMT, "sum"), qty=(cc.COL_QTY, "sum")).reset_index()
    g["up"] = g["spend"] / g["qty"]
    return g


def plot_group(ax, g, supers, name_map, all_months, xpos):
    for sp in supers:
        d = g[g["super"] == sp].sort_values("month")
        if len(d) < 2:
            continue
        first, last = d["up"].iloc[0], d["up"].iloc[-1]
        chg = (last / first - 1) * 100 if first else 0
        ax.plot([xpos[m] for m in d["month"]], d["up"].values, marker="o",
                label=f"{name_map[sp]} ({chg:+.0f}%)")
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("加權平均單價 (NT$)")
    ax.legend(fontsize=8, loc="best", ncol=2)


def main():
    df, emb = s3d.load_clean_with_emb()
    s = s3d.assign_super(df, emb, 25)
    names = naming.load_names(naming.super_names_path(25), "super")
    all_months = sorted(s["month"].unique())
    xpos = {m: i for i, m in enumerate(all_months)}

    for u in sorted(s[cc.COL_USER].unique()):
        sub = s[s[cc.COL_USER] == u]
        dec, cur, n_base = s3c.decompose_user(sub)   # 通過門檻的超類型
        if dec.empty:
            continue
        # 依本月花費排序
        dec = dec.reindex(dec["current_spend"].sort_values(ascending=False).index)
        supers = dec["topic"].tolist()   # step3d 把 topic 設為 super id
        name_map = {sp: names.get(sp, str(sp)) for sp in supers}
        g = monthly_unit_price(sub, supers)

        n = len(supers)
        if n <= MAX_PER_CHART:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            plot_group(ax, g, supers, name_map, all_months, xpos)
            ax.set_title(f"user {u}｜全部 {n} 個通過門檻超類型的單價成長歷史")
        else:
            half = (n + 1) // 2
            fig, axes = plt.subplots(2, 1, figsize=(11, 10))
            plot_group(axes[0], g, supers[:half], name_map, all_months, xpos)
            plot_group(axes[1], g, supers[half:], name_map, all_months, xpos)
            axes[0].set_title(f"user {u}｜通過門檻超類型單價歷史（共 {n} 個，上：花費較高 {half} 個）")
            axes[1].set_title(f"下：其餘 {n - half} 個")
        fig.tight_layout()
        fig.savefig(cc.OUT_DIR / f"history_user{u}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"user {u}: {n} 個超類型 → history_user{u}.png" + ("（分兩張子圖）" if n > MAX_PER_CHART else ""))


if __name__ == "__main__":
    main()
