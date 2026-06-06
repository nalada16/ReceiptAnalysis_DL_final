"""產生延伸洞察圖：① 每人每月總消費(季節性/寒假) ② user0 消費升級(類別單價分歧)。

輸出：outputs/insight_monthly_spend.png、outputs/insight_user0_upgrade.png
"""
from __future__ import annotations
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

import c_common as cc
import step3d_supertypes as s3d
import naming

for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        plt.rcParams["axes.unicode_minus"] = False
        break


def main():
    df, emb = s3d.load_clean_with_emb()
    s = s3d.assign_super(df, emb, 25)
    names = naming.load_names(naming.super_names_path(25), "super")
    s["name"] = s["super"].map(lambda t: names.get(t, str(t)))
    all_months = sorted(s["month"].unique())
    xpos = {m: i for i, m in enumerate(all_months)}

    # ── 圖1：每人每月總消費 ──
    fig, ax = plt.subplots(figsize=(11, 5.5))
    tot = s.groupby([cc.COL_USER, "month"])[cc.COL_AMT].sum().reset_index()
    for u in sorted(s[cc.COL_USER].unique()):
        d = tot[tot[cc.COL_USER] == u].sort_values("month")
        ax.plot([xpos[m] for m in d["month"]], d[cc.COL_AMT], marker="o", label=f"user {u}")
    jan = xpos.get(pd.Period("2026-01", "M"))
    if jan is not None:
        ax.axvline(jan, color="grey", ls="--", lw=1)
        ax.text(jan, ax.get_ylim()[1]*0.95, "2026-01 寒假", fontsize=9, color="grey", ha="center")
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha="right")
    ax.set_ylabel("每月總消費 (NT$)")
    ax.set_title("每人每月總消費趨勢（user 0 / 2 寒假暴跌，user 1 一月反而最高）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cc.OUT_DIR / "insight_monthly_spend.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── 圖2：user0 消費升級（正餐主食 vs 便宜類別 單價趨勢）──
    u0 = s[s[cc.COL_USER] == 0]
    g = u0.groupby(["name", "month"]).agg(spend=(cc.COL_AMT, "sum"), qty=(cc.COL_QTY, "sum")).reset_index()
    g["up"] = g["spend"] / g["qty"]
    focus = ["正餐主食", "超商零食點心", "優酪乳乳製品"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for nm in focus:
        d = g[g["name"] == nm].sort_values("month")
        if len(d) < 3:
            continue
        chg = (d["up"].iloc[-1] / d["up"].iloc[0] - 1) * 100
        ax.plot([xpos[m] for m in d["month"]], d["up"], marker="o", label=f"{nm} ({chg:+.0f}%)")
    ax.set_xticks(range(len(all_months)))
    ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha="right")
    ax.set_ylabel("加權平均單價 (NT$)")
    ax.set_title("user 0：正餐主食單價走高（5 月尤甚），零食/優酪乳長期走低")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cc.OUT_DIR / "insight_user0_upgrade.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("已輸出 insight_monthly_spend.png / insight_user0_upgrade.png")


if __name__ == "__main__":
    main()
