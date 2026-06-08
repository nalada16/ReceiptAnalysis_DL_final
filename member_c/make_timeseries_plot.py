# -*- coding: utf-8 -*-
"""新圖：三位 user × 三個指標（總花費 / 總數量 / 平均單價）的折線圖。

輸出：outputs/timeseries_all_users.png
版面：3 列（user 0/1/2）× 3 欄（總花費、總數量、平均單價）
每格顯示該 user 花費最高的前 TOP_N 個超類型的月時序。
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"D:\ReceiptAnalysis_DL_final\member_c")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

import c_common as cc
import step3d_supertypes as s3d

TOP_N = 5   # 每位 user 顯示前幾個超類型

# ---- 字體 ----
for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        plt.rcParams["axes.unicode_minus"] = False
        break

# ---- 載入資料（K=25 超類型）----
df, emb = s3d.load_clean_with_emb()
s = s3d.assign_super(df, emb, K=25)

users = sorted(s[cc.COL_USER].unique())
all_months = sorted(s["month"].unique())
mpos = {m: i for i, m in enumerate(all_months)}
xticks = list(range(len(all_months)))
xlabels = [str(m)[2:] for m in all_months]   # 短月份標籤 e.g. 25-09

METRICS = [
    ("amt",        "總花費 (NT$)",   "月總花費"),
    ("qty",        "購買數量",        "月購買數量"),
    ("unit_price", "平均單價 (NT$)", "月平均單價"),
]

fig, axes = plt.subplots(len(users), 3, figsize=(18, 4.5 * len(users)), squeeze=False)
fig.suptitle("各 User 主要超類型時序（總花費 / 購買數量 / 平均單價）",
             fontsize=14, fontweight="bold", y=1.01)

for row, u in enumerate(users):
    sub = s[s[cc.COL_USER] == u]

    # 選 TOP_N：以全期總花費排序
    top_types = (sub.groupby("super")[cc.COL_AMT].sum()
                 .sort_values(ascending=False)
                 .head(TOP_N).index.tolist())
    label_map = (sub.drop_duplicates("super")
                 .set_index("super")["type_label"].to_dict())

    # 月聚合
    g = (sub[sub["super"].isin(top_types)]
         .groupby(["super", "month"])
         .agg(amt=(cc.COL_AMT, "sum"), qty=(cc.COL_QTY, "sum"))
         .reset_index())
    g["unit_price"] = g["amt"] / g["qty"]

    for col_idx, (col, ylabel, title_suffix) in enumerate(METRICS):
        ax = axes[row][col_idx]
        for t in top_types:
            d = g[g["super"] == t].sort_values("month")
            if len(d) < 2:
                continue
            lab = str(label_map.get(t, t))[:12]
            ax.plot([mpos[m] for m in d["month"]], d[col].values,
                    marker="o", markersize=4, label=lab)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"user {u}｜{title_suffix}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
out = cc.OUT_DIR / "timeseries_all_users.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"已輸出 {out}")
