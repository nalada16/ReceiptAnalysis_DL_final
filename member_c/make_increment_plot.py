# -*- coding: utf-8 -*-
"""新增量圖：紅(變貴)、藍(買更多) 各自從 0 出發，不堆疊，避免跨 0 混淆。
另以黑色菱形標出總增量 ΔS = 變貴 + 買更多。

版面：1 列 × 3 欄（user 0 / 1 / 2），K=25 超類型。
輸出：outputs/increment_all_users.png（不覆蓋舊的 step3d_K25_user*.png）
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"D:\ReceiptAnalysis_DL_final\member_c")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

import c_common as cc
import step3c_personal_inflation as s3
import step3d_supertypes as s3d

TOP_N = 8   # 每位 user 顯示增量絕對值前 N 大的超類型

for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        plt.rcParams["axes.unicode_minus"] = False
        break

# ---- 載入 K=25 超類型 ----
df, emb = s3d.load_clean_with_emb()
s = s3d.assign_super(df, emb, K=25)
users = sorted(s[cc.COL_USER].unique())

fig, axes = plt.subplots(1, len(users), figsize=(7.5 * len(users), 8), squeeze=False)
axes = axes[0]

for ax, u in zip(axes, users):
    sub = s[s[cc.COL_USER] == u]
    dec, cur, n_base = s3.decompose_user(sub)
    if dec.empty:
        ax.set_title(f"user {u}｜無可分析類型")
        continue

    top = dec.reindex(dec["delta_spend"].abs().sort_values(ascending=False).index).head(TOP_N)
    top = top.iloc[::-1]   # 由小到大，畫在下→上
    y = np.arange(len(top))
    h = 0.38

    price = top["變貴_price"].to_numpy()
    qty = top["買更多_qty"].to_numpy()
    delta = top["delta_spend"].to_numpy()

    # 紅、藍各自從 0 出發（並列，不堆疊）
    ax.barh(y + h / 2, price, height=h, color="#d9534f", label="變貴 (價)")
    ax.barh(y - h / 2, qty, height=h, color="#4a90d9", label="買更多 (量)")
    # 總增量：黑色菱形 + 數字
    ax.scatter(delta, y, color="black", marker="D", s=28, zorder=5, label="總增量 ΔS")
    for yi, d in zip(y, delta):
        ax.text(d + (12 if d >= 0 else -12), yi, f"{d:+.0f}",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, fontweight="bold")

    labels = [str(l)[:12] for l in top["type_label"]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", lw=0.8)
    total = dec["delta_spend"].sum()
    sign = "多花" if total >= 0 else "少花"
    ax.set_title(f"user {u}｜本月({cur})比平常{sign} {abs(total):.0f} 元", fontsize=11)
    ax.set_xlabel("與平常月均的差 (NT$)", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

# 共用圖例
handles = [
    plt.Rectangle((0, 0), 1, 1, color="#d9534f"),
    plt.Rectangle((0, 0), 1, 1, color="#4a90d9"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=8),
]
fig.legend(handles, ["變貴 (價效應)", "買更多 (量效應)", "總增量 ΔS"],
           loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("個人化消費增量：價效應 vs 量效應（各自從 0 出發）",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out = cc.OUT_DIR / "increment_all_users.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"已輸出 {out}")
