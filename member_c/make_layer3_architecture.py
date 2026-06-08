# -*- coding: utf-8 -*-
"""生成 Layer 3（成員 C）專屬架構圖，風格：圓角色塊 + Step 分段 + 標題/副標。

輸出：outputs/layer3_architecture.png
內容對齊 report_outline 三階段：語意分群 → 合併超類型 → 個人化增量分析（兩因子）。
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import c_common as cc

for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        plt.rcParams["axes.unicode_minus"] = False
        break

SALMON = dict(fc="#EBA9A3", ec="#D98A82", tc="#6E2B25", sc="#8C463F")
PURPLE = dict(fc="#C9CBEC", ec="#A9ACDD", tc="#34346A", sc="#56568C")
TEAL = dict(fc="#8FBEB3", ec="#6FA99B", tc="#1E4A40", sc="#356B5E")
GREY = dict(fc="#F3F3F3", ec="#C2C2C2", tc="#555555", sc="#888888")

fig, ax = plt.subplots(figsize=(8.6, 13.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 17.5)
ax.axis("off")


def box(cx, cy, w, h, title, subtitle, style, tsize=12, ssize=8.5, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        fc=style["fc"], ec=style["ec"], lw=1.6,
        linestyle="--" if dashed else "-", zorder=2))
    if subtitle:
        ax.text(cx, cy + h * 0.17, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=style["tc"], zorder=3)
        ax.text(cx, cy - h * 0.25, subtitle, ha="center", va="center",
                fontsize=ssize, color=style["sc"], zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=tsize, fontweight="bold", color=style["tc"], zorder=3)
    return {"top": (cx, cy + h / 2), "bot": (cx, cy - h / 2)}


def arrow(p, q, color="#7A7A7A"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18,
                 lw=1.8, color=color, zorder=1))


def step_label(y, text):
    ax.text(0.3, y, text, ha="left", va="center", fontsize=12.5,
            fontweight="bold", color="#444444")


def dsep(y):
    ax.plot([0.3, 9.7], [y, y], ls=(0, (4, 4)), color="#CFCFCF", lw=1.2, zorder=0)


# ===== Step 1 =====
step_label(16.9, "Step 1　商品語意分群（精準層）")
b1 = box(5, 16.0, 5.4, 0.95, "BERT Embedding（768 維）",
         "每筆明細的語意向量 ← Task 1", SALMON)
b2 = box(5, 14.5, 6.6, 0.95, "UMAP 降維 → HDBSCAN 密度分群",
         "不需預設群數、label = −1 自動過濾離群品項", PURPLE)
arrow(b1["bot"], b2["top"])
arrow(b2["bot"], (5, 13.35))
labels1 = ["美式咖啡群", "麵店餐點群", "牛肉麵店群", "其他細群…"]
subs1 = ["≥10 次", "≥10 次", "≥10 次", "自動識別"]
for i, (lab, sub) in enumerate(zip(labels1, subs1)):
    cx = 1.55 + i * 2.3
    box(cx, 12.8, 2.05, 0.9, lab, sub, GREY, tsize=10, ssize=8, dashed=(i == 3))
ax.text(9.55, 12.8, "≈110 群", ha="right", va="center", fontsize=9,
        color="#888", style="italic")

dsep(11.95)

# ===== Step 2（拆成兩個框）=====
step_label(11.55, "Step 2　兩層合併（故事層）")
bf = box(5, 10.7, 6.4, 0.9, "去除 noise",
         "濾掉 HDBSCAN 標記為 label=−1 的離群品項", PURPLE)
arrow((5, 11.95 - 0.55), bf["top"])
ba = box(5, 9.2, 7.2, 0.95, "Agglomerative Hierarchical Clustering",
         "對細群中心向量做 Ward 合併（可指定 K、確定性）", PURPLE)
arrow(bf["bot"], ba["top"])
arrow(ba["bot"], (5, 8.05))
labels2 = ["正餐主食", "手搖飲料", "瓶裝茶飲", "…共 25 類"]
subs2 = ["38 款", "54 款", "64 款", "超類型"]
for i, (lab, sub) in enumerate(zip(labels2, subs2)):
    cx = 1.55 + i * 2.3
    box(cx, 7.5, 2.05, 0.9, lab, sub, GREY, tsize=10, ssize=8, dashed=(i == 3))

dsep(6.65)

# ===== Step 3 =====
step_label(6.25, "Step 3　個人化消費增量分析")
b4 = box(5, 5.4, 7.0, 0.95, "各超類型「這個月 vs 平常」月均單價 / 數量",
         "平常 = 之前各月月均；這個月 = 最新月", SALMON)
arrow((5, 6.65 - 0.55), b4["top"])
b5 = box(5, 3.85, 6.4, 0.95, "兩因子拆解公式",
         "總增量 = 價的效應（變貴） + 量的效應（買更多）", PURPLE)
arrow(b4["bot"], b5["top"])
o1 = box(2.9, 2.1, 3.0, 0.95, "價的效應", "東西變貴了多少", TEAL, tsize=11, ssize=8.5)
o2 = box(7.1, 2.1, 3.0, 0.95, "量的效應", "買更多了多少", TEAL, tsize=11, ssize=8.5)
arrow(b5["bot"], (2.9, 2.58))
arrow(b5["bot"], (7.1, 2.58))

ax.text(5, 1.1, "驗證：價效應 + 量效應 = 總增量　|　可下鑽到品項層級回查",
        ha="center", va="center", fontsize=9, color="#777", style="italic")

fig.tight_layout()
out = cc.OUT_DIR / "layer3_architecture.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"已輸出 {out}")
