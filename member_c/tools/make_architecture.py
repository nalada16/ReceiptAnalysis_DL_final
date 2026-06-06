"""生成系統架構圖（投影片用 PNG）。A 資料/分類 → B 時序 → C 分群/通膨/整合。"""
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

# 配色
C_DATA = "#9e9e9e"
C_A = "#5b8def"
C_B = "#3fae6e"
C_C = "#e08a4b"
C_INT = "#9b59b6"

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")


def band(x0, x1, y0, y1, color, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.02,rounding_size=0.15",
                 fc=color, ec=color, alpha=0.10, zorder=0))
    ax.text(x0 + 0.15, y1 - 0.28, title, fontsize=12, fontweight="bold",
            color=color, ha="left", va="center", zorder=1)


def box(cx, cy, w, h, text, color, fontsize=10, fc=None):
    fc = fc or color
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec=color, lw=1.6, alpha=0.92, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color="white", fontweight="bold", zorder=3, wrap=True)
    return {"top": (cx, cy + h / 2), "bot": (cx, cy - h / 2),
            "l": (cx - w / 2, cy), "r": (cx + w / 2, cy)}


def arrow(p, q, color="#444", style="-", lw=1.8, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                 lw=lw, color=color, linestyle=style,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))
    if label:
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        ax.text(mx, my, label, fontsize=8, color=color, ha="center",
                va="center", bbox=dict(fc="white", ec="none", alpha=0.85), zorder=4)


# ---- 標題 ----
ax.text(7, 10.7, "電子發票語意分類與個人消費行為診斷 — 系統架構",
        fontsize=15, fontweight="bold", ha="center")

# ---- 資料源 ----
data = box(7, 10.0, 4.6, 0.62,
           "原始電子發票 CSV  (2613 筆 · 3 人 · 2025-09~2026-05)", C_DATA, 10)

# ---- 成員 A ----
band(0.4, 13.6, 7.5, 9.6, C_A, "成員 A｜資料工程 + Task1 分類")
a1 = box(2.4, 9.0, 2.6, 0.6, "去識別化 + 欄位清洗", C_A, 9.5)
a2 = box(5.5, 9.0, 2.6, 0.6, "文字正規化\n全形→半形 / 小寫 / store", C_A, 9)
a3 = box(8.6, 9.0, 2.6, 0.6, "LLM 預標註 + 人工審核", C_A, 9.5)
a4 = box(11.7, 9.0, 2.8, 0.6, "BERT 分類 fine-tune\nckiplab/bert-base-chinese", C_A, 8.5)
out_label = box(4.5, 8.0, 3.2, 0.58, "輸出①  6 類消費標籤", "#3b6fd0", 9.5)
out_emb = box(9.5, 8.0, 3.6, 0.58, "輸出②  BERT [CLS] embedding 768d", "#3b6fd0", 9.5)

# ---- 成員 B ----
band(0.4, 6.85, 4.3, 6.95, C_B, "成員 B｜Task2 L1/L2")
b1 = box(3.5, 6.35, 3.0, 0.55, "每日消費矩陣 + mask", C_B, 9)
b2 = box(2.1, 5.2, 2.4, 0.62, "LSTM 短期預測\n(L1)", C_B, 9)
b3 = box(5.0, 5.2, 2.4, 0.62, "LSTM-AE 異常偵測\n(L2)", C_B, 9)

# ---- 成員 C ----
band(7.15, 13.6, 4.3, 6.95, C_C, "成員 C｜Task2 L3 + 整合")
c1 = box(10.3, 6.35, 4.2, 0.55, "BERTopic + UMAP + HDBSCAN 商品分群", C_C, 9)
c2 = box(8.9, 5.2, 2.7, 0.62, "重複購買群\n+ 單價時序", C_C, 9)
c3 = box(12.1, 5.2, 2.7, 0.62, "量 / 價效應\n通膨拆解 (L3)", C_C, 9)

# ---- 整合 ----
integ = box(7, 3.1, 7.2, 0.66, "整合報告 · 系統架構圖 · 簡報（全員）", C_INT, 11)

# ---- 箭頭 ----
arrow(data["bot"], (7, 9.32), C_DATA)
arrow(a1["r"], a2["l"], C_A)
arrow(a2["r"], a3["l"], C_A)
arrow(a3["r"], a4["l"], C_A)
arrow((6.5, 8.7), out_label["top"], C_A, rad=-0.15)   # 分類→標籤
arrow(a4["bot"], out_emb["top"], C_A)                  # BERT→embedding
arrow(out_label["bot"], b1["top"], C_B, rad=-0.1)      # 標籤→每日矩陣
arrow(out_emb["bot"], c1["top"], C_C, rad=0.1)         # embedding→分群
arrow(out_emb["bot"], (2.1, 5.55), "#888", style="--", lw=1.4,
      label="語意特徵(選配)", rad=0.25)                  # embedding→LSTM(虛線)
arrow(b1["bot"], b2["top"], C_B, rad=0.1)
arrow(b1["bot"], b3["top"], C_B, rad=-0.1)
arrow(c1["bot"], c2["top"], C_C, rad=0.1)
arrow(c2["r"], c3["l"], C_C)
arrow(b2["bot"], (6.2, 3.45), C_INT, rad=0.15)
arrow(b3["bot"], (6.6, 3.45), C_INT, rad=0.1)
arrow(c2["bot"], (7.6, 3.45), C_INT, rad=-0.1)
arrow(c3["bot"], (8.2, 3.45), C_INT, rad=-0.15)

# ---- 註腳 ----
ax.text(7, 2.3, "核心任務：Task1 分類 + Task2-L1 預測　|　延伸任務：L2 異常、L3 通膨拆解",
        fontsize=9.5, ha="center", color="#555", style="italic")

fig.tight_layout()
out = cc.OUT_DIR / "system_architecture.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"已輸出 {out}")
