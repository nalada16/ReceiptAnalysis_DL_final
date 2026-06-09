# -*- coding: utf-8 -*-
"""生成整體專案系統架構圖。

輸出：outputs/system_architecture_full.png
風格：與 layer3_architecture.png 相同（圓角色塊 + Step 分段 + 標題/副標）
三層架構：Task 1（成員 A）→ Task B（成員 B）+ Task C（成員 C）並行 → 整合輸出
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
sys.path.insert(0, r"D:\ReceiptAnalysis_DL_final\member_c")
import c_common as cc

for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [cand]
        plt.rcParams["axes.unicode_minus"] = False
        break

SALMON = dict(fc="#EBA9A3", ec="#D98A82", tc="#6E2B25", sc="#8C463F")
PURPLE = dict(fc="#C9CBEC", ec="#A9ACDD", tc="#34346A", sc="#56568C")
TEAL   = dict(fc="#8FBEB3", ec="#6FA99B", tc="#1E4A40", sc="#356B5E")
GREY   = dict(fc="#F3F3F3", ec="#C2C2C2", tc="#555555", sc="#888888")
ORANGE = dict(fc="#F6D5A8", ec="#E0B070", tc="#6B3A00", sc="#8C5A10")

W, H = 11.5, 20.0
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 12)
ax.set_ylim(0, 22)
ax.axis("off")


def box(cx, cy, w, h, title, subtitle, style, tsize=11, ssize=8.2, dashed=False):
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
    return {"top": (cx, cy + h / 2), "bot": (cx, cy - h / 2),
            "left": (cx - w / 2, cy), "right": (cx + w / 2, cy)}


def arrow(p, q, color="#7A7A7A"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18,
                 lw=1.8, color=color, zorder=1))


def step_label(y, text, x=0.3):
    ax.text(x, y, text, ha="left", va="center", fontsize=12.5,
            fontweight="bold", color="#444444")


def dsep(y):
    ax.plot([0.3, 11.7], [y, y], ls=(0, (4, 4)), color="#CFCFCF", lw=1.2, zorder=0)


def vsep(x, y0, y1):
    ax.plot([x, x], [y0, y1], ls=(0, (4, 4)), color="#CFCFCF", lw=1.2, zorder=0)


# ===== 輸入資料 =====
inp = box(6, 21.2, 8.0, 0.9, "原始電子發票資料",
          "3 位使用者 · 2613 筆明細 · 2025-09 ~ 2026-05", GREY, tsize=12)
arrow(inp["bot"], (6, 20.3))

dsep(20.1)

# ===== Task 1：成員 A =====
step_label(19.75, "Task 1　成員 A — 語意分類與向量化")
b_pre = box(6, 19.1, 7.2, 0.95, "文字前處理 + LLM 半自動標注",
            "全形轉半形、店名去雜訊、金額離散化（5 區間）", GREY, tsize=10.5)
arrow((6, 20.1), b_pre["top"])
b_bert = box(6, 17.7, 7.8, 0.95, "BERT Fine-tuning（ckiplab/bert-base-chinese）",
             "輸入：品名 [SEP] 店家 [SEP] 金額區間　|　Macro F1 = 0.92", SALMON)
arrow(b_pre["bot"], b_bert["top"])
arrow(b_bert["bot"], (6, 16.55))

# 兩個輸出框
out_label = box(3.2, 15.95, 4.8, 0.9, "6 類消費標籤",
                "飲食 / 交通 / 購物 / 娛樂 / 教育 / 醫療", TEAL, tsize=10.5)
out_emb   = box(8.8, 15.95, 4.8, 0.9, "BERT [CLS] 向量",
                "768 維語意表示（每筆明細）", TEAL, tsize=10.5)
arrow((6, 16.55), (3.2, 16.4))
arrow((6, 16.55), (8.8, 16.4))

dsep(15.45)

# ===== 兩欄並行 =====
# 左欄中心 x=3.0，右欄中心 x=9.0
LX, RX = 3.0, 9.0

step_label(15.1, "Task B　成員 B — 時序預測與異常偵測", x=0.3)
step_label(15.1, "Task C　成員 C — 商品分群與增量分析", x=6.3)

vsep(6.15, 3.0, 15.2)

# ---- Task B（左欄）----
arrow(out_label["bot"], (LX, 14.65))

bB1 = box(LX, 14.1, 5.2, 0.9, "每日消費時序建立",
          "補齊缺失日 · has_transaction mask · 時間特徵", PURPLE, tsize=10.5)
arrow((LX, 14.65), bB1["top"])

bB2 = box(LX, 12.65, 5.2, 0.95, "Routine Forecasting",
          "LSTM / MLP / ARIMA · 過去 30 天預測未來 7 天", SALMON, tsize=10.5)
arrow(bB1["bot"], bB2["top"])

bB3 = box(LX, 11.1, 5.2, 0.95, "Anomaly Detection",
          "LSTM Autoencoder · 個人化 p95 重建誤差門檻", SALMON, tsize=10.5)
arrow(bB2["bot"], bB3["top"])
arrow(bB3["bot"], (LX, 10.15))

obB1 = box(1.6, 9.6, 2.3, 0.85, "短期預測", "7 天逐日消費", ORANGE, tsize=10, ssize=8)
obB2 = box(4.4, 9.6, 2.3, 0.85, "異常分數", "值得回查的日期", ORANGE, tsize=10, ssize=8)
arrow((LX, 10.15), (1.6, 9.98))
arrow((LX, 10.15), (4.4, 9.98))

# ---- Task C（右欄）----
arrow(out_emb["bot"], (RX, 14.65))

bC1 = box(RX, 14.1, 5.2, 0.9, "UMAP(5D) + HDBSCAN 分群",
          "noise 6.9% · silhouette 0.88 · ~110 細群", PURPLE, tsize=10.5)
arrow((RX, 14.65), bC1["top"])

bC2 = box(RX, 12.65, 5.2, 0.95, "AHC（Ward）合併超類型",
          "細群中心向量合併 · 可指定 K=25 · 確定性", SALMON, tsize=10.5)
arrow(bC1["bot"], bC2["top"])

bC3 = box(RX, 11.1, 5.2, 0.95, "個人化消費增量分析",
          "兩因子拆解：變貴效應 + 買更多效應", SALMON, tsize=10.5)
arrow(bC2["bot"], bC3["top"])
arrow(bC3["bot"], (RX, 10.15))

obC1 = box(7.6, 9.6, 2.3, 0.85, "消費增量", "本月多花在哪類", ORANGE, tsize=10, ssize=8)
obC2 = box(10.4, 9.6, 2.3, 0.85, "時序趨勢", "單價/數量歷史變化", ORANGE, tsize=10, ssize=8)
arrow((RX, 10.15), (7.6, 9.98))
arrow((RX, 10.15), (10.4, 9.98))

dsep(9.1)

# ===== 整合輸出 =====
step_label(8.75, "整合診斷輸出")
arrow((LX, 9.18), (6, 8.35))
arrow((RX, 9.18), (6, 8.35))

bInt = box(6, 7.85, 9.0, 0.95, "個人化消費行為診斷 Pipeline",
           "分類 → 預測 / 異常偵測 → 增量歸因　|　可重跑 · 可比較 · 可解釋", TEAL, tsize=11.5)

ax.text(6, 6.9,
        "B 找出「哪天異常」· C 解釋「異常落在哪個類型、變貴還是買更多」",
        ha="center", va="center", fontsize=9, color="#666", style="italic")

fig.tight_layout()
out = cc.OUT_DIR / "system_architecture_full.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
plt.close(fig)
print(f"已輸出 {out}")
