"""步驟 3c：個人消費增量分析（Layer 3 主交付）。

對齊定義：解釋「這個月」比「平常」多花的錢，落在哪些**類型商品**，並拆解：
  變貴（50元珍奶→70元珍奶）  vs  買更多（一個月5杯→10杯）。

★ 獨立新檔，不影響 step3 / step3b。類型 = step1 的語意商品群（cluster）。
  期間：每人「這個月」= 最新月；「平常」= 之前所有月的月均。

每個類型群：
  P0 = 平常加權單價 = Σ基期金額 / Σ基期數量
  Q0 = 平常每月平均數量 = Σ基期數量 / 基期月數
  P1 = 這個月單價         Q1 = 這個月數量
  ΔS = P1·Q1 − P0·Q0      （這個月 − 一個典型月，在此類型上的花費差）
  變貴 price_effect = (P1−P0)·Q1     買更多 qty_effect = (Q1−Q0)·P0
  （新類型：P0:=P1 → 全歸買更多；本月停買：P1:=P0,Q1=0 → 負的買更多）

全部類型 ΣΔS = 這個月總花費 − 平常月均總花費 = 「這個月多花的錢」。

輸出（outputs/）：
  step3c_increment_breakdown.csv   每人每類型的 ΔS / 變貴 / 買更多
  step3c_user_summary.csv          每人：多花多少、變貴佔比、買更多佔比
  step3c_user{u}.png               每人雙圖：增量拆解 + 類型單價成長歷史
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import c_common as cc

TOP_BREAKDOWN = 8   # 增量拆解圖顯示前 N 個貢獻最大的類型
TOP_HISTORY = 5     # 成長歷史圖顯示前 N 個花費最高的類型
MAX_PRICE_RATIO = 6  # 類型一致性：群內單價 p90/p10 超過此倍數視為「不純的類型」剔除

# 增量分析的「值得分析」門檻（per-user、每類型）：
#   基期須出現 ≥ MIN_BASE_MONTHS 個月（「平常」要是多月平均，P0/Q0 才穩）
#   且（月均數量 ≥ MIN_MONTHLY_QTY 或 月均花費 ≥ MIN_MONTHLY_SPEND）
#   → 量大（如每天喝的飲料）或花錢多（如貴但少買的正餐）擇一即納入。
MIN_BASE_MONTHS = 2       # 基期至少出現幾個月
MIN_MONTHLY_QTY = 10      # 平常每月平均數量門檻
MIN_MONTHLY_SPEND = 100   # 平常每月平均花費門檻（NT$）

# 通用/佔位品名：這些不是可比較的真實商品（單價無意義），整列剔除。
GENERIC_KEYWORDS = [
    "代收", "兩用袋", "購物袋", "塑膠袋", "塑膠", "提袋", "紙袋",
    "餐飲費", "餐飲與食品", "餐飲", "餐廳", "f&b", "商品一批", "商品",
    "小吃", "food court", "美食街", "熟食", "升級", "服務費", "外送費", "運費",
    "代購", "雜項", "其他", "一批", "1百", "百元",
]


def load_typed():
    """載入明細並附上語意類型群（topic），丟掉 noise 與通用佔位品名，建立可讀標籤。"""
    df, _ = cc.load_data(require_embeddings=False)
    clus = pd.read_csv(cc.OUT_DIR / "step1_clusters.csv")
    df["topic"] = clus["topic"].values
    df = df[df["topic"] != -1].copy()

    # 剔除通用佔位品名（單價不可比）
    item_low = df[cc.COL_ITEM].astype(str).str.lower()
    mask_generic = item_low.apply(lambda s: any(k in s for k in GENERIC_KEYWORDS))
    n_before = len(df)
    df = df[~mask_generic].copy()
    print(f"剔除通用佔位品名 {n_before - len(df)} 筆，剩 {len(df)} 筆")

    # 類型一致性過濾：群內單價(p90/p10)落差過大代表「不純的類型」，單價不可比 → 剔除
    def coherent(s):
        p10, p90 = s.quantile(0.10), s.quantile(0.90)
        return (p90 / p10) <= MAX_PRICE_RATIO if p10 > 0 else False
    keep_topic = df.groupby("topic")[cc.COL_UNIT].apply(coherent)
    bad = keep_topic[~keep_topic].index.tolist()
    df = df[df["topic"].isin(keep_topic[keep_topic].index)].copy()
    print(f"剔除不純類型 {len(bad)} 群（單價落差>p90/p10>{MAX_PRICE_RATIO}），剩 {df['topic'].nunique()} 個類型")

    # 類型標籤 = 群內最高頻品名 +「等N款」，凸顯這是 cluster(類型)而非單品
    top_item = df.groupby("topic")[cc.COL_ITEM].agg(lambda s: s.value_counts().index[0])
    n_items = df.groupby("topic")[cc.COL_ITEM].nunique()
    df["type_label"] = df["topic"].map(
        lambda t: f"{top_item[t]} 等{n_items[t]}款" if n_items[t] > 1 else f"{top_item[t]}（單品）")

    # 人工命名覆寫：cluster_names.csv 有填 custom_name 就用你的名字（沒填維持自動）
    import naming
    names = naming.load_names(naming.FINE_NAMES, "topic")
    if names:
        df["type_label"] = df.apply(lambda r: names.get(r["topic"], r["type_label"]), axis=1)
        print(f"套用人工命名 {len(names)} 個細群（cluster_names.csv）")
    return df


def decompose_user(sub: pd.DataFrame):
    """對單一使用者：算每個類型的 P0/Q0/P1/Q1 與增量拆解。"""
    months = sorted(sub["month"].unique())
    cur, base_months = months[-1], months[:-1]
    n_base = len(base_months)
    base = sub[sub["month"].isin(base_months)]
    curm = sub[sub["month"] == cur]

    rows = []
    for t in sub["topic"].unique():
        b = base[base["topic"] == t]
        c = curm[curm["topic"] == t]
        b_qty, b_amt = b[cc.COL_QTY].sum(), b[cc.COL_AMT].sum()
        c_qty, c_amt = c[cc.COL_QTY].sum(), c[cc.COL_AMT].sum()
        if b_qty == 0 and c_qty == 0:
            continue
        # 「值得分析」門檻（per-user、每類型）：
        #   基期須出現 ≥ MIN_BASE_MONTHS 個月（平常要是多月平均）
        #   且（月均數量夠多 或 月均花費夠高）才納入
        b_months = b["month"].nunique()
        if b_months < MIN_BASE_MONTHS:
            continue
        monthly_qty = b_qty / n_base
        monthly_spend = b_amt / n_base
        if not (monthly_qty >= MIN_MONTHLY_QTY or monthly_spend >= MIN_MONTHLY_SPEND):
            continue
        P0 = b_amt / b_qty if b_qty else np.nan
        P1 = c_amt / c_qty if c_qty else np.nan
        Q0 = b_qty / n_base                      # 平常每月平均數量
        Q1 = c_qty                               # 這個月數量
        if np.isnan(P0):       # 新類型：沒有平常基準 → 全算「買更多」
            P0 = P1
        if np.isnan(P1):       # 本月沒買 → 視為單價不變、Q1=0
            P1 = P0
        dS = P1 * Q1 - P0 * Q0
        price_eff = (P1 - P0) * Q1
        qty_eff = (Q1 - Q0) * P0
        rows.append({
            "type_label": sub[sub.topic == t]["type_label"].iloc[0],
            "topic": t, "P0": round(P0, 1), "P1": round(P1, 1),
            "Q0_per_month": round(Q0, 2), "Q1_this_month": round(Q1, 2),
            "baseline_spend": round(P0 * Q0, 1), "current_spend": round(P1 * Q1, 1),
            "delta_spend": round(dS, 1),
            "變貴_price": round(price_eff, 1), "買更多_qty": round(qty_eff, 1),
        })
    out = pd.DataFrame(rows)
    return out, str(cur), n_base


def monthly_unit_price(sub: pd.DataFrame, topics):
    """某些類型的每月加權單價（成長歷史用）。"""
    s = sub[sub["topic"].isin(topics)]
    g = s.groupby(["topic", "month"]).agg(amt=(cc.COL_AMT, "sum"),
                                          qty=(cc.COL_QTY, "sum"))
    g["unit_price"] = g["amt"] / g["qty"]
    return g.reset_index()


def monthly_qty(sub: pd.DataFrame, topics):
    """某些類型的每月數量（數量成長歷史用）。"""
    s = sub[sub["topic"].isin(topics)]
    g = s.groupby(["topic", "month"]).agg(qty=(cc.COL_QTY, "sum"))
    return g.reset_index()


def set_font():
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break


def plot_user(u, dec, sub, cur, n_base, total_extra, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    set_font()

    fig = plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.1, 1], hspace=0.55, wspace=0.35)
    ax1 = fig.add_subplot(gs[:, 0])   # 左欄：增量拆解（佔全高）
    ax2 = fig.add_subplot(gs[0, 1])   # 右上：總花費
    ax3 = fig.add_subplot(gs[1, 1])   # 右中：總數量
    ax4 = fig.add_subplot(gs[2, 1])   # 右下：平均單價

    # --- 左：增量拆解（變貴 vs 買更多）---
    top = dec.reindex(dec["delta_spend"].abs().sort_values(ascending=False).index).head(TOP_BREAKDOWN)
    top = top.iloc[::-1]
    y = np.arange(len(top))
    labels = [str(s)[:14] for s in top["type_label"]]
    ax1.barh(y, top["變貴_price"], color="#d9534f", label="變貴 (價)")
    ax1.barh(y, top["買更多_qty"], left=top["變貴_price"], color="#4a90d9", label="買更多 (量)")
    for yi, d in zip(y, top["delta_spend"]):
        ax1.text(d + (8 if d >= 0 else -8), yi, f"{d:+.0f}",
                 va="center", ha="left" if d >= 0 else "right", fontsize=9, fontweight="bold")
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=10)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel("這個月 vs 平常月均  在此類型上的花費差 (NT$)")
    sign = "多花" if total_extra >= 0 else "少花"
    ax1.set_title(f"user {u}｜這個月({cur})比平常{sign} {abs(total_extra):.0f} 元，花在哪些類型")
    ax1.legend(loc="lower right")

    # 準備右側折線圖資料
    top_spend = dec.reindex(dec["current_spend"].sort_values(ascending=False).index).head(TOP_HISTORY)
    topic_list = top_spend["topic"].tolist()
    label_map = dict(zip(dec["topic"], dec["type_label"]))
    all_months = sorted(sub["month"].unique())
    mpos = {m: i for i, m in enumerate(all_months)}
    xticks = list(range(len(all_months)))
    xlabels = [str(m) for m in all_months]

    # 一次聚合三個指標
    s = sub[sub["topic"].isin(topic_list)]
    g = s.groupby(["topic", "month"]).agg(
        amt=(cc.COL_AMT, "sum"),
        qty=(cc.COL_QTY, "sum")
    ).reset_index()
    g["unit_price"] = g["amt"] / g["qty"]

    def draw_line(ax, col, ylabel, title):
        for t in topic_list:
            d = g[g["topic"] == t].sort_values("month")
            if len(d) < 2:
                continue
            first, last = d[col].iloc[0], d[col].iloc[-1]
            chg = (last / first - 1) * 100 if first else 0
            lab = f"{str(label_map[t])[:10]} ({chg:+.0f}%)"
            ax.plot([mpos[m] for m in d["month"]], d[col].values, marker="o", label=lab)
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")

    draw_line(ax2, "amt",        "總花費 (NT$)",   f"user {u}｜月總花費")
    draw_line(ax3, "qty",        "購買數量",        f"user {u}｜月購買數量")
    draw_line(ax4, "unit_price", "平均單價 (NT$)", f"user {u}｜月平均單價")

    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_typed()
    all_break, summary = [], []
    for u in sorted(df[cc.COL_USER].unique()):
        sub = df[df[cc.COL_USER] == u]
        dec, cur, n_base = decompose_user(sub)
        if dec.empty:
            continue
        dec.insert(0, "user_id", u)
        total_extra = dec["delta_spend"].sum()
        price_tot = dec["變貴_price"].sum()
        qty_tot = dec["買更多_qty"].sum()
        summary.append({
            "user_id": u, "current_month": cur, "baseline_months": n_base,
            "本月多花的錢": round(total_extra, 1),
            "變貴貢獻": round(price_tot, 1), "買更多貢獻": round(qty_tot, 1),
            "變貴佔比": round(price_tot / total_extra, 2) if total_extra else np.nan,
        })
        all_break.append(dec)
        plot_user(u, dec, sub, cur, n_base, total_extra, cc.OUT_DIR / f"step3c_user{u}.png")

    breakdown = pd.concat(all_break, ignore_index=True)
    breakdown.to_csv(cc.OUT_DIR / "step3c_increment_breakdown.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(summary)
    summ.to_csv(cc.OUT_DIR / "step3c_user_summary.csv", index=False, encoding="utf-8-sig")

    print("=== 個人消費增量摘要（這個月 vs 平常月均）===")
    print(summ.to_string(index=False))
    print("\n=== 各人「這個月多花錢」前 5 類型 ===")
    for u in sorted(breakdown["user_id"].unique()):
        d = breakdown[breakdown.user_id == u]
        top = d.reindex(d["delta_spend"].abs().sort_values(ascending=False).index).head(5)
        print(f"\n[user {u}]")
        print(top[["type_label", "P0", "P1", "Q0_per_month", "Q1_this_month",
                   "delta_spend", "變貴_price", "買更多_qty"]].to_string(index=False))
    print("\n已輸出 step3c_increment_breakdown.csv / step3c_user_summary.csv / step3c_user{0,1,2}.png")


if __name__ == "__main__":
    main()
