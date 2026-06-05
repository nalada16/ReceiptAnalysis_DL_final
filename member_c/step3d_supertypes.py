"""步驟 3d：兩層式商品分類 — 把細群合併成「超類型」，在超類型層跑增量分析。

第一層：step1 的細群（~110，精準追價用）。
第二層：對細群的「中心向量」做 Agglomerative(ward) 合併成 K 個超類型（好講故事用）。
同時跑 K=15 / 25 / 35，比較哪個最好講故事。

超類型層的單價=該大類的「每件平均花費」，price_effect 同時涵蓋真漲價與「換買更貴的款」。
（超類型本來就跨價位，故此層不套用細群的單價一致性過濾。）

輸出（outputs/）：
  supertypes_K{K}.csv          每個 K 的超類型 → 成員細群/品項 對照
  step3d_K{K}_user{u}.png      每個 K、每位使用者的增量拆解 + 成長歷史
  step3d_compare_summary.csv   三種 K 的逐人摘要（給你選）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import c_common as cc
import step3c_personal_inflation as s3

KS = [15, 25, 35]


def load_clean_with_emb():
    """載入明細+embedding，附細群 topic，剔除 noise、通用佔位品名、以及不純的細群。

    ★ 合併成超類型「之前」就要在細群層剔除不純群（如 topic 0 的代收/購物袋雜物群），
      否則它會在取消一致性過濾的超類型層復活並主導結果。
    """
    df, emb = cc.load_data(require_embeddings=True)
    df = df.reset_index(drop=True)
    clus = pd.read_csv(cc.OUT_DIR / "step1_clusters.csv")
    df["fine_topic"] = clus["topic"].values
    keep = df["fine_topic"] != -1
    item_low = df[cc.COL_ITEM].astype(str).str.lower()
    keep &= ~item_low.apply(lambda s: any(k in s for k in s3.GENERIC_KEYWORDS))
    df = df[keep].copy()
    emb = emb[df.index.to_numpy()]

    # 細群層一致性過濾：剔除單價落差過大的不純細群（與 step3c 同口徑）
    def coherent(s):
        p10, p90 = s.quantile(0.10), s.quantile(0.90)
        return (p90 / p10) <= s3.MAX_PRICE_RATIO if p10 > 0 else False
    ok = df.groupby("fine_topic")[cc.COL_UNIT].apply(coherent)
    good = ok[ok].index
    mask = df["fine_topic"].isin(good).to_numpy()
    df, emb = df[mask].copy(), emb[mask]
    print(f"細群層剔除不純群 {len(ok) - len(good)} 個，保留 {len(good)} 個細群")
    return df.reset_index(drop=True), emb


def assign_super(df: pd.DataFrame, emb: np.ndarray, K: int) -> pd.DataFrame:
    """對細群中心向量做 ward 合併成 K 個超類型，回傳附 topic(=super)+type_label 的 df。"""
    fines = sorted(df["fine_topic"].unique())
    cents = np.vstack([emb[(df["fine_topic"] == t).to_numpy()].mean(axis=0) for t in fines])
    sup = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(cents)
    fine2sup = dict(zip(fines, sup))

    s = df.copy()
    s["super"] = s["fine_topic"].map(fine2sup)
    # 超類型標籤 = 前 3 名品項，讓它讀起來像「大類」而非單一品項
    def lab(g):
        its = g[cc.COL_ITEM].value_counts().head(3).index.tolist()
        return "/".join(str(i)[:7] for i in its) + f"…({g[cc.COL_ITEM].nunique()}款)"
    label_map = s.groupby("super").apply(lab, include_groups=False)
    s["topic"] = s["super"]                                   # 讓 step3c 以 super 為單位
    s["type_label"] = s["super"].map(label_map)
    return s


def taxonomy(s: pd.DataFrame, K: int):
    """輸出超類型→成員細群/品項對照表。"""
    rows = []
    for sup, g in s.groupby("super"):
        fine_items = (g.groupby("fine_topic")[cc.COL_ITEM]
                      .agg(lambda x: x.value_counts().index[0]))
        rows.append({
            "super": sup,
            "label": g["type_label"].iloc[0],
            "n_fine_clusters": g["fine_topic"].nunique(),
            "n_distinct_items": g[cc.COL_ITEM].nunique(),
            "n_rows": len(g),
            "unit_price_p50": round(g[cc.COL_UNIT].median(), 1),
            "member_fine_types": " | ".join(map(str, fine_items.tolist()[:12])),
        })
    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    out.to_csv(cc.OUT_DIR / f"supertypes_K{K}.csv", index=False, encoding="utf-8-sig")
    return out


def run_increment(s: pd.DataFrame, K: int):
    """在超類型層跑增量分析（重用 step3c 邏輯），逐人出圖+摘要。"""
    summary = []
    for u in sorted(s[cc.COL_USER].unique()):
        sub = s[s[cc.COL_USER] == u]
        dec, cur, n_base = s3.decompose_user(sub)
        if dec.empty:
            summary.append({"K": K, "user_id": u, "可分析超類型": 0,
                            "本月多花": np.nan, "最大貢獻類型": "-"})
            continue
        dec.insert(0, "user_id", u)
        total = dec["delta_spend"].sum()
        top = dec.reindex(dec["delta_spend"].abs().sort_values(ascending=False).index).iloc[0]
        summary.append({
            "K": K, "user_id": u, "可分析超類型": len(dec),
            "本月多花": round(total, 0),
            "最大貢獻類型": f"{top['type_label']} ({top['delta_spend']:+.0f})",
        })
        s3.plot_user(u, dec, sub, cur, n_base, total, cc.OUT_DIR / f"step3d_K{K}_user{u}.png")
    return summary


def main():
    df, emb = load_clean_with_emb()
    print(f"清理後 {len(df)} 筆、{df['fine_topic'].nunique()} 個細群")
    all_sum = []
    for K in KS:
        s = assign_super(df, emb, K)
        tax = taxonomy(s, K)
        all_sum += run_increment(s, K)
        print(f"\n===== K={K}：{len(tax)} 個超類型（前 12 大）=====")
        print(tax.head(12)[["label", "n_fine_clusters", "n_rows", "unit_price_p50"]].to_string(index=False))

    summ = pd.DataFrame(all_sum)
    summ.to_csv(cc.OUT_DIR / "step3d_compare_summary.csv", index=False, encoding="utf-8-sig")
    print("\n\n===== 三種 K 逐人摘要（選哪個最好講故事）=====")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
