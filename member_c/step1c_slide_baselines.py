"""步驟 1c：補齊投影片「任務二 Layer3 比較對象」那 4 列的真實數字。

投影片的四個比較對象：
  1. Regex 關鍵字比對        (Rule-based 下限)  ── 本檔實作
  2. TF-IDF + HDBSCAN        (傳統 ML 對照)     ── 本檔實作
  3. BERT Embedding + K-Means(消融實驗)         ── 取自 exp_clustering.csv (UMAP5+KMeans)
  4. BERTopic (BERT+HDBSCAN) (Ours)             ── 取自 exp_clustering.csv (UMAP5+HDBSCAN)

評估指標與 step1b 一致：對 6 類消費標籤的 purity/homogeneity/NMI/ARI + 內部 silhouette + noise。
輸出：outputs/exp_slide_baselines.csv
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score, homogeneity_score,
    normalized_mutual_info_score, adjusted_rand_score,
)

import c_common as cc
from step1b_cluster_experiments import purity, run_hdbscan

# 關鍵字規則：每個 pattern 對應一個群（rule-based 下限）。
# 刻意只涵蓋常見品項，凸顯規則法「覆蓋率有限、無法泛化」的缺點。
KEYWORD_GROUPS = {
    "美式咖啡": r"美式",
    "拿鐵": r"拿鐵|latte",
    "其他咖啡": r"咖啡|coffee",
    "紅茶": r"紅茶",
    "綠茶": r"綠茶",
    "奶茶": r"奶茶",
    "青茶四季春": r"四季春|青茶",
    "燕麥": r"燕麥",
    "便當": r"便當",
    "飯糰": r"飯糰",
    "麵": r"麵",
    "飯類": r"飯",
    "雞肉餐": r"雞",
    "停車": r"parking|停車",
    "悠遊卡交通": r"悠遊|捷運|高鐵|台鐵",
    "美妝": r"粉底|眉筆|屈臣|洗髮|沐浴",
    "球類運動": r"桌球|球",
    "維他命保健": r"維他命|保健|藥",
    "列印文具": r"列印|ibon|文具|書",
}


def regex_keyword_labels(df: pd.DataFrame) -> np.ndarray:
    """依關鍵字規則給群號；都不匹配 → -1（視為 noise，無法處理）。"""
    text = (df[cc.COL_ITEM].astype(str) + " " + df[cc.COL_STORE].astype(str))
    labels = np.full(len(df), -1, dtype=int)
    for gid, (_, pat) in enumerate(KEYWORD_GROUPS.items()):
        hit = text.str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
        # 只填還沒被前面規則指派到的（規則有優先序）
        labels[(labels == -1) & hit.values] = gid
    return labels


def tfidf_hdbscan_labels(df: pd.DataFrame):
    """TF-IDF 向量化（jieba 斷詞）→ L2 正規化 → HDBSCAN。回傳 (labels, 稠密矩陣)。"""
    try:
        import jieba

        def tok(t):
            return [w for w in jieba.cut(t) if w.strip()]
        vec = TfidfVectorizer(tokenizer=tok, token_pattern=None, min_df=2)
    except Exception:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)
    text = (df[cc.COL_ITEM].astype(str) + " " + df[cc.COL_STORE].astype(str)).tolist()
    X = vec.fit_transform(text)
    Xd = normalize(X).toarray().astype(np.float64)   # L2 正規化後歐氏≈cosine
    labels = run_hdbscan(Xd)
    return labels, Xd


def evaluate(name, kind, labels, truth, X_metric=None):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    noise = float(np.mean(labels == -1))
    sil = np.nan
    if X_metric is not None and n_clusters >= 2 and mask.sum() > n_clusters:
        sil = round(silhouette_score(X_metric[mask], labels[mask]), 3)
    return {
        "method": name, "type": kind,
        "n_clusters": n_clusters,
        "noise_ratio": round(noise, 3), "coverage": round(1 - noise, 3),
        "purity": round(purity(labels, truth), 3),
        "homogeneity": round(homogeneity_score(truth[mask], labels[mask]), 3) if mask.sum() else np.nan,
        "NMI": round(normalized_mutual_info_score(truth[mask], labels[mask]), 3) if mask.sum() else np.nan,
        "ARI": round(adjusted_rand_score(truth[mask], labels[mask]), 3) if mask.sum() else np.nan,
        "silhouette": sil,
    }


def plot_slide(out: pd.DataFrame, path):
    """簡報用對照圖：coverage / purity / silhouette 分組長條，Ours 以邊框強調。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break

    metrics = [("coverage", "覆蓋率 (1-noise)"), ("purity", "純度"), ("silhouette", "silhouette")]
    x = np.arange(len(out))
    w = 0.25
    colors = ["#bdbdbd", "#8c9eff", "#c5896f"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (col, lab) in enumerate(metrics):
        vals = out[col].fillna(0).to_numpy()
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    # 標出 noise 比例（紅字）強調「雜訊干擾」主張
    for xi, nz in zip(x, out["noise_ratio"]):
        ax.text(xi, -0.08, f"noise {nz:.0%}", ha="center", va="top",
                fontsize=8, color="#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels(out["method"], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Layer3 分群方法對照：規則 / 傳統 ML / 消融 / Ours")
    ax.legend(loc="lower left")
    # Ours（最後一根）加灰底強調
    ax.axvspan(x[-1] - 0.45, x[-1] + 0.45, color="#fff3e0", zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df, emb = cc.load_data(require_embeddings=True)
    truth = df[cc.COL_LABELID].to_numpy()
    rows = []

    # 1. Regex 關鍵字（rule-based 下限）；silhouette 在 BERT 空間上評估群是否分得開
    emb_norm = normalize(emb.astype(np.float64))
    lab = regex_keyword_labels(df)
    rows.append(evaluate("Regex 關鍵字比對", "Rule-based 下限", lab, truth, X_metric=emb_norm))

    # 2. TF-IDF + HDBSCAN（傳統 ML 對照）
    lab, Xd = tfidf_hdbscan_labels(df)
    rows.append(evaluate("TF-IDF + HDBSCAN", "傳統 ML 對照", lab, truth, X_metric=Xd))

    # 3 & 4：直接取 exp_clustering.csv 的 UMAP5 兩列，確保與主實驗一致
    exp_path = cc.OUT_DIR / "exp_clustering.csv"
    if exp_path.exists():
        exp = pd.read_csv(exp_path).set_index("config")
        for cfg, name, kind in [
            ("UMAP5+KMeans(K=108)", "BERT Embedding + K-Means", "消融實驗"),
            ("UMAP5+HDBSCAN", "BERTopic (BERT + HDBSCAN)", "Ours"),
        ]:
            if cfg in exp.index:
                r = exp.loc[cfg]
                rows.append({
                    "method": name, "type": kind,
                    "n_clusters": int(r["n_clusters"]), "noise_ratio": r["noise_ratio"],
                    "coverage": r["coverage"], "purity": r["purity"],
                    "homogeneity": r["homogeneity"], "NMI": r["NMI"],
                    "ARI": r["ARI"], "silhouette": r["silhouette"],
                })
    else:
        print("[提醒] 找不到 exp_clustering.csv，請先跑 step1b 取得第 3、4 列")

    out = pd.DataFrame(rows)
    out.to_csv(cc.OUT_DIR / "exp_slide_baselines.csv", index=False, encoding="utf-8-sig")
    if len(out) == 4:
        plot_slide(out, cc.OUT_DIR / "exp_slide_baselines.png")
    print("=== 投影片「Layer3 比較對象」四列真實數字 ===")
    print(out.to_string(index=False))
    print("\n已輸出 exp_slide_baselines.csv / exp_slide_baselines.png")


if __name__ == "__main__":
    main()
