"""步驟 1b：分群方法對照實驗（給報告用的「嘗試紀錄」）。

商品分群是非監督任務，沒有直接準確率，因此用成員 A 的 6 類消費標籤當**外部基準**：
一個好的商品群不應該混到不同消費類別。指標：
  外部（對 6 類 label_id）：purity、homogeneity、NMI、ARI
  內部：silhouette（在分群所用的空間上、排除 noise）
  結構：n_clusters、noise 比例

為了公平比較，分群在兩種固定粒度下進行：
  細粒度 K=108（商品群，與 HDBSCAN 自動找到的群數同等級）
  粗粒度 K=6  （對齊成員 A 的 6 類消費，檢驗 embedding 能否還原類別）

實驗矩陣：
  降維 : raw768(L2正規化) / PCA30 / UMAP5
  分群 : HDBSCAN(自動) / KMeans(K=108,K=6) / Agglomerative-Ward(K=108,K=6)

輸出：
  outputs/exp_clustering.csv   結果對照表
  outputs/exp_clustering.png   purity / NMI / silhouette 長條圖
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, homogeneity_score,
    normalized_mutual_info_score, adjusted_rand_score,
)

import c_common as cc

RANDOM_STATE = 42
FINE_K = 108    # 細粒度：與 UMAP5+HDBSCAN 自動群數同級，公平比 HDBSCAN
COARSE_K = 6    # 粗粒度：對齊 6 類消費類別


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """排除 noise(-1) 後，每群取多數類別，多數樣本佔比。"""
    mask = labels != -1
    if mask.sum() == 0:
        return float("nan")
    d = pd.DataFrame({"c": labels[mask], "t": truth[mask]})
    maj = d.groupby("c")["t"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return maj / mask.sum()


def eval_clustering(name, space, labels, X_metric, truth):
    """算一組分群的全部指標。X_metric 是計算 silhouette 用的空間。"""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    noise = float(np.mean(labels == -1))
    sil = float("nan")
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = silhouette_score(X_metric[mask], labels[mask])
    # 外部指標一律在非 noise 子集上算，讓有/無 noise 的方法可比；coverage 另外報
    homo = homogeneity_score(truth[mask], labels[mask]) if mask.sum() else float("nan")
    nmi = normalized_mutual_info_score(truth[mask], labels[mask]) if mask.sum() else float("nan")
    ari = adjusted_rand_score(truth[mask], labels[mask]) if mask.sum() else float("nan")
    return {
        "config": name, "space": space,
        "n_clusters": n_clusters, "noise_ratio": round(noise, 3),
        "coverage": round(1 - noise, 3),
        "purity": round(purity(labels, truth), 3),
        "homogeneity": round(homo, 3),
        "NMI": round(nmi, 3), "ARI": round(ari, 3),
        "silhouette": round(sil, 3) if sil == sil else np.nan,
    }


def run_hdbscan(X):
    from hdbscan import HDBSCAN
    return HDBSCAN(min_cluster_size=10, min_samples=5,
                   metric="euclidean", cluster_selection_method="eom").fit_predict(X)


def make_spaces(emb):
    """回傳 {空間名: (分群用矩陣, silhouette用矩陣)}。"""
    raw = normalize(emb.astype(np.float64))                       # L2 正規化→歐氏≈cosine
    pca = PCA(n_components=30, random_state=RANDOM_STATE).fit_transform(emb)
    from umap import UMAP
    umap5 = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine", random_state=RANDOM_STATE).fit_transform(emb.astype(np.float64))
    return {"raw768": raw, "PCA30": pca, "UMAP5": umap5}


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df["purity"], w, label="purity (vs 6-class)")
    ax.bar(x, df["NMI"], w, label="NMI")
    ax.bar(x + w, df["silhouette"].fillna(0), w, label="silhouette")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("Clustering experiments: 3 reducers x 3 clusterers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    df, emb = cc.load_data(require_embeddings=True)
    truth = df[cc.COL_LABELID].to_numpy()
    print(f"資料 {len(df)} 筆，6 類分布：{np.bincount(truth)}")

    spaces = make_spaces(emb)
    results = []
    for sname, X in spaces.items():
        # --- 細粒度（商品群）---
        lab = run_hdbscan(X)
        results.append(eval_clustering(f"{sname}+HDBSCAN", sname, lab, X, truth))
        lab = KMeans(n_clusters=FINE_K, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        results.append(eval_clustering(f"{sname}+KMeans(K={FINE_K})", sname, lab, X, truth))
        lab = AgglomerativeClustering(n_clusters=FINE_K, linkage="ward").fit_predict(X)
        results.append(eval_clustering(f"{sname}+Agglo(K={FINE_K})", sname, lab, X, truth))
        # --- 粗粒度（對齊 6 類）---
        lab = KMeans(n_clusters=COARSE_K, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        results.append(eval_clustering(f"{sname}+KMeans(K={COARSE_K})", sname, lab, X, truth))
        print(f"  {sname} 完成")

    out = pd.DataFrame(results)
    out.to_csv(cc.OUT_DIR / "exp_clustering.csv", index=False, encoding="utf-8-sig")
    plot(results, cc.OUT_DIR / "exp_clustering.png")
    print("\n=== 分群方法對照 ===")
    print(out.to_string(index=False))
    print("\n已輸出 exp_clustering.csv / exp_clustering.png")


if __name__ == "__main__":
    main()
