"""對照實驗：BERT-UMAP 之後 concat 價格 feature 再分群。

流程：
  BERT 768d ──UMAP(5D, cosine)──► U  (每維 standardize → std=1)
                                  │
  price_bucket ─ordinal {VL=0..VH=4}─► standardize → p  (std=1)
                                  │
  X = [ U, w·p ]  → HDBSCAN（與主流程同參數）
                  → 比較 noise / silhouette / 群數 / 對 6 類的 purity

weight w 的設計理由：
  U 每維 std=1（5 維），p std=1（1 維）
  - w=1   : 與 UMAP 任一維等權，價格佔總 variance ≈ 1/6 = 16.7%
  - w=√5  : 價格 variance 等於 UMAP 5 維總和（50%）
  - w=0.5 : 弱影響當對照
  - w=2   : 強影響當對照
  - w=0   : 不加價格（基準＝純 UMAP）

輸出：
  outputs/pf_clustering_comparison.csv  各 w 的指標對照
  outputs/pf_clustering_comparison.png  視覺化
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, homogeneity_score, normalized_mutual_info_score

import pf_common as pf
import c_common as cc
from step1b_cluster_experiments import purity, run_hdbscan

RANDOM_STATE = 42
BUCKET_MAP = {"VERY_LOW": 0, "LOW": 1, "MID": 2, "HIGH": 3, "VERY_HIGH": 4}
WEIGHTS = [0.0, 0.5, 1.0, 2.0, math.sqrt(5)]   # 0 = 純 UMAP 基準線


def umap5(emb: np.ndarray) -> np.ndarray:
    from umap import UMAP
    return UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                metric="cosine", random_state=RANDOM_STATE).fit_transform(emb.astype(np.float64))


def metrics(labels: np.ndarray, space: np.ndarray, truth: np.ndarray) -> dict:
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    sil = np.nan
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = round(silhouette_score(space[mask], labels[mask]), 3)
    return {
        "n_clusters": n_clusters,
        "noise_ratio": round(float(np.mean(labels == -1)), 3),
        "coverage": round(float(np.mean(mask)), 3),
        "purity": round(purity(labels, truth), 3),
        "homogeneity": round(homogeneity_score(truth[mask], labels[mask]), 3) if mask.sum() else np.nan,
        "NMI": round(normalized_mutual_info_score(truth[mask], labels[mask]), 3) if mask.sum() else np.nan,
        "silhouette": sil,
    }


def plot(out: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for cand in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PingFang TC"]:
        if any(cand in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            plt.rcParams["axes.unicode_minus"] = False
            break

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(out))
    labels = [f"w={w}" + ("（純 UMAP）" if w == 0 else "") for w in out["weight"]]

    axes[0].bar(x, out["n_clusters"], color="#5b8def")
    axes[0].set_title("群數")
    for xi, v in zip(x, out["n_clusters"]):
        axes[0].text(xi, v + 0.5, f"{v}", ha="center", fontsize=9)

    axes[1].bar(x, out["noise_ratio"], color="#c0392b")
    axes[1].set_title("noise 比例（越低越好）")
    axes[1].set_ylim(0, max(out["noise_ratio"]) * 1.3 + 0.01)
    for xi, v in zip(x, out["noise_ratio"]):
        axes[1].text(xi, v + 0.003, f"{v:.2f}", ha="center", fontsize=9)

    axes[2].bar(x, out["silhouette"].fillna(0), color="#3fae6e")
    axes[2].set_title("silhouette（越高越好）")
    axes[2].set_ylim(0, 1.05)
    for xi, v in zip(x, out["silhouette"].fillna(0)):
        axes[2].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    fig.suptitle("BERT-UMAP + 價格 feature（不同 weight 對分群品質的影響）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def main():
    df, emb = cc.load_data(require_embeddings=True)
    df = df.reset_index(drop=True)
    truth = df[cc.COL_LABELID].to_numpy()

    print("跑 UMAP(5D)…")
    U = umap5(emb)
    U_std = StandardScaler().fit_transform(U)        # 每維 std=1

    # 價格 ordinal feature
    p_raw = df[cc.COL_BUCKET].map(BUCKET_MAP).to_numpy().reshape(-1, 1).astype(float)
    p_std = StandardScaler().fit_transform(p_raw)    # std=1

    rows = []
    for w in WEIGHTS:
        X = np.hstack([U_std, w * p_std]) if w > 0 else U_std
        labels = run_hdbscan(X)
        m = metrics(labels, X, truth)
        m = {"weight": round(w, 2), **m,
             "feature_dim": X.shape[1],
             "price_var_share": round((w**2) / (X.shape[1] - 1 + w**2), 3) if w > 0 else 0.0}
        rows.append(m)
        print(f"  w={w:.2f}: 群={m['n_clusters']:3d}  noise={m['noise_ratio']:.1%}  "
              f"silhouette={m['silhouette']}  purity={m['purity']}")

    out = pd.DataFrame(rows)
    out.to_csv(pf.OUT / "pf_clustering_comparison.csv", index=False, encoding="utf-8-sig")
    plot(out, pf.OUT / "pf_clustering_comparison.png")
    print("\n=== 對照表 ===")
    print(out.to_string(index=False))
    print(f"\n已輸出至 {pf.OUT}")


if __name__ == "__main__":
    main()
