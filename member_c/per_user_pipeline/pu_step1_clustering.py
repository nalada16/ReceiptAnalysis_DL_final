"""per-user 步驟1：三個人「各自」分群（不 pooled）。

每位使用者只用自己的 embedding 跑 UMAP+HDBSCAN（參數與主流程 step1 一致）。
群號加上 user 前綴避免跨人撞號；noise 標為 "noise"。
輸出：outputs/pu_clusters.csv（每列附 pu_topic）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pu_common as pu
import c_common as cc

RANDOM_STATE = 42


def cluster_one(emb_u: np.ndarray) -> np.ndarray:
    from umap import UMAP
    from hdbscan import HDBSCAN
    red = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
               metric="cosine", random_state=RANDOM_STATE).fit_transform(emb_u.astype(np.float64))
    return HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                   cluster_selection_method="eom").fit_predict(red)


def main():
    df, emb = cc.load_data(require_embeddings=True)
    df = df.reset_index(drop=True)
    pu_topic = np.empty(len(df), dtype=object)

    for u in sorted(df[cc.COL_USER].unique()):
        idx = (df[cc.COL_USER] == u).to_numpy()
        labels = cluster_one(emb[idx])
        tagged = [f"u{u}_{l}" if l != -1 else "noise" for l in labels]
        pu_topic[idx] = tagged
        n_clu = len(set(labels[labels != -1]))
        noise = float(np.mean(labels == -1))
        print(f"user {u}: {idx.sum()} 筆 → {n_clu} 群, noise {noise:.1%}")

    df["pu_topic"] = pu_topic
    df.to_csv(pu.OUT / "pu_clusters.csv", index=False, encoding="utf-8-sig")
    print(f"\n已輸出 {pu.OUT / 'pu_clusters.csv'}")


if __name__ == "__main__":
    main()
