"""步驟 1d（獨立對照實驗）：三人「分開跑分群」 vs 原本「pooled 一起跑」。

★ 本檔完全獨立，不修改也不影響 step1 / step1b / step1c 的主流程。
   產出全部寫到 outputs/per_user/ 子資料夾，前綴 exp_per_user_*。

驗證三件事：
  1. 小資料量 user（尤其 user 2）單獨跑時 noise 會不會飆高。
  2. 同一商品被多人購買時，分開跑被迫拆散 → 量化「失去跨人可比性」。
  3. 整體分群品質（silhouette / purity）分開跑變好還變差。

參數與 step1 完全一致（經確認）：UMAP(n_neighbors=15, n_components=5, cosine) +
HDBSCAN(min_cluster_size=10, min_samples=5)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, homogeneity_score, normalized_mutual_info_score,
)

import pu_common as pu  # 設定 path 並提供本資料夾 OUT
import c_common as cc
from step1b_cluster_experiments import purity, run_hdbscan  # 沿用相同評估與分群設定

OUT = pu.OUT
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5


def umap_reduce(emb: np.ndarray) -> np.ndarray:
    """與 step1 相同的 UMAP 設定。"""
    from umap import UMAP
    return UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_N_COMPONENTS,
        min_dist=0.0, metric="cosine", random_state=RANDOM_STATE,
    ).fit_transform(emb.astype(np.float64))


def metrics(labels: np.ndarray, space: np.ndarray, truth: np.ndarray) -> dict:
    """單一分群結果的指標（與 step1c 同口徑）。"""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    sil = np.nan
    if space is not None and n_clusters >= 2 and mask.sum() > n_clusters:
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


def run():
    df, emb = cc.load_data(require_embeddings=True)
    truth_all = df[cc.COL_LABELID].to_numpy()

    # pooled 結果（既有），含每列 topic；並重算一次 pooled UMAP 座標供「限定該 user」算 silhouette
    pooled = pd.read_csv(cc.OUT_DIR / "step1_clusters.csv")
    pooled_topic = pooled["topic"].to_numpy()
    print("重算 pooled UMAP 座標（供限定 user 的 silhouette）...")
    pooled_space = umap_reduce(emb)

    rows = []
    for u in sorted(df[cc.COL_USER].unique()):
        idx = (df[cc.COL_USER] == u).to_numpy()
        truth_u = truth_all[idx]

        # (A) pooled 限定該 user：用 pooled 的 topic 與 pooled UMAP 座標，只取該 user 的列
        m_pool = metrics(pooled_topic[idx], pooled_space[idx], truth_u)
        m_pool.update({"user_id": int(u), "method": "pooled(限定該user)", "n_rows": int(idx.sum())})
        rows.append(m_pool)

        # (B) 該 user 單獨跑：自己的 embedding → 自己的 UMAP + HDBSCAN
        emb_u = emb[idx]
        space_u = umap_reduce(emb_u)
        labels_u = run_hdbscan(space_u)
        m_solo = metrics(labels_u, space_u, truth_u)
        m_solo.update({"user_id": int(u), "method": "per-user(單獨跑)", "n_rows": int(idx.sum())})
        rows.append(m_solo)
        print(f"  user {u}: pooled noise {m_pool['noise_ratio']:.1%} / "
              f"per-user noise {m_solo['noise_ratio']:.1%}")

    out = pd.DataFrame(rows)[
        ["user_id", "method", "n_rows", "n_clusters", "noise_ratio",
         "coverage", "purity", "homogeneity", "NMI", "silhouette"]
    ].sort_values(["user_id", "method"])
    out.to_csv(OUT / "exp_per_user_clustering.csv", index=False, encoding="utf-8-sig")
    return df, out


def fragmentation(df: pd.DataFrame):
    """跨人破碎度：被 >=2 人買的商品，pooled 能否歸同群、per-user 必然拆散。"""
    pooled = pd.read_csv(cc.OUT_DIR / "step1_clusters.csv")
    combo = (df[cc.COL_ITEM].astype(str) + "@" + df[cc.COL_STORE].astype(str)).to_numpy()
    work = pd.DataFrame({
        "combo": combo,
        "user": df[cc.COL_USER].to_numpy(),
        "topic": pooled["topic"].to_numpy(),
        "amt": df[cc.COL_AMT].to_numpy(),
    })
    n_users_per_combo = work.groupby("combo")["user"].nunique()
    multi = n_users_per_combo[n_users_per_combo >= 2].index
    multi_rows = work[work["combo"].isin(multi)]

    # pooled 下：被多人買的商品，其非 noise 列是否落在「單一群」
    def single_cluster(g):
        nz = g[g["topic"] != -1]["topic"]
        return len(set(nz)) == 1 and len(nz) > 0
    pooled_same = multi_rows.groupby("combo").apply(single_cluster, include_groups=False)

    summary = {
        "被>=2人購買的商品種類數": int(len(multi)),
        "佔全部商品種類比例": round(len(multi) / n_users_per_combo.size, 3),
        "這些商品的明細筆數": int(len(multi_rows)),
        "佔總筆數比例": round(len(multi_rows) / len(work), 3),
        "佔總金額比例": round(multi_rows["amt"].sum() / work["amt"].sum(), 3),
        "pooled下_落單一群的比例": round(float(pooled_same.mean()), 3),
        "per-user下_可同群的比例": 0.0,  # 分開跑，跨人必然在不同 clustering，定義上=0
    }
    pd.DataFrame([summary]).to_csv(
        OUT / "exp_per_user_fragmentation.csv", index=False, encoding="utf-8-sig")
    return summary


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

    users = sorted(out["user_id"].unique())
    # 4 格：群數(粒度) 與 homogeneity 才是關鍵；noise/silhouette 補充
    metrics_show = [("n_clusters", "群數 / 商品粒度 (越多=解析越細)", False),
                    ("homogeneity", "homogeneity (群內越純越好)", True),
                    ("silhouette", "silhouette (越高越好)", True),
                    ("noise_ratio", "noise 比例 (需配合群數看)", True)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(users))
    w = 0.36
    for ax, (col, title, unit01) in zip(axes.ravel(), metrics_show):
        pool = [out[(out.user_id == u) & (out.method.str.startswith("pooled"))][col].values[0] for u in users]
        solo = [out[(out.user_id == u) & (out.method.str.startswith("per-user"))][col].values[0] for u in users]
        b1 = ax.bar(x - w / 2, pool, w, label="pooled(限定該user)", color="#5b8def")
        b2 = ax.bar(x + w / 2, solo, w, label="per-user(單獨跑)", color="#e08a4b")
        fmt = "{:.2f}" if unit01 else "{:.0f}"
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        fmt.format(b.get_height()), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels([f"user {u}" for u in users])
        ax.set_title(title)
        if unit01:
            ax.set_ylim(0, 1.08)
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("分開跑分群 vs pooled（逐人對照）", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df, out = run()
    frag = fragmentation(df)
    plot(out, OUT / "exp_per_user_clustering.png")

    print("\n=== 分群品質：pooled(限定該user) vs per-user(單獨跑) ===")
    print(out.to_string(index=False))
    print("\n=== 跨人破碎度 ===")
    for k, v in frag.items():
        print(f"  {k}: {v}")
    print(f"\n已輸出至 {OUT}：exp_per_user_clustering.csv / .png / exp_per_user_fragmentation.csv")


if __name__ == "__main__":
    main()
