"""步驟 1：商品語意分群（BERTopic + UMAP + HDBSCAN）。

輸入：成員 A 的 receipt_embeddings.npy（不需重跑 BERT）。
輸出：
  outputs/step1_clusters.csv      每筆明細的分群標籤
  outputs/step1_topics.csv        每群的主題詞與代表品項
  outputs/step1_umap_scatter.png  2D 視覺化

設計重點：
  * 直接餵現成 embedding 給 BERTopic（embedding_model=None），執行時不下載任何模型。
  * BERTopic 不可用時，自動退回「UMAP + HDBSCAN + 每群最高頻品名」的純手刻流程。
  * 中文 c-TF-IDF 主題詞需要斷詞，優先用 jieba；沒有 jieba 則退回字元 n-gram。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import c_common as cc

# ---- 可調超參數 ----------------------------------------------------------
UMAP_N_NEIGHBORS = 15      # 越小越看局部結構；資料少可調小
UMAP_N_COMPONENTS = 5      # 分群用的降維維度
HDBSCAN_MIN_CLUSTER = 10   # 一個商品群最少幾筆（對應企劃書 5~10 次）
HDBSCAN_MIN_SAMPLES = 5    # 越大噪音越多、群越保守
RANDOM_STATE = 42


def _build_documents(df: pd.DataFrame) -> list[str]:
    """每筆明細的文字表示：品名 + 店名（給 c-TF-IDF 抽主題詞用）。"""
    return (df[cc.COL_ITEM].astype(str) + " " + df[cc.COL_STORE].astype(str)).tolist()


def _make_vectorizer():
    """中文友善的 CountVectorizer：有 jieba 用 jieba，否則用字元 2-3 gram。"""
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        import jieba

        def tok(text: str):
            return [w for w in jieba.cut(text) if w.strip()]

        return CountVectorizer(tokenizer=tok, token_pattern=None, min_df=2)
    except Exception:
        # 退路：字元 n-gram，仍能抓到「燕麥」「奶茶」這類子字串
        return CountVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)


def run_bertopic(df: pd.DataFrame, emb: np.ndarray):
    """主路徑：BERTopic（內部用 UMAP+HDBSCAN），餵現成 embedding。"""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=None,            # 關鍵：用現成 embedding，不載模型
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=_make_vectorizer(),
        calculate_probabilities=False,
        verbose=True,
    )
    docs = _build_documents(df)
    topics, _ = topic_model.fit_transform(docs, embeddings=emb.astype(np.float64))
    info = topic_model.get_topic_info()
    return np.asarray(topics), info, topic_model


def run_umap_hdbscan(df: pd.DataFrame, emb: np.ndarray):
    """Fallback：純 UMAP + HDBSCAN，主題詞用每群最高頻品名。"""
    from umap import UMAP
    from hdbscan import HDBSCAN

    reducer = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    red = reducer.fit_transform(emb.astype(np.float64))
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(red)

    # 每群代表品項 = 群內最高頻 item_clean
    tmp = df.copy()
    tmp["topic"] = labels
    rows = []
    for t, sub in tmp.groupby("topic"):
        top_items = sub[cc.COL_ITEM].value_counts().head(5).index.tolist()
        rows.append({"Topic": t, "Count": len(sub), "Representation": ", ".join(top_items)})
    info = pd.DataFrame(rows).sort_values("Count", ascending=False)
    return labels, info, None


def plot_scatter(emb: np.ndarray, labels: np.ndarray, path):
    """把 embedding 降到 2D 畫散佈圖（噪音點以灰色 x 顯示）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    xy = UMAP(n_components=2, metric="cosine", random_state=RANDOM_STATE).fit_transform(
        emb.astype(np.float64)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    noise = labels == -1
    ax.scatter(xy[noise, 0], xy[noise, 1], c="lightgrey", s=6, marker="x", label="noise")
    sc = ax.scatter(xy[~noise, 0], xy[~noise, 1], c=labels[~noise], s=8, cmap="tab20")
    ax.set_title("Receipt item clusters (UMAP 2D)")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="cluster id")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    df, emb = cc.load_data(require_embeddings=True)
    print(f"載入 {len(df)} 筆明細，embedding {emb.shape}")

    try:
        labels, info, _ = run_bertopic(df, emb)
        engine = "BERTopic"
    except Exception as e:
        print(f"[警告] BERTopic 不可用（{type(e).__name__}: {e}），退回 UMAP+HDBSCAN")
        labels, info, _ = run_umap_hdbscan(df, emb)
        engine = "UMAP+HDBSCAN"

    n_clusters = len({l for l in labels if l != -1})
    noise_ratio = float(np.mean(labels == -1))
    print(f"[{engine}] 群數={n_clusters}  noise 比例={noise_ratio:.1%}")

    df_out = df.copy()
    df_out["topic"] = labels

    # 中文 c-TF-IDF 關鍵詞對短品名效果差，額外附「每群最高頻品名」當人類可讀標籤。
    top_items = (
        df_out.groupby("topic")[cc.COL_ITEM]
        .apply(lambda s: " | ".join(s.value_counts().head(5).index.astype(str)))
        .rename("top_items")
    )
    info = info.merge(top_items, left_on="Topic", right_index=True, how="left")

    df_out.to_csv(cc.OUT_DIR / "step1_clusters.csv", index=False, encoding="utf-8-sig")
    info.to_csv(cc.OUT_DIR / "step1_topics.csv", index=False, encoding="utf-8-sig")
    plot_scatter(emb, labels, cc.OUT_DIR / "step1_umap_scatter.png")
    print("已輸出 step1_clusters.csv / step1_topics.csv / step1_umap_scatter.png")


if __name__ == "__main__":
    main()
