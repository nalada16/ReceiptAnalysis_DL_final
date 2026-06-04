# 系統架構

投影片用圖：`outputs/system_architecture.png`（由 `make_architecture.py` 生成，可改座標/文字後重跑）。
下方為可編輯的 Mermaid 版（GitHub、VS Code、draw.io、Typora 等可直接渲染）。

```mermaid
flowchart TB
    DATA["原始電子發票 CSV<br/>2613 筆 · 3 人 · 2025-09~2026-05"]

    subgraph A["成員 A｜資料工程 + Task1 分類"]
        direction LR
        A1["去識別化 + 欄位清洗"] --> A2["文字正規化<br/>全形→半形 / 小寫 / store"]
        A2 --> A3["LLM 預標註 + 人工審核"]
        A3 --> A4["BERT 分類 fine-tune<br/>ckiplab/bert-base-chinese"]
    end

    OUTL["輸出①　6 類消費標籤"]
    OUTE["輸出②　BERT [CLS] embedding 768d"]

    subgraph B["成員 B｜Task2 L1/L2"]
        direction TB
        B1["每日消費矩陣 + mask"]
        B1 --> B2["LSTM 短期預測 (L1)"]
        B1 --> B3["LSTM-AE 異常偵測 (L2)"]
    end

    subgraph C["成員 C｜Task2 L3 + 整合"]
        direction TB
        C1["BERTopic + UMAP + HDBSCAN 商品分群"]
        C1 --> C2["重複購買群 + 單價時序"]
        C2 --> C3["量 / 價效應 通膨拆解 (L3)"]
    end

    INT["整合報告 · 系統架構圖 · 簡報（全員）"]

    DATA --> A
    A4 --> OUTL
    A4 --> OUTE
    OUTL --> B1
    OUTE --> C1
    OUTE -. 語意特徵（選配） .-> B2
    B2 --> INT
    B3 --> INT
    C2 --> INT
    C3 --> INT

    classDef a fill:#5b8def,stroke:#3b6fd0,color:#fff;
    classDef b fill:#3fae6e,stroke:#2e8c57,color:#fff;
    classDef c fill:#e08a4b,stroke:#c5733a,color:#fff;
    classDef o fill:#3b6fd0,stroke:#2c54a0,color:#fff;
    classDef d fill:#9e9e9e,stroke:#7d7d7d,color:#fff;
    classDef i fill:#9b59b6,stroke:#7d3c98,color:#fff;
    class A1,A2,A3,A4 a;
    class B1,B2,B3 b;
    class C1,C2,C3 c;
    class OUTL,OUTE o;
    class DATA d;
    class INT i;
```

## 重點說明（口頭報告用）

- **資料流主軸**：CSV →（A）清洗/標註/BERT 分類 → 產出**標籤**與**embedding** →
  （B）標籤彙整成每日矩陣做預測/異常 ＋（C）embedding 做商品分群與通膨拆解 → 全員整合。
- **成員 C 的關鍵接點**：直接複用 A 的 `receipt_embeddings.npy`，**不重跑 BERT**。
- **跨模組亮點**：embedding 也可作為 B 的 LSTM「語意輔助特徵」（虛線、選配），體現
  Task1 中間層向量被下游重用的設計（企劃書第二章核心設計）。
