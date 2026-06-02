"""成員 C 共用工具：載入成員 A 的資料與 embedding、欄位正規化、共用設定。

所有 member_c/ 下的腳本都從這裡載入資料，確保 CSV 與 embedding 逐列對齊。
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Windows 主控台預設 cp950，印中文會 UnicodeEncodeError；統一改成 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 路徑 ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # 專案根目錄
CSV_PATH = ROOT / "all_user_6_label.csv"               # 成員 A 清洗+標註後的明細
EMB_PATH = ROOT / "receipt_embeddings.npy"             # 成員 A 的 BERT [CLS] embedding
OUT_DIR = Path(__file__).resolve().parent / "outputs"  # 所有產出（圖、csv）
OUT_DIR.mkdir(exist_ok=True)

# ---- 欄位名稱（成員 A 的原始欄名，含中文）-------------------------------
COL_DATE = "發票日期"
COL_QTY = "消費明細_數量"
COL_UNIT = "消費明細_單價"
COL_AMT = "消費明細_金額"
COL_STORE = "store_clean"
COL_ITEM = "item_clean"
COL_LABEL = "label"
COL_USER = "user_id"
COL_BUCKET = "price_bucket"
COL_LABELID = "label_id"

# 通膨拆解的篩選門檻（重複購買群至少出現幾次、跨幾個月）
MIN_COUNT = 5      # 企劃書：每群至少 5~10 次
MIN_MONTHS = 2     # 至少跨 2 個月才能比較單價變化


def load_data(require_embeddings: bool = True):
    """載入明細 DataFrame 與（可選）embedding 矩陣，並驗證對齊。

    Returns
    -------
    df : pd.DataFrame  已加上 'month'（Period[M]）與 'date'（Timestamp）欄
    emb : np.ndarray | None  形狀 (N, 768)，與 df 逐列對齊；require_embeddings=False 時為 None
    """
    df = pd.read_csv(CSV_PATH)
    # 第一欄是匿名 index，丟掉避免干擾
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    df["date"] = pd.to_datetime(df[COL_DATE])
    df["month"] = df["date"].dt.to_period("M")

    # 數量缺失或 <=0 視為 1（成員 A 已濾掉金額<=0，但數量偶有空值）
    df[COL_QTY] = pd.to_numeric(df[COL_QTY], errors="coerce").fillna(1.0)
    df.loc[df[COL_QTY] <= 0, COL_QTY] = 1.0
    df[COL_AMT] = pd.to_numeric(df[COL_AMT], errors="coerce")

    emb = None
    if require_embeddings:
        emb = np.load(EMB_PATH)
        assert len(df) == len(emb), (
            f"CSV ({len(df)}) 與 embedding ({len(emb)}) 列數不一致，無法對齊！"
        )
    return df, emb


def group_key_exact(df: pd.DataFrame) -> pd.Series:
    """精確分群鍵：同一 (品名, 店名) 視為同一商品。

    這是通膨拆解最可靠的分群方式（避免語意分群把同商品因價格區間拆開）。
    BERTopic/HDBSCAN 的語意群則作為對照與「自動探索」用途。
    """
    return df[COL_ITEM].astype(str) + " @ " + df[COL_STORE].astype(str)


def monthly_unit_price(df: pd.DataFrame, group_col: str, by_user: bool = True):
    """對每個商品群、每個月計算加權平均單價 = sum(金額)/sum(數量)。

    用加權平均而非單價欄平均，避免一次買多件時被低估/高估。
    """
    keys = [group_col, "month"]
    if by_user:
        keys = [COL_USER] + keys
    g = df.groupby(keys, observed=True).agg(
        amt=(COL_AMT, "sum"),
        qty=(COL_QTY, "sum"),
        n=(COL_AMT, "size"),
    )
    g["unit_price"] = g["amt"] / g["qty"]
    return g.reset_index()
