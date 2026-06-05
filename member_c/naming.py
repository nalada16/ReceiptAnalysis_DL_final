"""人工命名（human-in-the-loop）共用工具。

命名表放在 member_c/ 根目錄（方便用 Excel 開）：
  cluster_names.csv          細群命名（key=topic）
  supertype_names_K{K}.csv   超類型命名（key=super）

每張表都有 custom_name 欄供你填；空白則下游用自動標籤。
重新產生命名表時會 merge：保留你已填的 custom_name，只補新出現的群。
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
FINE_NAMES = HERE / "cluster_names.csv"


def super_names_path(K: int) -> Path:
    return HERE / f"supertype_names_K{K}.csv"


def merge_template(path: Path, new_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """寫出/更新命名表：保留既有 custom_name，補上新 key，標記已消失的群。"""
    new_df = new_df.copy()
    if path.exists():
        old = pd.read_csv(path)
        if "custom_name" in old.columns:
            new_df = new_df.merge(old[[key, "custom_name"]], on=key, how="left")
        else:
            new_df["custom_name"] = ""
    else:
        new_df["custom_name"] = ""
    new_df["custom_name"] = new_df["custom_name"].fillna("")
    # custom_name 放最前面，方便填寫
    cols = [key, "custom_name"] + [c for c in new_df.columns if c not in (key, "custom_name")]
    new_df = new_df[cols]
    new_df.to_csv(path, index=False, encoding="utf-8-sig")
    return new_df


def load_names(path: Path, key: str) -> dict:
    """讀命名表，回傳 {key: custom_name}（只含有填的）。檔案不存在回空 dict。"""
    if not path.exists():
        return {}
    d = pd.read_csv(path)
    if "custom_name" not in d.columns:
        return {}
    out = {}
    for _, r in d.iterrows():
        name = str(r["custom_name"]).strip()
        if name and name.lower() != "nan":
            out[r[key]] = name
    return out
