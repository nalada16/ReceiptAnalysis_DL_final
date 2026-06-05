"""price-feature 管線共用設定：把上層 member_c 接進 path，輸出導向本資料夾。

★ 本資料夾是「BERT-UMAP 後 concat 價格 feature」的對照實驗，與主流程完全分開，
   所有產出寫到 price_feature_pipeline/outputs/。
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEMBER_C = HERE.parent
if str(MEMBER_C) not in sys.path:
    sys.path.insert(0, str(MEMBER_C))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
