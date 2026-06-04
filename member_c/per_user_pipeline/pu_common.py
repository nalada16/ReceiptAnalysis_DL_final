"""per-user 管線共用設定：把上層 member_c 接進 path，並把輸出導向本資料夾。

★ 本資料夾(per_user_pipeline/)是「三人分開分群 → 分開增量分析」的獨立對照管線，
   與 member_c 主流程(pooled)完全分開，所有產出寫到 per_user_pipeline/outputs/。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEMBER_C = HERE.parent
if str(MEMBER_C) not in sys.path:
    sys.path.insert(0, str(MEMBER_C))  # 讓本資料夾能 import c_common / step3c 等

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
