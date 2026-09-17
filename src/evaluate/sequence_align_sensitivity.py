# SCOPE: shared
"""査読対応: sequence_align.py のNW閾値・スコアリング定数を環境変数で上書きして
再実行するラッパー。本体 (sequence_align.py) は変更しない。

環境変数（省略時は元の既定値）:
  SEQALIGN_FAR_APART  - 秒。既定 0.15（150ms）
  SEQALIGN_MATCH      - 既定 2
  SEQALIGN_MISMATCH   - 既定 -1
  SEQALIGN_GAP        - 既定 -2

使い方:
  SEQALIGN_FAR_APART=0.10 python3 src/evaluate/sequence_align_sensitivity.py \
      --reference results/csj_units.tsv --hypothesis results/mfa_csj_units.tsv \
      --mapping csj --out /tmp/sens_far100.tsv
"""
from __future__ import annotations

import os
import sys

import sequence_align as sa

_far = float(os.environ.get("SEQALIGN_FAR_APART", sa.FAR_APART))
_match = float(os.environ.get("SEQALIGN_MATCH", sa.MATCH))
_mismatch = float(os.environ.get("SEQALIGN_MISMATCH", sa.MISMATCH))
_gap = float(os.environ.get("SEQALIGN_GAP", sa.GAP))

sa.FAR_APART = _far
sa.MATCH = _match
sa.MISMATCH = _mismatch
sa.GAP = _gap
sa.FAR_PENALTY = 2 * _gap - 1

print(
    f"[sensitivity] FAR_APART={_far} MATCH={_match} MISMATCH={_mismatch} "
    f"GAP={_gap} FAR_PENALTY={sa.FAR_PENALTY}",
    file=sys.stderr,
)

if __name__ == "__main__":
    raise SystemExit(sa.main())
