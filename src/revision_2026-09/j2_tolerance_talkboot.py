# SCOPE: journal-only（査読対応。事後。docs/decisions-log.md 2026-09-16 J1–J6）
"""J2: Table 1 の 10 ms 許容率（12クラス × onset/offset）に講演クラスタブートストラップの区間を付ける。

入力: results/rq1_table.tsv（Table 1 と同じ行: boundary_type ∈ {onset, offset} かつ abs_error_ms あり）
出力: results/revision_2026-09/j2_tolerance_talkboot.tsv（集計値のみ）
2,000 回、seed 20260916、講演を復元抽出、パーセンタイル 95% 区間。比率は講演をまたいで併合（Table 1 と同じ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision_2026-09"
TABLE1 = ["vowel_short", "vowel_long", "stop", "fricative", "affricate", "nasal", "flap",
          "approximant", "moraic_nasal", "moraic_nasal_fused", "geminate_stop", "devoiced_fused"]
B, SEED = 2000, 20260916
THRESHOLDS = (10, 20)  # 10 ms が主。20 ms は参考


def main() -> int:
    d = pd.read_csv(ROOT / "results" / "rq1_table.tsv", sep="\t",
                    usecols=["file_id", "phone_class", "boundary_type", "abs_error_ms"],
                    dtype={"file_id": str, "phone_class": str, "boundary_type": str})
    d = d[d.boundary_type.isin(["onset", "offset"]) & d.abs_error_ms.notna()]
    d = d[d.phone_class.isin(TABLE1)]
    for t in THRESHOLDS:
        d[f"hit{t}"] = (d.abs_error_ms <= t).astype(np.int64)
    g = d.groupby(["file_id", "phone_class", "boundary_type"]).agg(
        n=("abs_error_ms", "size"), **{f"hit{t}": (f"hit{t}", "sum") for t in THRESHOLDS}).reset_index()
    talks = sorted(g.file_id.unique())
    tidx = {t: i for i, t in enumerate(talks)}
    cells = [(c, b) for c in TABLE1 for b in ("onset", "offset")]
    T, C = len(talks), len(cells)
    N = np.zeros((T, C)); H = {t: np.zeros((T, C)) for t in THRESHOLDS}
    for r in g.itertuples(index=False):
        j = cells.index((r.phone_class, r.boundary_type))
        i = tidx[r.file_id]
        N[i, j] = r.n
        for t in THRESHOLDS:
            H[t][i, j] = getattr(r, f"hit{t}")
    rng = np.random.default_rng(SEED)
    W = np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(B)])  # B × T
    WN = W @ N
    rows = []
    for t in THRESHOLDS:
        pt = 100 * H[t].sum(0) / N.sum(0)
        bs = 100 * (W @ H[t]) / WN
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        for j, (c, b) in enumerate(cells):
            rows.append({"threshold_ms": t, "phone_class": c, "boundary": b, "n": int(N[:, j].sum()),
                         "n_talks_with_cell": int((N[:, j] > 0).sum()),
                         "rate_pct": pt[j], "ci_lo": lo[j], "ci_hi": hi[j],
                         "boot_se": np.nanstd(bs[:, j], ddof=1)})
    # 参考: 12クラス併合
    for t in THRESHOLDS:
        tot = 100 * (W @ H[t]).sum(1) / WN.sum(1)
        rows.append({"threshold_ms": t, "phone_class": "ALL_table1", "boundary": "both",
                     "n": int(N.sum()), "n_talks_with_cell": T,
                     "rate_pct": 100 * H[t].sum() / N.sum(),
                     "ci_lo": np.percentile(tot, 2.5), "ci_hi": np.percentile(tot, 97.5),
                     "boot_se": tot.std(ddof=1)})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "j2_tolerance_talkboot.tsv", sep="\t", index=False, float_format="%.3f")
    print(f"talks={T} B={B} seed={SEED}")
    print(out.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
