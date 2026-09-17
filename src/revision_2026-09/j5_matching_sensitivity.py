# SCOPE: journal-only（査読対応。事後。docs/decisions-log.md 2026-09-16 J1–J6）
"""J5: 対応付けの時間近接しきい値（FAR_APART）を 0.10 / 0.15（既定の再現）/ 0.20 s に振り、
クラス別 10 ms 許容率を比べる。本体 sequence_align.py は変更せず、
sequence_align_sensitivity.py を環境変数つきで実行するだけ。

  python3 src/revision_2026-09/j5_matching_sensitivity.py --work <dir outside repo>

トークン単位の中間ファイル（177講演に絞った units とアラインメント出力）は --work に置く。
出力（集計値のみ）: results/revision_2026-09/j5_matching_sensitivity.tsv
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "revision_2026-09"
CLASSES = ["devoiced_fused", "vowel_short", "stop"]
TABLE1 = ["vowel_short", "vowel_long", "stop", "fricative", "affricate", "nasal", "flap",
          "approximant", "moraic_nasal", "moraic_nasal_fused", "geminate_stop", "devoiced_fused"]
CONFIGS = {"far100": {"SEQALIGN_FAR_APART": "0.10"},
           "far150_default": {},
           "far200": {"SEQALIGN_FAR_APART": "0.20"}}
csv.field_size_limit(sys.maxsize)


def filter_main(src: Path, dst: Path) -> int:
    """style 列が aps/sps の行だけを写す（中身は表示しない）。"""
    n = 0
    with src.open(encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        header = fi.readline()
        fo.write(header)
        idx = header.rstrip("\n").split("\t").index("style")
        for line in fi:
            if line.split("\t")[idx] in ("aps", "sps"):
                fo.write(line)
                n += 1
    return n


def summarize(path: Path) -> tuple[Counter, dict]:
    types = Counter()
    cells: dict = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            types[r["error_type"]] += 1
            if r["error_type"] not in ("displacement", "substitution"):
                continue
            cls = r["phone_class"]
            for bd, col in (("onset", "onset_error_ms"), ("offset", "offset_error_ms")):
                if not r[col]:
                    continue
                hit = abs(float(r[col])) <= 10
                for key in ((cls, bd), ("ALL_table1" if cls in TABLE1 else "_other", "both")):
                    c = cells.setdefault(key, [0, 0])
                    c[0] += hit
                    c[1] += 1
    return types, cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args(argv)
    w = args.work
    w.mkdir(parents=True, exist_ok=True)
    ref, hyp = w / "csj_units_main.tsv", w / "mfa_csj_units_main.tsv"
    if not ref.exists():
        print("ref units (177 talks):", filter_main(RES / "csj_units.tsv", ref))
    if not hyp.exists():
        print("hyp units (177 talks):", filter_main(RES / "mfa_csj_units.tsv", hyp))

    procs = {}
    for name, env in CONFIGS.items():
        out = w / f"aln_{name}.tsv"
        if out.exists():
            continue
        e = {**os.environ, **env}
        cmd = [sys.executable, str(ROOT / "src/evaluate/sequence_align_sensitivity.py"),
               "--reference", str(ref), "--hypothesis", str(hyp), "--mapping", "csj", "--out", str(out)]
        log = open(w / f"aln_{name}.log", "w")
        procs[name] = (subprocess.Popen(cmd, env=e, cwd=ROOT / "src/evaluate", stdout=log, stderr=log), time.time())
    for name, (p, t0) in procs.items():
        rc = p.wait()
        print(f"{name}: rc={rc} {time.time() - t0:.0f}s")
        if rc:
            return rc

    rows = []
    for name in CONFIGS:
        types, cells = summarize(w / f"aln_{name}.tsv")
        tot = sum(types.values())
        base = {"config": name, "far_apart_s": CONFIGS[name].get("SEQALIGN_FAR_APART", "0.15"),
                "n_pairs": tot, **{f"pct_{k}": 100 * types[k] / tot for k in
                                   ("displacement", "substitution", "omission", "insertion")}}
        for key in [(c, b) for c in CLASSES for b in ("onset", "offset")] + [("ALL_table1", "both")]:
            h, n = cells.get(key, [0, 0])
            rows.append({**base, "phone_class": key[0], "boundary": key[1], "n_boundaries": n,
                         "within10_pct": 100 * h / n if n else float("nan")})
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with (OUT / "j5_matching_sensitivity.tsv").open("w", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"{r['config']:<15} {r['phone_class']:<15} {r['boundary']:<7} n={r['n_boundaries']:>8} "
              f"<=10ms {r['within10_pct']:6.2f}  (types d/s/o/i {r['pct_displacement']:.2f}/"
              f"{r['pct_substitution']:.2f}/{r['pct_omission']:.2f}/{r['pct_insertion']:.2f}, n={r['n_pairs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
