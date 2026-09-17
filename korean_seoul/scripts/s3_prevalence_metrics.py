#!/usr/bin/env python3
"""S3: prevalence-robust metrics (sens, spec, BA, MCC) per process and overall with
speaker-cluster bootstrap CIs (B=2000, seed 20260916); plus the overall figures
re-weighted to the process mix of the full candidate pool (69,542 environments).

The natural sample caps each process at 2,000 draws, so its process mix differs
from the pool; within a process the draw is random (natural within-process prevalence).
Weight for a candidate of process p = pool_share(p) / sample_share(p).

Also usable on other run tables: s3_prevalence_metrics.py <table.tsv> <out-prefix> [--no-pool-weight]
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402


def pool_shares():
    c = Counter(r["category"] for r in
                csv.DictReader((RC.SEOUL / "seoul_candidates.csv").open(encoding="utf-8")))
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}, c


def table(rows, weights=None, label_extra=""):
    out = []
    groups = [(p, [i for i, r in enumerate(rows) if r["process"] == p]) for p in RC.PROCS]
    groups.append(("ALL" + label_extra, list(range(len(rows)))))
    for name, idx in groups:
        if not idx:
            continue
        sub = [rows[i] for i in idx]
        h = np.array([r["human"] for r in sub]); a = np.array([r["aligner"] for r in sub])
        w0 = None if weights is None else weights[idx]
        m = RC.cls_metrics(h, a, w0)
        bs = {k: [] for k in ("sens", "spec", "ba", "mcc")}
        for w in RC.speaker_boot_weights(sub):
            ww = w if w0 is None else w * w0
            mm = RC.cls_metrics(h, a, ww)
            for k in bs:
                bs[k].append(mm[k])
        row = dict(process=name, n=len(sub), n_applied=int(h.sum()), n_not_applied=int((1 - h).sum()),
                   speakers=len({r["speaker"] for r in sub}),
                   agreement=round(m["agree"], 4), hand_applied=round(m["hand"], 4),
                   aligner_applied=round(m["aligner"], 4),
                   under=int(((h == 1) & (a == 0)).sum()), over=int(((h == 0) & (a == 1)).sum()))
        for k in bs:
            lo, hi, nv = RC.pct_ci(bs[k])
            row[k] = round(m[k], 4)
            row[f"{k}_lo"] = round(lo, 4)
            row[f"{k}_hi"] = round(hi, 4)
            row[f"{k}_valid_reps"] = nv
        out.append(row)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]) if args else RC.WORK / "candidates_natural.tsv"
    prefix = args[1] if len(args) > 1 else "s3_natural"
    rows = RC.read_tsv(src)
    out = table(rows)
    if "--no-pool-weight" not in sys.argv:
        shares, cnt = pool_shares()
        sc = Counter(r["process"] for r in rows)
        w = np.array([shares[r["process"]] / (sc[r["process"]] / len(rows)) for r in rows])
        allw = table(rows, w, label_extra="_pool_weighted")[-1]
        allw["n"] = f"{len(rows)} (weights: pool mix)"
        out.append(allw)
        # macro average across processes (unweighted mean of per-process metrics)
        per = [r for r in out if r["process"] in RC.PROCS]
        out.append(dict(process="MACRO_mean_of_processes", n=len(rows),
                        **{k: round(float(np.nanmean([r[k] for r in per])), 4)
                           for k in ("sens", "spec", "ba", "mcc")}))
        print("pool process shares:", {k: round(v, 4) for k, v in shares.items()})
    RC.write_tsv(RC.REV / f"{prefix}_metrics.tsv", out)
    keys = ["process", "n", "n_applied", "n_not_applied", "agreement", "under", "over",
            "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi",
            "ba", "ba_lo", "ba_hi", "mcc", "mcc_lo", "mcc_hi", "mcc_valid_reps"]
    print("\t".join(keys))
    for r in out:
        print("\t".join(str(r.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
