#!/usr/bin/env python3
"""S5 scoring: categorical-axis metrics for each prior setting on the natural sample.

Settings: v2 (published, equal prior), equal (re-run control), oos (out-of-sample
priors, halves A/B merged), extreme (0.9/0.1). Also the per-candidate decision
changes relative to v2, on the candidates audited in both runs.
Outputs: work/candidates_prior_<setting>.tsv, s5_prior_metrics.tsv, s5_prior_flips.tsv
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402

RUNS = [("equal_v2_published", ".mfa_seoul_v2"),
        ("equal_rerun", ".mfa_seoul_prior_rev_equal"),
        ("oos_priors", ".mfa_seoul_prior_rev_oos"),
        ("extreme_0.9_0.1", ".mfa_seoul_prior_rev_extreme")]


def metrics_rows(setting, rows, split=None):
    out = []
    groups = [(p, [r for r in rows if r["process"] == p]) for p in RC.PROCS] + [("ALL", rows)]
    if split:
        for h, spk in split.items():
            groups.append((f"ALL_half{h}", [r for r in rows if r["speaker"] in spk]))
    for name, sub in groups:
        if not sub:
            continue
        h = np.array([r["human"] for r in sub]); a = np.array([r["aligner"] for r in sub])
        m = RC.cls_metrics(h, a)
        row = dict(setting=setting, process=name, n=len(sub), negatives=int((h == 0).sum()),
                   agreement=round(m["agree"], 4), aligner_applied=round(m["aligner"], 4),
                   under=int(m["fn"]), over=int(m["fp"]))
        bs = {k: [] for k in ("agree", "sens", "spec", "ba", "mcc")}
        for w in RC.speaker_boot_weights(sub):
            mm = RC.cls_metrics(h, a, w)
            for k in bs:
                bs[k].append(mm[k])
        for k in bs:
            lo, hi, _ = RC.pct_ci(bs[k])
            row[k] = round(m[k], 4)
            row[f"{k}_lo"] = round(lo, 4)
            row[f"{k}_hi"] = round(hi, 4)
        out.append(row)
    return out


def main():
    split = json.loads((RC.REV / "s5_prior_split.json").read_text())
    halves = {"A": set(split["half_A"]), "B": set(split["half_B"])}
    tables = {}
    allm = []
    for setting, run in RUNS:
        rows = RC.build_table(RC.SEOUL / run, quiet=False)
        tables[setting] = rows
        if setting != "equal_v2_published":
            RC.write_tsv(RC.WORK / f"candidates_prior_{setting}.tsv", rows)
        allm += metrics_rows(setting, rows, halves)
    RC.write_tsv(RC.REV / "s5_prior_metrics.tsv", allm)
    keys = ["setting", "process", "n", "negatives", "agreement", "aligner_applied", "under", "over",
            "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi", "ba", "ba_lo", "ba_hi",
            "mcc", "mcc_lo", "mcc_hi"]
    print("\t".join(keys))
    for r in allm:
        if r["process"].startswith("ALL") or True:
            print("\t".join(str(r[k]) for k in keys))

    base = {r["cand_id"]: r for r in tables["equal_v2_published"]}
    flips = []
    for setting, _ in RUNS[1:]:
        common = [r for r in tables[setting] if r["cand_id"] in base]
        for p in RC.PROCS + ["ALL"]:
            sub = [r for r in common if p == "ALL" or r["process"] == p]
            if not sub:
                continue
            to_app = sum(1 for r in sub if base[r["cand_id"]]["aligner"] == 0 and r["aligner"] == 1)
            to_not = sum(1 for r in sub if base[r["cand_id"]]["aligner"] == 1 and r["aligner"] == 0)
            flips.append(dict(setting=setting, process=p, n_common=len(sub),
                              only_in_v2=len([k for k in base if base[k]["process"] == p or p == "ALL"]) - len(sub),
                              changed=to_app + to_not, to_applied=to_app, to_not_applied=to_not,
                              changed_share=round((to_app + to_not) / len(sub), 4)))
    RC.write_tsv(RC.REV / "s5_prior_flips.tsv", flips)
    print()
    for f in flips:
        print(f)


if __name__ == "__main__":
    main()
