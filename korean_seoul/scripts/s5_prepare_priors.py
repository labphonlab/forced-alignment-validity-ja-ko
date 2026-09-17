#!/usr/bin/env python3
"""S5: prepare prior-sensitivity re-alignments of the natural sample (.mfa_seoul_v2).

Variants and base dictionary are taken verbatim from .mfa_seoul_v2 (no new G2P).
Only the pronunciation-probability column of the candidate words is changed.

Settings / run dirs (seoul_corpus/):
  .mfa_seoul_prior_rev_equal    control: v2 dictionary unchanged (0.5/0.5), full corpus
  .mfa_seoul_prior_rev_oos      out-of-sample priors: speakers split in halves A/B
                                (random.Random(20260916)); p_cat estimated on one half
                                (candidate pool, seoul_candidates.csv), applied variant
                                prob = p, unapplied = 1 - p, used to align the other half
  .mfa_seoul_prior_rev_extreme  applied 0.9 / unapplied 0.1 for every process, full corpus

MFA 3.1.1 (kalpy 0.6.4, kalpy/fstext/lexicon.py) uses cost = |ln p| per pronunciation
with no renormalisation by the per-word maximum; p < 0.01 is floored to 0.01.

Corpora are hard links to the v2 wav/lab files (no extra disk). Prints aggregates only.
"""
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402

V2 = RC.SEOUL / ".mfa_seoul_v2"


def link_corpus(dst, speakers):
    n = 0
    for spk in speakers:
        (dst / spk).mkdir(parents=True, exist_ok=True)
        for f in (V2 / "corpus" / spk).iterdir():
            t = dst / spk / f.name
            if not t.exists():
                os.link(f, t)
            n += 1
    return n


def write_dict(path, prob_of):
    """prob_of(token, role) -> probability for candidate pronunciations."""
    vp = list(csv.DictReader((V2 / "variant_pronunciations.csv").open(encoding="utf-8")))
    cand = {r["token"] for r in vp}
    lines = []
    with (V2 / "variant_dict.txt").open(encoding="utf-8") as f:
        for line in f:
            if line.split("\t", 1)[0] in cand:
                continue
            lines.append(line.rstrip("\n"))
    probs = defaultdict(dict)   # token -> pron -> prob (max over roles/rows)
    for r in vp:
        for role, pron in (("applied", r["sandhi"]), ("unapplied", r["citation"])):
            p = prob_of(r["change_type"], role)
            probs[r["token"]][pron] = max(p, probs[r["token"]].get(pron, 0.0))
    multi = sum(1 for t in probs if len([r for r in vp if r["token"] == t]) > 1)
    for tok in sorted(probs):
        for pron, p in sorted(probs[tok].items()):
            lines.append(f"{tok}\t{p:.4f}\t0.5\t1.0\t1.0\t{pron}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(cand), multi


def copy_meta(run):
    run.mkdir(parents=True, exist_ok=True)
    for fn in ("pilot_candidates.csv", "variant_pronunciations.csv"):
        shutil.copy2(V2 / fn, run / fn)


def main():
    speakers = sorted(p.name for p in (V2 / "corpus").iterdir() if p.is_dir())
    rng = random.Random(RC.SEED)
    sh = speakers[:]
    rng.shuffle(sh)
    A, B = sorted(sh[:20]), sorted(sh[20:])
    print(f"speakers {len(speakers)}: half A {len(A)}, half B {len(B)}")

    pool = list(csv.DictReader((RC.SEOUL / "seoul_candidates.csv").open(encoding="utf-8")))
    nat = RC.read_tsv(RC.WORK / "candidates_natural.tsv")

    def est(half, src):
        hs = set(half)
        out = {}
        for p in RC.PROCS:
            if src == "pool":
                v = [int(c["applied"]) for c in pool if c["category"] == p and c["file"][:3] in hs]
            else:
                v = [r["human"] for r in nat if r["process"] == p and r["speaker"] in hs]
            out[p] = (sum(v) / len(v), len(v))
        return out
    pA, pB = est(A, "pool"), est(B, "pool")
    sA, sB = est(A, "sample"), est(B, "sample")
    print(f"{'process':<24}{'p(A,pool)':>12}{'n':>7}{'p(B,pool)':>12}{'n':>7}"
          f"{'p(A,sample)':>13}{'p(B,sample)':>13}")
    for p in RC.PROCS:
        print(f"{p:<24}{pA[p][0]:>12.4f}{pA[p][1]:>7}{pB[p][0]:>12.4f}{pB[p][1]:>7}"
              f"{sA[p][0]:>13.4f}{sB[p][0]:>13.4f}")

    # control
    run = RC.SEOUL / ".mfa_seoul_prior_rev_equal"
    copy_meta(run)
    shutil.copy2(V2 / "variant_dict.txt", run / "variant_dict.txt")
    n = link_corpus(run / "seoulprioreq_corpus", speakers)
    print(f"equal: linked {n} files")

    # out-of-sample
    run = RC.SEOUL / ".mfa_seoul_prior_rev_oos"
    copy_meta(run)
    # half A is aligned with priors estimated on half B, and vice versa
    for half, spk, est_on in (("A", A, pB), ("B", B, pA)):
        k, multi = write_dict(run / f"variant_dict_{half}.txt",
                              lambda cat, role, e=est_on: e[cat][0] if role == "applied" else 1 - e[cat][0])
        n = link_corpus(run / f"seoulprioroos{half}_corpus", spk)
        print(f"oos half {half}: {n} files; {k} candidate words ({multi} with >1 variant row)")

    # extreme
    run = RC.SEOUL / ".mfa_seoul_prior_rev_extreme"
    copy_meta(run)
    write_dict(run / "variant_dict.txt", lambda cat, role: 0.9 if role == "applied" else 0.1)
    n = link_corpus(run / "seoulpriorext_corpus", speakers)
    print(f"extreme: linked {n} files")

    meta = dict(seed=RC.SEED, half_A=A, half_B=B,
                p_pool_A={k: v[0] for k, v in pA.items()}, n_pool_A={k: v[1] for k, v in pA.items()},
                p_pool_B={k: v[0] for k, v in pB.items()}, n_pool_B={k: v[1] for k, v in pB.items()},
                p_sample_A={k: v[0] for k, v in sA.items()}, p_sample_B={k: v[0] for k, v in sB.items()})
    (RC.REV / "s5_prior_split.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
