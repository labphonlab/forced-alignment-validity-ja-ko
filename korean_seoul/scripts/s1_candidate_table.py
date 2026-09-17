#!/usr/bin/env python3
"""S1: one per-candidate table for the canonical run (.mfa_seoul_v2), and a check
that it reproduces the manuscript numbers using the ORIGINAL statistics functions
(seoul_metrics.boot_ci / seoul_two_axes.boot_median_diff / auc_ci, seed 20260910).

Output: work/candidates_natural.tsv (token-level, git-ignored), s1_verification.tsv
"""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402

_saved = sys.argv
sys.argv = [sys.argv[0]]
import seoul_metrics as SM          # noqa: E402
import seoul_two_axes as S2         # noqa: E402
sys.argv = _saved


def main():
    run = RC.SEOUL / ".mfa_seoul_v2"
    rows = RC.build_table(run, quiet=False)
    RC.write_tsv(RC.WORK / "candidates_natural.tsv", rows)

    # original metric function on original-shaped rows
    orig = [dict(human=r["human"], mfa="applied" if r["aligner"] else "notapplied",
                 speaker=r["speaker"], cat=r["process"]) for r in rows]
    M = SM.metrics(orig)
    new = {}
    new["candidates"] = f"{len(rows):,}"
    new["speakers"] = f"{len({r['speaker'] for r in rows})}"
    new["agreement"] = f"{M['ag']:.1%}"
    new["baseline"] = f"{M['base']:.1%}"
    new["balanced_accuracy"] = f"{M['bal']:.1%}"
    new["MCC"] = f"{M['mcc']:.3f}"
    new["negatives"] = f"{M['fp'] + M['tn']:,}"
    new["under:over"] = f"{M['fn']:,}:{M['fp']:,}"

    def pairs(sub, field):
        return [(r["speaker"], bool(r["correct"]), [r[field]]) for r in sub
                if r[field] is not None]

    J = [r for r in rows if r["has_boundary"]]
    new["joint_n"] = f"{len(J):,}"
    ok = [r["dev_word_ms"] for r in J if r["correct"]]
    ng = [r["dev_word_ms"] for r in J if not r["correct"]]
    lo, hi = S2.boot_median_diff([(s, c, [d[0] / 1000]) for s, c, d in pairs(J, "dev_word_ms")], "x")
    new["word_median_correct_vs_wrong"] = f"{st.median(ok):.2f} vs {st.median(ng):.2f}"
    new["word_diff_CI"] = f"{st.median(ng) - st.median(ok):+.2f} [{lo:+.2f}, {hi:+.2f}]"

    safe = [r for r in rows if r["identity_safe"]]
    new["identity_safe_candidates"] = f"{len(safe):,}"
    SS = [r for r in safe if r["has_boundary"] and r["dev_site_ms"] is not None]
    ok = [r["dev_site_ms"] for r in SS if r["correct"]]
    ng = [r["dev_site_ms"] for r in SS if not r["correct"]]
    lo, hi = S2.boot_median_diff([(s, c, [d[0] / 1000]) for s, c, d in pairs(SS, "dev_site_ms")], "x")
    new["safe_site_median_correct_vs_wrong"] = f"{st.median(ok):.2f} vs {st.median(ng):.2f}"
    new["safe_site_diff_CI"] = f"{st.median(ng) - st.median(ok):+.2f} [{lo:+.2f}, {hi:+.2f}]"

    au, alo, ahi = S2.auc_ci([(r["speaker"], bool(r["correct"]), r["dev_word_ms"]) for r in J])
    new["AUC_full_word"] = f"{au:.3f} [{alo:.3f}, {ahi:.3f}]"
    au, alo, ahi = S2.auc_ci([(r["speaker"], bool(r["correct"]), r["dev_site_ms"]) for r in SS])
    new["AUC_safe_site"] = f"{au:.3f} [{alo:.3f}, {ahi:.3f}]"

    old = {
        "candidates": "5,716", "speakers": "40", "agreement": "57.5%", "baseline": "97.2%",
        "balanced_accuracy": "50.7%", "MCC": "0.005", "negatives": "161",
        "under:over": "2,338:91", "joint_n": "5,538",
        "word_median_correct_vs_wrong": "9.00 vs 9.00",
        "word_diff_CI": "-0.01 [-0.36, +0.45]",
        "identity_safe_candidates": "2,645",
        "safe_site_median_correct_vs_wrong": "9.44 vs 10.00",
        "safe_site_diff_CI": "+0.56 [+0.02, +1.42]",
        "AUC_full_word": "0.498 [0.483, 0.515]",
        "AUC_safe_site": "0.532 [0.508, 0.559]",
    }
    out = []
    print(f"\n{'quantity':<36}{'manuscript':<26}{'recomputed':<26}")
    for k in old:
        match = old[k].replace(" ", "") == new[k].replace(" ", "")
        print(f"{k:<36}{old[k]:<26}{new[k]:<26}{'OK' if match else 'DIFF'}")
        out.append(dict(quantity=k, manuscript=old[k], recomputed=new[k],
                        match="OK" if match else "DIFF"))
    RC.write_tsv(RC.REV / "s1_verification.tsv", out)
    # note on the process-site analysis n
    print(f"\nidentity-safe with process-site deviation: n={len(SS):,} "
          f"(correct {sum(r['correct'] for r in SS):,}, wrong {sum(1 - r['correct'] for r in SS):,})")


if __name__ == "__main__":
    main()
