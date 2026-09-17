#!/usr/bin/env python3
"""S6 support: is the pronounced-form tier (pWord.prono., romanized) consistent with
the hand-corrected phoneme tier? Aggregate match rates only (no content printed).

For every candidate environment in seoul_candidates.csv (and for the natural-sample
subset), the phoneme-tier labels whose midpoints fall inside the word interval are
compared with the phoneme sequence of the romanized pronounced form (and, for
reference, of the romanized orthographic form).
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402

CORPUS = Path(os.environ.get("SEOUL_CORPUS_DIR", "seoul_corpus/raw")).expanduser()
TOK = re.compile(r"[A-Za-z][A-Za-z0-9]")
# romanized tiers keep spelling distinctions the phoneme tier merges (Yun et al. 2015 §2.3)
NORM = {"EE": "ee", "YE": "ye", "wE": "we", "WE": "we"}


def norm(seq):
    return [NORM.get(x, x) for x in seq]


def read_phonemes(path):
    t = path.read_bytes().decode("utf-16")
    for b in re.split(r"item\s*\[\d+\]:", t)[1:]:
        m = re.search(r'name\s*=\s*"([^"]*)"', b)
        if m and m.group(1) == "phoneme":
            return [(float(a), float(c), d) for a, c, d in re.findall(
                r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', b)]
    return []


def main():
    cands = list(csv.DictReader((RC.SEOUL / "seoul_candidates.csv").open(encoding="utf-8")))
    nat_keys = set()
    # natural-sample candidates, matched on (file, word start) via the replay in s4
    import s4_prepare_cc as S4
    for c in S4.replay_natural(cands):
        nat_keys.add((c["file"], c["start"], c["category"], c["syl_index"]))
    cache = {}
    res = defaultdict(Counter)
    subs = defaultdict(Counter)
    seen = set()
    for c in cands:
        key = (c["file"], c["start"], c["end"])
        f = c["file"]
        if f not in cache:
            cache[f] = read_phonemes(CORPUS / f"{f}.TextGrid")
        s, e = float(c["start"]), float(c["end"])
        ph = [p for a, b, p in cache[f] if s - 1e-6 <= (a + b) / 2 <= e + 1e-6
              and not p.startswith("<") and p.strip()]
        pr = TOK.findall(c["prono_roman"])
        orr = TOK.findall(c["ortho_roman"])
        groups = ["all_candidates"]
        if (c["file"], c["start"], c["category"], c["syl_index"]) in nat_keys:
            groups.append("natural_sample_candidates")
        if key not in seen:
            groups.append("distinct_word_tokens")
            seen.add(key)
        for g in groups:
            res[g]["n"] += 1
            res[g]["phoneme==prono"] += ph == pr
            res[g]["phoneme==ortho"] += ph == orr
            res[g]["prono==ortho"] += pr == orr
            res[g]["len_equal_prono"] += len(ph) == len(pr)
            res[g]["phoneme==prono_norm"] += ph == norm(pr)
            if len(ph) == len(pr) and ph != norm(pr):
                for x, y in zip(ph, norm(pr)):
                    if x != y:
                        subs[g][(y, x)] += 1
            if ph != pr:
                res[g][f"mismatch_{c['category']}"] += 1
    out = []
    for g, cnt in res.items():
        n = cnt["n"]
        row = dict(group=g, n=n,
                   phoneme_equals_prono=round(cnt["phoneme==prono"] / n, 4),
                   phoneme_equals_ortho=round(cnt["phoneme==ortho"] / n, 4),
                   phoneme_equals_prono_vowelnorm=round(cnt["phoneme==prono_norm"] / n, 4),
                   top_substitutions_prono_to_phoneme=";".join(f"{a}>{b}:{k}" for (a, b), k in subs[g].most_common(8)),
                   prono_equals_ortho=round(cnt["prono==ortho"] / n, 4),
                   same_length_phoneme_prono=round(cnt["len_equal_prono"] / n, 4),
                   raw_mismatches_by_process_before_vowelnorm=";".join(f"{k[9:]}={v}" for k, v in sorted(cnt.items())
                                                  if k.startswith("mismatch_")))
        out.append(row)
        print(row)
    RC.write_tsv(RC.REV / "s6_tier_consistency.tsv", out)


if __name__ == "__main__":
    main()
