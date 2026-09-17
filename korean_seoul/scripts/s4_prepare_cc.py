#!/usr/bin/env python3
"""S4: build the case-control (negative-enriched) alignment corpus.

Pool: seoul_corpus/seoul_candidates.csv (69,542 candidate environments).
Natural sample exclusion: the natural (.mfa_seoul_v2) draw is replayed exactly as
prepare_mfa.py does it (random.Random(20260909), <=2000 per category, categories
in sorted order), each drawn candidate is mapped to its utterance with the same
rule, and the result is checked against .mfa_seoul_v2/pilot_candidates.csv.

Draw (seed 20260916): per process, up to 150 human-not-applied candidates (all if
fewer) and the same number of human-applied candidates; one candidate per utterance
(across the whole case-control sample); utterances of the natural sample excluded.
Negatives are drawn first (all processes), then positives.

Audio/lab writing replicates prepare_mfa.py (16 kHz mono, 0.15 s pad, linear
resampling, uid = <file>_<k:05d>). Output run dir: seoul_corpus/.mfa_seoul_cc_rev
Prints aggregates only.
"""
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402

CORPUS = Path(os.environ.get("SEOUL_CORPUS_DIR", "seoul_corpus/raw")).expanduser()
RUN = RC.SEOUL / ".mfa_seoul_cc_rev"
CORPUS_DIR = RUN / "seoulccrev_corpus"
N_NEG = 150
PAD = 0.15
SR = 16000

# replicate prepare_mfa.read_tiers (cannot import: argparse runs at import time)
import re  # noqa: E402


def read_tiers(path):
    t = path.read_bytes().decode("utf-16")
    out = {}
    for b in re.split(r"item\s*\[\d+\]:", t)[1:]:
        m = re.search(r'name\s*=\s*"([^"]*)"', b)
        if not m:
            continue
        ivs = re.findall(
            r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"', b)
        out.setdefault(m.group(1), []).append(
            [(float(a), float(c), d) for a, c, d in ivs])
    return out


_utts = {}


def utt_of(c):
    fname = c["file"]
    if fname not in _utts:
        tg = CORPUS / f"{fname}.TextGrid"
        fl = CORPUS / f"{fname}.flac"
        if not tg.exists() or not fl.exists():
            _utts[fname] = None
        else:
            _utts[fname] = read_tiers(tg).get("utt.ortho.", [[]])[0]
    utts = _utts[fname]
    if utts is None:
        return None
    ws, we = float(c["start"]), float(c["end"])
    u = next((u for u in utts if u[0] <= ws and we <= u[1] and u[2].strip()
              and not u[2].startswith("<")), None)
    if u is None:
        return None
    return (fname, u[0], u[1], u[2])


def replay_natural(cands):
    by_cat = defaultdict(list)
    for c in cands:
        by_cat[c["category"]].append(c)
    rng = random.Random(20260909)
    picked = []
    for cat, lst in sorted(by_cat.items()):
        picked += lst if len(lst) <= 2000 else rng.sample(lst, 2000)
    return picked


def main():
    cands = list(csv.DictReader((RC.SEOUL / "seoul_candidates.csv").open(encoding="utf-8")))
    print(f"pool candidates: {len(cands):,}")
    for i, c in enumerate(cands):
        c["_i"] = i

    # ---- replay the natural sample and verify against .mfa_seoul_v2 ----
    nat = replay_natural(cands)
    by_file = defaultdict(list)
    for c in nat:
        by_file[c["file"]].append(c)
    written = []
    nat_utts = set()
    for fname, items in sorted(by_file.items()):
        for k, c in enumerate(items):
            u = utt_of(c)
            if u is None:
                continue
            s, e = max(0.0, u[1] - PAD), u[2] + PAD
            # prepare_mfa also drops segments shorter than 0.1 s (never binding here)
            written.append((f"{fname}_{k:05d}", c["ortho"], c["category"], c["applied"]))
            nat_utts.add(u[:3])
    v2 = list(csv.DictReader((RC.SEOUL / ".mfa_seoul_v2/pilot_candidates.csv").open(encoding="utf-8")))
    ref = [(r["utterance_id"], r["token"], r["change_type"], r["human_applied"]) for r in v2]
    same = written == ref
    print(f"natural replay: {len(written):,} written vs {len(ref):,} in .mfa_seoul_v2 -> "
          f"{'IDENTICAL' if same else 'MISMATCH'}; distinct natural utterances {len(nat_utts):,}")
    if not same:
        # tolerate only if the set of (uid, cat, applied) matches
        sys.exit("natural replay mismatch; aborting")

    # ---- eligible pool ----
    elig = []
    for c in cands:
        u = utt_of(c)
        if u is None:
            continue
        key = u[:3]
        if key in nat_utts:
            continue
        c["_u"] = key
        c["_utext"] = u[3]
        elig.append(c)
    ec = Counter((c["category"], c["applied"]) for c in elig)
    print(f"eligible (utterance found, not in natural sample): {len(elig):,}")
    for p in RC.PROCS:
        print(f"  {p:<24} applied {ec[(p, '1')]:>6,}  not applied {ec[(p, '0')]:>4,}")

    rng = random.Random(RC.SEED)
    used = set()
    chosen = []
    neg_n = {}
    for p in RC.PROCS:
        pool = [c for c in elig if c["category"] == p and c["applied"] == "0"]
        rng.shuffle(pool)
        k = 0
        for c in pool:
            if k >= N_NEG:
                break
            if c["_u"] in used:
                continue
            used.add(c["_u"]); chosen.append(c); k += 1
        neg_n[p] = k
    for p in RC.PROCS:
        pool = [c for c in elig if c["category"] == p and c["applied"] == "1"]
        rng.shuffle(pool)
        k = 0
        for c in pool:
            if k >= neg_n[p]:
                break
            if c["_u"] in used:
                continue
            used.add(c["_u"]); chosen.append(c); k += 1
    cc = Counter((c["category"], c["applied"]) for c in chosen)
    print(f"\ncase-control draw: {len(chosen):,} candidates, {len(used):,} utterances, "
          f"{len({c['file'][:3] for c in chosen})} speakers")
    for p in RC.PROCS:
        print(f"  {p:<24} applied {cc[(p, '1')]:>4}  not applied {cc[(p, '0')]:>4}")

    # ---- write corpus exactly like prepare_mfa.py ----
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    byf = defaultdict(list)
    for c in sorted(chosen, key=lambda c: c["_i"]):   # pool order within file, like prepare_mfa
        byf[c["file"]].append(c)
    rows, miss = [], 0
    for fname, items in sorted(byf.items()):
        flac = CORPUS / f"{fname}.flac"
        audio, sr0 = sf.read(str(flac), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        spk = fname[:3]
        (CORPUS_DIR / spk).mkdir(exist_ok=True)
        for k, c in enumerate(items):
            _, u0, u1 = c["_u"]
            text = c["_utext"]
            s, e = max(0.0, u0 - PAD), min(len(audio) / sr0, u1 + PAD)
            seg = audio[int(s * sr0):int(e * sr0)]
            if len(seg) < sr0 * 0.1:
                miss += 1; continue
            idx = np.linspace(0, len(seg) - 1, int(len(seg) * SR / sr0))
            seg16 = np.interp(idx, np.arange(len(seg)), seg).astype("float32")
            uid = f"{fname}_{k:05d}"
            sf.write(CORPUS_DIR / spk / f"{uid}.wav", seg16, SR)
            (CORPUS_DIR / spk / f"{uid}.lab").write_text(text.strip(), encoding="utf-8")
            rows.append(dict(utterance_id=uid, speaker_id=spk, token=c["ortho"],
                             syllable_boundary=c["syl_index"], change_type=c["category"],
                             transcript=text.strip(),
                             wav_path=str(CORPUS_DIR / spk / f"{uid}.wav"),
                             duration_sec=f"{e-s:.4f}", human_applied=c["applied"],
                             prono=c["prono"]))
    with (RUN / "pilot_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwritten {len(rows):,} utterances to {CORPUS_DIR.relative_to(RC.PROJ)} (short-segment drops {miss})")


if __name__ == "__main__":
    main()
