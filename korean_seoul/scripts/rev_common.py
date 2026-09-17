#!/usr/bin/env python3
"""Shared helpers for the 2026-09 revision reanalysis (Seoul Corpus).

Reuses the canonical pipeline functions without changing them:
  seoul_compare.load()      categorical axis (with the contrast audit)
  seoul_boundary.measure()  temporal axis (class-aligned boundary deviations)

LICENSING: nothing here prints corpus content. Token-level tables are written
only under WORK (git-ignored). Word types are stored as salted hashes.
"""
import csv
import hashlib
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path(os.environ.get("KOREAN_SOUND_CHANGE_ROOT", ".")).expanduser().resolve()
PIPE = PROJ / "scripts/pilot_variant_competition"
SEOUL = PROJ / "seoul_corpus"
REV = PROJ / "analysis/revision_2026-09/seoul"
WORK = REV / "work"
SEED = 20260916
B = 2000
SALT = "ksc-rev-2026-09"

# The canonical modules read sys.argv at import time; import them with a clean argv.
_saved = sys.argv
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(PIPE))
import seoul_compare as SC      # noqa: E402
import seoul_boundary as SB     # noqa: E402
from phone_classes import IPA   # noqa: E402
sys.argv = _saved

PROCS = ["aspiration", "liaison", "liquidization", "nasalization_liquid",
         "nasalization_obstruent", "tensification"]


def wtype(token):
    return hashlib.sha1((SALT + token).encode("utf-8")).hexdigest()[:12]


def _cls(seq):
    return [IPA.get(x) for x in str(seq).split()]


def build_table(run_dir: Path, quiet=True):
    """One row per audited candidate in a run directory (seoul_corpus/<run>)."""
    SC.RUN = SB.RUN = run_dir
    rows = SC.load()
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        meas = {m["utt"]: m for m in SB.measure("aligned")}
    vp = {v["token"]: v for v in
          csv.DictReader((run_dir / "variant_pronunciations.csv").open(encoding="utf-8"))}
    out = []
    for r in rows:
        v = vp.get(r["token"])
        safe = (v is not None and None not in _cls(v["citation"])
                and _cls(v["citation"]) == _cls(v["sandhi"]))
        m = meas.get(r["utt"])
        aligner = 1 if r["mfa"] == "applied" else 0
        out.append(dict(
            cand_id=r["utt"], process=r["cat"], speaker=r["speaker"],
            word_type=wtype(r["token"]), human=int(r["human"]), aligner=aligner,
            correct=int(aligner == int(r["human"])), identity_safe=int(safe),
            has_boundary=int(m is not None and bool(m["devs"])),
            dev_word_ms=(st.median(m["devs"]) * 1000 if m and m["devs"] else None),
            dev_site_ms=(st.median(m["devs_target"]) * 1000 if m and m["devs_target"] else None),
            n_bound_word=(len(m["devs"]) if m else 0),
            n_bound_site=(len(m["devs_target"]) if m else 0),
            coverage=(round(m["cov"], 6) if m else None),
        ))
    if not quiet:
        print(f"[build_table] {run_dir.name}: {len(out)} rows; {buf.getvalue().strip()}")
    return out


def write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def read_tsv(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    for r in rows:
        for k in ("human", "aligner", "correct", "identity_safe", "has_boundary",
                  "n_bound_word", "n_bound_site"):
            r[k] = int(r[k])
        for k in ("dev_word_ms", "dev_site_ms", "coverage"):
            r[k] = float(r[k]) if r[k] != "" else None
    return rows


# ---------- classification metrics (numpy, weighted) ----------
def conf(h, a, w=None):
    h = np.asarray(h); a = np.asarray(a)
    w = np.ones(len(h)) if w is None else np.asarray(w, float)
    tp = w[(h == 1) & (a == 1)].sum(); fn = w[(h == 1) & (a == 0)].sum()
    fp = w[(h == 0) & (a == 1)].sum(); tn = w[(h == 0) & (a == 0)].sum()
    return tp, fn, fp, tn


def cls_metrics(h, a, w=None):
    tp, fn, fp, tn = conf(h, a, w)
    n = tp + fn + fp + tn
    sens = tp / (tp + fn) if tp + fn else math.nan
    spec = tn / (tn + fp) if tn + fp else math.nan
    ba = (sens + spec) / 2
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else math.nan
    hp = (tp + fn) / n if n else math.nan
    return dict(n=n, tp=tp, fn=fn, fp=fp, tn=tn, agree=(tp + tn) / n if n else math.nan,
                hand=hp, base=max(hp, 1 - hp), aligner=(tp + fp) / n if n else math.nan,
                sens=sens, spec=spec, ba=ba, mcc=mcc)


def speaker_index(rows):
    spk = sorted({r["speaker"] for r in rows})
    idx = {s: i for i, s in enumerate(spk)}
    return spk, np.array([idx[r["speaker"]] for r in rows])


def speaker_boot_weights(rows, b=B, seed=SEED):
    """Yield per-row multiplicity vectors for a speaker-cluster bootstrap."""
    spk, si = speaker_index(rows)
    rng = np.random.default_rng(seed)
    for _ in range(b):
        draw = rng.integers(0, len(spk), len(spk))
        cnt = np.bincount(draw, minlength=len(spk))
        yield cnt[si]


def two_way_boot_weights(rows, b=B, seed=SEED, key2="word_type"):
    """Pigeonhole (two-way crossed) bootstrap: speakers and word types resampled
    independently; row weight = (#times speaker drawn) x (#times word type drawn)."""
    spk, si = speaker_index(rows)
    wt = sorted({r[key2] for r in rows})
    widx = {w: i for i, w in enumerate(wt)}
    wi = np.array([widx[r[key2]] for r in rows])
    rng = np.random.default_rng(seed)
    for _ in range(b):
        cs = np.bincount(rng.integers(0, len(spk), len(spk)), minlength=len(spk))
        cw = np.bincount(rng.integers(0, len(wt), len(wt)), minlength=len(wt))
        yield cs[si] * cw[wi]


def pct_ci(vals):
    v = np.array([x for x in vals if x is not None and not math.isnan(x)])
    if len(v) == 0:
        return (math.nan, math.nan, 0)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v))


def weighted_auc(score, label, w=None):
    """P(score_pos > score_neg) + 0.5 P(tie), label==1 positive; weighted."""
    s = np.asarray(score, float); y = np.asarray(label)
    w = np.ones(len(s)) if w is None else np.asarray(w, float)
    wp = np.where(y == 1, w, 0.0); wn = np.where(y == 0, w, 0.0)
    P, N = wp.sum(), wn.sum()
    if P == 0 or N == 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")
    s, wp, wn = s[order], wp[order], wn[order]
    uniq, start = np.unique(s, return_index=True)
    gp = np.add.reduceat(wp, start); gn = np.add.reduceat(wn, start)
    below_n = np.concatenate([[0.0], np.cumsum(gn)[:-1]])
    return float((gp * (below_n + 0.5 * gn)).sum() / (P * N))


def weighted_median(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = w > 0
    x, w = x[m], w[m]
    if len(x) == 0:
        return math.nan
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    half = c[-1] / 2
    i = np.searchsorted(c, half)
    # statistics.median convention for integer weights: average when exactly at half
    if abs(c[i] - half) < 1e-9 and i + 1 < len(x):
        return float((x[i] + x[i + 1]) / 2)
    return float(x[i])
