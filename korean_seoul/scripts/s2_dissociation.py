#!/usr/bin/env python3
"""S2: same-token dissociation between boundary error and variant-selection error.

Subsets: identity-safe (PRIMARY), full (secondary).
Measures: word-level median deviation (dev_word_ms), process-site deviation (dev_site_ms).
  (a) AUC (boundary error -> wrong selection): speaker-cluster and two-way
      (speaker x word type, pigeonhole) bootstrap CIs, B=2000, seed 20260916
  (c) leave-one-speaker-out CV AUC for process / boundary / process+boundary
      (+ process x boundary) logistic models; increment with speaker-bootstrap CI
  (d) median difference (wrong - correct) with speaker-cluster (and two-way) CI
(b) is in s2b_glmer.R.
Input: work/candidates_natural.tsv (from s1). Output: s2_*.tsv (aggregates only).
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_common as RC   # noqa: E402

warnings.filterwarnings("ignore")
EPS_MS = 1.0   # log(dev_ms + 1)


def subsets(rows):
    safe = [r for r in rows if r["identity_safe"]]
    return [("identity_safe", safe), ("full", rows)]


def usable(sub, field):
    return [r for r in sub if r[field] is not None]


def part_a_d(rows):
    out_auc, out_med = [], []
    for sname, sub in subsets(rows):
        for field in ("dev_site_ms", "dev_word_ms"):
            U = usable(sub, field)
            x = np.array([r[field] for r in U])
            wrong = np.array([1 - r["correct"] for r in U])
            auc = RC.weighted_auc(x, wrong)
            sp = [RC.weighted_auc(x, wrong, w) for w in RC.speaker_boot_weights(U)]
            tw = [RC.weighted_auc(x, wrong, w) for w in RC.two_way_boot_weights(U)]
            slo, shi, sn = RC.pct_ci(sp)
            tlo, thi, tn = RC.pct_ci(tw)
            out_auc.append(dict(subset=sname, measure=field, n=len(U),
                                n_correct=int((wrong == 0).sum()), n_wrong=int(wrong.sum()),
                                speakers=len({r["speaker"] for r in U}),
                                word_types=len({r["word_type"] for r in U}),
                                auc=round(auc, 4),
                                spk_lo=round(slo, 4), spk_hi=round(shi, 4), spk_valid=sn,
                                twoway_lo=round(tlo, 4), twoway_hi=round(thi, 4), twoway_valid=tn))

            def md(w):
                a = RC.weighted_median(x[wrong == 0], w[wrong == 0])
                b = RC.weighted_median(x[wrong == 1], w[wrong == 1])
                return b - a
            one = np.ones(len(x))
            d = md(one)
            sp = [md(w) for w in RC.speaker_boot_weights(U)]
            tw = [md(w) for w in RC.two_way_boot_weights(U)]
            slo, shi, _ = RC.pct_ci(sp)
            tlo, thi, _ = RC.pct_ci(tw)
            out_med.append(dict(subset=sname, measure=field, n=len(U),
                                median_correct=round(RC.weighted_median(x[wrong == 0], one[wrong == 0]), 3),
                                median_wrong=round(RC.weighted_median(x[wrong == 1], one[wrong == 1]), 3),
                                diff=round(d, 3), spk_lo=round(slo, 3), spk_hi=round(shi, 3),
                                twoway_lo=round(tlo, 3), twoway_hi=round(thi, 3)))
    return out_auc, out_med


def part_a2(rows):
    out = []
    for sname, sub in subsets(rows):
        for field in ("dev_site_ms", "dev_word_ms"):
            U = usable(sub, field)
            x = np.array([r[field] for r in U])
            wrong = np.array([1 - r["correct"] for r in U])
            proc = np.array([r["process"] for r in U])
            st_ = stratified_auc(x, wrong, proc)
            bs = [stratified_auc(x, wrong, proc, w) for w in RC.speaker_boot_weights(U)]
            lo, hi, _ = RC.pct_ci(bs)
            tw = [stratified_auc(x, wrong, proc, w) for w in RC.two_way_boot_weights(U)]
            tlo, thi, _ = RC.pct_ci(tw)
            out.append(dict(subset=sname, measure=field, stratum="within-process pairs only",
                            n=len(U), n_wrong=int(wrong.sum()), auc=round(st_, 4),
                            spk_lo=round(lo, 4), spk_hi=round(hi, 4),
                            twoway_lo=round(tlo, 4), twoway_hi=round(thi, 4)))
            for p in RC.PROCS:
                m = proc == p
                if m.sum() < 20 or wrong[m].sum() < 5 or (1 - wrong[m]).sum() < 5:
                    continue
                Up = [r for r, k in zip(U, m) if k]
                a = RC.weighted_auc(x[m], wrong[m])
                bs = [RC.weighted_auc(x[m], wrong[m], w) for w in RC.speaker_boot_weights(Up)]
                lo, hi, _ = RC.pct_ci(bs)
                tw = [RC.weighted_auc(x[m], wrong[m], w) for w in RC.two_way_boot_weights(Up)]
                tlo, thi, _ = RC.pct_ci(tw)
                out.append(dict(subset=sname, measure=field, stratum=p, n=int(m.sum()),
                                n_wrong=int(wrong[m].sum()), auc=round(a, 4),
                                spk_lo=round(lo, 4), spk_hi=round(hi, 4),
                                twoway_lo=round(tlo, 4), twoway_hi=round(thi, 4)))
    return out


def design(U, field, model, procs):
    z = np.log(np.array([r[field] for r in U]) + EPS_MS)
    P = np.array([[1.0 if r["process"] == p else 0.0 for p in procs[1:]] for r in U])
    if model == "process":
        return P
    if model == "boundary":
        return z[:, None]
    if model == "process+boundary":
        return np.column_stack([P, z])
    if model == "process*boundary":
        oh = np.array([[1.0 if r["process"] == p else 0.0 for p in procs] for r in U])
        return np.column_stack([P, oh * z[:, None]])
    raise ValueError(model)


MODELS = ["process", "boundary", "process+boundary", "process*boundary"]


def loso(U, field, folds=None):
    """folds=None: leave-one-speaker-out; folds=k: k speaker-grouped folds (seeded)."""
    procs = sorted({r["process"] for r in U})
    y = np.array([r["correct"] for r in U])
    spk = np.array([r["speaker"] for r in U])
    us = np.unique(spk)
    if folds is None:
        fold_of = {s: i for i, s in enumerate(us)}
    else:
        perm = np.random.default_rng(RC.SEED).permutation(us)
        fold_of = {s: i % folds for i, s in enumerate(perm)}
    fid = np.array([fold_of[s] for s in spk])
    preds = {}
    for m in MODELS:
        X = design(U, field, m, procs)
        p = np.zeros(len(U))
        for f in np.unique(fid):
            te = fid == f
            tr = ~te
            clf = LogisticRegression(C=1e6, max_iter=5000)
            clf.fit(X[tr], y[tr])
            p[te] = clf.predict_proba(X[te])[:, 1]
        preds[m] = p
    return y, preds, spk


def within_speaker_auc(pred, y, spk):
    """Mean of per-held-out-speaker AUCs (removes the between-fold intercept artifact)."""
    v = [RC.weighted_auc(pred[spk == s], y[spk == s]) for s in np.unique(spk)]
    v = [x for x in v if not math.isnan(x)]
    return float(np.mean(v)), len(v)


def stratified_auc(score, label, strata, w=None):
    """AUC using only (positive, negative) pairs from the same stratum."""
    score = np.asarray(score); label = np.asarray(label); strata = np.asarray(strata)
    w = np.ones(len(score)) if w is None else np.asarray(w, float)
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        P = w[m & (label == 1)].sum(); N = w[m & (label == 0)].sum()
        if P == 0 or N == 0:
            continue
        num += RC.weighted_auc(score[m], label[m], w[m]) * P * N
        den += P * N
    return num / den if den else math.nan


def part_c(rows):
    out = []
    for sname, sub in subsets(rows):
        for field in ("dev_site_ms", "dev_word_ms"):
            U = usable(sub, field)
            # drop processes too sparse to estimate in every fold (<20 candidates)
            from collections import Counter
            cnt = Counter(r["process"] for r in U)
            U = [r for r in U if cnt[r["process"]] >= 20]
            y, preds, spk = loso(U, field)
            aucs = {m: RC.weighted_auc(preds[m], y) for m in MODELS}
            y10, p10, _ = loso(U, field, folds=10)
            for m in MODELS:
                ws, ns = within_speaker_auc(preds[m], y, spk)
                out.append(dict(subset=sname, measure=field, n=len(U), processes="",
                                model=m, cv=f"LOSO, mean within-held-out-speaker AUC ({ns} speakers)",
                                auc=round(ws, 4), spk_lo="", spk_hi=""))
                out.append(dict(subset=sname, measure=field, n=len(U), processes="",
                                model=m, cv="10-fold speaker-grouped (seeded), pooled OOF",
                                auc=round(RC.weighted_auc(p10[m], y10), 4), spk_lo="", spk_hi=""))
            ws_inc = within_speaker_auc(preds["process+boundary"], y, spk)[0] - \
                within_speaker_auc(preds["process"], y, spk)[0]
            us = np.unique(spk)
            dv = np.array([RC.weighted_auc(preds["process+boundary"][spk == s_], y[spk == s_])
                           - RC.weighted_auc(preds["process"][spk == s_], y[spk == s_]) for s_ in us])
            dv = dv[~np.isnan(dv)]
            rng = np.random.default_rng(RC.SEED)
            bm = [dv[rng.integers(0, len(dv), len(dv))].mean() for _ in range(RC.B)]
            lo_, hi_, _ = RC.pct_ci(bm)
            out.append(dict(subset=sname, measure=field, n=len(U), processes="",
                            model="increment: (process+boundary) - process",
                            cv="LOSO, mean within-held-out-speaker AUC", auc=round(ws_inc, 4),
                            spk_lo=round(lo_, 4), spk_hi=round(hi_, 4)))
            boots = {m: [] for m in MODELS}
            inc = []
            for w in RC.speaker_boot_weights(U):
                a = {m: RC.weighted_auc(preds[m], y, w) for m in MODELS}
                for m in MODELS:
                    boots[m].append(a[m])
                inc.append(a["process+boundary"] - a["process"])
            for m in MODELS:
                lo, hi, _ = RC.pct_ci(boots[m])
                out.append(dict(subset=sname, measure=field, n=len(U),
                                processes="+".join(sorted(cnt_k for cnt_k in {r["process"] for r in U})),
                                model=m, cv="leave-one-speaker-out (40 folds), pooled OOF",
                                auc=round(aucs[m], 4), spk_lo=round(lo, 4), spk_hi=round(hi, 4)))
            lo, hi, _ = RC.pct_ci(inc)
            out.append(dict(subset=sname, measure=field, n=len(U),
                            processes="", model="increment: (process+boundary) - process",
                            cv="same", auc=round(aucs["process+boundary"] - aucs["process"], 4),
                            spk_lo=round(lo, 4), spk_hi=round(hi, 4)))
    return out


def show(title, rows):
    print(f"\n== {title} ==")
    keys = list(rows[0].keys())
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r[k]) for k in keys))


def main():
    rows = RC.read_tsv(RC.WORK / "candidates_natural.tsv")
    a, d = part_a_d(rows)
    RC.write_tsv(RC.REV / "s2a_auc.tsv", a)
    RC.write_tsv(RC.REV / "s2d_median_diff.tsv", d)
    show("S2a AUC (wrong selection detected by larger boundary error)", a)
    show("S2d median difference wrong - correct (ms)", d)
    a2 = part_a2(rows)
    RC.write_tsv(RC.REV / "s2a_auc_by_process.tsv", a2)
    show("S2a' AUC within process (stratified) and per process", a2)
    c = part_c(rows)
    RC.write_tsv(RC.REV / "s2c_cv_auc.tsv", c)
    show("S2c LOSO CV AUC (outcome = correct selection)", c)


if __name__ == "__main__":
    main()
