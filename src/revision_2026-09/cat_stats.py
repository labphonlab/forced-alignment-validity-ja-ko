# SCOPE: journal-only
"""C2・C3・C4(d)・C5：範疇軸の検定（docs/decisions-log.md 2026-09-17 事前登録）。

入力はトークン表（results/revision_2026-09/work/cat_tokens_*.tsv）だけ。出力は集計値のみ。

区間
  二元：話者と語タイプの pigeonhole ブートストラップ（各水準を独立に復元抽出し、
        トークンの重みを 話者の回数 × 語の回数 にする）
  話者：話者だけを復元抽出
  いずれも 2,000 回、seed 20260917、95% パーセンタイル区間
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 20260917
B = 2000
WORK = Path(__file__).resolve().parents[2] / "results" / "revision_2026-09" / "work"
OUT = WORK.parent
ROWS: list[dict] = []


def add(analysis, subset, quantity, est, ci2=(np.nan, np.nan), cis=(np.nan, np.nan), n=None, note=""):
    ROWS.append(dict(analysis=analysis, subset=subset, quantity=quantity, estimate=est,
                     ci2way_low=ci2[0], ci2way_high=ci2[1], cispk_low=cis[0], cispk_high=cis[1],
                     n=n, note=note))


def metrics(tp, fn, tn, fp):
    tp, fn, tn, fp = (np.asarray(x, dtype=float) for x in (tp, fn, tn, fp))
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        ba = (sens + spec) / 2
        acc = (tp + tn) / (tp + fn + tn + fp)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(sensitivity=sens, specificity=spec, balanced_accuracy=ba, agreement=acc, mcc=mcc)


def draws(n_levels, rng, B=B):
    idx = rng.integers(0, n_levels, size=(B, n_levels))
    out = np.zeros((B, n_levels))
    for b in range(B):
        out[b] = np.bincount(idx[b], minlength=n_levels)
    return out


def cell_counts(d: pd.DataFrame, spk_codes, word_codes, n_s, n_w):
    """4セル（tp fn tn fp）の 話者×語 疎行列。"""
    cells = {}
    for name, rl, hl in (("tp", "devoiced", "devoiced"), ("fn", "devoiced", "voiced"),
                         ("tn", "voiced", "voiced"), ("fp", "voiced", "devoiced")):
        m = ((d.ref_label == rl) & (d.hyp_label == hl)).to_numpy()
        cells[name] = sparse.csr_matrix((np.ones(m.sum()), (spk_codes[m], word_codes[m])), shape=(n_s, n_w))
    return cells


def boot_confusion(d: pd.DataFrame, two_way: bool):
    s_codes, s_lev = pd.factorize(d.speaker)
    w_codes, w_lev = pd.factorize(d.word_id)
    rng = np.random.default_rng(SEED)
    ms = draws(len(s_lev), rng)
    mw = draws(len(w_lev), rng) if two_way else np.ones((B, len(w_lev)))
    cells = cell_counts(d, s_codes, w_codes, len(s_lev), len(w_lev))
    res = {}
    for k, c in cells.items():
        res[k] = np.einsum("bs,bs->b", ms, (c @ mw.T).T)
    return metrics(res["tp"], res["fn"], res["tn"], res["fp"])


def point_confusion(d):
    tp = ((d.ref_label == "devoiced") & (d.hyp_label == "devoiced")).sum()
    fn = ((d.ref_label == "devoiced") & (d.hyp_label == "voiced")).sum()
    tn = ((d.ref_label == "voiced") & (d.hyp_label == "voiced")).sum()
    fp = ((d.ref_label == "voiced") & (d.hyp_label == "devoiced")).sum()
    return metrics(tp, fn, tn, fp), (tp, fn, tn, fp)


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def report_confusion(analysis, subset, d, two_way=True):
    pt, (tp, fn, tn, fp) = point_confusion(d)
    b2 = boot_confusion(d, True) if two_way else None
    bs = boot_confusion(d, False)
    n_s, n_w = d.speaker.nunique(), d.word_id.nunique()
    note = f"tp={tp};fn={fn};tn={tn};fp={fp};speakers={n_s};word_types={n_w}"
    for q in ("sensitivity", "specificity", "balanced_accuracy", "mcc", "agreement"):
        add(analysis, subset, q, float(pt[q]), ci(b2[q]) if b2 else (np.nan, np.nan), ci(bs[q]), len(d), note)
    return pt, b2, bs


# ---------------------------------------------------------------- P2 用
def weighted_median(x_sorted, w_sorted):
    cw = np.cumsum(w_sorted)
    if cw[-1] <= 0:
        return np.nan
    return x_sorted[np.searchsorted(cw, cw[-1] / 2)]


def weighted_auc(inv, n_u, y_wrong, w):
    wc = np.bincount(inv, weights=w * (~y_wrong), minlength=n_u)
    ww = np.bincount(inv, weights=w * y_wrong, minlength=n_u)
    below = np.cumsum(wc) - wc
    den = ww.sum() * wc.sum()
    return np.sum(ww * (below + 0.5 * wc)) / den if den > 0 else np.nan


def p2(d: pd.DataFrame, label: str):
    d = d.copy()
    d["err"] = (d.onset_error_ms.astype(float).abs() + d.offset_error_ms.astype(float).abs()) / 2
    d["correct"] = (d.ref_label == d.hyp_label).astype(int)
    d = d.sort_values("err").reset_index(drop=True)
    y_wrong = (d.correct == 0).to_numpy()
    err = d.err.to_numpy()
    s_codes, s_lev = pd.factorize(d.speaker)
    w_codes, w_lev = pd.factorize(d.word_id)
    uniq, inv = np.unique(err, return_inverse=True)

    def stats(w):
        mw_ = weighted_median(err[y_wrong], w[y_wrong])
        mc_ = weighted_median(err[~y_wrong], w[~y_wrong])
        return mw_ - mc_, weighted_auc(inv, len(uniq), y_wrong, w)

    one = np.ones(len(d))
    md, auc = stats(one)
    rng = np.random.default_rng(SEED)
    ms = draws(len(s_lev), rng)
    mw = draws(len(w_lev), rng)
    bmd, bauc, smd, sauc = [], [], [], []
    for b in range(B):
        w2 = ms[b][s_codes] * mw[b][w_codes]
        x, y = stats(w2)
        bmd.append(x)
        bauc.append(y)
        x, y = stats(ms[b][s_codes])
        smd.append(x)
        sauc.append(y)
    n_note = (f"n_correct={int(d.correct.sum())};n_wrong={int(y_wrong.sum())};speakers={len(s_lev)};"
              f"word_types={len(w_lev)};median_err_correct={np.median(err[~y_wrong]):.2f};"
              f"median_err_wrong={np.median(err[y_wrong]):.2f}")
    add("P2", label, "median_err_diff_wrong_minus_correct_ms", md, ci(bmd), ci(smd), len(d), n_note)
    add("P2", label, "auc_err_predicts_wrong", auc, ci(bauc), ci(sauc), len(d), n_note)

    # 話者を1人ずつ抜く交差検証
    m = 5.0
    d["zl"] = np.log(d.err + 1)
    aucs = []
    oof = np.full((len(d), 2), np.nan)
    for s in s_lev:
        te_mask = (d.speaker == s).to_numpy()
        tr = d[~te_mask]
        te = d[te_mask]
        g = tr.correct.mean()
        agg = tr.groupby("word_id").correct.agg(["sum", "count"])
        k_tr = tr.word_id.map(agg["sum"]).to_numpy() - tr.correct.to_numpy()
        n_tr = tr.word_id.map(agg["count"]).to_numpy() - 1
        p_tr = (k_tr + m * g) / (n_tr + m)
        k_te = te.word_id.map(agg["sum"]).fillna(0).to_numpy()
        n_te = te.word_id.map(agg["count"]).fillna(0).to_numpy()
        p_te = (k_te + m * g) / (n_te + m)
        lo = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))  # noqa: E731
        mu, sd = tr.zl.mean(), tr.zl.std()
        X0_tr, X0_te = lo(p_tr)[:, None], lo(p_te)[:, None]
        X1_tr = np.column_stack([lo(p_tr), (tr.zl - mu) / sd])
        X1_te = np.column_stack([lo(p_te), (te.zl - mu) / sd])
        m0 = LogisticRegression(penalty=None, max_iter=1000).fit(X0_tr, tr.correct)
        m1 = LogisticRegression(penalty=None, max_iter=1000).fit(X1_tr, tr.correct)
        q0 = m0.predict_proba(X0_te)[:, 1]
        q1 = m1.predict_proba(X1_te)[:, 1]
        oof[te_mask, 0] = q0
        oof[te_mask, 1] = q1
        if te.correct.nunique() == 2:
            aucs.append((s, len(te), roc_auc_score(te.correct, q0), roc_auc_score(te.correct, q1)))
    a = pd.DataFrame(aucs, columns=["speaker", "n", "auc0", "auc1"])
    a["inc"] = a.auc1 - a.auc0
    rng = np.random.default_rng(SEED)
    bi = [a.inc.to_numpy()[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    b0 = [a.auc0.to_numpy()[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    note = f"speakers_with_both_classes={len(a)} of {len(s_lev)};target_encoding_m=5"
    add("P2", label, "loso_auc_word_only_mean_within_speaker", a.auc0.mean(), cis=ci(b0), n=len(d), note=note)
    add("P2", label, "loso_auc_word_plus_err_mean_within_speaker", a.auc1.mean(), n=len(d), note=note)
    add("P2", label, "loso_auc_increment", a.inc.mean(), cis=ci(bi), n=len(d),
        note=note + f";verdict={'not_a_proxy(<0.02)' if a.inc.mean() < 0.02 else 'increment>=0.02'}")
    add("P2", label, "loso_pooled_oof_auc_word_only", roc_auc_score(d.correct, oof[:, 0]), n=len(d))
    add("P2", label, "loso_pooled_oof_auc_word_plus_err", roc_auc_score(d.correct, oof[:, 1]), n=len(d))
    add("P2", label, "accuracy", d.correct.mean(), n=len(d))
    d[["speaker", "word_id", "correct", "err", "ref_label", "env"]].to_csv(
        WORK / f"cat_p2_{label}.tsv", sep="\t", index=False)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main177")
    ap.add_argument("--p3", nargs="*", default=["B1", "B1_prior_rerun", "B1_prior_equal", "B1_prior_oos"])
    args = ap.parse_args()

    d = pd.read_csv(WORK / f"cat_tokens_{args.main}.tsv", sep="\t", dtype=str, keep_default_na=False)
    comp = d[d.status == "competing"]
    # ---------------- P1
    report_confusion("P1", "all_tokens(reference)", d)
    report_confusion("P1", "competing", comp)
    for env in ("voiceless_both", "other"):
        report_confusion("P1", f"competing|env={env}", comp[comp.env == env])
    for t in ("symbol", "presence", "mixed"):
        report_confusion("P1", f"competing|type={t}", comp[comp.ctype == t])
    report_confusion("P1", "competing|type=symbol+mixed", comp[comp.ctype.isin(["symbol", "mixed"])])
    lex = d[d.status == "lexical"]
    report_confusion("P1", "lexically_determined", lex)
    for env in ("voiceless_both", "other"):
        report_confusion("P1", f"lexically_determined|env={env}", lex[lex.env == env])
    for r in ROWS:
        if r["subset"] == "competing" and r["quantity"] == "balanced_accuracy":
            lo, hi = r["ci2way_low"], r["ci2way_high"]
            add("P1", "competing", "verdict_BA_2way_CI_excludes_0.5", float(not (lo <= 0.5 <= hi)),
                note=f"BA={r['estimate']:.4f} [{lo:.4f},{hi:.4f}]")
    # 組成
    for (st, ct, rl), g in d.groupby(["status", "ctype", "ref_label"]):
        add("composition", f"{st}|{ct or '-'}", f"n_ref_{rl}", len(g))
    for (st, env), g in d.groupby(["status", "env"]):
        add("composition", f"{st}|env={env}", "n", len(g))
    add("composition", "competing", "n_speakers", comp.speaker.nunique())
    add("composition", "competing", "n_word_types", comp.word_id.nunique())

    # ---------------- P2
    sym = comp[comp.ctype == "symbol"]
    p2(sym, "symbol")
    supp = comp[comp.ctype.isin(["symbol", "mixed"]) & comp.hyp_value.isin(["V", "D"])]
    p2(supp, "supp_symbol_or_mixed_vowel_present")

    # ---------------- C5 探索
    rng = np.random.default_rng(SEED)
    both = comp.groupby("word_id").ref_label.nunique()
    cc_words = both[both == 2].index
    cc = comp[comp.word_id.isin(cc_words)]
    bas, mccs, ns = [], [], []
    for _ in range(200):
        parts = []
        for w, g in cc.groupby("word_id"):
            k = min((g.ref_label == "devoiced").sum(), (g.ref_label == "voiced").sum())
            for lab in ("devoiced", "voiced"):
                gg = g[g.ref_label == lab]
                parts.append(gg.iloc[rng.choice(len(gg), k, replace=False)])
        s = pd.concat(parts)
        pt, _ = point_confusion(s)
        bas.append(pt["balanced_accuracy"])
        mccs.append(pt["mcc"])
        ns.append(len(s))
    add("C5", "case_control(200 draws)", "balanced_accuracy_mean", float(np.mean(bas)),
        cis=(float(np.min(bas)), float(np.max(bas))), n=int(np.mean(ns)),
        note=f"range over draws in ci columns;word_types={len(cc_words)}")
    add("C5", "case_control(200 draws)", "mcc_mean", float(np.mean(mccs)),
        cis=(float(np.min(mccs)), float(np.max(mccs))), n=int(np.mean(ns)))
    comp2 = comp.assign(correct=(comp.ref_label == comp.hyp_label).astype(int),
                        dev=(comp.ref_label == "devoiced").astype(int))
    pw = comp2.groupby("word_id").agg(n=("correct", "size"), acc=("correct", "mean"), p=("dev", "mean"))
    pw["majority"] = np.maximum(pw.p, 1 - pw.p)
    for nmin in (5, 20):
        q = pw[pw.n >= nmin]
        qs = np.percentile(q.acc, [10, 25, 50, 75, 90])
        add("C5", f"per_word_accuracy|n>={nmin}", "quantiles_10_25_50_75_90", float(qs[2]),
            n=len(q), note=";".join(f"{v:.3f}" for v in qs))
        add("C5", f"per_word_accuracy|n>={nmin}", "token_weighted_accuracy",
            float(np.average(q.acc, weights=q.n)), n=int(q.n.sum()))
        add("C5", f"per_word_accuracy|n>={nmin}", "token_weighted_majority_baseline",
            float(np.average(q.majority, weights=q.n)), n=int(q.n.sum()))
        add("C5", f"per_word_accuracy|n>={nmin}", "share_words_acc_below_majority",
            float((q.acc < q.majority).mean()), n=len(q))
    cnt = pw.n.sort_values(ascending=False)
    cum = cnt.cumsum() / cnt.sum()
    for share in (0.5, 0.8, 0.9):
        add("C5", "concentration", f"word_types_for_{int(share*100)}pct_tokens", int((cum < share).sum() + 1),
            n=int(cnt.sum()), note=f"of {len(cnt)} types")
    add("C5", "concentration", "top10_types_token_share", float(cum.iloc[min(9, len(cum) - 1)]), n=int(cnt.sum()))
    add("C5", "concentration", "types_with_1_token", int((cnt == 1).sum()), n=len(cnt))

    # ---------------- P3
    tabs = {}
    for s in args.p3:
        f = WORK / f"cat_tokens_{s}.tsv"
        if f.exists():
            tabs[s] = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False).set_index("ref_unit_id")
    if "B1" in tabs:
        base = tabs["B1"]
        ids = base.index[base.status == "competing"]
        for s, t in tabs.items():
            ids = ids.intersection(t.index)
        stat_dis = {s: int((t.loc[ids, "status"] != "competing").sum()) for s, t in tabs.items()}
        add("P3", "token_set", "n_competing_in_B1_and_present_in_all", len(ids),
            note=";".join(f"{k}_status_not_competing={v}" for k, v in stat_dis.items()))
        ref = base.loc[ids]
        spk_codes, spk_lev = pd.factorize(ref.speaker)
        rng = np.random.default_rng(SEED)
        ms = draws(len(spk_lev), rng)
        boots = {}
        for s, t in tabs.items():
            dd = ref[["speaker", "word_id", "ref_label"]].assign(hyp_label=t.loc[ids, "hyp_label"].to_numpy())
            pt, _ = point_confusion(dd)
            cells = {}
            for name, rl, hl in (("tp", "devoiced", "devoiced"), ("fn", "devoiced", "voiced"),
                                 ("tn", "voiced", "voiced"), ("fp", "voiced", "devoiced")):
                m = ((dd.ref_label == rl) & (dd.hyp_label == hl)).to_numpy()
                cells[name] = np.bincount(spk_codes[m], minlength=len(spk_lev))
            bm = metrics(*(ms @ cells[k] for k in ("tp", "fn", "tn", "fp")))
            boots[s] = (pt, bm, dd.hyp_label.to_numpy())
            for q in ("agreement", "sensitivity", "specificity", "balanced_accuracy", "mcc"):
                add("P3", s, q, float(pt[q]), cis=ci(bm[q]), n=len(dd), note=f"speakers={len(spk_lev)}")
        pairs = [("B1_prior_oos", "B1_prior_equal"), ("B1_prior_oos", "B1"), ("B1_prior_equal", "B1"),
                 ("B1_prior_rerun", "B1")]
        for a_, b_ in pairs:
            if a_ not in boots or b_ not in boots:
                continue
            pa, ba_, ha = boots[a_]
            pb, bb, hb = boots[b_]
            lab = f"{a_}-minus-{b_}"
            chg = ha != hb
            add("P3", lab, "share_decisions_changed", float(chg.mean()), n=len(chg), note=f"n_changed={int(chg.sum())}")
            for q in ("agreement", "balanced_accuracy", "sensitivity", "specificity", "mcc"):
                add("P3", lab, f"delta_{q}", float(pa[q] - pb[q]), cis=ci(ba_[q] - bb[q]), n=len(chg))
            dd_ = (pa["agreement"] - pb["agreement"]) - (pa["balanced_accuracy"] - pb["balanced_accuracy"])
            bdd = (ba_["agreement"] - bb["agreement"]) - (ba_["balanced_accuracy"] - bb["balanced_accuracy"])
            add("P3", lab, "delta_agreement_minus_delta_BA", float(dd_), cis=ci(bdd), n=len(chg))
            # 型別の変化
            for t in ("symbol", "presence", "mixed"):
                mt = (ref.ctype == t).to_numpy()
                if mt.sum():
                    add("P3", lab, f"share_decisions_changed|type={t}", float(chg[mt].mean()), n=int(mt.sum()))
        if "B1_prior_oos" in boots and "B1_prior_equal" in boots:
            pa, ba_, _ = boots["B1_prior_oos"]
            pb, bb, _ = boots["B1_prior_equal"]
            da = pa["agreement"] - pb["agreement"]
            db = pa["balanced_accuracy"] - pb["balanced_accuracy"]
            lo, hi = ci((ba_["agreement"] - bb["agreement"]) - (ba_["balanced_accuracy"] - bb["balanced_accuracy"]))
            verdict = "supported" if (da > db and da > 0 and lo > 0) else (
                "direction_only(CI includes 0)" if da > db else "not_supported")
            add("P3", "verdict(oos vs equal)", "delta_agreement_gt_delta_BA", float(da > db),
                note=f"dAgree={da:.4f};dBA={db:.4f};diff CI=[{lo:.4f},{hi:.4f}];{verdict}")
        # P3 の探索：学習用講演で語の率を推定できた語に限る
    out = pd.DataFrame(ROWS)
    out.to_csv(OUT / "cat_REPORT_numbers.tsv", sep="\t", index=False, float_format="%.5f")
    for a_ in ("P1", "P2", "P3", "C5", "composition"):
        out[out.analysis == a_].to_csv(OUT / f"cat_{a_.lower()}.tsv", sep="\t", index=False, float_format="%.5f")
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 70)
    print(out.drop(columns=["analysis"]).to_string())


if __name__ == "__main__":
    main()
