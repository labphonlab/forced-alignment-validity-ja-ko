# SCOPE: pilot（段階0–1）
"""整列結果を CSJ 人手ラベルと突き合わせ、事前登録（docs/decisions-log.md D5・D6・D11）の量を出す。

突き合わせは mfa-ja-validation の変換器（mfa_textgrid_to_units.py）と系列アラインメント
（sequence_align.py）を呼ぶだけで、改変しない。トークン単位の表は work/eval/ に置き、
results/ には集計値だけを書く。

praatio が要るので、MFA 3.4.1 の環境の python で実行する。

    PY=python3
    $PY scripts/02_evaluate.py prepare-ref
    $PY scripts/02_evaluate.py align --system B0 --units ../mfa-ja-validation/results/mfa_csj_units.tsv
    $PY scripts/02_evaluate.py align --system B1 --textgrids work/align/B1_pretrained_spk
    $PY scripts/02_evaluate.py summarize --systems B0 B1 A B
    $PY scripts/02_evaluate.py replicates --a A A_r2 A_r3 --b B B_r2 B_r3
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT.parent / "mfa-ja-validation"
SPLIT = ROOT / "splits" / "csj_core_split.tsv"
EVAL = ROOT / "work" / "eval"
RESULTS = ROOT / "results"
REF_ALL = VALID / "results" / "csj_units.tsv"
REF_TEST = EVAL / "ref_test_units.tsv"

# D5 のクラス定義
TARGET = {
    "devoiced_fused", "devoiced_fused_geminate", "devoiced_fused_long",
    "devoiced_high", "devoiced_nonhigh",
}
CONTROL = {"vowel_short", "vowel_long", "stop", "fricative", "affricate", "nasal", "flap"}
MATCHED = {"displacement", "substitution"}
THRESHOLDS = (10, 20, 25, 50)

# D6 の範疇判定の定義
REF_DEVOICED_MARKS = {"i̥", "ɯ̥"}
REF_VOICED_HIGH = {"i", "ɯ"}
VOICED_VOWELS = {"a", "aː", "e", "eː", "i", "iː", "o", "oː", "ɯ", "ɯː", "ɨ", "ɨː"}
DEVOICED_VOWELS = {"i̥", "ɯ̥", "ɨ̥"}
VOICELESS_CONTEXT_CLASSES = {"stop", "fricative", "affricate", "geminate_stop", "geminate_affricate"}
VOICED_OBSTRUENT_INITIALS = ("b", "d", "ɡ", "g", "z", "ʑ", "v")

SEED = 20260915
N_BOOT = 2000

csv.field_size_limit(sys.maxsize)


def test_talks() -> list[str]:
    with SPLIT.open(encoding="utf-8") as fh:
        return sorted(r["file_id"] for r in csv.DictReader(fh, delimiter="\t") if r["split"] == "test")


def filter_rows(src: Path, dst: Path, talks: set[str]) -> int:
    """先頭列（file_id）が評価用講演の行だけを写す。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        fo.write(fi.readline())
        for line in fi:
            if line.split("\t", 1)[0] in talks:
                fo.write(line)
                n += 1
    return n


# MFA は話者名つきの層（"spk123 - phones"）を出すことがある。変換器は "phones"/"words" を探すので揃える。
TIER_RE = re.compile(r'(name = ")[^"\n]* - (words|phones)(")')


def normalize_textgrids(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.TextGrid"):
        old.unlink()
    paths = sorted(src.rglob("*.TextGrid"))
    for p in paths:
        text = p.read_text(encoding="utf-8")
        (dst / p.name).write_text(TIER_RE.sub(r"\1\2\3", text), encoding="utf-8")
    return len(paths)


def run(cmd: list) -> None:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    # 変換器・系列アラインメントの標準出力は件数の要約だけなので、そのまま見せる
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print(proc.stderr.strip()[-2000:], file=sys.stderr)
        raise SystemExit(proc.returncode)


def cmd_prepare_ref(_args) -> int:
    n = filter_rows(REF_ALL, REF_TEST, set(test_talks()))
    print(f"reference units (評価用講演): {n} -> {REF_TEST}")
    return 0


def cmd_align(args) -> int:
    if not REF_TEST.exists():
        print("[error] 先に prepare-ref を実行すること。", file=sys.stderr)
        return 1
    talks = set(test_talks())
    out = EVAL / args.system
    out.mkdir(parents=True, exist_ok=True)
    units = out / "units.tsv"

    if args.textgrids:
        n = normalize_textgrids(args.textgrids, out / "textgrids")
        got = {p.stem for p in (out / "textgrids").glob("*.TextGrid")}
        print(f"TextGrids: {n}（評価用 {len(talks)} 講演のうち欠落 {len(talks - got)}）")
        run([
            sys.executable, VALID / "src" / "convert" / "mfa_textgrid_to_units.py", out / "textgrids",
            "--corpus", "csj", "--metric", "accuracy", "--style", "aps", "--speaker-id", "csj",
            "--out", units,
        ])
    elif args.units:
        n = filter_rows(args.units, units, talks)
        print(f"hypothesis units (評価用講演): {n}")
    else:
        print("[error] --textgrids か --units が必要。", file=sys.stderr)
        return 1

    run([
        sys.executable, VALID / "src" / "evaluate" / "sequence_align.py",
        "--reference", REF_TEST, "--hypothesis", units, "--mapping", "csj",
        "--out", out / "alignment.tsv",
    ])
    return 0


def load_ref_info() -> dict[str, tuple[str, str, str, str]]:
    """参照単位ID → (クラス, 正規ラベル, 直前単位のクラス, 直前単位の正規ラベル)。"""
    by_file: dict[str, list[tuple[float, str, str, str]]] = defaultdict(list)
    with REF_TEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            by_file[r["file_id"]].append(
                (float(r["t_start"]), r["unit_id"], r["phone_class"], r["label_canonical"])
            )
    info: dict[str, tuple[str, str, str, str]] = {}
    for rows in by_file.values():
        rows.sort()
        prev_cls, prev_lab = "", ""
        for _, uid, cls, lab in rows:
            info[uid] = (cls, lab, prev_cls, prev_lab)
            prev_cls, prev_lab = cls, lab
    return info


def ref_category(cls: str, lab: str) -> str | None:
    tokens = set(lab.split())
    if tokens & REF_DEVOICED_MARKS:
        return "devoiced"
    if cls == "vowel_short" and lab in REF_VOICED_HIGH:
        return "voiced"
    return None


def hyp_category(error_type: str, hyp_label: str, ref_lab: str) -> str | None:
    if error_type == "omission":
        return "devoiced"
    labels = set(hyp_label.split())
    if labels & VOICED_VOWELS:
        return "voiced"
    if labels & DEVOICED_VOWELS:
        return "devoiced"
    # 母音記号が無い: 参照が融合区間（子音＋母音）なら母音の削除、母音だけの単位なら判定しない
    return "devoiced" if len(ref_lab.split()) > 1 else None


def voiceless_context(prev_cls: str, prev_lab: str) -> bool:
    return prev_cls in VOICELESS_CONTEXT_CLASSES and not prev_lab.startswith(VOICED_OBSTRUENT_INITIALS)


class SystemStats:
    def __init__(self, name: str) -> None:
        self.name = name
        self.errors: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.types: dict[str, Counter] = defaultdict(Counter)
        # (講演, 群) → [10 ms 以内の数, 境界数]。群は T・C・ALL
        self.talk_group: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        # (講演, クラス, onset/offset) → [10 ms 以内の数, 境界数]。クラス別の区間に使う
        self.talk_class: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
        # (講演, 環境, 参照カテゴリ, 仮説カテゴリ) → 件数。環境は all・voiceless_ctx
        self.cat: Counter = Counter()
        self.cat_excluded = 0


def collect(system: str, info: dict[str, tuple[str, str, str, str]]) -> SystemStats:
    st = SystemStats(system)
    path = EVAL / system / "alignment.tsv"
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            et, cls, talk = r["error_type"], r["phone_class"], r["file_id"]
            st.types[cls if et != "insertion" else f"{cls} (hyp)"][et] += 1

            if et in MATCHED:
                group = "T" if cls in TARGET else "C" if cls in CONTROL else None
                for bd, col in (("onset", "onset_error_ms"), ("offset", "offset_error_ms")):
                    e = float(r[col])
                    st.errors[(cls, bd)].append(e)
                    hit = int(abs(e) <= 10)
                    tc = st.talk_class[(talk, cls, bd)]
                    tc[0] += hit
                    tc[1] += 1
                    for g in filter(None, (group, "ALL")):
                        c = st.talk_group[(talk, g)]
                        c[0] += hit
                        c[1] += 1

            uid = r["ref_unit_id"]
            if uid and uid in info:
                rcls, rlab, pcls, plab = info[uid]
                rc = ref_category(rcls, rlab)
                if rc is None:
                    continue
                hc = hyp_category(et, r["hyp_label"], rlab)
                if hc is None:
                    st.cat_excluded += 1
                    continue
                st.cat[(talk, "all", rc, hc)] += 1
                if rc == "devoiced" or voiceless_context(pcls, plab):
                    st.cat[(talk, "voiceless_ctx", rc, hc)] += 1
    return st


def pooled_rate(counts: dict, keys) -> float:
    h = n = 0
    for k in keys:
        c = counts.get(k)
        if c:
            h += c[0]
            n += c[1]
    return 100.0 * h / n if n else math.nan


def rate(tg: dict, talks: list[str], group: str) -> float:
    return pooled_rate(tg, ((t, group) for t in talks))


def class_rate(st: SystemStats, talks: list[str], cls: str, bd: str) -> float:
    return pooled_rate(st.talk_class, ((t, cls, bd) for t in talks))


def cat_metrics(cat: Counter, talks: list[str], env: str) -> dict[str, float]:
    tp = fn = tn = fp = 0
    for t in talks:
        tp += cat.get((t, env, "devoiced", "devoiced"), 0)
        fn += cat.get((t, env, "devoiced", "voiced"), 0)
        tn += cat.get((t, env, "voiced", "voiced"), 0)
        fp += cat.get((t, env, "voiced", "devoiced"), 0)
    sens = tp / (tp + fn) if tp + fn else math.nan
    spec = tn / (tn + fp) if tn + fp else math.nan
    return {
        "n_ref_devoiced": tp + fn, "n_ref_voiced": tn + fp,
        "sensitivity": 100 * sens, "specificity": 100 * spec,
        "balanced_accuracy": 100 * (sens + spec) / 2,
    }


def bootstrap(fn, talks: list[str]) -> tuple[float, float, float]:
    """評価用講演を単位とするクラスタブートストラップ（D5）。点推定と 95% パーセンタイル区間。"""
    rng = random.Random(SEED)
    point = fn(talks)
    draws = sorted(fn([rng.choice(talks) for _ in talks]) for _ in range(N_BOOT))
    draws = [d for d in draws if not math.isnan(d)]
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return point, lo, hi


def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.3f}" if isinstance(v, float) else v for v in row])


def cmd_summarize(args) -> int:
    talks = test_talks()
    info = load_ref_info()
    stats = {s: collect(s, info) for s in args.systems if (EVAL / s / "alignment.tsv").exists()}
    missing = [s for s in args.systems if s not in stats]
    if missing:
        print(f"[warn] alignment.tsv が無いので飛ばす: {missing}", file=sys.stderr)

    class_rows = []
    for s, st in stats.items():
        for (cls, bd), errs in sorted(st.errors.items()):
            absd = [abs(e) for e in errs]
            class_rows.append([
                s, cls, bd, len(errs), statistics.median(absd), statistics.fmean(absd),
                statistics.fmean(errs),
                *[100.0 * sum(a <= t for a in absd) / len(absd) for t in THRESHOLDS],
            ])
    write_tsv(
        RESULTS / "pilot_class_summary.tsv",
        ["system", "phone_class", "boundary", "n", "median_abs_ms", "mean_abs_ms", "mean_signed_ms",
         *[f"within{t}" for t in THRESHOLDS]],
        class_rows,
    )

    type_rows = []
    for s, st in stats.items():
        for cls, cnt in sorted(st.types.items()):
            type_rows.append([s, cls, *(cnt[k] for k in ("displacement", "substitution", "omission", "insertion"))])
    write_tsv(RESULTS / "pilot_error_types.tsv",
              ["system", "phone_class", "displacement", "substitution", "omission", "insertion"], type_rows)

    cat_rows = []
    for s, st in stats.items():
        for env in ("all", "voiceless_ctx"):
            m = cat_metrics(st.cat, talks, env)
            cat_rows.append([s, env, m["n_ref_devoiced"], m["n_ref_voiced"], m["sensitivity"],
                             m["specificity"], m["balanced_accuracy"], st.cat_excluded if env == "all" else ""])
    write_tsv(RESULTS / "pilot_devoicing_categorical.tsv",
              ["system", "context", "n_ref_devoiced", "n_ref_voiced", "sensitivity", "specificity",
               "balanced_accuracy", "excluded"], cat_rows)

    contrast_rows = []
    print("\n== 10 ms 以内率（onset・offset 併合）")
    for s, st in stats.items():
        print(f"  {s:<3} ALL {rate(st.talk_group, talks, 'ALL'):6.2f}  "
              f"T {rate(st.talk_group, talks, 'T'):6.2f}  C {rate(st.talk_group, talks, 'C'):6.2f}")

    def diff(a: str, b: str, group: str):
        return lambda ts: rate(stats[b].talk_group, ts, group) - rate(stats[a].talk_group, ts, group)

    if "A" in stats and "B" in stats:
        did = lambda ts: diff("A", "B", "T")(ts) - diff("A", "B", "C")(ts)  # noqa: E731
        for label, fn in (("(B-A)_T", diff("A", "B", "T")), ("(B-A)_C", diff("A", "B", "C")),
                          ("DiD (B-A)_T-(B-A)_C  [D5 主要]", did)):
            contrast_rows.append(["A vs B", label, *bootstrap(fn, talks)])
        for env in ("all", "voiceless_ctx"):
            for metric in ("sensitivity", "balanced_accuracy"):
                fn = (lambda e, m: lambda ts: cat_metrics(stats["B"].cat, ts, e)[m]
                      - cat_metrics(stats["A"].cat, ts, e)[m])(env, metric)
                contrast_rows.append(["A vs B", f"devoicing {metric} B-A ({env})", *bootstrap(fn, talks)])
    if "B0" in stats and "B1" in stats:
        for group in ("ALL", "T", "C"):
            contrast_rows.append(["B0 vs B1", f"(B1-B0)_{group} within10", *bootstrap(diff("B0", "B1", group), talks)])

    write_tsv(RESULTS / "pilot_contrasts.tsv", ["comparison", "quantity", "estimate", "ci_low", "ci_high"],
              contrast_rows)
    if contrast_rows:
        print("\n== 対比（評価用講演クラスタブートストラップ 2,000 回、95% 区間）")
        for comp, q, est, lo, hi in contrast_rows:
            print(f"  {comp:<9} {q:<45} {est:+7.2f} [{lo:+7.2f}, {hi:+7.2f}]")
    print("\n== 無声化の範疇判定（D6）")
    for row in cat_rows:
        s, env, nd, nv, sens, spec, ba, exc = row
        print(f"  {s:<3} {env:<14} devoiced n={nd:<6} voiced n={nv:<6} "
              f"sens {sens:6.2f}  spec {spec:6.2f}  BA {ba:6.2f}  excluded {exc}")
    print(f"\nwrote {RESULTS}/pilot_*.tsv")
    return 0


def cmd_replicates(args) -> int:
    """学習の反復に対する頑健性（docs/decisions-log.md D11）。"""
    talks = test_talks()
    info = load_ref_info()
    stats = {s: collect(s, info) for s in args.a + args.b}

    def t_rate(s: str, ts: list[str]) -> float:
        return rate(stats[s].talk_group, ts, "T")

    def c_rate(s: str, ts: list[str]) -> float:
        return rate(stats[s].talk_group, ts, "C")

    def sens(s: str, ts: list[str]) -> float:
        return cat_metrics(stats[s].cat, ts, "all")["sensitivity"]

    def bar_diff(value) -> callable:
        """条件Bの平均 − 条件Aの平均。value(system, talks) を受け取る。"""
        return lambda ts: (statistics.fmean(value(b, ts) for b in args.b)
                           - statistics.fmean(value(a, ts) for a in args.a))

    def did_bar(ts: list[str]) -> float:
        return bar_diff(t_rate)(ts) - bar_diff(c_rate)(ts)

    def fused(bd: str):
        return lambda s, ts: class_rate(stats[s], ts, "devoiced_fused", bd)

    model_rows = []
    print("== モデルごと（10 ms 以内率、評価用講演全体）")
    for cond, names in (("A", args.a), ("B", args.b)):
        for s in names:
            row = [cond, s, t_rate(s, talks), c_rate(s, talks),
                   fused("onset")(s, talks), fused("offset")(s, talks), sens(s, talks)]
            model_rows.append(row)
            print(f"  {cond} {s:<6} T {row[2]:6.2f}  C {row[3]:6.2f}  "
                  f"devoiced_fused onset {row[4]:6.2f} offset {row[5]:6.2f}  無声化の感度 {row[6]:6.2f}")
    write_tsv(RESULTS / "pilot_replicates_models.tsv",
              ["condition", "system", "T_within10", "C_within10", "devoiced_fused_onset_within10",
               "devoiced_fused_offset_within10", "devoicing_sensitivity"], model_rows)

    print("\n== 学習のばらつき（条件内の標準偏差）")
    for cond in ("A", "B"):
        rows = [r for r in model_rows if r[0] == cond]
        if len(rows) > 1:
            print(f"  {cond}: T {statistics.stdev(r[2] for r in rows):.2f}  C {statistics.stdev(r[3] for r in rows):.2f}  "
                  f"onset {statistics.stdev(r[4] for r in rows):.2f}  offset {statistics.stdev(r[5] for r in rows):.2f}  "
                  f"感度 {statistics.stdev(r[6] for r in rows):.2f}")
        else:
            print(f"  {cond}: モデルが1つなので算出しない")

    pair_rows = []
    for a in args.a:
        for b in args.b:
            d = (t_rate(b, talks) - t_rate(a, talks)) - (c_rate(b, talks) - c_rate(a, talks))
            pair_rows.append([a, b, d, sens(b, talks) - sens(a, talks)])
    write_tsv(RESULTS / "pilot_replicates_pairs.tsv",
              ["A_system", "B_system", "DiD_within10", "sensitivity_B_minus_A"], pair_rows)
    print(f"\n== 全{len(pair_rows)}組")
    for a, b, d, sdiff in pair_rows:
        print(f"  {a:<6} vs {b:<6} 差の差 {d:+6.2f}  感度 B−A {sdiff:+6.2f}")

    did_est = bootstrap(did_bar, talks)
    sens_est = bootstrap(bar_diff(sens), talks)
    onset_est = bootstrap(bar_diff(fused("onset")), talks)
    offset_est = bootstrap(bar_diff(fused("offset")), talks)
    all_pos = all(r[2] > 0 for r in pair_rows)
    all_neg_sens = all(r[3] < 0 for r in pair_rows)
    verdict = "支持を維持" if all_pos and did_est[1] > 0 else "学習のばらつきに対して頑健でない"
    sens_verdict = "逆向きの結果が反復した" if all_neg_sens and sens_est[2] < 0 else "逆向きの結果は反復しなかった"
    write_tsv(RESULTS / "pilot_replicates_contrasts.tsv", ["quantity", "estimate", "ci_low", "ci_high", "note"], [
        ["DiD_bar within10 [D11 主要]", *did_est, f"全組で正: {all_pos} → {verdict}"],
        ["devoicing sensitivity B_bar-A_bar", *sens_est, f"全組で B<A: {all_neg_sens} → {sens_verdict}"],
        ["devoiced_fused onset within10 B_bar-A_bar [探索]", *onset_est, ""],
        ["devoiced_fused offset within10 B_bar-A_bar [探索]", *offset_est, ""],
    ])
    print("\n== D11 の判定（評価用講演クラスタブートストラップ 2,000 回、95% 区間）")
    print(f"  Δ̄ = {did_est[0]:+.2f} [{did_est[1]:+.2f}, {did_est[2]:+.2f}]、全組で正: {all_pos} → {verdict}")
    print(f"  無声化の感度 B̄−Ā = {sens_est[0]:+.2f} [{sens_est[1]:+.2f}, {sens_est[2]:+.2f}]、"
          f"全組で B<A: {all_neg_sens} → {sens_verdict}")
    print(f"  探索: devoiced_fused onset B̄−Ā = {onset_est[0]:+.2f} [{onset_est[1]:+.2f}, {onset_est[2]:+.2f}]、"
          f"offset B̄−Ā = {offset_est[0]:+.2f} [{offset_est[1]:+.2f}, {offset_est[2]:+.2f}]")
    print(f"\nwrote {RESULTS}/pilot_replicates_*.tsv")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare-ref")
    a = sub.add_parser("align")
    a.add_argument("--system", required=True)
    a.add_argument("--textgrids", type=Path)
    a.add_argument("--units", type=Path)
    s = sub.add_parser("summarize")
    s.add_argument("--systems", nargs="+", default=["B0", "B1", "A", "B"])
    r = sub.add_parser("replicates")
    r.add_argument("--a", nargs="+", required=True, help="条件Aの系統名（例: A A_r2 A_r3）")
    r.add_argument("--b", nargs="+", required=True, help="条件Bの系統名（例: B B_r2 B_r3）")
    args = ap.parse_args(argv)
    handlers = {"prepare-ref": cmd_prepare_ref, "align": cmd_align, "summarize": cmd_summarize,
                "replicates": cmd_replicates}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
