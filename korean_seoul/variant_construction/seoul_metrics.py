#!/usr/bin/env python3
"""範疇軸の過程別指標と、話者を単位にした再現性（2026-09-10 新設）。

査読指摘への対応。
  (a) 人手適用率が 95.5% と極端に偏るので、生の一致率だけでは足りない。
      感度・特異度・均衡正解率・MCC を併記する。
  (b) 4,811 トークンは 40 話者から来ており、同一語も繰り返される。独立観測ではない
      ので McNemar の p は過大に小さい。**話者を単位に方向の再現性**を数え、
      話者クラスタ・ブートストラップで区間を出す。
"""
import random
import statistics as st
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seoul_compare import load   # noqa: E402

SEED = 20260910
B = 2000


def cells(g):
    tp = sum(1 for r in g if r["human"] == 1 and r["mfa"] == "applied")
    fn = sum(1 for r in g if r["human"] == 1 and r["mfa"] == "notapplied")
    fp = sum(1 for r in g if r["human"] == 0 and r["mfa"] == "applied")
    tn = sum(1 for r in g if r["human"] == 0 and r["mfa"] == "notapplied")
    return tp, fn, fp, tn


def metrics(g):
    tp, fn, fp, tn = cells(g)
    n = tp + fn + fp + tn
    ag = (tp + tn) / n
    hp = (tp + fn) / n
    base = max(hp, 1 - hp)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    bal = (sens + spec) / 2 if tp + fn and tn + fp else float("nan")
    den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else float("nan")
    return dict(n=n, hp=hp, mfa=(tp + fp) / n, ag=ag, base=base, diff=100 * (ag - base),
                sens=sens, spec=spec, bal=bal, mcc=mcc, tp=tp, fn=fn, fp=fp, tn=tn)


def boot_ci(rows, stat, b=B):
    """話者クラスタ・ブートストラップ。話者ごと丸ごと再抽出する。"""
    by = defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    spk = list(by)
    rng = random.Random(SEED)
    vals = []
    for _ in range(b):
        s = [x for k in (rng.choice(spk) for _ in spk) for x in by[k]]
        try:
            vals.append(stat(s))
        except (ZeroDivisionError, ValueError):
            pass
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def main():
    rows = load()
    cats = sorted({r["cat"] for r in rows})
    print(f"n={len(rows):,} / 話者 {len({r['speaker'] for r in rows})} 名\n")
    hdr = (f"{'過程':<24}{'n':>6}{'人手':>7}{'MFA':>7}{'一致':>7}{'基準':>7}"
           f"{'差':>7}{'感度':>7}{'特異度':>8}{'均衡':>7}{'MCC':>7}")
    print(hdr)
    print("-" * len(hdr))
    for cat in cats + ["全体"]:
        g = rows if cat == "全体" else [r for r in rows if r["cat"] == cat]
        m = metrics(g)
        print(f"{cat:<24}{m['n']:>6,}{m['hp']:>7.1%}{m['mfa']:>7.1%}{m['ag']:>7.1%}"
              f"{m['base']:>7.1%}{m['diff']:>+7.1f}{m['sens']:>7.1%}{m['spec']:>8.1%}"
              f"{m['bal']:>7.1%}{m['mcc']:>7.3f}")

    if "--markdown" in sys.argv:
        # 2026-09-11: JAKLE 表1 と揃える。Hand と Baseline は全過程で一致する（多数派が全過程で
        # 適用）ので1列に統合し、陰性数と過小:過大を加える。行は基準との差が小さい順（悪い順）。
        EN = {"tensification": "Tensification",
              "nasalization_obstruent": "Nasalization (obstr.)",
              "aspiration": "Aspiration", "liquidization": "Liquidization",
              "nasalization_liquid": "Nasalization (liquid)",
              "liaison": "Liaison", "全体": "All"}
        print("\n<!-- 原稿貼り付け用 -->\n")
        print("| Process | *n* | Hand applied (= baseline) | Aligner applied | Agreement | Diff. "
              "| Sens. | Spec. | Bal. acc. | MCC | Negatives | Under : over |")
        print("|" + "---|" * 12)
        order = sorted(cats, key=lambda c: metrics([r for r in rows if r["cat"] == c])["diff"])
        for cat in order + ["全体"]:
            g = rows if cat == "全体" else [r for r in rows if r["cat"] == cat]
            m = metrics(g)
            cells = [EN[cat], f"{m['n']:,}", f"{m['hp']:.1%}", f"{m['mfa']:.1%}", f"{m['ag']:.1%}",
                     f"{m['diff']:+.1f}".replace("-", "\u2212"), f"{m['sens']:.1%}",
                     f"{m['spec']:.1%}", f"{m['bal']:.1%}",
                     f"{m['mcc']:.3f}".replace("-", "\u2212"), f"{m['fp'] + m['tn']:,}",
                     f"{m['fn']:,} : {m['fp']:,}"]
            if cat == "全体":
                cells = [f"**{c}**" for c in cells]
            print("| " + " | ".join(cells) + " |")

    print("\n【話者を単位にした再現性】")
    same = tot = 0
    diffs = []
    for s in sorted({r["speaker"] for r in rows}):
        g = [r for r in rows if r["speaker"] == s]
        tp, fn, fp, tn = cells(g)
        tot += 1
        if fn > fp:
            same += 1
        diffs.append(fn - fp)
    print(f"  「MFA非適用/人手適用」が逆方向を上回る話者: {same}/{tot} 名")
    print(f"  話者ごとの差(見落とし−逆)の中央値: {st.median(diffs):.1f} 件")

    ci = boot_ci(rows, lambda s: metrics(s)["ag"])
    cd = boot_ci(rows, lambda s: metrics(s)["diff"])
    print(f"\n【話者クラスタ・ブートストラップ（{B:,}回）】")
    print(f"  一致率 {metrics(rows)['ag']:.1%}  95%CI [{ci[0]:.1%}, {ci[1]:.1%}]")
    print(f"  基準との差 {metrics(rows)['diff']:+.1f}pt  95%CI [{cd[0]:+.1f}, {cd[1]:+.1f}]")
    print("\n※ 話者を単位に取り直しても区間は基準を下回る側に収まる。"
          "\n  トークン単位の p 値ではなく、この再現性を主要な証拠とする。")


if __name__ == "__main__":
    main()
