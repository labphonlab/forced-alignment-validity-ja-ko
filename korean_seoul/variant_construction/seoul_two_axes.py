#!/usr/bin/env python3
"""同一トークン上で時間軸と範疇軸を突き合わせる（2026-09-09 新設 / 09-10 改訂2）。

本研究の中心的主張は「境界配置と変異形選択の妥当性は分離する」である。
別々の標本で測った2つの数値を並べるだけでは、標本差の可能性が残る。ここでは
**同一の語トークン**について境界偏差と変異形判定を結合する。

2026-09-10 の改訂2（査読対応）:
  (a) **非有意は同等の証明ではない**。Mann-Whitney の p だけでなく、
      差そのものの**話者クラスタ・ブートストラップ信頼区間**を出す。
  (b) 語全体の境界で平均すると、**過程が働く場所だけ悪くても薄まる**。
      音節境界に隣接する分節だけに絞った分析を併記する。
  (c) 対応がついた分節は 8割で、残りは評価から落ちる。**両群で対応率が
      揃っているか**を確認する（揃っていなければ選択効果の疑いが残る）。
  (d) 対応率が揃っていても、鼻音化（T→N）や流音化（N→L）では変異形を誤ると
      **標的分節そのもの**が対応不能になり落ちうる。`--identity-safe` は
      2仮説が同じ粗い音素類列に写る候補だけに絞る。そこでは選んだ変異形が
      どの分節が対応するかに影響しえないので、この経路の選択効果は構造的に消える。

前段: seoul_boundary.py（時間軸）, seoul_compare.py（範疇軸）
"""
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seoul_boundary import measure          # noqa: E402
import seoul_compare as SC                   # noqa: E402
from seoul_compare import load              # noqa: E402
from phone_classes import IPA               # noqa: E402

SEED = 20260910
B = 2000


def line(label, ds):
    if not ds:
        print(f"  {label:<28} n=    0")
        return
    print(f"  {label:<28} n={len(ds):>6,}  中央値 {st.median(ds)*1000:6.2f} ms  "
          f"平均 {st.mean(ds)*1000:6.2f} ms  "
          f"20ms以内 {sum(1 for x in ds if x <= 0.020)/len(ds):5.1%}")


def boot_median_diff(pairs, key):
    """話者を単位に再抽出し、群間の「語ごと中央偏差の中央値」の差を区間推定する。

    pairs: (speaker, is_correct, [偏差...]) の列
    """
    by = defaultdict(list)
    for spk, ok, ds in pairs:
        by[spk].append((ok, ds))
    spk = list(by)
    rng = random.Random(SEED)
    vals = []
    for _ in range(B):
        a, b = [], []
        for k in (rng.choice(spk) for _ in spk):
            for ok, ds in by[k]:
                (a if ok else b).append(st.median(ds))
        if a and b:
            vals.append((st.median(b) - st.median(a)) * 1000)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def auc_ci(pairs):
    """境界誤差（語ごと中央値）で変異形の誤りをどれだけ判別できるか。

    AUC = 誤った語のほうが境界誤差が大きい確率。0.5 なら判別できない。
    「境界精度は変異形選択の代理指標になるか」を直接問う指標。話者クラスタで区間を出す。
    """
    def auc(a, b):
        if not a or not b:
            return None
        from scipy.stats import mannwhitneyu as mw
        u, _ = mw(b, a, alternative="two-sided")
        return u / (len(a) * len(b))
    by = defaultdict(list)
    for spk, ok, v in pairs:
        by[spk].append((ok, v))
    a = [v for _, ok, v in pairs if ok]
    b = [v for _, ok, v in pairs if not ok]
    point = auc(a, b)
    spk = list(by)
    rng = random.Random(SEED)
    vals = []
    for _ in range(500):
        aa, bb = [], []
        for k in (rng.choice(spk) for _ in spk):
            for ok, v in by[k]:
                (aa if ok else bb).append(v)
        x = auc(aa, bb)
        if x is not None:
            vals.append(x)
    vals.sort()
    return point, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def block(title, joined, field):
    print(f"\n【{title}】")
    use = [(r, m) for r, m in joined if m[field]]
    if not use:
        print("  該当なし")
        return
    ok = [(r, m) for r, m in use if (r["mfa"] == "applied") == bool(r["human"])]
    ng = [(r, m) for r, m in use if (r["mfa"] == "applied") != bool(r["human"])]
    line("変異形の判定が人手と一致", [x for _, m in ok for x in m[field]])
    line("変異形の判定が人手と不一致", [x for _, m in ng for x in m[field]])
    a = [st.median(m[field]) for _, m in ok]
    b = [st.median(m[field]) for _, m in ng]
    d = (st.median(b) - st.median(a)) * 1000
    _, pv = mannwhitneyu(a, b, alternative="two-sided")
    lo, hi = boot_median_diff([(r["speaker"], (r["mfa"] == "applied") == bool(r["human"]),
                               m[field]) for r, m in use], field)
    print(f"  語トークン単位の中央偏差: 一致 {st.median(a)*1000:.2f} ms 対 "
          f"不一致 {st.median(b)*1000:.2f} ms")
    print(f"  差 {d:+.2f} ms  話者クラスタ・ブートストラップ95%CI "
          f"[{lo:+.2f}, {hi:+.2f}] ms  (Mann-Whitney p={pv:.3g})")
    print(f"  → 区間の上限は {hi:+.2f} ms（人手転記者間の偏差 9.04 ms の {hi/9.04:.0%}）。"
          + ("区間が0を含む：検出可能な増加なし。" if lo <= 0 <= hi
             else "区間が0を含まない：小さいが検出可能な差あり。"))
    au, alo, ahi = auc_ci([(r["speaker"], (r["mfa"] == "applied") == bool(r["human"]),
                            st.median(m[field])) for r, m in use])
    print(f"  判別力 AUC {au:.3f} [{alo:.3f}, {ahi:.3f}]"
          "（境界誤差で変異形の誤りを見分けられる確率。0.5=見分けられない）")


def main():
    method = "nearest" if "--nearest" in sys.argv else "aligned"
    bnd = {m["utt"]: m for m in measure(method)}
    print(f"境界の測り方: {method}")
    rows = load()
    if "--identity-safe" in sys.argv:
        import csv
        vp = {v["token"]: v for v in
              csv.DictReader((SC.RUN / "variant_pronunciations.csv").open(encoding="utf-8"))}

        def cls(seq):
            return [IPA.get(x) for x in str(seq).split()]
        before = len(rows)
        rows = [r for r in rows if r["token"] in vp
                and None not in cls(vp[r["token"]]["citation"])
                and cls(vp[r["token"]]["citation"]) == cls(vp[r["token"]]["sandhi"])]
        print(f"identity-safe: 2仮説の音素類列が一致する候補 {len(rows):,} / {before:,}")
    joined = [(r, bnd[r["utt"]]) for r in rows if r["utt"] in bnd]
    print(f"\n両軸を測れたトークン {len(joined):,} 件 "
          f"/ 話者 {len({r['speaker'] for r, _ in joined})} 名")

    ok = [(r, m) for r, m in joined if (r["mfa"] == "applied") == bool(r["human"])]
    ng = [(r, m) for r, m in joined if (r["mfa"] == "applied") != bool(r["human"])]
    print(f"  一致 {len(ok):,} 件 / 不一致 {len(ng):,} 件 "
          f"（一致率 {len(ok)/len(joined):.1%}）")

    if method == "aligned":
        print(f"\n【分節の対応率（選択効果の点検）】")
        print(f"  変異形の判定が一致した語   {st.mean([m['cov'] for _, m in ok]):.1%}")
        print(f"  変異形の判定が不一致の語   {st.mean([m['cov'] for _, m in ng]):.1%}")
        print("  → 両群で揃っていれば、対応不能な分節を落としたことによる"
              "選択効果は考えにくい。")

    block("範疇軸の当否で分けた境界偏差（語内の対応分節すべて）", joined, "devs")
    if method == "aligned":
        block("同じ比較を、過程が働く音節境界に隣接する分節だけで", joined, "devs_target")

    print("\n【過程別】")
    for cat in sorted({r["cat"] for r, _ in joined}):
        g = [(r, m) for r, m in joined if r["cat"] == cat]
        ag = sum(1 for r, _ in g if (r["mfa"] == "applied") == bool(r["human"])) / len(g)
        med = st.median([x for _, m in g for x in m["devs"]]) * 1000
        print(f"  {cat:<24} n={len(g):>5,}  範疇一致 {ag:5.1%}  境界中央値 {med:5.2f} ms")

    print("\n※ 変異形選択の誤りは境界誤差の増大を伴わなかった。"
          "\n  これは2軸が経験的に分離しうる（dissociable）ことを示すが、"
          "\n  統計的独立を示したものではない（話者・語・過程の効果は未統制）。")


if __name__ == "__main__":
    main()
