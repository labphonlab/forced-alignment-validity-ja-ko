#!/usr/bin/env python3
"""2仮説の「違いの種類」で一致率を整理する（2026-09-10 全面改訂）。

査読指摘への対応。5.1 の説明（モデルが素性として持つ次元でどれだけ離れているかで
成否が決まる）は、過程ごとの順位を**結果を見たあとで**説明したものだった。
過程名ではなく、**2仮説の音素列の違いそのもの**で分類すれば、過程をまたいで
同じ主張を確かめられる。連音化は内部で違いの種類が割れるので、
**同一過程内での対照**にもなる。

これは予測モデルではなく記述である。「事前に予測できる」と主張するには、
違いの大きさを連続量として定義し、hold-out で検証する必要がある。

2026-09-10 の改訂（査読対応・重要）: 当初は多数派ベースラインとの差だけを見ていたが、
**この指標は人手側の出現比に影響される**。4.2 で生の一致率を退けた理由と同じものが
ここにも当てはまる。

ただし均衡正解率・MCC はこの層別では使えない。**陰性（規則が適用されなかった事例）が
群によって2〜97件しかない**からである（解放・有声性群は陰性2件）。特異度がその数件に
支配され、均衡正解率も MCC もそこを経由する。正準環境から標本を取る設計上の帰結であり、
層を細かくするほど効く。

そこで**感度（適用事例をどれだけ検出したか）**を主指標にする。出現比に依存せず、
分母は各群で数百あるので安定して推定できる。話者クラスタ・ブートストラップで
区間も出す。均衡正解率・MCC は参考として併記するが、陰性の数を必ず添える。
"""
import csv
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seoul_compare as SC        # noqa: E402
from phone_classes import IPA     # noqa: E402
from seoul_metrics import cells, metrics  # noqa: E402

SEED = 20260910
B = 2000


def sens_ci(rows):
    """感度の話者クラスタ・ブートストラップ95%区間。"""
    by = defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    spk = list(by)
    rng = random.Random(SEED)
    v = []
    for _ in range(B):
        s = [x for k in (rng.choice(spk) for _ in spk) for x in by[k]]
        tp, fn, _, _ = cells(s)
        if tp + fn:
            v.append(tp / (tp + fn))
    v.sort()
    return v[int(.025 * len(v))], v[int(.975 * len(v))]

OBSTRUENT = {"P", "T", "K", "S", "C"}
# 喉頭素性のみで対立する組（平音・濃音・激音・有声異音）は、同じ場所類に写る。
# 有声性・解放のみの組（終声の未解放音 対 語中の有声音）を別に取る。
RELEASE = {("p̚", "b"), ("t̚", "d"), ("k̚", "ɡ"), ("t̚", "dʑ"), ("k̚", "ɟ"),
           ("p̚", "bʲ"), ("k̚", "ɡʷ"), ("t̚", "d"), ("p̚", "bʷ")}


def kind(cit, san):
    a, b = cit.split(), san.split()
    if len(a) != len(b):
        return "分節数が動く"
    d = [(x, y) for x, y in zip(a, b) if x != y]
    if not d:
        return "対立なし"
    if len(d) > 1:
        return "2箇所以上動く"
    x, y = d[0]
    if (x, y) in RELEASE or (y, x) in RELEASE:
        return "解放・有声性のみ"
    cx, cy = IPA.get(x), IPA.get(y)
    if cx is None or cy is None:
        return "その他"
    if cx == cy and cx in OBSTRUENT:
        return "喉頭素性のみ"
    return "調音様式・場所が動く"


def main():
    vp = {r["token"]: r for r in
          csv.DictReader((SC.RUN / "variant_pronunciations.csv").open(encoding="utf-8"))}
    rows = [r for r in SC.load() if r["token"] in vp]
    g = defaultdict(list)
    for r in rows:
        v = vp[r["token"]]
        g[kind(v["citation"], v["sandhi"])].append(r)

    def block(title, groups):
        print(f"\n【{title}】")
        print(f"  {'違いの種類':<16}{'n':>7}{'感度':>8}{'95%CI':>16}"
              f"{'陰性数':>8}{'均衡正解率':>10}{'MCC':>8}")
        for k in sorted(groups, key=lambda k: -len(groups[k])):
            sub = groups[k]
            if len(sub) < 30:
                continue
            m = metrics(sub)
            lo, hi = sens_ci(sub)
            neg = m["fp"] + m["tn"]
            bal = f"{m['bal']:.1%}" if m["bal"] == m["bal"] else "—"
            mcc = f"{m['mcc']:.3f}" if m["mcc"] == m["mcc"] else "—"
            print(f"  {k:<16}{m['n']:>7,}{m['sens']:>8.1%}"
                  f"{f'[{lo:.1%}, {hi:.1%}]':>16}{neg:>8}{bal:>10}{mcc:>8}")

    block(f"全過程 n={len(rows):,}", g)

    li = defaultdict(list)
    for r in rows:
        if r["cat"] == "liaison":
            li[kind(vp[r["token"]]["citation"], vp[r["token"]]["sandhi"])].append(r)
    block(f"連音化のみ n={sum(len(v) for v in li.values()):,}（同一過程内の対照）", li)

    print("\n※ 主指標は感度。均衡正解率と MCC は**陰性数が2〜97件しかないため**"
          "\n  この層別では読めない（正準環境から標本を取る設計の帰結）。参考値である。")
    print("※ korean_mfa の素性設定は配布モデルの meta.json で確認できる"
          "\n  （type: mfcc / use_pitch: False / use_voicing: False / use_energy: False）。"
          "\n  F0・有声性・エネルギーを**明示的な入力素性として与えていない**という意味であり、"
          "\n  MFCC が間接的に関連情報を含む可能性は否定しない。")


if __name__ == "__main__":
    main()
