#!/usr/bin/env python3
"""連音化の一致率を、対立が何の違いかで分けて見る（2026-09-10 新設）。

連音化は全過程中もっとも一致率が低い（38.4%、ベースライン比 −57.8pt）。
説明の当否を確かめる。korean_mfa は有声性を素性として持たない。連音化の2仮説は
多くの場合 [t̚] 対 [d] のように**閉鎖の解放と有声性**で分かれる。もしこの説明が
正しいなら、ɭ 対 ɾ のような調音様式の違いで分かれる対では一致率が高いはずである。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seoul_compare as SC   # noqa: E402

# 有声性・解放だけで分かれる対（モデルが素性として持たない次元）
VOICING = {("p̚", "b"), ("t̚", "d"), ("k̚", "ɡ"), ("t̚", "dʑ"), ("k̚", "g")}
# 調音様式・場所が動く対（モデルが素性として持つ次元）
MANNER = {("ɭ", "ɾ"), ("ɭ", "l")}


def kind(cit, san):
    a, b = cit.split(), san.split()
    if len(a) != len(b):
        return "音素数が動く"
    d = [(x, y) for x, y in zip(a, b) if x != y]
    if len(d) != 1:
        return "その他"
    if d[0] in VOICING:
        return "有声性・解放のみ"
    if d[0] in MANNER:
        return "調音様式（ɭ 対 ɾ）"
    return "その他"


def main():
    vp = {r["token"]: r for r in
          csv.DictReader((SC.RUN / "variant_pronunciations.csv").open(encoding="utf-8"))}
    rows = SC.load()
    g = defaultdict(list)
    for r in rows:
        v = vp.get(r["token"])
        if v:
            g[kind(v["citation"], v["sandhi"])].append(r)
    print(f"連音化 {len(rows):,} 件を対立の種類で分ける\n")
    for k in sorted(g, key=lambda k: -len(g[k])):
        sub = g[k]
        ag = sum(1 for r in sub if (r["mfa"] == "applied") == bool(r["human"])) / len(sub)
        hp = sum(r["human"] for r in sub) / len(sub)
        base = max(hp, 1 - hp)
        print(f"  {k:<18} n={len(sub):>5,}  一致 {ag:5.1%}  "
              f"多数派 {base:5.1%}（差 {100*(ag-base):+5.1f}pt）")
    print("\n※ korean_mfa は use_pitch/use_voicing/use_energy をいずれも false とする。")


if __name__ == "__main__":
    main()
