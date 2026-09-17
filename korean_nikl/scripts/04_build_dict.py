"""変異形競合辞書を作る。

対象語の form 綴りに、G2P(form) と G2P(original_form) の**2発音だけ**を与える。
2つが同一音素列になる対は判定不能なので除外し、その件数も報告する
（判定不能を黙って混ぜると検出率が水増しされる）。
"""
import argparse
import collections
from pathlib import Path

import pandas as pd


def load(path):
    ent = collections.defaultdict(list)
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            ent[parts[0]].append(parts[-1])
    return ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2p", default="data/g2p_all.dict")
    ap.add_argument("--prepared", default="data/prepared.csv")
    ap.add_argument("--out", default="data/competition.dict")
    ap.add_argument("--map", default="data/target_prons.csv")
    args = ap.parse_args()

    g2p = load(args.g2p)
    d = pd.read_csv(args.prepared)
    d = d[d.status == "ok"].copy()

    ent = {w: list(dict.fromkeys(p)) for w, p in g2p.items()}
    rows, undecidable, missing = [], 0, 0
    for (form, orig), _ in d.groupby(["form", "orig"]):
        pf = g2p.get(form, [None])[0]
        po = g2p.get(orig, [None])[0]
        if not pf or not po:
            # G2P が発音を返さなかった対。判定不能として明示的に残す
            # （黙って落とすと NaN が「どちらでもない」に化ける）
            missing += 1
            rows.append({"form": form, "orig": orig, "pron_form": pf or "",
                         "pron_orig": po or "", "decidable": False})
            continue
        if pf == po:  # G2P が差を潰す対
            undecidable += 1
            rows.append({"form": form, "orig": orig, "pron_form": pf,
                         "pron_orig": po, "decidable": False})
            continue
        ent[form] = [pf, po]
        rows.append({"form": form, "orig": orig, "pron_form": pf,
                     "pron_orig": po, "decidable": True})

    with open(args.out, "w", encoding="utf-8") as f:
        for w in sorted(ent):
            for p in ent[w]:
                f.write(f"{w}\t{p}\n")
    m = pd.DataFrame(rows)
    m.to_csv(args.map, index=False)
    print(f"対象対 {len(m)}  判定可能 {int(m.decidable.sum())}  "
          f"G2Pが同一で判定不能 {undecidable}  発音欠落 {missing}")
    if undecidable:
        print("\n判定不能の例:")
        print(m[~m.decidable].head(8).to_string(index=False))
    print(f"\n辞書エントリ {len(ent):,} -> {args.out}")


if __name__ == "__main__":
    main()
