"""関門B用のパイロット標本。過程カテゴリごとに語彙タイプ均等で抽出する。

仮説: 強制アライメントは置換系（濃音化・母音上昇）の変異は選べるが、
削除・縮約系（音節脱落・終声脱落・語頭子音脱落）では転写者の判定に追随できない。
→ カテゴリ間の検出率差として検定する。
"""
import argparse
import collections
import random

import pandas as pd

CATS = ["tensification", "vowel_raising", "vowel_other",
        "syllable_deletion", "coda_deletion", "onset_liaison", "coda_addition"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=400)
    ap.add_argument("--min-speakers", type=int, default=5)
    ap.add_argument("--min-tokens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out", default="data/pilot_sample.csv")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    inv = pd.read_csv("data/pair_inventory.csv")
    ok = inv[(inv.n_speakers >= args.min_speakers) & (inv.n_tokens >= args.min_tokens)]
    # 変異形が複数ある form 型を除く。辞書に3つ以上の発音が入ると
    # 語型ごとに競合の選択肢数が変わり、偶然水準がカテゴリ間で揃わなくなる。
    nvar = ok.groupby("form").orig.nunique()
    single = set(nvar[nvar == 1].index)
    dropped = ok[~ok.form.isin(single)]
    ok = ok[ok.form.isin(single)]
    print(f"複数変異形の form 型を除外: 対 {len(dropped)} / 残り {len(ok)}")
    keep = {(r.form, r.orig) for r in ok.itertuples()}

    tok = pd.read_csv("data/discrepancy_tokens.csv")
    tok = tok[[(f, o) in keep for f, o in zip(tok.form, tok.orig)]]

    out = []
    for cat in CATS:
        sub = tok[tok.category == cat]
        if sub.empty:
            continue
        by_type = collections.defaultdict(list)
        for r in sub.itertuples(index=False):
            by_type[(r.form, r.orig)].append(r)
        for v in by_type.values():
            rng.shuffle(v)
        types = list(by_type)
        rng.shuffle(types)
        taken, used, spk = [], set(), collections.Counter()
        rank = 0
        while len(taken) < args.per_cat and rank < 50:
            progressed = False
            for t in sorted(types, key=lambda x: rng.random()):
                if len(taken) >= args.per_cat:
                    break
                pool = [r for r in by_type[t]
                        if (r.speaker_id, t) not in used and spk[r.speaker_id] <= rank]
                if not pool:
                    continue
                pick = min(pool, key=lambda r: (spk[r.speaker_id], rng.random()))
                taken.append(pick); used.add((pick.speaker_id, t)); spk[pick.speaker_id] += 1
                progressed = True
            if not progressed:
                break
            rank += 1
        df = pd.DataFrame(taken)
        out.append(df)
        print(f"{cat:20s} n={len(df):4d} types={df.groupby(['form','orig']).ngroups:4d} "
              f"speakers={df.speaker_id.nunique():4d}")

    res = pd.concat(out, ignore_index=True)
    res.to_csv(args.out, index=False)
    print(f"\ntotal n={len(res)} utterances={res.utterance_id.nunique()} -> {args.out}")


if __name__ == "__main__":
    main()
