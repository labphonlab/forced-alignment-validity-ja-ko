"""どちらの発音が選ばれたかを判定する。

対象語区間の音素列を読み、G2P(form)（＝正書法どおり）と
G2P(original_form)（＝転写者が聴き取った変異形）のどちらと一致するかを見る。
一致＝MFAが転写者の判定に追随した、とみなす。
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from praatio import textgrid as tgio

MARKUP = re.compile(r"\{[^}]*\}|\([^)]*\)|&[^&]*&|[.,!?~'/]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="data/prepared.csv")
    ap.add_argument("--map", default="data/target_prons.csv")
    ap.add_argument("--aligned", default=".mfa/aligned")
    ap.add_argument("--out", default="data/scored.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.prepared)
    d = d[d.status == "ok"].merge(pd.read_csv(args.map), on=["form", "orig"], how="left")
    rows = []
    for r in d.itertuples(index=False):
        rec = {"utterance_id": r.utterance_id, "speaker_id": r.speaker_id,
               "form": r.form, "orig": r.orig, "category": r.category,
               "decidable": r.decidable, "chosen": "", "outcome": ""}
        tg_path = Path(args.aligned) / r.speaker_id / f"{r.utterance_id}.TextGrid"
        if not tg_path.exists():
            rec["outcome"] = "no_textgrid"; rows.append(rec); continue
        tg = tgio.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        wt = tg.getTier([n for n in tg.tierNames if "word" in n.lower()][0])
        pt = tg.getTier([n for n in tg.tierNames if "phone" in n.lower()][0])
        words = [iv for iv in wt.entries]
        lab = str(r.lab).split()
        tgt = None
        if r.word_index < len(words) and words[r.word_index].label == r.form:
            tgt = words[r.word_index]
        else:
            hits = [iv for iv in words if iv.label == r.form]
            if hits:
                tgt = hits[0]
        if tgt is None:
            rec["outcome"] = "word_not_found"; rows.append(rec); continue
        phones = " ".join(p.label for p in pt.entries
                          if p.start >= tgt.start - 1e-4 and p.end <= tgt.end + 1e-4)
        rec["chosen"] = phones
        if not r.decidable:
            rec["outcome"] = "undecidable"
        elif phones == r.pron_orig:
            rec["outcome"] = "followed_transcriber"
        elif phones == r.pron_form:
            rec["outcome"] = "chose_citation"
        else:
            rec["outcome"] = "neither"
        rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    dec = res[res.outcome.isin(["followed_transcriber", "chose_citation"])]
    print(res.outcome.value_counts().to_dict())
    print(f"\n判定できたトークン {len(dec)}\n")
    t = dec.groupby("category").outcome.value_counts().unstack(fill_value=0)
    t["n"] = t.sum(axis=1)
    t["detection_rate"] = (t.get("followed_transcriber", 0) / t["n"]).round(3)
    print(t.sort_values("detection_rate", ascending=False).to_string())


if __name__ == "__main__":
    main()
