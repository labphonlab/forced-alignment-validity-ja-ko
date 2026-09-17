"""アライメント材料の作成。

各対象トークンについて、.lab では **form（正書法）綴りを置き**、辞書側に
G2P(form) と G2P(original_form) の2発音だけを与える。他のトークンは
original_form 綴りにして文脈の整列精度を上げる。
どちらの発音が選ばれたかが、MFA が転写者の判定に追随できたかの指標になる。
"""
import argparse
import os
import json
import re
import wave
from pathlib import Path

import pandas as pd

PCM_ROOT = Path(os.environ.get("NIKL_PCM_ROOT", "data/raw_nikl_pcm")).expanduser()
JSON_ROOT = Path(os.environ.get("NIKL_JSON_ROOT", "data/raw_nikl_json")).expanduser()
MARKUP = re.compile(r"\{[^}]*\}|\([^)]*\)|&[^&]*&|[.,!?~'/]")
FRAG = re.compile(r"-([가-힣]+)-")
_C = {}


def utt(uid):
    fid = uid.split(".")[0]
    if fid not in _C:
        data = json.load(open(JSON_ROOT / f"{fid}.json", encoding="utf-8"))
        _C[fid] = {u["id"]: u for d in data.get("document", []) for u in d.get("utterance", [])}
    return _C[fid][uid]


def clean_tok(t):
    t = FRAG.sub(r"\1", t)
    t = MARKUP.sub("", t).strip().strip("-")
    return t if t and all("가" <= c <= "힣" for c in t) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/pilot_sample.csv")
    ap.add_argument("--corpus", default=".mfa/corpus")
    ap.add_argument("--out", default="data/prepared.csv")
    args = ap.parse_args()

    d = pd.read_csv(args.sample)
    corpus = Path(args.corpus); corpus.mkdir(parents=True, exist_ok=True)
    rows, need = [], set()
    for r in d.itertuples(index=False):
        u = utt(r.utterance_id)
        toks = (u.get("original_form") or "").split()
        if r.word_index >= len(toks):
            rows.append({**r._asdict(), "status": "index_out_of_range"}); continue
        words = []
        bad = False
        for i, t in enumerate(toks):
            c = clean_tok(str(r.form) if i == r.word_index else t)
            if c is None:
                bad = True; break
            words.append(c)
        if bad:
            rows.append({**r._asdict(), "status": "markup"}); continue
        src = PCM_ROOT / r.utterance_id.split(".")[0] / f"{r.utterance_id}.pcm"
        if not src.exists():
            rows.append({**r._asdict(), "status": "pcm_missing"}); continue
        sd = corpus / r.speaker_id; sd.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        with wave.open(str(sd / f"{r.utterance_id}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(data)
        (sd / f"{r.utterance_id}.lab").write_text(" ".join(words) + "\n", encoding="utf-8")
        need.update(words); need.add(str(r.orig))
        rows.append({**r._asdict(), "status": "ok", "lab": " ".join(words)})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    Path("data/need_pron.txt").write_text("\n".join(sorted(need)) + "\n", encoding="utf-8")
    print(out.status.value_counts().to_dict())
    print(f"発音が必要な語型 {len(need):,} -> data/need_pron.txt")


if __name__ == "__main__":
    main()
