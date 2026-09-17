# SCOPE: pilot（段階0–1）
"""学習済みモデルと整列結果が、コーパスを実際に使えているかを点検する（D10 の再発防止）。

言語を指定せずに日本語で学習すると、発話が分かち書きされず発話全体が1語の未知語になり、
その発話は学習から黙って外れる。2026-09-15 の初回は、22.68時間のうち0.85時間しか使われず、
それでも MFA はエラーを出さずにモデルを書き出した。ここでは件数と割合で止める。

    python3 scripts/check_model.py model work/models/B_pitch_voicing.zip --corpus work/corpus/train
    python3 scripts/check_model.py alignment work/align/B_pitch_voicing --corpus work/corpus/test

中身（書き起こし）は表示しない。件数・時間・割合だけを出す。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

INTERVAL = re.compile(r'xmin = ([0-9.eE+-]+)\s*\n\s*xmax = ([0-9.eE+-]+)\s*\n\s*text = "(.*)"')
PHONE_TIER = re.compile(r'name = "(?:[^"\n]* - )?phones"(.*?)(?=\n\s*item \[|\Z)', re.S)
NON_SPEECH_PHONES = {"spn"}


def corpus_stats(corpus: Path) -> tuple[int, int, float]:
    """(ファイル数, 文字のある発話区間の数, その合計秒)。"""
    files = sorted(corpus.glob("*.TextGrid"))
    n = 0
    dur = 0.0
    for tg in files:
        for s, e, text in INTERVAL.findall(tg.read_text(encoding="utf-8")):
            if text.strip():
                n += 1
                dur += float(e) - float(s)
    return len(files), n, dur


def check_model(args) -> int:
    with zipfile.ZipFile(args.model) as z:
        name = next(n for n in z.namelist() if n.endswith("meta.json"))
        meta = json.loads(z.read(name))
    training = meta.get("training", {})
    _, n_utt, dur = corpus_stats(args.corpus)
    utt_ratio = training.get("num_utterances", 0) / n_utt if n_utt else 0.0
    dur_ratio = training.get("audio_duration", 0.0) / dur if dur else 0.0

    print(f"model            : {args.model}")
    print(f"language         : {meta.get('language')}（期待値 {args.language}）")
    print(f"utterances used  : {training.get('num_utterances')} / {n_utt}（{100 * utt_ratio:.1f}%）")
    print(f"audio used       : {training.get('audio_duration', 0) / 3600:.2f} / {dur / 3600:.2f} 時間（{100 * dur_ratio:.1f}%）")
    feats = meta.get("features", {})
    print(f"use_pitch/voicing: {feats.get('use_pitch')} / {feats.get('use_voicing')}")

    problems = []
    if meta.get("language") != args.language:
        problems.append(f"言語が {meta.get('language')}（{args.language} であるべき）")
    if utt_ratio < args.min_ratio:
        problems.append(f"学習に使われた発話が {100 * utt_ratio:.1f}%（下限 {100 * args.min_ratio:.0f}%）")
    if dur_ratio < args.min_ratio:
        problems.append(f"学習に使われた音声が {100 * dur_ratio:.1f}%（下限 {100 * args.min_ratio:.0f}%）")
    return report(problems)


def check_alignment(args) -> int:
    n_files, _, _ = corpus_stats(args.corpus)
    outputs = sorted(args.alignment.glob("*.TextGrid"))
    n_phones = n_spn = 0
    for tg in outputs:
        m = PHONE_TIER.search(tg.read_text(encoding="utf-8"))
        if not m:
            continue
        for _, _, label in INTERVAL.findall(m.group(1)):
            label = label.strip()
            if not label or label == "sil":
                continue
            n_phones += 1
            n_spn += label in NON_SPEECH_PHONES
    spn_ratio = n_spn / n_phones if n_phones else 1.0

    print(f"alignment        : {args.alignment}")
    print(f"TextGrids        : {len(outputs)} / {n_files}")
    print(f"phone intervals  : {n_phones}（うち spn {n_spn}、{100 * spn_ratio:.2f}%）")

    problems = []
    if len(outputs) < n_files:
        problems.append(f"TextGrid が {n_files - len(outputs)} 本足りない")
    if spn_ratio > args.max_spn:
        problems.append(f"spn の割合が {100 * spn_ratio:.2f}%（上限 {100 * args.max_spn:.0f}%）")
    return report(problems)


def report(problems: list[str]) -> int:
    if problems:
        for p in problems:
            print(f"[NG] {p}", file=sys.stderr)
        return 1
    print("[OK]")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("model", type=Path)
    m.add_argument("--corpus", type=Path, required=True)
    m.add_argument("--language", default="japanese")
    m.add_argument("--min-ratio", type=float, default=0.9)
    a = sub.add_parser("alignment")
    a.add_argument("alignment", type=Path)
    a.add_argument("--corpus", type=Path, required=True)
    a.add_argument("--max-spn", type=float, default=0.05)
    args = ap.parse_args(argv)
    return check_model(args) if args.cmd == "model" else check_alignment(args)


if __name__ == "__main__":
    raise SystemExit(main())
