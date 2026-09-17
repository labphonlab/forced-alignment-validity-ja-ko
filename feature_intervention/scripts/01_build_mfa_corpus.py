# SCOPE: pilot（段階0）
"""分割表の講演から MFA 入力コーパスを組む。TextGrid の層名を話者IDにする。

mfa-ja-validation の既存コーパスは全講演の層名が `utt` だったため、MFA は183講演を
話者1名として扱い、CMVN と話者適応を講演・話者をまたいで推定していた
（../mfa-ja-validation/results/mfa_align_183.log「Found 1 speaker across 183 files」）。
ここでは層名を話者IDにして、MFA に本当の話者を渡す（docs/decisions-log.md D3）。

発話区間と入力テキストの作り方は既存と同じ（prepare_csj_corpus.ipu_segments を呼ぶ）。

    python3 scripts/01_build_mfa_corpus.py --split test --out work/corpus/test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT.parent / "mfa-ja-validation"
sys.path.insert(0, str(VALID / "src" / "align"))

from prepare_csj_corpus import ipu_segments, wav_duration  # noqa: E402

CSJ_DIR = VALID / "data" / "csj"


def write_textgrid(
    path: Path, segments: list[tuple[float, float, str]], total: float, tier: str
) -> None:
    """発話区間1層だけの TextGrid を書く。層名が MFA にとっての話者名になる。"""
    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for s, e, text in segments:
        s, e = max(s, cursor), min(e, total)
        if e <= s:
            continue
        if s > cursor:
            intervals.append((cursor, s, ""))
        intervals.append((s, e, text.replace('"', '""')))
        cursor = e
    if cursor < total:
        intervals.append((cursor, total, ""))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {total}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        f'        name = "{tier}"',
        "        xmin = 0",
        f"        xmax = {total}",
        f"        intervals: size = {len(intervals)}",
    ]
    for i, (s, e, text) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {s}",
            f"            xmax = {e}",
            f'            text = "{text}"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--split-table", type=Path, default=ROOT / "splits" / "csj_core_split.tsv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="先頭N講演のみ（動作確認用）")
    args = ap.parse_args(argv)

    with args.split_table.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]

    # 前回の実行の残り（講演数を変えて作り直した場合など）を消してから組む
    args.out.mkdir(parents=True, exist_ok=True)
    for old in list(args.out.glob("*.TextGrid")) + list(args.out.glob("*.wav")):
        old.unlink()

    n_talks = n_seg = 0
    speakers: set[str] = set()
    speech = 0.0
    for r in rows:
        xml = CSJ_DIR / f"{r['file_id']}.xml"
        wav = xml.with_suffix(".wav")
        if not (xml.exists() and wav.exists()):
            print(f"[warn] XML または wav がない: {r['file_id']}", file=sys.stderr)
            continue
        segs = ipu_segments(xml)
        if not segs:
            print(f"[warn] 有効な発話区間がない: {r['file_id']}", file=sys.stderr)
            continue
        (args.out / wav.name).symlink_to(wav.resolve())
        write_textgrid(
            args.out / f"{r['file_id']}.TextGrid", segs, wav_duration(wav), f"spk{r['speaker_id']}"
        )
        n_talks += 1
        n_seg += len(segs)
        speakers.add(r["speaker_id"])
        speech += sum(e - s for s, e, _ in segs)

    print(f"split              : {args.split}")
    print(f"talks              : {n_talks}")
    print(f"speakers           : {len(speakers)}")
    print(f"utterance segments : {n_seg}")
    print(f"speech duration    : {speech / 3600:.2f} 時間")
    print(f"corpus             : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
