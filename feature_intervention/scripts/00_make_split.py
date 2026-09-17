# SCOPE: pilot（段階0）
"""CSJ core の独話（APS/SPS）を、話者単位で学習用と評価用に分ける。

使う列は講演ID・話者ID・スタイル・IPU時刻だけで、CSJ 本体は読まない。
入力は mfa-ja-validation の派生表 results/csj_units.tsv。
READ（再朗読）6講演は予備分析から外す（docs/decisions-log.md D1）。

    python3 scripts/00_make_split.py
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT.parent / "mfa-ja-validation" / "results" / "csj_units.tsv"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", type=Path, default=UNITS)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--out", type=Path, default=ROOT / "splits" / "csj_core_split.tsv")
    args = ap.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    meta: dict[str, tuple[str, str]] = {}
    spans: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with args.units.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["style"] not in ("aps", "sps"):
                continue
            meta[r["file_id"]] = (r["speaker_id"], r["style"])
            spans[r["file_id"]].add((r["ipu_start"], r["ipu_end"]))

    hours = {
        t: sum(float(e) - float(s) for s, e in v if s and e) / 3600
        for t, v in spans.items()
    }

    talks_of: dict[str, list[str]] = defaultdict(list)
    for talk, (spk, _) in meta.items():
        talks_of[spk].append(talk)

    # スタイルの組み合わせ（APSのみ・SPSのみ・両方）で話者を層に分け、層ごとに同じ割合を評価用に回す
    strata: dict[str, list[str]] = defaultdict(list)
    for spk, talks in talks_of.items():
        strata["+".join(sorted({meta[t][1] for t in talks}))].append(spk)

    rng = random.Random(args.seed)
    test: set[str] = set()
    for key in sorted(strata):
        speakers = sorted(strata[key])
        rng.shuffle(speakers)
        test.update(speakers[: round(len(speakers) * args.test_frac)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["file_id", "speaker_id", "style", "speech_hours", "split"])
        for talk in sorted(meta):
            spk, style = meta[talk]
            split = "test" if spk in test else "train"
            w.writerow([talk, spk, style, f"{hours[talk]:.4f}", split])

    summary: dict[tuple[str, str], list] = defaultdict(lambda: [0, set(), 0.0])
    for talk, (spk, style) in meta.items():
        row = summary[("test" if spk in test else "train", style)]
        row[0] += 1
        row[1].add(spk)
        row[2] += hours[talk]
    for (split, style), (n, spks, h) in sorted(summary.items()):
        print(f"{split:<5} {style}: talks={n:>3} speakers={len(spks):>3} speech_hours={h:6.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
