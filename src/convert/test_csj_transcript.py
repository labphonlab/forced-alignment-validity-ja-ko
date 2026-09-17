# SCOPE: shared
"""csj_transcript.py のユニットテスト。

CSJ実データを使わずに検証する（CLAUDE.md §1）。テストケースは
`data/samples/csj_transcript_cases.tsv` にあり、仕様書 transcription.pdf の
実例と、仕様の記述から構成した異常系からなる。

    python3 src/convert/test_csj_transcript.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from csj_transcript import parse, strip_tags  # noqa: E402

CASES = Path(__file__).resolve().parents[2] / "data" / "samples" / "csj_transcript_cases.tsv"


def load_cases() -> list[dict[str, str]]:
    with CASES.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def run() -> int:
    cases = load_cases()
    failures: list[str] = []

    for c in cases:
        cid = c["id"]
        got = strip_tags(c["input"])
        exp_text = c.get("expected_text") or ""
        exp_tags = {t for t in (c.get("expected_tags") or "").split(",") if t}
        got_tags = {s.tag for s in got.spans}

        # r_masked_nest は前半のみ検証する（残りは伏せ字で本文なし）
        if cid == "r_masked_nest":
            if got.text.endswith("というタイトル"):
                pass
            else:
                failures.append(f"{cid}: text={got.text!r} が『というタイトル』で終わらない")
        elif got.text != exp_text:
            failures.append(f"{cid}: text 期待={exp_text!r} 実際={got.text!r}")

        if got_tags != exp_tags:
            failures.append(f"{cid}: tags 期待={sorted(exp_tags)} 実際={sorted(got_tags)}")

        if got.warnings:
            failures.append(f"{cid}: 予期しない警告 {got.warnings}")

    # --- 個別の意味的テスト ---
    # (R) を含む転記単位は丸ごと除外する必要がある
    for cid, src, expect in [
        ("r_plain", "(R 佐藤)が発表します", True),
        ("r_masked", "(R××)が発表します", True),
        ("f_simple", "(F あの)千九百九十五年", False),
    ]:
        if strip_tags(src).has_masked_region is not expect:
            failures.append(f"{cid}: has_masked_region 期待={expect}")

    # 促音のF/D末尾判定
    for src, pos, expect in [("(F あっ)", 1, "F"), ("(D コッ)", 1, "D"), ("まっすぐ", 1, None)]:
        got = strip_tags(src).ends_with_tracked(pos)
        if got != expect:
            failures.append(f"ends_with_tracked({src!r},{pos}) 期待={expect} 実際={got}")

    # (W) 左項の (F) は基本形に現れないため、発音形を併用しないと取りこぼす
    basic = "チェコ・スロバキアの"
    phonetic = "(W セコスロバ(F アノー)スロバキア;チェコスロバキア)ノ"
    if strip_tags(basic).tracked_spans():
        failures.append("基本形のみで (F) が検出された（テスト前提が崩れている）")
    merged = parse(basic=basic, phonetic=phonetic)
    if not any(s.tag == "F" and s.source == "phonetic" for s in merged.spans):
        failures.append("parse(basic, phonetic) が発音形の (F) を拾えていない")

    print(f"cases : {len(cases)}")
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  {f}")
        return 1
    print("すべて通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
