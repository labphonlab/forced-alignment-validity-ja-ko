# SCOPE: shared
"""japanese_mfa辞書の同一母音連続（V V）表記揺れを監査する。

背景
----
japanese_mfa辞書は長音を原則としてIPA長音記号つきの単一音素で書く（`東京 t oː c oː`）。
しかし一部の項目は長音を母音2個で書いており、表記が一様でない。決定的な例:

    お姉さん     o n e e s a ɴ    ← 母音2個
    おねえさん   o n eː s a ɴ     ← 単一音素（同一発音・同一語）

`mapping/decisions.md`「長音の分割方針」の付随決定3に従い、本スクリプトは
該当語を機械的に洗い出して**人手判断の対象を絞る**。一括正規化はしない。
同一母音連続には正当なものが多数含まれるためである:

    あいだをおいて  a i d a o o i t e   ← 形態素境界をまたぐ。統合してはならない
    ｎａｎ          e n ɯ e e n ɯ       ← 略語の字読み。統合してはならない

出力
----
標準出力: 集計値のみ。
`--out` 指定時: 該当エントリの一覧をTSVで書き出す（人手レビュー用）。
japanese_mfa辞書は公開リソースであり、CLAUDE.md §1 のCSJ制約の対象ではない。

使い方
------
    python3 src/convert/audit_vv_entries.py path/to/japanese_mfa.dict \\
        --out results/vv_audit.tsv
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# japanese_mfa の短母音。長音記号つき（aː 等）は対象外。
SHORT_VOWELS = {"a", "e", "i", "o", "ɯ", "ɨ"}

# 無声化母音。i̥ 等は結合文字 U+0325 を含むため、正規化して基底文字で判定する。
COMBINING_RING_BELOW = "̥"


def _base_vowel(phone: str) -> str | None:
    """音素が短母音ならその基底文字を返す。長音・子音・無声化母音は None。"""
    if COMBINING_RING_BELOW in phone:
        return None  # 無声化母音は長音化しないので対象外
    normalized = unicodedata.normalize("NFC", phone)
    return normalized if normalized in SHORT_VOWELS else None


def parse_line(line: str) -> tuple[str, list[str]] | None:
    """辞書1行を (表記, 音素列) に分解する。

    japanese_mfa辞書にはプレーン版（2フィールド）とリリース版
    （表記のあとに確率4列が入る）がある。数値列を読み飛ばして両方に対応する。
    """
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    fields = line.split("\t")
    if len(fields) < 2:
        return None
    word, rest = fields[0], fields[1:]

    # 末尾フィールドが空白区切りの音素列。手前に数値列があれば飛ばす。
    tokens = " ".join(rest).split()
    start = 0
    for i, tok in enumerate(tokens):
        try:
            float(tok)
        except ValueError:
            start = i
            break
    else:
        return None
    phones = tokens[start:]
    return (word, phones) if phones else None


def find_vv(phones: list[str]) -> list[tuple[int, str]]:
    """同一短母音が隣接する箇所を [(位置, 母音)] で返す。"""
    hits = []
    for i in range(len(phones) - 1):
        v = _base_vowel(phones[i])
        if v is not None and v == _base_vowel(phones[i + 1]):
            hits.append((i, v))
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dict_path", type=Path, help="japanese_mfa.dict へのパス")
    ap.add_argument("--out", type=Path, help="該当エントリのTSV出力先（人手レビュー用）")
    args = ap.parse_args(argv)

    if not args.dict_path.exists():
        print(f"[error] not found: {args.dict_path}", file=sys.stderr)
        return 1

    n_entries = 0
    n_long_mark = 0          # ː を含むエントリ数
    by_vowel: Counter = Counter()
    affected: list[tuple[str, str, str]] = []

    with args.dict_path.open(encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_line(line)
            if parsed is None:
                continue
            word, phones = parsed
            n_entries += 1
            if any("ː" in p for p in phones):
                n_long_mark += 1
            hits = find_vv(phones)
            if hits:
                for _, v in hits:
                    by_vowel[v] += 1
                affected.append((word, " ".join(phones), ",".join(v for _, v in hits)))

    print(f"dictionary entries          : {n_entries}")
    print(f"entries containing 'ː'      : {n_long_mark} ({100.0 * n_long_mark / n_entries:.2f}%)")
    print(f"entries with V V sequence   : {len(affected)} ({100.0 * len(affected) / n_entries:.2f}%)")
    print()
    print(f"{'vowel':>8} {'occurrences':>12}")
    print("-" * 22)
    for v, c in by_vowel.most_common():
        print(f"{v:>8} {c:>12}")
    print()
    print(f"total V V occurrences       : {sum(by_vowel.values())}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            fh.write("word\tpronunciation\tvv_vowels\tdecision\n")
            for word, pron, vowels in affected:
                # decision列は人手で埋める: merge / keep / exclude
                fh.write(f"{word}\t{pron}\t{vowels}\t\n")
        print()
        print(f"wrote {len(affected)} rows to {args.out}")
        print("decision列を人手で埋めること: merge（長音として統合）/ keep（正当な母音連続）/ exclude（判断保留・分析から除外）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
