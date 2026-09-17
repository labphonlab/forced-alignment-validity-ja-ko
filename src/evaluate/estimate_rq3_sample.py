# SCOPE: shared
"""RQ3（母音長×アクセント型）の分析対象がどれだけ残るかを事前見積もりする。

目的
----
`mapping/decisions.md` で確定した4つの決定のうち、次の2つがRQ3の母音長分析の
対象を直接削る。CSJ本番XMLを配置した直後に本スクリプトを実行し、**モデルを
組む前に**サンプルサイズが十分かを確認する。

  1. **無声化母音の除外**（「無声化母音の脱落分類」）
     無声化融合モーラは内部境界に真値が存在しないため母音長が測定できない。
     → 母音長分析から除外する。

  2. **撥音の後続環境による層別**（「撥音の異音統合」）
     撥音に先行する母音は、撥音の onset が母音の offset にあたる。特に `ɰ̃`
     環境（母音・摩擦音・接近音の前）では境界が原理的に不明瞭なため、母音長の
     誤差が系統的に大きくなる可能性がある。→ 層別が必要で、層ごとの件数が要る。

あわせて、長音・促音の決定に由来する除外（転記タグF/D末尾の融合促音など）と、
`StartTimeUncertain` 由来の不確実境界の件数も出す。

**サンプルサイズが想定より小さければ、この時点で報告すること。**
ICPhS版のRQ3は母音長×アクセント型のみであり、ここが痩せると論文の中心が崩れる。

出力について
------------
CLAUDE.md §1 に従い、**集計値以外は一切出力しない**。

使い方
------
    # 合成サンプルで動作確認
    python3 src/evaluate/estimate_rq3_sample.py data/samples/sample_csj.xml

    # CSJ本番
    python3 src/evaluate/estimate_rq3_sample.py data/csj/XML --glob '*.xml'
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# CSJの母音ラベル。大文字は無声化母音（segment.pdf 表1）。
VOICED_VOWELS = {"a", "i", "u", "e", "o"}
DEVOICED_VOWELS = {"A", "I", "U", "E", "O"}
LONG_VOWEL_SECOND = "H"

# 撥音の後続環境の層。CSJ側は撥音を N としか書かないため、後続セグメントから導出する。
# MFAの異音対応は mapping/csj_mfa_map.tsv を参照（ここでは層の粒度だけ決める）。
NASAL_CONTEXT_LABELS = {
    "vowel_or_fricative": "ɰ̃相当（境界不明瞭・誤差最大の予測）",
    "stop_or_affricate": "m/n/ŋ/ɲ相当",
    "nasal_fused": "鼻音後続（N,m/N,n の融合）",
    "final_or_pause": "ɴ相当（語末・ポーズ前）",
}


def _flag(elem: ET.Element, name: str) -> bool:
    return elem.get(name) == "1"


def classify_nasal_context(next_entity: str | None) -> str:
    """撥音の後続環境を層に落とす。next_entity は次のPhoneのPhoneEntity。"""
    if next_entity is None:
        return "final_or_pause"
    if next_entity in {"N", "m", "n", "nj", "ny", "my"}:
        return "nasal_fused"
    if next_entity in VOICED_VOWELS | DEVOICED_VOWELS | {LONG_VOWEL_SECOND}:
        return "vowel_or_fricative"
    if next_entity in {"s", "sj", "sy", "h", "hj", "hy", "F", "z", "zj", "zy"}:
        return "vowel_or_fricative"
    return "stop_or_affricate"


def scan(xml_paths: list[Path]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    n_files = 0

    for path in xml_paths:
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            print(f"[warn] parse error, skipped: {path.name} ({exc.code})", file=sys.stderr)
            continue
        n_files += 1

        phones = list(tree.iter("Phone"))
        for i, ph in enumerate(phones):
            entity = ph.get("PhoneEntity") or ""
            nxt = phones[i + 1].get("PhoneEntity") if i + 1 < len(phones) else None

            is_vowel = entity in VOICED_VOWELS or entity in DEVOICED_VOWELS
            if not is_vowel:
                if entity == "N":
                    counts["nasal_total"] += 1
                    counts[f"nasal_ctx::{classify_nasal_context(nxt)}"] += 1
                continue

            counts["vowel_total"] += 1

            # --- 除外条件を順に適用（排他的に数える） ---
            if entity in DEVOICED_VOWELS or _flag(ph, "Devoiced"):
                counts["excluded::devoiced"] += 1
                continue
            if _flag(ph, "StartTimeUncertain") or _flag(ph, "EndTimeUncertain"):
                counts["excluded::uncertain_boundary"] += 1
                continue

            counts["analyzable_vowel"] += 1
            if nxt == LONG_VOWEL_SECOND:
                counts["analyzable::long_vowel"] += 1
            else:
                counts["analyzable::short_vowel"] += 1
            # 撥音が後続する母音は層別が必要
            if nxt == "N":
                counts["analyzable::followed_by_nasal"] += 1

    return counts, n_files


def report(counts: Counter, n_files: int) -> None:
    total = counts["vowel_total"]
    print(f"files parsed              : {n_files}")
    print(f"vowel Phone total         : {total}")
    if not total:
        print("\n[warn] 母音が0件。入力が想定と異なる可能性がある。")
        return

    print()
    print("--- RQ3 母音長分析からの除外 ---")
    for key, label in [
        ("excluded::devoiced", "無声化（真値なし → 除外）"),
        ("excluded::uncertain_boundary", "不確実境界（等分割等 → 除外）"),
    ]:
        c = counts[key]
        print(f"  {label:<34} {c:>9}  ({100.0 * c / total:5.1f}%)")

    remain = counts["analyzable_vowel"]
    print()
    print(f"  {'分析可能な母音':<34} {remain:>9}  ({100.0 * remain / total:5.1f}%)")
    print(f"    {'うち短母音':<32} {counts['analyzable::short_vowel']:>9}")
    print(f"    {'うち長母音':<32} {counts['analyzable::long_vowel']:>9}")
    print(f"    {'うち撥音が後続（要層別）':<31} {counts['analyzable::followed_by_nasal']:>9}")

    print()
    print("--- 撥音の後続環境層（RQ1因子・RQ3層別に使う） ---")
    n_nasal = counts["nasal_total"]
    print(f"  {'撥音 total':<34} {n_nasal:>9}")
    for key, label in NASAL_CONTEXT_LABELS.items():
        c = counts[f"nasal_ctx::{key}"]
        pct = f"({100.0 * c / n_nasal:5.1f}%)" if n_nasal else ""
        print(f"    {label:<32} {c:>9}  {pct}")

    print()
    print("--- 判定 ---")
    if remain < 1000:
        print("  ⚠ 分析可能な母音が1,000件未満。ICPhS版RQ3（母音長×アクセント型）の")
        print("    サンプルサイズとして不足の可能性が高い。**この時点で報告すること。**")
    elif remain < 5000:
        print("  ⚠ 分析可能な母音が5,000件未満。アクセント型との交差で層が痩せる恐れ。")
        print("    アクセント型の分布を確認してから混合モデルの構造を決めること。")
    else:
        print("  分析可能な母音は十分な規模。ただしアクセント型・話者との交差を確認すること。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", type=Path)
    ap.add_argument("--glob", default="*.xml")
    args = ap.parse_args(argv)

    paths = sorted(args.target.rglob(args.glob)) if args.target.is_dir() else [args.target]
    if not paths:
        print(f"[error] no XML found under {args.target}", file=sys.stderr)
        return 1

    counts, n_files = scan(paths)
    report(counts, n_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
