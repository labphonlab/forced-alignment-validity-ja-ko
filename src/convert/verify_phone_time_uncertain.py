# SCOPE: shared
"""CSJ XMLの Phone/@StartTimeUncertain・@EndTimeUncertain の性質を検証する。

目的
----
`mapping/decisions.md`「長音の分割方針」で、長母音の内部境界はCSJ側が等分割の
人工物であると判断した（segment.pdf §7「長母音を分割する。この場合も等分割する」）。
XMLには `StartTimeUncertain` / `EndTimeUncertain` 属性が存在するが（xml.pdf 表1）、
これが等分割由来の境界に立つかどうかはマニュアルに明示がない。

本スクリプトは `PhoneEntity == "H"`（長母音の第二要素）と不確実フラグの
クロス集計を取り、以下のどちらであるかを判定する材料を出す。

  (a) フラグが等分割由来の境界の印である
      → 統合対象の特定をコーパス自身の申告に基づいて行える
  (b) そうでない
      → `PhoneEntity == "H"` を直接トリガにする実装に落とす

出力について
------------
CLAUDE.md §1 に従い、**集計値（件数・比率）以外は一切出力しない**。
音素列・時刻・話者ID等の実データ内容は標準出力にも戻り値にも載せない。

使い方
------
    # 合成サンプルで動作確認（先にこれを通すこと）
    python3 src/convert/verify_phone_time_uncertain.py data/samples/sample_csj.xml

    # CSJ本番（内容は読まず、集計値だけが出る）
    python3 src/convert/verify_phone_time_uncertain.py data/csj/XML --glob '*.xml'
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# 長母音の第二要素。segment.pdf 表1 脚注「ラベル"H"は長母音の第二要素を表す」
LONG_VOWEL_SECOND_ELEMENT = "H"


def _flag(elem: ET.Element, name: str) -> bool:
    """0/1属性を読む。xml.pdf 4.1「"0"をとる場合は属性そのものを記述しない」。"""
    return elem.get(name) == "1"


def tabulate(xml_paths: list[Path]) -> tuple[Counter, Counter, int, int]:
    """Phone要素を走査し、(不確実フラグ集計, PhoneEntity×PhoneClass集計, ファイル数, 総Phone数)。

    第1の集計キーは (is_long_vowel_H, start_uncertain, end_uncertain) の3つ組。
    第2の集計は (PhoneEntity, PhoneClass, Devoiced) の3つ組。促音の閉鎖区間 "<cl>" が
    Phone層にどう現れるか（属性値の記法・PhoneClassの値）がマニュアルから確定できない
    ため、値の分布を実データで確認する。
    """
    flag_table: Counter = Counter()
    entity_table: Counter = Counter()
    n_phones = 0
    n_files = 0

    for path in xml_paths:
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            # ファイル名は出すが、中身は出さない
            print(f"[warn] parse error, skipped: {path.name} ({exc.code})", file=sys.stderr)
            continue
        n_files += 1
        for phone in tree.iter("Phone"):
            n_phones += 1
            flag_table[(
                phone.get("PhoneEntity") == LONG_VOWEL_SECOND_ELEMENT,
                _flag(phone, "StartTimeUncertain"),
                _flag(phone, "EndTimeUncertain"),
            )] += 1
            entity_table[(
                phone.get("PhoneEntity") or "(none)",
                phone.get("PhoneClass") or "(none)",
                _flag(phone, "Devoiced"),
            )] += 1

    return flag_table, entity_table, n_files, n_phones


def report(table: Counter, n_files: int, n_phones: int) -> None:
    """集計値のみを標準出力に書く。"""
    print(f"files parsed : {n_files}")
    print(f"total Phone  : {n_phones}")
    print()
    print(f"{'PhoneEntity':>12} {'StartUncert':>12} {'EndUncert':>10} {'count':>10} {'pct':>7}")
    print("-" * 56)
    for is_h, su, eu in sorted(table, key=lambda k: (not k[0], k[1], k[2])):
        count = table[(is_h, su, eu)]
        pct = 100.0 * count / n_phones if n_phones else 0.0
        label = '"H"' if is_h else "other"
        print(f"{label:>12} {str(su):>12} {str(eu):>10} {count:>10} {pct:>6.2f}%")

    n_h = sum(c for (is_h, _, _), c in table.items() if is_h)
    n_h_start = sum(c for (is_h, su, _), c in table.items() if is_h and su)
    print()
    if n_h:
        rate = 100.0 * n_h_start / n_h
        print(f'"H" phones                      : {n_h}')
        print(f'"H" with StartTimeUncertain=1   : {n_h_start} ({rate:.2f}%)')
        print()
        # 判定の目安。しきい値は報告用であって、実装の分岐はこの出力を見て人が決める。
        if rate >= 99.0:
            print("=> ほぼ全ての H に不確実フラグ。仮説(a)と整合: フラグを統合対象の判定に使える。")
        elif rate <= 1.0:
            print("=> H にフラグがほぼ立たない。仮説(b): PhoneEntity=='H' を直接トリガにする。")
        else:
            print("=> 中間的。フラグは等分割以外の要因も拾っている。PhoneEntity=='H' を主トリガとし、")
            print("   フラグは補助情報として保持する実装が安全。")
    else:
        print('[warn] PhoneEntity=="H" が0件。入力が想定と異なる可能性がある。')


def report_entities(entity_table: Counter, n_phones: int) -> None:
    """PhoneEntity × PhoneClass × Devoiced の分布。促音の <cl> の記法確認用。"""
    print()
    print("=" * 56)
    print("PhoneEntity × PhoneClass × Devoiced")
    print("=" * 56)
    print(f"{'PhoneEntity':>16} {'PhoneClass':>14} {'Devoiced':>9} {'count':>10}")
    print("-" * 56)
    for (entity, cls, dv), count in sorted(entity_table.items(), key=lambda kv: -kv[1]):
        print(f"{entity:>16} {cls:>14} {str(dv):>9} {count:>10}")

    # 促音の閉鎖区間がどの記法で現れるかの手掛かり
    closure_like = {e for (e, _, _) in entity_table if "cl" in e.lower()}
    print()
    if closure_like:
        print(f"閉鎖区間らしき PhoneEntity: {sorted(closure_like)}")
        print("→ この値を src/evaluate/ の閉鎖区間判定に用いる。mapping/decisions.md に追記すること。")
    else:
        print("[warn] 'cl' を含む PhoneEntity が見つからない。")
        print("→ Phone層に閉鎖区間が現れない可能性。.seg 側の確認が必要。")

    moraic_n = {e for (e, _, _) in entity_table if e in {"N", "Q"}}
    if moraic_n:
        print(f"特殊モーラの PhoneEntity: {sorted(moraic_n)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", type=Path, help="XMLファイル、またはXMLを含むディレクトリ")
    ap.add_argument("--glob", default="*.xml", help="ディレクトリ指定時の探索パターン")
    args = ap.parse_args(argv)

    if args.target.is_dir():
        paths = sorted(args.target.rglob(args.glob))
    else:
        paths = [args.target]

    if not paths:
        print(f"[error] no XML found under {args.target}", file=sys.stderr)
        return 1

    flag_table, entity_table, n_files, n_phones = tabulate(paths)
    report(flag_table, n_files, n_phones)
    report_entities(entity_table, n_phones)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
