# SCOPE: shared
"""XML Phone層とTextGrid `seg` tier を突合し、等分割由来の内部境界を同定する。

なぜこれが要るか
----------------
`mapping/decisions.md` の3項目（長音・撥音・無声化母音）に、実装トリガーが未確定の
⚠ が残っている。いずれも「XML Phone層の内部境界が、等分割の人工物か実測かを
どう判別するか」という同じ問いである。

`Phone/@StartTimeUncertain` がその印である**可能性**はあるが、xml.pdf に明示がない。
そこで、フラグに頼らず**構造の突合で直接判定する**。

    CSJ配布のTextGrid `seg` tier = 分節音ラベルの**原形**
        融合ラベル（`Q,t`・`N,m`・`s,U`）がカンマ区切りのまま1区間として入っている
    XML Phone層 = そこから segment.pdf §7 の手順で機械生成したもの
        融合ラベルは等分割され、長母音 `oH` も `o`|`H` に等分割される

したがって **TextGridの1区間に対しXMLが複数Phoneを持つ場合、その内部境界は
定義上すべて等分割の人工物**である。フラグの意味を推測する必要がない。

本スクリプトはその対応関係を集計し、あわせて
`StartTimeUncertain` / `EndTimeUncertain` が実際に等分割境界の印になっているかを
検証する（なっていればフラグで安価に判定でき、なっていなければ本突合を
本番パイプラインに組み込む必要がある）。

出力について
------------
CLAUDE.md §1 に従い、**集計値のみを標準出力に出す**。ラベル種別は
「どの融合パターンが何件あるか」を出すが、これは分節音ラベル体系の情報であって
発話内容ではない。時刻・話者・転記テキストは一切出さない。

使い方
------
    python3 src/convert/verify_equal_division.py data/csj --limit 5
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TOL = 1e-4  # 時刻突合の許容誤差（秒）

# TextGrid の interval を素朴に読む（praatio を使わず依存を増やさない）
_TIER_RE = re.compile(r'name\s*=\s*"([^"]*)"')
_INTERVAL_RE = re.compile(
    r"xmin\s*=\s*([0-9.eE+-]+)\s*\n\s*xmax\s*=\s*([0-9.eE+-]+)\s*\n\s*text\s*=\s*\"((?:[^\"]|\"\")*)\"",
    re.MULTILINE,
)


def read_seg_tier(path: Path) -> list[tuple[float, float, str]]:
    """TextGrid の `seg` tier の区間を (xmin, xmax, text) で返す。"""
    txt = path.read_text(encoding="utf-8", errors="replace")
    # tier ごとに分割し、name="seg" のブロックだけを取る
    parts = re.split(r'(?=name\s*=\s*")', txt)
    for block in parts:
        m = _TIER_RE.match(block.lstrip())
        if not m or m.group(1) != "seg":
            continue
        return [
            (float(a), float(b), c.replace('""', '"'))
            for a, b, c in _INTERVAL_RE.findall(block)
        ]
    return []


def read_xml_phones(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for ph in ET.parse(path).iter("Phone"):
        s, e = ph.get("PhoneStartTime"), ph.get("PhoneEndTime")
        if s is None or e is None:
            continue
        out.append((float(s), float(e), ph.get("PhoneEntity") or ""))
    out.sort()
    return out


def read_xml_flags(path: Path) -> dict[tuple[float, float], tuple[bool, bool]]:
    d = {}
    for ph in ET.parse(path).iter("Phone"):
        s, e = ph.get("PhoneStartTime"), ph.get("PhoneEndTime")
        if s is None or e is None:
            continue
        d[(float(s), float(e))] = (
            ph.get("StartTimeUncertain") == "1",
            ph.get("EndTimeUncertain") == "1",
        )
    return d


def analyse(xml_path: Path, tg_path: Path, stats: Counter, patterns: Counter,
            flag_tab: Counter) -> None:
    segs = read_seg_tier(tg_path)
    phones = read_xml_phones(xml_path)
    flags = read_xml_flags(xml_path)
    if not segs or not phones:
        stats["files_skipped"] += 1
        return
    stats["files"] += 1

    pi = 0
    for s_start, s_end, s_text in segs:
        label = s_text.strip()
        if not label:
            continue
        # この seg 区間に収まる Phone を集める
        group = []
        while pi < len(phones) and phones[pi][0] < s_end - TOL:
            if phones[pi][1] <= s_end + TOL and phones[pi][0] >= s_start - TOL:
                group.append(phones[pi])
                pi += 1
            elif phones[pi][1] <= s_start + TOL:
                pi += 1
            else:
                break
        if not group:
            stats["seg_without_phone"] += 1
            continue

        stats["seg_total"] += 1
        n = len(group)
        stats[f"seg_to_{min(n, 4)}phone"] += 1

        if n >= 2:
            # TextGridで1区間 = XMLで複数Phone → 内部境界はすべて等分割由来
            stats["derived_boundaries"] += n - 1
            patterns[f"{label} -> {n}"] += 1
            # 内部境界に不確実フラグが立っているか
            for k in range(n - 1):
                _, _, _ = group[k]
                end_flag = flags.get((group[k][0], group[k][1]), (False, False))[1]
                start_flag = flags.get((group[k + 1][0], group[k + 1][1]), (False, False))[0]
                flag_tab[("derived", end_flag or start_flag)] += 1
        else:
            # 1:1 の対応 → その両端は実測
            f = flags.get((group[0][0], group[0][1]), (False, False))
            flag_tab[("measured", f[0] or f[1])] += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    xmls = sorted(args.corpus_dir.glob("*.xml"))
    if args.limit:
        xmls = xmls[: args.limit]
    if not xmls:
        print(f"[error] no XML under {args.corpus_dir}", file=sys.stderr)
        return 1

    stats: Counter = Counter()
    patterns: Counter = Counter()
    flag_tab: Counter = Counter()

    for x in xmls:
        tg = x.with_suffix(".TextGrid")
        if not tg.exists():
            stats["missing_textgrid"] += 1
            continue
        analyse(x, tg, stats, patterns, flag_tab)

    print(f"files analysed        : {stats['files']}")
    print(f"seg intervals         : {stats['seg_total']}")
    for k in (1, 2, 3, 4):
        c = stats[f"seg_to_{k}phone"]
        lbl = f"{k}" if k < 4 else "4+"
        pct = 100.0 * c / stats["seg_total"] if stats["seg_total"] else 0
        print(f"  seg 1区間 -> Phone {lbl:<2}    {c:>8}  ({pct:5.2f}%)")
    print(f"等分割由来と確定した内部境界 : {stats['derived_boundaries']}")

    print()
    print("=== 不確実フラグは等分割境界の印か ===")
    print(f"{'境界の種別':<14}{'フラグ有':>10}{'フラグ無':>10}{'フラグ的中率':>14}")
    for kind in ("derived", "measured"):
        yes = flag_tab[(kind, True)]
        no = flag_tab[(kind, False)]
        tot = yes + no
        rate = 100.0 * yes / tot if tot else 0.0
        print(f"{kind:<14}{yes:>10}{no:>10}{rate:>13.2f}%")

    d_yes, d_no = flag_tab[("derived", True)], flag_tab[("derived", False)]
    m_yes, m_no = flag_tab[("measured", True)], flag_tab[("measured", False)]
    print()
    if d_yes + d_no and m_yes + m_no:
        sens = 100.0 * d_yes / (d_yes + d_no)
        spec = 100.0 * m_no / (m_yes + m_no)
        print(f"感度（等分割をフラグで拾える割合）: {sens:.2f}%")
        print(f"特異度（実測にフラグが立たない割合）: {spec:.2f}%")
        if sens >= 99 and spec >= 99:
            print("=> 仮説(a)成立。フラグ判定で実装してよい。")
        else:
            print("=> フラグ単独では不十分。**TextGrid seg tier との突合を本番に組み込むこと。**")

    print()
    print("=== 融合パターン上位（ラベル体系の情報。発話内容ではない）===")
    for pat, c in patterns.most_common(20):
        print(f"  {pat:<28} {c:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
