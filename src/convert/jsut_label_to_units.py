# SCOPE: shared
"""jsut-label（HTSフルコンテキストラベル）→ 中間表現。

**このコーパスは境界精度の参照に使えない。** jsut-labelの時間情報はJuliusによる
自動forced alignmentであり（公式README明記）、人手校正版は存在しない
（`docs/decisions-log.md` A節）。したがって出力レコードには一律
`metric_type=CONCORDANCE` を立てる。`schema.require_accuracy()` がこれを検出して
精度としての集計を拒否する。

適用する統合（`mapping/decisions.md` の確定事項）
------------------------------------------------
  長音    同一母音の隣接 `e e` を1単位に統合（MFAの `eː` に対応させる）
  促音    `cl` + 後続子音 を1単位に統合（MFAの `tː` に対応させる）
  撥音    `N` + 後続鼻音 を1単位に統合（MFAの `mː` / `nː` に対応させる）

いずれも統合で捨てた内部境界は `*_internal_boundaries.tsv` に保持する。

入力形式
--------
1行が `<開始> <終了> <左々>^<左>-<中心>+<右>=<右々>/A:.../B:...` の形。
時刻は100ns単位（1秒 = 10,000,000）。中心音素のみを使う。

使い方
------
    python3 src/convert/jsut_label_to_units.py data/jsut-label/labels/basic5000 \\
        --limit 50 --out results/jsut_units.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapping_loader import UnmappedLabelError, load_jsut_mapping  # noqa: E402
from schema import (  # noqa: E402
    BoundarySource,
    Corpus,
    ExclusionReason,
    InternalBoundary,
    MetricType,
    Style,
    Unit,
    internal_boundary_path,
    validate,
    write_internal_boundaries,
    write_units,
)

HTS_LINE = re.compile(r"^(\d+)\s+(\d+)\s+(?:[^^]*)\^(?:[^-]*)-([^+]*)\+")
HTU_PER_SECOND = 10_000_000  # 100ns単位

VOWELS = {"a", "i", "u", "e", "o"}
NASALS = {"m", "n", "ny", "my"}  # 撥音と融合しうるオンセット鼻音
SILENCES = {"pau", "sil"}
GEMINATE_CLOSURE = "cl"
MORAIC_NASAL = "N"


class Segment:
    """ラベル1行 = 統合前の原セグメント。"""

    __slots__ = ("start", "end", "phone")

    def __init__(self, start: float, end: float, phone: str) -> None:
        self.start = start
        self.end = end
        self.phone = phone


def parse_lab(path: Path) -> list[Segment]:
    segs: list[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HTS_LINE.match(line)
        if not m:
            continue
        start, end, phone = m.group(1), m.group(2), m.group(3)
        segs.append(Segment(int(start) / HTU_PER_SECOND, int(end) / HTU_PER_SECOND, phone))
    return segs


def _phone_class(phones: list[str]) -> str:
    """統合後の単位から音素クラスを決める。RQ1では使わないが整合性のために付ける。"""
    head = phones[0]
    if len(phones) == 2 and head in VOWELS and phones[1] == head:
        return "vowel_long"
    if head == GEMINATE_CLOSURE:
        return "geminate"
    if head == MORAIC_NASAL and len(phones) == 2:
        return "moraic_nasal_fused"
    if head == MORAIC_NASAL:
        return "moraic_nasal"
    if head in SILENCES:
        return "silence"
    if head in VOWELS:
        return "vowel_short"
    return "consonant"


def merge_segments(segs: list[Segment]) -> list[tuple[list[Segment], str]]:
    """確定した3決定に従って原セグメントを統合する。

    戻り値は (統合対象の原セグメント列, 音素クラス) の列。
    """
    out: list[tuple[list[Segment], str]] = []
    i = 0
    while i < len(segs):
        cur = segs[i]
        nxt = segs[i + 1] if i + 1 < len(segs) else None

        group = [cur]
        if nxt is not None:
            same_vowel = cur.phone in VOWELS and nxt.phone == cur.phone
            geminate = cur.phone == GEMINATE_CLOSURE
            nasal_fused = cur.phone == MORAIC_NASAL and nxt.phone in NASALS
            if same_vowel or geminate or nasal_fused:
                group = [cur, nxt]

        out.append((group, _phone_class([s.phone for s in group])))
        i += len(group)
    return out


def convert_file(path: Path, mapping) -> tuple[list[Unit], list[InternalBoundary]]:
    file_id = path.stem
    segs = parse_lab(path)
    units: list[Unit] = []
    boundaries: list[InternalBoundary] = []

    for idx, (group, phone_class) in enumerate(merge_segments(segs)):
        unit_id = f"{file_id}_{idx:04d}"
        label_source = " ".join(s.phone for s in group)

        # 写像は mapping/jsut_mfa_map.tsv のみを参照する（CLAUDE.md §3）。
        # **恒等写像のフォールバックは置かない。** jsut-labelはローマ字表記、MFAはIPA表記で
        # 単独音素も対応が異なるため（jsut sh → MFA ɕ、jsut y → MFA j 等）、
        # 素通しは規約差を誤差として静かに計上する事故を招く。写像漏れは例外で露出させる。
        excluded = None
        entries = mapping.entries_for_jsut(label_source)  # 未写像なら UnmappedLabelError
        label_canonical = entries[0].mfa_phone
        category = entries[0].category

        if group[0].phone in SILENCES:
            excluded = ExclusionReason.UNMAPPED_LABEL  # 無音は評価対象外

        units.append(
            Unit(
                file_id=file_id,
                speaker_id="jsut_single_speaker",  # JSUTは単一話者コーパス
                corpus=Corpus.JSUT,
                style=Style.NA,
                metric_type=MetricType.CONCORDANCE,  # ← 精度ではない
                ipu_id=None,
                ipu_start=None,
                ipu_end=None,
                unit_id=unit_id,
                t_start=group[0].start,
                t_end=group[-1].end,
                label_canonical=label_canonical,
                label_source=label_source,
                phone_class=category or phone_class,
                boundary_source_start=BoundarySource.MEASURED,
                boundary_source_end=BoundarySource.MEASURED,
                merged_from=len(group),
                excluded_reason=excluded,
            )
        )

        # 統合で捨てた内部境界を保持（感度分析用）
        for pos in range(len(group) - 1):
            boundaries.append(
                InternalBoundary(
                    file_id=file_id,
                    unit_id=unit_id,
                    t=group[pos].end,
                    source=BoundarySource.DERIVED,
                    position=pos,
                    note=f"merged {label_source}",
                )
            )

    return units, boundaries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("label_dir", type=Path, help="jsut-label の labels/basic5000 等")
    ap.add_argument("--limit", type=int, default=0, help="先頭N件のみ処理（0で全件）")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    paths = sorted(args.label_dir.glob("*.lab"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"[error] no .lab found under {args.label_dir}", file=sys.stderr)
        return 1

    mapping = load_jsut_mapping()
    all_units: list[Unit] = []
    all_boundaries: list[InternalBoundary] = []
    unmapped: dict[str, int] = {}
    for p in paths:
        try:
            u, b = convert_file(p, mapping)
        except UnmappedLabelError as exc:
            # 1ファイルの写像漏れで全体を止めず、まとめて報告する
            key = str(exc).strip("\"'")
            unmapped[key] = unmapped.get(key, 0) + 1
            continue
        all_units.extend(u)
        all_boundaries.extend(b)

    if unmapped:
        print(f"[error] 写像漏れで {sum(unmapped.values())} ファイルをスキップ:", file=sys.stderr)
        for msg, n in sorted(unmapped.items(), key=lambda kv: -kv[1]):
            print(f"  ({n:>4}件) {msg}", file=sys.stderr)
        print("  → mapping/jsut_mfa_map.tsv に追記すること。", file=sys.stderr)
        return 1

    problems = validate(all_units)
    if problems:
        print(f"[error] スキーマ検証で {len(problems)} 件の問題:", file=sys.stderr)
        for p in problems[:10]:
            print(f"  {p}", file=sys.stderr)
        return 1

    write_units(all_units, args.out)
    write_internal_boundaries(all_boundaries, internal_boundary_path(args.out))

    n_merged = sum(1 for u in all_units if u.merged_from > 1)
    n_excluded = sum(1 for u in all_units if u.excluded_reason is not None)
    print(f"files            : {len(paths)}")
    print(f"units            : {len(all_units)}")
    print(f"  merged (>1seg) : {n_merged}")
    print(f"  excluded       : {n_excluded}")
    print(f"  analyzable     : {sum(1 for u in all_units if u.is_analyzable)}")
    print(f"internal bounds  : {len(all_boundaries)}")
    print()
    print(f"wrote {args.out}")
    print(f"wrote {internal_boundary_path(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
