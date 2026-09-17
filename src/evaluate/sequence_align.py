# SCOPE: shared
"""参照ラベルとMFA出力の系列アラインメント。4類型分類の前段。

なぜ正規化ラベル同士を単純比較しないか
--------------------------------------
写像表には**文脈依存の一対多**が含まれる。

    jsut u   -> MFA ɯ / ɨ      （ɨ は歯茎・口蓋（化）子音の後。辞書実測で全体の34%）
    jsut z   -> MFA z / dz     （摩擦実現 / 破擦実現）
    jsut N   -> MFA ɴ/m/n/ŋ/ɲ/ɰ̃（撥音の異音6種）

変換器は表の先頭行を採るため、参照側の /u/ は一律 `ɯ` になる。ここで正規化ラベル同士を
比較すると、MFAが `ɨ` を出すたびに偽の substitution が立つ。**これは規約差であって
MFAの誤りではない。**

そこで本モジュールは、参照側の**原ラベル**（`label_source`）とMFA音素の組が
写像表に行として存在するかで照合する。対応は非対称に扱われるため、
「jsut N は MFA m と両立するが、jsut m は MFA ɴ と両立しない」を正しく表現できる。

撥音の記号衝突について
----------------------
MFAの `m`/`n`/`ɲ`/`mʲ` はオンセット子音と撥音の異音で記号が同一である。
本方式では参照側が `N` か `m` かで両立性が決まるため、**系列の文脈から自動的に解決される**
（`decisions.md`「撥音の異音統合」問題2の対処）。単語層からの辞書引きは、
系列アラインメントで曖昧性が残る場合の追加手段としてフェーズ3で実装する。

使い方
------
    python3 src/evaluate/sequence_align.py \\
        --reference results/jsut_units.tsv \\
        --hypothesis results/mfa_jsut_units.tsv \\
        --mapping jsut \\
        --out results/jsut_alignment.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))

from mapping_loader import load_csj_mapping, load_jsut_mapping  # noqa: E402

MATCH, MISMATCH, GAP = 2, -1, -2


@dataclass
class Row:
    """中間表現TSVの1行（必要な列だけ保持する）。"""

    file_id: str
    unit_id: str
    t_start: float
    t_end: float
    label_canonical: str
    label_source: str
    phone_class: str
    excluded_reason: str
    boundary_source_start: str
    boundary_source_end: str
    metric_type: str
    ipu_id: str = ""
    ipu_start: float | None = None
    ipu_end: float | None = None
    word: str = ""

    @property
    def evaluable(self) -> bool:
        return (
            not self.excluded_reason
            and self.boundary_source_start == "measured"
            and self.boundary_source_end == "measured"
        )


def read_units(path: Path) -> list[Row]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = []
    for r in rows:
        out.append(
            Row(
                file_id=r["file_id"],
                unit_id=r["unit_id"],
                t_start=float(r["t_start"]),
                t_end=float(r["t_end"]),
                label_canonical=r["label_canonical"],
                label_source=r["label_source"],
                phone_class=r["phone_class"],
                excluded_reason=r.get("excluded_reason", ""),
                boundary_source_start=r["boundary_source_start"],
                boundary_source_end=r["boundary_source_end"],
                metric_type=r["metric_type"],
                ipu_id=r.get("ipu_id", "") or "",
                ipu_start=float(r["ipu_start"]) if r.get("ipu_start") else None,
                ipu_end=float(r["ipu_end"]) if r.get("ipu_end") else None,
                word=r.get("word", ""),
            )
        )
    return out


class Compatibility:
    """(参照の原ラベル, MFA音素) が写像表の行として存在するかを判定する。"""

    def __init__(self, mapping, side: str) -> None:
        self._pairs: set[tuple[str, str]] = set()
        for e in mapping._entries:  # noqa: SLF001 — 同一パッケージ内の意図的な参照
            src = e.jsut_label if side == "jsut" else e.csj_label
            if src and e.mfa_phone and e.mfa_phone != "-":
                self._pairs.add((src, e.mfa_phone))

    def compatible(self, ref_source: str, hyp_phone: str) -> bool:
        return (ref_source, hyp_phone) in self._pairs


def split_parts(ref_row: Row) -> list[str]:
    """参照単位の統合前の原ラベル列を返す。

    区切りはコーパスで異なる。jsut-label は空白（`"cl t"`）、CSJは `+`（`"Q+SclS+t"`）。
    `mapping/*.tsv` の csj_label / jsut_label 列の表記規約に合わせてある。
    """
    return [p for p in re.split(r"[+\s]+", ref_row.label_source) if p]


def multi_ok(ref_row: Row, hyps: list[Row], compat: Compatibility) -> bool:
    """参照の統合単位が、複数のMFA音素と順に両立するか（1:N の判定）。

    MFAの発音バリアント選択は非決定的で、撥音＋鼻音を `nː` に融合する場合と
    `ɴ n` の2分節で出す場合が混在する（docs/decisions-log.md A節）。
    参照側の統合ルールを固定できないため、対応づけの側で吸収する。
    """
    parts = split_parts(ref_row)
    if len(parts) != len(hyps) or len(parts) < 2:
        return False
    return all(compat.compatible(p, h.label_canonical) for p, h in zip(parts, hyps))


# 時間的に離れたペアを対応させない閾値（秒）。IPU窓内でも同一ラベルが複数あると、
# NWが遠いインスタンスに対応させ、数百ms規模の偽 displacement を生む。開始時刻が
# これ以上離れたペアは「両立しない」とみなし、omission+insertion に分解させる。
FAR_APART = 0.15


# 時間的に離れたペアの対角スコア。gap を2回引く（omission+insertion）より悪くして、
# 遠いインスタンスへのマッチを排除する。substitution（ラベル違いだが時刻が近い）は
# MISMATCH のまま許容する。
FAR_PENALTY = 2 * GAP - 1


def _far(r: Row, h_first: Row, h_last: Row) -> bool:
    """参照単位と対応MFA単位群が、開始・終了のどちらかで大きく離れているか。

    onset だけで判定すると、1:N対応や長い区間で終了側だけが数百msずれたペアが
    マッチとして残る（onsetの外れ値は消えるがoffsetに残る）。両端で判定する。
    """
    return (abs(h_first.t_start - r.t_start) > FAR_APART
            or abs(h_last.t_end - r.t_end) > FAR_APART)


def _match_score(r: Row, h: Row, ok: bool) -> float:
    """1:1 の対角スコア。"""
    if _far(r, h, h):
        return FAR_PENALTY  # 遠いペアは対応させない → omission+insertion に分解
    return MATCH if ok else MISMATCH


def align(
    ref: list[Row], hyp: list[Row], compat: Compatibility, max_span: int = 3
) -> list[tuple[Row | None, list[Row]]]:
    """Needleman-Wunsch。両立性を一致条件に使い、1:N の遷移も許容する。

    戻り値は (参照行 or None, 対応するMFA行のリスト) の列。
    1:1 なら リストの長さは1、1:N なら N、omission なら空、insertion なら参照が None。
    """
    n, m = len(ref), len(hyp)
    NEG = float("-inf")
    score = [[NEG] * (m + 1) for _ in range(n + 1)]
    score[0][0] = 0
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ok = compat.compatible(ref[i - 1].label_source, hyp[j - 1].label_canonical)
            best = max(
                score[i - 1][j - 1] + _match_score(ref[i - 1], hyp[j - 1], ok),
                score[i - 1][j] + GAP,
                score[i][j - 1] + GAP,
            )
            # 1:N の遷移。両立し、かつ両端が時間的に近い場合のみ、部分数に比例した加点。
            for k in range(2, min(max_span, j) + 1):
                if multi_ok(ref[i - 1], hyp[j - k : j], compat) and \
                        not _far(ref[i - 1], hyp[j - k], hyp[j - 1]):
                    cand = score[i - 1][j - k] + MATCH * k
                    if cand > best:
                        best = cand
            score[i][j] = best

    out: list[tuple[Row | None, list[Row]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            matched = False
            for k in range(2, min(max_span, j) + 1):
                if multi_ok(ref[i - 1], hyp[j - k : j], compat) and \
                        not _far(ref[i - 1], hyp[j - k], hyp[j - 1]) and \
                        score[i][j] == score[i - 1][j - k] + MATCH * k:
                    out.append((ref[i - 1], list(hyp[j - k : j])))
                    i, j = i - 1, j - k
                    matched = True
                    break
            if matched:
                continue
            ok = compat.compatible(ref[i - 1].label_source, hyp[j - 1].label_canonical)
            if score[i][j] == score[i - 1][j - 1] + _match_score(ref[i - 1], hyp[j - 1], ok):
                out.append((ref[i - 1], [hyp[j - 1]]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + GAP:
            out.append((ref[i - 1], []))
            i -= 1
            continue
        out.append((None, [hyp[j - 1]]))
        j -= 1
    out.reverse()
    return out


def classify(r: Row | None, hs: list[Row], compat: Compatibility) -> str:
    """4類型に排他的に分類する。定義は src/evaluate/error_types.py の docstring が正。"""
    if r is not None and not hs:
        return "omission"
    if r is None:
        return "insertion"
    if len(hs) > 1:
        # 1:N が成立している時点で両立性は確認済み（multi_ok）。境界ずれとして扱う。
        return "displacement"
    if not compat.compatible(r.label_source, hs[0].label_canonical):
        return "substitution"
    return "displacement"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--hypothesis", type=Path, required=True)
    ap.add_argument("--mapping", choices=["jsut", "csj"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    mapping = load_jsut_mapping() if args.mapping == "jsut" else load_csj_mapping()
    compat = Compatibility(mapping, args.mapping)

    ref_all = read_units(args.reference)
    hyp_all = read_units(args.hypothesis)

    # --- IPU窓ごとに突合する ---
    # 講演全体を1系列としてアラインメントすると、1箇所の脱落から経路が横滑りして
    # 数秒規模の偽の誤差を生む。参照側はIPU（原則200ms以上のポーズで区切られた単位）の
    # 時刻を持つので、それを窓として参照・仮説の双方を区切り、窓ごとに独立に突き合わせる。
    # MFA側はIPU情報を持たないため、各MFA単位の中点が入る参照IPU窓に割り当てる。
    def ref_windows(rows: list[Row]) -> dict[str, dict[str, tuple[float, float, list[Row]]]]:
        # まず file_id×ipu_id ごとに単位を集める。
        grouped: dict[str, dict[str, list[Row]]] = {}
        for r in rows:
            if not r.evaluable:
                continue
            key = r.ipu_id or "_nofile"
            grouped.setdefault(r.file_id, {}).setdefault(key, []).append(r)
        # 窓の時間範囲は、その窓に属する全単位から張る。
        # IPU時刻を持つ場合（CSJ）は全単位が同一のIPU開始/終了を持つので min/max は
        # そのIPU範囲に一致する（従来と同一）。IPU時刻が無い場合（JSUT: 発話単位で
        # ポーズ区切りが無く ipu_start/ipu_end が空）は、単位の t_start/t_end から
        # 発話全体を窓とする。従来は先頭単位の span を窓にしていたため、窓が1音素幅に
        # 縮退し、後続音素がすべて omission になっていた（JSUT経路のみのバグ）。
        d: dict[str, dict[str, tuple[float, float, list[Row]]]] = {}
        for fid, fw in grouped.items():
            d[fid] = {}
            for key, rs in fw.items():
                rs.sort(key=lambda x: x.t_start)
                starts = [r.ipu_start if r.ipu_start is not None else r.t_start for r in rs]
                ends = [r.ipu_end if r.ipu_end is not None else r.t_end for r in rs]
                d[fid][key] = (min(starts), max(ends), rs)
        return d

    ref_by = ref_windows(ref_all)

    # 仮説側を file_id ごとに時刻順で持ち、IPU窓に割り当てる
    hyp_by_file: dict[str, list[Row]] = {}
    for h in hyp_all:
        if h.evaluable:
            hyp_by_file.setdefault(h.file_id, []).append(h)
    for v in hyp_by_file.values():
        v.sort(key=lambda x: x.t_start)

    def assign_hyp(fid: str, start: float, end: float) -> list[Row]:
        """参照IPU窓 [start, end) と時間的に重なる仮説単位を返す。

        MFAと参照でIPU境界がフレーム単位でずれるため、中点判定だと窓端の音素を
        取りこぼし、末尾に連続omissionを生んで経路を横滑りさせる。重なり判定にして、
        各仮説単位は「中点が最も近い窓」に一意に属させる（重複割り当てを避ける）。
        """
        out = []
        for h in hyp_by_file.get(fid, []):
            if h.t_end <= start or h.t_start >= end:
                continue
            mid = (h.t_start + h.t_end) / 2.0
            # 窓に完全に収まる、または中点が窓内なら含める
            if start <= mid < end or (h.t_start >= start and h.t_end <= end):
                out.append(h)
        return out

    common = sorted(set(ref_by) & set(hyp_by_file))
    if not common:
        print("[error] 参照と仮説で共通する file_id がない。", file=sys.stderr)
        return 1

    from collections import Counter

    counts: Counter = Counter()
    out_rows: list[list[str]] = []
    metric = ref_all[0].metric_type if ref_all else "unknown"

    pairs_iter = []
    for fid in common:
        for ipu_id, (w_start, w_end, ref_rows) in sorted(ref_by[fid].items()):
            hyp_rows = assign_hyp(fid, w_start, w_end)
            counts["_ipu_windows"] += 1
            for r, hs in align(ref_rows, hyp_rows, compat):
                pairs_iter.append((fid, r, hs))

    for fid, r, hs in pairs_iter:
        if True:
            etype = classify(r, hs, compat)
            counts[etype] += 1
            if len(hs) > 1:
                counts["_multi_span"] += 1
            onset_err = offset_err = ""
            if r is not None and hs:
                # 1:N のときは外側境界どうしで比較する（内部境界は評価対象外）
                onset_err = f"{(hs[0].t_start - r.t_start) * 1000:.3f}"
                offset_err = f"{(hs[-1].t_end - r.t_end) * 1000:.3f}"
            out_rows.append([
                fid,
                r.unit_id if r else "",
                ";".join(h.unit_id for h in hs),
                r.label_source if r else "",
                r.label_canonical if r else "",
                " ".join(h.label_canonical for h in hs),
                r.phone_class if r else (hs[0].phone_class if hs else ""),
                etype,
                onset_err,
                offset_err,
                str(len(hs)),
                metric,
            ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow([
            "file_id", "ref_unit_id", "hyp_unit_id", "ref_label_source",
            "ref_label_canonical", "hyp_label", "phone_class", "error_type",
            "onset_error_ms", "offset_error_ms", "n_hyp_units", "metric_type",
        ])
        w.writerows(out_rows)

    total = sum(v for k, v in counts.items() if not k.startswith("_"))
    print(f"files aligned : {len(common)}")
    print(f"IPU windows   : {counts['_ipu_windows']}")
    print(f"pairs         : {total}")
    for k in ("displacement", "substitution", "omission", "insertion"):
        c = counts[k]
        print(f"  {k:<14} {c:>7}  ({100.0 * c / total:5.2f}%)" if total else f"  {k}: 0")
    print(f"  (うち 1:N 対応 {counts['_multi_span']} 件)")
    print(f"\nmetric_type   : {metric}")
    if metric == "concordance":
        print("  ※ これは精度ではなくアライナ間の一致度である（docs/decisions-log.md A節）")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
