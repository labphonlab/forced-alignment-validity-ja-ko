# SCOPE: shared
"""MFA出力TextGrid → 中間表現。

MFA音素は既に正規形なので、CSJ/jsut側のような統合は不要である
（長音 `oː`・促音 `tː`・撥音融合 `mː` はいずれもMFA側では既に1音素）。
本変換器の仕事は次の3つ。

  1. phones tier の各区間を Unit に落とす
  2. `mapping/mfa_phone_class.tsv` から音素クラスを割り当てる
  3. words tier を突き合わせ、各音素に単語と単語内位置を付ける
     （撥音の記号衝突を下流で解消するために必要。decisions.md「撥音の異音統合」問題2）

`metric_type` は呼び出し側が指定する。JSUTを対象にするときは必ず `concordance`
を指定すること（`accuracy` を指定すると schema.validate が弾く）。

TextGridの解析には praatio を使う。MFA環境（conda `mfa`）で実行すること。

    conda activate mfa
    python3 src/convert/mfa_textgrid_to_units.py results/mfa_jsut/aligned \\
        --corpus jsut --metric concordance --out results/mfa_jsut_units.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import (  # noqa: E402
    BoundarySource,
    Corpus,
    ExclusionReason,
    MetricType,
    Style,
    Unit,
    validate,
    write_units,
)

MAPPING_DIR = Path(__file__).resolve().parents[2] / "mapping"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# MFA辞書で長音が母音2個に書かれている語の統合対象。VV監査の merge 判定を使う。
# mapping/decisions.md「長音の分割方針」付随決定3、および
# results/vv_audit_classified.tsv（classify_vv_entries.py の出力）。
SHORT_VOWELS = {"a", "i", "e", "o", "ɯ", "ɨ"}
LONG_OF = {v: v + "ː" for v in SHORT_VOWELS}


def load_vv_merge_words(path: Path | None = None) -> set[str]:
    """VV監査で `merge` と判定された語の集合を返す。

    `keep`（形態素境界をまたぐ正当な母音連続）と `uncertain`（人手判断待ち）は
    統合しない。統合しすぎる方が統合し損ねるより誤差指標への害が大きいため
    （偽の境界を消すと誤差が過小評価される）。
    """
    path = path or RESULTS_DIR / "vv_audit_classified.tsv"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {
            r["word"]
            for r in csv.DictReader(fh, delimiter="\t")
            if r.get("verdict") == "merge"
        }

# MFAが無音・非音声に使うラベル。評価対象外。
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>", "silence"}

# RQ1の誤差解剖から除外するクラス（mapping/mfa_phone_class.tsv の方針）
EXCLUDED_CLASSES = {"non_phonemic"}


def load_phone_classes(path: Path | None = None) -> dict[str, str]:
    """mapping/mfa_phone_class.tsv を読む。`#` 行はコメント。"""
    path = path or MAPPING_DIR / "mfa_phone_class.tsv"
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return {
        r["mfa_phone"].strip(): r["category"].strip()
        for r in csv.DictReader(lines, delimiter="\t")
        if r.get("mfa_phone")
    }


def _read_textgrid(path: Path):
    """praatio でTextGridを開き、(words, phones) の区間リストを返す。"""
    try:
        from praatio import textgrid as tgio
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "praatio が見つからない。MFA環境で実行すること: conda activate mfa"
        ) from exc

    tg = tgio.openTextgrid(str(path), includeEmptyIntervals=False)
    names = {n.lower(): n for n in tg.tierNames}
    if "phones" not in names:
        raise ValueError(f"{path.name}: phones tier がない（tiers={tg.tierNames}）")
    phones = tg.getTier(names["phones"]).entries
    words = tg.getTier(names["words"]).entries if "words" in names else []
    return words, phones


def _word_at(words, t_start: float, t_end: float) -> tuple[str | None, int]:
    """音素区間の中点を含む単語を返す。見つからなければ (None, -1)。"""
    mid = (t_start + t_end) / 2.0
    for idx, w in enumerate(words):
        if w.start <= mid < w.end:
            return (w.label or None), idx
    return None, -1


def convert_file(
    path: Path, phone_classes: dict[str, str], corpus: Corpus,
    metric: MetricType, style: Style, speaker_id: str,
    vv_merge_words: set[str] | None = None,
) -> tuple[list[Unit], list[str]]:
    vv_merge_words = vv_merge_words or set()
    file_id = path.stem
    words, phones = _read_textgrid(path)
    units: list[Unit] = []
    unknown: list[str] = []

    word_counters: dict[int, int] = {}

    # --- VV統合（修正3a）: 同一語内で同一短母音が隣接し、その語がVV監査で merge 判定
    #     ならば、MFA側も1区間に統合して参照側の長母音1単位と対応づける。
    merged_intervals = []
    skip_next = False
    for k, iv in enumerate(phones):
        if skip_next:
            skip_next = False
            continue
        nxt = phones[k + 1] if k + 1 < len(phones) else None
        lab = (iv.label or "").strip()
        if nxt is not None and lab in SHORT_VOWELS and (nxt.label or "").strip() == lab:
            w_cur, _ = _word_at(words, iv.start, iv.end)
            w_nxt, _ = _word_at(words, nxt.start, nxt.end)
            if w_cur is not None and w_cur == w_nxt and w_cur in vv_merge_words:
                merged_intervals.append((iv.start, nxt.end, LONG_OF[lab], 2))
                skip_next = True
                continue
        merged_intervals.append((iv.start, iv.end, lab, 1))
    phones = merged_intervals

    for idx, iv in enumerate(phones):
        start_t, end_t, label, n_merged = iv
        is_silence = label in SILENCE_LABELS

        phone_class = phone_classes.get(label)
        if phone_class is None and not is_silence:
            # 写像表にない音素は黙って通さない。まとめて報告する。
            unknown.append(label)
            phone_class = "UNKNOWN"

        excluded = None
        if is_silence:
            excluded = ExclusionReason.UNMAPPED_LABEL
        elif phone_class in EXCLUDED_CLASSES or phone_class == "UNKNOWN":
            excluded = ExclusionReason.UNMAPPED_LABEL

        word, word_idx = _word_at(words, start_t, end_t)
        if word_idx >= 0:
            pos = word_counters.get(word_idx, 0)
            word_counters[word_idx] = pos + 1
        else:
            pos = None

        units.append(
            Unit(
                file_id=file_id,
                speaker_id=speaker_id,
                corpus=corpus,
                style=style,
                metric_type=metric,
                ipu_id=None,
                ipu_start=None,
                ipu_end=None,
                unit_id=f"{file_id}_mfa_{idx:04d}",
                t_start=float(start_t),
                t_end=float(end_t),
                label_canonical=label,
                label_source=label,  # MFA音素は既に正規形
                phone_class=phone_class or "silence",
                # MFAが主張する境界。人手かどうかは metric_type が担う（schema.py 参照）
                boundary_source_start=BoundarySource.MEASURED,
                boundary_source_end=BoundarySource.MEASURED,
                merged_from=n_merged,  # 通常1。VV統合が効いた場合のみ2
                excluded_reason=excluded,
                word=word,
                word_phone_index=pos,
            )
        )

    return units, unknown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("textgrid_dir", type=Path)
    ap.add_argument("--corpus", choices=[c.value for c in Corpus], required=True)
    ap.add_argument("--metric", choices=[m.value for m in MetricType], required=True)
    ap.add_argument("--style", choices=[s.value for s in Style], default=Style.NA.value)
    ap.add_argument("--speaker-id", default="jsut_single_speaker")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    paths = sorted(args.textgrid_dir.rglob("*.TextGrid"))
    if not paths:
        print(f"[error] no TextGrid under {args.textgrid_dir}", file=sys.stderr)
        return 1

    phone_classes = load_phone_classes()
    vv_merge_words = load_vv_merge_words()
    print(f'VV merge 対象語: {len(vv_merge_words)}語')
    all_units: list[Unit] = []
    unknown: dict[str, int] = {}

    for p in paths:
        u, unk = convert_file(
            p, phone_classes, Corpus(args.corpus), MetricType(args.metric),
            Style(args.style), args.speaker_id, vv_merge_words,
        )
        all_units.extend(u)
        for label in unk:
            unknown[label] = unknown.get(label, 0) + 1

    if unknown:
        print("[error] mapping/mfa_phone_class.tsv にない音素:", file=sys.stderr)
        for label, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
            print(f"  {label!r}  {n}件", file=sys.stderr)
        return 1

    problems = validate(all_units)
    if problems:
        print(f"[error] スキーマ検証で {len(problems)} 件の問題:", file=sys.stderr)
        for p in problems[:10]:
            print(f"  {p}", file=sys.stderr)
        return 1

    write_units(all_units, args.out)

    from collections import Counter
    by_class = Counter(u.phone_class for u in all_units)
    print(f"TextGrids        : {len(paths)}")
    print(f"units            : {len(all_units)}")
    print(f"  excluded       : {sum(1 for u in all_units if u.excluded_reason)}")
    print(f"  analyzable     : {sum(1 for u in all_units if u.is_analyzable)}")
    print(f"  with word ctx  : {sum(1 for u in all_units if u.word)}")
    print(f"  VV merged      : {sum(1 for u in all_units if u.merged_from > 1)}")
    print()
    print("音素クラス上位:")
    for k, v in by_class.most_common(10):
        print(f"  {k:<22} {v:>7}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
