# SCOPE: shared
"""RQ3（誤差伝播）の母音長測定。手作業境界版とMFA版の母音長を対にする。

RQ3は「手作業境界版・MFA版で同一モデルを並行実行し、固定効果推定値・信頼区間を
比較」する（CLAUDE.md §6）。ICPhS版は**母音長×アクセント型のみ**。本モジュールは
その入力（母音ごとに hand/MFA 両方の持続時間 + アクセント型）を作る。

問い: MFAの境界誤差は、母音長とアクセント型の関係の推定にどれだけ伝播するか。
  - hand 版: CSJ人手境界から測った母音長 ~ アクセント型 のモデル
  - MFA 版: MFA境界から測った母音長 ~ アクセント型 のモデル
  両者の固定効果を比べ、MFA使用時に関係が歪むかを見る。

対応付け
--------
sequence_align.py の出力（ref_unit_id ↔ hyp_unit_id）で母音を対にする。
hand 長 = CSJ unit の t_end - t_start、MFA 長 = 対応する MFA unit の t_end - t_start。
母音長は境界の差分で得られるので Parselmouth 不要（境界はCSJ/MFAが持つ）。ジャーナル版で
フォルマント等を測る際に Parselmouth を追加する。

除外
----
- 無声化融合母音: 内部境界に真値がないため母音長が測れない（segment.pdf §5.1）。csj側で除外済み。
- omission（対応するMFA母音がない）: MFA長が測れないため落とす。
- アクセント型が付与されていない母音（コア外・マスク等）: 落とす。
- 撥音が後続する母音は落とさず following_context で層別（RQ3層別。decisions.md）。

アクセントの由来（論文Limitationsに明記）
-----------------------------------------
知覚アクセント（辞書規範ではない）。CSJコアのみ・東京方言話者に限定。

使い方
------
    python3 src/acoustics/vowel_measures.py \\
        --alignment results/csj_alignment.tsv \\
        --csj-units results/csj_units.tsv \\
        --mfa-units results/mfa_csj_units.tsv \\
        --out results/rq3_vowel_table.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))
from paths import assert_writable  # noqa: E402

VOWEL_CLASSES = {"vowel_short", "vowel_long"}


def load_units(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return {r["unit_id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def dur(u: dict[str, str]) -> float:
    return float(u["t_end"]) - float(u["t_start"])


def measure_vowel_duration(unit: dict[str, str]) -> float:
    """1母音unitの持続時間（秒）。hand版・MFA版のどちらの入力でも同じ関数を通す
    （RQ3の比較可能性のため分岐しない、というスタブ時の設計意図を保持）。"""
    return dur(unit)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--csj-units", type=Path, required=True)
    ap.add_argument("--mfa-units", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    csj = load_units(args.csj_units)
    mfa = load_units(args.mfa_units)

    rows: list[dict[str, object]] = []
    n_no_accent = n_devoiced = n_nasal_ctx = 0

    with args.alignment.open(encoding="utf-8") as fh:
        for a in csv.DictReader(fh, delimiter="\t"):
            if a["phone_class"] not in VOWEL_CLASSES:
                continue
            if a["error_type"] not in ("displacement", "substitution"):
                continue  # 対応するMFA母音がない（omission）→ 母音長を測れない
            ref = csj.get(a["ref_unit_id"])
            if ref is None:
                continue
            if not ref.get("accent_type"):
                n_no_accent += 1
                continue
            if ref.get("devoiced") == "1":
                n_devoiced += 1
                continue

            hyp_ids = [h for h in a["hyp_unit_id"].split(";") if h]
            hyp_units = [mfa[h] for h in hyp_ids if h in mfa]
            if not hyp_units:
                continue
            mfa_start = min(float(h["t_start"]) for h in hyp_units)
            mfa_end = max(float(h["t_end"]) for h in hyp_units)
            mfa_dur = mfa_end - mfa_start

            following = ref.get("following_context") or ""
            if following == "moraic_nasal":
                n_nasal_ctx += 1

            hand_dur = measure_vowel_duration(ref)
            rows.append({
                "file_id": ref["file_id"],
                "speaker_id": ref["speaker_id"],
                "style": ref["style"],
                "vowel": a["ref_label_canonical"],
                "length_class": a["phone_class"],
                "accent_type": ref["accent_type"],
                "is_accent_nucleus": ref.get("is_accent_nucleus", "0"),
                "following_context": following,
                "dur_hand_ms": f"{hand_dur * 1000:.3f}",
                "dur_mfa_ms": f"{mfa_dur * 1000:.3f}",
                "dur_diff_ms": f"{(hand_dur - mfa_dur) * 1000:.3f}",
            })

    assert_writable(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["file_id", "speaker_id", "style", "vowel", "length_class",
            "accent_type", "is_accent_nucleus", "following_context",
            "dur_hand_ms", "dur_mfa_ms", "dur_diff_ms"]
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"母音ペア（hand/MFA対）: {len(rows)}")
    print(f"  アクセント型なしで除外 : {n_no_accent}")
    print(f"  無声化で除外           : {n_devoiced}")
    print(f"  撥音後続（層別対象）   : {n_nasal_ctx}")
    print(f"  長さクラス             : {dict(Counter(r['length_class'] for r in rows))}")
    print(f"  アクセント型           : {dict(sorted(Counter(r['accent_type'] for r in rows).items()))}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
