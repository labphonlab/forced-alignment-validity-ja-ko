# SCOPE: shared
"""査読対応: RQ2の accent-nucleus / following-nasal 条件別に、符号付き境界誤差
(MFA - manual, ms) と符号付き持続時間差 (hand - MFA, ms) を集計する。

絶対誤差（§3.1/Table 1）だけでは RQ2 の係数がどちらの向きに動くかを説明できない
という査読指摘に対応する分析。出力は docs/supplement/supplement.md の
「Signed boundary and duration errors」節の数値と対応する。

入力:
  results/csj_alignment_main.tsv （ref_unit_id, phone_class, error_type,
    onset_error_ms, offset_error_ms）
  results/csj_units.tsv （unit_id, phone_class, accent_type, is_accent_nucleus,
    devoiced, following_context）
  results/rq3_vowel_table.tsv （is_accent_nucleus, following_context, dur_diff_ms）

出力: results/signed_error_by_condition.tsv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"

VOWEL_CLASSES = {"vowel_short", "vowel_long", "short vowel", "long vowel"}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def signed_boundary_errors() -> dict[tuple[str, bool], dict[str, list[float]]]:
    units = {}
    with open(RESULTS / "csj_units.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["phone_class"] in VOWEL_CLASSES:
                units[row["unit_id"]] = row

    out: dict[tuple[str, bool], dict[str, list[float]]] = defaultdict(
        lambda: {"onset": [], "offset": []}
    )
    with open(RESULTS / "csj_alignment_main.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["phone_class"] not in VOWEL_CLASSES:
                continue
            if row["error_type"] not in ("displacement", "substitution"):
                continue
            u = units.get(row["ref_unit_id"])
            if u is None or not u.get("accent_type") or u.get("devoiced") == "1":
                continue
            try:
                oe = float(row["onset_error_ms"])
                fe = float(row["offset_error_ms"])
            except (ValueError, KeyError):
                continue
            is_nuc = u.get("is_accent_nucleus", "") in ("1", "TRUE", "True")
            is_nasal = u.get("following_context", "") == "moraic_nasal"
            out[("nucleus", is_nuc)]["onset"].append(oe)
            out[("nucleus", is_nuc)]["offset"].append(fe)
            out[("nasal", is_nasal)]["onset"].append(oe)
            out[("nasal", is_nasal)]["offset"].append(fe)
    return out


def signed_duration_diff() -> dict[tuple[str, bool], list[float]]:
    out: dict[tuple[str, bool], list[float]] = defaultdict(list)
    with open(RESULTS / "rq3_vowel_table.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                d = float(row["dur_diff_ms"])
            except (ValueError, KeyError):
                continue
            is_nuc = row.get("is_accent_nucleus", "") in ("1", "TRUE", "True")
            is_nasal = row.get("following_context", "") == "moraic_nasal"
            out[("nucleus", is_nuc)].append(d)
            out[("nasal", is_nasal)].append(d)
    return out


def main() -> None:
    onoff = signed_boundary_errors()
    durdiff = signed_duration_diff()

    rows = []
    for key in [("nucleus", True), ("nucleus", False), ("nasal", True), ("nasal", False)]:
        on = onoff[key]["onset"]
        off = onoff[key]["offset"]
        dd = durdiff[key]
        rows.append(
            {
                "condition": f"{key[0]}={key[1]}",
                "n_matched_vowels": len(on),
                "onset_error_mean_ms": round(mean(on), 2),
                "onset_error_median_ms": round(median(on), 2),
                "offset_error_mean_ms": round(mean(off), 2),
                "offset_error_median_ms": round(median(off), 2),
                "n_dur_tokens": len(dd),
                "dur_diff_hand_minus_mfa_mean_ms": round(mean(dd), 2),
                "dur_diff_hand_minus_mfa_median_ms": round(median(dd), 2),
            }
        )

    out_path = RESULTS / "signed_error_by_condition.tsv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
