# SCOPE: journal-only（査読対応。事後。docs/decisions-log.md 2026-09-16 J1–J6）
"""J4 前段: vowel_measures.py の包含規則を件数だけ数えるモードで再現し、語の識別子を付けた
側表（整数コードのみ）を作る。

- 規則は src/acoustics/vowel_measures.py と同一の順序で当てる（omission → 参照なし →
  アクセント型なし → 無声化 → MFA 単位なし）。本体は変更しない。
- 側表は rq3_vowel_table.tsv と行順が一致することを、持続時間の列で全行照合して確かめる。
  語は文字列を持たず、(file_id に依らない) 語形の整数コードと、講演内の語トークン番号の整数コードだけを書く。
  側表はトークン単位なのでリポジトリ外（引数 --side-out、既定はセッションの一時領域）に置く。

出力（集計値）: results/revision_2026-09/j4_exclusions.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "revision_2026-09"
VOWEL_CLASSES = {"vowel_short", "vowel_long"}
csv.field_size_limit(sys.maxsize)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side-out", type=Path, required=True)
    args = ap.parse_args(argv)

    csj: dict[str, dict[str, str]] = {}
    total_vowel_units = Counter()
    with (RES / "csj_units.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["style"] not in ("aps", "sps"):
                continue
            if r["phone_class"] in VOWEL_CLASSES:
                ev = (not r["excluded_reason"] and r["boundary_source_start"] == "measured"
                      and r["boundary_source_end"] == "measured")
                total_vowel_units["all"] += 1
                total_vowel_units["evaluable" if ev else "not_evaluable"] += 1
                total_vowel_units["accent_type_blank"] += int(not r.get("accent_type"))
                total_vowel_units["devoiced_flag"] += int(r.get("devoiced") == "1")
                total_vowel_units["word_blank"] += int(not r.get("word"))
            csj[r["unit_id"]] = {k: r[k] for k in ("file_id", "accent_type", "devoiced", "word",
                                                   "word_phone_index", "is_accent_nucleus",
                                                   "t_start", "t_end", "ipu_id")}
    mfa: dict[str, tuple[float, float]] = {}
    with (RES / "mfa_csj_units.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            mfa[r["unit_id"]] = (float(r["t_start"]), float(r["t_end"]))

    c = Counter()
    side = []
    word_codes: dict[str, int] = {}
    with (RES / "csj_alignment_main.tsv").open(encoding="utf-8") as fh:
        for a in csv.DictReader(fh, delimiter="\t"):
            if a["phone_class"] not in VOWEL_CLASSES:
                continue
            c["alignment_vowel_rows"] += 1
            if a["error_type"] not in ("displacement", "substitution"):
                c[f"excluded_{a['error_type']}"] += 1
                continue
            ref = csj.get(a["ref_unit_id"])
            if ref is None:
                c["excluded_ref_unit_missing"] += 1
                continue
            if not ref.get("accent_type"):
                c["excluded_no_accent_type"] += 1
                continue
            if ref.get("devoiced") == "1":
                c["excluded_devoiced"] += 1
                continue
            hyps = [mfa[h] for h in a["hyp_unit_id"].split(";") if h and h in mfa]
            if not hyps:
                c["excluded_no_mfa_unit"] += 1
                continue
            c["included"] += 1
            c[f"included_{a['error_type']}"] += 1
            c[f"included_{a['phone_class']}"] += 1
            c[f"included_nucleus_{ref['is_accent_nucleus'] or '0'}"] += 1
            hand = (float(ref["t_end"]) - float(ref["t_start"])) * 1000
            mdur = (max(h[1] for h in hyps) - min(h[0] for h in hyps)) * 1000
            w = ref["word"]
            wc = word_codes.setdefault(w, len(word_codes) + 1) if w else 0
            side.append((f"{hand:.3f}", f"{mdur:.3f}", wc))
    # 行順・値の照合
    mism = 0
    n_tab = 0
    with (RES / "rq3_vowel_table.tsv").open(encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter="\t")):
            n_tab += 1
            if i >= len(side) or (r["dur_hand_ms"], r["dur_mfa_ms"]) != side[i][:2]:
                mism += 1
    c["rq3_vowel_table_rows"] = n_tab
    c["row_mismatches_vs_rq3_vowel_table"] = mism
    c["distinct_word_forms_included"] = len(word_codes)
    c["included_word_blank"] = sum(1 for s in side if s[2] == 0)

    args.side_out.parent.mkdir(parents=True, exist_ok=True)
    with args.side_out.open("w", encoding="utf-8") as fh:
        fh.write("row\tword_code\n")
        for i, s in enumerate(side, 1):
            fh.write(f"{i}\t{s[2]}\n")

    rows = [("vowel_reference_units_177_talks", total_vowel_units["all"]),
            ("  of_which_evaluable_boundaries", total_vowel_units["evaluable"]),
            ("  of_which_not_evaluable", total_vowel_units["not_evaluable"]),
            ("  of_which_accent_type_blank", total_vowel_units["accent_type_blank"]),
            ("  of_which_devoiced_flag_1", total_vowel_units["devoiced_flag"]),
            ("  of_which_word_blank", total_vowel_units["word_blank"])]
    order = ["alignment_vowel_rows", "excluded_omission", "excluded_ref_unit_missing",
             "excluded_no_accent_type", "excluded_devoiced", "excluded_no_mfa_unit", "included",
             "included_displacement", "included_substitution", "included_vowel_short",
             "included_vowel_long", "included_nucleus_1", "included_nucleus_0",
             "rq3_vowel_table_rows", "row_mismatches_vs_rq3_vowel_table",
             "distinct_word_forms_included", "included_word_blank"]
    rows += [(k, c.get(k, 0)) for k in order]
    rows += [(k, v) for k, v in sorted(c.items()) if k not in order]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "j4_exclusions.tsv").open("w", encoding="utf-8") as fh:
        fh.write("step\tn\n")
        for k, v in rows:
            fh.write(f"{k}\t{v}\n")
            print(f"{k:<45} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
