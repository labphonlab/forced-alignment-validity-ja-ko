# SCOPE: journal-only（査読対応。事後。docs/decisions-log.md 2026-09-16 J1–J6）
"""J1: Table 1 の分母と共有境界の割合。

入力（実行するだけで中身は表示しない）:
  results/csj_units.tsv, results/mfa_csj_units.tsv, results/csj_alignment_main.tsv
出力（集計値のみ）:
  results/revision_2026-09/j1_denominators.tsv
  results/revision_2026-09/j1_shared_boundaries.tsv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = RES / "revision_2026-09"

TABLE1 = ["vowel_short", "vowel_long", "stop", "fricative", "affricate", "nasal", "flap",
          "approximant", "moraic_nasal", "moraic_nasal_fused", "geminate_stop", "devoiced_fused"]
TOL = 0.0005  # 秒。時刻の一致とみなす幅


def main() -> int:
    u = pd.read_csv(RES / "csj_units.tsv", sep="\t", dtype=str, keep_default_na=False,
                    usecols=["file_id", "style", "ipu_id", "unit_id", "t_start", "t_end",
                             "phone_class", "excluded_reason", "boundary_source_start",
                             "boundary_source_end"])
    u = u[u["style"].isin(["aps", "sps"])].copy()
    u["t_start"] = u["t_start"].astype(float)
    u["t_end"] = u["t_end"].astype(float)
    u["evaluable"] = ((u["excluded_reason"] == "") & (u["boundary_source_start"] == "measured")
                      & (u["boundary_source_end"] == "measured"))
    print(f"talks={u.file_id.nunique()} units={len(u)} evaluable={int(u.evaluable.sum())}")

    a = pd.read_csv(RES / "csj_alignment_main.tsv", sep="\t", dtype=str, keep_default_na=False,
                    usecols=["file_id", "ref_unit_id", "hyp_unit_id", "phone_class", "error_type"])
    print(f"alignment rows={len(a)} talks={a.file_id.nunique()}")
    # csj_alignment_main.tsv は READ 6講演の insertion 行（参照なし）を含む（2026-09-16 判明）。
    # 主分析の177講演に限定する。
    main_talks = set(u.file_id)
    n_read = int((~a.file_id.isin(main_talks)).sum())
    types_read = a[~a.file_id.isin(main_talks)].error_type.value_counts().to_dict()
    print(f"rows from non-main (READ) talks dropped: {n_read} {types_read}")
    a = a[a.file_id.isin(main_talks)].copy()

    # --- 分母 ---
    ref_all = u.groupby("phone_class").size()
    ref_eval = u[u.evaluable].groupby("phone_class").size()
    et = a[a.error_type != "insertion"].groupby(["phone_class", "error_type"]).size().unstack(fill_value=0)
    ins = a[a.error_type == "insertion"].groupby("phone_class").size()
    classes = sorted(set(ref_all.index) | set(et.index) | set(ins.index))
    rows = []
    for c in classes:
        r = {
            "phone_class": c,
            "in_table1": c in TABLE1,
            "ref_segments_all": int(ref_all.get(c, 0)),
            "ref_segments_evaluable": int(ref_eval.get(c, 0)),
            "displacement": int(et.get("displacement", pd.Series(dtype=int)).get(c, 0)),
            "substitution": int(et.get("substitution", pd.Series(dtype=int)).get(c, 0)),
            "omission": int(et.get("omission", pd.Series(dtype=int)).get(c, 0)),
            "insertion_by_mfa_class": int(ins.get(c, 0)),
        }
        rows.append(r)
    df = pd.DataFrame(rows)

    def total(sub: pd.DataFrame, name: str) -> dict:
        s = sub.drop(columns=["phone_class", "in_table1"]).sum()
        return {"phone_class": name, "in_table1": "", **{k: int(v) for k, v in s.items()}}

    df = pd.concat([
        df[df.in_table1].set_index("phone_class").loc[TABLE1].reset_index(),
        pd.DataFrame([total(df[df.in_table1], "TOTAL_table1_classes")]),
        df[~df.in_table1],
        pd.DataFrame([total(df[~df.in_table1], "TOTAL_other_classes"),
                      total(df[df.in_table1 != ""], "TOTAL_all_classes")]),
    ], ignore_index=True)
    df["ref_rows_in_alignment"] = df.displacement + df.substitution + df.omission
    df["matched_in_table1"] = df.displacement + df.substitution
    df["pct_of_evaluable_in_table1"] = 100 * df.matched_in_table1 / df.ref_segments_evaluable
    df["pct_of_all_in_table1"] = 100 * df.matched_in_table1 / df.ref_segments_all
    df["pct_evaluable_omitted"] = 100 * df.omission / df.ref_segments_evaluable
    df["pct_evaluable_substituted"] = 100 * df.substitution / df.ref_segments_evaluable
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "j1_denominators.tsv", sep="\t", index=False, float_format="%.3f")
    print(df.to_string())
    chk = df[df.phone_class == "TOTAL_all_classes"].iloc[0]
    print(f"check: evaluable ref units={chk.ref_segments_evaluable}, ref rows in alignment={chk.ref_rows_in_alignment}")

    # --- 共有境界 ---
    ev = u[u.evaluable].sort_values(["file_id", "t_start", "t_end"]).reset_index(drop=True)
    same = (ev.file_id.shift() == ev.file_id) & (ev.ipu_id.shift() == ev.ipu_id)
    ev["prev_uid"] = np.where(same, ev.unit_id.shift(), "")
    ev["prev_t_end"] = np.where(same, ev.t_end.shift(), np.nan)

    m = a[a.error_type.isin(["displacement", "substitution"])].copy()
    hyp = pd.read_csv(RES / "mfa_csj_units.tsv", sep="\t", dtype=str, keep_default_na=False,
                      usecols=["unit_id", "t_start", "t_end"])
    hs = dict(zip(hyp.unit_id, hyp.t_start.astype(float)))
    he = dict(zip(hyp.unit_id, hyp.t_end.astype(float)))
    del hyp
    ids = m.hyp_unit_id.str.split(";")
    m["h_start"] = ids.str[0].map(hs)
    m["h_end"] = ids.str[-1].map(he)
    info = ev.set_index("unit_id")[["t_start", "prev_uid", "prev_t_end"]]
    m = m.join(info, on="ref_unit_id")
    mh = m.set_index("ref_unit_id")[["h_end"]].rename(columns={"h_end": "prev_h_end"})
    m = m.join(mh, on="prev_uid")
    m["prev_matched"] = m.prev_h_end.notna()
    m["shared_ref"] = m.prev_matched & ((m.t_start - m.prev_t_end).abs() <= TOL)
    m["shared_both"] = m.shared_ref & ((m.h_start - m.prev_h_end).abs() <= TOL)
    m["tab1"] = m.phone_class.isin(TABLE1)
    out = []
    for name, sub in (("all_classes", m), ("table1_classes", m[m.tab1])):
        n = len(sub)
        out.append({
            "scope": name, "onset_rows": n,
            "has_preceding_segment_in_ipu": int((sub.prev_uid != "").sum()),
            "preceding_segment_matched": int(sub.prev_matched.sum()),
            "ref_onset_equals_prev_ref_offset": int(sub.shared_ref.sum()),
            "pct_shared_ref_of_onset_rows": 100 * sub.shared_ref.sum() / n,
            "also_mfa_boundary_shared": int(sub.shared_both.sum()),
            "pct_identical_error_counted_twice_of_onset_rows": 100 * sub.shared_both.sum() / n,
        })
    so = pd.DataFrame(out)
    so.to_csv(OUT / "j1_shared_boundaries.tsv", sep="\t", index=False, float_format="%.3f")
    print(so.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
