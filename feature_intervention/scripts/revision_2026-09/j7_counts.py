# SCOPE: 改訂用の事後解析（docs/decisions-log.md D17）
"""J7 前段: 14系統について、講演×群（T・C）の 10 ms 以内件数と、系統ごとの T・C・範疇判定を出す。
02_evaluate.py の collect() 等をそのまま呼ぶ（改変しない）。

    PY=python3
    $PY scripts/revision_2026-09/j7_counts.py

出力:
  work/revision_2026-09/talk_group_counts.tsv（講演単位の件数）
  results/revision_2026-09/j7_systems_run_level.tsv（系統ごとの集計値）
"""
import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "02_evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

SYSTEMS = {  # 系統 → (比較の組, 条件)
    "B0": ("public", "B0"), "B1": ("public", "B1"),
    "A": ("A_vs_B", "A"), "A_r2": ("A_vs_B", "A"), "A_r3": ("A_vs_B", "A"),
    "B": ("A_vs_B", "B"), "B_r2": ("A_vs_B", "B"), "B_r3": ("A_vs_B", "B"),
    "A2": ("A2_vs_B2", "A2"), "A2_r2": ("A2_vs_B2", "A2"), "A2_r3": ("A2_vs_B2", "A2"),
    "B2": ("A2_vs_B2", "B2"), "B2_r2": ("A2_vs_B2", "B2"), "B2_r3": ("A2_vs_B2", "B2"),
}

talks = ev.test_talks()
info = ev.load_ref_info()
cnt_rows, sys_rows = [], []
for s, (pair, cond) in SYSTEMS.items():
    st = ev.collect(s, info)
    for t in talks:
        for g in ("T", "C"):
            h, n = st.talk_group.get((t, g), [0, 0])
            cnt_rows.append([pair, cond, s, t, g, h, n - h, n])
    cm = ev.cat_metrics(st.cat, talks, "all")
    cv = ev.cat_metrics(st.cat, talks, "voiceless_ctx")
    sys_rows.append([pair, cond, s, round(ev.rate(st.talk_group, talks, "T"), 3),
                     round(ev.rate(st.talk_group, talks, "C"), 3),
                     round(ev.rate(st.talk_group, talks, "ALL"), 3),
                     round(cm["sensitivity"], 3), round(cm["specificity"], 3), round(cm["balanced_accuracy"], 3),
                     round(cv["specificity"], 3), round(cv["balanced_accuracy"], 3),
                     cm["n_ref_devoiced"], cm["n_ref_voiced"], st.cat_excluded])
    print(s, sys_rows[-1][3:9])

out = ROOT / "work" / "revision_2026-09" / "talk_group_counts.tsv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["pair", "condition", "run", "talk", "classgroup", "hit", "miss", "n"])
    w.writerows(cnt_rows)
res = ROOT / "results" / "revision_2026-09" / "j7_systems_run_level.tsv"
res.parent.mkdir(parents=True, exist_ok=True)
with res.open("w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(["pair", "condition", "system", "T_within10", "C_within10", "ALL_within10",
                "sensitivity", "specificity", "balanced_accuracy",
                "specificity_voiceless_ctx", "balanced_accuracy_voiceless_ctx",
                "n_ref_devoiced", "n_ref_voiced", "cat_excluded"])
    w.writerows(sys_rows)
print("wrote", out, res)
