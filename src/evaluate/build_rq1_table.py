# SCOPE: shared
"""RQ1（音素クラス別誤差）の分析入力テーブルを組む。

sequence_align.py の出力（境界誤差・4類型）に、参照 unit の属性
（スタイル・話者・後続環境・IPUタイミング）を結合し、R側の混合モデルが
そのまま読める long 形式のTSVにする。

モデル（CLAUDE.md §6、2026-07-20 簡略化後）:
    log(誤差) ~ 音素クラス * スタイル + 局所話速 + 境界タイプ + (1 | 話者) + (1 | 音素)

出力の1行 = 1境界。各評価単位は onset と offset の2境界を持つので、
boundary_type 列で区別して2行に展開する（モデルの「境界タイプ」因子）。

局所話速
--------
IPU内の分析可能な音素数 / IPU実時間（articulation rate 相当）。IPUは原則200ms以上の
ポーズで区切られるので、IPU内は概ね無ポーズ区間とみなせる（RQ2の調音速度に近い）。
RQ2の厳密な発話速度／調音速度の分離はジャーナル版で別途行う（ここは局所話速の代理）。

投入データ
----------
**発話フィルタを適用しない全量**（docs/decisions-log.md「RQ1は発話フィルタを適用しない
全量で行う」）。RQ0の先行研究比較で使う短発話除外は、RQ1には適用しない。

displacement 以外の扱い
-----------------------
誤差の大きさ（連続値）はマッチしたペア（displacement / substitution）にのみ定義される。
omission / insertion は「誤差 = 位置ずれ」を持たないため、log(誤差) のモデルには入れない。
ただし別カラム error_type を残し、R側で類型の生起率（ロジスティック）を別途分析できるようにする。

使い方
------
    python3 src/evaluate/build_rq1_table.py \\
        --alignment results/csj_alignment.tsv \\
        --units results/csj_units.tsv \\
        --out results/rq1_table.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))
from paths import assert_writable  # noqa: E402

# 誤差の大きさが定義される類型（マッチしたペア）
MATCHED_TYPES = {"displacement", "substitution"}


def load_units(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return {r["unit_id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def local_speech_rate(units: dict[str, dict[str, str]]) -> dict[tuple[str, str], float]:
    """(file_id, ipu_id) → 局所話速（分析可能音素数 / IPU実時間, 個/秒）。"""
    count: dict[tuple[str, str], int] = {}
    dur: dict[tuple[str, str], float] = {}
    for u in units.values():
        if u.get("excluded_reason"):
            continue
        key = (u["file_id"], u["ipu_id"])
        count[key] = count.get(key, 0) + 1
        if key not in dur and u.get("ipu_start") and u.get("ipu_end"):
            dur[key] = float(u["ipu_end"]) - float(u["ipu_start"])
    rate: dict[tuple[str, str], float] = {}
    for key, n in count.items():
        d = dur.get(key, 0.0)
        if d > 0:
            rate[key] = n / d
    return rate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--units", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    units = load_units(args.units)
    rate = local_speech_rate(units)

    out_rows: list[dict[str, object]] = []
    n_no_unit = 0
    with args.alignment.open(encoding="utf-8") as fh:
        for a in csv.DictReader(fh, delimiter="\t"):
            ref_id = a["ref_unit_id"]
            if not ref_id:  # insertion（参照なし）
                out_rows.append(_row(a, None, None, "insertion_marker"))
                continue
            u = units.get(ref_id)
            if u is None:
                n_no_unit += 1
                continue
            key = (u["file_id"], u["ipu_id"])
            lsr = rate.get(key)
            if a["error_type"] in MATCHED_TYPES:
                # onset と offset を別行に展開（境界タイプ因子）
                for btype, col in (("onset", "onset_error_ms"), ("offset", "offset_error_ms")):
                    val = a.get(col)
                    if val:
                        out_rows.append(_row(a, u, lsr, btype, abs(float(val))))
            else:
                # omission は類型分析用に1行だけ残す（誤差なし）
                out_rows.append(_row(a, u, lsr, "none"))

    cols = [
        "file_id", "ipu_id", "speaker_id", "style", "phone_class",
        "error_type", "boundary_type", "abs_error_ms", "local_speech_rate",
        "following_context", "devoiced", "tag_vlong", "tag_clong",
        "ref_label_canonical", "n_hyp_units",
    ]
    assert_writable(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    from collections import Counter
    bt = Counter(r["boundary_type"] for r in out_rows)
    print(f"alignment rows in     : (読み込み)")
    print(f"output rows (long)    : {len(out_rows)}")
    print(f"  boundary_type       : {dict(bt)}")
    print(f"  参照unit未発見       : {n_no_unit}")
    matched = [r for r in out_rows if r["boundary_type"] in ("onset", "offset")]
    print(f"  誤差つき境界         : {len(matched)}")
    print(f"wrote {args.out}")
    return 0


def _row(a, u, lsr, btype, err=None):
    return {
        "file_id": a["file_id"],
        "ipu_id": u["ipu_id"] if u else "",
        "speaker_id": u["speaker_id"] if u else "",
        "style": u["style"] if u else "",
        "phone_class": a["phone_class"],
        "error_type": a["error_type"],
        "boundary_type": btype,
        "abs_error_ms": f"{err:.3f}" if err is not None else "",
        "local_speech_rate": f"{lsr:.4f}" if lsr else "",
        "following_context": u["following_context"] if u else "",
        "devoiced": u["devoiced"] if u else "",
        "tag_vlong": u["tag_vlong"] if u else "",
        "tag_clong": u["tag_clong"] if u else "",
        "ref_label_canonical": a["ref_label_canonical"],
        "n_hyp_units": a["n_hyp_units"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
