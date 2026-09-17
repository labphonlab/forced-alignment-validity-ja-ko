#!/usr/bin/env python3
"""citation（規則未適用）側の発音仮説を点検する（2026-09-08 新設）。

`audit_variant_validity.py` は sandhi 側が宣言した過程を体現しているかを見る。
本スクリプトはその対で、**citation 側が音韻的に成立する形か**を見る。

判定基準: 阻害音終声（k̚ p̚ t̚）の直後に有声阻害音の初声が来る連続。
韓国語の平音は有声音間でのみ有声異音になるため、この連続は生じえない。
音節単位でG2Pを引くと、左右の音節が独立に音声化されるためこの誤りが混入する。

出力: カテゴリ別の違反率と、違反の有無で自動適用率が動くかどうか。
後者が小さければ、citation 側の欠陥は過小検出バイアスの主因ではない、と言える。
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = [".mfa_newcats_v2", ".mfa_balanced2_v2", ".mfa_crosseojeol_v3"]

CODA_OBSTRUENT = {"k̚", "p̚", "t̚"}
VOICED_OBSTRUENT = {"b", "bʲ", "bʷ", "d", "dʲ", "dʷ", "dʑ", "dʑʷ",
                    "ɟ", "ɡ", "ɡʷ", "ɣ", "β", "βʷ", "ʝ"}


def violations(phone_seq):
    """成立しない異音連続の位置を (index, 左, 右) で返す。"""
    p = str(phone_seq).split()
    return [(i, p[i - 1], p[i]) for i in range(1, len(p))
            if p[i - 1] in CODA_OBSTRUENT and p[i] in VOICED_OBSTRUENT]


def load():
    frames = []
    for run in RUNS:
        f = ROOT / run / "pilot_results.csv"
        if f.exists():
            d = pd.read_csv(f).dropna(subset=["citation", "sandhi"])
            d["_run"] = run
            frames.append(d)
    return pd.concat(frames).drop_duplicates(["token", "utterance_id"])


def main():
    d = load()
    d["cit_viol"] = d["citation"].map(lambda s: len(violations(s)))
    d["san_viol"] = d["sandhi"].map(lambda s: len(violations(s)))

    t = d.groupby("change_type").agg(
        n=("citation", "size"),
        citation違反=("cit_viol", lambda s: int((s > 0).sum())),
        sandhi違反=("san_viol", lambda s: int((s > 0).sum())))
    t["citation違反率"] = (t["citation違反"] / t["n"]).map("{:.1%}".format)
    print("═" * 66)
    print("citation 側の異音連続の点検（阻害音終声＋有声阻害音初声）")
    print("═" * 66)
    print(t.to_string())
    print(f"\n全体: {int((d.cit_viol > 0).sum()):,} / {len(d):,} 件 "
          f"({(d.cit_viol > 0).mean():.1%})")

    sub, rest = d[d.cit_viol > 0], d[d.cit_viol == 0]
    if len(sub):
        ap, ap0 = [(x["result"] == "SANDHI (rule applied)").mean() for x in (sub, rest)]
        print("\n判定への影響")
        print(f"  違反あり (n={len(sub):,}) 自動適用率 {ap:.1%}")
        print(f"  違反なし (n={len(rest):,}) 自動適用率 {ap0:.1%}")
        print(f"  差 {ap - ap0:+.1%}")


if __name__ == "__main__":
    main()
