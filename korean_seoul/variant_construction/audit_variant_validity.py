#!/usr/bin/env python3
"""変異形競合法に与えた2つの発音仮説が、宣言したカテゴリの対立を実際に作っているかを検査する。

## なぜ必要か（2026-08-31 発見）

変異形競合法は、各候補語について「引用形（規則が適用されていない発音）」と
「適用形（適用された発音）」をMFAの発音辞書に等確率で登録し、どちらが選ばれるかを
音韻過程の生起判定とみなす。この設計は、**2つの仮説が実際にその音韻過程の対立を
作っている**ことを前提とする。

しかし辞書を点検すると、この前提が成り立たない項目がある。適用形は規則から導いた
形ではなく **G2P の出力** をそのまま用いており、宣言したカテゴリの変化を含まない
ことがある。例（`.mfa_newcats/variant_dict.txt`）:

    맛보는  m ɐ t̚ p o n ɨ n   /  m ɐ t̚ pʰ o n ɨ n   ← 濃音ではなく激音の対立
    갖고는  k ɐ t̚ k o n ɨ n   /  k ɐ ɡ o n ɨ n      ← 終声の有無と有声化の対立
    앞세우는 ɐ p̚ sʰ e u n ɨ n  /  ɐ p s e u n ɨ n     ← 開放・帯気の記号差のみ

これらの項目でMFAが答えているのは「濃音化が起きたか」ではない。したがって
その判定を濃音化の測定値として集計することはできない。

## 使い方

    python3 audit_variant_validity.py            # 検査のみ
    python3 audit_variant_validity.py --recompute  # 妥当な項目のみで一致度を再計算
"""
import argparse
import os
import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(os.environ.get("KOREAN_SOUND_CHANGE_ROOT", ".")).expanduser().resolve()

TENSE = {"p͈", "t͈", "k͈", "s͈", "tɕ͈", "c͈", "ɕ͈"}
ASP = {"pʰ", "tʰ", "kʰ", "tɕʰ", "cʰ"}
NAS = {"m", "n", "ŋ", "ɲ"}
LIQ = {"ɭ", "ʎ", "l"}
PAL = {"tɕ", "tɕʰ", "dʑ"}


# 子音を「素の類」に正規化する表。有声・二次調音・不放音・長さの差を消す。
CLS = {"k": "K", "ɡ": "K", "ɣ": "K", "k̚": "K", "c": "K", "ɟ": "K",
       "kʰ": "KH", "cʰ": "KH", "k͈": "KK", "x": "H",
       "t": "T", "d": "T", "t̚": "T", "tʰ": "TH", "t͈": "TT",
       "p": "P", "b": "P", "p̚": "P", "pʰ": "PH", "p͈": "PP",
       "β": "H", "ɸ": "H",
       "s": "S", "sʰ": "S", "ɕʰ": "S", "s͈": "SS", "ɕ͈": "SS",
       "tɕ": "C", "dʑ": "C", "tɕʰ": "CH", "tɕ͈": "CC",
       "h": "H", "ɦ": "H", "ç": "H", "ʝ": "H",
       "m": "M", "n": "N", "ɲ": "N", "ŋ": "NG",
       "ɾ": "R", "l": "R", "ɭ": "R", "ʎ": "R"}


def _classes(seq):
    import collections
    c = collections.Counter()
    for raw in seq:
        p = raw.replace("ː", "")
        if len(p) > 1 and p[-1] in "ʲʷ":
            p = p[:-1]
        k = CLS.get(p)
        if k:
            c[k] += 2 if "ː" in raw else 1
    return c


def liaison_is_valid(citation: str, sandhi: str) -> bool:
    """連音化は**再音節化**なので、子音の類は保存されるはずである。

    以前は「2形が違えば対立あり」としていたが、これでは終声が単に脱落した項目
    （신호를읽어: ɾ ɨ ɭ i → ɾ ɨ i）も妥当と数えてしまう。子音の類の多重集合が
    減っていないことを条件にする。/h/ の脱落だけは規則として許す。
    なお終声と滑音が1音素に融合する場合（ɭ + ɥ → ɾʷ）は音素数が減るが
    子音の類は保たれるので、正しく妥当と判定される。
    """
    lost = _classes(str(citation).split()) - _classes(str(sandhi).split())
    lost.pop("H", None)
    return (not lost) and str(citation) != str(sandhi)


def _norm(seq):
    """長音記号と二次調音を落として素の音素にする。"""
    out = []
    for p in seq:
        p = p.replace("ː", "")
        if len(p) > 1 and p[-1] in "ʲʷ":
            p = p[:-1]
        out.append(p)
    return out


def _count(seq, S):
    """該当する音素の数。重子音（ː 付き）は終声＋初声の2つ分として数える。

    例: 업무를 の適用形 `ʌ mː u ɾ ɨ ɭ` は ㅁ+ㅁ であり、引用形 `ʌ p̚ m u ɾ ɨ ɭ`
    より鼻音が1つ増えている。ː を単なる長さとして1つに数えると鼻音化を見落とす。
    """
    n = 0
    for raw, p in zip(seq, _norm(seq)):
        if p in S:
            n += 2 if "ː" in raw else 1
    return n


def variant_is_valid(cat: str, citation: str, sandhi: str) -> bool:
    """適用形が、宣言されたカテゴリの変化を実際に含むか。

    **存在ではなく個数で比べる。** 語中の別の位置に既に激音や破擦音があると、
    「適用形に含まれ引用形に含まれない」という条件では偽陰性が出る
    （例: 파악할 は引用形にも pʰ がある）。
    """
    c, s = str(citation).split(), str(sandhi).split()
    if cat == "tensification":
        return _count(s, TENSE) > _count(c, TENSE)
    if cat == "aspiration":
        return _count(s, ASP) > _count(c, ASP)
    if cat.startswith("nasalization"):
        return _count(s, NAS) > _count(c, NAS)
    if cat == "liquidization":
        return (_has_gem(s) or _count(s, LIQ) > _count(c, LIQ))
    if cat == "palatalization":
        return _count(s, PAL) > _count(c, PAL)
    if cat.startswith("liaison"):
        return liaison_is_valid(citation, sandhi)
    return True


def _has_gem(seq):
    return any(p in {"ʎː", "ɭː", "lː"} for p in seq)


def load_variants() -> pd.DataFrame:
    """変異形の正典。**修復後のランを優先**する（2026-08-31）。

    `.mfa_*_v2` と `.mfa_crosseojeol_v3` は、規則ベースのフォールバックを
    正しく発動させて作り直した辞書でMFAを再実行した結果である。旧ランは
    宣言したカテゴリの対立を作れていない項目を多く含むため、後方互換のために
    残すが、優先順位は最も低い。
    """
    order = [".mfa_newcats_v2", ".mfa_balanced2_v2", ".mfa_crosseojeol_v3",
             ".mfa_scale1", ".mfa_balanced", ".mfa_balanced2", ".mfa_newcats",
             ".mfa_crosseojeol", ".mfa_pilot"]
    frames = []
    for run in order:
        for f in sorted(glob.glob(str(ROOT / run / "pilot_results*.csv"))):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if {"token", "utterance_id", "citation", "sandhi"} <= set(d.columns):
                d = d[["token", "utterance_id", "citation", "sandhi"]].copy()
                d["_src"] = run
                frames.append(d)
    return pd.concat(frames).drop_duplicates(["token", "utterance_id"])


def audit(check_dir: Path, allv: pd.DataFrame) -> pd.DataFrame:
    k = pd.read_csv(check_dir / "answer_key_PRIVATE.csv")
    m = k.merge(allv, on=["token", "utterance_id"], how="left")
    m["valid"] = [
        variant_is_valid(r["category"], r["citation"], r["sandhi"])
        if pd.notna(r["citation"]) else False
        for _, r in m.iterrows()
    ]
    return m


def summarize(name: str, m: pd.DataFrame) -> None:
    g = m.groupby("category")["valid"].agg(n="size", 妥当="sum")
    g["妥当率"] = (g["妥当"] / g["n"]).map(lambda x: f"{x:.0%}")
    print(f"\n########## {name}")
    print("  適用形がカテゴリ名どおりの変化を含むか:")
    print("   " + g.to_string().replace("\n", "\n    "))


def stats(label: str, d: pd.DataFrame) -> None:
    ct = (pd.crosstab(d["mfa"], d["human"])
            .reindex(index=["applied", "notapplied"], columns=["applied", "notapplied"])
            .fillna(0).astype(int))
    ag = (d["mfa"] == d["human"]).mean()
    kp = cohen_kappa_score(d["mfa"], d["human"], labels=["applied", "notapplied"])
    b, c = ct.loc["applied", "notapplied"], ct.loc["notapplied", "applied"]
    p = mcnemar([[ct.loc["applied", "applied"], b],
                 [c, ct.loc["notapplied", "notapplied"]]], exact=True).pvalue
    base = d["human"].value_counts(normalize=True).max()
    print(f"\n=== {label}  n={len(d)} ===")
    print(f"  一致率 {ag:.1%} / Cohen's κ = {kp:.3f}")
    print(f"  多数派ベースライン {base:.1%}（差 {ag - base:+.1%}）")
    print(f"  不一致: MFA非適用-聴取適用 {c}件 / 逆 {b}件, McNemar 正確検定 p={p:.3g}")


def recompute(check_dir: Path, allv: pd.DataFrame) -> None:
    m = audit(check_dir, allv)
    r = pd.read_csv(check_dir / "response_sheet_filled.csv", encoding="utf-8-sig")
    r.columns = [c.strip() for c in r.columns]
    jc = [c for c in r.columns if "判定" in c][0]
    d = m.merge(r[["item_id", jc]], on="item_id")
    d["human"] = d[jc].map({"적용됨": "applied", "적용안됨": "notapplied"})
    d["mfa"] = d["mfa_result"].map({"SANDHI (rule applied)": "applied",
                                     "CITATION (rule not applied)": "notapplied"})
    d = d.dropna(subset=["human", "mfa"])
    stats("現行の公表値（全件）", d)
    stats("変異形が妥当な項目のみ", d[d["valid"]])
    print("\n  除外された項目のカテゴリ別内訳:")
    print("   " + d[~d["valid"]]["category"].value_counts().to_string().replace("\n", "\n    "))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    allv = load_variants()
    for d in ["manual_check_2026-08-05", "manual_check_2026-08-10_crosseojeol"]:
        summarize(d, audit(ROOT / d, allv))
    if a.recompute:
        recompute(ROOT / "manual_check_2026-08-05", allv)


if __name__ == "__main__":
    main()
