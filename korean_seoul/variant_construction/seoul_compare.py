#!/usr/bin/env python3
"""Seoul Corpus 上で MFA の変異形判定と人手転記を突き合わせる（2026-09-09 新設）。

Seoul Corpus を使う理由は3つ。
  1. 人手の音素ラベル。9名の訓練された転記者が15か月、音素同定の一致は κ=0.980
     （Yun et al. 2015）。評定者パネルを募る必要がない。
  2. **境界精度が既報**。McAuliffe et al.(2026) が同コーパスで korean_mfa の平均境界誤差
     14.78 ms を報告している。**時間軸と範疇軸を同一データ上で対比できる。**
  3. 規模。候補は27,132件（NIKL の聴取検証は106件）。

標本は**自然な出現比**で取る。陰性を人為的に増やすと多数派ベースラインが働かず、
偏りの有無を NIKL と比較できなくなる（2026-09-09 の試走で確認済み）。

前段: seoul_corpus/prepare_mfa.py → build_variant_dict.py → mfa align
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pilot_variant_competition"))
from audit_variant_validity import variant_is_valid, liaison_is_valid  # noqa: E402

# 過程を追加したランは別ディレクトリに分けてある（.mfa_seoul は検証済み数値の再現元）。
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
RUN = ROOT / "seoul_corpus" / (_args[0] if _args else ".mfa_seoul")


def phones_of(tg: Path, token: str):
    """TextGrid から対象語に含まれる音素列を取り出す。"""
    t = tg.read_text(encoding="utf-8")
    tiers = {}
    for b in re.split(r"item\s*\[\d+\]:", t)[1:]:
        m = re.search(r'name = "([^"]*)"', b)
        if m:
            tiers[m.group(1)] = re.findall(
                r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"', b)
    for s, e, w in tiers.get("words", []):
        if w == token:
            return " ".join(p for ps, pe, p in tiers.get("phones", [])
                            if float(ps) >= float(s) - 1e-6
                            and float(pe) <= float(e) + 1e-6 and p.strip())
    return None


def load():
    vp = {r["token"]: r for r in
          csv.DictReader((RUN / "variant_pronunciations.csv").open(encoding="utf-8"))}
    rows = []
    for c in csv.DictReader((RUN / "pilot_candidates.csv").open(encoding="utf-8")):
        tg = RUN / "aligned" / c["speaker_id"] / f"{c['utterance_id']}.TextGrid"
        v = vp.get(c["token"])
        if not tg.exists() or not v:
            continue
        seq = phones_of(tg, c["token"])
        if not seq:
            continue
        cat, cit, san = c["change_type"], v["citation"], v["sandhi"]
        ok = (liaison_is_valid(cit, san) if cat.startswith("liaison")
              else variant_is_valid(cat, cit, san))
        if not ok:
            continue                       # 2仮説が対立を作っていない
        mfa = "applied" if seq == san else ("notapplied" if seq == cit else None)
        if mfa is None:
            continue                       # どちらの変異形でもない（稀）
        rows.append(dict(cat=cat, human=int(c["human_applied"]), mfa=mfa,
                         speaker=c["speaker_id"], token=c["token"],
                         utt=c["utterance_id"]))   # 境界測定との結合キー
    return rows


def block(g, label):
    tp = sum(1 for r in g if r["human"] == 1 and r["mfa"] == "applied")
    fn = sum(1 for r in g if r["human"] == 1 and r["mfa"] == "notapplied")
    fp = sum(1 for r in g if r["human"] == 0 and r["mfa"] == "applied")
    tn = sum(1 for r in g if r["human"] == 0 and r["mfa"] == "notapplied")
    ag = (tp + tn) / len(g)
    hp = (tp + fn) / len(g)
    base = max(hp, 1 - hp)
    p = mcnemar([[tp, fp], [fn, tn]], exact=False).pvalue
    print(f"  {label:<26}n={len(g):>5,}  一致 {ag:5.1%}  人手適用率 {hp:5.1%}  "
          f"多数派 {base:5.1%} (差 {100*(ag-base):+5.1f}pt)")
    print(f"  {'':<26}見落とし {fn:>4} 対 逆 {fp:<4}  McNemar p={p:.3g}")


def main():
    rows = load()
    print(f"照合できた候補 {len(rows):,} 件 / 話者 {len({r['speaker'] for r in rows})} 名\n")
    print("【Seoul Corpus・同一データ上での範疇軸】")
    for cat in sorted({r["cat"] for r in rows}):
        block([r for r in rows if r["cat"] == cat], cat)
    print()
    block(rows, "全体")
    print("\n※ 同コーパスの境界精度は McAuliffe et al.(2026) が 14.78 ms と報告。"
          "\n  人手転記者間の境界偏差は 9.04 ms、音素同定の一致は κ=0.980（Yun et al. 2015）。")


if __name__ == "__main__":
    main()
