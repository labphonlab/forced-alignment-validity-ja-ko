"""liaison_crosseojeolのvariant辞書構築、修正版(2026-08-10)。

当初版(select_and_prepare_crosseojeol.py → build_variant_dict.py流用)は、
2つのeojeolを結合した人工的な偽単語をそのままG2Pに渡してsandhi変異形を
得ていたが、G2Pが境界の子音を正しく再音節化(coda→onset移動)せず単純に
脱落させる誤った予測をすることが人手検証との突合せで判明した
(README §12「人手検証完了」参照)。

修正: hangul_jamo.resyllabify_at_boundary()で境界の綴りを先に正しく
連音化した形に変換してからG2Pに渡す。これによりG2Pは通常の(見た目上)
単語に近い綴りを処理するだけでよくなり、境界の子音を正しく保持した
sandhi変異形が得られる。

citation変異形は変更なし(left_token・right_tokenをそれぞれ独立にG2P)。
複合終声(ㄺ等)でresyllabify_at_boundary()がNoneを返す場合は、当初版と
同じ(旧・欠陥あり)方式にフォールボックスし、note列に明記する。

入力: .mfa_crosseojeol/pilot_candidates.csv (token=結合形, syllable_boundary)
出力: .mfa_crosseojeol/variant_dict_v2.txt, variant_pronunciations_v2.csv
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from hangul_jamo import resyllabify_at_boundary  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_variant_validity import liaison_is_valid  # noqa: E402
from build_variant_dict import apply_liaison  # noqa: E402

MFA_ENV = "mfa"
BASE_DICT = Path(os.environ.get("KOREAN_MFA_DICTIONARY", "korean_mfa.dict")).expanduser()
PILOT_DIR = Path(os.environ.get("KOREAN_LIAISON_WORK_DIR", ".mfa_crosseojeol")).expanduser()


def run_g2p(words, workdir: Path):
    wordlist_path = workdir / "g2p_wordlist_v2.txt"
    wordlist_path.write_text("\n".join(sorted(set(words))), encoding="utf-8")
    out_path = workdir / "g2p_out_v2.dict"
    subprocess.run(
        ["mamba", "run", "-n", MFA_ENV, "mfa", "g2p", str(wordlist_path), "korean_mfa", str(out_path), "--num_pronunciations", "1"],
        check=True, capture_output=True, text=True,
    )
    pron = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        word, phones = parts[0], parts[-1]
        pron.setdefault(word, phones)
    return pron


def main():
    df = pd.read_csv(PILOT_DIR / "pilot_candidates.csv")
    df = df.drop_duplicates(subset=["token", "syllable_boundary"])

    lefts, rights, resyl_lefts, resyl_rights = set(), set(), set(), set()
    splits = {}
    for _, row in df.iterrows():
        token, boundary = row["token"], int(row["syllable_boundary"])
        left = token[: boundary + 1]
        right = token[boundary + 1:]
        resyl = resyllabify_at_boundary(left, right)
        splits[(token, boundary)] = (left, right, resyl)
        lefts.add(left)
        rights.add(right)
        if resyl:
            resyl_lefts.add(resyl[0])
            resyl_rights.add(resyl[1])

    all_words = lefts | rights | resyl_lefts | resyl_rights
    print(f"g2p query words: {len(all_words)} (incl. {len(resyl_lefts)+len(resyl_rights)} resyllabified)")
    g2p_pron = run_g2p(sorted(all_words), PILOT_DIR)

    entries = []
    n_resyl_ok, n_fallback, n_ipa_fix = 0, 0, 0
    for _, row in df.iterrows():
        token, boundary, change_type = row["token"], int(row["syllable_boundary"]), row["change_type"]
        left, right, resyl = splits[(token, boundary)]

        left_ph = g2p_pron.get(left)
        right_ph = g2p_pron.get(right)
        if left_ph is None or right_ph is None:
            print(f"  WARN: g2p missing for split of '{token}' ({left}/{right}); skipping")
            continue
        citation = f"{left_ph} {right_ph}"

        note = ""
        if resyl:
            rl_ph = g2p_pron.get(resyl[0])
            rr_ph = g2p_pron.get(resyl[1])
            if rl_ph is not None and rr_ph is not None:
                sandhi = f"{rl_ph} {rr_ph}"
                n_resyl_ok += 1
            else:
                sandhi = None
        else:
            sandhi = None

        if sandhi is None:
            # 複合終声等でresyllabify不可、またはG2Pが欠損 -> 旧方式(結合偽単語への
            # 直接G2P)にフォールバック。既にpilot_results.csvにある値を再利用。
            n_fallback += 1
            note = "resyllabify unavailable, fell back to old (merged-word G2P) method"
            old_results = pd.read_csv(PILOT_DIR / "pilot_results.csv")
            old_row = old_results[old_results["token"] == token]
            sandhi = old_row.iloc[0]["sandhi"] if len(old_row) else citation

        # 2026-08-31 追加:
        # 綴りを正しく再音節化しても、G2P が終声の子音を落とした発音を返すことがある
        # （신호를읽어: … ɾ ɨ ɭ i … → … ɾ ɨ i …）。連音化は再音節化なので子音は
        # 保存されるはずであり、保存されていなければ IPA 水準で終声を初声へ移す。
        if not liaison_is_valid(citation, sandhi):
            fixed = apply_liaison(citation, len(left_ph.split(" ")))
            if liaison_is_valid(citation, fixed):
                sandhi = fixed
                n_ipa_fix += 1
                note = (note + "; " if note else "") + "IPA-level liaison fallback applied"
            else:
                note = (note + "; " if note else "") + "STILL_INVALID: no liaison contrast"

        if sandhi.replace(" ", "") == citation.replace(" ", ""):
            note = (note + "; " if note else "") + "WARNING: citation == sandhi, no acoustic contrast to test"

        entries.append({
            "token": token, "boundary": boundary, "change_type": change_type,
            "citation": citation, "sandhi": sandhi, "note": note,
        })

    print(f"\nresyllabified correctly: {n_resyl_ok}, fell back to old method: {n_fallback}, IPA-level liaison fix: {n_ipa_fix}")

    entries_df = pd.DataFrame(entries).drop_duplicates(subset=["token", "boundary"])
    entries_df.to_csv(PILOT_DIR / "variant_pronunciations_v2.csv", index=False)

    candidate_words = set(entries_df["token"])
    out_lines = []
    with open(BASE_DICT, "r", encoding="utf-8") as f:
        for line in f:
            word = line.split("\t", 1)[0]
            if word in candidate_words:
                continue
            out_lines.append(line.rstrip("\n"))

    for _, row in entries_df.iterrows():
        for phones in {row["sandhi"], row["citation"]}:
            out_lines.append(f"{row['token']}\t0.5\t0.5\t1.0\t1.0\t{phones}")

    out_dict_path = PILOT_DIR / "variant_dict_v2.txt"
    out_dict_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nmerged dictionary -> {out_dict_path} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
