"""
変異形競合方式: パイロット候補語について「引用形(citation, 規則未適用)」と
「連続音声形(sandhi, 規則適用済み)」の2つの発音変異形を持つMFA辞書を構築する。

方法(korean_mfaのG2Pモデル・辞書の挙動を実測して採用):
  - sandhi変異形: 基本辞書(korean_mfa.dict)に語全体のエントリがあればそれを採用
    (基本辞書はG2Pモデルの訓練元でもあり、実測で規則適用済みの発音を収録している
    ことが多い; 例 그렇게→kɨɾʌkʰe(激音化済), 엘에이→eɾei(連音化済))。
    なければ mfa g2p で語全体を推定。
  - citation変異形: 音節境界(syllable_boundary)で語を左右に分割し、それぞれを
    個別に mfa g2p へ渡して得た発音を連結する。孤立音節としてG2Pすると規則が
    適用されにくいため、規則「未適用」形の近似として使う。
  - 両変異形が同一になった場合(G2Pが訓練データの偏りで語全体でも規則を適用しない
    ケース、例: 토픽만の鼻音化脱落)は、citation形にNASAL_MAP等を直接適用して
    sandhi形を明示的に合成するフォールバックを行う。

出力: .mfa_pilot/variant_dict.txt (基本辞書のコピーから候補語の既存エントリを
除去し、2変異形×確率0.5を追加した完全なMFA辞書)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from hangul_jamo import NASAL_MAP, decompose  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_variant_validity import variant_is_valid  # noqa: E402

MFA_ENV = "mfa"
BASE_DICT = Path(os.environ.get("KOREAN_MFA_DICTIONARY", "korean_mfa.dict")).expanduser()


def run_g2p(words, workdir: Path):
    wordlist_path = workdir / "g2p_wordlist.txt"
    wordlist_path.write_text("\n".join(words), encoding="utf-8")
    out_path = workdir / "g2p_out.dict"
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


def load_base_dict_best(words, base_dict_path: Path):
    """基本辞書から各語の最も確率の高いエントリの発音を取得する。"""
    best = {}
    best_prob = {}
    wanted = set(words)
    with open(base_dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            word = parts[0]
            if word not in wanted:
                continue
            try:
                prob = float(parts[1])
            except ValueError:
                prob = 0.0
            phones = parts[-1]
            if word not in best or prob > best_prob[word]:
                best[word] = phones
                best_prob[word] = prob
    return best


def apply_nasalization_fallback(citation_phones: str, token: str, boundary: int):
    """citation形にnasalization未適用が疑われる場合、終声音素をNASAL_MAPで
    明示的に置換してsandhi形を合成するフォールバック。coda音素は終声位置では
    通常「不放音(k̚/t̚/p̚)」で表記されるため、対応する鼻音素(ŋ/n/m)に置換する。
    """
    coda_target = {"k̚": "ŋ", "t̚": "n", "p̚": "m", "k": "ŋ", "t": "n", "p": "m"}
    phones = citation_phones.split(" ")
    # citation形の音素列のうち、境界に最も近いと推定される破裂音を後ろから探索する
    for i in range(len(phones) - 1, -1, -1):
        if phones[i] in coda_target:
            phones[i] = coda_target[phones[i]]
            return " ".join(phones)
    return citation_phones


# G2P は母音間・鼻音後で有声異音（d ɡ b dʑ ɣ β ɟ）を出す。引用形の初声が
# 有声で書かれていても濃音化の対象なので、写像に含める（2026-08-31 追加）。
ONSET_TO_TENSE = {
    "k": "k͈", "ɡ": "k͈", "ɣ": "k͈", "c": "k͈", "ɟ": "k͈",
    "t": "t͈", "d": "t͈",
    "p": "p͈", "b": "p͈", "β": "p͈",
    "s": "s͈", "sʰ": "s͈", "ɕʰ": "s͈",
    "tɕ": "tɕ͈", "dʑ": "tɕ͈",
}


def apply_onset_fallback(citation_phones: str, left_len: int, target_map=None, target_value=None):
    """citation形の右側(次音節側)先頭音素、すなわちboundary直後の初声を
    置換してsandhi形を合成する汎用フォールバック(tensification/liquidization/
    nasalization_liquid向け)。target_mapが与えられればそれで置換語を決定し、
    なければtarget_valueで一律置換する。
    """
    phones = citation_phones.split(" ")
    if left_len >= len(phones):
        return citation_phones
    onset = phones[left_len]
    if target_map is not None:
        if onset not in target_map:
            return citation_phones
        phones[left_len] = target_map[onset]
    else:
        phones[left_len] = target_value
    return " ".join(phones)


# 音響モデル（korean_mfa 辞書）に実在する二次調音つき音素。
# 存在しない組み合わせ（ɡʲ など）を作ると MFA がその発音を丸ごと無視する。
MODEL_PHONES = {"bʲ", "bʷ", "dʲ", "dʷ", "kʷ", "ɡʷ", "mʲ", "pʲ", "pʷ",
                "sʷ", "tʲ", "tʷ", "ɾʲ", "ɾʷ", "ɟ", "ɲ", "ʎ"}
# 口蓋化した軟口蓋閉鎖音は ɡʲ ではなく ɟ と書かれる。
PALATAL_ONSET = {"ɡ": "ɟ", "k": "ɟ"}

ASPIRATE = {"k̚": "kʰ", "k": "kʰ", "ɡ": "kʰ", "t̚": "tʰ", "t": "tʰ", "d": "tʰ",
            "p̚": "pʰ", "p": "pʰ", "b": "pʰ", "tɕ": "tɕʰ", "dʑ": "tɕʰ"}
H_SET = {"h", "ɦ", "ç", "x", "ɸ", "β", "ʝ"}


def apply_aspiration(citation_phones: str, left_len: int):
    """격음화: 阻害音と /h/ が隣接して激音1つに融合する。

    파악할: … k̚ h ɐ … → … kʰ ɐ …（終声＋ㅎ、2音素が1音素になる）
    逆順（ㅎ終声＋平音初声）も同じ位置で起こる。
    """
    ph = citation_phones.split(" ")
    if left_len >= len(ph) or left_len == 0:
        return citation_phones
    coda, onset = ph[left_len - 1], ph[left_len]
    if coda in ASPIRATE and onset in H_SET:
        merged = ASPIRATE[coda]
    elif coda in H_SET and onset in ASPIRATE:
        merged = ASPIRATE[onset]
    else:
        return citation_phones
    return " ".join(ph[:left_len - 1] + [merged] + ph[left_len + 1:])


CODA_TO_ONSET = {"k̚": "ɡ", "k": "ɡ", "t̚": "d", "t": "d", "p̚": "b", "p": "b",
                 "ɭ": "ɾ", "s": "sʰ", "sʰ": "sʰ", "tɕ": "dʑ", "tɕʰ": "tɕʰ",
                 "n": "n", "m": "m", "ŋ": "ŋ"}
GLIDE_SET = {"j", "w", "ɥ", "ɰ"}
PAL_TARGET = {"t̚": "dʑ", "t": "dʑ", "d": "dʑ", "tʰ": "tɕʰ"}


def apply_liaison(citation_phones: str, left_len: int):
    """연음화: 終声を次音節の初声に移す（再音節化）。

    右側が滑音で始まる場合、終声と滑音は1音素に融合する
    （ɭ + ɥ → ɾʷ、p̚ + j → pʲ）。korean_mfa の表記に合わせる。
    """
    ph = citation_phones.split(" ")
    if left_len >= len(ph) or left_len == 0:
        return citation_phones
    coda = ph[left_len - 1]
    if coda not in CODA_TO_ONSET:
        return citation_phones
    onset = CODA_TO_ONSET[coda]
    nxt = ph[left_len]
    if nxt in GLIDE_SET:                       # 終声＋滑音 → 二次調音つき1音素
        sec = "ʲ" if nxt == "j" else "ʷ"
        merged = PALATAL_ONSET.get(onset, onset + sec) if sec == "ʲ" else onset + sec
        if merged not in MODEL_PHONES:         # モデルに無い組み合わせは融合しない
            return " ".join(ph[:left_len - 1] + [onset] + ph[left_len:])
        return " ".join(ph[:left_len - 1] + [merged] + ph[left_len + 1:])
    return " ".join(ph[:left_len - 1] + [onset] + ph[left_len:])


def apply_palatalization(citation_phones: str, left_len: int):
    """구개음화: 終声 ㄷ/ㅌ が ㅣ の前で ㅈ/ㅊ になり、次音節の初声に移る。"""
    ph = citation_phones.split(" ")
    if left_len >= len(ph) or left_len == 0:
        return citation_phones
    coda = ph[left_len - 1]
    if coda not in PAL_TARGET:
        return citation_phones
    return " ".join(ph[:left_len - 1] + [PAL_TARGET[coda]] + ph[left_len:])


PALATALIZED = {"ɾʲ", "ʎ", "ʎː", "ɲ", "ʝ"}
CODA_TO_NASAL = {"k̚": "ŋ", "k": "ŋ", "ɡ": "ŋ", "t̚": "n", "t": "n", "d": "n",
                 "p̚": "m", "p": "m", "b": "m"}


def apply_nasalization_at(citation_phones: str, left_len: int):
    """비음화(阻害音由来): 境界直前の終声阻害音を、対応する鼻音に替える。

    以前の実装は音素列を後ろから探索して最初に見つかった破裂音を置換しており、
    対象の境界とは無関係な位置を変えることがあった。位置で指定する。
    """
    ph = citation_phones.split(" ")
    if left_len >= len(ph) or left_len == 0:
        return citation_phones
    coda = ph[left_len - 1]
    if coda not in CODA_TO_NASAL:
        return citation_phones
    ph[left_len - 1] = CODA_TO_NASAL[coda]
    return " ".join(ph)


def apply_liquidization(citation_phones: str, left_len: int):
    """유음화: 終声と次の初声を、単一の重子音（ɭː / ʎː）に置き換える。

    korean_mfa は ㄹㄹ を1つの長子音として書く（분류 → p u ʎː u）。
    以前の実装は初声を `l` に替えるだけだったが、`l` は音響モデルに無い音素で、
    MFA がその発音を丸ごと無視していた（2026-08-31 発見、482件）。
    """
    ph = citation_phones.split(" ")
    if left_len >= len(ph) or left_len == 0:
        return citation_phones
    onset = ph[left_len]
    gem = "ʎː" if (onset in PALATALIZED or onset.endswith("ʲ")) else "ɭː"
    return " ".join(ph[:left_len - 1] + [gem] + ph[left_len + 1:])


def apply_nasalization_liquid(citation_phones: str, left_len: int):
    """비음화(유음 유래): 初声 ㄹ を鼻音にし、直前の阻害音終声も鼻音化する。

    입력부터: i p̚ ɾʲ ʌ … → i m ɲ ʌ …（p̚→m と ɾʲ→ɲ の両方が起きる）
    以前は初声だけを n に替えており、終声が阻害音のまま残っていた。
    """
    ph = citation_phones.split(" ")
    if left_len >= len(ph) or left_len == 0:
        return citation_phones
    onset = ph[left_len]
    ph[left_len] = "ɲ" if (onset in PALATALIZED or onset.endswith("ʲ")) else "n"
    coda = ph[left_len - 1]
    if coda in CODA_TO_NASAL:
        ph[left_len - 1] = CODA_TO_NASAL[coda]
    return " ".join(ph)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", default=".mfa_pilot")
    args = parser.parse_args()

    pilot_dir = Path(args.pilot_dir)
    df = pd.read_csv(pilot_dir / "pilot_candidates.csv")
    df = df.drop_duplicates(subset=["token", "syllable_boundary"])

    lefts, rights, wholes = set(), set(), set()
    splits = {}
    for _, row in df.iterrows():
        token, boundary = row["token"], int(row["syllable_boundary"])
        left = token[: boundary + 1]
        right = token[boundary + 1 :]
        splits[(token, boundary)] = (left, right)
        lefts.add(left)
        rights.add(right)
        wholes.add(token)

    all_words = lefts | rights | wholes
    print(f"g2p query words: {len(all_words)}")
    g2p_pron = run_g2p(sorted(all_words), pilot_dir)
    dict_pron = load_base_dict_best(wholes, BASE_DICT)

    entries = []
    for _, row in df.iterrows():
        token, boundary, change_type = row["token"], int(row["syllable_boundary"]), row["change_type"]
        left, right = splits[(token, boundary)]

        left_ph = g2p_pron.get(left)
        right_ph = g2p_pron.get(right)
        if left_ph is None or right_ph is None:
            print(f"  WARN: g2p missing for split of '{token}' ({left}/{right}); skipping")
            continue
        citation = f"{left_ph} {right_ph}"

        sandhi = dict_pron.get(token) or g2p_pron.get(token)
        if sandhi is None:
            print(f"  WARN: no sandhi pronunciation for '{token}'; skipping")
            continue

        note = ""
        # 2026-08-31 修正（重大）:
        # 以前は `sandhi == citation` のときだけ規則ベースのフォールバックを使っていた。
        # しかしG2Pが「引用形とは違うが、その規則とも無関係な形」を返すことが多く、
        # その場合フォールバックが発動せず、**宣言した音韻過程の対立になっていない
        # 2形**を辞書に入れていた。実測での妥当率は 경음화 7% / 격음화 60% /
        # 비음화 63% / 유음화 70%（母集団、`audit_variant_validity.py`）。
        # したがって判定条件を「同一か」から「**その規則の変化を実際に含むか**」に
        # 変更する。含まなければ規則ベースで合成する。
        if not variant_is_valid(change_type, citation, sandhi):
            why = ("citation と同一" if sandhi.replace(" ", "") == citation.replace(" ", "")
                   else "g2p 出力が当該規則の変化を含まない")
            left_len = len(left_ph.split(" "))
            if change_type in ("nasalization_obstruent", "nasalization"):
                sandhi = apply_nasalization_at(citation, left_len)
                note = f"nasalization fallback applied ({why})"
            elif change_type == "nasalization_liquid":
                sandhi = apply_nasalization_liquid(citation, left_len)
                note = f"nasalization_liquid fallback applied (coda->nasal, onset->n/ɲ, {why})"
            elif change_type == "tensification":
                sandhi = apply_onset_fallback(citation, left_len, target_map=ONSET_TO_TENSE)
                note = f"tensification fallback applied (onset->tense, {why})"
            elif change_type == "liquidization":
                sandhi = apply_liquidization(citation, left_len)
                note = f"liquidization fallback applied (coda+onset->ɭː/ʎː, {why})"
            elif change_type == "aspiration":
                sandhi = apply_aspiration(citation, left_len)
                note = f"aspiration fallback applied (阻害音+h -> 激音, {why})"
            elif change_type.startswith("liaison"):
                sandhi = apply_liaison(citation, left_len)
                note = f"liaison fallback applied (coda->onset, {why})"
            elif change_type == "palatalization":
                sandhi = apply_palatalization(citation, left_len)
                note = f"palatalization fallback applied (coda->tɕ/tɕʰ, {why})"
            else:
                note = f"WARNING: no rule-based fallback for {change_type} ({why})"
            # 合成しても規則の変化を作れなければ、対立が無いので後段で除外できるよう印を残す
            if not variant_is_valid(change_type, citation, sandhi):
                note += " | STILL_INVALID"

        entries.append({
            "token": token, "boundary": boundary, "change_type": change_type,
            "citation": citation, "sandhi": sandhi, "note": note,
        })

    entries_df = pd.DataFrame(entries).drop_duplicates(subset=["token", "boundary"])
    entries_df.to_csv(pilot_dir / "variant_pronunciations.csv", index=False)
    print(entries_df[["token", "change_type", "citation", "sandhi", "note"]].to_string(index=False))

    # --- 辞書合成: 基本辞書から候補語の既存行を除去し、2変異形(確率0.5/0.5)を追加 ---
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

    out_dict_path = pilot_dir / "variant_dict.txt"
    out_dict_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nmerged dictionary -> {out_dict_path} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
