"""form / original_form の差分を抽出し、音韻過程に分類する。

NIKL Dialogue 2025 は正書法正規化した `form` と発音準拠の `original_form` を
両方持つ。差分は転写者によるトークン単位の発音変異注記であり（語型ごとの慣習では
ないことは検証済み: 同一語5回以上の話者の72-89%で変異形が混在）、
人手の gold standard として使える。

ここでは語数が一致する発話に限って対を取り、字母分解で過程を判定する。
"""
import collections
import csv
import glob
import json
import re

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
LENIS = {0: 1, 3: 4, 7: 8, 9: 10, 12: 13}          # 평음 → 경음
ASPIR = {0: 15, 3: 16, 7: 17, 12: 14}              # 평음 → 격음
RAISE = {("ㅗ", "ㅜ"), ("ㅓ", "ㅡ"), ("ㅐ", "ㅔ"), ("ㅏ", "ㅐ"), ("ㅕ", "ㅔ"),
         ("ㅑ", "ㅐ"), ("ㅗ", "ㅚ"), ("ㅜ", "ㅡ")}
MARKUP = re.compile(r"\{[^}]*\}|\([^)]*\)|&[^&]*&|[.,!?~\-'/]")


def clean(tok):
    t = MARKUP.sub("", tok).strip()
    return t if t and all("가" <= c <= "힣" for c in t) else None


def jamo(syl):
    c = ord(syl) - 0xAC00
    return c // 588, (c % 588) // 28, c % 28


def classify(a, b):
    """form a → original_form b をラベル付けする。"""
    if len(a) != len(b):
        return "syllable_deletion" if len(b) < len(a) else "syllable_addition"
    labels = []
    for x, y in zip(a, b):
        cx, vx, jx = jamo(x)
        cy, vy, jy = jamo(y)
        if cx != cy:
            if LENIS.get(cx) == cy:
                labels.append("tensification")
            elif ASPIR.get(cx) == cy:
                labels.append("aspiration")
            elif cx == 11 or cy == 11:
                labels.append("onset_liaison")     # ㅇ の出入り＝再音節化
            else:
                labels.append("onset_other")
        if vx != vy:
            pair = (JUNG[vx], JUNG[vy])
            labels.append("vowel_raising" if pair in RAISE else "vowel_other")
        if jx != jy:
            labels.append("coda_deletion" if jy == 0 else
                          ("coda_addition" if jx == 0 else "coda_change"))
    if not labels:
        return "identical"
    return labels[0] if len(set(labels)) == 1 else "multiple:" + "+".join(sorted(set(labels)))


def main():
    inv = collections.Counter()
    spk = collections.defaultdict(set)
    rows = []
    for path in sorted(glob.glob("data/raw_nikl_json/*.json")):
        data = json.load(open(path, encoding="utf-8"))
        for doc in data.get("document", []):
            for utt in doc.get("utterance", []):
                fa = (utt.get("form") or "").split()
                fb = (utt.get("original_form") or "").split()
                if len(fa) != len(fb):
                    continue
                sid = utt.get("speaker_id", "")
                for i, (ta, tb) in enumerate(zip(fa, fb)):
                    ca, cb = clean(ta), clean(tb)
                    if not ca or not cb or ca == cb:
                        continue
                    cat = classify(ca, cb)
                    inv[(ca, cb, cat)] += 1
                    spk[(ca, cb)].add(sid)
                    rows.append({"utterance_id": utt.get("id", ""), "speaker_id": sid,
                                 "word_index": i, "form": ca, "orig": cb, "category": cat})
    with open("data/pair_inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["form", "orig", "category", "n_tokens", "n_speakers"])
        for (a, b, c), n in inv.most_common():
            w.writerow([a, b, c, n, len(spk[(a, b)])])
    with open("data/discrepancy_tokens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utterance_id", "speaker_id", "word_index",
                                          "form", "orig", "category"])
        w.writeheader()
        w.writerows(rows)
    cat = collections.Counter()
    for (a, b, c), n in inv.items():
        cat[c] += n
    print(f"差分トークン {sum(inv.values()):,} / 異なり対 {len(spk):,}\n")
    for c, n in cat.most_common(20):
        print(f"  {c:45s} {n:8,}  ({100*n/sum(cat.values()):5.1f}%)")


if __name__ == "__main__":
    main()
