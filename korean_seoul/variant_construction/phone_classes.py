#!/usr/bin/env python3
"""人手ラベル（Seoul ローマ字）と MFA 出力（IPA）を共通の粗い音素類に写す。

境界を突き合わせるには、まずどの分節がどの分節に対応するかを決める必要がある。
2つの表記体系は直接比較できないので、**対応付けに足るだけの粗さ**の類へ写す。
喉頭素性（平音/激音/濃音）は意図的に潰す——そこは範疇軸の争点であり、
時間軸の対応付けの手がかりにしてはならない。
"""

# --- 人手（Yun et al. 2015 の表記） ---
HAND = {}
for k in ("p0", "ph", "pp"): HAND[k] = "P"
for k in ("t0", "th", "tt"): HAND[k] = "T"
for k in ("k0", "kh", "kk"): HAND[k] = "K"
for k in ("s0", "ss"): HAND[k] = "S"
for k in ("c0", "cc", "ch"): HAND[k] = "C"
HAND["hh"] = "H"
HAND["mm"], HAND["nn"], HAND["ng"] = "M", "N", "NG"
HAND["ll"] = "L"
# 母音。滑音つきの記号は核へ寄せる（MFA 側では滑音が独立音素になるため、
# その差は対応付けの挿入として扱われる）
for k, v in (("ii", "I"), ("ee", "E"), ("EE", "E"), ("aa", "A"), ("oo", "O"),
             ("uu", "U"), ("vv", "V^"), ("xx", "U^"), ("xi", "U^"),
             ("ya", "A"), ("ye", "E"), ("yo", "O"), ("yu", "U"), ("yv", "V^"),
             ("wa", "A"), ("we", "E"), ("wi", "I"), ("wv", "V^"),
             ("wE", "E"), ("WE", "E"), ("YE", "E")):
    HAND[k] = v

# --- MFA（korean_mfa v3.0 の IPA） ---
IPA = {}
for k in ("p", "pʰ", "p͈", "b", "p̚", "pʲ", "pʷ", "ɸʷ", "βʷ"): IPA[k] = "P"
for k in ("t", "tʰ", "t͈", "d", "t̚", "tʷ"): IPA[k] = "T"
# c/cʰ/c͈/ɟ は ㄱ の口蓋化異音であって ㅈ ではない
for k in ("k", "kʰ", "k͈", "ɡ", "k̚", "kʷ", "kʷː", "ɡʷ", "c", "cʰ", "cʰː", "c͈", "ɟ"):
    IPA[k] = "K"
for k in ("s", "sʰ", "s͈", "sː", "ɕʰ", "ɕ͈"): IPA[k] = "S"
for k in ("tɕ", "tɕʰ", "tɕ͈", "tɕ͈ː", "dʑ", "tɕʷ"): IPA[k] = "C"
for k in ("h", "ç", "ɦ", "ʝ"): IPA[k] = "H"
for k in ("m", "mʲ"): IPA[k] = "M"
for k in ("n", "nː", "ɲ"): IPA[k] = "N"
IPA["ŋ"] = "NG"
for k in ("l", "ɭ", "ɭː", "ɾ", "ɾʲ", "ʎ", "ʎː"): IPA[k] = "L"
for k in ("j", "w", "ɰ"): IPA[k] = "G"
for k, v in (("i", "I"), ("iː", "I"), ("e", "E"), ("eː", "E"), ("ɛː", "E"),
             ("ɐ", "A"), ("o", "O"), ("oː", "O"), ("u", "U"), ("uː", "U"),
             ("ʌ", "V^"), ("ʌː", "V^"), ("ɨ", "U^")):
    IPA[k] = v

NONPHONE = ("<", "spn", "sil", "sp", "")


def is_phone(label: str) -> bool:
    """人手層の <SIL> <LAUGH-…> <NOISE> 等や MFA の spn を落とす。"""
    s = label.strip()
    return bool(s) and not s.startswith("<") and s not in ("spn", "sil", "sp")


def align(a, b):
    """クラス列 a, b の編集距離アライメントを返す。

    戻り値は (i, j) の組の列で、**クラスが一致した対だけ**を含む。
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]
