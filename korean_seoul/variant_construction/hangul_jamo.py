"""ハングル音節分解ユーティリティと7種音変化(liaison/nasalization_obstruent/
nasalization_liquid/aspiration/tensification/liquidization/palatalization)の
環境判定規則。extract_candidates.py(転写差分ベース)とfind_environments.py
(綴りの音韻環境ベース)で共有する。

2026-08-05: JAKLE 2026年7月大会の討論(呉相珉氏)で「鼻音化は阻害音由来と流音由来に
区別できる」との指摘を受け、当初の nasalization を nasalization_obstruent
(阻害音終声+鼻音初声、例: 밥만→밤만) と nasalization_liquid(終声+ㄹ初声→ㄴに、
例: 종로→종노)に分割。あわせてv3(PDLIジャーナル草稿)Table2の6過程のうち
未実装だった tensification(濃音化)・liquidization(流音化)を追加し、6過程
全てをカバーする。
"""

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3


def decompose(char: str):
    """1音節ハングルを (cho, jung, jong) に分解する。非ハングルは None。"""
    code = ord(char)
    if not (HANGUL_BASE <= code <= HANGUL_LAST):
        return None
    offset = code - HANGUL_BASE
    jong_idx = offset % 28
    jung_idx = (offset // 28) % 21
    cho_idx = offset // (28 * 21)
    return CHO[cho_idx], JUNG[jung_idx], JONG[jong_idx]


def compose(cho: str, jung: str, jong: str = ""):
    """(cho, jung, jong) から1音節ハングルを合成する(decomposeの逆演算)。
    2026-08-10: eojeol境界を跨ぐliaisonのcorrect resyllabification綴り生成用に追加。
    """
    cho_idx = CHO.index(cho)
    jung_idx = JUNG.index(jung)
    jong_idx = JONG.index(jong)
    code = HANGUL_BASE + (cho_idx * 21 + jung_idx) * 28 + jong_idx
    return chr(code)



# 複合終声(겹받침)の分解。(基本, 削除される終声側) の順。
# 一般則: 第2子音が次音節の初声に移動し、第1子音がそのまま終声として残る
# (例: 닭이→달기[ㄹ残/ㄱ移動], 값이→갑씨[ㅂ残/ㅅ移動])。
# ただしㅎで終わる複合終声(ㄶ・ㅀ)はㅎが脱落し、第1子音の方が初声に移動する
# (例: 많아→마나[ㅎ脱落・ㄴ移動]、앓아→아라[ㅎ脱落・ㄹ移動])——例外として個別指定。
COMPOUND_JONG_STAY_MOVE = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅄ": ("ㅂ", "ㅅ"),
}
COMPOUND_JONG_H_DELETE_MOVE = {"ㄶ": "ㄴ", "ㅀ": "ㄹ"}


def resyllabify_at_boundary(left_token: str, right_token: str):
    """left_tokenの最終音節の終声を、right_token先頭音節(null onset=ㅇ)の
    初声へ移動させた「連音化後の正しい綴り」を生成する(2026-08-10新設、
    2026-08-13に複合終声(겹받침)対応を追加)。

    G2Pに人工的に結合した偽単語をそのまま渡すと、境界の子音を正しく再音節化
    せず単純に脱落させる誤った予測をすることが判明した(README §12参照)。
    この関数で綴りの時点で正しく再音節化してからG2Pに渡すことで、G2Pが
    通常の単語と同様に扱えるようにする。

    単純終声はそのまま初声へ移動。複合終声(ㄳㄵㄺㄻㄼㄽㄾㄿㅄ)は第2子音が
    初声へ移動し第1子音が終声として残る。ㅎを含む複合終声(ㄶㅀ)はㅎが脱落し
    第1子音(ㄴ/ㄹ)が初声へ移動する。
    """
    if not left_token or not right_token:
        return None
    left_dec = decompose(left_token[-1])
    right_dec = decompose(right_token[0])
    if left_dec is None or right_dec is None:
        return None
    cho, jung, jong = left_dec
    cho2, jung2, jong2 = right_dec
    if jong == "" or cho2 != "ㅇ":
        return None

    if jong in CHO:
        stay_jong, move_cho = "", jong
    elif jong in COMPOUND_JONG_STAY_MOVE:
        stay, move_cho = COMPOUND_JONG_STAY_MOVE[jong]
        stay_jong = stay
    elif jong in COMPOUND_JONG_H_DELETE_MOVE:
        stay_jong = ""
        move_cho = COMPOUND_JONG_H_DELETE_MOVE[jong]
    else:
        return None

    new_last_left = compose(cho, jung, stay_jong)
    new_first_right = compose(move_cho, jung2, jong2)
    new_left = left_token[:-1] + new_last_left
    new_right = new_first_right + right_token[1:]
    return new_left, new_right


NASAL_MAP = {  # 閉鎖音終声 -> 鼻音化後の終声 (先行子音の調音位置を保持)
    "ㄱ": "ㅇ", "ㄲ": "ㅇ", "ㅋ": "ㅇ", "ㄳ": "ㅇ", "ㄺ": "ㅇ",
    "ㄷ": "ㄴ", "ㅅ": "ㄴ", "ㅆ": "ㄴ", "ㅈ": "ㄴ", "ㅊ": "ㄴ", "ㅌ": "ㄴ", "ㅎ": "ㄴ",
    "ㅂ": "ㅁ", "ㅍ": "ㅁ", "ㄼ": "ㅁ", "ㄿ": "ㅁ", "ㅄ": "ㅁ",
}
ASPIRATE_MAP = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
PALATAL_MAP = {"ㄷ": "ㅈ", "ㅌ": "ㅊ"}


def find_environments(token: str):
    """1つのeojeol(語節)内で、隣接音節境界に生じうる音変化の「環境」を
    綴り(citation form)だけから機械的に検出する(実際に変化が生じたかは問わない)。
    転写の差分に頼らないため、正書法が形態音素的な韓国語でも候補を取りこぼさない。

    戻り値: [(i, change_type), ...]  # i = 終声を持つ側の音節インデックス(0始まり)
    """
    decomposed = [decompose(c) for c in token]
    sites = []
    for i in range(len(token) - 1):
        cur = decomposed[i]
        nxt = decomposed[i + 1]
        if cur is None or nxt is None:
            continue
        cur_cho, cur_jung, cur_jong = cur
        nxt_cho, nxt_jung, nxt_jong = nxt
        if cur_jong == "":
            continue

        if cur_jong == "ㅆ":
            # ㅆ終声(-았/었/였/했/있-等の用言語幹末)は除外する。変異形競合方式の
            # 大規模実行(2026-08-05, n=2,000)で、korean_mfaのG2Pがこの形態素の
            # 終声を鼻音化(nasalization環境)でもliaison環境でも教科書的規則
            # (鼻音化/有声化・流音化)通りに予測せず、代わりに濃音の摩擦音(s͈/s)
            # をほぼ一貫して予測することが判明した(例: 있는→sʰ͈保持、있어→sのまま)。
            # 候補分類を汚染する系統的な癖のため、環境検出の時点で除外する。
            continue
        if cur_jong in NASAL_MAP and nxt_cho in ("ㄴ", "ㅁ"):
            sites.append((i, "nasalization_obstruent"))
            continue
        if cur_jong not in ("", "ㄴ", "ㄹ") and nxt_cho == "ㄹ":
            # 流音由来の鼻音化(유음의 비음화): ㄹ以外の終声+ㄹ初声 -> 初声がㄴに
            # (例: 종로→종노, 능력→능녁)。終声がㄴの場合はㄴ+ㄹ→ㄹㄹ(流音化)に
            # なるため除外(liquidizationとして別途判定)。
            sites.append((i, "nasalization_liquid"))
            continue
        if cur_jong in PALATAL_MAP and nxt_cho == "ㅇ" and nxt_jung == "ㅣ":
            sites.append((i, "palatalization"))
            continue
        if cur_jong == "ㅎ" and nxt_cho in ASPIRATE_MAP:
            sites.append((i, "aspiration"))
            continue
        if cur_jong in ASPIRATE_MAP and nxt_cho == "ㅎ":
            sites.append((i, "aspiration"))
            continue
        if cur_jong in NASAL_MAP and nxt_cho in ("ㄱ", "ㄷ", "ㅂ", "ㅅ", "ㅈ"):
            # 濃音化(농음화): 阻害音終声(閉鎖音化する終声集合を流用)+平音初声
            # -> 初声が濃音に(例: 학교→학꾜)。ㅎとの結合は激音化が優先(直前で処理済み)。
            sites.append((i, "tensification"))
            continue
        if (cur_jong == "ㄴ" and nxt_cho == "ㄹ") or (cur_jong == "ㄹ" and nxt_cho == "ㄴ"):
            # 流音化(유음화): ㄴ+ㄹ・ㄹ+ㄴ -> ㄹㄹ(例: 신라→실라, 칼날→칼랄)。
            sites.append((i, "liquidization"))
            continue
        if cur_jong not in ("ㅇ", "ㄴ", "ㅁ") and nxt_cho == "ㅇ":
            # /ŋ/(終声ㅇ)は音節初声になれないため除外(単純脱落と区別できない)。
            # ㄴ/ㅁ終声もパイロット検証(2026-08-05)で確認済み: 資格化しても
            # 音素表記が変わらず(共に単純にn/mのまま)、音響的コントラストを
            # 生まないため除外(48発話パイロットのNO_CONTRAST 4/8はすべてこの型)。
            sites.append((i, "liaison"))
            continue
    return sites
