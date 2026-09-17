# SCOPE: journal-only
"""共通部品：範疇軸（無声化の変異形選択）の検定（docs/decisions-log.md 2026-09-17 事前登録）。

- 辞書の読み込み（MFA 3.4.1 と同じ正規化：NFKC、小文字化）
- 発音どうしの位置対応（重みつき編集距離）
- 当該母音位置の値：V（有声記号）・D（無声化記号）・A（母音なし）
- D6 の定義（参照側のカテゴリ、無声阻害音）
"""

from __future__ import annotations

import hashlib
import os
import unicodedata
from pathlib import Path

DICT_PATH = Path(os.environ.get("MFA_JAPANESE_DICT", "japanese_mfa.dict")).expanduser()

# D6（fa-validity-jako docs/decisions-log.md）
REF_DEVOICED_MARKS = {"i̥", "ɯ̥"}
REF_VOICED_HIGH = {"i", "ɯ"}
VOICED_VOWELS = {"a", "aː", "e", "eː", "i", "iː", "o", "oː", "ɯ", "ɯː", "ɨ", "ɨː"}
DEVOICED_VOWELS = {"i̥", "ɯ̥", "ɨ̥"}
ALL_VOWELS = VOICED_VOWELS | DEVOICED_VOWELS
VOICELESS_CONTEXT_CLASSES = {"stop", "fricative", "affricate", "geminate_stop", "geminate_affricate"}
# D6 の有声頭文字に ɟ を加えた（D6 の列挙は ɟ を落としており、有声の硬口蓋破裂音を無声扱いしてしまう）
VOICED_OBSTRUENT_INITIALS = ("b", "d", "ɡ", "g", "z", "ʑ", "v", "ɟ")
# 記号で見る無声阻害音（融合区間の子音部と、辞書内の環境判定に使う）
VOICELESS_OBSTRUENT_SYMBOLS = {
    "k", "t", "p", "c", "tʲ", "pʲ", "kː", "tː", "pː", "cː", "tʲː", "pʲː",
    "s", "ɕ", "h", "ç", "ɸ", "ɸʲ", "sː", "ɕː", "hː", "çː", "ɸː", "ɸʲː",
    "ts", "tɕ", "tsː", "tɕː",
}
HIGH_BASE = {"i": "I", "i̥": "I", "iː": "I", "ɯ": "U", "ɯ̥": "U", "ɨ": "U", "ɨ̥": "U", "ɯː": "U", "ɨː": "U"}


def norm_word(w: str) -> str:
    return unicodedata.normalize("NFKC", w).lower()


def word_hash(w: str) -> str:
    return hashlib.sha1(norm_word(w).encode("utf-8")).hexdigest()[:12]


def value(ph: str | None) -> str | None:
    if ph is None:
        return "A"
    if ph in VOICED_VOWELS:
        return "V"
    if ph in DEVOICED_VOWELS:
        return "D"
    return None


def load_dictionary(path: Path = DICT_PATH) -> dict[str, list[tuple[str, ...]]]:
    """語 → 発音の一覧（重複除去、出現順）。MFA の parse_dictionary_file と同じ規則で確率列を外す。"""
    import re

    prob = re.compile(r"\b(\d+\.\d+|1)\b")
    out: dict[str, list[tuple[str, ...]]] = {}
    seen: set[tuple[str, tuple[str, ...]]] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            w = norm_word(parts.pop(0))
            for _ in range(4):
                if parts and prob.match(parts[0]):
                    parts.pop(0)
                else:
                    break
            pron = tuple(parts)
            if (w, pron) in seen:
                continue
            seen.add((w, pron))
            out.setdefault(w, []).append(pron)
    return out


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    av, bv = a in ALL_VOWELS, b in ALL_VOWELS
    if av != bv:
        return 9.0  # 母音と子音は対応させない
    if av and HIGH_BASE.get(a) and HIGH_BASE.get(a) == HIGH_BASE.get(b):
        return 0.4  # 有声・無声だけが違う同じ高母音
    return 1.0


def align_prons(a: tuple[str, ...], b: tuple[str, ...]):
    """a の各位置に対応する b の位置（なければ None）と、a の各すき間 g（0..len(a)）に挿入された b の音素列を返す。"""
    n, m = len(a), len(b)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            if i > 0 and j > 0:
                best = min(best, D[i - 1][j - 1] + _sub_cost(a[i - 1], b[j - 1]))
            if i > 0:
                best = min(best, D[i - 1][j] + 1.0)
            if j > 0:
                best = min(best, D[i][j - 1] + 1.0)
            D[i][j] = best
    amap: list[int | None] = [None] * n
    ins: list[list[str]] = [[] for _ in range(n + 1)]
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + _sub_cost(a[i - 1], b[j - 1]))) < 1e-9:
            amap[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + 1.0)) < 1e-9:
            i -= 1
        else:
            ins[i].insert(0, b[j - 1])
            j -= 1
    return amap, ins


def values_at(chosen: tuple[str, ...], variants: list[tuple[str, ...]], pos: int | None, gap: int | None):
    """chosen の位置 pos（または すき間 gap）に対する、各発音の値の集合。"""
    vals = set()
    for v in variants:
        if v == chosen:
            vals.add(value(chosen[pos]) if pos is not None else "A")
            continue
        amap, ins = align_prons(chosen, v)
        if pos is not None:
            j = amap[pos]
            vals.add(value(v[j]) if j is not None else "A")
        else:
            inserted = [value(p) for p in ins[gap] if p in ALL_VOWELS]
            if "V" in inserted:
                vals.add("V")
            elif "D" in inserted:
                vals.add("D")
            else:
                vals.add("A")
    vals.discard(None)
    return vals


def competition(vals: set[str]) -> tuple[str, str]:
    """(status, type)。status: competing / lexical；type: symbol / presence / mixed / ''。"""
    dev = vals & {"D", "A"}
    if "V" in vals and dev:
        if dev == {"D"}:
            return "competing", "symbol"
        if dev == {"A"}:
            return "competing", "presence"
        return "competing", "mixed"
    return "lexical", ""
