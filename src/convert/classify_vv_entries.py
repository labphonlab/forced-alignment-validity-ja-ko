# SCOPE: shared
"""VV監査（audit_vv_entries.py）の結果を機械的に事前分類し、人手判断の残数を絞る。

前提の訂正
----------
当初は「MFA辞書の形態素境界情報を使って自動分類する」方針だったが、
japanese_mfa辞書を実地に検査した結果、**形態素境界情報もアクセント句情報も
存在しない**ことが確認された。辞書の構造は次の2形式のみである。

    表記 \\t 音素列
    表記 \\t p1 \\t p2 \\t p3 \\t p4 \\t 音素列      (確率つきエントリ)

音素列・表記のいずれにも境界マーカー（# + / = ^ ' 等）は0件。
（CSJ側の csj.htkdic は `１０ｄＢ＋名詞/数詞` のようにPOSを持つが、それとは別物。）

そこで、辞書内で実際に得られる次の4シグナルで代替する。

  S1 cross-reference（最強）: `V V` を `Vː` に正規化した音素列が、辞書内の
     他エントリの音素列として実在するか。実在すれば、同一の音韻語が2通りに
     書かれている証拠であり merge 候補（例: お姉さん o n e e s a ɴ に対し
     おねえさん o n eː s a ɴ が実在）。
  S2 表記の母音重ね: 表記に同一母音の仮名連続（かあ・ねえ・とお 等）または
     長音符「ー」が含まれる → merge 候補。
  S3 字読み: 表記が全角英数字・ラテン文字主体（略語の一字ずつ読み。
     例: ｎａｎ → e n ɯ e e n ɯ）→ keep。
  S4 句エントリ: 表記に助詞（を・は・が・に・へ・と・も・の）が含まれる
     → 複数語からなる句であり形態素境界をまたぐ → keep。

シグナルが競合した場合は keep 側（統合しない側）を優先する。統合しすぎる方が
統合し損ねるより誤差指標への害が大きいため（偽の境界を消すと誤差が過小評価される）。

出力
----
標準出力: 分類件数の内訳。
`--out-uncertain`: どのシグナルにも当たらなかった残りをTSVで出力（人手判断用）。
`--out-classified`: 自動分類できた分もTSVで出力（抜き取り検証用）。

使い方
------
    python3 src/convert/classify_vv_entries.py path/to/japanese_mfa.dict \\
        --out-uncertain results/vv_audit_uncertain.tsv \\
        --out-classified results/vv_audit_classified.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_vv_entries import find_vv, parse_line  # noqa: E402

# 仮名 -> その仮名が担う母音。表記側の母音重ね検出に使う。
KANA_VOWEL = {}
for _vowel, _rows in {
    "a": "あかがさざただなはばぱまやらわアカガサザタダナハバパマヤラワ",
    "i": "いきぎしじちぢにひびぴみりイキギシジチヂニヒビピミリ",
    "u": "うくぐすずつづぬふぶぷむゆるウクグスズツヅヌフブプムユル",
    "e": "えけげせぜてでねへべぺめれエケゲセゼテデネヘベペメレ",
    "o": "おこごそぞとどのほぼぽもよろオコゴソゾトドノホボポモヨロ",
}.items():
    for _k in _rows:
        KANA_VOWEL[_k] = _vowel

# 表記上、長音を表す第二要素になりうる仮名
VOWEL_KANA = {"あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
              "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o"}

PARTICLES = set("をはがにへともの")
LATIN_ALNUM = re.compile(r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９]")


def normalize_vv(phones: list[str]) -> list[str]:
    """同一短母音の隣接を長音記号つき単一音素に畳む。"""
    out: list[str] = []
    i = 0
    while i < len(phones):
        if i + 1 < len(phones):
            a = unicodedata.normalize("NFC", phones[i])
            b = unicodedata.normalize("NFC", phones[i + 1])
            if a == b and a in {"a", "e", "i", "o", "ɯ", "ɨ"}:
                out.append(a + "ː")
                i += 2
                continue
        out.append(phones[i])
        i += 1
    return out


def has_doubled_vowel_kana(surface: str) -> bool:
    """表記に同一母音の仮名連続、または長音符が含まれるか。"""
    if "ー" in surface:
        return True
    for i in range(len(surface) - 1):
        second = VOWEL_KANA.get(surface[i + 1])
        if second is None:
            continue
        if KANA_VOWEL.get(surface[i]) == second:
            return True
    return False


def is_letter_spelling(surface: str) -> bool:
    """略語の一字ずつ読みらしい表記か（ラテン英数字が主体）。"""
    if not surface:
        return False
    latin = len(LATIN_ALNUM.findall(surface))
    return latin >= 2 and latin / len(surface) >= 0.6


def contains_particle(surface: str) -> bool:
    return any(ch in PARTICLES for ch in surface)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dict_path", type=Path)
    ap.add_argument("--out-uncertain", type=Path)
    ap.add_argument("--out-classified", type=Path)
    args = ap.parse_args(argv)

    entries: list[tuple[str, list[str]]] = []
    all_prons: set[str] = set()

    with args.dict_path.open(encoding="utf-8") as fh:
        for line in fh:
            parsed = parse_line(line)
            if parsed is None:
                continue
            word, phones = parsed
            entries.append((word, phones))
            all_prons.add(" ".join(phones))

    counts: Counter = Counter()
    classified: list[tuple[str, str, str, str]] = []
    uncertain: list[tuple[str, str, str]] = []

    for word, phones in entries:
        hits = find_vv(phones)
        if not hits:
            continue
        pron = " ".join(phones)
        normalized = " ".join(normalize_vv(phones))

        # keep 側のシグナルを先に評価する（競合時は keep 優先）
        if "・" in word:
            verdict, signal = "keep", "S5:nakaguro-compound"
        elif is_letter_spelling(word):
            verdict, signal = "keep", "S3:letter-spelling"
        elif contains_particle(word):
            verdict, signal = "keep", "S4:particle-phrase"
        elif normalized != pron and normalized in all_prons:
            verdict, signal = "merge", "S1:cross-reference"
        elif has_doubled_vowel_kana(word):
            verdict, signal = "merge", "S2:doubled-kana"
        else:
            counts["uncertain"] += 1
            uncertain.append((word, pron, ",".join(v for _, v in hits)))
            continue

        counts[f"{verdict} / {signal}"] += 1
        counts[f"_total_{verdict}"] += 1
        classified.append((word, pron, verdict, signal))

    total = counts["uncertain"] + counts["_total_merge"] + counts["_total_keep"]
    print(f"V V entries total           : {total}")
    print()
    for key in sorted(k for k in counts if not k.startswith("_") and k != "uncertain"):
        print(f"  {key:<28} {counts[key]:>6}")
    print()
    print(f"  {'auto-classified merge':<28} {counts['_total_merge']:>6}")
    print(f"  {'auto-classified keep':<28} {counts['_total_keep']:>6}")
    print(f"  {'requires manual review':<28} {counts['uncertain']:>6}"
          f"  ({100.0 * counts['uncertain'] / total:.1f}%)")

    if args.out_uncertain:
        args.out_uncertain.parent.mkdir(parents=True, exist_ok=True)
        with args.out_uncertain.open("w", encoding="utf-8") as fh:
            fh.write("word\tpronunciation\tvv_vowels\tdecision\n")
            for row in uncertain:
                fh.write("\t".join(row) + "\t\n")
        print(f"\nwrote {len(uncertain)} rows to {args.out_uncertain}")

    if args.out_classified:
        args.out_classified.parent.mkdir(parents=True, exist_ok=True)
        with args.out_classified.open("w", encoding="utf-8") as fh:
            fh.write("word\tpronunciation\tverdict\tsignal\n")
            for row in classified:
                fh.write("\t".join(row) + "\n")
        print(f"wrote {len(classified)} rows to {args.out_classified}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
