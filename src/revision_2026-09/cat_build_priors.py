# SCOPE: journal-only
"""C4(b)：学習用139講演の参照ラベルから語ごとの無声化率を推定し、辞書の写しを2つ作る。

  oos   : 競合する語の各発音に、競合位置ごとの積 Π(無声化型なら p、有声型なら 1−p) を確率列として書く
  equal : 競合する語の各発音の確率列を 1.0 に揃える
  それ以外の行はそのまま写す（確率列・無音の列を含めて変えない）。

語ごとの率 p は、学習用講演の「競合あり」トークン（main177 の整列で語を同定）の無声化の割合。
出現5回未満の語は、環境別の全体率（学習用講演の競合ありトークン）で置き換える。環境は辞書の
基準発音（母音記号の数が最大の発音）の中で、その位置の前後が両方とも無声阻害音かで決める。
p は [0.01, 0.99] に切る（MFA 3.4.1 は 0.01 未満を 0.01 に切り上げる）。

出力：辞書（fa-validity-jako/work/dict_rev/）、語ごとの率の集計（語はハッシュ）。
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from cat_common import (
    ALL_VOWELS, DICT_PATH, HIGH_BASE, VOICELESS_OBSTRUENT_SYMBOLS, align_prons, load_dictionary,
    norm_word, value, word_hash,
)

MIN_N = 5
PROB = re.compile(r"\b(\d+\.\d+|1)\b")


def anchor_positions(variants):
    """基準発音と、競合する位置 → {発音: 値}。"""
    anchor = max(variants, key=lambda v: (sum(p in ALL_VOWELS for p in v), len(v)))
    per_pos = defaultdict(dict)
    for v in variants:
        amap, _ = align_prons(anchor, v)
        for j, ph in enumerate(anchor):
            if ph not in ALL_VOWELS or not HIGH_BASE.get(ph):
                continue
            k = amap[j]
            per_pos[j][v] = value(v[k]) if k is not None else "A"
    comp = {}
    for j, vals in per_pos.items():
        s = set(vals.values()) - {None}
        if "V" in s and s & {"D", "A"}:
            comp[j] = vals
    return anchor, comp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True, help="main177 のトークン表")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args(argv)

    n_dev = Counter()
    n_all = Counter()
    env_dev = Counter()
    env_all = Counter()
    with args.tokens.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["split"] != "train" or r["status"] != "competing":
                continue
            d = r["ref_label"] == "devoiced"
            n_all[r["word_id"]] += 1
            n_dev[r["word_id"]] += d
            env_all[r["env"]] += 1
            env_dev[r["env"]] += d
    env_rate = {e: env_dev[e] / env_all[e] for e in env_all}

    lex = load_dictionary()
    comp_words = {}
    n_symbol_words = 0
    for w, variants in lex.items():
        if len(variants) < 2:
            continue
        anchor, comp = anchor_positions(variants)
        if comp:
            comp_words[w] = (anchor, comp)
            if any(set(vals.values()) >= {"V", "D"} for vals in comp.values()):
                n_symbol_words += 1

    prior = {}
    stats = Counter()
    for w, (anchor, comp) in comp_words.items():
        h = word_hash(w)
        n = n_all.get(h, 0)
        pos_p = {}
        for j in comp:
            if n >= MIN_N:
                p = n_dev[h] / n
            else:
                prev_vl = j > 0 and anchor[j - 1] in VOICELESS_OBSTRUENT_SYMBOLS
                next_vl = j + 1 < len(anchor) and anchor[j + 1] in VOICELESS_OBSTRUENT_SYMBOLS
                p = env_rate["voiceless_both" if prev_vl and next_vl else "other"]
            pos_p[j] = min(0.99, max(0.01, p))
        stats["words_word_rate" if n >= MIN_N else "words_env_fallback"] += 1
        probs = {}
        for v in {v for vals in comp.values() for v in vals}:
            pr = 1.0
            for j, vals in comp.items():
                val = vals.get(v)
                if val in ("D", "A"):
                    pr *= pos_p[j]
                elif val == "V":
                    pr *= 1 - pos_p[j]
            probs[v] = pr
        prior[w] = probs

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_oos = (args.outdir / "japanese_mfa_prior_oos.dict").open("w", encoding="utf-8")
    out_eq = (args.outdir / "japanese_mfa_prior_equal.dict").open("w", encoding="utf-8")
    changed = Counter()
    with DICT_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            w = norm_word(parts[0])
            if w not in prior:
                out_oos.write(line)
                out_eq.write(line)
                continue
            has_prob = len(parts) > 2 and PROB.fullmatch(parts[1]) is not None
            pron = tuple(parts[-1].split())
            p = prior[w].get(pron, 1.0)
            for fo, val in ((out_oos, f"{p:.4f}"), (out_eq, "1.0")):
                new = [parts[0], val] + (parts[2:] if has_prob else parts[1:])
                fo.write("\t".join(new) + "\n")
            changed["lines_changed"] += 1
            changed["lines_had_prob"] += has_prob
            changed["lines_not_in_competing_positions"] += pron not in prior[w]
    out_oos.close()
    out_eq.close()

    with args.summary.open("w", encoding="utf-8") as fo:
        fo.write("quantity\tvalue\n")
        rows = [
            ("dict_words", len(lex)),
            ("competing_words_any_type", len(comp_words)),
            ("competing_words_with_devoiced_symbol_variant", n_symbol_words),
            ("train_competing_tokens", sum(env_all.values())),
            ("train_word_types_with_competing_tokens", len(n_all)),
            ("train_word_types_n_ge_5", sum(1 for h in n_all if n_all[h] >= MIN_N)),
            ("env_rate_voiceless_both", env_rate.get("voiceless_both")),
            ("env_rate_other", env_rate.get("other")),
            ("env_n_voiceless_both", env_all["voiceless_both"]),
            ("env_n_other", env_all["other"]),
            *stats.items(), *changed.items(),
        ]
        for k, v in rows:
            fo.write(f"{k}\t{v:.4f}\n" if isinstance(v, float) else f"{k}\t{v}\n")
    for k, v in rows:
        print(f"{k:<50} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
