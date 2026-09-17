# SCOPE: journal-only
"""C1：参照側の高母音トークン表を作る（docs/decisions-log.md 2026-09-17 事前登録）。

入力（どれも中身は表示しない。集計だけを標準出力に出す）
  --ref        参照単位（csj_units.tsv 形式）
  --alignment  sequence_align.py の出力
  --textgrids  MFA の TextGrid（phones/words 層。話者名つきの層名でも可）
  --talks      対象講演（style で絞る場合は --styles）
出力
  --out        トークン表（results/revision_2026-09/work/ のみ）
  --attrition  減少の記録（集計）

整列側の単位ID（<file>_mfa_NNNN）は src/convert/mfa_textgrid_to_units.py の convert_file を
そのまま呼んで再現する（改変しない）。再現した単位が --hyp-units と一致するかを確かめる。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "convert"))

from cat_common import (  # noqa: E402
    ALL_VOWELS, HIGH_BASE, REF_DEVOICED_MARKS, REF_VOICED_HIGH, VOICED_OBSTRUENT_INITIALS,
    VOICELESS_CONTEXT_CLASSES, VOICELESS_OBSTRUENT_SYMBOLS, competition, load_dictionary,
    norm_word, value, values_at, word_hash,
)
import mfa_textgrid_to_units as m2u  # noqa: E402
from schema import Corpus, MetricType, Style  # noqa: E402

csv.field_size_limit(sys.maxsize)
TIER_RE = re.compile(r'(name = ")[^"\n]* - (words|phones)(")')
SKIP_NEXT_CLASSES = {"auxiliary_closure", "geminate_no_closure"}


def ref_category(cls: str, lab: str) -> str | None:
    if set(lab.split()) & REF_DEVOICED_MARKS:
        return "devoiced"
    if cls == "vowel_short" and lab in REF_VOICED_HIGH:
        return "voiced"
    return None


def voiceless_unit(cls: str, lab: str) -> bool:
    first = lab.split()[0] if lab.strip() else ""
    if cls in VOICELESS_CONTEXT_CLASSES:
        return not first.startswith(VOICED_OBSTRUENT_INITIALS)
    if cls.startswith("devoiced_fused"):
        return first in VOICELESS_OBSTRUENT_SYMBOLS
    return False


def load_ref(path: Path, talks: set[str]):
    by_file = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["file_id"] in talks:
                by_file[r["file_id"]].append(r)
    info = {}
    for fid, rows in by_file.items():
        rows.sort(key=lambda r: float(r["t_start"]))
        for k, r in enumerate(rows):
            cat = ref_category(r["phone_class"], r["label_canonical"])
            if cat is None:
                continue
            lab = r["label_canonical"]
            fused = len(lab.split()) > 1
            ipu = r["ipu_id"]
            # 先行：融合区間なら単位内の子音、母音だけの単位なら同じ IPU の直前の単位
            if fused:
                prev_vl = lab.split()[0] in VOICELESS_OBSTRUENT_SYMBOLS
            else:
                p = rows[k - 1] if k > 0 else None
                prev_vl = bool(p and p["ipu_id"] == ipu and voiceless_unit(p["phone_class"], p["label_canonical"]))
            # 後続：同じ IPU の直後の単位（促音の閉鎖区間は読み飛ばす）
            nxt = None
            q = k + 1
            while q < len(rows) and rows[q]["ipu_id"] == ipu:
                if rows[q]["phone_class"] in SKIP_NEXT_CLASSES:
                    q += 1
                    continue
                nxt = rows[q]
                break
            next_vl = bool(nxt and voiceless_unit(nxt["phone_class"], nxt["label_canonical"]))
            info[r["unit_id"]] = {
                "file_id": fid, "speaker": r["speaker_id"], "style": r["style"], "ref_cat": cat,
                "ref_class": r["phone_class"], "fused": fused,
                "env": "voiceless_both" if prev_vl and next_vl else "other",
            }
    return by_file, info


def read_hyp(tg_dir: Path, talks: set[str], hyp_units: Path | None):
    """単位ID → (語インスタンス番号, 語内の生の位置, ラベル)、語インスタンス → (語, 発音)。"""
    classes = m2u.load_phone_classes()
    vv = m2u.load_vv_merge_words()
    unit_pos: dict[str, tuple[str, int | None, int | None]] = {}
    unit_seq: dict[str, list[str]] = {}  # file → 評価可能な単位IDの順序
    words_of: dict[str, tuple[str, tuple[str, ...]]] = {}
    unit_label: dict[str, tuple[str, float]] = {}
    paths = [p for p in sorted(tg_dir.rglob("*.TextGrid")) if p.stem in talks]
    with tempfile.TemporaryDirectory() as td:
        for p in paths:
            text = p.read_text(encoding="utf-8")
            norm = TIER_RE.sub(r"\1\2\3", text)
            q = Path(td) / p.name
            q.write_text(norm, encoding="utf-8")
            units, unknown = m2u.convert_file(q, classes, Corpus("csj"), MetricType("accuracy"),
                                              Style("aps"), "csj", vv)
            words, phones = m2u._read_textgrid(q)  # noqa: SLF001
            raw_word = []
            for iv in phones:
                _, widx = m2u._word_at(words, iv.start, iv.end)  # noqa: SLF001
                raw_word.append(widx)
            prons = defaultdict(list)
            raw_pos = []
            for iv, widx in zip(phones, raw_word):
                if widx >= 0:
                    raw_pos.append(len(prons[widx]))
                    prons[widx].append((iv.label or "").strip())
                else:
                    raw_pos.append(None)
            for widx, ph in prons.items():
                words_of[f"{p.stem}#{widx}"] = (words[widx].label or "", tuple(ph))
            k = 0
            seq = []
            for u in units:
                widx = raw_word[k]
                unit_pos[u.unit_id] = (f"{p.stem}#{widx}" if widx >= 0 else None, raw_pos[k], u.label_canonical)
                unit_label[u.unit_id] = (u.label_canonical, round(u.t_start, 4))
                if u.is_analyzable:
                    seq.append(u.unit_id)
                k += u.merged_from
            assert k == len(phones), p
            unit_seq[p.stem] = seq
    mismatch = None
    if hyp_units is not None:
        mismatch = [0, 0]
        with hyp_units.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["file_id"] not in talks:
                    continue
                mismatch[1] += 1
                got = unit_label.get(r["unit_id"])
                if got is None or got[0] != r["label_canonical"] or abs(got[1] - float(r["t_start"])) > 1e-3:
                    mismatch[0] += 1
    return unit_pos, unit_seq, words_of, mismatch, len(paths)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--textgrids", type=Path, required=True)
    ap.add_argument("--hyp-units", type=Path)
    ap.add_argument("--styles", nargs="*", default=None)
    ap.add_argument("--split", type=Path, help="fa-validity-jako の分割表（split 列を付ける）")
    ap.add_argument("--split-filter", default=None, help="この split の講演だけ")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attrition", type=Path, required=True)
    args = ap.parse_args(argv)

    split = {}
    if args.split:
        with args.split.open(encoding="utf-8") as fh:
            split = {r["file_id"]: r["split"] for r in csv.DictReader(fh, delimiter="\t")}
    talks = set()
    with args.ref.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if args.styles and r["style"] not in args.styles:
                continue
            if args.split_filter and split.get(r["file_id"]) != args.split_filter:
                continue
            talks.add(r["file_id"])
    by_file, info = load_ref(args.ref, talks)
    unit_pos, unit_seq, words_of, mismatch, n_tg = read_hyp(args.textgrids, talks, args.hyp_units)
    seq_index = {u: (f, i) for f, s in unit_seq.items() for i, u in enumerate(s)}
    lexicon = load_dictionary()

    att = Counter()
    for v in info.values():
        att[f"0_ref_high_vowel_{v['ref_cat']}"] += 1

    rows_out = []
    with args.alignment.open(encoding="utf-8") as fh:
        align_rows = [r for r in csv.DictReader(fh, delimiter="\t") if r["file_id"] in talks]
    by_file_rows = defaultdict(list)
    for r in align_rows:
        by_file_rows[r["file_id"]].append(r)

    cache: dict = {}

    def word_values(wkey, pos, gap):
        word, pron = words_of[wkey]
        key = norm_word(word)
        variants = lexicon.get(key)
        if not variants or pron not in variants:
            return None
        ck = (key, pron, pos, gap)
        if ck not in cache:
            cache[ck] = values_at(pron, variants, pos, gap)
        return cache[ck]

    for fid, rows in by_file_rows.items():
        for idx, r in enumerate(rows):
            uid = r["ref_unit_id"]
            if not uid or uid not in info:
                continue
            ti = info[uid]
            att[f"1_in_alignment_{ti['ref_cat']}"] += 1
            et = r["error_type"]
            hyp_ids = [h for h in r["hyp_unit_id"].split(";") if h]
            onset = offset = ""
            wkey = pos = gap = None
            reason = ""
            if et != "omission":
                onset, offset = r["onset_error_ms"], r["offset_error_ms"]
                vowel_units = [h for h in hyp_ids if unit_pos[h][2] in ALL_VOWELS]
                if len(vowel_units) > 1:
                    high = [h for h in vowel_units if HIGH_BASE.get(unit_pos[h][2])]
                    vowel_units = high if len(high) == 1 else vowel_units
                if len(vowel_units) == 1:
                    wkey, pos, _ = unit_pos[vowel_units[0]]
                elif len(vowel_units) > 1:
                    reason = "ambiguous_vowel_units"
                elif ti["fused"]:
                    # 母音記号なし：参照が融合区間なら母音の削除（D6）。すき間は最後の単位の後ろ
                    wkey, p_last, _ = unit_pos[hyp_ids[-1]]
                    gap = None if p_last is None else p_last + 1
                    if gap is None:
                        wkey = None
                else:
                    reason = "d6_excluded_no_vowel"
                if not reason and (wkey is None or (pos is None and gap is None)):
                    reason = "no_word"
            else:
                # 脱落：前後の最も近い整列側単位のあいだのすき間
                prev_u = next((h for rr in reversed(rows[:idx]) for h in reversed([x for x in rr["hyp_unit_id"].split(";") if x])), None)
                next_u = next((h for rr in rows[idx + 1:] for h in [x for x in rr["hyp_unit_id"].split(";") if x]), None)
                if prev_u is None or next_u is None:
                    reason = "omission_edge"
                else:
                    fp, ip = seq_index[prev_u]
                    fn, in_ = seq_index[next_u]
                    wp, pp, _ = unit_pos[prev_u]
                    wn, pn, _ = unit_pos[next_u]
                    if fp != fn or in_ != ip + 1:
                        reason = "omission_not_adjacent"
                    elif wp is not None and wp == wn:
                        wkey, gap = wp, pn
                    else:
                        cands = []
                        if wp is not None and pp == len(words_of[wp][1]) - 1:
                            cands.append((wp, len(words_of[wp][1])))
                        if wn is not None and pn == 0:
                            cands.append((wn, 0))
                        if not cands:
                            reason = "omission_no_word"
                        else:
                            pick = None
                            for wk, g in cands:
                                vals = word_values(wk, None, g)
                                if vals and "V" in vals:
                                    pick = (wk, g)
                                    break
                            wkey, gap = pick or cands[0]
            if reason:
                att[f"2_excl_{reason}"] += 1
                continue
            vals = word_values(wkey, pos, gap)
            word, pron = words_of[wkey]
            if vals is None:
                att["2_excl_word_not_in_dictionary"] += 1
                continue
            hyp_val = value(pron[pos]) if pos is not None else "A"
            if hyp_val is None:
                att["2_excl_position_not_vowel"] += 1
                continue
            status, ctype = competition(vals)
            att[f"3_{status}_{ti['ref_cat']}"] += 1
            rows_out.append([
                args.system, fid, ti["speaker"], ti["style"], split.get(fid, ""), uid, word_hash(word),
                ti["ref_cat"], ti["ref_class"], ti["env"], "devoiced" if hyp_val in ("D", "A") else "voiced",
                hyp_val, status, ctype, "".join(sorted(vals)), et, onset, offset,
                pos if pos is not None else "", gap if gap is not None else "",
            ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(["system", "file_id", "speaker", "style", "split", "ref_unit_id", "word_id", "ref_label",
                    "ref_class", "env", "hyp_label", "hyp_value", "status", "ctype", "values", "error_type",
                    "onset_error_ms", "offset_error_ms", "pos", "gap"])
        w.writerows(rows_out)
    args.attrition.parent.mkdir(parents=True, exist_ok=True)
    with args.attrition.open("w", encoding="utf-8") as fo:
        fo.write("system\tstep\tn\n")
        for k in sorted(att):
            fo.write(f"{args.system}\t{k}\t{att[k]}\n")
        if mismatch:
            fo.write(f"{args.system}\tcheck_unit_reproduction_mismatch\t{mismatch[0]}\n")
            fo.write(f"{args.system}\tcheck_unit_reproduction_total\t{mismatch[1]}\n")
        fo.write(f"{args.system}\tn_talks\t{len(talks)}\n{args.system}\tn_textgrids\t{n_tg}\n")
    print(f"talks {len(talks)}  textgrids {n_tg}  unit reproduction mismatch {mismatch}")
    for k in sorted(att):
        print(f"  {k:<45} {att[k]}")
    print(f"tokens written: {len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
