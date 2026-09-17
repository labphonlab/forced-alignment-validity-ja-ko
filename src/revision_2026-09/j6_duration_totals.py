# SCOPE: journal-only（査読対応。docs/decisions-log.md 2026-09-16 J1–J6）
"""J6: 「約38.6時間」が何の合計かを確かめる。
(a) 177講演の音声ファイル長の合計（WAV ヘッダの長さだけを読む。音声の中身は読まない）
(b) IPU 時間の合計（csj_units.tsv の (file_id, ipu_start, ipu_end) の一意な組の長さの和）
合計値だけを表示する。
"""
import csv
import sys
import wave
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
csv.field_size_limit(sys.maxsize)
style = {}
spans = defaultdict(set)
with (ROOT / "results/csj_units.tsv").open(encoding="utf-8") as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if r["style"] in ("aps", "sps"):
            style[r["file_id"]] = r["style"]
            if r["ipu_start"] and r["ipu_end"]:
                spans[r["file_id"]].add((r["ipu_start"], r["ipu_end"]))
audio = defaultdict(float)
missing = 0
for t, s in style.items():
    p = ROOT / "data/csj" / f"{t}.wav"
    try:
        with wave.open(str(p), "rb") as w:
            audio[s] += w.getnframes() / w.getframerate()
    except (FileNotFoundError, wave.Error):
        missing += 1
ipu = defaultdict(float)
for t, v in spans.items():
    ipu[style[t]] += sum(float(e) - float(b) for b, e in v)
print(f"talks={len(style)} wav_unreadable={missing}")
print(f"(a) total audio duration: {sum(audio.values())/3600:.2f} h  (APS {audio['aps']/3600:.2f}, SPS {audio['sps']/3600:.2f})")
print(f"(b) total IPU time:       {sum(ipu.values())/3600:.2f} h  (APS {ipu['aps']/3600:.2f}, SPS {ipu['sps']/3600:.2f})")
with (ROOT / "results/revision_2026-09/j6_duration_totals.tsv").open("w", encoding="utf-8") as fh:
    fh.write("quantity\ttotal_h\taps_h\tsps_h\n")
    fh.write(f"audio_duration\t{sum(audio.values())/3600:.3f}\t{audio['aps']/3600:.3f}\t{audio['sps']/3600:.3f}\n")
    fh.write(f"ipu_time\t{sum(ipu.values())/3600:.3f}\t{ipu['aps']/3600:.3f}\t{ipu['sps']/3600:.3f}\n")
