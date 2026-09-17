# SCOPE: shared
"""CSJ XML + wav → MFAが受け付けるコーパスを組む。

なぜTextGridを作るのか
----------------------
CSJのwavは講演1本まるごと（最長796秒）である。MFAに長時間ファイルをそのまま渡すと
アラインメントが破綻するため、**発話区間を区切ったTextGridを添える**（MFAは
wavと同名のTextGridがあれば、その区間ごとにアラインメントする）。

区切りにはCSJのIPU（Inter-Pausal Unit、原則200ms以上のポーズで区切られた単位）を
そのまま使う。RQ2の休止閾値としても同じ定義を採用しており（docs/decisions-log.md）、
評価単位と入力単位が一致するので突合が単純になる。

入力テキスト
------------
`SUW/@PlainOrthographicTranscription`（タグ除去済みの出現形）を連結する。
発音形（カタカナ）は使わない — japanese_mfa辞書は標準表記でキーされており、
発音形の辞書照合率は27.1%しかない（`PlainOrthographicTranscription` は97.5%）。
経緯は docs/decisions-log.md「CSJのMFA入力は基本形を使う」。

除外
----
`TransSUW/@TagMaskStart` を持つIPUは、音声全体が白色雑音に置換されているため
区間ごと落とす（transcription.pdf §5「タグ(R)」）。

使い方
------
    python3 src/align/prepare_csj_corpus.py data/csj results/mfa_csj/corpus --limit 3
"""

from __future__ import annotations

import argparse
import sys
import wave
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))

from csj_xml_to_units import ipu_is_masked  # noqa: E402
from paths import assert_writable  # noqa: E402


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def ipu_segments(xml_path: Path) -> list[tuple[float, float, str]]:
    """(開始, 終了, テキスト) を返す。マスク付きIPUと空テキストは除く。

    マスク判定は csj_xml_to_units.ipu_is_masked と同じ（Start/Midst/End のいずれか）。
    Start だけだとIPU境界をまたぐマスクの継続IPUがすり抜け、伏せ字 × が入力に漏れる。
    """
    out: list[tuple[float, float, str]] = []
    for ipu in ET.parse(xml_path).iter("IPU"):
        if ipu_is_masked(ipu):
            continue
        text = "".join(
            (s.get("PlainOrthographicTranscription") or "").strip()
            for s in ipu.iter("SUW")
        ).strip()
        if not text:
            continue
        s = float(ipu.get("IPUStartTime") or 0.0)
        e = float(ipu.get("IPUEndTime") or 0.0)
        if e > s:
            out.append((s, e, text))
    return out


def write_textgrid(path: Path, segments: list[tuple[float, float, str]], total: float) -> None:
    """発話区間1層だけのTextGridを書く。区間の隙間は空文字で埋める。"""
    assert_writable(path)
    intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for s, e, text in segments:
        s, e = max(s, cursor), min(e, total)
        if e <= s:
            continue
        if s > cursor:
            intervals.append((cursor, s, ""))
        intervals.append((s, e, text.replace('"', '""')))
        cursor = e
    if cursor < total:
        intervals.append((cursor, total, ""))

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {total}",
        "tiers? <exists>",
        "size = 1",
        "item []:",
        "    item [1]:",
        '        class = "IntervalTier"',
        '        name = "utt"',
        "        xmin = 0",
        f"        xmax = {total}",
        f"        intervals: size = {len(intervals)}",
    ]
    for i, (s, e, text) in enumerate(intervals, 1):
        lines += [
            f"        intervals [{i}]:",
            f"            xmin = {s}",
            f"            xmax = {e}",
            f'            text = "{text}"',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus_dir", type=Path, help="CSJのXML/wavがあるディレクトリ")
    ap.add_argument("out_dir", type=Path, help="組み立て先（MFAに渡すディレクトリ）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    assert_writable(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xmls = sorted(args.corpus_dir.glob("*.xml"))
    if args.limit:
        xmls = xmls[: args.limit]

    n_talks = n_seg = 0
    total_dur = 0.0
    for x in xmls:
        wav = x.with_suffix(".wav")
        if not wav.exists():
            print(f"[warn] wav がない: {x.stem}", file=sys.stderr)
            continue
        segs = ipu_segments(x)
        if not segs:
            print(f"[warn] 有効な発話区間がない: {x.stem}", file=sys.stderr)
            continue
        dur = wav_duration(wav)
        link = args.out_dir / wav.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(wav.resolve())          # 原本は複製しない（920MB）
        write_textgrid(args.out_dir / f"{x.stem}.TextGrid", segs, dur)
        n_talks += 1
        n_seg += len(segs)
        total_dur += sum(e - s for s, e, _ in segs)

    print(f"talks              : {n_talks}")
    print(f"utterance segments : {n_seg}")
    print(f"speech duration    : {total_dur/3600:.2f} 時間")
    print(f"corpus             : {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
