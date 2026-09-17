# SCOPE: shared
"""MFA実行ラッパー。コーパス準備・アラインメント・再現情報の記録を行う。

再現性について
--------------
CLAUDE.md §2 に従い、実行のたびに使用したMFAバージョンとモデル・辞書のSHA-256を
出力ディレクトリの `run_metadata.json` に残す。`docs/versions.md` に記録された値と
食い違った場合は警告する（モデルを更新したのに versions.md を直し忘れる事故を防ぐ）。

MFAはconda環境 `mfa` に導入されている。本スクリプトはその環境内で実行すること。

    conda activate mfa
    python3 src/align/run_mfa.py --help

使い方
------
    # JSUT: wavとtranscript_utf8.txtからコーパスを組んでアラインメント
    python3 src/align/run_mfa.py \\
        --wav-dir data/jsut/jsut_ver1.1/basic5000/wav \\
        --transcript data/jsut/jsut_ver1.1/basic5000/transcript_utf8.txt \\
        --out-dir results/mfa_jsut \\
        --limit 50
"""

from __future__ import annotations

import argparse
import os
import hashlib
import re
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# docs/versions.md に記録済みの値。ここと実物が食い違ったら警告する。
EXPECTED = {
    "acoustic_sha256": "85928ffb1024486872a677a92e1fa94d2f997462d0b84c73a58aa9bb0e35179a",
    "dictionary_sha256": "4a0c66760576e4b7f3748f3169e7d5217c0135f857f0b5d34637c93a85fd1c91",
    "mfa_version": "3.4.1",
}

MFA_HOME = Path(os.environ.get("MFA_MODEL_DIR", str(Path.home() / "Documents" / "MFA" / "pretrained_models"))).expanduser()
ACOUSTIC = MFA_HOME / "acoustic" / "japanese_mfa.zip"
DICTIONARY = MFA_HOME / "dictionary" / "japanese_mfa.dict"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mfa_version() -> str:
    return subprocess.run(
        ["mfa", "version"], capture_output=True, text=True, check=True
    ).stdout.strip()


def load_transcripts(path: Path) -> dict[str, str]:
    """JSUTの transcript_utf8.txt を読む。1行が `ID:本文` の形。

    ⚠ 生の転記には算用数字が含まれ、MFAはこれを読みに展開できず `spn` に潰す。
    精度評価には使わないこと。`load_kana()` を使う（docs/decisions-log.md A節）。
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        utt_id, text = line.split(":", 1)
        out[utt_id.strip()] = text.strip()
    return out


_DIGITS = "〇一二三四五六七八九"
_SMALL_UNITS = ["", "十", "百", "千"]
_BIG_UNITS = ["", "万", "億", "兆"]


def _int_to_kanji(n: int) -> str:
    """算用数字を漢数字に直す。MFA辞書は標準表記でキーされているため。

    `1473` → `千四百七十三`。十・百・千の前の「一」は落とす（日本語の通常表記）。
    """
    if n == 0:
        return "〇"
    groups: list[int] = []
    while n > 0:
        groups.append(n % 10000)
        n //= 10000
    out = []
    for gi in range(len(groups) - 1, -1, -1):
        g = groups[gi]
        if g == 0:
            continue
        s = ""
        for pos in range(3, -1, -1):
            d = (g // (10 ** pos)) % 10
            if d == 0:
                continue
            # 十/百/千の前の「一」は書かない
            s += ("" if d == 1 and pos > 0 else _DIGITS[d]) + _SMALL_UNITS[pos]
        out.append(s + _BIG_UNITS[gi])
    return "".join(out)


_NUM_RE = re.compile(r"[0-9０-９]+")


def expand_numerals(text: str) -> str:
    """テキスト中の算用数字（半角・全角）を漢数字に置き換える。

    MFAは算用数字を読みに展開できず、未知語トークン `spn` を1つ出力する。
    参照側の手動アノテーションは読みに展開済みのため、この差がそのまま
    omission として計上される（docs/troubleshooting.md）。

    かな読み（`load_kana`）を入力にする案は**誤りだったので採らない**。
    japanese_mfa辞書は標準表記（漢字・カタカナ）でキーされており、
    全ひらがなを与えると内容語の大半が未知語になる（`しすてぃな` `まれーしあ` 等は辞書に無い）。
    """

    def repl(m: re.Match[str]) -> str:
        digits = m.group(0).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        return _int_to_kanji(int(digits))

    return _NUM_RE.sub(repl, text)


def load_kana(path: Path) -> dict[str, str]:
    """jsut-label の text_kana/*.yaml から `kana_level0`（かな読み）を読む。

    ⚠ **MFAの入力には使わないこと。** japanese_mfa辞書は標準表記でキーされているため、
    かな読みを与えると内容語の大半が未知語になり、`spn` がかえって増える（実測で確認）。
    本関数は参照側の検証・デバッグ用に残してある。

    YAMLの構造は次の形。依存を増やさないため、必要な2行だけを拾う簡易パーサとする。

        BASIC5000_0001:
          text_level0: 水をマレーシアから買わなくてはならないのです。
          kana_level0: みずをまれーしあからかわなくてわならないのです
          ...

    MFAが算用数字を `spn` に潰す問題を回避するため、こちらを入力テキストにする。
    """
    out: dict[str, str] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        if not raw.startswith((" ", "\t")):
            current = raw.split(":", 1)[0].strip()
            continue
        stripped = raw.strip()
        if current and stripped.startswith("kana_level0:"):
            out[current] = stripped.split(":", 1)[1].strip()
    return out


def build_corpus(
    wav_dir: Path, transcripts: dict[str, str], corpus_dir: Path, limit: int,
    utt_ids: set[str] | None = None,
) -> int:
    """MFAが期待する「wavと同名の.labが並ぶディレクトリ」を組む。"""
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    corpus_dir.mkdir(parents=True)

    wavs = sorted(wav_dir.glob("*.wav"))
    if utt_ids is not None:
        wavs = [w for w in wavs if w.stem in utt_ids]
    elif limit:
        wavs = wavs[:limit]

    n = 0
    missing: list[str] = []
    for wav in wavs:
        text = transcripts.get(wav.stem)
        if text is None:
            missing.append(wav.stem)
            continue
        # シンボリックリンクで済ませる（数GBのwavを複製しない）
        (corpus_dir / wav.name).symlink_to(wav.resolve())
        (corpus_dir / f"{wav.stem}.lab").write_text(text + "\n", encoding="utf-8")
        n += 1

    if missing:
        print(f"[warn] 転記が見つからず除外: {len(missing)}件 (例: {missing[:3]})", file=sys.stderr)
    return n


def check_versions() -> dict[str, str]:
    """実物のバージョン・ハッシュを取り、docs/versions.md の記録と照合する。"""
    meta = {
        "mfa_version": mfa_version(),
        "acoustic_sha256": sha256(ACOUSTIC),
        "dictionary_sha256": sha256(DICTIONARY),
        "acoustic_path": str(ACOUSTIC),
        "dictionary_path": str(DICTIONARY),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    for key, expected in EXPECTED.items():
        if meta[key] != expected:
            print(
                f"[warn] {key} が docs/versions.md の記録と異なる。\n"
                f"       記録: {expected}\n"
                f"       実物: {meta[key]}\n"
                f"       → docs/versions.md に変更を追記すること（上書きせず追記）。",
                file=sys.stderr,
            )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--transcript", type=Path,
                    help="JSUT transcript_utf8.txt（生の転記。算用数字がspnに潰れる）")
    ap.add_argument("--kana", type=Path,
                    help="jsut-label text_kana/*.yaml（⚠MFA入力には不適。デバッグ用）")
    ap.add_argument("--no-expand-numerals", action="store_true",
                    help="算用数字→漢数字の展開を行わない（既定は展開する）")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="先頭N発話のみ（0で全件）")
    ap.add_argument("--utt-list", type=Path,
                    help="処理する発話IDを1行1件で並べたファイル（--limit より優先）")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--beam", type=int, default=100)
    ap.add_argument("--retry-beam", type=int, default=400)
    args = ap.parse_args(argv)

    for p in (ACOUSTIC, DICTIONARY):
        if not p.exists():
            print(f"[error] not found: {p}\n  mfa model download で取得すること。", file=sys.stderr)
            return 1

    meta = check_versions()

    corpus_dir = args.out_dir / "corpus"
    align_dir = args.out_dir / "aligned"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.transcript:
        transcripts = load_transcripts(args.transcript)
        text_source = f"transcript:{args.transcript}"
        if not args.no_expand_numerals:
            n_expanded = sum(1 for v in transcripts.values() if _NUM_RE.search(v))
            transcripts = {k: expand_numerals(v) for k, v in transcripts.items()}
            text_source += "+numerals_expanded"
            print(f"算用数字を漢数字に展開: {n_expanded} 発話")
    elif args.kana:
        print("[warn] かな読みをMFA入力にするのは誤り。japanese_mfa辞書は標準表記で"
              "キーされており、内容語の大半が未知語になる（docs/troubleshooting.md）。",
              file=sys.stderr)
        transcripts = load_kana(args.kana)
        text_source = f"kana:{args.kana}"
    else:
        print("[error] --transcript が必要。", file=sys.stderr)
        return 1
    utt_ids = None
    if args.utt_list:
        utt_ids = {ln.strip() for ln in args.utt_list.read_text(encoding='utf-8').splitlines() if ln.strip()}
        print(f'発話IDリスト指定: {len(utt_ids)}件')
    n = build_corpus(args.wav_dir, transcripts, corpus_dir, args.limit, utt_ids)
    print(f"corpus prepared : {n} utterances -> {corpus_dir}")
    if n == 0:
        print("[error] コーパスが空。--wav-dir と --transcript を確認すること。", file=sys.stderr)
        return 1

    cmd = [
        "mfa", "align", "--clean",
        str(corpus_dir), str(DICTIONARY), str(ACOUSTIC), str(align_dir),
        "--num_jobs", str(args.jobs),
        "--beam", str(args.beam),
        "--retry_beam", str(args.retry_beam),
        "--output_format", "long_textgrid",
    ]
    print("running:", " ".join(cmd))
    proc = subprocess.run(cmd)

    meta.update(
        {
            "command": cmd,
            "returncode": proc.returncode,
            "n_utterances": n,
            "wav_dir": str(args.wav_dir),
            "text_source": text_source,
        }
    )
    (args.out_dir / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_out = len(list(align_dir.glob("*.TextGrid"))) if align_dir.exists() else 0
    print(f"\naligned TextGrids : {n_out}")
    print(f"metadata          : {args.out_dir / 'run_metadata.json'}")
    if n_out < n:
        print(f"[warn] {n - n_out} 発話がアラインメントされなかった（beamを上げる必要があるかも）。",
              file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
