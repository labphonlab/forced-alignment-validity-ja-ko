# SCOPE: shared
"""VV監査の残差を、CSJ評価対象に実際に出現する語彙だけに絞り込む。

背景
----
`classify_vv_entries.py` は japanese_mfa辞書544,160件から `V V` 表記揺れを3,017件
検出し、うち749件が自動分類できずに残った。しかしこの749件は**辞書全体**に対する
残差であり、その多くは CSJ の学会講演・模擬講演には出現しない語である
（例: アパラチア山脈、ウォッチドッグ・タイマ、ジョアンナ）。

人手レビューが本当に必要なのは「残差 ∩ CSJ評価対象語彙」だけである。
本スクリプトはその積集合を取り、レビュー対象件数を実務的な規模まで落とす。

出力について
------------
CLAUDE.md §1 に従い、**標準出力には集計値しか書かない**。
積集合の語リストはファイルにのみ書き出す。書き出される語はいずれも
japanese_mfa辞書（公開リソース）に存在する表記だが、「CSJに出現する」という
情報を含むため、出力先は `results/`（.gitignore でデフォルト除外）とする。

CSJ語彙の取り出し方
-------------------
CSJ XML の SUW要素から表記を集める。既定では `@SUWDictionaryForm`（短単位の代表形）を
使う。MFA辞書の見出しは代表形に近いため一致率が高い。`--field` で
`@OrthographicTranscription`（基本形）や `@PlainOrthographicTranscription`
（タグ無し出現形）に切り替えられる。一致率が低い場合は複数フィールドを試すこと。

使い方
------
    # 先に合成サンプルで動作確認
    python3 src/convert/intersect_vv_with_corpus.py \\
        results/vv_audit_uncertain.tsv data/samples/sample_csj.xml \\
        --out results/vv_review_queue.tsv

    # CSJ本番（内容は読まず、集計値だけが標準出力に出る）
    python3 src/convert/intersect_vv_with_corpus.py \\
        results/vv_audit_uncertain.tsv data/csj/XML \\
        --out results/vv_review_queue.tsv
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_FIELD = "SUWDictionaryForm"


def load_audit(path: Path) -> tuple[list[str], list[list[str]]]:
    """監査TSVを (ヘッダ, 行) で返す。1列目が表記であることを前提とする。"""
    with path.open(encoding="utf-8") as fh:
        lines = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    if not lines:
        raise ValueError(f"empty audit file: {path}")
    return lines[0], lines[1:]


def collect_corpus_vocab(xml_paths: list[Path], field: str) -> set[str]:
    """CSJ XMLのSUW要素から表記の集合を作る。内容は返り値の外に出さない。"""
    vocab: set[str] = set()
    for path in xml_paths:
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            print(f"[warn] parse error, skipped: {path.name} ({exc.code})", file=sys.stderr)
            continue
        for suw in tree.iter("SUW"):
            value = suw.get(field)
            if value:
                vocab.add(value)
    return vocab


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("audit_tsv", type=Path, help="vv_audit_uncertain.tsv 等")
    ap.add_argument("corpus", type=Path, help="CSJ XMLファイル、またはそれを含むディレクトリ")
    ap.add_argument("--field", default=DEFAULT_FIELD, help=f"SUWの表記属性名（既定: {DEFAULT_FIELD}）")
    ap.add_argument("--glob", default="*.xml")
    ap.add_argument("--out", type=Path, help="積集合の出力先TSV")
    args = ap.parse_args(argv)

    header, rows = load_audit(args.audit_tsv)

    paths = sorted(args.corpus.rglob(args.glob)) if args.corpus.is_dir() else [args.corpus]
    if not paths:
        print(f"[error] no XML found under {args.corpus}", file=sys.stderr)
        return 1

    vocab = collect_corpus_vocab(paths, args.field)
    hits = [r for r in rows if r and r[0] in vocab]

    print(f"audit rows              : {len(rows)}")
    print(f"corpus files parsed     : {len(paths)}")
    print(f"corpus vocabulary size  : {len(vocab)}  (field: {args.field})")
    print(f"rows present in corpus  : {len(hits)}")
    if rows:
        print(f"reduction               : {len(rows)} -> {len(hits)}"
              f"  ({100.0 * len(hits) / len(rows):.1f}% remain)")
    if not hits:
        print()
        print("[note] 積集合が0件。--field を OrthographicTranscription 等に変えて再試行するか、")
        print("       入力コーパスが想定と異なっていないか確認すること。")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for r in hits:
                fh.write("\t".join(r) + "\n")
        print(f"\nwrote {len(hits)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
