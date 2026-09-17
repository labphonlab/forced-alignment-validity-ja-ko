# SCOPE: shared
"""保護対象パスへの書き込みを防ぐガード。

なぜ必要か
----------
`data/csj` は **Dropboxの原本（`~/Dropbox/Research/Corpora/csj`）へのシンボリックリンク**
である。ここへ誤って書き込むと、再取得できないCSJの原本が破壊される。
CSJは再配布不可のため、失うと復旧手段がない。

`data/jsut`・`data/jsut-label` も再取得コストが高い（2.7GBのダウンロード）ため保護する。

すべての書き出しは `assert_writable()` を通すこと。`schema.write_units()` /
`write_internal_boundaries()` は既に内部で呼んでいる。新しく書き出し処理を作る場合も
必ず通すこと。

補足: 読み取りは妨げない。CLAUDE.md §1 の「CSJの内容をコンテキストに載せない」は
別の規約であり、本モジュールの責務ではない（そちらは各スクリプトが集計値のみを
標準出力に出すことで担保する）。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 書き込み禁止。相対パスはプロジェクトルート基準。
PROTECTED = (
    PROJECT_ROOT / "data" / "csj",
    PROJECT_ROOT / "data" / "jsut",
    PROJECT_ROOT / "data" / "jsut-label",
)


class ProtectedPathError(RuntimeError):
    """保護対象パスへ書き込もうとしたときに送出される。"""


def _resolved_targets() -> list[Path]:
    """保護対象を、シンボリックリンクを辿った実体パスも含めて列挙する。"""
    out: list[Path] = []
    for p in PROTECTED:
        out.append(p)
        try:
            out.append(p.resolve())
        except OSError:
            pass
    return out


def is_protected(path: Path | str) -> bool:
    """path が保護対象の配下（またはそれ自身）なら True。

    シンボリックリンク経由の書き込みも検出するため、リンクを辿った実体でも判定する。
    """
    p = Path(path)
    candidates = [p]
    try:
        # 未作成のファイルでも親までは解決できる
        candidates.append(p.resolve() if p.exists() else p.parent.resolve() / p.name)
    except OSError:
        pass

    for cand in candidates:
        for target in _resolved_targets():
            if cand == target or target in cand.parents:
                return True
    return False


def assert_writable(path: Path | str) -> Path:
    """書き込み先として安全なら Path を返し、保護対象なら例外を投げる。"""
    p = Path(path)
    if is_protected(p):
        raise ProtectedPathError(
            f"保護対象への書き込みを拒否した: {p}\n"
            f"  data/csj は Dropbox の CSJ 原本へのシンボリックリンクであり、"
            f"CSJは再配布不可のため破壊すると復旧できない。\n"
            f"  出力先は results/ 以下にすること。"
        )
    return p


if __name__ == "__main__":
    cases = [
        ("results/out.tsv", False),
        ("data/samples/x.xml", False),
        ("data/csj/A01F0055.xml", True),
        ("data/csj/sub/new.tsv", True),
        (str(PROJECT_ROOT / "data" / "csj" / "x"), True),
        ("data/jsut/jsut_ver1.1/x", True),
        ("data/jsut-label/labels/x.lab", True),
    ]
    ok = True
    for path, expected in cases:
        got = is_protected(PROJECT_ROOT / path if not path.startswith("/") else path)
        mark = "OK " if got == expected else "NG "
        if got != expected:
            ok = False
        print(f"  {mark}{path:<52} protected={got} (期待 {expected})")
    print("\n全ケース通過" if ok else "\n失敗あり")
