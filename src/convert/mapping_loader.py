# SCOPE: shared
"""`mapping/*.tsv` を唯一の参照元として写像を適用する。

CLAUDE.md §3 の「音素対応の判断はコード内で個別に行わない」を実装で強制する。
変換スクリプトは音素対応を自前で判断せず、必ずこのモジュールを経由する。

**未知のラベルは黙って通さず例外を投げる。** 写像漏れが誤差として静かに計上されると、
規約差なのかMFAの誤りなのかが後から区別できなくなるため。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MAPPING_DIR = Path(__file__).resolve().parents[2] / "mapping"


class UnmappedLabelError(KeyError):
    """写像表に存在しないラベルが来たときに送出される。"""


@dataclass(frozen=True)
class MappingEntry:
    csj_label: str | None
    jsut_label: str | None
    mfa_phone: str
    category: str
    notes: str


class PhoneMapping:
    """写像表1枚分。方向ごとの索引を持つ。

    同一の csj_label に複数の mfa_phone が対応する場合がある
    （例: uH ← ɯː / ɨː、N ← ɴ/m/n/ŋ/ɲ/ɰ̃）。逆に mfa_phone から引くと
    csj_label は一意に定まる。この非対称性がそのまま索引の形になっている。
    """

    def __init__(self, entries: list[MappingEntry]) -> None:
        self._entries = entries
        self._by_mfa: dict[str, MappingEntry] = {}
        self._by_csj: dict[str, list[MappingEntry]] = {}
        self._by_jsut: dict[str, list[MappingEntry]] = {}

        for e in entries:
            if e.mfa_phone and e.mfa_phone != "-":
                self._by_mfa[e.mfa_phone] = e
            if e.csj_label and e.csj_label != "-":
                self._by_csj.setdefault(e.csj_label, []).append(e)
            if e.jsut_label and e.jsut_label != "-":
                self._by_jsut.setdefault(e.jsut_label, []).append(e)

    def canonical_from_mfa(self, mfa_phone: str) -> MappingEntry:
        """MFA音素から正規化ラベルを引く。撥音の異音統合はここで効く。"""
        try:
            return self._by_mfa[mfa_phone]
        except KeyError as exc:
            raise UnmappedLabelError(
                f"MFA音素 {mfa_phone!r} が写像表にない。"
                f" mapping/ に追記するか、mapping/decisions.md で除外を決めること。"
            ) from exc

    def entries_for_csj(self, csj_label: str) -> list[MappingEntry]:
        try:
            return self._by_csj[csj_label]
        except KeyError as exc:
            raise UnmappedLabelError(f"CSJラベル {csj_label!r} が写像表にない。") from exc

    def entries_for_jsut(self, jsut_label: str) -> list[MappingEntry]:
        try:
            return self._by_jsut[jsut_label]
        except KeyError as exc:
            raise UnmappedLabelError(f"jsut-labelラベル {jsut_label!r} が写像表にない。") from exc

    def category_of_mfa(self, mfa_phone: str) -> str:
        return self.canonical_from_mfa(mfa_phone).category

    def has_mfa(self, mfa_phone: str) -> bool:
        return mfa_phone in self._by_mfa

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def categories(self) -> set[str]:
        return {e.category for e in self._entries}


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """`#` で始まる行はコメントとして読み飛ばす（表の途中の区切りコメントも含む）。"""
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def load_csj_mapping(path: Path | None = None) -> PhoneMapping:
    path = path or MAPPING_DIR / "csj_mfa_map.tsv"
    entries = [
        MappingEntry(
            csj_label=(r.get("csj_label") or "").strip(),
            jsut_label=(r.get("jsut_label") or "").strip(),
            mfa_phone=(r.get("mfa_phone") or "").strip(),
            category=(r.get("category") or "").strip(),
            notes=(r.get("notes") or "").strip(),
        )
        for r in _read_tsv(path)
    ]
    return PhoneMapping(entries)


def load_jsut_mapping(path: Path | None = None) -> PhoneMapping:
    path = path or MAPPING_DIR / "jsut_mfa_map.tsv"
    entries = [
        MappingEntry(
            csj_label=None,
            jsut_label=(r.get("jsut_label") or "").strip(),
            mfa_phone=(r.get("mfa_phone") or "").strip(),
            category=(r.get("category") or "").strip(),
            notes=(r.get("notes") or "").strip(),
        )
        for r in _read_tsv(path)
    ]
    return PhoneMapping(entries)


if __name__ == "__main__":
    # 写像表の健全性チェック。CIから呼べるようにしてある。
    for name, loader in [("csj", load_csj_mapping), ("jsut", load_jsut_mapping)]:
        m = loader()
        print(f"{name:5} entries={len(m):3}  categories={sorted(m.categories)}")
