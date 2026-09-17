# SCOPE: shared
"""CSJ XML → 中間表現。境界精度評価の参照側を作る。

本モジュールが `mapping/decisions.md` の4決定（長音・促音・撥音・無声化母音）を
実際に適用する場所である。判断そのものは書かず、決定に従うだけにすること。

適用する統合
------------
  長音      `V`|`H`   を1単位に統合（Phone層の等分割を戻す）
  促音      `Q`|`SclS` を1単位に統合。バースト（`SclS` の終端）は `burst_t` に保持
  撥音      `N`|鼻音   を1単位に統合（鼻音融合の場合）
  無声化    融合モーラ（`s`|`U` 等）を1単位に統合。内部境界は真値なし

等分割境界の判定（2026-07-20 確定）
-----------------------------------
    内部境界が等分割由来 ⟺ 不確実フラグが立つ OR いずれかのPhoneが長さ0

感度・特異度ともに100%であることをTextGrid `seg` tier との突合で検証済み
（`mapping/decisions.md`「等分割境界の判定方法」）。TextGridを読む必要はない。

実データ由来の注意
------------------
  - 補助ラベルは `<cl>` ではなく **`SclS`**（山括弧が `S…S` に符号化されている）
  - 長さ0のPhoneが存在する（VOT実質ゼロ。`zero_vot` 属性で保持し除外しない）
  - MFA入力テキストは `SUW/@PlainOrthographicTranscription`（タグ除去済み・
    算用数字とラテン文字を含まない・辞書照合率97.5%）
  - `TransSUW/@TagMaskStart` を持つIPUは音声全体が白色雑音のため**丸ごと除外**

使い方
------
    python3 src/convert/csj_xml_to_units.py data/csj --limit 3 \\
        --out results/csj_units.tsv
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapping_loader import UnmappedLabelError, load_csj_mapping  # noqa: E402
from schema import (  # noqa: E402
    BoundarySource,
    Corpus,
    ExclusionReason,
    InternalBoundary,
    MetricType,
    Style,
    Unit,
    internal_boundary_path,
    validate,
    write_internal_boundaries,
    write_units,
)

ZERO = 1e-9

# ⚠ 実データ由来の相違（2026-07-20 確認）:
#   1. PhoneClass は小文字 'vowel' / 'consonant' / 'others' / 'special'
#   2. **無声化母音の PhoneEntity は小文字のまま**（`i` `u` `a` `o` `e`）で、
#      segment.pdf 表1 の大文字 `I` `U` `A` `E` `O` は XML には現れない。
#      無声化は `Devoiced="1"` 属性のみで表される。
#      対応表 mapping/csj_mfa_map.tsv はマニュアルの表記（大文字）に従っているため、
#      変換器側で正規化して吸収する（表を実装都合で歪めない）。
CLS_VOWEL = "vowel"
CLS_CONSONANT = "consonant"

LONG_VOWEL_SECOND = "H"
CLOSURE = "SclS"
MORAIC_NASAL = "N"
GEMINATE = "Q"
NASALS = {"m", "n", "nj", "ny", "my", "mj"}
DEVOICED_VOWELS = {"A", "I", "U", "E", "O"}
VOICED_VOWELS = {"a", "i", "u", "e", "o"}
# 評価対象外の補助ラベル（閉鎖 SclS は別扱い）
AUX_EXCLUDED = {"SpzS", "SuvS", "SsvS", "SfrS", "SfvS", "S?S", "SNS", "SbS", "#"}


class Ph:
    """Phone要素1つ分。"""

    __slots__ = ("start", "end", "entity", "cls", "devoiced", "su", "eu",
                 "mora_vlong", "mora_clong", "burst_t", "accent_type", "is_nucleus")

    def __init__(self, e: ET.Element, mora_vlong: bool, mora_clong: bool,
                 accent_type: int | None = None, is_nucleus: bool = False) -> None:
        self.start = float(e.get("PhoneStartTime") or 0.0)
        self.end = float(e.get("PhoneEndTime") or 0.0)
        self.entity = e.get("PhoneEntity") or ""
        self.cls = e.get("PhoneClass") or ""
        self.devoiced = e.get("Devoiced") == "1"
        self.su = e.get("StartTimeUncertain") == "1"
        self.eu = e.get("EndTimeUncertain") == "1"
        self.mora_vlong = mora_vlong
        self.mora_clong = mora_clong
        self.burst_t: float | None = None  # 閉鎖を吸収した破裂音のバースト位置
        self.accent_type = accent_type    # 語のアクセント型（0/1/2..、無核=0）
        self.is_nucleus = is_nucleus       # このPhoneのモーラがアクセント核か

    @property
    def is_zero(self) -> bool:
        return abs(self.end - self.start) < ZERO

    @property
    def label(self) -> str:
        """対応表の表記に正規化したラベル。

        XMLは無声化母音を小文字＋`Devoiced` 属性で表すが、対応表は segment.pdf 表1 に
        従って大文字（`I` `U` `A` `E` `O`）で書いてある。ここで吸収する。
        """
        if self.devoiced and self.entity in VOICED_VOWELS:
            return self.entity.upper()
        return self.entity


def boundary_is_derived(a: Ph, b: Ph) -> bool:
    """a と b の間の内部境界が等分割由来か。mapping/decisions.md で確定した判定条件。"""
    return a.eu or b.su or a.is_zero or b.is_zero


def ipu_is_masked(ipu: ET.Element) -> bool:
    """(R) マスクを含むIPUか。音声全体が白色雑音のため丸ごと除外する対象。

    ⚠ マスクはIPU境界をまたぐ（transcription.pdf §5「タグ(R)」）。継続IPUは
    `TagMaskStart` を持たず `TagMaskMidst` / `TagMaskEnd` だけを持つため、Start だけで
    判定すると継続IPUがすり抜け、伏せ字 `×` が入力に漏れる（実測: 全31講演で6 IPU・
    47 SUW が漏れていた）。Start / Midst / End のいずれかで判定する。
    """
    return any(
        t.get("TagMaskStart") == "1"
        or t.get("TagMaskMidst") == "1"
        or t.get("TagMaskEnd") == "1"
        for t in ipu.iter("TransSUW")
    )


def style_of(talk_id: str) -> Style:
    """講演IDの先頭文字から講演種別を決める。

    A=学会講演(APS), S=模擬講演(SPS), R=再朗読(READ、読み上げ)。
    RQ1主分析はAPS/SPSの2水準のみ（READは補助分析用。docs/decisions-log.md 2026-07-24）。
    """
    if talk_id.startswith("A"):
        return Style.APS
    if talk_id.startswith("R"):
        return Style.READ
    return Style.SPS


def collect_phones(ipu: ET.Element) -> list[Ph]:
    """IPU内のPhoneを集める。アクセント型（語）と核（モーラ）も付ける。

    アクセント型は各SUWの XJToBILabelWord/@PerceivedAccPos（短単位終端に付与）から取る。
    核は Mora/@PerceivedAcc==1。PerceivedAccPos が核モーラ位置と一致することは実データで
    100%確認済み（知覚アクセント。CSJコア・東京方言限定。docs/decisions-log.md）。
    """
    out: list[Ph] = []
    for suw in ipu.iter("SUW"):
        # 短単位終端のアクセントラベルを語のアクセント型とする
        acc_labels = [w.get("PerceivedAccPos") for w in suw.iter("XJToBILabelWord")]
        acc_type: int | None = None
        if acc_labels and acc_labels[-1] is not None:
            try:
                acc_type = int(acc_labels[-1])
            except ValueError:
                acc_type = None
        for mora in suw.iter("Mora"):
            vl = mora.get("TagVLong") == "1"
            cl = mora.get("TagCLong") == "1"
            is_nuc = mora.get("PerceivedAcc") == "1"
            for ph in mora.iter("Phone"):
                out.append(Ph(ph, vl, cl, accent_type=acc_type, is_nucleus=is_nuc))
    out.sort(key=lambda p: (p.start, p.end))
    return out


# 破裂音・破擦音: MFAは閉鎖とバーストを分けず1音素で表す（閉鎖専用音素を持たない）。
# CSJは閉鎖を SclS、バースト以降を子音ラベルで別々に持つ。両者の onset を対応させるには、
# **CSJ側の子音の onset を SclS の開始（閉鎖開始）まで遡らせる**必要がある。
# そうしないと CSJ の子音 onset（バースト）と MFA の onset（閉鎖開始）が閉鎖長ぶん
# （実測で平均約40ms）系統的にずれ、破裂音の許容内率が壊滅する（10ms以内 3.2%）。
# バースト時刻は burst_t に保持し RQ3 のVOT分析で使う（docs/decisions-log.md「RQ3拡張の再定義」）。
STOP_AFFRICATE = {
    "k", "g", "t", "d", "b", "p", "kj", "gj", "ky", "gy", "ty", "dy", "py", "by", "gw", "kw",
    "c", "cj", "cy", "z",  # z は破擦実現時。摩擦実現でも閉鎖は付かないので影響なし
}


def absorb_closures(phones: list[Ph]) -> list[Ph]:
    """単独の SclS を直後の破裂音・破擦音に吸収する（Qを伴う促音の SclS は触らない）。

    子音側の start を閉鎖開始まで遡らせ、burst 時刻（SclS の終端）を子音に記録する。
    """
    out: list[Ph] = []
    i = 0
    while i < len(phones):
        p = phones[i]
        prev = phones[i - 1] if i > 0 else None
        is_geminate_closure = prev is not None and prev.entity == GEMINATE
        if (p.entity == CLOSURE and not is_geminate_closure
                and i + 1 < len(phones) and phones[i + 1].entity in STOP_AFFRICATE):
            nxt = phones[i + 1]
            nxt.burst_t = p.end        # 閉鎖終端＝バースト位置
            nxt.start = p.start        # onset を閉鎖開始まで遡らせる
            out.append(nxt)
            i += 2
            continue
        out.append(p)
        i += 1
    return out


def group_phones(phones: list[Ph]) -> list[list[Ph]]:
    """4決定に従って Phone を統合単位にまとめる。"""
    groups: list[list[Ph]] = []
    i = 0
    while i < len(phones):
        cur = phones[i]
        grp = [cur]

        # 後続を貪欲に取り込む。取り込み条件は決定ごとに異なる。
        while i + len(grp) < len(phones):
            nxt = phones[i + len(grp)]
            prev = grp[-1]
            merge = False

            # 長音: V + H
            if nxt.entity == LONG_VOWEL_SECOND and prev.cls == CLS_VOWEL:
                merge = True
            # 促音: Q + SclS + 子音
            elif prev.entity == GEMINATE and nxt.entity == CLOSURE:
                merge = True
            elif prev.entity == CLOSURE and grp[0].entity == GEMINATE:
                merge = True
            # 撥音 + 鼻音
            elif prev.entity == MORAIC_NASAL and nxt.entity in NASALS:
                merge = True
            # 無声化融合モーラ: 子音 + 無声化母音（内部境界に真値がない）
            elif nxt.devoiced and nxt.cls == CLS_VOWEL and prev.cls == CLS_CONSONANT:
                merge = True

            if not merge:
                break
            grp.append(nxt)

        groups.append(grp)
        i += len(grp)
    return groups


def convert_file(path: Path, mapping, stats: Counter) -> tuple[list[Unit], list[InternalBoundary]]:
    root = ET.parse(path).getroot()
    talk_id = root.get("TalkID") or path.stem
    speaker_id = root.get("SpeakerID") or "unknown"
    style = style_of(talk_id)

    units: list[Unit] = []
    bounds: list[InternalBoundary] = []

    for ipu in root.iter("IPU"):
        # (R) を含むIPUは音声全体が白色雑音に置換されている → 丸ごと除外
        if ipu_is_masked(ipu):
            stats["ipu_masked_excluded"] += 1
            continue
        stats["ipu"] += 1

        ipu_id = ipu.get("IPUID")
        ipu_start = float(ipu.get("IPUStartTime") or 0.0)
        ipu_end = float(ipu.get("IPUEndTime") or 0.0)

        # MFA入力に使う語形（タグ除去済み）。単語文脈として各Phoneに付ける
        words = [
            (s.get("PlainOrthographicTranscription") or "").strip()
            for s in ipu.iter("SUW")
        ]
        word_str = "".join(w for w in words if w) or None

        phones = absorb_closures(collect_phones(ipu))
        groups = group_phones(phones)
        ipu_units: list[Unit] = []
        for gi, grp in enumerate(groups):
            head = grp[0]
            unit_id = f"{talk_id}_{ipu_id}_{gi:04d}"
            label_source = "+".join(p.label for p in grp)

            excluded = None
            if head.entity in AUX_EXCLUDED:
                excluded = ExclusionReason.UNMAPPED_LABEL
            elif head.entity == CLOSURE and len(grp) == 1:
                # 促音に取り込まれなかった単独の閉鎖ラベル（通常の破裂音の閉鎖区間）。
                # **MFA側に対応物が存在しない**（japanese_mfa は閉鎖専用音素を持たない）ため
                # 4類型分類の対象外とする。mapping/decisions.md「促音の閉鎖区間の帰属」。
                # 除外しないと閉鎖のたびに偽の omission が立つ（実測で omission の65%を占めた）。
                # バースト時刻は burst_t に保持され RQ3 のVOT分析で使う。
                excluded = ExclusionReason.UNMAPPED_LABEL
                stats["closure_excluded"] += 1

            # 写像は mapping/csj_mfa_map.tsv のみを参照（CLAUDE.md §3）
            label_canonical = label_source
            phone_class = "unknown"
            try:
                entries = mapping.entries_for_csj(label_source)
                label_canonical = entries[0].mfa_phone
                phone_class = entries[0].category
            except UnmappedLabelError:
                stats[f"unmapped::{label_source}"] += 1
                if excluded is None:
                    excluded = ExclusionReason.UNMAPPED_LABEL

            # バースト時刻: 促音は SclS の終端、単独破裂音は absorb_closures が記録した burst_t
            burst_t = next((p.end for p in grp if p.entity == CLOSURE), None)
            if burst_t is None:
                burst_t = next((p.burst_t for p in grp if p.burst_t is not None), None)
            zero_vot = any(p.is_zero for p in grp if p.entity == CLOSURE)
            # 閉鎖以外で長さ0の単位は測れる区間を持たないため除外する
            # （実測8件: Q 5 / k g b 各1。§5.2 と同じ「同一時刻への融合」現象）
            if abs(grp[-1].end - head.start) < ZERO and not zero_vot and excluded is None:
                excluded = ExclusionReason.ZERO_DURATION
                stats['zero_duration_excluded'] += 1

            # 無声化融合は内部境界に真値がない（segment.pdf §5.1）→ undeterminable
            has_devoiced = any(p.devoiced for p in grp)
            inner_kind = (
                BoundarySource.UNDETERMINABLE if has_devoiced else BoundarySource.DERIVED
            )

            ipu_units.append(
                Unit(
                    file_id=talk_id,
                    speaker_id=speaker_id,
                    corpus=Corpus.CSJ,
                    style=style,
                    metric_type=MetricType.ACCURACY,  # CSJのみ精度を論じられる
                    ipu_id=ipu_id,
                    ipu_start=ipu_start,
                    ipu_end=ipu_end,
                    unit_id=unit_id,
                    t_start=head.start,
                    t_end=grp[-1].end,
                    label_canonical=label_canonical,
                    label_source=label_source,
                    phone_class=phone_class,
                    boundary_source_start=BoundarySource.MEASURED,
                    boundary_source_end=BoundarySource.MEASURED,
                    merged_from=len(grp),
                    burst_t=burst_t,
                    devoiced=has_devoiced,
                    tag_vlong=any(p.mora_vlong for p in grp),
                    tag_clong=any(p.mora_clong for p in grp),
                    zero_vot=zero_vot,
                    word=word_str,
                    # アクセント: 統合単位は元モーラ由来のアクセント型を引き継ぐ。
                    # 核は統合単位内のどれかのPhoneが核モーラなら真（長音 V+H は同一モーラ）。
                    accent_type=next((p.accent_type for p in grp if p.accent_type is not None), None),
                    is_accent_nucleus=any(p.is_nucleus for p in grp),
                    excluded_reason=excluded,
                )
            )
            stats["units"] += 1
            if len(grp) > 1:
                stats["merged"] += 1

            for k in range(len(grp) - 1):
                src = inner_kind
                if not boundary_is_derived(grp[k], grp[k + 1]):
                    # 判定条件を満たさない内部境界＝実測。撥音の分離ラベリング等
                    src = BoundarySource.MEASURED
                    stats["internal_measured"] += 1
                bounds.append(
                    InternalBoundary(
                        file_id=talk_id, unit_id=unit_id, t=grp[k].end,
                        source=src, position=k, note=f"merged {label_source}",
                    )
                )

        # --- following_context を後付けする ---
        # 「次に何が来るか」を音素クラスで表す。RQ1では撥音の異音を予測する因子として、
        # RQ3では撥音に先行する母音の層別として使う（mapping/decisions.md「撥音の異音統合」）。
        # 撥音自身が後続鼻音と融合している場合は unit 側の phone_class が
        # moraic_nasal_fused になっているので、ここでは後続のクラスだけを見ればよい。
        for k, u in enumerate(ipu_units):
            u.following_context = (
                ipu_units[k + 1].phone_class if k + 1 < len(ipu_units) else "final"
            )
        units.extend(ipu_units)

    return units, bounds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    xmls = sorted(args.corpus_dir.glob("*.xml"))
    if args.limit:
        xmls = xmls[: args.limit]
    if not xmls:
        print(f"[error] no XML under {args.corpus_dir}", file=sys.stderr)
        return 1

    mapping = load_csj_mapping()
    stats: Counter = Counter()
    all_units: list[Unit] = []
    all_bounds: list[InternalBoundary] = []

    for x in xmls:
        u, b = convert_file(x, mapping, stats)
        all_units.extend(u)
        all_bounds.extend(b)
        stats["files"] += 1

    problems = validate(all_units)
    if problems:
        print(f"[error] スキーマ検証で {len(problems)} 件の問題:", file=sys.stderr)
        for p in problems[:10]:
            print(f"  {p}", file=sys.stderr)
        return 1

    write_units(all_units, args.out)
    write_internal_boundaries(all_bounds, internal_boundary_path(args.out))

    print(f"files              : {stats['files']}")
    print(f"IPU                : {stats['ipu']}  (マスク除外 {stats['ipu_masked_excluded']})")
    print(f"units              : {stats['units']}")
    print(f"  merged (>1)      : {stats['merged']}")
    print(f"  excluded         : {sum(1 for u in all_units if u.excluded_reason)}")
    print(f"  analyzable       : {sum(1 for u in all_units if u.is_analyzable)}")
    print(f"  zero_vot         : {sum(1 for u in all_units if u.zero_vot)}")
    print(f"internal bounds    : {len(all_bounds)}")
    print(f"  derived          : {sum(1 for b in all_bounds if b.source is BoundarySource.DERIVED)}")
    print(f"  undeterminable   : {sum(1 for b in all_bounds if b.source is BoundarySource.UNDETERMINABLE)}")
    print(f"  measured(実測)   : {stats['internal_measured']}")

    nasal_ctx = Counter(
        u.following_context for u in all_units
        if u.phone_class in ("moraic_nasal", "moraic_nasal_fused")
    )
    if nasal_ctx:
        print()
        print("撥音の後続環境（RQ1因子・RQ3層別）:")
        for k, v in nasal_ctx.most_common(8):
            print(f"  {k:<22} {v:>7}")

    unmapped = {k: v for k, v in stats.items() if k.startswith("unmapped::")}
    if unmapped:
        print(f"\n[warn] 未写像ラベル {len(unmapped)} 種 / 延べ {sum(unmapped.values())}:")
        for k, v in sorted(unmapped.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {k.split('::', 1)[1]!r:<24} {v:>7}")
        print("  → mapping/csj_mfa_map.tsv に追記すること。")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
