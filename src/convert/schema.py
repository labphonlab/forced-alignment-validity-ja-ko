# SCOPE: shared
"""中間表現のスキーマ定義。**このモジュールが変換層の唯一の真実源。**

設計の要点
----------
`mapping/decisions.md` で確定した4つの決定（長音・促音・撥音・無声化母音）は、
すべて「どの境界が信頼でき、どの境界を評価対象にするか」の判断に帰着する。
その判断を個々の変換スクリプトに散らさず、**このスキーマに畳み込む**。
CSJ変換器・JSUT変換器・MFA出力変換器は、コーパスが違っても同じスキーマを出力する。

全体を統制するのは `BoundarySource` の3値である。

    measured        人手が置いた境界。**評価対象**
    derived         等分割の人工物。**評価対象外**
                      - 長音の o|H（segment.pdf §7 手順3）
                      - 促音の Q|<cl>（同 手順1-2）
                      - 撥音の鼻音融合 N|m（同 手順2）※実測の場合もある。decisions.md参照
    undeterminable  人手が「決定不可能」と宣言。**評価対象外**
                      - 無声化融合モーラの内部（segment.pdf §5.1
                        「時間軸上で区分することはできない」）
    none            対応する境界が存在しない（発話端など）

`derived` と `undeterminable` を分けるのは、除外理由として意味が違うためである。
前者は「1単位を便宜的に割った」、後者は「境界が音声学的に存在しない」。
論文のLimitationsで別々に説明する必要があり、件数も別に報告する。

**`BoundarySource` は「参照がその境界を主張しているか、それを評価に使うか」を表す。
参照が人手か自動かは `boundary_source` ではなく `metric_type` が担う。**
JSUTはすべての境界がJuliusの自動アラインメントだが、外側境界は
`measured`（＝一致度の算出に使う）、統合で捨てた内部境界は `derived` を与える。
JSUTが金標準でないことは `metric_type=concordance` が一手に引き受けており、
`boundary_source` に二重に持たせない。

物理フォーマット
----------------
TSV 2テーブル構成。第三者が arrow 等の追加依存なしに再現できることを優先する
（JSUTを公開実行例として提供する目的上の要請）。

    <name>.tsv                      1行 = 1評価単位。UNIT_COLUMNS の順で出力
    <name>_internal_boundaries.tsv  統合で捨てた内部境界。unit_id で主テーブルに結合

metric_type について
--------------------
JSUTは境界精度の参照に使えない（時間情報がJuliusの自動アラインメントであるため。
`docs/decisions-log.md` A節）。したがってJSUT経路の出力は精度ではなく**一致度**である。
これを型で担保するため全レコードに `metric_type` を持たせ、集計側は
`require_accuracy()` を通してから精度として扱うこと。一致度を精度として集計しようと
すると例外が送出される。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import assert_writable  # noqa: E402


class BoundarySource(str, Enum):
    """境界の出所。評価対象かどうかを決める。"""

    MEASURED = "measured"
    DERIVED = "derived"
    UNDETERMINABLE = "undeterminable"
    NONE = "none"

    @property
    def is_evaluable(self) -> bool:
        return self is BoundarySource.MEASURED


class MetricType(str, Enum):
    """このレコードから算出できる指標の種別。"""

    ACCURACY = "accuracy"        # 人手校正済み参照との比較（CSJのみ）
    CONCORDANCE = "concordance"  # 自動アライナ同士の一致度（JSUT）


class Corpus(str, Enum):
    CSJ = "csj"
    JSUT = "jsut"


class Style(str, Enum):
    """CSJ内の講演種別。JSUTを評価から外したことに伴うスタイル要因の代替。"""

    APS = "aps"    # 学会講演（自発・独話）
    SPS = "sps"    # 模擬講演（自発・独話）
    READ = "read"  # 再朗読（読み上げ。両スタイル話者6名の自発発話転記の朗読。RQ補助分析用）
    NA = "na"      # JSUT等、該当なし


class ExclusionReason(str, Enum):
    """分析から除外する理由。null（除外しない）は None で表す。"""

    FD_FUSED_GEMINATE = "fd_fused_geminate"    # 転記タグF/D末尾。母音境界が失われる
    DICT_ERROR = "dict_error"                  # MFA辞書の記述誤り（VV監査のexclude）
    DEVOICED_NO_DURATION = "devoiced_no_duration"  # 無声化融合。母音長が測れない
    UNMAPPED_LABEL = "unmapped_label"          # mapping/*.tsv に写像がない
    ZERO_DURATION = "zero_duration"            # 長さ0で測れる区間がない（閉鎖以外）


@dataclass
class Unit:
    """1評価単位。統合後のセグメント1つに対応する。"""

    # --- 同定 ---
    file_id: str
    speaker_id: str
    corpus: Corpus
    style: Style
    metric_type: MetricType

    # --- 発話単位（CSJのIPU。RQ2の休止判定に使う。200ms以上のポーズで区切られる）---
    ipu_id: str | None
    ipu_start: float | None
    ipu_end: float | None

    # --- 単位そのもの ---
    unit_id: str
    t_start: float
    t_end: float
    label_canonical: str   # mapping/*.tsv 経由の正規化ラベル
    label_source: str      # 変換前の原ラベル（監査用。写像の誤りを後から追える）
    phone_class: str       # RQ1の音素クラス

    # --- 境界の出所（評価対象かどうかを決める）---
    boundary_source_start: BoundarySource
    boundary_source_end: BoundarySource

    # --- 統合の痕跡 ---
    merged_from: int = 1           # 統合した原セグメント数。1なら統合なし
    burst_t: float | None = None   # 促音のバースト時刻（RQ3のVOT用）

    # --- 属性 ---
    devoiced: bool = False
    tag_vlong: bool = False        # 非語彙的な母音引き延ばし <H>
    tag_clong: bool = False        # 非語彙的な子音引き延ばし <Q>

    # 閉鎖区間の長さが0だった（＝VOTが実質ゼロ）。segment.pdf §5.2:
    # 「無声破裂音であってもVOTが実際上ゼロになることがある。その場合、<cl>と子音ラベルは
    #   融合ラベルとなって同一の時刻に付与されている」
    # CSJ実データで全Phoneの0.97%（331/33,976、すべて SclS）。**バグではなく仕様。**
    # RQ3のVOT分析では「VOT=0」の実測値として意味を持つため、除外せず属性で保持する。
    zero_vot: bool = False
    following_context: str | None = None  # 撥音の後続環境層（RQ1因子・RQ3層別）

    # --- アクセント（RQ3: 母音長×アクセント型の誤差伝播。CSJコアのみ）---
    # accent_type: 語（短単位）の知覚アクセント型。0=無核, 1=頭高, 2..=N型（核モーラの1始まり位置）。
    #   ⚠ 辞書の規範アクセントではなく X-JToBI ラベラーが実発音から付けた知覚アクセント。
    #   ⚠ CSJコアにのみ付与。話者は東京方言に限定（docs/decisions-log.md, ICPhS Limitations）。
    # is_accent_nucleus: この単位のモーラがアクセント核か（Mora/@PerceivedAcc==1）。
    accent_type: int | None = None
    is_accent_nucleus: bool = False

    # --- 単語文脈 ---
    # 撥音の記号衝突（MFAの m/n/ɲ/mʲ がオンセット子音と同記号）を下流で解消するために要る。
    # sequence_align.py が word + word_phone_index から辞書エントリを辿って撥音か否かを判定する。
    # mapping/decisions.md「撥音の異音統合」問題2を参照。
    word: str | None = None
    # ⚠ 未実装（2026-07-20）。将来の精度改善候補として保留。
    #   CSJ変換器は word をIPU単位でしか埋めておらず、SUW単位の対応付けと
    #   単語内位置の付与を行っていない。撥音の記号衝突（MFAの m/n/ɲ/mʲ が
    #   オンセット子音と同記号）の解消は、現状 sequence_align.py の系列文脈だけで
    #   足りている（jsut側の検証で N vs m の非対称性が正しく解けることを確認済み）。
    #   曖昧性が残るケースが実データで見つかった場合に、MFA単語層から辞書エントリを
    #   辿る判別を追加する。その際に本フィールドが必要になる。
    word_phone_index: int | None = None  # 単語内での音素位置（0始まり）

    excluded_reason: ExclusionReason | None = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def is_analyzable(self) -> bool:
        """両端が実測で、除外理由がない単位のみ誤差算出の対象。"""
        return (
            self.excluded_reason is None
            and self.boundary_source_start.is_evaluable
            and self.boundary_source_end.is_evaluable
        )


@dataclass
class InternalBoundary:
    """統合で捨てた内部境界。感度分析のために保持する。

    主テーブルとは `unit_id` で結合する。1単位が複数の内部境界を持ちうるため
    別テーブルに正規化している（TSVで可変長フィールドを避けるため）。
    """

    file_id: str
    unit_id: str
    t: float
    source: BoundarySource          # derived か undeterminable のいずれか
    position: int = 0               # 単位内での順序（0始まり）
    note: str = ""


UNIT_COLUMNS: list[str] = [f.name for f in fields(Unit)]
INTERNAL_BOUNDARY_COLUMNS: list[str] = [f.name for f in fields(InternalBoundary)]


class MetricTypeError(RuntimeError):
    """一致度データを精度として扱おうとしたときに送出される。"""


def require_accuracy(units: list[Unit]) -> list[Unit]:
    """精度として集計してよいレコードだけを通す。違反があれば例外。

    JSUT由来の一致度データが精度指標に紛れ込む事故を、レビューではなくコードで防ぐ。
    """
    offenders = {u.corpus.value for u in units if u.metric_type is not MetricType.ACCURACY}
    if offenders:
        raise MetricTypeError(
            f"精度として集計できないレコードが含まれている（corpus={sorted(offenders)}）。"
            " JSUT+jsut-labelの時間情報はJuliusの自動アラインメントであり、"
            " MFAとの差は一致度であって精度ではない。docs/decisions-log.md A節を参照。"
        )
    return units


def _serialize(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Enum):
        return value.value
    return str(value)


def write_units(units: list[Unit], path: Path) -> None:
    """主テーブルをTSVで書き出す。保護対象パスへの書き込みは拒否する。"""
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(UNIT_COLUMNS)
        for u in units:
            w.writerow([_serialize(getattr(u, c)) for c in UNIT_COLUMNS])


def write_internal_boundaries(rows: list[InternalBoundary], path: Path) -> None:
    """内部境界テーブルをTSVで書き出す。主テーブルと unit_id で結合する。"""
    path = assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(INTERNAL_BOUNDARY_COLUMNS)
        for r in rows:
            w.writerow([_serialize(getattr(r, c)) for c in INTERNAL_BOUNDARY_COLUMNS])


def internal_boundary_path(unit_path: Path) -> Path:
    """主テーブルのパスから内部境界テーブルのパスを導く（命名規約を一箇所に固定）。"""
    return unit_path.with_name(f"{unit_path.stem}_internal_boundaries.tsv")


def validate(units: list[Unit]) -> list[str]:
    """スキーマ上の不整合を検出して問題の一覧を返す。空リストなら問題なし。"""
    problems: list[str] = []
    seen: set[str] = set()

    for u in units:
        # 長さ0は CSJ の仕様上ありうる（VOT実質ゼロで <cl> と子音が同時刻。segment.pdf §5.2）。
        # zero_vot が立っている場合のみ許容し、それ以外の長さ0・負値は不整合とする。
        if u.t_end < u.t_start:
            problems.append(f"{u.unit_id}: 時刻が逆転 (start={u.t_start}, end={u.t_end})")
        elif u.t_end == u.t_start and not u.zero_vot and u.excluded_reason is None:
            problems.append(
                f"{u.unit_id}: 長さ0だが zero_vot も excluded_reason も無い "
                f"(VOT実質ゼロなら zero_vot=True、測定不能なら excluded_reason を設定すること)"
            )
        if u.unit_id in seen:
            problems.append(f"{u.unit_id}: unit_id が重複")
        seen.add(u.unit_id)
        if u.merged_from < 1:
            problems.append(f"{u.unit_id}: merged_from が 1 未満")
        if u.burst_t is not None and not (u.t_start <= u.burst_t <= u.t_end):
            problems.append(f"{u.unit_id}: burst_t が単位の区間外")
        if u.corpus is Corpus.JSUT and u.metric_type is MetricType.ACCURACY:
            problems.append(f"{u.unit_id}: JSUTに accuracy が設定されている（一致度のみ許容）")
        if u.corpus is Corpus.CSJ and u.style is Style.NA:
            problems.append(f"{u.unit_id}: CSJに style=na が設定されている（aps/spsが必要）")

    return problems
