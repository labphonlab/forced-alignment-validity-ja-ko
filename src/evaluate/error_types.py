# SCOPE: shared
"""
誤差4類型の定義と分類ロジック（RQ1）。

この docstring がプロジェクト全体における誤差類型の「唯一の真実源」。
定義を変更する場合は CLAUDE.md 5節・docs/decisions-log.md に理由を残してから
ここを変更する。

類型定義:
    1. displacement  境界は対応しているが位置がずれている
    2. omission      手作業ラベルの分節がMFA出力に対応物を持たない（例: 無声化母音の消失）
    3. insertion     MFA出力にのみ存在する分節
    4. substitution  境界位置は近いがラベルが異なる

分類は上記4種類に対して排他的に行うこと（一つの誤差事例が複数類型に
またがらないようにする）。

omission の下位分類（2026-07-20 追加）:
    omission は来歴の異なる2現象を含むため、**類型は増やさずに属性で下位分類する**。
    5類型目を作らなかった理由は docs/decisions-log.md A節を参照。

    omission_provenance:
        "lexical"    MFAが選択した発音バリアントに当該分節がそもそも存在しない。
                     japanese_mfa辞書は無声化母音を3通りに符号化しており（無声化音素
                     i̥/ɨ̥/ɯ̥、母音の完全削除、有声のまま）、多くの語で削除形が最高確率の
                     既定形である（例: した = ɕ t a が 0.99、ɕ i t a が 0.01）。
                     forced alignment は選択された音素列の全音素を必ず出力するため、
                     辞書にない母音は決して現れない。辞書カバレッジの問題。
        "alignment"  音素列には当該分節が存在するが、境界が縮退している。
                     音響モデルの問題。

    両者は MFA 利用者にとって意味が異なり（改善の処方箋が別）、RQ5の修正効率曲線でも
    lexical 起因の脱落は「手修正しても直らない」クラスとして現れるはずである。
    なお複数バリアントがある場合 MFA は音響尤度でバリアントを選ぶため、
    lexical omission も純粋な辞書の問題とは言い切れない点は論文で明示すること。

substitution に計上しないもの（2026-07-20 追加）:
    以下はラベル規約の差であって MFA の誤りではないため、写像で吸収し
    substitution に計上しない。詳細は mapping/decisions.md を参照。
    - 撥音の異音: MFA の ɴ/m/n/ŋ/ɲ/ɰ̃ はいずれも CSJ の N に対応する
    - 非高母音の無声化: CSJ の A/E/O に対し MFA には対応する無声化音素が存在しない
    - MFA が有声形バリアントを選択した場合（devoiced_voiced_mfa）

真値が存在しない境界（評価対象外）:
    以下は参照側に真値がないため、いずれの類型にも計上せず評価から除外する。
    - 長音の内部境界（CSJ Phone層の等分割）
    - 促音の Q|<cl> 内部境界（同）、およびバースト境界（MFA側に対応物なし）
    - 無声化融合モーラの内部境界（segment.pdf §5.1 が「時間軸上で区分することは
      できない」と明記。人手ラベラーが決定不可能と宣言したもの）
"""

from dataclasses import dataclass
from enum import Enum


class ErrorType(str, Enum):
    DISPLACEMENT = "displacement"
    OMISSION = "omission"
    INSERTION = "insertion"
    SUBSTITUTION = "substitution"


class OmissionProvenance(str, Enum):
    """omission の来歴。error_type が OMISSION のときのみ意味を持つ。"""

    LEXICAL = "lexical"
    ALIGNMENT = "alignment"


@dataclass
class BoundaryErrorRecord:
    phone_class: str
    corpus: str  # "csj" or "jsut"
    onset_or_offset: str
    error_type: ErrorType
    displacement_ms: float | None  # displacement/substitutionのみ意味を持つ
    omission_provenance: OmissionProvenance | None = None  # omissionのみ意味を持つ
    following_context: str | None = None  # 撥音の後続環境（RQ1の因子・RQ3の層別に使う）


def classify_error(manual_interval, mfa_interval) -> ErrorType:
    """TODO: mapping/csj_mfa_map.tsv, mapping/jsut_mfa_map.tsv を参照して実装する。
    data/csj/ の実データ内容はここで読み込まず、上流の変換スクリプトが
    出力した中間表現（時刻・ラベルのみ）を入力として受け取ること。
    """
    raise NotImplementedError
