# SCOPE: journal-only
"""
RQ5: 音素クラスを段階的に手修正した場合のコスト対効果曲線。
査読で新設が提案されたRQ。ICPhS版では使用しない。
"""


def simulate_partial_correction(error_records, correction_order):
    """correction_order: 手修正する音素クラスを追加していく順序のリスト。
    各段階での推定作業時間と許容内率の変化を返す。
    """
    raise NotImplementedError
