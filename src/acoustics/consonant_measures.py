# SCOPE: journal-only
"""
VOT・閉鎖区間・破裂区間の音響測定。RQ3拡張（ジャーナル版のみ）。
査読指摘: MFA利用者の実利用に閉鎖音研究が多いことを反映して追加。
ICPhS原稿には本モジュールの出力を含めないこと（docs/rq-version-map.md参照）。
"""


def measure_vot(textgrid_path, phone_class_map):
    raise NotImplementedError


def measure_closure_duration(textgrid_path, phone_class_map):
    raise NotImplementedError
