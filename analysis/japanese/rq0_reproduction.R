# SCOPE: shared
# RQ0: McAuliffe & Sonderegger (2026) のCSJ集計値（境界誤差・許容内率）を
# 独立実装で再現する。これがパイプライン全体の受け入れ基準になる。
# 通過条件: results/ 以下の誤差テーブルから算出した中央値・10/20/25/50ms許容内率が
# 先行研究の報告値と比較可能な形で出力されること。

# --- 依存パッケージ（renv.lock 固定対象）---
library(data.table)  # TSV中間表現の読み込み（fread）。arrow等の追加依存を避けるためTSVを採用
library(ggplot2)
