# SCOPE: icphs
# ICPhS原稿v2の Table[X]（音素クラス別・境界タイプ別の境界誤差要約）を清書する。
# 入力: results/rq1_summary_by_class.tsv（rq1_error_by_class.R の出力）
# 出力: results/table_rq1_boundary_error.tsv（論文貼り込み用、英語ラベル）
#
# 二重盲検（CLAUDE.md §8）: 著者・所属情報を含めない。ラベルは英語。
# CLAUDE.md §5 の要約統計を裾込みで報告する（中央値・平均・RMSE・MAD・95パーセンタイル）
# ＋許容内率 10/20/25/50 ms（4閾値固定）。
# Figure 1 と同じ主要11クラス（境界数 >= 4000）に揃える。

library(data.table)

d <- fread("results/rq1_summary_by_class.tsv")

# 主要11クラス（Figure 1 と一致）
keep <- d[, .(n_total = sum(n)), by = phone_class][n_total >= 4000, phone_class]
d <- d[phone_class %in% keep]

# 英語ラベル（Figure 1 と統一）
class_labels <- c(
  vowel_short = "short vowel", vowel_long = "long vowel", stop = "stop",
  fricative = "fricative", affricate = "affricate", nasal = "nasal (onset)",
  flap = "flap", approximant = "approximant", moraic_nasal = "moraic nasal /N/",
  moraic_nasal_fused = "moraic nasal (fused)", geminate_stop = "geminate stop",
  devoiced_fused = "devoiced (fused)"
)
d[, phone_class := class_labels[phone_class]]
d[, boundary_type := factor(boundary_type, levels = c("onset", "offset"))]

# 音素クラスの並びは onset の中央値昇順（Figure 1 と同じ視覚順）
ord <- d[boundary_type == "onset"][order(median), phone_class]
d[, phone_class := factor(phone_class, levels = ord)]

out <- d[order(phone_class, boundary_type), .(
  `Phone class`   = phone_class,
  Boundary        = boundary_type,
  N               = n,
  `Median (ms)`   = round(median, 1),
  `Mean (ms)`     = round(mean, 1),
  `RMSE (ms)`     = round(rmse, 1),
  `MAD (ms)`      = round(mad, 1),
  `95th pct (ms)` = round(p95, 1),
  `<=10 ms (%)`   = round(within10, 1),
  `<=20 ms (%)`   = round(within20, 1),
  `<=25 ms (%)`   = round(within25, 1),
  `<=50 ms (%)`   = round(within50, 1)
)]

fwrite(out, "results/table_rq1_boundary_error.tsv", sep = "\t")

# 全体行（参考。全評価境界のプール）
cat("Table[X] written: results/table_rq1_boundary_error.tsv\n")
cat(sprintf("rows: %d (11 classes x 2 boundary types)\n", nrow(out)))
cat(sprintf("total N across shown classes: %d\n", sum(out$N)))
print(out)
