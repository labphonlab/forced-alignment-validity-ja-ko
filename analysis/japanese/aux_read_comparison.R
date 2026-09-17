# SCOPE: icphs
# 補助分析（探索的, n=6）: 両スタイル+再朗読を揃える6話者に限定し、
# 話者内で APS/SPS（自発・独話）と R（再朗読・読み上げ）の
# MFA境界精度・母音長を比較する。
#
# ⚠ n=6 話者・各1講演の探索的分析。主分析（177講演 APS/SPS）とは別枠。
#   統計的検出力は限定的。RQ1/RQ3 の補助節として位置づける（docs/decisions-log.md 2026-07-24）。
#
# 位置づけの意義: CSJの再朗読は「話者自身の自発発話の転記の読み上げ」であり、
#   同一話者・テキスト統制された read-vs-spontaneous 対を与える。現行の主分析が
#   APS/SPS対比を自発性の代理としているのに対し、これは真の読み上げ条件を含む。
#
# 入力: results/csj_alignment.tsv（全183講演の突合）, results/csj_units.tsv,
#       results/rq3_vowel_table_all.tsv（後述: READ含む母音テーブル）

library(data.table)
library(ggplot2)

# 対象6話者
SPK <- c("19","423","463","471","514","685")

# --- 境界精度: 全183講演突合から6話者分を抽出し、スタイル別に集計 ---
al <- fread("results/csj_alignment.tsv")
un <- fread("results/csj_units.tsv", select = c("unit_id","speaker_id","style"))
setkey(un, unit_id)
al <- merge(al, un, by.x = "ref_unit_id", by.y = "unit_id", all.x = TRUE)
sub <- al[speaker_id %in% SPK & error_type == "displacement" &
          !is.na(onset_error_ms)]

# onset/offset を縦持ちに
long <- rbind(
  sub[, .(speaker_id, style, phone_class, err = abs(onset_error_ms), bt = "onset")],
  sub[, .(speaker_id, style, phone_class, err = abs(offset_error_ms), bt = "offset")]
)
long <- long[!is.na(err)]
long[, style := factor(style, levels = c("aps","sps","read"),
                       labels = c("APS (spont.)","SPS (spont.)","READ"))]

acc <- long[, .(
  n = .N,
  median = round(median(err),1),
  mean   = round(mean(err),1),
  within10 = round(100*mean(err<=10),1),
  within25 = round(100*mean(err<=25),1)
), by = style][order(style)]
fwrite(acc, "results/aux_read_accuracy.tsv", sep = "\t")
cat("== 補助分析: スタイル別 境界精度（6話者・displacement）==\n")
print(acc)

# 話者内対比（各話者でスタイル別中央値）
per_spk <- long[, .(median_err = round(median(err),1)), by = .(speaker_id, style)]
per_spk_wide <- dcast(per_spk, speaker_id ~ style, value.var = "median_err")
fwrite(per_spk_wide, "results/aux_read_per_speaker.tsv", sep = "\t")
cat("\n== 話者内 中央境界誤差(ms): APS/SPS/READ ==\n")
print(per_spk_wide)

# --- 母音長誤差: READ含む母音テーブルから6話者分 ---
if (file.exists("results/rq3_vowel_table_all.tsv")) {
  vt <- fread("results/rq3_vowel_table_all.tsv")
  vt <- vt[speaker_id %in% SPK]
  vt[, style := factor(style, levels = c("aps","sps","read"),
                       labels = c("APS (spont.)","SPS (spont.)","READ"))]
  vd <- vt[, .(
    n = .N,
    hand_median = round(median(dur_hand_ms),1),
    mfa_median  = round(median(dur_mfa_ms),1),
    diff_median = round(median(dur_diff_ms),1)
  ), by = style][order(style)]
  fwrite(vd, "results/aux_read_vowel_duration.tsv", sep = "\t")
  cat("\n== 補助分析: スタイル別 母音長 hand/MFA（6話者）==\n")
  print(vd)
}

# --- 図: スタイル別 境界誤差分布（6話者）---
p <- ggplot(long[err <= 60], aes(x = style, y = err, fill = style)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, linewidth = 0.35, colour = "grey20") +
  scale_fill_manual(values = c("#0072B2","#56B4E9","#D55E00"), guide = "none") +
  labs(x = NULL, y = "Absolute boundary error (ms)",
       title = "Auxiliary: boundary error by speaking style (6 speakers, exploratory)") +
  theme_minimal(base_size = 11)
ggsave("results/aux_read_boundary_error.png", p, width = 6, height = 4, dpi = 150)

cat("\n出力: aux_read_accuracy.tsv, aux_read_per_speaker.tsv,\n")
cat("      aux_read_vowel_duration.tsv, aux_read_boundary_error.png\n")
