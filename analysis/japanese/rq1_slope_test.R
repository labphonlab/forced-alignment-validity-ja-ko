# SCOPE: icphs
# RQ1: 話者ランダムスロープ (1 + style | speaker) の識別可能性テスト（177講演版）。
# 6名の両スタイル話者が入ったことで、話者内スタイル対比の情報が初めて存在する。
# 切片のみ (1|speaker) と スロープ入り (1+style|speaker) を両方フィットし、
# 収束・singular・スロープ分散・モデル比較(anova)を報告する。
# 結果に応じて主モデルの random effects 構造と原稿の記述を更新する。

library(data.table)
library(lme4)

d <- fread("results/rq1_table.tsv")
matched <- d[boundary_type %in% c("onset","offset") & !is.na(abs_error_ms)]
m <- matched[!is.na(local_speech_rate) & phone_class != "" & speaker_id != ""]
m[, log_err := log(abs_error_ms + 5)]
rare <- m[, .N, by = phone_class][N < 200, phone_class]
m[phone_class %in% rare, phone_class := "other_rare"]

# 主分析と同一の subsample（seed=42, CAP=40000）
CAP <- 40000L; set.seed(42)
m <- m[, .SD[if (.N > CAP) sample(.N, CAP) else seq_len(.N)], by = phone_class]
m[, phone_class := relevel(factor(phone_class), ref = "vowel_short")]
m[, style := factor(style)]
m[, boundary_type := factor(boundary_type)]
m[, ref_phone := factor(ref_label_canonical)]

# 両スタイルを持つ話者の確認（スロープ識別の前提）
by_spk <- m[, .(nstyle = uniqueN(style)), by = speaker_id]
cat(sprintf("話者総数 %d, うち両スタイル(APS&SPS)を持つ話者 %d名\n",
            nrow(by_spk), by_spk[nstyle >= 2, .N]))

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

cat("\n=== m0: (1 | speaker) 切片のみ（現行主モデル）===\n")
m0 <- lmer(log_err ~ phone_class * style + local_speech_rate + boundary_type +
             (1 | speaker_id) + (1 | ref_phone), data = m, REML = FALSE, control = ctrl)
cat(sprintf("  収束: %s / singular: %s\n",
            is.null(m0@optinfo$conv$lme4$messages), isSingular(m0)))

cat("\n=== m1: (1 + style | speaker) スロープ入り ===\n")
m1 <- tryCatch(
  lmer(log_err ~ phone_class * style + local_speech_rate + boundary_type +
         (1 + style | speaker_id) + (1 | ref_phone), data = m, REML = FALSE, control = ctrl),
  error = function(e) { cat("  [error]", conditionMessage(e), "\n"); NULL })

if (!is.null(m1)) {
  cat(sprintf("  収束: %s / singular: %s\n",
              is.null(m1@optinfo$conv$lme4$messages), isSingular(m1)))
  cat("\n  -- m1 の話者ランダム効果（スロープ分散・相関）--\n")
  vc <- VarCorr(m1)
  print(vc, comp = c("Variance","Std.Dev."))
  cat("\n  -- m0 vs m1 尤度比検定 --\n")
  print(anova(m0, m1))
  cat("\n=== 判定 ===\n")
  sl_sd <- attr(vc$speaker_id, "stddev")
  slope_sd <- if (length(sl_sd) >= 2) sl_sd[2] else NA
  cat(sprintf("  style スロープ SD = %.5f\n", slope_sd))
  if (isSingular(m1)) {
    cat("  → singular fit。スロープ分散が事実上ゼロ。切片のみ (1|speaker) を維持する。\n")
  } else if (!is.na(slope_sd) && slope_sd < 0.01) {
    cat("  → スロープ分散が極小。切片のみで実質同等。(1|speaker) を維持する。\n")
  } else {
    cat("  → スロープ分散が非自明かつ収束。(1+style|speaker) の採用を検討。原稿の識別不能記述を更新。\n")
  }
}
