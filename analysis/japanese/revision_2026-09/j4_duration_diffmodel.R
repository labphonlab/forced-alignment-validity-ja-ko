# SCOPE: journal-only（査読対応の補完モデル。事後。docs/decisions-log.md 2026-09-16 J1–J6）
# J4: トークン内差分 d = log(MFA) − log(hand) の混合モデル、is_nuc 話者スロープ、(1|word) 感度、
#     階層 Bland–Altman、paired model 係数の補足表。
# 使い方: Rscript j4_duration_diffmodel.R <word_side_tsv>
suppressPackageStartupMessages({library(data.table); library(lme4)})
args <- commandArgs(trailingOnly = TRUE)
side_path <- args[1]
OUT <- "results/revision_2026-09"
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5), calc.derivs = FALSE)
t_all <- Sys.time()

d <- fread("results/rq3_vowel_table.tsv")   # rq3_paired.R と同じ入力・同じ変数定義
d[, nasal_following := as.integer(following_context == "moraic_nasal")]
d[, is_nuc := as.integer(is_accent_nucleus)]
d[, style := factor(style)]
d[, vowel := factor(vowel)]
d[, length_class := factor(length_class, levels = c("vowel_short", "vowel_long"))]
d[, dlog := log(dur_mfa_ms) - log(dur_hand_ms)]
side <- fread(side_path)
stopifnot(nrow(side) == nrow(d))
d[, word := factor(side$word_code)]
cat(sprintf("n=%d speakers=%d talks=%d vowels=%d words=%d\n", nrow(d), uniqueN(d$speaker_id),
            uniqueN(d$file_id), nlevels(d$vowel), nlevels(d$word)))

tidy <- function(fit, model) {
  co <- summary(fit)$coefficients
  msgs <- c(fit@optinfo$conv$lme4$messages, fit@optinfo$warnings)
  data.table(model = model, term = rownames(co), est = co[, 1], se = co[, 2],
             ci_lo = co[, 1] - 1.96 * co[, 2], ci_hi = co[, 1] + 1.96 * co[, 2], t = co[, 3],
             converged = if (length(msgs)) paste(unlist(msgs), collapse = " | ") else "ok",
             singular = isSingular(fit))
}
vcs <- function(fit, model) {
  v <- as.data.frame(VarCorr(fit))
  data.table(model = model, grp = v$grp, var1 = v$var1, var2 = v$var2, vcov = v$vcov, sdcor = v$sdcor)
}
timed <- function(expr) { t0 <- Sys.time(); r <- expr; list(fit = r, min = as.numeric(difftime(Sys.time(), t0, units = "mins"))) }

res <- list(); vc <- list(); tm <- list()
# (a) 差分モデル（依頼の式）
f1 <- timed(lmer(dlog ~ is_nuc + nasal_following + style + length_class +
                   (1 | speaker_id) + (1 | file_id) + (1 | vowel), data = d, REML = TRUE, control = ctrl))
res$a <- tidy(f1$fit, "diff_spk_file_vowel"); vc$a <- vcs(f1$fit, "diff_spk_file_vowel"); tm$a <- f1$min
# (b) + (1|word)
f2 <- timed(lmer(dlog ~ is_nuc + nasal_following + style + length_class +
                   (1 | speaker_id) + (1 | file_id) + (1 | vowel) + (1 | word), data = d, REML = TRUE, control = ctrl))
res$b <- tidy(f2$fit, "diff_plus_word"); vc$b <- vcs(f2$fit, "diff_plus_word"); tm$b <- f2$min
# (c) is_nuc の話者ランダムスロープ
f3 <- timed(lmer(dlog ~ is_nuc + nasal_following + style + length_class +
                   (1 + is_nuc | speaker_id) + (1 | file_id) + (1 | vowel), data = d, REML = TRUE, control = ctrl))
res$c <- tidy(f3$fit, "diff_nuc_slope_speaker"); vc$c <- vcs(f3$fit, "diff_nuc_slope_speaker"); tm$c <- f3$min
# 参考: 固定効果だけの OLS（rq3_paired.R の等価性チェックと同じ）
ols <- lm(dlog ~ is_nuc + nasal_following + style + length_class, data = d)
oc <- summary(ols)$coefficients
res$ols <- data.table(model = "ols_no_random", term = rownames(oc), est = oc[, 1], se = oc[, 2],
                      ci_lo = oc[, 1] - 1.96 * oc[, 2], ci_hi = oc[, 1] + 1.96 * oc[, 2], t = oc[, 3],
                      converged = "ok", singular = NA)
# paired model の交互作用（既存の結果）を並べる
pc <- fread("results/rq3_paired_coefficients.tsv")
map <- c("sourcemfa" = "(Intercept)", "sourcemfa:is_nuc" = "is_nuc",
         "sourcemfa:nasal_following" = "nasal_following", "sourcemfa:stylesps" = "stylesps",
         "sourcemfa:length_classvowel_long" = "length_classvowel_long")
pp <- pc[term %in% names(map)]
res$paired <- data.table(model = "paired_model_existing (source x term)", term = map[pp$term],
                         est = pp$est, se = pp$se, ci_lo = pp$ci_lo, ci_hi = pp$ci_hi, t = pp$t,
                         converged = "see rq3_paired.log", singular = NA)
R <- rbindlist(res)
R[, minutes := c(a = tm$a, b = tm$b, c = tm$c)[match(model, c("diff_spk_file_vowel", "diff_plus_word", "diff_nuc_slope_speaker"))]]
fwrite(R, file.path(OUT, "j4_duration_diffmodel.tsv"), sep = "\t")
fwrite(rbindlist(vc), file.path(OUT, "j4_duration_diffmodel_varcomp.tsv"), sep = "\t")
print(R[term != "(Intercept)" | TRUE], digits = 4)
print(rbindlist(vc), digits = 4)

# 階層 Bland–Altman（ms、符号は既存どおり hand − MFA）
d[, diff_ms := dur_hand_ms - dur_mfa_ms]
ba <- timed(lmer(diff_ms ~ 1 + (1 | speaker_id) + (1 | file_id), data = d, REML = TRUE, control = ctrl))
v <- as.data.frame(VarCorr(ba$fit))
vs <- v$vcov[v$grp == "speaker_id"]; vf <- v$vcov[v$grp == "file_id"]; vr <- v$vcov[v$grp == "Residual"]
b0 <- fixef(ba$fit)[1]; seb <- sqrt(vcov(ba$fit)[1, 1])
sd_tot <- sqrt(vs + vf + vr)
naive_sd <- sd(d$diff_ms)
BA <- data.table(
  quantity = c("bias_mixed", "bias_mixed_ci_lo", "bias_mixed_ci_hi", "var_speaker", "var_talk",
               "var_residual", "sd_total", "loa_lo_mixed", "loa_hi_mixed",
               "bias_naive_mean", "sd_naive", "loa_lo_naive", "loa_hi_naive",
               "median_diff_short", "median_diff_long", "n", "minutes"),
  value = c(b0, b0 - 1.96 * seb, b0 + 1.96 * seb, vs, vf, vr, sd_tot,
            b0 - 1.96 * sd_tot, b0 + 1.96 * sd_tot,
            mean(d$diff_ms), naive_sd, mean(d$diff_ms) - 1.96 * naive_sd, mean(d$diff_ms) + 1.96 * naive_sd,
            d[length_class == "vowel_short", median(diff_ms)], d[length_class == "vowel_long", median(diff_ms)],
            nrow(d), ba$min))
fwrite(BA, file.path(OUT, "j4_bland_altman_hier.tsv"), sep = "\t")
print(BA, digits = 5)

# 補足表: paired model の全係数
S <- copy(pc)
S[, term_label := c("(Intercept)" = "Intercept (hand, short vowel, APS, non-nucleus, no following /N/)",
  "sourcemfa" = "Boundary source: MFA",
  "is_nuc" = "Accent nucleus",
  "nasal_following" = "Following moraic nasal",
  "stylesps" = "Style: SPS",
  "length_classvowel_long" = "Length: long vowel",
  "sourcemfa:is_nuc" = "MFA x accent nucleus",
  "sourcemfa:nasal_following" = "MFA x following moraic nasal",
  "sourcemfa:stylesps" = "MFA x SPS",
  "sourcemfa:length_classvowel_long" = "MFA x long vowel")[term]]
setcolorder(S, c("term", "term_label"))
setnames(S, c("est", "se", "ci_lo", "ci_hi", "t"), c("Estimate", "SE", "CI95_lo", "CI95_hi", "t"))
fwrite(S, file.path(OUT, "tableS_paired_model_coefficients.tsv"), sep = "\t")
cat(sprintf("total minutes %.1f\n", as.numeric(difftime(Sys.time(), t_all, units = "mins"))))
