# SCOPE: shared
# 査読対応: RQ2主モデルに (1|talk) を追加した感度分析。
# 主モデル（analysis/rq3_paired.R）が (1|token)+(1|speaker)+(1|vowel) のみで
# (1|file_id)（=talk）を持たない点を査読者が指摘。cluster bootstrapはtalk単位でも
# 行っているため、主モデル本体にtalkランダム切片を加えても主要な交互作用推定値が
# 変わらないかを確認する。
#
# 入力: results/rq3_vowel_table.tsv（analysis/rq3_paired.R と同一）
# 出力: results/rq3_paired_talk_sensitivity.tsv

library(data.table)
library(lme4)

d <- fread("results/rq3_vowel_table.tsv")
d[, token := .I]
d[, nasal_following := as.integer(following_context == "moraic_nasal")]
d[, is_nuc := as.integer(is_accent_nucleus)]
d[, style := factor(style)]
d[, vowel := factor(vowel)]
d[, length_class := factor(length_class, levels = c("vowel_short", "vowel_long"))]

long <- rbind(
  d[, .(token, speaker_id, file_id, vowel, is_nuc, nasal_following, style, length_class,
        source = "hand", y = log(dur_hand_ms))],
  d[, .(token, speaker_id, file_id, vowel, is_nuc, nasal_following, style, length_class,
        source = "mfa",  y = log(dur_mfa_ms))]
)
long[, source := factor(source, levels = c("hand", "mfa"))]

cat(sprintf("トークン n = %d, 話者 = %d, 講演 = %d, 母音 = %d\n",
            nrow(d), uniqueN(d$speaker_id), uniqueN(d$file_id), uniqueN(d$vowel)))

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

t0 <- Sys.time()
cat("\n[(1|file_id) 追加版をフィット中...]\n")
m_talk <- lmer(y ~ source * (is_nuc + nasal_following + style + length_class) +
                 (1 | token) + (1 | speaker_id) + (1 | vowel) + (1 | file_id),
               data = long, REML = TRUE, control = ctrl)
t1 <- Sys.time()
cat(sprintf("所要時間: %.1f 分\n", as.numeric(difftime(t1, t0, units = "mins"))))
cat("singular?", isSingular(m_talk), "\n")

co <- summary(m_talk)$coefficients
est <- co[, "Estimate"]; se <- co[, "Std. Error"]
res <- data.table(
  term  = rownames(co),
  est   = round(est, 5),
  se    = round(se, 5),
  ci_lo = round(est - 1.96 * se, 5),
  ci_hi = round(est + 1.96 * se, 5)
)
fwrite(res, "results/rq3_paired_talk_sensitivity.tsv", sep = "\t")

cat("\n交互作用項（主要3項）:\n")
print(res[grepl("^source", term) & grepl(":", term)])

# 主モデルとの比較
main <- fread("results/rq3_paired_coefficients.tsv")
cat("\n主モデル（talkなし）との比較:\n")
key_terms <- c("sourcemfa:is_nuc", "sourcemfa:nasal_following", "sourcemfa:length_classvowel_long")
cmp <- merge(main[term %in% key_terms, .(term, est_main = est, ci_lo_main = ci_lo, ci_hi_main = ci_hi)],
             res[term %in% key_terms, .(term, est_talk = est, ci_lo_talk = ci_lo, ci_hi_talk = ci_hi)],
             by = "term")
print(cmp)

vc <- as.data.frame(VarCorr(m_talk))
cat("\n分散成分:\n")
print(vc[, c("grp", "vcov", "sdcor")])
