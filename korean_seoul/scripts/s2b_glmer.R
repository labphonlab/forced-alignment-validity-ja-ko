#!/usr/bin/env Rscript
# S2(b): correct ~ z(log boundary error) * process + (1|speaker) + (1|word_type)
# Input : work/candidates_natural.tsv (token-level, git-ignored)
# Output: s2b_glmer_fixed.tsv, s2b_glmer_models.tsv (aggregates only)
# log error = log(dev_ms + 1); z-scored within each analysed subset.
suppressPackageStartupMessages({ library(lme4) })
rev <- Sys.getenv("SEOUL_ANALYSIS_DIR", unset = ".")
d0 <- read.delim(file.path(rev, "work/candidates_natural.tsv"), stringsAsFactors = FALSE)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

conv_msg <- function(m) {
  msgs <- c(m@optinfo$conv$lme4$messages, m@optinfo$warnings)
  msgs <- unlist(msgs)
  s <- if (length(msgs)) paste(msgs, collapse = " | ") else "ok"
  if (isSingular(m)) s <- paste(s, "| singular")
  s
}

fixed_rows <- list(); model_rows <- list()
for (sub in c("identity_safe", "full")) {
  for (meas in c("dev_site_ms", "dev_word_ms")) {
    d <- d0
    if (sub == "identity_safe") d <- d[d$identity_safe == 1, ]
    d <- d[!is.na(d[[meas]]), ]
    tab <- table(d$process)
    d <- d[d$process %in% names(tab)[tab >= 20], ]
    d$zlog <- as.numeric(scale(log(d[[meas]] + 1)))
    # reference = most frequent process; sum-to-zero would also work
    ref <- names(sort(table(d$process), decreasing = TRUE))[1]
    d$process <- relevel(factor(d$process), ref = ref)
    fits <- list()
    fits$main <- glmer(correct ~ zlog + process + (1 | speaker) + (1 | word_type),
                       data = d, family = binomial, control = ctrl)
    fits$interaction <- glmer(correct ~ zlog * process + (1 | speaker) + (1 | word_type),
                              data = d, family = binomial, control = ctrl)
    # same model as `interaction`, reparametrised to give one slope per process
    fits$nested_slopes <- glmer(correct ~ process + process:zlog + (1 | speaker) + (1 | word_type),
                                data = d, family = binomial, control = ctrl)
    lrt <- anova(fits$main, fits$interaction)
    for (nm in names(fits)) {
      m <- fits[[nm]]
      cf <- summary(m)$coefficients
      ci <- confint(m, method = "Wald", parm = "beta_")
      vc <- as.data.frame(VarCorr(m))
      model_rows[[length(model_rows) + 1]] <- data.frame(
        subset = sub, measure = meas, model = nm, n = nrow(d),
        speakers = length(unique(d$speaker)), word_types = length(unique(d$word_type)),
        reference = ref, logLik = round(as.numeric(logLik(m)), 2), AIC = round(AIC(m), 2),
        sd_speaker = round(vc$sdcor[vc$grp == "speaker"], 3),
        sd_word_type = round(vc$sdcor[vc$grp == "word_type"], 3),
        LRT_interaction_chisq = if (nm == "interaction") round(lrt$Chisq[2], 3) else NA,
        LRT_df = if (nm == "interaction") lrt$Df[2] else NA,
        LRT_p = if (nm == "interaction") signif(lrt$`Pr(>Chisq)`[2], 3) else NA,
        convergence = conv_msg(m))
      for (i in seq_len(nrow(cf))) {
        term <- rownames(cf)[i]
        fixed_rows[[length(fixed_rows) + 1]] <- data.frame(
          subset = sub, measure = meas, model = nm, term = term,
          estimate = round(cf[i, 1], 4), se = round(cf[i, 2], 4),
          ci_lo = round(ci[term, 1], 4), ci_hi = round(ci[term, 2], 4),
          OR = round(exp(cf[i, 1]), 3), p = signif(cf[i, 4], 3))
      }
    }
  }
}
F <- do.call(rbind, fixed_rows); M <- do.call(rbind, model_rows)
write.table(F, file.path(rev, "s2b_glmer_fixed.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(M, file.path(rev, "s2b_glmer_models.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
print(M, row.names = FALSE)
print(F[F$model != "interaction" & grepl("zlog", F$term), ], row.names = FALSE)
cat("R", R.version.string, "lme4", as.character(packageVersion("lme4")), "\n")
