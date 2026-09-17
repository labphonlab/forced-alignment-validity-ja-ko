# SCOPE: journal-only
# P2（docs/decisions-log.md 2026-09-17 事前登録）：correct ~ z(log(err+1)) + (1|speaker) + (1|word)
# 入力: results/revision_2026-09/work/cat_p2_<label>.tsv（src/revision_2026-09/cat_stats.py が出す）
# 出力: results/revision_2026-09/cat_p2_glmer.tsv（集計値のみ）
suppressPackageStartupMessages({ library(lme4); library(data.table) })
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
res_dir <- file.path(here, "..", "..", "results", "revision_2026-09")
out <- list()
for (lab in c("symbol", "supp_symbol_or_mixed_vowel_present")) {
  d <- fread(file.path(res_dir, "work", paste0("cat_p2_", lab, ".tsv")))
  d[, zlerr := as.numeric(scale(log(err + 1)))]
  t0 <- Sys.time()
  m <- glmer(correct ~ zlerr + (1 | speaker) + (1 | word_id), data = d, family = binomial,
             control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cf <- summary(m)$coefficients
  ci <- confint(m, parm = "beta_", method = "Wald")
  vc <- as.data.frame(VarCorr(m))
  conv <- if (length(m@optinfo$conv$lme4$messages)) paste(m@optinfo$conv$lme4$messages, collapse = "|") else "ok"
  for (term in rownames(cf)) {
    out[[length(out) + 1]] <- data.table(subset = lab, term = term, estimate = cf[term, "Estimate"],
      se = cf[term, "Std. Error"], z = cf[term, "z value"], p = cf[term, "Pr(>|z|)"],
      ci_low = ci[term, 1], ci_high = ci[term, 2], or = exp(cf[term, "Estimate"]),
      sd_speaker = vc$sdcor[vc$grp == "speaker"], sd_word = vc$sdcor[vc$grp == "word_id"],
      n = nrow(d), n_speaker = uniqueN(d$speaker), n_word = uniqueN(d$word_id),
      singular = isSingular(m), convergence = conv, seconds = round(el, 1))
  }
}
res <- rbindlist(out)
fwrite(res, file.path(res_dir, "cat_p2_glmer.tsv"), sep = "\t")
print(res)
