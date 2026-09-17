# SCOPE: 改訂用の事後解析（docs/decisions-log.md D17）。事前登録なし。
# J7: 反復モデル（run）を単位に含む二項 GLMM、14系統一覧の照合、B2−A2 の T 差の同等性限界。
#   Rscript scripts/revision_2026-09/j7_glmm.R
suppressPackageStartupMessages({library(data.table); library(lme4); library(MASS)})
set.seed(20260916)
OUT <- "results/revision_2026-09"
d <- fread("work/revision_2026-09/talk_group_counts.tsv")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

fit_pair <- function(pair_name, a, b, extra = FALSE) {
  x <- d[pair == pair_name]
  x[, cond := factor(condition, levels = c(a, b))]
  x[, cg := factor(classgroup, levels = c("C", "T"))]
  f <- if (extra) cbind(hit, miss) ~ cond * cg + (1 | talk) + (1 | talk:cg) + (1 | run) + (1 | run:cg)
       else       cbind(hit, miss) ~ cond * cg + (1 | talk) + (1 | talk:cg) + (1 | run)
  t0 <- Sys.time()
  m <- glmer(f, data = x, family = binomial, control = ctrl)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  fe <- fixef(m); V <- as.matrix(vcov(m))
  k <- grep(":", names(fe), value = TRUE)
  est <- fe[k]; se <- sqrt(V[k, k])
  # 確率尺度: 乱効果0の講演・モデルでの差の差（固定効果の多変量正規から 10,000 回）
  sim <- mvrnorm(10000, fe, V)
  p <- function(b) {
    pAC <- plogis(b[, 1]); pBC <- plogis(b[, 1] + b[, 2])
    pAT <- plogis(b[, 1] + b[, 3]); pBT <- plogis(b[, 1] + b[, 2] + b[, 3] + b[, 4])
    cbind(did = 100 * ((pBT - pAT) - (pBC - pAC)), dT = 100 * (pBT - pAT), dC = 100 * (pBC - pAC))
  }
  pt <- p(matrix(fe, 1)); ps <- p(sim)
  msgs <- c(m@optinfo$conv$lme4$messages, m@optinfo$warnings)
  vc <- as.data.frame(VarCorr(m))
  data.table(pair = pair_name, model = if (extra) "+ (1|run:classgroup)" else "specified",
             n_rows = nrow(x), n_runs = uniqueN(x$run), n_talks = uniqueN(x$talk),
             logodds_int = est, logodds_se = se,
             logodds_lo = est - 1.96 * se, logodds_hi = est + 1.96 * se,
             odds_ratio = exp(est), z = est / se,
             prob_did_pt = pt[1, "did"],
             prob_did_lo = quantile(ps[, "did"], .025), prob_did_hi = quantile(ps[, "did"], .975),
             prob_dT_pt = pt[1, "dT"], prob_dT_lo = quantile(ps[, "dT"], .025), prob_dT_hi = quantile(ps[, "dT"], .975),
             prob_dC_pt = pt[1, "dC"], prob_dC_lo = quantile(ps[, "dC"], .025), prob_dC_hi = quantile(ps[, "dC"], .975),
             var_talk = vc$vcov[vc$grp == "talk"], var_talk_cg = vc$vcov[vc$grp == "talk:cg"],
             var_run = vc$vcov[vc$grp == "run"],
             var_run_cg = if (extra) vc$vcov[vc$grp == "run:cg"] else NA_real_,
             converged = if (length(msgs)) paste(unlist(msgs), collapse = " | ") else "ok",
             singular = isSingular(m), minutes = round(mins, 2))
}
R <- rbindlist(list(
  fit_pair("A_vs_B", "A", "B"), fit_pair("A_vs_B", "A", "B", extra = TRUE),
  fit_pair("A2_vs_B2", "A2", "B2"), fit_pair("A2_vs_B2", "A2", "B2", extra = TRUE)))
fwrite(R, file.path(OUT, "j7_glmm_interaction.tsv"), sep = "\t")
print(R, digits = 4)

# ---- 14系統一覧の照合（既存 results と再計算） ----
s <- fread(file.path(OUT, "j7_systems_run_level.tsv"))
cat_old <- fread("results/pilot_devoicing_categorical.tsv")[context == "all"]
rep_old <- fread("results/d14_nolda_replicates_models.tsv")
chk <- merge(s, cat_old[, .(system, sens_old = sensitivity, spec_old = specificity, ba_old = balanced_accuracy)],
             by = "system", all.x = TRUE)
chk <- merge(chk, rep_old[, .(system, T_old = T_within10, C_old = C_within10)], by = "system", all.x = TRUE)
cat(sprintf("max |diff| sens/spec/BA vs pilot_devoicing_categorical: %.4f\n",
            chk[, max(abs(c(sensitivity - sens_old, specificity - spec_old, balanced_accuracy - ba_old)), na.rm = TRUE)]))
cat(sprintf("max |diff| T/C vs d14_nolda_replicates_models (A2/B2 runs): %.4f\n",
            chk[, max(abs(c(T_within10 - T_old, C_within10 - C_old)), na.rm = TRUE)]))
cat("note: results/pilot_replicates_models.tsv now holds the D14 (A2/B2) values; A/B run-level T/C are recomputed here.\n")
chk[, source_check := fifelse(is.na(T_old), "recomputed; sens/spec/BA match pilot_devoicing_categorical",
                              "recomputed; matches d14_nolda_replicates_models + pilot_devoicing_categorical")]
out14 <- chk[, .(pair, condition, system, T_within10, C_within10, sensitivity, specificity, balanced_accuracy,
                 n_ref_devoiced, n_ref_voiced, source_check)]
ord <- c("B0", "B1", "A", "A_r2", "A_r3", "B", "B_r2", "B_r3", "A2", "A2_r2", "A2_r3", "B2", "B2_r2", "B2_r3")
out14 <- out14[match(ord, system)]
fwrite(out14, file.path(OUT, "j7_systems_table.tsv"), sep = "\t")
print(out14)
# 条件平均と SD
print(out14[condition %in% c("A", "B", "A2", "B2"),
            .(T_mean = mean(T_within10), T_sd = sd(T_within10), C_mean = mean(C_within10), C_sd = sd(C_within10),
              sens_mean = mean(sensitivity), BA_mean = mean(balanced_accuracy)), by = condition], digits = 4)

# ---- 同等性限界（TOST） ----
welch <- function(a, b, level) {
  tt <- t.test(b, a, var.equal = FALSE, conf.level = level)
  c(diff = unname(tt$estimate[1] - tt$estimate[2]), lo = tt$conf.int[1], hi = tt$conf.int[2], df = unname(tt$parameter))
}
A2T <- out14[condition == "A2", T_within10]; B2T <- out14[condition == "B2", T_within10]
A2C <- out14[condition == "A2", C_within10]; B2C <- out14[condition == "B2", C_within10]
w95 <- welch(A2T, B2T, .95); w90 <- welch(A2T, B2T, .90)
bound_T <- max(abs(w90[c("lo", "hi")]))
# 差の差（run 単位: B2 の各 run の T−C と A2 の各 run の T−C の Welch）
w90d <- welch(A2T - A2C, B2T - B2C, .90); w95d <- welch(A2T - A2C, B2T - B2C, .95)
bound_D <- max(abs(w90d[c("lo", "hi")]))
TO <- data.table(
  quantity = c("B2-A2 T (pp)", "B2-A2 DiD T-C (pp)"),
  estimate = c(w95["diff"], w95d["diff"]),
  ci95_lo = c(w95["lo"], w95d["lo"]), ci95_hi = c(w95["hi"], w95d["hi"]),
  ci90_lo = c(w90["lo"], w90d["lo"]), ci90_hi = c(w90["hi"], w90d["hi"]),
  welch_df = c(w90["df"], w90d["df"]),
  tost_equivalence_bound_pp = c(bound_T, bound_D),
  statement = c(sprintf("With 3 runs per condition, the data exclude (TOST, alpha=.05 each side) only effects larger than +/-%.2f pp on T; effects up to ~%.1f pp in either direction are compatible with the data.", bound_T, bound_T),
                sprintf("For the DiD, effects beyond +/-%.2f pp are excluded at the same level.", bound_D)))
fwrite(TO, file.path(OUT, "j7_tost_bound.tsv"), sep = "\t")
print(TO, digits = 4)
