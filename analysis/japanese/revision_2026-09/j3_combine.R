# SCOPE: journal-only（J3 の各 seed の集計を1表にまとめる）
suppressPackageStartupMessages(library(data.table))
p <- "results/revision_2026-09/j3_parts"
diffs <- rbindlist(lapply(list.files(p, "_diffs.tsv$", full.names = TRUE), fread))
meta  <- rbindlist(lapply(list.files(p, "_meta.tsv$", full.names = TRUE), fread))
diffs[, ratio := exp(onset_minus_offset)]
# seed 間の最大差（offset ごと）
rng <- diffs[, .(max_across_seed_diff = max(onset_minus_offset) - min(onset_minus_offset)),
             by = .(offset, phone_class)]
out <- merge(diffs, rng, by = c("offset", "phone_class"))
out <- merge(out, meta[, .(seed, offset, converged, singular, minutes, var_speaker, var_file,
                           var_phone, var_resid, wald_int_chisq, wald_int_df)],
             by = c("seed", "offset"))
setorder(out, offset, phone_class, seed)
fwrite(out, "results/revision_2026-09/j3_rq1_interaction_seeds.tsv", sep = "\t")
fwrite(meta, "results/revision_2026-09/j3_rq1_model_meta.tsv", sep = "\t")
print(meta)
w <- dcast(diffs, offset + phone_class ~ seed, value.var = "onset_minus_offset")
w <- merge(w, rng, by = c("offset", "phone_class"))
print(w, digits = 3)
cat("overall max across-seed diff by offset:\n"); print(rng[, .(max = max(max_across_seed_diff)), by = offset])
# 符号が seed 間で変わるクラス、CI が 0 を含むクラス
print(diffs[, .(sign_consistent = length(unique(sign(onset_minus_offset))) == 1,
                any_ci_includes0 = any(ci_lo < 0 & ci_hi > 0)), by = .(offset, phone_class)][
                  sign_consistent == FALSE | any_ci_includes0 == TRUE])
