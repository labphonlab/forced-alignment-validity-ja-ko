# SCOPE: journal-only（査読対応の感度分析。事後。docs/decisions-log.md 2026-09-16 J1–J6）
# J3: RQ1 モデルに phone_class × boundary_type を入れ、講演の切片を加える。
#   log(abs_error_ms + OFFSET) ~ phone_class * boundary_type + phone_class * style +
#       local_speech_rate + (1|speaker_id) + (1|file_id) + (1|ref_phone)
# 抽出: phone_class × boundary_type 層別、各層 CAP=20,000（少なければ全件）。
# 使い方: Rscript j3_rq1_interaction_seeds.R <seed> <offset>
# 出力: results/revision_2026-09/j3_parts/seed<seed>_off<offset>.{tsv,rds-free summary}
#   集計値（係数・分散成分・クラス別 onset−offset 差）のみ。トークン行は書かない。
suppressPackageStartupMessages({library(data.table); library(lme4)})
args <- commandArgs(trailingOnly = TRUE)
SEED <- as.integer(args[1]); OFFSET <- as.numeric(args[2])
CAP <- 20000L
t0 <- Sys.time()
d <- fread("results/rq1_table.tsv",
           select = c("file_id", "speaker_id", "style", "phone_class", "boundary_type",
                      "abs_error_ms", "local_speech_rate", "ref_label_canonical"))
m <- d[boundary_type %in% c("onset", "offset") & !is.na(abs_error_ms)]
rm(d); gc()
m <- m[!is.na(local_speech_rate) & phone_class != "" & speaker_id != ""]
n_full <- nrow(m)
# 稀少クラスの畳み込みは既存 rq1_error_by_class.R と同一（全データで境界数 < 200）
rare <- m[, .N, by = phone_class][N < 200, phone_class]
m[phone_class %in% rare, phone_class := "other_rare"]
set.seed(SEED)
setorder(m, phone_class, boundary_type)   # 群の順序を固定
m <- m[, .SD[if (.N > CAP) sample(.N, CAP) else seq_len(.N)], by = .(phone_class, boundary_type)]
m[, y := log(abs_error_ms + OFFSET)]
m[, phone_class := relevel(factor(phone_class), ref = "vowel_short")]
m[, boundary_type := relevel(factor(boundary_type), ref = "offset")]
m[, style := factor(style)]
m[, ref_phone := factor(ref_label_canonical)]
m[, file_id := factor(file_id)]; m[, speaker_id := factor(speaker_id)]
cat(sprintf("seed=%d offset=%g: %d -> %d rows; speakers=%d talks=%d phones=%d classes=%d; rare=%s\n",
            SEED, OFFSET, n_full, nrow(m), nlevels(m$speaker_id), nlevels(m$file_id),
            nlevels(m$ref_phone), nlevels(m$phone_class), paste(rare, collapse = ",")))

fit <- lmer(y ~ phone_class * boundary_type + phone_class * style + local_speech_rate +
              (1 | speaker_id) + (1 | file_id) + (1 | ref_phone),
            data = m, REML = TRUE,
            control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                                  calc.derivs = FALSE))
el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
msgs <- c(fit@optinfo$conv$lme4$messages, fit@optinfo$warnings)
conv <- if (length(msgs)) paste(unlist(msgs), collapse = " | ") else "ok"
sing <- isSingular(fit)
vc <- as.data.frame(VarCorr(fit))
fe <- fixef(fit); V <- as.matrix(vcov(fit))
# クラス別の onset − offset 差（モデルが含意する値。style・話速とは交互作用しないので一意）
classes <- levels(m$phone_class)
rows <- lapply(classes, function(cl) {
  L <- setNames(rep(0, length(fe)), names(fe))
  L["boundary_typeonset"] <- 1
  nm <- paste0("phone_class", cl, ":boundary_typeonset")
  if (cl != "vowel_short") L[nm] <- 1
  est <- sum(L * fe); se <- sqrt(drop(t(L) %*% V %*% L))
  data.table(phone_class = cl, onset_minus_offset = est, se = se,
             ci_lo = est - 1.96 * se, ci_hi = est + 1.96 * se,
             n_onset = m[phone_class == cl & boundary_type == "onset", .N],
             n_offset = m[phone_class == cl & boundary_type == "offset", .N])
})
res <- rbindlist(rows)
res[, `:=`(seed = SEED, offset = OFFSET)]
# 交互作用の全体検定（ML でなく REML 同士は不可なので、Wald χ² で）
int_names <- grep(":boundary_typeonset$", names(fe), value = TRUE)
b <- fe[int_names]; Vi <- V[int_names, int_names]
wald <- drop(t(b) %*% solve(Vi) %*% b)
meta <- data.table(seed = SEED, offset = OFFSET, n_rows = nrow(m), n_full = n_full,
                   converged = conv, singular = sing, minutes = round(el, 1),
                   var_speaker = vc$vcov[vc$grp == "speaker_id"],
                   var_file = vc$vcov[vc$grp == "file_id"],
                   var_phone = vc$vcov[vc$grp == "ref_phone"],
                   var_resid = vc$vcov[vc$grp == "Residual"],
                   wald_int_chisq = wald, wald_int_df = length(b),
                   b_onset_ref = fe["boundary_typeonset"],
                   b_rate = fe["local_speech_rate"])
dir.create("results/revision_2026-09/j3_parts", showWarnings = FALSE)
stem <- sprintf("results/revision_2026-09/j3_parts/seed%d_off%g", SEED, OFFSET)
fwrite(res, paste0(stem, "_diffs.tsv"), sep = "\t")
fwrite(meta, paste0(stem, "_meta.tsv"), sep = "\t")
co <- summary(fit)$coefficients
fwrite(data.table(term = rownames(co), co, seed = SEED, offset = OFFSET),
       paste0(stem, "_fixef.tsv"), sep = "\t")
print(meta); print(res)
