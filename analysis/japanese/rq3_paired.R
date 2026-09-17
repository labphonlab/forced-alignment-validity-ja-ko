# SCOPE: shared（ICPhS版の主結果。ジャーナル版も土台に使う）
# RQ2（母音長への誤差伝播）: paired analysis。
#
# 査読対応（2026-07-25）: 従来は hand版・MFA版のモデルを別々にフィットし、点推定の
#   相対差（+104% / -98%）を報告していた。これは「2つの点推定の差」であって、その差が
#   統計的に有意かを検定していない（Gelman & Stern 2006）。
#   本スクリプトは boundary_source をトークン内要因とする paired model を組み、
#   boundary_source × accent_nucleus, boundary_source × following_nasal の
#   交互作用（=誤差伝播の大きさそのもの）を CI つきで主結果として推定する。
#
# モデル（査読者指定）:
#   log(duration) ~ boundary_source * (accent_nucleus + following_nasal + style)
#                   + (1 | token) + (1 | speaker) + (1 | phone)
#
# 頑健性: 話者クラスタ／講演クラスタの cluster bootstrap で交互作用CIを再確認。
#   （paired model の交互作用 = トークン内差分 dlog=log(dur_mfa)-log(dur_hand) を
#     アウトカムにした差分モデルの主効果と数学的に一致する。(1|speaker)/(1|phone)/
#     (1|token) はトークン内で不変のため差分で相殺する。この等価性をスクリプト内で
#     数値確認し、bootstrap は高速な差分モデルで回す。）
#
# 入力: results/rq3_vowel_table.tsv（1行=1母音トークン, dur_hand_ms/dur_mfa_ms）

library(data.table)
library(lme4)

d <- fread("results/rq3_vowel_table.tsv")
d[, token := .I]
d[, nasal_following := as.integer(following_context == "moraic_nasal")]
d[, is_nuc := as.integer(is_accent_nucleus)]
d[, style := factor(style)]
d[, vowel := factor(vowel)]
d[, length_class := factor(length_class, levels = c("vowel_short", "vowel_long"))]
d[, dlog := log(dur_mfa_ms) - log(dur_hand_ms)]

cat(sprintf("トークン n = %d, 話者 = %d, 講演 = %d, 母音 = %d\n",
            nrow(d), uniqueN(d$speaker_id), uniqueN(d$file_id), uniqueN(d$vowel)))

# ===== 1. Paired model（主結果） =====
long <- rbind(
  d[, .(token, speaker_id, file_id, vowel, is_nuc, nasal_following, style, length_class,
        source = "hand", y = log(dur_hand_ms))],
  d[, .(token, speaker_id, file_id, vowel, is_nuc, nasal_following, style, length_class,
        source = "mfa",  y = log(dur_mfa_ms))]
)
long[, source := factor(source, levels = c("hand", "mfa"))]

cat("\n[paired lmer をフィット中... (1|token) は大規模なので時間がかかる]\n")
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
m <- lmer(y ~ source * (is_nuc + nasal_following + style + length_class) +
            (1 | token) + (1 | speaker_id) + (1 | vowel),
          data = long, REML = TRUE, control = ctrl)

co <- summary(m)$coefficients
est <- co[, "Estimate"]; se <- co[, "Std. Error"]
res <- data.table(
  term  = rownames(co),
  est   = round(est, 5),
  se    = round(se, 5),
  ci_lo = round(est - 1.96 * se, 5),
  ci_hi = round(est + 1.96 * se, 5),
  t     = round(co[, "t value"], 2)
)
fwrite(res, "results/rq3_paired_coefficients.tsv", sep = "\t")
cat("\n== Paired model 固定効果（全項）==\n"); print(res)

cat("\n== 主結果: boundary_source × 各予測子の交互作用（=誤差伝播）==\n")
key <- res[grepl("^sourcemfa:", term)]
print(key)

# hand基準の主効果に対する相対シフト（従来の +104% / -98% に対応、CI付き）
hand_nuc   <- est["is_nuc"]
hand_nasal <- est["nasal_following"]
int_nuc    <- co["sourcemfa:is_nuc", ]
int_nasal  <- co["sourcemfa:nasal_following", ]
cat(sprintf("\naccent_nucleus: hand主効果 = %.5f\n", hand_nuc))
cat(sprintf("  source×nucleus 交互作用 = %.5f [%.5f, %.5f], t=%.1f\n",
            int_nuc["Estimate"], int_nuc["Estimate"]-1.96*int_nuc["Std. Error"],
            int_nuc["Estimate"]+1.96*int_nuc["Std. Error"], int_nuc["t value"]))
cat(sprintf("  相対シフト = %.1f%% [%.1f%%, %.1f%%]\n",
            100*int_nuc["Estimate"]/abs(hand_nuc),
            100*(int_nuc["Estimate"]-1.96*int_nuc["Std. Error"])/abs(hand_nuc),
            100*(int_nuc["Estimate"]+1.96*int_nuc["Std. Error"])/abs(hand_nuc)))
cat(sprintf("following_nasal: hand主効果 = %.5f\n", hand_nasal))
cat(sprintf("  source×nasal 交互作用 = %.5f [%.5f, %.5f], t=%.1f\n",
            int_nasal["Estimate"], int_nasal["Estimate"]-1.96*int_nasal["Std. Error"],
            int_nasal["Estimate"]+1.96*int_nasal["Std. Error"], int_nasal["t value"]))
cat(sprintf("  相対シフト = %.1f%% [%.1f%%, %.1f%%]\n",
            100*int_nasal["Estimate"]/abs(hand_nasal),
            100*(int_nasal["Estimate"]-1.96*int_nasal["Std. Error"])/abs(hand_nasal),
            100*(int_nasal["Estimate"]+1.96*int_nasal["Std. Error"])/abs(hand_nasal)))

hand_len <- est["length_classvowel_long"]
int_len  <- co["sourcemfa:length_classvowel_long", ]
cat(sprintf("length_class(long): hand主効果 = %.5f\n", hand_len))
cat(sprintf("  source×length 交互作用 = %.5f [%.5f, %.5f], t=%.1f\n",
            int_len["Estimate"], int_len["Estimate"]-1.96*int_len["Std. Error"],
            int_len["Estimate"]+1.96*int_len["Std. Error"], int_len["t value"]))
cat(sprintf("  相対シフト = %.1f%% [%.1f%%, %.1f%%]\n",
            100*int_len["Estimate"]/abs(hand_len),
            100*(int_len["Estimate"]-1.96*int_len["Std. Error"])/abs(hand_len),
            100*(int_len["Estimate"]+1.96*int_len["Std. Error"])/abs(hand_len)))

saveRDS(m, "results/rq3_paired_model.rds")

# ===== 2. 差分モデルとの等価性チェック =====
# dlog ~ is_nuc + nasal_following + style の主効果は paired model の source×X 交互作用と一致するはず
diff_fit <- lm(dlog ~ is_nuc + nasal_following + style + length_class, data = d)
dc <- coef(diff_fit)
cat("\n== 等価性チェック: 差分モデル dlog~... の係数 vs paired 交互作用 ==\n")
cat(sprintf("  is_nuc:          diff=%.5f  paired(source:is_nuc)=%.5f\n",
            dc["is_nuc"], est["sourcemfa:is_nuc"]))
cat(sprintf("  nasal_following: diff=%.5f  paired(source:nasal)=%.5f\n",
            dc["nasal_following"], est["sourcemfa:nasal_following"]))
cat(sprintf("  stylesps:        diff=%.5f  paired(source:stylesps)=%.5f\n",
            dc["stylesps"], est["sourcemfa:stylesps"]))
cat(sprintf("  length(long):    diff=%.5f  paired(source:length)=%.5f\n",
            dc["length_classvowel_long"], est["sourcemfa:length_classvowel_long"]))

# ===== 3. Cluster bootstrap（話者／講演）=====
boot_cluster <- function(clustvar, B = 1000L, seed = 42L) {
  set.seed(seed)
  cl <- unique(d[[clustvar]])
  keep <- c("is_nuc", "nasal_following", "stylesps", "length_classvowel_long")
  out <- matrix(NA_real_, B, length(keep), dimnames = list(NULL, keep))
  for (b in seq_len(B)) {
    samp <- sample(cl, length(cl), replace = TRUE)
    db <- d[data.table(cl = samp), on = setNames("cl", clustvar),
            allow.cartesian = TRUE]
    fb <- lm(dlog ~ is_nuc + nasal_following + style + length_class, data = db)
    out[b, ] <- coef(fb)[keep]
  }
  out
}

for (cv in c("speaker_id", "file_id")) {
  bb <- boot_cluster(cv, B = 1000L)
  qs <- apply(bb, 2, quantile, c(0.025, 0.5, 0.975), na.rm = TRUE)
  cat(sprintf("\n== Cluster bootstrap by %s (B=1000): 2.5%%/50%%/97.5%% ==\n", cv))
  print(round(qs, 5))
  fwrite(as.data.table(round(t(qs), 5), keep.rownames = "term"),
         sprintf("results/rq3_paired_bootstrap_%s.tsv", cv), sep = "\t")
}

cat("\n出力: rq3_paired_coefficients.tsv, rq3_paired_model.rds,\n")
cat("      rq3_paired_bootstrap_speaker_id.tsv, rq3_paired_bootstrap_file_id.tsv\n")
