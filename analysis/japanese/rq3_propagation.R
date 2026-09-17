# SCOPE: shared（ICPhS版は母音長のみ、ジャーナル版はVOT・閉鎖区間まで拡張 + ICC）
# RQ3: 手作業境界版・MFA境界版で同一モデルを並行実行し、固定効果推定値・
# 信頼区間の一致度をBland-Altman型で比較。ジャーナル版はICC（irr::icc 等）も併用。
#
# ICPhS版の問い（母音長×アクセント型のみ）:
#   MFAの境界誤差は、母音長とアクセント型の関係の推定にどれだけ伝播するか。
#   同一モデルを hand（人手境界の母音長）と MFA（MFA境界の母音長）で並行推定し、
#   固定効果を比較する。歪みが小さければ「MFA境界でも母音長×アクセント研究に使える」。
#
# モデル: 母音長 ~ 長さクラス * アクセント核 + スタイル + 後続環境 + (1 | 話者) + (1 | 母音)
#   ※ アクセント型は 0..8 だが、母音長に効くのは「その母音が核かどうか」なので
#      is_accent_nucleus を主効果に使う（アクセント型番号そのものは語レベルの属性）。
#      長さクラス（短/長）は母音長の主要因なので必ず入れる。
#
# 入力: results/rq3_vowel_table.tsv（src/acoustics/vowel_measures.py の出力）
#
# ⚠ Limitations（ICPhS原稿v2に明記）: アクセントは知覚アクセント（辞書規範ではない）。
#   CSJコアのみ・東京方言話者に限定。docs/decisions-log.md 参照。

# --- 依存パッケージ（renv.lock 固定対象）---
library(data.table)
library(lme4)
library(irr)         # ICC（ジャーナル版のみ）
library(ggplot2)

d <- fread("results/rq3_vowel_table.tsv")

# 撥音後続の母音は境界が不明瞭になりうる（ɰ̃環境）。層別のため印を付ける（落とさない）。
d[, nasal_following := as.integer(following_context == "moraic_nasal")]
d[, is_nuc := as.integer(is_accent_nucleus)]
d[, length_class := factor(length_class, levels = c("vowel_short", "vowel_long"))]
d[, style := factor(style)]
d[, vowel := factor(vowel)]

cat(sprintf("母音ペア n = %d, 話者 = %d, 母音 = %d\n",
            nrow(d), uniqueN(d$speaker_id), uniqueN(d$vowel)))

# ---- 1. 記述: アクセント核 × 長さクラス 別の母音長（hand vs MFA）----
summ <- d[, .(
  n = .N,
  hand_median = median(dur_hand_ms),
  mfa_median  = median(dur_mfa_ms),
  diff_median = median(dur_diff_ms),
  diff_mean   = mean(dur_diff_ms),
  diff_sd     = sd(dur_diff_ms)
), by = .(length_class, is_nuc)][order(length_class, is_nuc)]
fwrite(summ, "results/rq3_duration_summary.tsv", sep = "\t")
cat("\n== 長さクラス×アクセント核 別 母音長（hand/MFA中央値, ms）==\n")
print(summ)

# ---- 2. 同一モデルを hand / MFA で並行推定 ----
fit_one <- function(dt, ycol) {
  dt <- copy(dt)
  dt[, y := log(get(ycol))]
  lmer(y ~ length_class * is_nuc + style + nasal_following +
         (1 | speaker_id) + (1 | vowel),
       data = dt, REML = TRUE,
       control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
}

cat("\n-- hand（人手境界）版モデル --\n")
fit_hand <- fit_one(d, "dur_hand_ms")
print(summary(fit_hand)$coefficients)

cat("\n-- MFA（MFA境界）版モデル --\n")
fit_mfa <- fit_one(d, "dur_mfa_ms")
print(summary(fit_mfa)$coefficients)

# ---- 3. 固定効果の比較（誤差伝播の量）----
ch <- summary(fit_hand)$coefficients
cm <- summary(fit_mfa)$coefficients
common <- intersect(rownames(ch), rownames(cm))
compare <- data.table(
  term      = common,
  hand_est  = ch[common, "Estimate"],
  mfa_est   = cm[common, "Estimate"],
  hand_se   = ch[common, "Std. Error"],
  mfa_se    = cm[common, "Std. Error"]
)
compare[, est_diff := mfa_est - hand_est]
# hand の推定値に対する相対的なずれ（誤差伝播の指標）
compare[, rel_shift_pct := 100 * est_diff / abs(hand_est)]
fwrite(compare, "results/rq3_fixef_comparison.tsv", sep = "\t")
cat("\n== 固定効果の hand vs MFA 比較（誤差伝播）==\n")
print(compare)

saveRDS(list(hand = fit_hand, mfa = fit_mfa), "results/rq3_models.rds")

# ---- 4. Bland–Altman: 母音長そのものの hand vs MFA 一致度 ----
d[, mean_dur := (dur_hand_ms + dur_mfa_ms) / 2]
ba_bias <- mean(d$dur_diff_ms)
ba_loa  <- 1.96 * sd(d$dur_diff_ms)
cat(sprintf("\n== Bland-Altman（母音長 hand-MFA）==\n  bias = %.2f ms, LoA = [%.2f, %.2f] ms\n",
            ba_bias, ba_bias - ba_loa, ba_bias + ba_loa))

p <- ggplot(d[sample(.N, min(.N, 20000))], aes(x = mean_dur, y = dur_diff_ms)) +
  geom_point(alpha = 0.1, size = 0.4) +
  geom_hline(yintercept = ba_bias, colour = "blue") +
  geom_hline(yintercept = c(ba_bias - ba_loa, ba_bias + ba_loa),
             linetype = "dashed", colour = "red") +
  coord_cartesian(xlim = c(0, 250), ylim = c(-100, 100)) +
  labs(x = "母音長 平均 (hand, MFA) ms", y = "母音長差 hand - MFA (ms)",
       title = "RQ3: 母音長の hand vs MFA 一致度（Bland-Altman・CSJ 31講演暫定）") +
  theme_minimal()
ggsave("results/rq3_bland_altman.png", p, width = 8, height = 6, dpi = 150)

cat("\n出力: rq3_duration_summary.tsv, rq3_fixef_comparison.tsv,\n")
cat("      rq3_models.rds, rq3_bland_altman.png\n")
