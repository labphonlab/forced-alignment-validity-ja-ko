# SCOPE: shared
# Figure 2 を paired model（analysis/rq3_paired.R の rq3_paired_model.rds）から再生成する。
# 従来の Figure 2 は hand版・MFA版を別々にフィットした2モデルの係数を並べていたが、
# 査読対応（2026-07-25）で主結果を paired model の交互作用に統一したため、図も
# paired model から導く hand推定 vs MFA推定（=hand+source×X交互作用）に差し替える。
#
# 各予測子について:
#   hand推定 = 主効果 β_X （source=hand が基準）
#   MFA 推定 = β_X + β_{source:X}
#   MFA推定のSEは線形結合の分散 Var(a+b)=Var(a)+Var(b)+2Cov(a,b) を vcov から計算。

library(lme4)
library(ggplot2)

m <- readRDS("results/rq3_paired_model.rds")
b  <- fixef(m)
V  <- as.matrix(vcov(m))

lincomb <- function(terms) {
  est <- sum(b[terms])
  se  <- sqrt(sum(V[terms, terms]))
  c(est = est, lo = est - 1.96 * se, hi = est + 1.96 * se)
}

preds <- list(
  `Accent nucleus`         = c(main = "is_nuc",          inter = "sourcemfa:is_nuc"),
  `Following moraic nasal` = c(main = "nasal_following",  inter = "sourcemfa:nasal_following")
)

rows <- list()
for (nm in names(preds)) {
  p <- preds[[nm]]
  hand <- lincomb(p["main"])
  mfa  <- lincomb(c(p["main"], p["inter"]))
  rows[[length(rows) + 1]] <- data.frame(predictor = nm, source = "Hand (reference)",
                                          est = hand["est"], lo = hand["lo"], hi = hand["hi"])
  rows[[length(rows) + 1]] <- data.frame(predictor = nm, source = "MFA",
                                          est = mfa["est"],  lo = mfa["lo"],  hi = mfa["hi"])
}
df <- do.call(rbind, rows)
df$source <- factor(df$source, levels = c("Hand (reference)", "MFA"))

# 数値をテキスト用に出力
cat("== Figure 2 用の hand vs MFA 推定（paired model 由来）==\n")
print(df, row.names = FALSE, digits = 4)

p <- ggplot(df, aes(x = est, y = predictor, colour = source, shape = source)) +
  geom_vline(xintercept = 0, linewidth = 0.3, colour = "grey60") +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18,
                 position = position_dodge(width = 0.5), linewidth = 0.5) +
  geom_point(size = 2.4, position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = c("Hand (reference)" = "#0072B2", "MFA" = "#D55E00"),
                      name = NULL) +
  scale_shape_manual(values = c("Hand (reference)" = 16, "MFA" = 17), name = NULL) +
  labs(x = "Effect on log vowel duration (95% CI)", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "top", panel.grid.major.y = element_blank())

ggsave("results/figure2_vowel_length_fixef.pdf", p, width = 4.2, height = 1.55)
ggsave("results/figure2_vowel_length_fixef.png", p, width = 4.2, height = 1.55, dpi = 150)
cat("\n出力: results/figure2_vowel_length_fixef.pdf / .png\n")
