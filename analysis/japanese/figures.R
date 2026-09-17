# SCOPE: icphs
# ICPhS原稿v2の Figure 1・2 を生成する。
#
# 二重盲検（CLAUDE.md §8）: 図中に所属・著者を特定できる情報を入れない。ラベルは英語。
# 配色: Okabe-Ito 由来の CVD 安全な2色（青 #0072B2 / 朱 #D55E00、dataviz validator でPASS）。
#   ICPhS予稿はグレースケール印刷されうるため、色に加えて位置（dodge）・形状で冗長符号化する。
#
# Figure 1: 音素クラス × 境界タイプ（onset/offset）別の境界誤差 箱ひげ図
# Figure 2: 母音長モデルの固定効果 フォレストプロット（hand vs MFA、95%CI）

library(data.table)
library(ggplot2)

BLUE <- "#0072B2"   # onset / hand（人手参照）
VERM <- "#D55E00"   # offset / MFA

# ============================================================
# Figure 1: Boundary error by phone class and boundary type
# ============================================================
d <- fread("results/rq1_table.tsv")
m <- d[boundary_type %in% c("onset", "offset") & !is.na(abs_error_ms)]

# 主要な音素クラスに絞る（境界数 < 2000 の稀少クラスと、境界を持たない
# vocalfry_fused/geminate_no_closure 等の評価不能クラスは除外して可読性を確保）。
# 除外は図の注に明記する（下のキャプション案参照）。
keep <- m[, .N, by = phone_class][N >= 4000, phone_class]
m1 <- m[phone_class %in% keep]

# 音素クラスを中央値順（onset基準）で並べる
ord <- m1[boundary_type == "onset", .(med = median(abs_error_ms)), by = phone_class][order(med), phone_class]
m1[, phone_class := factor(phone_class, levels = ord)]
m1[, boundary_type := factor(boundary_type, levels = c("onset", "offset"))]

# 読みやすい英語ラベル
class_labels <- c(
  vowel_short = "short vowel", vowel_long = "long vowel", stop = "stop",
  fricative = "fricative", affricate = "affricate", nasal = "nasal (onset)",
  flap = "flap", approximant = "approximant", moraic_nasal = "moraic nasal /N/",
  moraic_nasal_fused = "moraic nasal (fused)", geminate_stop = "geminate stop",
  devoiced_fused = "devoiced (fused)"
)

fig1 <- ggplot(m1, aes(x = phone_class, y = abs_error_ms, fill = boundary_type)) +
  # n が大きい（各20万境界）ため個別の外れ値点は描かない（黒い帯になり可読性を損なう）。
  # 箱＋ひげ（1.5 IQR）で分布を示し、裾は数値（95パーセンタイル、rq1_summary）で報告する。
  geom_boxplot(outlier.shape = NA, coef = 1.5,
               position = position_dodge(width = 0.78), width = 0.7,
               linewidth = 0.35, colour = "grey20") +
  scale_fill_manual(values = c(onset = BLUE, offset = VERM),
                    labels = c(onset = "onset", offset = "offset"),
                    name = "Boundary") +
  scale_x_discrete(labels = class_labels) +
  coord_flip(ylim = c(0, 80)) +
  labs(x = NULL, y = "Absolute boundary error (ms)") +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = c(0.86, 0.14),
    legend.background = element_rect(fill = "white", colour = "grey80", linewidth = 0.3),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text = element_text(colour = "grey20")
  )

ggsave("results/figure1_boundary_error_by_class.png", fig1,
       width = 7.2, height = 4.6, dpi = 300)
ggsave("results/figure1_boundary_error_by_class.pdf", fig1, width = 7.2, height = 4.6)
cat("Figure 1 written. classes shown:", paste(ord, collapse = ", "), "\n")
cat("n(onset)=", m1[boundary_type=="onset", .N], " n(offset)=", m1[boundary_type=="offset", .N], "\n")

# ============================================================
# Figure 2: Fixed-effect estimates, hand vs MFA vowel-length model
# ============================================================
cmp <- fread("results/rq3_fixef_comparison.tsv")

# 表示する項（アクセント核・撥音後続環境）。ICPhSの主眼はこの2つ。
# 参考として母音長クラス主効果も入れると誤差伝播の相対的大きさが見えるが、
# 指示どおりアクセント核・撥音後続に絞る。
want <- c(
  is_nuc = "Accent nucleus",
  nasal_following = "Following moraic nasal"
)
sub <- cmp[term %in% names(want)]
sub[, label := want[term]]

# long 形に展開し 95%CI を SE から作る（est ± 1.96 SE）
long <- rbind(
  sub[, .(label, version = "Hand (reference)", est = hand_est, se = hand_se)],
  sub[, .(label, version = "MFA",              est = mfa_est,  se = mfa_se)]
)
long[, `:=`(lo = est - 1.96 * se, hi = est + 1.96 * se)]
long[, version := factor(version, levels = c("Hand (reference)", "MFA"))]
long[, label := factor(label, levels = rev(unname(want)))]

fig2 <- ggplot(long, aes(x = est, y = label, colour = version, shape = version)) +
  geom_vline(xintercept = 0, colour = "grey60", linewidth = 0.4, linetype = "22") +
  geom_errorbarh(aes(xmin = lo, xmax = hi),
                 position = position_dodge(width = 0.55),
                 height = 0.18, linewidth = 0.7) +
  geom_point(position = position_dodge(width = 0.55), size = 2.8, stroke = 0.4,
             fill = "white") +
  scale_colour_manual(values = c("Hand (reference)" = BLUE, "MFA" = VERM), name = NULL) +
  scale_shape_manual(values = c("Hand (reference)" = 16, "MFA" = 17), name = NULL) +
  labs(x = "Fixed-effect estimate on log vowel duration (95% CI)", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    legend.justification = "left",
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(colour = "grey20", size = 11)
  )

ggsave("results/figure2_vowel_length_fixef.png", fig2,
       width = 7.0, height = 3.2, dpi = 300)
ggsave("results/figure2_vowel_length_fixef.pdf", fig2, width = 7.0, height = 3.2)
cat("Figure 2 written. terms:", paste(long$label |> unique(), collapse = ", "), "\n")
print(long[, .(label, version, est = round(est,4), lo = round(lo,4), hi = round(hi,4))])
