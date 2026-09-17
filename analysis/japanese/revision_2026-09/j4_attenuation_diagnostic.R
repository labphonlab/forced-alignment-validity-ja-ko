# SCOPE: journal-only（J4 補助: 差分モデルで is_nuc が −0.038 → −0.028 に縮む原因の切り分け。係数のみ出力）
suppressPackageStartupMessages(library(data.table))
d <- fread("results/rq3_vowel_table.tsv")
d[, `:=`(nas = as.integer(following_context == "moraic_nasal"), nuc = as.integer(is_accent_nucleus),
         dlog = log(dur_mfa_ms) - log(dur_hand_ms))]
f <- list(
  ols            = dlog ~ nuc + nas + style + length_class,
  vowel_fixed    = dlog ~ nuc + nas + style + factor(vowel),
  speaker_fixed  = dlog ~ nuc + nas + style + length_class + factor(speaker_id),
  talk_fixed     = dlog ~ nuc + nas + factor(file_id) + length_class,
  vowel_talk_fix = dlog ~ nuc + nas + factor(vowel) + factor(file_id))
out <- rbindlist(lapply(names(f), function(n) {
  co <- summary(lm(f[[n]], data = d))$coefficients
  data.table(spec = n, term = c("nuc", "nas"), est = co[c("nuc", "nas"), 1], se = co[c("nuc", "nas"), 2])
}))
# 母音ごとの核の割合と平均 dlog（母音記号は CSJ ラベルではなく音素カテゴリの集計）
vt <- d[, .(n = .N, nuc_share = mean(nuc), mean_dlog = mean(dlog)), by = vowel][order(vowel)]
fwrite(out, "results/revision_2026-09/j4_attenuation_diagnostic.tsv", sep = "\t")
fwrite(vt, "results/revision_2026-09/j4_vowel_nucleus_share.tsv", sep = "\t")
print(out, digits = 4); print(vt, digits = 3)
