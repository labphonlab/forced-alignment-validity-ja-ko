# 過程カテゴリごとの検出率。語型・話者を変量効果に入れて、
# カテゴリ差が少数の語彙タイプに由来しないことを確認する。
suppressPackageStartupMessages({library(dplyr); library(lme4)})

s <- read.csv("data/scored.csv", fileEncoding="UTF-8")
m <- read.csv("data/target_prons.csv", fileEncoding="UTF-8")
d <- s |> filter(outcome %in% c("followed_transcriber","chose_citation")) |>
  left_join(m, by=c("form","orig")) |>
  mutate(hit = as.integer(outcome=="followed_transcriber"),
         nf = lengths(strsplit(pron_form, " ")),
         no = lengths(strsplit(pron_orig, " ")),
         dlen = no - nf,
         abs_dlen = abs(dlen),
         category = relevel(factor(category), ref="tensification"))

cat("n =", nrow(d), " 語型 =", n_distinct(d$form), " 話者 =", n_distinct(d$speaker_id), "\n\n")

cat("### 混合ロジスティック: カテゴリ（語型・話者を変量効果）\n")
m1 <- glmer(hit ~ category + (1|form) + (1|speaker_id), data=d, family=binomial,
            control=glmerControl(optimizer="bobyqa"))
print(round(summary(m1)$coefficients, 4))

cat("\n### 音素数差を統制してもカテゴリ差は残るか\n")
m2 <- glmer(hit ~ category + abs_dlen + (1|form) + (1|speaker_id), data=d, family=binomial,
            control=glmerControl(optimizer="bobyqa"))
print(round(summary(m2)$coefficients, 4))

cat("\n### 語型間のばらつき（変量効果の分散）\n")
print(VarCorr(m1))

cat("\n### 長さ不変トークンのみ（置換系の比較）\n")
d0 <- d |> filter(dlen==0) |> mutate(category=droplevels(category))
m3 <- glmer(hit ~ category + (1|form) + (1|speaker_id), data=d0, family=binomial,
            control=glmerControl(optimizer="bobyqa"))
print(round(summary(m3)$coefficients, 4))

cat("\n### 濃音化: 語型ごとの検出率上位/下位（少数語型への依存確認）\n")
d |> filter(category=="tensification") |> group_by(form, orig) |>
  summarise(n=n(), rate=mean(hit), .groups="drop") |> filter(n>=5) |>
  arrange(desc(rate)) |> (\(x) rbind(head(x,5), tail(x,5)))() |> as.data.frame() |> print()
