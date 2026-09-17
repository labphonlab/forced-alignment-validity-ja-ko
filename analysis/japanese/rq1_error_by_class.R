# SCOPE: shared（ICPhS版は結果の一部のみ抜粋、ジャーナル版は全体使用）
# RQ1: 音素クラス別・境界タイプ別の誤差解剖 + 4類型（displacement/omission/
# insertion/substitution）の生起率とクラスによる偏り。
#
# モデル（CLAUDE.md §6、2026-07-21 簡略化後）:
#   log(誤差) ~ 音素クラス * スタイル + 局所話速 + 境界タイプ + (1 | 話者) + (1 | 音素)
#   ※ 話者ランダムスロープは暫定的に落としている（31講演/30話者で識別不能。
#      docs/decisions-log.md「RQ1モデルの話者ランダム効果を切片のみに簡略化」）。
#
# 投入データ: 発話フィルタを適用しない全量（docs/decisions-log.md
#   「RQ1は発話フィルタを適用しない全量で行う」）。RQ0比較用の短発話除外は使わない。
#
# 入力: results/rq1_table.tsv（src/evaluate/build_rq1_table.py の出力）
#   1行 = 1境界。boundary_type ∈ {onset, offset, none(=omission), insertion_marker}
#   abs_error_ms は onset/offset のみ値を持つ。

# --- 依存パッケージ（renv.lock 固定対象）---
library(data.table)
library(lme4)        # 誤差モデル lmer
library(ggplot2)

d <- fread("results/rq1_table.tsv")

# ---- 1. 記述統計: 音素クラス × 境界タイプ 別の誤差 ----
# CLAUDE.md §5 の要約統計: 中央値・平均・RMSE・MAD・95パーセンタイル（裾を隠さない）
matched <- d[boundary_type %in% c("onset", "offset") & !is.na(abs_error_ms)]

summ <- matched[, .(
  n        = .N,
  median   = median(abs_error_ms),
  mean     = mean(abs_error_ms),
  rmse     = sqrt(mean(abs_error_ms^2)),
  mad      = median(abs(abs_error_ms - median(abs_error_ms))),
  p95      = quantile(abs_error_ms, 0.95),
  within10 = 100 * mean(abs_error_ms <= 10),
  within20 = 100 * mean(abs_error_ms <= 20),
  within25 = 100 * mean(abs_error_ms <= 25),
  within50 = 100 * mean(abs_error_ms <= 50)
), by = .(phone_class, boundary_type)][order(phone_class, boundary_type)]

fwrite(summ, "results/rq1_summary_by_class.tsv", sep = "\t")
cat("== 音素クラス×境界タイプ 別 誤差要約 ==\n")
print(summ)

# ---- 2. 4類型の生起率（音素クラス別）----
# 各評価単位につき1つの類型。onset/offset展開前の単位数で数えるため、
# boundary_type=onset の行（マッチペア）と none/insertion_marker を使う。
type_tab <- d[boundary_type %in% c("onset", "none", "insertion_marker"),
              .N, by = .(phone_class, error_type)]
type_wide <- dcast(type_tab, phone_class ~ error_type, value.var = "N", fill = 0)
fwrite(type_wide, "results/rq1_error_types_by_class.tsv", sep = "\t")
cat("\n== 音素クラス別 4類型 生起数 ==\n")
print(type_wide)

# ---- 3. 混合モデル: log(誤差) ~ 音素クラス*スタイル + 局所話速 + 境界タイプ + (1|話者)+(1|音素) ----
# 誤差0（完全一致）は log で -Inf になるため、10ms格子の半分（5ms）を下限に加える
# （MFAの量子化限界。docs/decisions-log.md「MFAの境界は10ms格子に量子化される」）。
m <- matched[!is.na(local_speech_rate) & phone_class != "" & speaker_id != ""]
m[, log_err := log(abs_error_ms + 5)]
# 稀少クラス（境界数 < 200 ≒ 単位数 < 100）を other_rare に畳む。
# モデルの音素クラス×スタイル交互作用が n=十数 のクラスで不安定化するのを防ぐ。
# 記述統計（section 1）は畳まず全クラスを個別に報告済み。
rare <- m[, .N, by = phone_class][N < 200, phone_class]
m[phone_class %in% rare, phone_class := "other_rare"]
cat(sprintf("モデルで other_rare に畳んだクラス: %s\n", paste(rare, collapse = ", ")))

# --- 層別 subsampling（183講演版のみ）---
# 誤差つき境界は約247万件。crossed random effects (話者137 × 音素130) つき lmer を
# この規模で回すと収束コスト・メモリが過大。記述統計（section 1, Table 1）は全データを
# 使うが、混合モデルのみ **音素クラスごとに上限 CAP 件で層別無作為抽出**する。
# 稀少クラスは全件残し（推定を痩せさせない）、優勢クラス（vowel_short 等）のみ間引く。
# 固定効果の点推定はこの抽出で実質不変（大クラスは十分な n が残る）。
# 再現性のため set.seed 固定。方法は docs/decisions-log.md に記録。
CAP <- 40000L
set.seed(42)
n_before <- nrow(m)
m <- m[, .SD[if (.N > CAP) sample(.N, CAP) else seq_len(.N)], by = phone_class]
cat(sprintf("lmer用 subsampling: %d -> %d 行（音素クラス上限 CAP=%d, seed=42）\n",
            n_before, nrow(m), CAP))

m[, phone_class := relevel(factor(phone_class), ref = "vowel_short")]
m[, style := factor(style)]
m[, boundary_type := factor(boundary_type)]
m[, ref_phone := factor(ref_label_canonical)]

cat(sprintf("\nモデル投入 n = %d, 話者 = %d, 音素 = %d, 音素クラス = %d\n",
            nrow(m), uniqueN(m$speaker_id), uniqueN(m$ref_phone), uniqueN(m$phone_class)))

fit <- lmer(
  log_err ~ phone_class * style + local_speech_rate + boundary_type +
    (1 | speaker_id) + (1 | ref_phone),
  data = m, REML = TRUE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

cat("\n== RQ1 混合モデル 要約 ==\n")
print(summary(fit))
if (isSingular(fit)) {
  cat("\n[warn] singular fit。ランダム効果構造の見直しが必要な可能性。\n")
}

saveRDS(fit, "results/rq1_model.rds")

# ---- 4. 図: 音素クラス別の誤差分布（中央値・裾込み）----
p <- ggplot(matched, aes(x = reorder(phone_class, abs_error_ms, median),
                         y = abs_error_ms)) +
  geom_boxplot(outlier.size = 0.3, coef = 1.5) +
  coord_flip(ylim = c(0, 100)) +
  labs(x = "音素クラス", y = "境界誤差 (ms)",
       title = "RQ1: 音素クラス別 境界誤差（CSJ 31講演・暫定）") +
  theme_minimal()
ggsave("results/rq1_error_by_class.png", p, width = 8, height = 6, dpi = 150)

cat("\n出力: results/rq1_summary_by_class.tsv, rq1_error_types_by_class.tsv,\n")
cat("      rq1_model.rds, rq1_error_by_class.png\n")
