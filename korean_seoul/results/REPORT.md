# Seoul Corpus reanalysis for the 2026-09 revision (S1–S6)

- **Date:** 2026-09-16.
- **Outputs:** aggregate results are in this folder; token-level tables are in `work/`, which git ignores.
- **Bootstrap:** B = 2,000, seed 20260916.

## Environment and commands

**Inputs**
- Candidate pool: `seoul_corpus/seoul_candidates.csv` (69,542 environments).
- Natural sample: `.mfa_seoul_v2` (6,911 utterances, reduced to 5,716 audited candidates).

**Software**
- MFA 3.1.1 (env `mfa`, kalpy 0.6.4) with korean_mfa acoustic model v3.0 (GMM-HMM) and G2P v3.0.
- Beams were never set, so MFA's defaults applied: beam 10, retry_beam 40 (`alignment/mixins.py` L74–75).
- `MFA_ROOT_DIR=$HOME/MFA`.

**Original v2 commands** (from `~/MFA/_moved_from_dropbox_2026-09-15/command_history.yaml`)
```
mfa g2p .mfa_seoul_v2/g2p_wordlist.txt korean_mfa .mfa_seoul_v2/g2p_out.dict --num_pronunciations 1
mfa align --clean -j 8 --output_format long_textgrid .mfa_seoul_v2/corpus .mfa_seoul_v2/variant_dict.txt korean_mfa .mfa_seoul_v2/aligned   # 88 s
```

**New alignment runs.** Each used `-j 6` and took 43–72 s.

| Run directory | Purpose |
|---|---|
| `.mfa_seoul_cc_rev` | case-control sample |
| `.mfa_seoul_prior_rev_equal` | equal priors, repeated |
| `.mfa_seoul_prior_rev_oos` (A/B) | priors estimated on the other half of speakers |
| `.mfa_seoul_prior_rev_extreme` | extreme priors |

The exact commands are in `~/MFA/command_history.yaml`.

**How the pronunciation-probability column works in MFA 3.1.1**
- The arc cost is |ln p|, with no renormalisation to the per-word maximum.
- p is floored at 0.01.
- Values are parsed only when they match `\d+\.\d+`.
- Sources: `utils.py::parse_dictionary_file` and `kalpy/fstext/lexicon.py`.

**Scripts** (all in `scripts/`): `rev_common.py`, `s1_candidate_table.py`, `s2_dissociation.py`, `s2b_glmer.R`, `s3_prevalence_metrics.py`, `s4_prepare_cc.py`, `s5_prepare_priors.py`, `s5_align.sh`, `s5_score.py`, `s6_tier_consistency.py`.

## S1 Reproduction

- All 16 manuscript quantities reproduce exactly (`s1_verification.tsv`).
- The identity-safe process-site analysis uses n = 2,564 of the 2,645 identity-safe candidates (1,287 correct, 1,277 wrong).
- The identity-safe subset contains only aspiration (417), liaison (1,022) and tensification (1,205), plus 1 obstruent-nasalization candidate.

## S2 Same-token dissociation (identity-safe subset primary)

**(a) AUC: does boundary error flag wrong selections?**

The two-way CI resamples speakers and word types.

| Subset / measure | n (wrong) | AUC | Speaker CI | Two-way CI |
|---|---|---|---|---|
| Identity-safe / site | 2,564 (1,277) | .532 | [.507, .558] | [.485, .580] |
| Identity-safe / word | 2,636 (1,307) | .502 | [.482, .523] | [.460, .546] |
| Full / site | 5,414 (2,297) | .496 | [.482, .512] | [.462, .531] |
| Full / word | 5,538 (2,367) | .498 | [.484, .515] | [.467, .529] |

Within-process AUC (identity-safe, site):
- Pooled: .547, speaker CI [.519, .575], two-way CI [.501, .597].
- By process: liaison .599 [.567, .634] (two-way [.521, .662]), aspiration .546, tensification .510.
- In the full sample at the site, liquidization reverses to .385 [.326, .452].

**(b) glmer: `correct ~ z(log(dev+1)) * process + (1|speaker) + (1|word_type)`**

All fits converged.

| Subset / measure | Slope per SD | Interaction test |
|---|---|---|
| Identity-safe / site | −0.207 [−0.342, −0.072] (OR 0.81, p = .003) | χ²(2) = 2.82, p = .245 |
| Identity-safe / word | −0.112 [−0.242, 0.017] | — |
| Full / site | −0.102 [−0.183, −0.020] | χ²(5) = 16.5, p = .0055 (liquidization slope +0.38) |
| Full / word | −0.086 [−0.164, −0.007] | — |

Random-effect SDs: speaker ≈ 0.16–0.20; word type ≈ 1.7–2.1 logits.

**(c) Cross-validation**

The table gives the mean AUC within each held-out speaker (leave-one-speaker-out).

| Subset / measure | Process | Boundary | Process + boundary | Increment |
|---|---|---|---|---|
| Identity-safe / site | .651 | .531 | .668 | +.017 [+.006, +.028] |
| Identity-safe / word | .653 | .463 | .661 | +.008 |
| Full / site | .675 | .504 | .681 | +.006 |
| Full / word | .673 | .486 | .676 | +.003 |

**(d) Median difference, wrong − correct (ms)**

| Subset / measure | Difference | Speaker CI | Two-way CI |
|---|---|---|---|
| Identity-safe / site | +0.56 | [+0.02, +1.39] | [−0.33, +2.20] |
| Identity-safe / word | +0.32 | [−0.28, +0.82] | [−0.82, +1.16] |
| Full / site | −0.09 | [−0.42, +0.39] | [−1.08, +0.73] |
| Full / word | −0.01 | [−0.35, +0.43] | [−0.70, +0.75] |

## S3 Natural sample: metrics with speaker-cluster CIs (`s3_natural_metrics.tsv`)

| Process | n | Neg. | Sens. | Spec. | BA | MCC |
|---|---|---|---|---|---|---|
| Aspiration | 1,228 | 77 | .624 | .234 | .429 [.371, .492] | −.072 |
| Liaison | 1,135 | 15 | .368 | .800 | .584 [.481, .688] | .040 |
| Liquidization | 483 | 21 | .842 | .857 | .850 [.774, .923] | .365 |
| Nasalization (liquid) | 185 | 23 | .914 | .261 | .587 [.483, .705] | .185 |
| Nasalization (obstr.) | 1,267 | 3 | .712 | 1 | .856 | .076 |
| Tensification | 1,418 | 22 | .466 | .591 | .528 [.380, .637] | .014 |
| **All** | 5,716 | 161 | .579 [.561, .596] | .435 [.336, .544] | .507 [.459, .556] | .005 [−.027, .039] |
| All, pool-weighted | | | .440 | .591 | .516 [.453, .573] | .008 [−.023, .040] |

## S4 Case-control sample (new alignment; `s4_casecontrol_metrics.tsv`)

**Draw**
- The pool excluded all utterances in the natural sample. The natural draw was replayed and reproduced exactly.
- Eligible negatives: aspiration 74, liaison 440, tensification 136, obstruent nasalization 4, liquidization 0, liquid nasalization 0.
- Sampling: up to 150 negatives per process, each matched with the same number of positives, one per utterance. This gave 708 candidates, of which 611 remained after the audit.

**Results**

| Process | Applied / not | Under : over | Sens. | Spec. | BA | MCC |
|---|---|---|---|---|---|---|
| Aspiration | 50 / 69 | 20:53 | .600 | .232 | .416 [.325, .501] | −.181 |
| Liaison | 116 / 118 | 65:69 | .440 | .415 | .428 [.371, .494] | −.145 |
| Tensification | 128 / 122 | 57:52 | .555 | .574 | .564 [.509, .625] | .128 [.018, .251] |
| Nasalization (obstr.) | 4 / 4 | 0:0 | 1 | 1 | 1 | — |
| **All** | 298 / 313 | 142:174 | .524 [.468, .579] | .444 [.388, .512] | .484 [.445, .529] | −.033 [−.111, .058] |

## S5 Prior sensitivity (new alignments; `s5_prior_metrics.tsv`, `s5_prior_flips.tsv`, `s5_prior_split.json`)

**Out-of-sample priors**
- The 40 speakers were split into halves A and B with seed 20260916.
- p for each process was estimated on one half and applied to the other.

| p | Aspiration | Liaison | Liquidization | Nasalization (liquid) | Nasalization (obstr.) | Tensification |
|---|---|---|---|---|---|---|
| Estimated on A | .957 | .982 | .964 | .837 | .998 | .984 |
| Estimated on B | .955 | .978 | .953 | .897 | .996 | .983 |

The extreme setting used 0.9 / 0.1.

| Setting | n | Agreement | Under : over | Sens. | Spec. | BA | MCC |
|---|---|---|---|---|---|---|---|
| Equal (v2) | 5,716 | .575 | 2,338:91 | .579 | .435 | .507 | .005 |
| Equal, re-run | 5,717 | .574 | 2,347:89 | .578 | .447 | .512 | .008 |
| Out-of-sample | 5,751 | .923 | 312:132 | .944 | .185 [.122, .257] | .565 [.533, .600] | .091 [.047, .137] |
| Extreme | 5,742 | .887 | 523:126 | .906 | .222 | .564 [.531, .599] | .072 |

**Out-of-sample priors, by process (BA)**
- Aspiration .468 (specificity .026)
- Liaison .655
- Liquidization .884
- Nasalization (liquid) .562
- Nasalization (obstr.) .811
- Tensification .465 (specificity .000)

**Decisions changed relative to v2**
- Equal-prior re-run: 40 of 5,716 (0.7%) changed. v2 is not bit-reproducible (it used `-j 8`, the re-run `-j 6`).
- Out-of-sample priors: 36.1% changed, all toward "applied" (liaison 60.6%, tensification 46.6%).
- Extreme priors: 32.4% changed.

## S6 Seoul Corpus annotation workflow

According to Yun et al. (2015) and the corpus manual, the annotation was a single pipeline:
- A pronounced-form hangul transcription went to an HTK segmenter.
- Nine labellers then corrected the phone symbols and boundaries. These corrections also fixed listening errors in the pronounced transcription.
- The manual (p.28) states that all tiers are synced.

The documents do not say how symbol corrections were propagated between tiers.

Tier consistency (`s6_tier_consistency.tsv`):
- The phone tier and the romanised pronounced form agree symbol for symbol in 99.98% of the 69,542 candidates, after mapping EE/YE/wE/WE.
- The phone tier matches the orthographic form in only 23%.

**Conclusion:** the categorical and temporal references come from one human labelling.

## Manuscript implications

1. Report two-way cluster CIs.
   - The identity-safe site difference is +0.56, with a two-way CI of [−0.33, +2.20].
   - The identity-safe site AUC is .532 [.485, .580].
2. Replace "AUC ≤ .53".
   - Within-process AUC is .547, and .599 for liaison.
   - The conditional association is OR 0.81 per SD.
   - Adding boundary error to process raises cross-validated AUC by only +.017.
3. The under-application asymmetry depends on prevalence. In the case-control sample it is 142:174, with BA .484.
4. Priors dominate the decisions.
   - Out-of-sample priors change 36% of decisions and raise agreement to .923.
   - BA rises only to .565, and specificity falls to .185.
5. Methods: add the probability semantics, the commands, and the default beams (10/40).
6. The two Seoul references are one annotation, so they are procedurally dependent.
