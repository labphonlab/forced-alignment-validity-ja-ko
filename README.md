# Forced-alignment validity in Japanese and Korean spontaneous speech

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22806127.svg)](https://doi.org/10.5281/zenodo.22806127)

Code, configurations, phone-correspondence tables, and aggregate results supporting the article:

**Accurate boundaries do not validate variant selection: two axes of forced-alignment validity in Japanese and Korean spontaneous speech**

## Contents

- 'src/', 'analysis/japanese/', 'mapping/': Japanese boundary evaluation, correspondence, error decomposition, and vowel-duration analyses.
- 'feature_intervention/': Japanese acoustic-model configurations and analysis of the f0/voicing intervention.
- 'korean_nikl/': scripts for the NIKL paired-transcription study and aggregate results.
- 'korean_seoul/': variant-construction, audit, boundary, prior-sensitivity, and two-axis analyses for the Seoul Corpus, with aggregate outputs.
- 'supplementary_tables/': publication-ready aggregate CSV tables S1-S4.
- 'figures/': vector versions of Figures 1-4.

## Data availability and restrictions

The Corpus of Spontaneous Japanese, NIKL Dialogue Speech Corpus 2025, and Seoul Corpus are distributed by their providers under licences that do not permit redistribution here. This repository therefore contains no corpus audio, transcripts, phone labels, word forms, speaker identifiers, listening-task data, or token-level derivatives. Users must obtain the source corpora from their providers and configure the local input paths expected by the scripts.

Only aggregate results that do not disclose restricted corpus content are included. The public JSUT corpus may be used as a worked example for the Japanese conversion and scoring pipeline; JSUT is not treated as a hand-corrected accuracy reference because its published boundary timings are automatic alignments.

## Environments

Python dependencies are recorded in 'environment.yml' and 'requirements.txt'. R dependencies are recorded in 'renv.lock' and can be restored with 'renv::restore()'.

The analyses used Montreal Forced Aligner versions 3.1.1 and 3.4.1 as documented in the article and 'docs/versions.md'. Model and dictionary versions should be verified before rerunning because pretrained resources can change independently of the code.

Local paths are configured with environment variables rather than author-specific paths. The principal variables are 'MFA_MODEL_DIR', 'MFA_JAPANESE_DICT', 'NIKL_PCM_ROOT', 'NIKL_JSON_ROOT', 'KOREAN_SOUND_CHANGE_ROOT', 'SEOUL_CORPUS_DIR', 'SEOUL_ANALYSIS_DIR', and 'KOREAN_MFA_DICTIONARY'.

## Reproduction scope

The included aggregate tables can be inspected without restricted data. Full regeneration from raw audio requires lawful access to the corresponding corpus and is intentionally not automated as a download. Scripts retain the analysis logic, random seeds, model specifications, audit procedures, and output schemas used for the article.

## Licensing

Code is licensed under the MIT License ('LICENSES/MIT.txt'). Documentation, figures, and aggregate result tables are licensed under CC BY 4.0 ('LICENSES/CC-BY-4.0.txt'). Third-party corpora and pretrained models are not covered by these licences.

## Citation

Please cite the archived version supporting the article as Ishihara (2026), *Forced-alignment validity in Japanese and Korean spontaneous speech: code and aggregate results*, version 1.0.0, Zenodo, https://doi.org/10.5281/zenodo.22806127. The development repository is https://github.com/labphonlab/forced-alignment-validity-ja-ko.
