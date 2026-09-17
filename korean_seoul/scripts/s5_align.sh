#!/bin/bash
# S5 alignments. Same env/model as .mfa_seoul_v2 (MFA 3.1.1, korean_mfa v3.0), default beams (10/40).
set -e
: "${SEOUL_CORPUS_DIR:?Set SEOUL_CORPUS_DIR to the licensed local Seoul Corpus work directory}"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$HOME/MFA}"
if [ -n "${MFA_BIN_DIR:-}" ]; then export PATH="$MFA_BIN_DIR:$PATH"; fi
cd "$SEOUL_CORPUS_DIR"
LOG=../analysis/revision_2026-09/seoul/work/s5_align_log.txt
: > $LOG
run() {  # corpus dict out
  echo "=== $(date '+%F %T') mfa align --clean -j 6 --output_format long_textgrid $1 $2 korean_mfa $3" | tee -a $LOG
  s=$(date +%s)
  mfa align --clean -j 6 --output_format long_textgrid "$1" "$2" korean_mfa "$3" >> $LOG 2>&1
  echo "=== elapsed $(( $(date +%s) - s )) s; TextGrids $(find "$3" -name '*.TextGrid' | wc -l)" | tee -a $LOG
}
R=.mfa_seoul_prior_rev_equal;   run $R/seoulprioreq_corpus $R/variant_dict.txt $R/aligned
R=.mfa_seoul_prior_rev_oos;     run $R/seoulprioroosA_corpus $R/variant_dict_A.txt $R/aligned_A
                                run $R/seoulprioroosB_corpus $R/variant_dict_B.txt $R/aligned_B
mkdir -p $R/aligned
for d in $R/aligned_A/* $R/aligned_B/*; do [ -d "$d" ] && ln -sfn "../$(basename $(dirname $d))/$(basename $d)" "$R/aligned/$(basename $d)"; done
R=.mfa_seoul_prior_rev_extreme; run $R/seoulpriorext_corpus $R/variant_dict.txt $R/aligned
echo done | tee -a $LOG
