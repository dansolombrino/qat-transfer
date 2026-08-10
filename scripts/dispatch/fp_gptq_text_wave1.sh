#!/usr/bin/env bash
# Rebuttal WP2 text wave: fp_gptq baseline, google-bert/bert-base-uncased x 12
# datasets at 3-bit/channel/skip=[classifier], behemoth GPUs 0/2/4 ONLY --
# WP3's 005 sweep owns GPUs 5/6/7; never touch them.
# Same design as fp_gptq_wave1.sh (pull-queue + flock + rssh; artifact count is
# the ground truth, never ssh exit status). Heavy-eval datasets first.
set -u

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
PY=.venv/bin/python
MODEL_SAN=google_bert_bert_base_uncased

SP="$LOCAL_ROOT/logs/dispatch/fp_gptq_text_wave1"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"

# AmazonPolarity has an FP checkpoint but was retired from the text task set
# (commented out of DATASET_NAME_TO_EPOCHS) -- it has no RTN twin either. 11
# active datasets, matching the 11 existing fp_ptq baselines.
DATASETS=(AmazonReviewsClassification IMDB ToxicConversations Banking77 MassiveIntent MassiveScenario MTOPIntent MTOPDomain Emotion TweetSentimentExtraction AmazonCounterfactual)

# Hard precondition (skill rule 3): every FP backbone checkpoint must exist on
# behemoth before launching, at the canonical bs=32/ml=128 fragment.
CK=$($RSSH behemoth "n=0; for d in ${DATASETS[*]}; do ls $B/storage/checkpoints/text/ilharco_automodelforsequenceclassification/fp/$MODEL_SAN/\$d/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128/mult=*/seed=2038/backbone_epoch_*.pt >/dev/null 2>&1 && n=\$((n+1)); done; echo \$n")
if [ "${CK:-0}" -ne "${#DATASETS[@]}" ]; then
  echo "PRECONDITION FAILED: $CK/${#DATASETS[@]} FP backbones on behemoth -- aborting" | tee "$DONE"
  exit 1
fi
echo "precondition OK: $CK/${#DATASETS[@]} FP backbones present on behemoth"

printf '%s\n' "${DATASETS[@]}" > "$Q"
: > "$DONE"

pop() {
  flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"
}

feeder() {
  local GPU="$1"
  while true; do
    local DS
    DS="$(pop)"
    [ -z "$DS" ] && break
    local T0 T1 RC
    T0=$(date +%s)
    $RSSH behemoth "bash ~/runners/t_fp_gptq.sh '$DS' $GPU '$PY' '$B'" >> "$SP/gpu$GPU.log" 2>&1
    RC=$?
    T1=$(date +%s)
    echo "GPU$GPU|$DS|rc=$RC|$((T1 - T0))s" >> "$DONE"
  done
  echo "GPU$GPU|FEEDER_DONE" >> "$DONE"
}

feeder 0 & P0=$!
feeder 2 & P2=$!
feeder 4 & P4=$!
echo "feeder pids: $P0 $P2 $P4" | tee "$SP/pids.txt"
wait "$P0" "$P2" "$P4"

N=$($RSSH behemoth "find $B/evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text/fp_gptq/$MODEL_SAN -name eval_results.json 2>/dev/null | wc -l")
echo "ARTIFACTS|${N:-0}/${#DATASETS[@]}" | tee -a "$DONE"
echo "TEXT WAVE1 COMPLETE"
