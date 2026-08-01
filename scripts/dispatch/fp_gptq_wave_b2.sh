#!/usr/bin/env bash
# Rebuttal WP2: 2-bit fp_gptq for BOTH wave-1 models in one mixed queue --
# vit_base_patch16_224.orig_in21k x 22 vision datasets and
# google-bert/bert-base-uncased x 11 active text datasets, 2-bit/channel,
# behemoth GPUs 0/2/4 ONLY (WP3's 005 sweep owns 5/6/7; never touch them).
# FP checkpoints are bit-independent, so no new precondition beyond the one
# already verified by the 3-bit waves; artifact counts below filter on the
# gptq=bits=2 fragment so 3-bit results cannot mask a missing 2-bit cell.
# Same design as fp_gptq_wave1.sh (pull-queue + flock + rssh; artifact count
# is the ground truth, never ssh exit status).
set -u

BITS=2
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
PY=.venv/bin/python

SP="$LOCAL_ROOT/logs/dispatch/fp_gptq_wave_b2"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"

V_DATASETS=(ImageNet SUN397 TinyImageNet Food101 PCAM EMNIST Cars CIFAR100 CIFAR10 SVHN FashionMNIST KMNIST MNIST GTSRB Flowers102 STL10 FER2013 RESISC45 OxfordIIITPet DTD RenderedSST2 EuroSAT)
T_DATASETS=(AmazonReviewsClassification IMDB ToxicConversations Banking77 MassiveIntent MassiveScenario MTOPIntent MTOPDomain Emotion TweetSentimentExtraction AmazonCounterfactual)

{
  for d in "${V_DATASETS[@]}"; do echo "v $d"; done
  for d in "${T_DATASETS[@]}"; do echo "t $d"; done
} > "$Q"
: > "$DONE"

pop() {
  flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"
}

feeder() {
  local GPU="$1"
  while true; do
    local ITEM KIND DS RUNNER
    ITEM="$(pop)"
    [ -z "$ITEM" ] && break
    KIND="${ITEM%% *}"; DS="${ITEM#* }"
    RUNNER=v_fp_gptq.sh; [ "$KIND" = t ] && RUNNER=t_fp_gptq.sh
    local T0 T1 RC
    T0=$(date +%s)
    $RSSH behemoth "bash ~/runners/$RUNNER $BITS '$DS' $GPU '$PY' '$B'" >> "$SP/gpu$GPU.log" 2>&1
    RC=$?
    T1=$(date +%s)
    echo "GPU$GPU|$KIND:$DS|rc=$RC|$((T1 - T0))s" >> "$DONE"
  done
  echo "GPU$GPU|FEEDER_DONE" >> "$DONE"
}

feeder 0 & P0=$!
feeder 2 & P2=$!
feeder 4 & P4=$!
echo "feeder pids: $P0 $P2 $P4" | tee "$SP/pids.txt"
wait "$P0" "$P2" "$P4"

NV=$($RSSH behemoth "find $B/evaluations/vision/ilharco_timm_supervised/000_baselines/vision/fp_gptq/vit_base_patch16_224_orig_in21k -path '*gptq=bits=2_*' -name eval_results.json 2>/dev/null | wc -l")
NT=$($RSSH behemoth "find $B/evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text/fp_gptq/google_bert_bert_base_uncased -path '*gptq=bits=2_*' -name eval_results.json 2>/dev/null | wc -l")
echo "ARTIFACTS|vision ${NV:-0}/${#V_DATASETS[@]}|text ${NT:-0}/${#T_DATASETS[@]}" | tee -a "$DONE"
echo "B2 WAVE COMPLETE"
