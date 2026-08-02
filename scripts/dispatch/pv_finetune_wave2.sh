#!/usr/bin/env bash
# Phase 008 stage B (wave 2): PV-Tuning finetune sweep,
# vit_base_patch16_224.orig_in21k x 22 datasets x 2 settings, seed 2038,
# 3-bit / channel / skip=[head].
#
#   delta=0.0  tau=0.01
#   delta=0.0  tau=0.10
#
# Why delta=0. Wave 1 used delta=0.9 (AQLM's LLM regime) and produced a frozen
# backbone: the pull-to-grid term is O(delta*scale) ~ 0.045 per step against an
# AdamW step of ~1e-5, so no code ever moved and accuracy landed near the
# no-finetuning baseline (CIFAR10 0.395 vs QAT 0.868; SUN397 0.031, below even
# FP+PTQ 0.054). At delta=0 the buffer accumulates exactly as in QAT, and tau
# is the single knob: tau=1 is bitwise QAT, tau<1 rations the discrete moves.
# Those wave-1 checkpoints are kept deliberately as a negative control.
#
# Design per .claude/skills/multi-rig-dispatch: dynamic pull-queue under flock;
# rssh.sh for every remote call; completion judged by artifact count, never by
# ssh exit status (wave 1 lost Flowers102 to a dropped ssh at epoch 27/147 and
# recorded it as complete -- the runner's artifact check is what caught it);
# PIDs recorded at launch; behemoth via .venv/bin/python, never uv.
set -u

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"

B_ROOT=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
T_ROOT=/home/dansolombrino/PARA/Projects/quantization/qat-transfer
B_PY=.venv/bin/python
T_PY=.venv/bin/python
L_PY="$LOCAL_ROOT/.venv/bin/python"

MODEL_SAN=vit_base_patch16_224_orig_in21k

SP="$LOCAL_ROOT/logs/dispatch/pv_finetune_wave2"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"; PIDS="$SP/pids.txt"

# ImageNet first (~4x every other dataset); the Ilharco schedule normalises the
# rest to ~300k training images each, so the tail is near-uniform and cheap.
DATASETS=(ImageNet SUN397 TinyImageNet Food101 PCAM OxfordIIITPet STL10 Cars \
          RenderedSST2 EMNIST GTSRB FER2013 RESISC45 CIFAR100 CIFAR10 SVHN \
          MNIST FashionMNIST KMNIST EuroSAT DTD Flowers102)
TAUS=(0.01 0.1)
DELTA=0.0

: > "$Q"
for T in "${TAUS[@]}"; do
  for D in "${DATASETS[@]}"; do
    echo "$D|$DELTA|$T" >> "$Q"
  done
done
TOTAL=$(wc -l < "$Q")
: > "$DONE"
: > "$PIDS"

pop() {
  flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"
}

feeder() {
  local LABEL="$1" HOST="$2" GPU="$3" PY="$4" ROOT="$5" NW="$6"
  while true; do
    ITEM="$(pop)"
    [ -z "$ITEM" ] && break
    DS="${ITEM%%|*}"; REST="${ITEM#*|}"; D="${REST%%|*}"; T="${REST##*|}"
    echo "$(date +%H:%M:%S) START $LABEL $DS d=$D t=$T" >> "$DONE"
    if [ -z "$HOST" ]; then
      bash "$LOCAL_ROOT/scripts/dispatch/runners/v_pv.sh" "$DS" "$GPU" "$PY" "$ROOT" "$NW" "$D" "$T"
    else
      $RSSH "$HOST" "bash ~/runners/v_pv.sh '$DS' $GPU '$PY' '$ROOT' '$NW' '$D' '$T'"
    fi
    RC=$?
    echo "$(date +%H:%M:%S) $LABEL|$DS|d=$D|t=$T|ssh_rc=$RC" >> "$DONE"
  done
  echo "$(date +%H:%M:%S) $LABEL|FEEDER_DONE" >> "$DONE"
}

echo "Launching 8 workers over $TOTAL cells (22 datasets x ${#TAUS[@]} taus)" | tee -a "$DONE"

feeder "beh-gpu0" behemoth 0 "$B_PY" "$B_ROOT" 96 >> "$SP/beh0.log" 2>&1 & echo "beh0 $!" >> "$PIDS"
feeder "beh-gpu2" behemoth 2 "$B_PY" "$B_ROOT" 96 >> "$SP/beh2.log" 2>&1 & echo "beh2 $!" >> "$PIDS"
feeder "beh-gpu4" behemoth 4 "$B_PY" "$B_ROOT" 96 >> "$SP/beh4.log" 2>&1 & echo "beh4 $!" >> "$PIDS"
feeder "beh-gpu5" behemoth 5 "$B_PY" "$B_ROOT" 96 >> "$SP/beh5.log" 2>&1 & echo "beh5 $!" >> "$PIDS"
feeder "beh-gpu6" behemoth 6 "$B_PY" "$B_ROOT" 96 >> "$SP/beh6.log" 2>&1 & echo "beh6 $!" >> "$PIDS"
feeder "beh-gpu7" behemoth 7 "$B_PY" "$B_ROOT" 96 >> "$SP/beh7.log" 2>&1 & echo "beh7 $!" >> "$PIDS"
feeder "rig4090"  ""        0 "$L_PY" "$LOCAL_ROOT" "" >> "$SP/rig4090.log" 2>&1 & echo "rig4090 $!" >> "$PIDS"
feeder "rig3090"  rig-3090-ti 0 "$T_PY" "$T_ROOT" 12 >> "$SP/rig3090.log" 2>&1 & echo "rig3090 $!" >> "$PIDS"

wait

echo "=== feeders finished; counting artifacts (ground truth, not ssh status) ===" | tee -a "$DONE"
for T in "${TAUS[@]}"; do
  F="pv=bits=3_gran=channel_skip=head_delta=${DELTA}_tau=${T}_trust=none_pevery=1_temp=0.0"
  NB=$($RSSH behemoth "find $B_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$F*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
  NT=$($RSSH rig-3090-ti "find $T_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$F*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
  NL=$(find "$LOCAL_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN" -path "*$F*" -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l)
  echo "ARTIFACTS|tau=$T|behemoth=${NB:-0} rig3090=${NT:-0} rig4090=${NL:-0} total=$(( ${NB:-0} + ${NT:-0} + ${NL:-0} ))/${#DATASETS[@]}" | tee -a "$DONE"
done
echo "PV FINETUNE WAVE2 COMPLETE" | tee -a "$DONE"
