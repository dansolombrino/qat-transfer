#!/usr/bin/env bash
# Phase 008 stage B: PV-Tuning finetune sweep, vit_base_patch16_224.orig_in21k
# x 22 datasets at the reference setting (3-bit / channel / skip=[head],
# delta=0.9, tau=0.01, trust=none, p_every=1, temp=0.0), seed 2038.
#
# Workers (8): behemoth GPUs 0,2,4,5,6,7 + local rig-4090 + rig-3090-ti.
# behemoth GPU 1 is deliberately excluded -- another user's vLLM engine lives
# there.
#
# Design per .claude/skills/multi-rig-dispatch:
#   * dynamic pull-queue under flock; per-dataset cost is roughly uniform
#     (the Ilharco schedule normalises every dataset to ~300k training images)
#     EXCEPT ImageNet at ~1.28M, so ImageNet goes first and the queue tail is
#     cheap.
#   * every remote call goes through rssh.sh (behemoth MaxSessions 10).
#   * ssh exit status is recorded but NEVER trusted as completion -- the
#     runner re-checks for the checkpoint artifact and the final tally counts
#     artifacts on each host.
#   * PIDs are recorded at launch; never pkill -f (rule 4).
#   * behemoth is invoked via .venv/bin/python, never uv (rule 5).
set -u

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"

B_ROOT=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
T_ROOT=/home/dansolombrino/PARA/Projects/quantization/qat-transfer
B_PY=.venv/bin/python          # behemoth: never `uv run` (cu129 pin, rule 5)
T_PY=.venv/bin/python
L_PY="$LOCAL_ROOT/.venv/bin/python"

MODEL_SAN=vit_base_patch16_224_orig_in21k
PVFRAG="pv=bits=3_gran=channel_skip=head_delta=0.9_tau=0.01_trust=none_pevery=1_temp=0.0"

SP="$LOCAL_ROOT/logs/dispatch/pv_finetune_wave1"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"; PIDS="$SP/pids.txt"

# ImageNet first (~4x every other dataset); the rest are near-uniform cost.
DATASETS=(ImageNet SUN397 TinyImageNet Food101 PCAM OxfordIIITPet STL10 Cars \
          RenderedSST2 EMNIST GTSRB FER2013 RESISC45 CIFAR100 CIFAR10 SVHN \
          MNIST FashionMNIST KMNIST EuroSAT DTD Flowers102)

printf '%s\n' "${DATASETS[@]}" > "$Q"
: > "$DONE"
: > "$PIDS"

pop() {
  flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"
}

# $1 label  $2 how-to-run prefix ("" = local)  $3 gpu  $4 py  $5 root  $6 workers
feeder() {
  local LABEL="$1" HOST="$2" GPU="$3" PY="$4" ROOT="$5" NW="$6"
  while true; do
    DS="$(pop)"
    [ -z "$DS" ] && break
    echo "$(date +%H:%M:%S) START $LABEL $DS" >> "$DONE"
    if [ -z "$HOST" ]; then
      bash "$LOCAL_ROOT/scripts/dispatch/runners/v_pv.sh" "$DS" "$GPU" "$PY" "$ROOT" "$NW"
    else
      $RSSH "$HOST" "bash ~/runners/v_pv.sh '$DS' $GPU '$PY' '$ROOT' '$NW'"
    fi
    RC=$?
    echo "$(date +%H:%M:%S) $LABEL|$DS|ssh_rc=$RC" >> "$DONE"
  done
  echo "$(date +%H:%M:%S) $LABEL|FEEDER_DONE" >> "$DONE"
}

echo "Launching 8 workers over ${#DATASETS[@]} datasets..." | tee -a "$DONE"

feeder "beh-gpu0" behemoth 0 "$B_PY" "$B_ROOT" 96 >> "$SP/beh0.log" 2>&1 & echo "beh0 $!" >> "$PIDS"
feeder "beh-gpu2" behemoth 2 "$B_PY" "$B_ROOT" 96 >> "$SP/beh2.log" 2>&1 & echo "beh2 $!" >> "$PIDS"
feeder "beh-gpu4" behemoth 4 "$B_PY" "$B_ROOT" 96 >> "$SP/beh4.log" 2>&1 & echo "beh4 $!" >> "$PIDS"
feeder "beh-gpu5" behemoth 5 "$B_PY" "$B_ROOT" 96 >> "$SP/beh5.log" 2>&1 & echo "beh5 $!" >> "$PIDS"
feeder "beh-gpu6" behemoth 6 "$B_PY" "$B_ROOT" 96 >> "$SP/beh6.log" 2>&1 & echo "beh6 $!" >> "$PIDS"
feeder "beh-gpu7" behemoth 7 "$B_PY" "$B_ROOT" 96 >> "$SP/beh7.log" 2>&1 & echo "beh7 $!" >> "$PIDS"
feeder "rig4090"  ""        0 "$L_PY" "$LOCAL_ROOT" "" >> "$SP/rig4090.log" 2>&1 & echo "rig4090 $!" >> "$PIDS"
feeder "rig3090"  rig-3090-ti 0 "$T_PY" "$T_ROOT" 12 >> "$SP/rig3090.log" 2>&1 & echo "rig3090 $!" >> "$PIDS"

wait

echo "=== all feeders finished; counting artifacts (ground truth) ===" | tee -a "$DONE"
NB=$($RSSH behemoth "find $B_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
NT=$($RSSH rig-3090-ti "find $T_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
NL=$(find "$LOCAL_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN" -path "*$PVFRAG*" -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l)
echo "ARTIFACTS|behemoth=${NB:-0} rig3090=${NT:-0} rig4090=${NL:-0} total=$(( ${NB:-0} + ${NT:-0} + ${NL:-0} ))/${#DATASETS[@]}" | tee -a "$DONE"
echo "PV FINETUNE WAVE1 COMPLETE" | tee -a "$DONE"
