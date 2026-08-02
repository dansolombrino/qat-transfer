#!/usr/bin/env bash
# Phase 008 stage B REPAIR: finish the 4 PV-Tuning cells wave 1 never landed.
#
# Wave 1 (scripts/dispatch/pv_finetune_wave1.sh) drained its queue at 22:23 but
# its parent died before the artifact tally ran, and three behemoth feeders
# exited ssh_rc=255 ("Session open refused by peer"). Ground-truth artifact
# count across the three hosts afterwards: 18/22.
#
#   behemoth (15): CIFAR10 CIFAR100 EMNIST EuroSAT FashionMNIST FER2013 Food101
#                  GTSRB MNIST OxfordIIITPet PCAM RenderedSST2 SUN397 SVHN
#                  TinyImageNet
#   rig-3090-ti (1): Cars
#   rig-4090 (2):    RESISC45 STL10
#
# Note EuroSAT and FashionMNIST ARE present despite their feeders reporting
# ssh_rc=255 -- the runs had finished and saved before the transport dropped.
# That is precisely why completion is counted on the artifact, never on an ssh
# exit status (multi-rig-dispatch rule 2).
#
# Missing (4): ImageNet Flowers102 DTD KMNIST.
#
# One worker per missing dataset, so no queue is needed. ImageNet goes to
# behemoth GPU 0 (~4x every other dataset, and the 288-core host is the only
# one that can keep its dataloader fed). runners/v_pv.sh is idempotent -- it
# exits early if the checkpoint already exists -- so re-running this is safe.
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

SP="$LOCAL_ROOT/logs/dispatch/pv_finetune_repair"
mkdir -p "$SP"
DONE="$SP/done.txt"; PIDS="$SP/pids.txt"
: > "$DONE"; : > "$PIDS"

# $1 label  $2 host ("" = local)  $3 gpu  $4 py  $5 root  $6 workers  $7 dataset
one() {
  local LABEL="$1" HOST="$2" GPU="$3" PY="$4" ROOT="$5" NW="$6" DS="$7"
  echo "$(date +%H:%M:%S) START $LABEL $DS" >> "$DONE"
  if [ -z "$HOST" ]; then
    bash "$LOCAL_ROOT/scripts/dispatch/runners/v_pv.sh" "$DS" "$GPU" "$PY" "$ROOT" "$NW"
  else
    $RSSH "$HOST" "bash ~/runners/v_pv.sh '$DS' $GPU '$PY' '$ROOT' '$NW'"
  fi
  echo "$(date +%H:%M:%S) $LABEL|$DS|ssh_rc=$?" >> "$DONE"
}

echo "Repairing 4 missing PV cells..." | tee -a "$DONE"

one "beh-gpu0" behemoth    0 "$B_PY" "$B_ROOT" 96 ImageNet   >> "$SP/beh0.log"    2>&1 & echo "beh0 $!"    >> "$PIDS"
one "beh-gpu2" behemoth    2 "$B_PY" "$B_ROOT" 96 Flowers102 >> "$SP/beh2.log"    2>&1 & echo "beh2 $!"    >> "$PIDS"
one "rig4090"  ""          0 "$L_PY" "$LOCAL_ROOT" "" DTD    >> "$SP/rig4090.log" 2>&1 & echo "rig4090 $!" >> "$PIDS"
one "rig3090"  rig-3090-ti 0 "$T_PY" "$T_ROOT" 12 KMNIST     >> "$SP/rig3090.log" 2>&1 & echo "rig3090 $!" >> "$PIDS"

wait

echo "=== all workers finished; counting artifacts (ground truth) ===" | tee -a "$DONE"
NB=$($RSSH behemoth "find $B_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
NT=$($RSSH rig-3090-ti "find $T_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
NL=$(find "$LOCAL_ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN" -path "*$PVFRAG*" -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l)
echo "ARTIFACTS|behemoth=${NB:-0} rig3090=${NT:-0} rig4090=${NL:-0} total=$(( ${NB:-0} + ${NT:-0} + ${NL:-0} ))/22" | tee -a "$DONE"
echo "PV FINETUNE REPAIR COMPLETE" | tee -a "$DONE"
