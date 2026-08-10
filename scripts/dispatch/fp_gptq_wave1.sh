#!/usr/bin/env bash
# Rebuttal WP2 wave 1: fp_gptq baseline, vit_base_patch16_224.orig_in21k x 22
# datasets at 3-bit/channel/skip=[head], behemoth GPUs 0/2/4 ONLY -- WP3's 005
# sweep owns GPUs 5/6/7 (tmux qat_005_full_gpu{5,6,7}); never touch them.
#
# Design per .claude/skills/multi-rig-dispatch: dynamic pull-queue under flock;
# every remote call goes through rssh.sh (behemoth MaxSessions 10); ssh exit
# status is recorded but NEVER trusted as a completion signal -- the final
# artifact count on behemoth is the ground truth. Heavy-eval datasets go first
# so the queue tail is cheap.
set -u

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
PY=.venv/bin/python
MODEL_SAN=vit_base_patch16_224_orig_in21k

SP="$LOCAL_ROOT/logs/dispatch/fp_gptq_wave1"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"

DATASETS=(ImageNet SUN397 TinyImageNet Food101 PCAM EMNIST Cars CIFAR100 CIFAR10 SVHN FashionMNIST KMNIST MNIST GTSRB Flowers102 STL10 FER2013 RESISC45 OxfordIIITPet DTD RenderedSST2 EuroSAT)

# Hard precondition (skill rule 3): every FP checkpoint must exist on behemoth
# before launching -- the eval script would die per-cell otherwise, and a grid
# against an incomplete set silently omits rows.
CK=$($RSSH behemoth "n=0; for d in ${DATASETS[*]}; do ls $B/storage/checkpoints/vision/ilharco_timm_supervised/fp/$MODEL_SAN/\$d/optim=*/mult=*/seed=2038/classifier_epoch_*.pt >/dev/null 2>&1 && n=\$((n+1)); done; echo \$n")
if [ "${CK:-0}" -ne "${#DATASETS[@]}" ]; then
  echo "PRECONDITION FAILED: $CK/${#DATASETS[@]} FP checkpoints on behemoth -- aborting" | tee "$DONE"
  exit 1
fi
echo "precondition OK: $CK/${#DATASETS[@]} FP checkpoints present on behemoth"

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
    $RSSH behemoth "bash ~/runners/v_fp_gptq.sh '$DS' $GPU '$PY' '$B'" >> "$SP/gpu$GPU.log" 2>&1
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

# Ground truth: artifact count on behemoth, scoped to the wave's own subtree.
N=$($RSSH behemoth "find $B/evaluations/vision/ilharco_timm_supervised/000_baselines/vision/fp_gptq/$MODEL_SAN -name eval_results.json 2>/dev/null | wc -l")
echo "ARTIFACTS|${N:-0}/${#DATASETS[@]}" | tee -a "$DONE"
echo "WAVE1 COMPLETE"
