#!/usr/bin/env bash
# Phase 008 stage D: the PV baselines that fill the heatmap's two blank columns.
#   evaluate_pv.py + evaluate_pv_ptq.py x 22 datasets, delta=0, tau=$TAU,
#   vit_base_patch16_224.orig_in21k, seed 2038, 3-bit/channel/skip=[head].
#
# Usage: pv_baselines_wave1.sh <TAU>        e.g. 0.01
#
# behemoth's 6 free GPUs; the PV checkpoints all live there. Cheap compared to
# the transfer grid: one checkpoint load and one test pass per script.
set -u

TAU="${1:?usage: pv_baselines_wave1.sh <TAU>}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
PY=.venv/bin/python
MODEL_SAN=vit_base_patch16_224_orig_in21k
PVFRAG="delta=0.0_tau=${TAU}_trust"

SP="$LOCAL_ROOT/logs/dispatch/pv_baselines_t${TAU}"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"; PIDS="$SP/pids.txt"

DATASETS=(ImageNet SUN397 PCAM Food101 EMNIST TinyImageNet CIFAR100 CIFAR10 SVHN \
          Cars GTSRB FER2013 RESISC45 FashionMNIST KMNIST MNIST STL10 \
          OxfordIIITPet RenderedSST2 EuroSAT DTD Flowers102)

# Hard precondition: every PV checkpoint must exist, or a dataset silently
# yields no baseline and the column stays half-blank (rule 3's shape).
NPV=$($RSSH behemoth "find $B/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'classifier_epoch_*.pt' | sed 's|.*/$MODEL_SAN/||; s|/optim.*||' | sort -u | wc -l")
echo "PRECONDITION pv checkpoints=$NPV (need ${#DATASETS[@]})"
if [ "${NPV:-0}" -ne "${#DATASETS[@]}" ]; then
  echo "PRECONDITION FAILED — refusing to launch."; exit 1
fi

printf '%s\n' "${DATASETS[@]}" > "$Q"
: > "$DONE"; : > "$PIDS"

pop() { flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"; }

feeder() {
  local GPU="$1"
  while true; do
    DS="$(pop)"; [ -z "$DS" ] && break
    echo "$(date +%H:%M:%S) START gpu$GPU $DS" >> "$DONE"
    $RSSH behemoth "bash ~/runners/v_pv_baseline.sh '$DS' $GPU '$PY' '$B' '$TAU' 96"
    echo "$(date +%H:%M:%S) gpu$GPU|$DS|ssh_rc=$?" >> "$DONE"
  done
  echo "$(date +%H:%M:%S) gpu$GPU|FEEDER_DONE" >> "$DONE"
}

echo "Launching 6 behemoth workers over ${#DATASETS[@]} datasets (tau=$TAU)" | tee -a "$DONE"
for G in 0 2 4 5 6 7; do
  feeder "$G" >> "$SP/gpu$G.log" 2>&1 &
  echo "gpu$G $!" >> "$PIDS"
done
wait

EV="$B/evaluations/vision/ilharco_timm_supervised/000_baselines/vision"
N1=$($RSSH behemoth "find $EV/pv/$MODEL_SAN -path '*tau=${TAU}_*' -name eval_results.json 2>/dev/null | wc -l")
N2=$($RSSH behemoth "find $EV/pv_ptq/$MODEL_SAN -path '*tau=${TAU}_*' -name eval_results.json 2>/dev/null | wc -l")
echo "ARTIFACTS|pv=${N1:-0}/${#DATASETS[@]} pv_ptq=${N2:-0}/${#DATASETS[@]}" | tee -a "$DONE"
echo "PV BASELINES WAVE tau=$TAU COMPLETE" | tee -a "$DONE"
