#!/usr/bin/env bash
# Phase 008 stage C: the 22x22 PV-QV transfer grid.
#   vit_base_patch16_224.orig_in21k, seed 2038, 3-bit/channel/skip=[head],
#   delta=0, tau=$TAU, alphas [0.0, 1.0], RTN apply_ptq_ at eval.
#
# Usage: pv_transfer_wave1.sh <TAU> <SPLIT>       e.g. 0.01 test
#
# 506 cells: alpha=1 on all 484 donor-receiver pairs, plus alpha=0 on the 22
# self-pairs (alpha=0 erases the donor, so it is the ptq(FP_target) baseline and
# is donor-independent).
#
# Runs on behemoth's 6 free GPUs only. rig-4090 and rig-3090-ti are excluded
# deliberately: a transfer cell needs the FP checkpoint, the PV checkpoint AND
# the PV sidecar for *every* donor co-located, and qv_transfer returns exit 0
# while silently skipping donors whose checkpoints are absent (rule 3). A host
# with a partial donor set would therefore emit quietly incomplete rows that
# look identical to complete ones. behemoth is also the fastest of the three.
#
# Per multi-rig-dispatch: hard precondition on artifact counts before any work;
# dynamic pull-queue under flock; rssh.sh for every call; completion judged by
# cell count, never ssh status; PIDs recorded.
set -u

TAU="${1:?usage: pv_transfer_wave1.sh <TAU> <SPLIT>}"
SPLIT="${2:?usage: pv_transfer_wave1.sh <TAU> <SPLIT>}"

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"
B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
PY=.venv/bin/python
MODEL_SAN=vit_base_patch16_224_orig_in21k
CK="$B/storage/checkpoints/vision/ilharco_timm_supervised"
PVFRAG="delta=0.0_tau=${TAU}_trust"

SP="$LOCAL_ROOT/logs/dispatch/pv_transfer_t${TAU}_${SPLIT}"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"; PIDS="$SP/pids.txt"

# Heavy receivers first: cost is dominated by the receiver's eval-set size,
# since every cell runs 4 evaluations (2 head variants x pre/post PTQ) over it.
TARGETS=(ImageNet SUN397 TinyImageNet Food101 PCAM EMNIST CIFAR100 CIFAR10 SVHN \
         Cars GTSRB FER2013 RESISC45 FashionMNIST KMNIST MNIST STL10 \
         OxfordIIITPet RenderedSST2 EuroSAT DTD Flowers102)

# --- hard precondition (rule 3): every donor artifact must exist ------------
NFP=$($RSSH behemoth "find $CK/fp/$MODEL_SAN -name 'classifier_epoch_*.pt' -path '*optim=adamw_lr=1e-05*' | sed 's|.*/$MODEL_SAN/||; s|/optim.*||' | sort -u | wc -l")
NPV=$($RSSH behemoth "find $CK/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'classifier_epoch_*.pt' | sed 's|.*/$MODEL_SAN/||; s|/optim.*||' | sort -u | wc -l")
NSD=$($RSSH behemoth "find $CK/pv/$MODEL_SAN -path '*$PVFRAG*' -name 'pv_state_epoch_*.pt' | sed 's|.*/$MODEL_SAN/||; s|/optim.*||' | sort -u | wc -l")
echo "PRECONDITION fp=$NFP pv=$NPV sidecar=$NSD (need ${#TARGETS[@]} each)"
if [ "${NFP:-0}" -ne "${#TARGETS[@]}" ] || [ "${NPV:-0}" -ne "${#TARGETS[@]}" ] || [ "${NSD:-0}" -ne "${#TARGETS[@]}" ]; then
  echo "PRECONDITION FAILED — refusing to launch a grid that would silently omit donors."
  exit 1
fi

printf '%s\n' "${TARGETS[@]}" > "$Q"
: > "$DONE"; : > "$PIDS"

pop() { flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"; }

feeder() {
  local GPU="$1"
  while true; do
    TGT="$(pop)"; [ -z "$TGT" ] && break
    echo "$(date +%H:%M:%S) START gpu$GPU $TGT" >> "$DONE"
    $RSSH behemoth "bash ~/runners/v_qv_pv.sh '$TGT' $GPU '$PY' '$B' '$TAU' '$SPLIT' 96"
    echo "$(date +%H:%M:%S) gpu$GPU|$TGT|ssh_rc=$?" >> "$DONE"
  done
  echo "$(date +%H:%M:%S) gpu$GPU|FEEDER_DONE" >> "$DONE"
}

echo "Launching 6 behemoth workers over ${#TARGETS[@]} receivers (tau=$TAU split=$SPLIT)" | tee -a "$DONE"
for G in 0 2 4 5 6 7; do
  feeder "$G" >> "$SP/gpu$G.log" 2>&1 &
  echo "gpu$G $!" >> "$PIDS"
done
wait

EV="$B/evaluations/vision/ilharco_timm_supervised/008_pv_transfer/vision/qv_transfer_pv"
N=$($RSSH behemoth "find $EV -path '*tau=${TAU}_*' -path '*split=${SPLIT}*' -name eval_results.json | wc -l")
echo "CELLS|${N:-0}/506 (484 cross+self at alpha=1, plus 22 self-pairs at alpha=0)" | tee -a "$DONE"
echo "PV TRANSFER WAVE tau=$TAU split=$SPLIT COMPLETE" | tee -a "$DONE"
