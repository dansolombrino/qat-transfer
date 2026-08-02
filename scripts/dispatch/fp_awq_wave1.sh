#!/usr/bin/env bash
# Rebuttal WP7 stage A: AWQ(FP) baseline sweep, vit_base_patch16_224.orig_in21k
# x 22 datasets, seed 2038, 3-bit / channel / skip=[head], ncal=4, ngrid=20,
# clip=true. Same scope as the fp_gptq wave, so the two competitor columns of
# the Task-1 table are directly comparable.
#
# Workers: behemoth GPUs 4,5,6,7 ONLY. GPUs 0 and 2, the local rig-4090 and
# rig-3090-ti are deliberately excluded -- another session's PV-Tuning wave is
# running there and this must not contend with it. GPU 1 holds another user's
# vLLM engine, as always. Adding more workers later is a one-line change; with
# 4 GPUs and ~45 s/dataset the whole wave is a few minutes anyway.
#
# Design per .claude/skills/multi-rig-dispatch:
#   * dynamic pull-queue under flock; ImageNet and SUN397 first (largest eval
#     sets -- AWQ's own cost is dataset-independent, the *evaluation* is not).
#   * every remote call goes through rssh.sh (behemoth MaxSessions 10).
#   * ssh exit status is recorded but NEVER trusted as completion -- the final
#     tally counts eval_results.json artifacts on the host.
#   * PIDs are recorded at launch; never pkill -f (rule 4).
#   * behemoth is invoked via .venv/bin/python, never uv (rule 5).
#   * Hydra overrides live in runners/v_fp_awq.sh, not here (rule 6).
#
# Smoke already passed on GPU 4: EuroSAT AWQ(FP)=0.9574 vs recorded RTN
# fp_ptq=0.3007 and fp_gptq=0.9752, FP=0.9826 -- the sanity anchor of
# plans/rebuttal_competitor_ptq.md §5. EuroSAT is left in the queue; the runner
# simply recomputes it (evaluate_fp_awq.py has no skip_existing).
set -u

LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"

B=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
PY=.venv/bin/python
BITS="${1:-3}"

MODEL_SAN=vit_base_patch16_224_orig_in21k
GPUS=(4 5 6 7)

SP="$LOCAL_ROOT/logs/dispatch/fp_awq_wave_b${BITS}"
mkdir -p "$SP"
Q="$SP/q.txt"; DONE="$SP/done.txt"; LOCK="$SP/q.lock"; PIDS="$SP/pids.txt"

DATASETS=(ImageNet SUN397 TinyImageNet Food101 PCAM OxfordIIITPet STL10 Cars \
          RenderedSST2 EMNIST GTSRB FER2013 RESISC45 CIFAR100 CIFAR10 SVHN \
          MNIST FashionMNIST KMNIST EuroSAT DTD Flowers102)

# Precondition: the FP checkpoints this wave quantizes must already be there.
# Failing loudly up front beats 22 per-dataset warnings.
CK=$($RSSH behemoth "find $B/storage/checkpoints/vision/ilharco_timm_supervised/fp/$MODEL_SAN -name 'classifier_epoch_*.pt' 2>/dev/null | wc -l")
if [ "${CK:-0}" -lt "${#DATASETS[@]}" ]; then
  echo "PRECONDITION FAILED: ${CK:-0}/${#DATASETS[@]} FP checkpoints on behemoth -- aborting" | tee "$DONE"
  exit 1
fi
echo "precondition OK: $CK FP checkpoints present on behemoth"

printf '%s\n' "${DATASETS[@]}" > "$Q"
: > "$DONE"; : > "$PIDS"

pop() {
  flock "$LOCK" bash -c "head -n1 '$Q' 2>/dev/null; sed -i '1d' '$Q' 2>/dev/null"
}

feeder() {
  local GPU="$1"
  while true; do
    local DS T0 T1 RC
    DS="$(pop)"
    [ -z "$DS" ] && break
    T0=$(date +%s)
    $RSSH behemoth "bash ~/runners/v_fp_awq.sh $BITS '$DS' $GPU '$PY' '$B'" >> "$SP/gpu$GPU.log" 2>&1
    RC=$?
    T1=$(date +%s)
    echo "$(date +%H:%M:%S) GPU$GPU|$DS|rc=$RC|$((T1 - T0))s" >> "$DONE"
  done
  echo "$(date +%H:%M:%S) GPU$GPU|FEEDER_DONE" >> "$DONE"
}

echo "Launching ${#GPUS[@]} workers over ${#DATASETS[@]} datasets at ${BITS}-bit..." | tee -a "$DONE"
for g in "${GPUS[@]}"; do
  feeder "$g" & echo "gpu$g $!" >> "$PIDS"
done
wait

# Ground truth: artifact count on behemoth, scoped to this wave's own subtree.
FRAG="awq=bits=${BITS}_gran=channel_skip=head_ncal=4_ngrid=20_clip=True"
N=$($RSSH behemoth "find $B/evaluations/vision/ilharco_timm_supervised/000_baselines/vision/fp_awq/$MODEL_SAN -path '*${FRAG}*' -name eval_results.json 2>/dev/null | wc -l")
echo "ARTIFACTS|${N:-0}/${#DATASETS[@]}" | tee -a "$DONE"
echo "FP_AWQ WAVE1 (${BITS}-bit) COMPLETE" | tee -a "$DONE"
