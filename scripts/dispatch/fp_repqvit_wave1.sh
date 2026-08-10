#!/usr/bin/env bash
# Rebuttal WP4 wave 1: fp_repqvit baseline, vit_base_patch16_224.orig_in21k x 22
# datasets at three W/A settings, LOCAL 4090 (single GPU, no ssh).
#
# Settings, and why these three: RepQ-ViT is weight+activation PTQ while the RTN
# and GPTQ columns are weight-only, so no single cell is bit-matched to them.
# W4/A4 is the method as published (the honest "strong competitor" column);
# W3/A8 matches the weight budget of the RTN/GPTQ columns with activations near
# lossless (isolates the weight path); W3/A4 is the intermediate point on the
# W3 activation-sensitivity curve. W3/A3 is deliberately excluded from the wave
# -- it collapses to near chance and would read as a strawman.
#
# Single GPU => sequential queue, no flock needed. Resume guard: a cell whose
# eval_results.json already exists is skipped, so this is safe to re-run and it
# absorbs the EuroSAT cells already run by hand during the smoke.
# Ground truth for completion is the artifact count, never the exit status.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=.venv/bin/python
GPU=0
MODEL_SAN=vit_base_patch16_224_orig_in21k
OPTIM_FRAG='optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128'
# Training-budget multiplier, given in its CANONICAL form (1, 0.25, 4) -- the
# same token mult_path_frag emits. Do not re-derive it in shell: canonicalising
# floats is the one thing src/duration.py exists to keep in a single place, and
# a bash approximation is how "mult=4" and "mult=4.0" become two trees.
MULT=${MULT:-1}
MULT_FRAG="mult=$MULT"
EVAL_ROOT="$ROOT/evaluations/vision/ilharco_timm_supervised/000_baselines/vision/fp_repqvit/$MODEL_SAN"
CK_ROOT="$ROOT/storage/checkpoints/vision/ilharco_timm_supervised/fp/$MODEL_SAN"

SP="$ROOT/logs/dispatch/fp_repqvit_wave1"
mkdir -p "$SP"
DONE="$SP/done.txt"

# Heavy-eval datasets first so the queue tail is cheap.
DATASETS=(ImageNet SUN397 TinyImageNet Food101 PCAM EMNIST Cars CIFAR100 CIFAR10 SVHN FashionMNIST KMNIST MNIST GTSRB Flowers102 STL10 FER2013 RESISC45 OxfordIIITPet DTD RenderedSST2 EuroSAT)
SETTINGS=(4:4 3:8 3:4)

# Hard precondition: every FP checkpoint must exist before launching, otherwise
# the grid silently omits rows.
CK=0
for DS in "${DATASETS[@]}"; do
  ls "$CK_ROOT/$DS/$OPTIM_FRAG/$MULT_FRAG/seed=2038/classifier_epoch_"*.pt >/dev/null 2>&1 && CK=$((CK + 1))
done
if [ "$CK" -ne "${#DATASETS[@]}" ]; then
  echo "PRECONDITION FAILED: $CK/${#DATASETS[@]} FP checkpoints present -- aborting" | tee "$DONE"
  exit 1
fi
echo "precondition OK: $CK/${#DATASETS[@]} FP checkpoints present"

: > "$DONE"
TOTAL=$((${#DATASETS[@]} * ${#SETTINGS[@]}))

for WA in "${SETTINGS[@]}"; do
  WB="${WA%%:*}"; AB="${WA##*:}"
  for DS in "${DATASETS[@]}"; do
    ART="$EVAL_ROOT/$DS/$OPTIM_FRAG/$MULT_FRAG/repqvit=wbits=${WB}_abits=${AB}_skip=head_cbs=32/seed=2038/eval_results.json"
    if [ -f "$ART" ]; then
      echo "w${WB}a${AB}|$DS|SKIP_EXISTING" >> "$DONE"
      continue
    fi
    T0=$(date +%s)
    bash "$ROOT/scripts/dispatch/runners/v_fp_repqvit.sh" "$WB" "$AB" "$DS" "$GPU" "$PY" "$ROOT" >> "$SP/gpu$GPU.log" 2>&1
    RC=$?
    T1=$(date +%s)
    echo "w${WB}a${AB}|$DS|rc=$RC|$((T1 - T0))s" >> "$DONE"
  done
done

N=$(find "$EVAL_ROOT" -name eval_results.json 2>/dev/null | wc -l)
echo "ARTIFACTS|${N:-0}/$TOTAL (plus any pre-existing off-grid settings)" | tee -a "$DONE"
echo "WAVE1 COMPLETE"
