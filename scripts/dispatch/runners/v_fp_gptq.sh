#!/usr/bin/env bash
# One fp_gptq baseline cell (rebuttal WP2).
# Usage: v_fp_gptq.sh <BITS> <DATASET> <GPU> <PY> <ROOT>
# Model is fixed to vit_base_patch16_224.orig_in21k (wave-1 scope). All Hydra
# overrides live here, not in the dispatcher, to keep them out of nested ssh
# quoting (multi-rig-dispatch rule 6).
set -u
BITS="$1"; DS="$2"; GPU="$3"; PY="$4"; ROOT="$5"
cd "$ROOT" || exit 1
"$PY" code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_gptq.py \
  model_name=vit_base_patch16_224.orig_in21k \
  dataset_name="$DS" \
  seed=2038 \
  gpu="$GPU" \
  batch_size=128 \
  lr=1e-05 \
  wd=0.1 \
  ls=0.0 \
  wl=500 \
  max_grad_norm=1.0 \
  'gptq.skip_modules=[head]' \
  gptq.bits="$BITS" \
  gptq.granularity=channel \
  hydra.run.dir="logs/dispatch/fp_gptq_wave_b${BITS}/${DS}_gpu${GPU}"
echo "V_FP_GPTQ_DONE bits$BITS $DS gpu$GPU"
