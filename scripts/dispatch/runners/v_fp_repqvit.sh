#!/usr/bin/env bash
# One fp_repqvit baseline cell (rebuttal WP4).
# Usage: v_fp_repqvit.sh <W_BITS> <A_BITS> <DATASET> <GPU> <PY> <ROOT> [EPOCH_MULT]
#
# EPOCH_MULT (canonical form: 1, 0.25, 4) defaults to 1 -- the schedule every
# existing run used -- so omitting it reproduces the previous behaviour.
# Model is fixed to vit_base_patch16_224.orig_in21k (wave-1 scope, same as the
# WP2 gptq runner). All Hydra overrides live here, not in the dispatcher, to
# keep them out of nested quoting (multi-rig-dispatch rule 6).
set -u
WB="$1"; AB="$2"; DS="$3"; GPU="$4"; PY="$5"; ROOT="$6"; MULT="${7:-1}"
cd "$ROOT" || exit 1
"$PY" code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_repqvit.py \
  model_name=vit_base_patch16_224.orig_in21k \
  dataset_name="$DS" \
  seed=2038 \
  gpu="$GPU" \
  epoch_mult="$MULT" \
  batch_size=128 \
  lr=1e-05 \
  wd=0.1 \
  ls=0.0 \
  wl=500 \
  max_grad_norm=1.0 \
  'repqvit.skip_modules=[head]' \
  repqvit.w_bits="$WB" \
  repqvit.a_bits="$AB" \
  repqvit.calib_batch_size=32 \
  hydra.run.dir="logs/dispatch/fp_repqvit_wave1/w${WB}a${AB}_${DS}_gpu${GPU}"
echo "V_FP_REPQVIT_DONE w${WB}a${AB} $DS gpu$GPU"
