#!/usr/bin/env bash
# One fp_awq baseline cell (rebuttal WP7).
# Usage: v_fp_awq.sh <BITS> <DATASET> <GPU> <PY> <ROOT> [LIMIT_NUM_BATCHES] [EPOCH_MULT]
#
# EPOCH_MULT (canonical form: 1, 0.25, 4) defaults to 1 -- the schedule every
# existing run used -- so omitting it reproduces the previous behaviour.
#
# Model is fixed to vit_base_patch16_224.orig_in21k (wave-1 scope, matching the
# fp_gptq wave so the two competitor columns are directly comparable). All Hydra
# overrides live here, not in the dispatcher, to keep them out of nested ssh
# quoting (multi-rig-dispatch rule 6).
#
# NO apply_ptq_ runs after AWQ -- AWQ is the quantizer. See code/src/awq.py.
#
# Passing LIMIT_NUM_BATCHES makes this a smoke run: evaluate_fp_awq.py then
# writes under experiment_type=fp_awq_dryrun and reads the fp_dryrun checkpoint
# tree, so a smoke can never contaminate real results -- but it also means a
# smoke needs a dryrun FP checkpoint to exist. Leave it unset for real runs.
set -u
BITS="$1"; DS="$2"; GPU="$3"; PY="$4"; ROOT="$5"; LNB="${6:-}"; MULT="${7:-1}"
cd "$ROOT" || exit 1

EXTRA=()
TAG="wave_b${BITS}"
if [ -n "$LNB" ]; then
  EXTRA+=("limit_num_batches=$LNB")
  TAG="smoke_b${BITS}"
fi

"$PY" code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_awq.py \
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
  'awq.skip_modules=[head]' \
  awq.bits="$BITS" \
  awq.granularity=channel \
  "${EXTRA[@]}" \
  hydra.run.dir="logs/dispatch/fp_awq_${TAG}/${DS}_gpu${GPU}"
RC=$?
echo "V_FP_AWQ_DONE bits$BITS $DS gpu$GPU rc=$RC"
