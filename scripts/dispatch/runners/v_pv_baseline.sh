#!/usr/bin/env bash
# PV baselines for one dataset: evaluate_pv.py and evaluate_pv_ptq.py.
# Usage: v_pv_baseline.sh <DATASET> <GPU> <PY> <ROOT> <TAU> [TORCH_NUM_WORKERS] [EPOCH_MULT]
#
# EPOCH_MULT (canonical form: 1, 0.25, 4) defaults to 1 -- the schedule every
# existing run used -- so omitting it reproduces the previous behaviour.
#
# These fill the `PV` and `PV+PTQ` columns of the 008 heatmap. They are also a
# real check, not just plotting furniture: finetune_pv.py settles the model onto
# the quantization grid before saving, so apply_ptq_ must be a no-op on the
# checkpoint. evaluate_pv_ptq.py records ptq_max_abs_weight_delta, which must
# come back exactly 0.0 -- a non-zero value means the checkpoint did not come
# from the settle path and must not be used to build a QV.
#
# Consequently the two scripts must report the SAME accuracy for a given
# dataset. That equality is the point; it is verified after the wave.
#
# Overrides live here, not in the dispatcher (multi-rig-dispatch rule 6).
set -u
DS="$1"; GPU="$2"; PY="$3"; ROOT="$4"; TAU="$5"; NW="${6:-}"; MULT="${7:-1}"

cd "$ROOT" || exit 1
[ -n "$NW" ] && export TORCH_NUM_WORKERS="$NW"

PV_ARGS=(

  epoch_mult="$MULT"
  model_name=vit_base_patch16_224.orig_in21k
  dataset_name="$DS"
  batch_size=128
  lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0
  seed=2038
  gpu="$GPU"
  pv.bits=3 pv.granularity=channel 'pv.skip_modules=[head]'
  pv.delta_decay=0.0 pv.max_code_change_per_step="$TAU"
  pv.trust_ratio=null pv.p_every=1 pv.temperature=0.0
)

"$PY" code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pv.py \
  "${PV_ARGS[@]}" \
  "hydra.run.dir=logs/dispatch/pvbase_${DS}_t${TAU}_gpu${GPU}_plain"
RC1=$?

"$PY" code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pv_ptq.py \
  "${PV_ARGS[@]}" \
  ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' \
  "hydra.run.dir=logs/dispatch/pvbase_${DS}_t${TAU}_gpu${GPU}_ptq"
RC2=$?

EV="$ROOT/evaluations/vision/ilharco_timm_supervised/000_baselines/vision"
N1=$(find "$EV/pv" -path "*/$DS/*" -path "*tau=${TAU}_*" -name eval_results.json 2>/dev/null | wc -l)
N2=$(find "$EV/pv_ptq" -path "*/$DS/*" -path "*tau=${TAU}_*" -name eval_results.json 2>/dev/null | wc -l)
echo "V_PVBASE_DONE $DS gpu$GPU rc=$RC1/$RC2 pv=$N1 pv_ptq=$N2"
