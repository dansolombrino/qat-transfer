#!/bin/bash
# Vision timm baseline evaluation at arbitrary bits and training budget.
# Usage: v_baseline.sh <VARIANT> <DATASET> <GPU> <PY> <ROOT> <BITS> [EPOCH_MULT] [NW]
#
# VARIANT is one of fp, fp_ptq, qat, qat_ptq, pretrained, pretrained_ptq -- the
# six the heatmap and win/loss scripts read. They are needed once per (dataset,
# budget): the plotting scripts locate them through paths that carry `mult=`, so
# a new budget without new baselines produces empty figures rather than an error.
#
# Every override lives here rather than in the caller. Overrides that cross two
# levels of shell quoting (local -> ssh -> remote) fail to parse, so the caller
# passes values only.
set -u
VARIANT=$1; DS=$2; GPU=$3; PY=$4; ROOT=$5; BITS=$6; MULT=${7:-1.0}; NW=${8:-}

cd "$ROOT" || exit 1
[ -n "$NW" ] && export TORCH_NUM_WORKERS="$NW"

BASE="code/experiments/vision/ilharco_timm_supervised/000_baselines"
COMMON=(
  model_name=vit_base_patch16_224.orig_in21k
  dataset_name="$DS"
  seed=2038
  gpu="$GPU"
  batch_size=128
  lr=1e-5
  wd=0.1
  ls=0.0
  wl=500
  max_grad_norm=1.0
  epoch_mult="$MULT"
)
QATG=(qat.bits="$BITS" qat.granularity=channel 'qat.skip_modules=[head]')
PTQG=(ptq.bits="$BITS" ptq.granularity=channel 'ptq.skip_modules=[head]')

# No '=' inside a Hydra override *value*, so the run dir uses mult0.25 not mult=0.25.
RUNDIR="logs/experiments/vision/ilharco_timm_supervised/000_baselines/${VARIANT}/gpu${GPU}/${DS}/mult${MULT}"

case "$VARIANT" in
  fp|pretrained)              EXTRA=() ;;
  fp_ptq)                     EXTRA=("${PTQG[@]}") ;;
  qat)                        EXTRA=("${QATG[@]}") ;;
  qat_ptq|pretrained_ptq)     EXTRA=("${QATG[@]}" "${PTQG[@]}") ;;
  *) echo "V_BL_FAIL unknown variant $VARIANT" >&2; exit 2 ;;
esac

# Idempotency and success are both judged on the artifact, through the same
# module the writer uses -- never by exit status, which a dropped ssh corrupts.
CHECK=(scripts/dispatch/budget_axis/bx_check.py baseline "$VARIANT" "$DS" "$MULT")
if "$PY" "${CHECK[@]}"; then
  echo "V_BL_SKIP variant=$VARIANT ds=$DS mult=$MULT (already present)"
  exit 0
fi

"$PY" "$BASE/evaluate_${VARIANT}.py" "${COMMON[@]}" "${EXTRA[@]}" \
  hydra.run.dir="$RUNDIR"

if "$PY" "${CHECK[@]}"; then
  echo "V_BL_OK variant=$VARIANT ds=$DS mult=$MULT"
else
  echo "V_BL_FAIL variant=$VARIANT ds=$DS mult=$MULT (no eval_results.json)" >&2
  exit 1
fi
