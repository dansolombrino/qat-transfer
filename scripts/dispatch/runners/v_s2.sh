#!/bin/bash
# Vision Stage-2 gate at arbitrary bits.
# Usage: v_s2.sh <BITS> <DATASET> <GPU> <PY> <ROOT> [EPOCH_MULT]
# fp_ptq/pretrained_ptq at 2 bits already exist; only qat and qat_ptq are re-run.
#
# EPOCH_MULT defaults to 1, the schedule every existing run used, so omitting
# it reproduces the previous behaviour exactly. It rides in the shared $C
# override string, so all three evaluations necessarily agree on the budget.
set -eu
BITS=$1; DS=$2; GPU=$3; PY=$4; ROOT=$5; MULT=${6:-1}
cd "$ROOT"
B=code/experiments/vision/ilharco_timm_supervised/000_baselines
C="model_name=vit_base_patch16_224.orig_in21k dataset_name=$DS batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=$GPU epoch_mult=$MULT"
$PY $B/evaluate_qat.py     $C qat.bits=$BITS qat.granularity=channel 'qat.skip_modules=[head]'
$PY $B/evaluate_qat_ptq.py $C qat.bits=$BITS qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=$BITS ptq.granularity=channel 'ptq.skip_modules=[head]'
$PY $B/evaluate_fp_ptq.py  $C ptq.bits=$BITS ptq.granularity=channel 'ptq.skip_modules=[head]'
echo "V_S2_OK bits=$BITS ds=$DS mult=$MULT"
