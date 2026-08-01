#!/bin/bash
# Text Stage-2 gate at arbitrary bits. Usage: t_s2.sh <BITS> <MODEL> <DS> <SKIP> <GPU> <PY> <ROOT>
set -eu
BITS=$1; MODEL=$2; DS=$3; SKIP=$4; GPU=$5; PY=$6; ROOT=$7
cd "$ROOT"
B=code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines
C="model_name=$MODEL dataset_name=$DS batch_size=32 lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=$GPU"
$PY $B/evaluate_qat.py     $C qat.bits=$BITS qat.granularity=channel "qat.skip_modules=[$SKIP]"
$PY $B/evaluate_qat_ptq.py $C qat.bits=$BITS qat.granularity=channel "qat.skip_modules=[$SKIP]" ptq.bits=$BITS ptq.granularity=channel "ptq.skip_modules=[$SKIP]"
$PY $B/evaluate_fp_ptq.py  $C ptq.bits=$BITS ptq.granularity=channel "ptq.skip_modules=[$SKIP]"
echo "T_S2_OK bits=$BITS model=$MODEL ds=$DS"
