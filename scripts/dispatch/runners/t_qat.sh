#!/bin/bash
# Text QAT finetune at arbitrary bits. Usage: t_qat.sh <BITS> <MODEL> <DS> <SKIP> <GPU> <PY> <ROOT>
set -eu
BITS=$1; MODEL=$2; DS=$3; SKIP=$4; GPU=$5; PY=$6; ROOT=$7; MULT=${8:-1}
cd "$ROOT"
$PY code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py \
  epoch_mult="$MULT" \
  model_name="$MODEL" dataset_name="$DS" seed=2038 gpu="$GPU" \
  batch_size=32 lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 \
  qat.bits="$BITS" qat.granularity=channel "qat.skip_modules=[$SKIP]"
echo "T_QAT_OK bits=$BITS model=$MODEL ds=$DS"
