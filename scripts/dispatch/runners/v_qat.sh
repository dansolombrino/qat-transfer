#!/bin/bash
# Vision QAT finetune at arbitrary bits. Usage: v_qat.sh <BITS> <DATASET> <GPU> <PY> <ROOT>
set -eu
BITS=$1; DS=$2; GPU=$3; PY=$4; ROOT=$5
cd "$ROOT"
$PY code/src/vision/ilharco_timm_supervised/finetune_qat.py \
  model_name=vit_base_patch16_224.orig_in21k dataset_name="$DS" seed=2038 gpu="$GPU" \
  qat.bits="$BITS" qat.granularity=channel 'qat.skip_modules=[head]' \
  hydra.run.dir="logs/src/vision/ilharco_timm_supervised/finetune_qat/bits${BITS}_gpu${GPU}/${DS}"
echo "V_QAT_OK bits=$BITS ds=$DS"
