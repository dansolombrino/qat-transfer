#!/bin/bash
# Vision QV transfer at arbitrary bits. Usage: v_qv.sh <BITS> <TARGET> <SOURCES> <GPU> <PY> <ROOT>
set -eu
BITS=$1; TGT=$2; SRC=$3; GPU=$4; PY=$5; ROOT=$6
cd "$ROOT"
$PY code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py \
  model_name=vit_base_patch16_224.orig_in21k batch_size=128 eval_split=test \
  lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu="$GPU" \
  "source.dataset_names=[$SRC]" source.seed=2038 "target.dataset_names=[$TGT]" target.seed=2038 \
  qat.bits="$BITS" qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.0 \
  ptq.bits="$BITS" ptq.granularity=channel 'ptq.skip_modules=[head]'
echo "V_QV_OK bits=$BITS tgt=$TGT"
