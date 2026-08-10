#!/bin/bash
# Text QV transfer at arbitrary bits. Usage: t_qv.sh <BITS> <MODEL> <TGT> <SRCS> <SKIP> <GPU> <PY> <ROOT>
set -eu
BITS=$1; MODEL=$2; TGT=$3; SRC=$4; SKIP=$5; GPU=$6; PY=$7; ROOT=$8; SMULT=${9:-1}; TMULT=${10:-1}
cd "$ROOT"
$PY code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py \
  model_name="$MODEL" batch_size=32 max_length=128 eval_split=test \
  lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 gpu="$GPU" \
  "source.dataset_names=[$SRC]" source.seed=2038 source.epoch_mult=$SMULT "target.dataset_names=[$TGT]" target.seed=2038 target.epoch_mult=$TMULT \
  qat.bits="$BITS" qat.granularity=channel "qat.skip_modules=[$SKIP]" qv.alpha=1.0 \
  ptq.bits="$BITS" ptq.granularity=channel "ptq.skip_modules=[$SKIP]"
echo "T_QV_OK bits=$BITS model=$MODEL tgt=$TGT"
