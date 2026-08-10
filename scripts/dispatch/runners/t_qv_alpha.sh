#!/bin/bash
# Text QV transfer at arbitrary bits, alpha and split.
#
# t_qv.sh hard-codes qv.alpha=1.0 and eval_split=test; the negative-lambda sweep
# needs both parameterized.  Hydra overrides stay inside this script on purpose:
# only bare values cross the ssh boundary, because a nested local -> ssh -> remote
# quoting of an override containing '=' produces "mismatched input '=' expecting <EOF>".
#
# Usage: t_qv_alpha.sh <BITS> <MODEL> <TGT> <SRCS> <SKIP> <ALPHA> <SPLIT> <GPU> <PY> <ROOT> [SRC_MULT] [TGT_MULT]
#
# SRC_MULT / TGT_MULT (canonical form: 1, 0.25, 4) are independent -- that is the
# point of the epoch_mult axis, since donor and receiver share one optim= fragment
# and would otherwise have nowhere to disagree. Both default to 1.
set -eu
BITS=$1; MODEL=$2; TGT=$3; SRC=$4; SKIP=$5; ALPHA=$6; SPLIT=$7; GPU=$8; PY=$9; ROOT=${10}; SMULT=${11:-1}; TMULT=${12:-1}
cd "$ROOT"
$PY code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py \
  model_name="$MODEL" batch_size=32 max_length=128 eval_split="$SPLIT" \
  lr=1e-5 wd=0.1 ls=0.0 max_grad_norm=1.0 gpu="$GPU" \
  "source.dataset_names=[$SRC]" source.seed=2038 source.epoch_mult=$SMULT "target.dataset_names=[$TGT]" target.seed=2038 target.epoch_mult=$TMULT \
  qat.bits="$BITS" qat.granularity=channel "qat.skip_modules=[$SKIP]" qv.alpha="$ALPHA" \
  ptq.bits="$BITS" ptq.granularity=channel "ptq.skip_modules=[$SKIP]" \
  hydra.run.dir="logs/neglambda/bits${BITS}_gpu${GPU}/${TGT}/a${ALPHA}_${SPLIT}"
echo "T_QV_OK bits=$BITS model=$MODEL tgt=$TGT alpha=$ALPHA split=$SPLIT"
