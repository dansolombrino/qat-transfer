#!/usr/bin/env bash
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1; NAME=$2; DS=$3; AVG=${4:-3.5}; SEED=${5:-2038}; METH=${6:-rtn}; HAWQ=${7:-false}
export CUDA_VISIBLE_DEVICES=$PHYS
LOG=/data02/users/lzhou/qat-transfer/logs/latest
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
say "START $NAME (gpu $PHYS, Qwen3-0.6B $DS text-clsf $METH avg=$AVG seed=$SEED)"
./.venv/bin/python code/experiments/text/ilharco_automodelforsequenceclassification/004_input_fragility/gap_allocation_clsf.py \
  model_name=Qwen/Qwen3-Embedding-0.6B dataset_name="$DS" \
  batch_size=32 lr=1e-05 wd=0.1 ls=0.0 max_grad_norm=1.0 max_length=128 seed=$SEED gpu=0 \
  avg_bits=$AVG ptq.method=$METH add_hawq=$HAWQ calib_batches=8 > "$LOG/$NAME.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say "DONE  $NAME (gpu$PHYS)"; echo "$NAME OK" >> "$LOG/STATUS"
else say "FAIL  $NAME (gpu$PHYS) rc=$rc"; echo "$NAME FAIL rc=$rc" >> "$LOG/STATUS"; fi
