#!/usr/bin/env bash
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1; NAME=$2; MODEL=$3; DS=$4
export CUDA_VISIBLE_DEVICES=$PHYS
LOG=/data02/users/lzhou/qat-transfer/logs/latest
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
say "START $NAME (gpu $PHYS, $MODEL $DS rtn+hawq avg=3.5)"
./.venv/bin/python code/experiments/text/sentence_transformers/004_input_fragility/gap_allocation.py \
  model_name="$MODEL" dataset_name="$DS" gpu=0 ptq.bits=3 ptq.method=rtn add_hawq=true \
  avg_bits=3.5 seed=2038 max_docs=8000 max_queries=1000 n_random=${5:-3} \
  calib_queries=128 calib_docs=512 > "$LOG/$NAME.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say "DONE  $NAME (gpu$PHYS)"; echo "$NAME OK" >> "$LOG/STATUS"
else say "FAIL  $NAME (gpu$PHYS) rc=$rc"; echo "$NAME FAIL rc=$rc" >> "$LOG/STATUS"; fi
