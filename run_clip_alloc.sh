#!/usr/bin/env bash
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1; NAME=$2; MODEL=$3; PRE=$4; METH=$5; SEED=${6:-2038}; AVG=${7:-3.5}
export CUDA_VISIBLE_DEVICES=$PHYS
LOG=/data02/users/lzhou/qat-transfer/logs/latest
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
say "START $NAME (gpu $PHYS, $MODEL/$PRE clip $METH avg=$AVG seed=$SEED)"
./.venv/bin/python code/experiments/vision/ilharco_hf_clip/004_input_fragility/gap_allocation_clip.py \
  model_name="$MODEL" pretrained="$PRE" gpu=0 ptq.bits=3 ptq.method=$METH \
  avg_bits=$AVG seed=$SEED > "$LOG/$NAME.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say "DONE  $NAME (gpu$PHYS)"; echo "$NAME OK" >> "$LOG/STATUS"
else say "FAIL  $NAME (gpu$PHYS) rc=$rc"; echo "$NAME FAIL rc=$rc" >> "$LOG/STATUS"; fi
