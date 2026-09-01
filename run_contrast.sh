#!/usr/bin/env bash
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1; NAME=$2; DS=$3; BITS=$4
export CUDA_VISIBLE_DEVICES=$PHYS
LOG=/data02/users/lzhou/qat-transfer/logs/latest
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
say "START $NAME (gpu $PHYS, same-checkpoint contrast $DS W$BITS)"
./.venv/bin/python code/experiments/text/sentence_transformers/004_input_fragility/same_checkpoint_contrast.py \
  dataset_name="$DS" ptq.bits=$BITS gpu=0 > "$LOG/$NAME.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say "DONE  $NAME (gpu$PHYS)"; echo "$NAME OK" >> "$LOG/STATUS"
else say "FAIL  $NAME (gpu$PHYS) rc=$rc"; echo "$NAME FAIL rc=$rc" >> "$LOG/STATUS"; fi
