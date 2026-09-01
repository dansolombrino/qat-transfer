#!/usr/bin/env bash
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1; NAME=$2; MODEL=$3; DS=$4; AVG=${5:-3.5}; SEED=${6:-2038}; METH=${7:-rtn}
GRAN=${8:-group_128}; LOBIT=${9:-3}   # damage levers for the boundary sweep
HAWQ=${10:-false}
HIBIT=$((LOBIT+1))
# The FP checkpoints were fine-tuned with different batch sizes per backbone, and the batch
# size is baked into the checkpoint path: ViT-B used 128, ViT-L used 64.
case "$MODEL" in *large*) BS=64;; *) BS=128;; esac
export CUDA_VISIBLE_DEVICES=$PHYS
LOG=/data02/users/lzhou/qat-transfer/logs/latest
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
say "START $NAME (gpu $PHYS, $MODEL $DS clsf $METH avg=$AVG seed=$SEED)"
./.venv/bin/python code/experiments/vision/ilharco_timm_supervised/004_input_fragility/gap_allocation_clsf.py \
  model_name="$MODEL" dataset_name="$DS" batch_size=$BS lr=1e-05 wd=0.1 ls=0.0 wl=500 \
  max_grad_norm=1.0 seed=$SEED gpu=0 avg_bits=$AVG ptq.method=$METH calib_batches=8 \
  ptq.granularity=$GRAN ptq.bits=$LOBIT "bit_choices=[$LOBIT,$HIBIT]" add_hawq=$HAWQ \
  > "$LOG/$NAME.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then say "DONE  $NAME (gpu$PHYS)"; echo "$NAME OK" >> "$LOG/STATUS"
else say "FAIL  $NAME (gpu$PHYS) rc=$rc"; echo "$NAME FAIL rc=$rc" >> "$LOG/STATUS"; fi
