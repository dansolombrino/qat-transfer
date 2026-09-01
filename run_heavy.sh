#!/usr/bin/env bash
# The 4B/8B HAWQ jobs need ~60GB (Fisher-trace gradients) and cannot share a GPU.
# Run them strictly one at a time on GPU 5, after the main sweep has drained.
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
LOG=/data02/users/lzhou/qat-transfer/logs/recalib_20260830
export CUDA_VISIBLE_DEVICES=5
set -a; . ./.env; set +a
while IFS=$'\t' read -r TAG MODEL DS METH AVG SEED HAWQ; do
  [ -z "${TAG:-}" ] && continue
  echo "[$(date +%H:%M:%S)] START $TAG (heavy, serialized)" | tee -a "$LOG/chain.log"
  t0=$SECONDS
  ./.venv/bin/python code/experiments/text/sentence_transformers/004_input_fragility/gap_allocation.py \
    model_name="$MODEL" dataset_name="$DS" gpu=0 ptq.bits=3 ptq.method="$METH" \
    avg_bits="$AVG" seed="$SEED" max_docs=8000 max_queries=1000 \
    calib_queries=128 calib_docs=512 add_hawq=true n_random=10 > "$LOG/$TAG.log" 2>&1
  rc=$?; d=$(( (SECONDS-t0)/60 ))
  if [ $rc -eq 0 ]; then echo "[$(date +%H:%M:%S)] DONE  $TAG (${d}m)" | tee -a "$LOG/chain.log"; echo "$TAG OK" >> "$LOG/STATUS"
  else echo "[$(date +%H:%M:%S)] FAIL  $TAG (${d}m) rc=$rc" | tee -a "$LOG/chain.log"; echo "$TAG FAIL rc=$rc" >> "$LOG/STATUS"; fi
done < /home/lzhou/qat-transfer/heavy_jobs.tsv
echo "heavy jobs complete"
