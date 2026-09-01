#!/usr/bin/env bash
# One worker per GPU. Pops jobs atomically from a shared queue so the three GPUs
# self-balance: the longest jobs are at the head of the file, short ones at the tail.
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1
LOG=/data02/users/lzhou/qat-transfer/logs/recalib_20260830
Q=/home/lzhou/qat-transfer/recalib_jobs.tsv
LK=$LOG/.queue.lock
export CUDA_VISIBLE_DEVICES=$PHYS
set -a; . ./.env; set +a
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }

pop(){  # atomically take the first line
  flock 9
  local line; line=$(head -1 "$Q")
  [ -z "$line" ] && return 1
  tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  printf '%s' "$line"
}

while :; do
  JOB=$(pop 9>"$LK") || { say "gpu$PHYS: queue empty, worker exiting"; break; }
  [ -z "$JOB" ] && { say "gpu$PHYS: queue empty, worker exiting"; break; }
  IFS=$'\t' read -r TAG MODEL DS METH AVG SEED HAWQ <<< "$JOB"
  EXTRA=""; [ "$HAWQ" = "1" ] && EXTRA="add_hawq=true n_random=10"
  say "START $TAG (gpu$PHYS)"
  t0=$SECONDS
  ./.venv/bin/python code/experiments/text/sentence_transformers/004_input_fragility/gap_allocation.py \
    model_name="$MODEL" dataset_name="$DS" gpu=0 ptq.bits=3 ptq.method="$METH" \
    avg_bits="$AVG" seed="$SEED" max_docs=8000 max_queries=1000 \
    calib_queries=128 calib_docs=512 $EXTRA > "$LOG/$TAG.log" 2>&1
  rc=$?; d=$(( (SECONDS-t0)/60 ))
  if [ $rc -eq 0 ]; then say "DONE  $TAG (gpu$PHYS, ${d}m)"; echo "$TAG OK" >> "$LOG/STATUS"
  else say "FAIL  $TAG (gpu$PHYS, ${d}m) rc=$rc"; echo "$TAG FAIL rc=$rc" >> "$LOG/STATUS"; fi
done
