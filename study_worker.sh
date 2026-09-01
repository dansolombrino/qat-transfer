#!/usr/bin/env bash
# Study A (all criteria under RTN) and Study B (gap vs HAWQ composed with GPTQ/AWQ).
# Fields: TAG MODEL CORPUS METHOD AVG SEED HAWQ BASELINES NRANDOM
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
PHYS=$1
LOG=/data02/users/lzhou/qat-transfer/logs/study_20260831
Q=/home/lzhou/qat-transfer/study_jobs.tsv
LK=$LOG/.queue.lock
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$PHYS
set -a; . ./.env; set +a
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/chain.log"; }
pop(){ flock 9; local l; l=$(head -1 "$Q"); [ -z "$l" ] && return 1
       tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"; printf '%s' "$l"; }
while :; do
  JOB=$(pop 9>"$LK") || { say "gpu$PHYS: queue empty"; break; }
  [ -z "$JOB" ] && { say "gpu$PHYS: queue empty"; break; }
  IFS=$'\t' read -r TAG MODEL DS METH AVG SEED HAWQ BASE NR <<< "$JOB"
  say "START $TAG (gpu$PHYS)"
  t0=$SECONDS
  ./.venv/bin/python code/experiments/text/sentence_transformers/004_input_fragility/gap_allocation.py \
    model_name="$MODEL" dataset_name="$DS" gpu=0 ptq.bits=3 ptq.method="$METH" \
    avg_bits="$AVG" seed="$SEED" max_docs=8000 max_queries=1000 calib_queries=128 calib_docs=512 \
    add_hawq=$([ "$HAWQ" = 1 ] && echo true || echo false) \
    add_baselines=$([ "$BASE" = 1 ] && echo true || echo false) \
    baselines_lite=$([ "$BASE" = 2 ] && echo true || echo false) \
    n_random="$NR" > "$LOG/$TAG.log" 2>&1
  rc=$?; d=$(( (SECONDS-t0)/60 ))
  if [ $rc -eq 0 ]; then say "DONE  $TAG (gpu$PHYS, ${d}m)"; echo "$TAG OK" >> "$LOG/STATUS"
  else say "FAIL  $TAG (gpu$PHYS, ${d}m) rc=$rc"; echo "$TAG FAIL rc=$rc" >> "$LOG/STATUS"; fi
done
