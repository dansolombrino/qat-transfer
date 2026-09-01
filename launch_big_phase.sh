#!/usr/bin/env bash
# Chained after the mid-model study: 4B/8B with HAWQ needs ~60GB of gradients,
# so these run strictly one job per GPU.
set -uo pipefail
while [ -s /home/lzhou/qat-transfer/study_jobs.tsv ] || pgrep -f 'gap_allocation\.py' >/dev/null 2>&1; do
  sleep 60
done
sleep 20
cp /home/lzhou/qat-transfer/study_jobs_big.tsv /home/lzhou/qat-transfer/study_jobs.tsv
for g in 5 6 7; do
  tmux new-session -d -s "big$g" \
    "bash /home/lzhou/qat-transfer/study_worker.sh $g 2>&1 | tee -a /data02/users/lzhou/qat-transfer/logs/study_20260831/big$g.log"
done
echo "big phase launched at $(date)"
