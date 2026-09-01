#!/usr/bin/env bash
# 4B/8B x {GPTQ, AWQ} with HAWQ: ~60GB of gradients plus a slow quantizer.
# Strictly one job per GPU. Chained after the current phase drains.
set -uo pipefail
while pgrep -f 'gap_allocation\.py' >/dev/null 2>&1; do sleep 60; done
sleep 30
cp /home/lzhou/qat-transfer/deferred_jobs.tsv /home/lzhou/qat-transfer/study_jobs.tsv
for g in 5 6 7; do
  tmux new-session -d -s "def$g" \
    "bash /home/lzhou/qat-transfer/study_worker.sh $g 2>&1 | tee -a /data02/users/lzhou/qat-transfer/logs/study_20260831/def$g.log"
done
echo "deferred big-model cross launched at $(date)"
