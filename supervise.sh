#!/usr/bin/env bash
# Restart the dispatcher if it dies while unclaimed jobs remain. The dispatcher records
# CLAIMED jobs in .jobq.done, so a restart never re-runs finished work.
set -uo pipefail
cd /home/lzhou/qat-transfer || exit 1
LOG=/data02/users/lzhou/qat-transfer/logs/latest
say(){ echo "[$(date +%H:%M:%S)] supervisor: $*" | tee -a "$LOG/chain.log"; }
restarts=0
while true; do
  sleep 60
  alive=$(ps -eo cmd | grep -c '^bash dispatcher\.sh' || true)
  [ "$alive" -gt 0 ] && continue
  # any job name in JOBS.txt not yet claimed?
  pending=$(awk '{print $1}' JOBS.txt | grep -vxF -f .jobq.done 2>/dev/null | wc -l)
  if [ "${pending:-0}" -eq 0 ]; then say "queue fully claimed; supervisor exiting"; exit 0; fi
  if [ "$restarts" -ge 20 ]; then say "restart cap reached, giving up"; exit 1; fi
  restarts=$((restarts+1))
  say "dispatcher gone with $pending unclaimed; restart #$restarts"
  nohup bash dispatcher.sh >> "$LOG/dispatcher.log" 2>&1 &
done
