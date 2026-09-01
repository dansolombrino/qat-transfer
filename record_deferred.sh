#!/usr/bin/env bash
# Second milestone: the 4B/8B x {GPTQ,AWQ} cross. Waits until that phase has actually
# started before watching for drain, so it cannot fire in the gap between phases.
set -uo pipefail
M=/home/lzhou/qat-transfer/DEFERRED_COMPLETE.md
L=/data02/users/lzhou/qat-transfer/logs/study_20260831
while ! tmux has-session -t def5 2>/dev/null && ! tmux has-session -t def6 2>/dev/null; do sleep 60; done
sleep 60
while [ -s /home/lzhou/qat-transfer/study_jobs.tsv ] || pgrep -f 'gap_allocation\.py' >/dev/null 2>&1; do sleep 120; done
{
  echo "# Deferred 4B/8B cross complete: $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "Z-runs OK: $(grep -c '^Z_.* OK$' "$L/STATUS" 2>/dev/null)"
  echo "Z-runs failed: $(grep '^Z_' "$L/STATUS" 2>/dev/null | grep -c FAIL || echo 0)"
  grep '^Z_' "$L/STATUS" 2>/dev/null | grep FAIL | sed 's/^/  /'
  echo
  ./.venv/bin/python compare_criteria.py 2>&1
} > "$M"
