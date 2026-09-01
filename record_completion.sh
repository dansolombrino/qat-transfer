#!/usr/bin/env bash
# Detached recorder: survives independently of the agent session (like the tmux workers).
# Writes a completion marker with the final summary so the result is never lost.
set -uo pipefail
M=/home/lzhou/qat-transfer/BAKEOFF_COMPLETE.md
L=/data02/users/lzhou/qat-transfer/logs/study_20260831
while ! grep -q 'big phase launched' /home/lzhou/.claude/jobs/87ad23a4/tmp/big_phase.log 2>/dev/null; do sleep 120; done
while [ -s /home/lzhou/qat-transfer/study_jobs.tsv ] || pgrep -f 'gap_allocation\.py' >/dev/null 2>&1; do sleep 120; done
{
  echo "# Bake-off complete: $(date '+%Y-%m-%d %H:%M')"
  echo
  echo "runs OK: $(grep -c ' OK$' "$L/STATUS" 2>/dev/null)"
  echo "failures: $(grep -c FAIL "$L/STATUS" 2>/dev/null || echo 0)"
  grep FAIL "$L/STATUS" 2>/dev/null | sed 's/^/  /'
  echo
  ./.venv/bin/python compare_criteria.py 2>&1
} > "$M"
