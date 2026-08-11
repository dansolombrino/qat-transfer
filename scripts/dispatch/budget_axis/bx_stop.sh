#!/bin/bash
# Stop budget-axis workers on one rig, correctly.
#
# Usage: bx_stop.sh <wave-id> <rig> [ft|ev|all]
#
# Why this script exists rather than a tmux command
# -------------------------------------------------
# `tmux kill-session` is NOT sufficient and this cost a real incident on this
# wave. Killing the session removes the pane, but the worker shell inside it can
# survive reparenting and keep pulling from its queue: after a kill-session that
# `tmux ls` confirmed gone, a worker on behemoth started a fresh 21-cell batch
# 46 seconds later, from the queue contents it had before the queue was
# corrected. Two stale workers ran for another eight minutes while `tmux ls`
# showed nothing.
#
# So: kill the recorded pane pids -- which are the worker shells -- and then
# reap the children they leave behind. `pkill -f ev_worker` is not an option
# either: the pattern matches the shell doing the killing.
#
# The workers are safe to stop at any point. Every guard is an artifact check, so
# relaunching with the same wave id resumes exactly where this left off.
set -u
WAVE=$1; RIG=$2; WHICH=${3:-all}

case "$RIG" in
  behemoth)    ROOT=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer ;;
  rig-4090)    ROOT=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer ;;
  rig-3090-ti) ROOT=/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer ;;
  *) echo "unknown rig $RIG" >&2; exit 2 ;;
esac
QDIR="$ROOT/scripts/dispatch/budget_axis/waves/$WAVE/$RIG"

# Child process patterns, matched with a bracket so this script's own command line
# cannot match them.
# pids.txt lines are "<session-name> <pane-pid>", so the session pattern must be
# anchored to the end of the *first field*, not the end of the line. Anchoring to
# the line matched nothing, so an earlier run of this script killed the children
# while leaving the worker shells alive -- and they immediately started new work.
# The pattern is applied to $1 in awk for exactly that reason.
case "$WHICH" in
  ft)  SESS_RX='_ft$';         CHILD_RX='finetune_(fp|qat)[.]py' ;;
  ev)  SESS_RX='_ev[0-9]+$';   CHILD_RX='qv_transfer[.]py|evaluate_[a-z_]*[.]py' ;;
  all) SESS_RX='_(ft|ev[0-9]+)$';
       CHILD_RX='finetune_(fp|qat)[.]py|qv_transfer[.]py|evaluate_[a-z_]*[.]py' ;;
  *) echo "third arg must be ft, ev or all" >&2; exit 2 ;;
esac

run() {
  if [ "$RIG" = "rig-4090" ]; then bash -c "$1"; else
    bash /mnt/KS_2TB/PARA/Projects/quantization/qat-transfer/scripts/dispatch/rssh.sh "$RIG" "$1"
  fi
}

run "
set -u
PIDS=\$(awk '\$1 ~ /$SESS_RX/ {print \$2}' '$QDIR/pids.txt' 2>/dev/null)
echo \"worker shells: \$PIDS\"
if [ -z \"\$PIDS\" ]; then
  echo 'ERROR: no worker shells matched -- refusing to kill children, which would'
  echo 'only make the live workers restart the work they were doing.' >&2
  exit 3
fi
kill \$PIDS 2>/dev/null
sleep 4
# Reap children the shells left behind, by explicit pid.
CH=\$(ps -eo pid,args | grep -E '$CHILD_RX' | awk '{print \$1}')
echo \"children: \$CH\"
[ -n \"\$CH\" ] && kill \$CH 2>/dev/null
sleep 5
STILL=\$(ps -eo pid,args | grep -cE '$CHILD_RX')
echo \"children remaining: \$STILL\"
# Any worker shell that ignored SIGTERM gets SIGKILL -- a survivor here is the
# whole reason this script exists.
for p in \$PIDS; do
  if ps -o pid= -p \$p >/dev/null 2>&1; then echo \"SIGKILL \$p (ignored TERM)\"; kill -9 \$p 2>/dev/null; fi
done
tmux ls 2>/dev/null | grep bxaxis || echo 'no bxaxis sessions'
"
