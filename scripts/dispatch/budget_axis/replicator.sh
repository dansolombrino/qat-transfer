#!/bin/bash
# Checkpoint replication for the budget-axis wave. Runs on rig-4090 only.
#
# Usage: replicator.sh <QDIR> [INTERVAL_S]
#
# Why it exists: a transfer cell on rig X needs the donor and receiver
# checkpoints for that cell, and the finetunes that produce them are spread over
# three rigs. Without replication each rig could only run the cells whose two
# checkpoints it happened to train, which is a small and awkward fraction of the
# grid.
#
# Why it runs on rig-4090 rather than on each producer: rig-4090 is the only host
# with outbound ssh to both others (behemoth is off-site behind a public IP and
# has no route back). So rig-4090 pulls from each rig and pushes to each rig,
# acting as the hub. Measured link speeds make this comfortable: 106 MB/s to
# rig-3090-ti on the LAN and 10.4 MB/s to behemoth, against 44 checkpoints of
# 343 MB -- about 25 minutes of transfer spread over the ~17 hours of finetuning.
#
# Only the new budgets are listed (`bx_check.py replicate-list` restricts to the
# 44 items this wave produces). The mult=1 tree is already byte-identical on all
# three rigs, so including it would make every pass re-stat 176 large files to
# discover that nothing changed.
set -u
QDIR=$1
INTERVAL=${2:-240}

ROOT=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer
PY="$ROOT/.venv/bin/python"
RSSH="$ROOT/scripts/dispatch/rssh.sh"
CHECK=scripts/dispatch/budget_axis/bx_check.py
LOCAL_CB=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer/storage/checkpoints
LOG="$QDIR/logs/replicator.log"
mkdir -p "$QDIR/logs" "$QDIR/tmp"

# rig -> repo root ; rig -> checkpoint base (they differ: rig-3090-ti keeps
# checkpoints on /mnt/WD_4TB and evaluations on /mnt/KS_960GB)
declare -A RROOT=(
  [behemoth]=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
  [rig-3090-ti]=/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer
)
declare -A RCB=(
  [behemoth]=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer/storage/checkpoints
  [rig-3090-ti]=/mnt/WD_4TB/cache/PARA/Projects/quantization/qat-transfer/storage/checkpoints
)

say() { printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG"; }

say "REPLICATOR_START pid=$$ interval=${INTERVAL}s"

CYCLE=0
while :; do
  CYCLE=$((CYCLE + 1))
  # A heartbeat every cycle, even when there is nothing to move. Without it the
  # log stays empty for the first ~20 minutes -- correct behaviour, since nothing
  # is settled yet -- and an empty log is indistinguishable from a hung loop. On
  # an unattended run that ambiguity is the problem, not the silence.
  say "CYCLE $CYCLE start"

  # Skip unreachable rigs cheaply. rssh.sh retries with backoff (up to ~105 s per
  # call), which is right for a transient session refusal but wrong for a host
  # that is off: two calls per rig per cycle would add ~3.5 minutes and slow
  # replication for the rigs that are still up. One 5-second probe decides.
  LIVE=""
  for RIG in behemoth rig-3090-ti; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$RIG" true 2>/dev/null; then
      LIVE="$LIVE $RIG"
    else
      say "SKIP $RIG unreachable"
    fi
  done

  # ---- pull: each rig's settled new-budget checkpoints -> rig-4090 ----
  for RIG in $LIVE; do
    LIST="$QDIR/tmp/pull.$RIG.list"
    if bash "$RSSH" "$RIG" "cd ${RROOT[$RIG]} && .venv/bin/python $CHECK replicate-list" \
         >"$LIST" 2>>"$LOG"; then
      N=$(grep -c . "$LIST" 2>/dev/null || echo 0)
      if [ "${N:-0}" -gt 0 ]; then
        rsync -a --partial --files-from="$LIST" \
          "$RIG:${RCB[$RIG]}/" "$LOCAL_CB/" >>"$LOG" 2>&1 \
          && say "PULL_OK $RIG files=$N" || say "PULL_ERR $RIG files=$N"
      fi
    else
      say "PULL_LIST_FAIL $RIG"
    fi
  done

  # ---- push: rig-4090's full set -> each rig ----
  LIST="$QDIR/tmp/push.list"
  PUSH_OK=1
  if (cd "$ROOT" && "$PY" "$CHECK" replicate-list) >"$LIST" 2>>"$LOG"; then
    N=$(grep -c . "$LIST" 2>/dev/null || echo 0)
    if [ "${N:-0}" -gt 0 ]; then
      for RIG in $LIVE; do
        if rsync -a --partial --files-from="$LIST" \
             "$LOCAL_CB/" "$RIG:${RCB[$RIG]}/" >>"$LOG" 2>&1; then
          say "PUSH_OK $RIG files=$N"
        else
          say "PUSH_ERR $RIG files=$N"
          PUSH_OK=0
        fi
      done
    else
      PUSH_OK=0
    fi
  else
    PUSH_OK=0
  fi

  # Stop only when rig-4090 holds all 88 files *and* the last push to both peers
  # succeeded -- otherwise a transient rsync failure on the final pass would exit
  # leaving a peer short, and every cell needing that checkpoint would then sit in
  # the deferred queue forever.
  EXPECT=88   # 44 runs x (classifier + head)
  HAVE=$(grep -c . "$LIST" 2>/dev/null || echo 0)
  if [ "${HAVE:-0}" -ge "$EXPECT" ] && [ "$PUSH_OK" -eq 1 ]; then
    say "REPLICATOR_COMPLETE all $EXPECT files present on all three rigs"
    break
  fi

  sleep "$INTERVAL"
done
