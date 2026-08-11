#!/bin/bash
# Finetune worker for the budget-axis wave. Runs ON the rig, pulling from a
# host-local flock queue.
#
# Usage: ft_worker.sh <QDIR> <GPU> <PY> <ROOT> <NW> <WID>
#
# Why a pull queue rather than a static per-worker assignment: the cost estimates
# that would drive a static split are unreliable (a 4x DTD run is 304 epochs over
# 3,384 images -- dominated by per-epoch eval, not by steps), and a queue lets a
# lagging rig simply take fewer items instead of becoming the critical path.
#
# Why the queue is host-local rather than shared: there is no shared filesystem
# across these three machines, and one ssh per work item would hit behemoth's
# stock `MaxSessions 10` immediately.
set -u
QDIR=$1; GPU=$2; PY=$3; ROOT=$4; NW=$5; WID=$6

Q="$QDIR/ft.q"
LOCK="$QDIR/ft.lock"
LEDGER="$QDIR/done.txt"
LOG="$QDIR/logs/ft.$WID.log"
mkdir -p "$QDIR/logs"

# The authorized-GPU guard. behemoth has eight cards and only GPU 0 is ours; the
# other seven belong to other people and are in use. Exiting 64 rather than
# falling back to another card makes a misconfiguration loud and harmless.
AUTHORIZED_GPUS="${AUTHORIZED_GPUS:-0}"
case " $AUTHORIZED_GPUS " in
  *" $GPU "*) ;;
  *) echo "FT_ABORT gpu=$GPU not in AUTHORIZED_GPUS='$AUTHORIZED_GPUS'" >&2; exit 64 ;;
esac

pop() { flock "$LOCK" bash -c "head -n1 '$Q'; sed -i '1d' '$Q'"; }

note() { printf '%s\t%s\t%s\n' "$(date -Is)" "$WID" "$*" | flock "$LOCK" tee -a "$LEDGER" >/dev/null; }

echo "FT_WORKER_START wid=$WID gpu=$GPU pid=$$ $(date -Is)" | tee -a "$LOG"

while :; do
  ITEM=$(pop)
  [ -z "$ITEM" ] && break

  KIND=$(printf '%s' "$ITEM" | cut -f1)
  DS=$(printf '%s' "$ITEM" | cut -f2)
  MULT=$(printf '%s' "$ITEM" | cut -f3)

  T0=$(date +%s)
  echo "=== FT_RUN kind=$KIND ds=$DS mult=$MULT gpu=$GPU $(date -Is)" >>"$LOG"

  if [ "$KIND" = "fp" ]; then
    bash "$ROOT/scripts/dispatch/runners/v_fp.sh" "$DS" "$GPU" "$PY" "$ROOT" "$MULT" "$NW" >>"$LOG" 2>&1
  else
    bash "$ROOT/scripts/dispatch/runners/v_qat.sh" 3 "$DS" "$GPU" "$PY" "$ROOT" "$MULT" "$NW" >>"$LOG" 2>&1
  fi
  RC=$?
  T1=$(date +%s)

  # The artifact is the verdict, not $?. A dropped connection or a non-zero exit
  # from a step after the save would otherwise record a finished run as failed.
  if "$PY" "$ROOT/scripts/dispatch/budget_axis/bx_check.py" ckpt-one "$KIND" "$DS" "$MULT"; then
    note "FT_DONE	$KIND	$DS	mult=$MULT	rc=$RC	$((T1-T0))s"
  else
    note "FT_FAIL	$KIND	$DS	mult=$MULT	rc=$RC	$((T1-T0))s"
    # One retry, appended to the tail so it does not block the queue behind it.
    RETRIES="$QDIR/ft.retried"
    if ! grep -qxF "$KIND	$DS	$MULT" "$RETRIES" 2>/dev/null; then
      printf '%s\t%s\t%s\n' "$KIND" "$DS" "$MULT" | flock "$LOCK" tee -a "$RETRIES" >/dev/null
      printf '%s\t%s\t%s\n' "$KIND" "$DS" "$MULT" | flock "$LOCK" tee -a "$Q" >/dev/null
      note "FT_REQUEUE	$KIND	$DS	mult=$MULT"
    else
      note "FT_PARKED	$KIND	$DS	mult=$MULT	(already retried once)"
    fi
  fi
done

echo "FT_WORKER_EXIT wid=$WID queue empty $(date -Is)" | tee -a "$LOG"
note "FT_WORKER_EXIT	$WID"
