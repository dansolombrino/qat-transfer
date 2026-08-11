#!/bin/bash
# Evaluation worker for the budget-axis wave: baselines and alpha=1 transfer
# cells. Runs ON the rig alongside a finetune worker on the same GPU.
#
# Usage: ev_worker.sh <QDIR> <GPU> <PY> <ROOT> <NW> <WID>
#
# Why it shares the GPU with a finetune worker rather than waiting its turn:
# a vit_base train step needs ~10 GB and an eval ~3 GB, so both fit on a 24 GB
# card. Phase-separating them would idle every GPU that finished its finetunes
# while others still trained, and -- the point of the exercise -- it would delay
# every heatmap until the very end instead of filling them column by column as
# checkpoints land.
#
# Why five queues rather than one sorted queue: priority here is a fixed, tiny
# ladder (baselines, then the two grids that were asked for, then the two
# cross-budget grids), and one file per rung makes "“take the most important
# runnable thing" a `for` loop instead of a sort under a lock.
set -u
QDIR=$1; GPU=$2; PY=$3; ROOT=$4; NW=$5; WID=$6

LOCK="$QDIR/ev.lock"
LEDGER="$QDIR/done.txt"
LOG="$QDIR/logs/ev.$WID.log"
PRIOS="0 1 2 3 4"
BACKOFF=90
mkdir -p "$QDIR/logs"

AUTHORIZED_GPUS="${AUTHORIZED_GPUS:-0}"
case " $AUTHORIZED_GPUS " in
  *" $GPU "*) ;;
  *) echo "EV_ABORT gpu=$GPU not in AUTHORIZED_GPUS='$AUTHORIZED_GPUS'" >&2; exit 64 ;;
esac

CHECK="$ROOT/scripts/dispatch/budget_axis/bx_check.py"

pop() { flock "$LOCK" bash -c "head -n1 '$1'; sed -i '1d' '$1'"; }
defer() { printf '%s\n' "$2" | flock "$LOCK" tee -a "$1" >/dev/null; }
note() { printf '%s\t%s\t%s\n' "$(date -Is)" "$WID" "$*" | flock "$LOCK" tee -a "$LEDGER" >/dev/null; }

echo "EV_WORKER_START wid=$WID gpu=$GPU pid=$$ $(date -Is)" | tee -a "$LOG"

IDLE=0
while :; do
  PROGRESSED=0

  for P in $PRIOS; do
    Q="$QDIR/ev.q.$P"
    D="$QDIR/ev.d.$P"
    [ -f "$Q" ] || continue

    while :; do
      ITEM=$(pop "$Q")
      [ -z "$ITEM" ] && break
      TYPE=$(printf '%s' "$ITEM" | cut -f1)

      if [ "$TYPE" = "BL" ]; then
        VARIANT=$(printf '%s' "$ITEM" | cut -f2)
        DS=$(printf '%s' "$ITEM" | cut -f3)
        MULT=$(printf '%s' "$ITEM" | cut -f4)

        if "$PY" "$CHECK" baseline "$VARIANT" "$DS" "$MULT"; then
          continue                        # already on disk, drop it
        fi
        # A baseline needs the checkpoint it describes. `pretrained` variants read
        # only the finetuned head, which ships with the same checkpoint dir, so
        # the same gate covers all six.
        if ! "$PY" "$CHECK" ckpt "$DS" "$MULT"; then
          defer "$D" "$ITEM"; continue
        fi

        T0=$(date +%s)
        echo "=== BL_RUN variant=$VARIANT ds=$DS mult=$MULT $(date -Is)" >>"$LOG"
        bash "$ROOT/scripts/dispatch/runners/v_baseline.sh" \
          "$VARIANT" "$DS" "$GPU" "$PY" "$ROOT" 3 "$MULT" "$NW" >>"$LOG" 2>&1
        RC=$?
        T1=$(date +%s)
        if "$PY" "$CHECK" baseline "$VARIANT" "$DS" "$MULT"; then
          note "BL_DONE	$VARIANT	$DS	mult=$MULT	rc=$RC	$((T1-T0))s"
        else
          note "BL_FAIL	$VARIANT	$DS	mult=$MULT	rc=$RC	$((T1-T0))s"
          defer "$D" "$ITEM"
        fi
        PROGRESSED=1
        break

      elif [ "$TYPE" = "QV" ]; then
        RECV=$(printf '%s' "$ITEM" | cut -f2)
        TMULT=$(printf '%s' "$ITEM" | cut -f3)
        SMULT=$(printf '%s' "$ITEM" | cut -f4)
        GRID=$(printf '%s' "$ITEM" | cut -f5)
        DONORS=$(printf '%s' "$ITEM" | cut -f6)

        # Which of this batch's donors can run right now: checkpoints settled at
        # the donor budget, receiver settled at its own, cell not already written.
        # This is the whole progressive-fill mechanism -- the batch runs with
        # whatever is ready and comes back later for the rest.
        READY=$("$PY" "$CHECK" runnable-donors "$RECV" "$TMULT" "$SMULT" "$DONORS")

        if [ -z "$READY" ]; then
          PENDING=$("$PY" "$CHECK" pending-donors "$RECV" "$TMULT" "$SMULT" "$DONORS")
          if [ -n "$PENDING" ]; then
            defer "$D" "$(printf 'QV\t%s\t%s\t%s\t%s\t%s' "$RECV" "$TMULT" "$SMULT" "$GRID" "$PENDING")"
          fi
          continue
        fi

        T0=$(date +%s)
        NCELL=$(printf '%s' "$READY" | tr ',' '\n' | grep -c .)
        echo "=== QV_RUN grid=$GRID recv=$RECV tmult=$TMULT smult=$SMULT n=$NCELL donors=$READY $(date -Is)" >>"$LOG"
        bash "$ROOT/scripts/dispatch/runners/v_qv.sh" \
          3 "$RECV" "$READY" "$GPU" "$PY" "$ROOT" "$SMULT" "$TMULT" >>"$LOG" 2>&1
        RC=$?
        T1=$(date +%s)

        # qv_transfer.py exits 0 when a checkpoint is missing, so the only honest
        # completion signal is which cells now exist on disk.
        PENDING=$("$PY" "$CHECK" pending-donors "$RECV" "$TMULT" "$SMULT" "$DONORS")
        WROTE=$(( NCELL - $(printf '%s' "$PENDING" | tr ',' '\n' | grep -c .) ))
        note "QV_BATCH	$GRID	$RECV	tmult=$TMULT	smult=$SMULT	tried=$NCELL	wrote=$WROTE	rc=$RC	$((T1-T0))s"
        if [ -n "$PENDING" ]; then
          defer "$D" "$(printf 'QV\t%s\t%s\t%s\t%s\t%s' "$RECV" "$TMULT" "$SMULT" "$GRID" "$PENDING")"
        fi
        PROGRESSED=1
        break
      fi
    done

    # Queue rung drained: fold the deferred items back so the next cycle
    # re-examines them against a filesystem that has moved on.
    if [ -s "$D" ]; then
      flock "$LOCK" bash -c "cat '$D' >> '$Q'; : > '$D'"
    fi

    [ "$PROGRESSED" -eq 1 ] && break
  done

  if [ "$PROGRESSED" -eq 1 ]; then
    IDLE=0
    continue
  fi

  # Nothing runnable anywhere. Either checkpoints are still training (wait) or
  # the wave is finished (exit). Distinguish on the artifact counts, not a guess.
  REMAIN=$("$PY" "$CHECK" remaining 2>/dev/null)
  if [ "${REMAIN:-1}" = "0" ]; then
    note "EV_WORKER_EXIT	$WID	all baselines and cells present"
    break
  fi

  IDLE=$((IDLE + 1))
  if [ $((IDLE % 10)) -eq 1 ]; then
    note "EV_IDLE	$WID	waiting on checkpoints	remaining=$REMAIN"
  fi
  sleep "$BACKOFF"
done

echo "EV_WORKER_EXIT wid=$WID $(date -Is)" | tee -a "$LOG"
