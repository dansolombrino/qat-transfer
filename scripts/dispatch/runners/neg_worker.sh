#!/bin/bash
# One sweep slot.  Runs ON the target host, pulling from a queue file that also
# lives on that host.
#
# Why the queue is host-local rather than driven by an ssh feeder per slot:
# behemoth's sshd runs MaxSessions 10 with no sudo to raise it, and MaxSessions
# counts *channels per connection*, which ControlMaster multiplexing makes worse
# rather than better.  18 slots each holding an ssh session for the duration of a
# job is a hard refusal, not something rssh.sh's retry can paper over.  So ssh is
# used only to launch these workers and to poll them, and the work-stealing
# happens locally under flock.
#
# Usage: neg_worker.sh <QUEUE> <DONE> <LOG> <GPU> <NAME> <RUNNER> <PY> <ROOT> [SRC_MULT] [TGT_MULT]
#
# The two multipliers are only passed through to t_qv_alpha.sh; this worker
# builds no paths of its own. Both default to 1.
set -u
QUEUE=$1; DONE=$2; LOG=$3; GPU=$4; NAME=$5; RUNNER=$6; PY=$7; ROOT=$8; SMULT=${9:-1}; TMULT=${10:-1}
LOCK="${QUEUE%.txt}.lock"
MODEL="google-bert/bert-base-uncased"
SKIP="classifier"
BITS=3
ALLT="Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction"

touch "$LOCK"

pop() {
  flock "$LOCK" bash -c 'if [ -s "$1" ]; then head -1 "$1"; sed -i 1d "$1"; exit 0; fi; exit 1' _ "$QUEUE"
}

# Queue line: "<ALPHA> <SPLIT> <TARGET> [SOURCES]".  SOURCES is optional and
# defaults to all 11: the lambda grid sweeps a whole target row per job, while
# the follow-up that evaluates each cell's selected lambda* on test needs a
# single (source, target) cell at a time.
while item=$(pop); do
  [ -z "$item" ] && continue
  ALPHA=$(echo "$item" | awk '{print $1}')
  SPLIT=$(echo "$item" | awk '{print $2}')
  TGT=$(echo "$item"   | awk '{print $3}')
  SRCS=$(echo "$item"  | awk '{print $4}')
  # Must not assign back into ALLT: it would leak this item's single source
  # into every later item that omits the field.
  [ -z "$SRCS" ] && SRCS="$ALLT"
  echo "### [$(date +%T)] $NAME gpu$GPU <- alpha=$ALPHA split=$SPLIT tgt=$TGT" >> "$LOG"
  START=$(date +%s)
  if bash "$RUNNER" "$BITS" "$MODEL" "$TGT" "$SRCS" "$SKIP" "$ALPHA" "$SPLIT" "$GPU" "$PY" "$ROOT" "$SMULT" "$TMULT" >> "$LOG" 2>&1; then
    RC=OK
  else
    RC=FAILED
  fi
  # Exit status is recorded but is NOT the completion signal -- jobs here have
  # historically reported FAILED after writing every result and OK after writing
  # a fraction.  Completion is decided by artifact count in the reconcile pass.
  echo "$NAME|$RC|$((($(date +%s) - START)))s|$ALPHA|$SPLIT|$TGT" >> "$DONE"
done

echo "### [$(date +%T)] $NAME gpu$GPU drained" >> "$LOG"
