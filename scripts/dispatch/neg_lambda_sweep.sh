#!/bin/bash
# NEGATIVE-LAMBDA SWEEP -- bert-base, text, 001_qat_transfer, bits=3.
#
# Every lambda ever evaluated for this family is positive (0.05 .. 2.00 step 0.05
# on val).  The left arm of every sensitivity curve is assumed, not measured:
# lambda_curves_common.py anchors it at (0, 0).  17% of bert-base's cross-task
# cells have their val optimum pinned at the left grid edge, so we currently
# cannot tell "this donor QV is useless here" from "this donor QV is
# anti-aligned here" -- under Proposition 1's cos^2 law, very different claims.
#
# Topology.  Workers run ON each host, pulling from a host-local flock queue.
# The obvious design -- one ssh feeder per slot -- does not work: behemoth's
# sshd runs MaxSessions 10 with no sudo to raise it, MaxSessions counts channels
# per connection (so ControlMaster multiplexing makes it worse), and 18 slots
# each holding a session for a job's duration is a hard refusal that no amount of
# retrying fixes.  Here ssh is used only to deploy, launch and poll.
#
# Ordering.  The queue is ordered fine tier -> coarse tier -> test split, not by
# lambda.  The fine tier is what resolves the censored cells; the coarse mirror
# only buys the symmetry figure.  A window that closes early should cost the
# figure, not the answer.
set -u

SP="${SP:-/tmp/claude-1000/-mnt-KS-2TB-PARA-Projects-quantization-qat-transfer/a4a0cd93-06a8-4ad7-91cd-1756136be5ed/scratchpad}"
LOCAL_ROOT=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer
RSSH="bash $LOCAL_ROOT/scripts/dispatch/rssh.sh"
RUNDIR="$SP/neglambda"; mkdir -p "$RUNDIR"
MASTER="$RUNDIR/jobs_all.txt"

MODEL_DIR=google_bert_bert_base_uncased
OPTIM='optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128'
QATF='qat=bits=3_gran=channel_skip=classifier'
PTQF='ptq=bits=3_gran=channel_skip=classifier'
QVREL="text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

TARGETS="Emotion IMDB Banking77 AmazonReviewsClassification AmazonCounterfactual MassiveIntent MassiveScenario MTOPDomain MTOPIntent ToxicConversations TweetSentimentExtraction"
FINE="-0.05 -0.1 -0.15 -0.2 -0.25 -0.3 -0.35 -0.4 -0.45 -0.5"
COARSE="-0.75 -1.0 -1.25 -1.5 -1.75 -2.0"
TEST_ALPHAS="-1.0"

# host | ssh-name (or "local") | root | ckpt base | eval base | gpus | procs-per-gpu
HOSTS=(
  "local|local|$LOCAL_ROOT|$LOCAL_ROOT/storage/checkpoints|$LOCAL_ROOT/evaluations|0|2"
  "rig3090|rig-3090-ti|/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer|/mnt/WD_4TB/cache/PARA/Projects/quantization/qat-transfer/storage/checkpoints|/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer/evaluations|0|2"
  "behemoth|behemoth|/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer|/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer/storage/checkpoints|/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer/evaluations|0 2 4 5 6 7|3"
)

say() { echo "### [$(date +%T)] $*"; }

run_on() {   # run_on <sshname> <command>
  if [ "$1" = "local" ]; then shift; bash -c "$*"; else local h=$1; shift; $RSSH "$h" "$*"; fi
}

# ---------------------------------------------------------------- checkpoint gate
# qv_transfer.py warns and exits 0 when a checkpoint is missing, so a mis-staged
# host produces nothing while looking healthy.  Gate on the artifact, per host.
GATED=()
for spec in "${HOSTS[@]}"; do
  IFS='|' read -r name ssh root ckpt evalb gpus ppg <<< "$spec"
  B="$ckpt/text/ilharco_automodelforsequenceclassification"
  n_qat=$(run_on "$ssh" "find $B/qat/$MODEL_DIR -path '*$QATF*' -name 'backbone_epoch_5.pt' 2>/dev/null | wc -l" 2>/dev/null | tr -d ' \r')
  n_fp=$(run_on "$ssh" "find $B/fp/$MODEL_DIR -path '*bs=32_ml=128*' -name 'backbone_epoch_5.pt' 2>/dev/null | wc -l" 2>/dev/null | tr -d ' \r')
  if [ "${n_qat:-0}" -ge 11 ] && [ "${n_fp:-0}" -ge 11 ]; then
    say "GATE OK   $name  qat=$n_qat fp=$n_fp"
    GATED+=("$spec")
  else
    say "GATE FAIL $name  qat=${n_qat:-?} fp=${n_fp:-?}  -- excluded"
  fi
done
[ ${#GATED[@]} -eq 0 ] && { say "no host passed the gate"; exit 1; }

# ------------------------------------------------------------------- build queue
# Resume-safe: a (alpha, split, target) job is enqueued unless all 11 of its
# source cells already exist somewhere in the fleet.
say "scanning existing negative-alpha cells across the fleet"
HAVE="$RUNDIR/have.txt"; : > "$HAVE"
for spec in "${GATED[@]}"; do
  IFS='|' read -r name ssh root ckpt evalb gpus ppg <<< "$spec"
  run_on "$ssh" "find $evalb/$QVREL/$MODEL_DIR -path '*$QATF*' -path '*$PTQF*' -path '*qv=alpha=-*' -name eval_results.json 2>/dev/null" \
    | sed -E 's#.*/src=([^_]+)_seed=[0-9]+/tgt=([^_]+)_seed=[0-9]+/.*/qv=alpha=([^/]+)/split=([^/]+)/.*#\3 \4 \2#' >> "$HAVE"
done
sort "$HAVE" | uniq -c | awk '$1>=11 {print $2, $3, $4}' | sort -u > "$RUNDIR/done_jobs.txt"
say "already complete: $(wc -l < "$RUNDIR/done_jobs.txt") jobs"

: > "$MASTER"
emit() {  # emit <alpha> <split>
  for t in $TARGETS; do
    # -- is required: every alpha starts with '-', which grep reads as an option.
    grep -qxF -- "$1 $2 $t" "$RUNDIR/done_jobs.txt" || echo "$1 $2 $t" >> "$MASTER"
  done
}
for a in $FINE;        do emit "$a" val;  done   # tier 1: resolves the censoring
for a in $COARSE;      do emit "$a" val;  done   # tier 2: symmetry figure
for a in $TEST_ALPHAS; do emit "$a" test; done   # tier 3: mirror-of-unit on test
NJOBS=$(wc -l < "$MASTER")
say "queue: $NJOBS jobs"
[ "$NJOBS" -eq 0 ] && { say "nothing to do"; exit 0; }

# ------------------------------------------------------------------------- shard
# Interleave by slot count so every host's local queue stays in priority order.
SLOTLIST="$RUNDIR/slots.txt"; : > "$SLOTLIST"
for spec in "${GATED[@]}"; do
  IFS='|' read -r name ssh root ckpt evalb gpus ppg <<< "$spec"
  for g in $gpus; do for p in $(seq 1 "$ppg"); do echo "$name"; done; done
done > "$SLOTLIST"
NSLOTS=$(wc -l < "$SLOTLIST")
say "slots: $NSLOTS"

for spec in "${GATED[@]}"; do
  IFS='|' read -r name ssh root ckpt evalb gpus ppg <<< "$spec"; : > "$RUNDIR/q.$name.txt"
done
i=0
while read -r line; do
  owner=$(sed -n "$(( i % NSLOTS + 1 ))p" "$SLOTLIST")
  echo "$line" >> "$RUNDIR/q.$owner.txt"
  i=$((i+1))
done < "$MASTER"

# ------------------------------------------------------- deploy, launch, record
PIDFILE="$RUNDIR/pids.txt"; : > "$PIDFILE"
for spec in "${GATED[@]}"; do
  IFS='|' read -r name ssh root ckpt evalb gpus ppg <<< "$spec"
  Q="$RUNDIR/q.$name.txt"; NQ=$(wc -l < "$Q")
  say "$name: $NQ jobs"
  [ "$NQ" -eq 0 ] && continue

  if [ "$ssh" = "local" ]; then
    RD="$root/scripts/dispatch/runners"; RQ="$RUNDIR/q.$name.txt"; RDONE="$RUNDIR/done.$name.txt"; RLOG="$RUNDIR"
  else
    RD='~/runners'; RQ='~/negsweep/q.txt'; RDONE='~/negsweep/done.txt'; RLOG='~/negsweep'
    scp -q "$LOCAL_ROOT/scripts/dispatch/runners/t_qv_alpha.sh" "$LOCAL_ROOT/scripts/dispatch/runners/neg_worker.sh" "$ssh:runners/" \
      || { say "$name: runner deploy FAILED -- skipping"; continue; }
    $RSSH "$ssh" "mkdir -p ~/negsweep && rm -f ~/negsweep/q.txt ~/negsweep/q.lock"
    scp -q "$Q" "$ssh:negsweep/q.txt" || { say "$name: queue push FAILED -- skipping"; continue; }
  fi

  # PY: behemoth's torch is hand-pinned for sm_120; never uv run / uv sync there.
  PY="$root/.venv/bin/python"
  for g in $gpus; do
    for p in $(seq 1 "$ppg"); do
      W="${name}_g${g}_$p"
      CMD="nohup setsid bash $RD/neg_worker.sh $RQ $RDONE $RLOG/w.$W.log $g $W $RD/t_qv_alpha.sh $PY $root </dev/null >/dev/null 2>&1 &"
      if [ "$ssh" = "local" ]; then eval "$CMD"; echo "local $! $W" >> "$PIDFILE"
      else $RSSH "$ssh" "$CMD exit 0"; echo "$ssh - $W" >> "$PIDFILE"; fi
    done
  done
  say "$name: launched $(( $(echo $gpus | wc -w) * ppg )) workers"
done

say "LAUNCH COMPLETE -- $NJOBS jobs over $NSLOTS slots"
say "workers recorded in $PIDFILE (never pkill -f neg_worker: it matches the caller)"
