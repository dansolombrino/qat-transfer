#!/bin/bash
# Move a downed rig's remaining work onto the rigs that are still up.
#
# Usage: bx_reassign.sh <wave-id> <down-rig>
#
# Why appending to a live queue is safe
# -------------------------------------
# The workers pop with `flock <lock> ... sed -i '1d'`, so appending under the same
# lock cannot interleave with a pop and corrupt the file. The queues are plain
# text precisely so that this kind of mid-flight correction is possible.
#
# Why this is needed at all, beyond the rig being unreachable
# ----------------------------------------------------------
# A worker pops an item *out* of its queue before running it. If the machine dies
# mid-item, that line is already gone from the queue file, so simply restarting
# the rig later would silently skip it -- no artifact, no queue entry, no error.
# rig-3090-ti died 35% into Cars FP, which is exactly that case. Reassigning from
# the pristine generated copy on the hub restores the item.
#
# Placement of the reassigned work follows the same two rules as the original:
# a dataset's FP and QAT stay together (so it becomes usable as a donor without
# waiting on replication), and baselines follow their dataset.
set -u
WAVE=$1; DOWN=$2

HUB=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer
W="$HUB/scripts/dispatch/budget_axis/waves/$WAVE"
SRC="$W/$DOWN"
[ -d "$SRC" ] || { echo "no generated queues for $DOWN at $SRC" >&2; exit 2; }

declare -A ROOT=(
  [behemoth]=/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer
  [rig-4090]=/mnt/KS_2TB/PARA/Projects/quantization/qat-transfer
  [rig-3090-ti]=/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer
)

# append <rig> <queue-basename> <lockname>   (content on stdin)
append() {
  local rig=$1 q=$2 lock=$3 tmp
  tmp=$(mktemp)
  cat > "$tmp"
  [ -s "$tmp" ] || { rm -f "$tmp"; return 0; }
  local qdir="${ROOT[$rig]}/scripts/dispatch/budget_axis/waves/$WAVE/$rig"
  if [ "$rig" = "rig-4090" ]; then
    flock "$qdir/$lock" bash -c "cat >> '$qdir/$q'" < "$tmp"
  else
    # Stream through ssh into a flock on the remote side; no path or value crosses
    # two levels of quoting.
    bash "$HUB/scripts/dispatch/rssh.sh" "$rig" \
      "flock $qdir/$lock bash -c 'cat >> $qdir/$q'" < "$tmp"
  fi
  echo "  -> $rig $q : $(grep -c . "$tmp") lines"
  rm -f "$tmp"
}

echo "reassigning $DOWN work for wave $WAVE"

# --- finetunes: whole datasets, split by measured speed (behemoth ~1.3x rig-4090)
BEH_DS="Cars DTD"
R4090_DS="OxfordIIITPet"

for ds in $BEH_DS; do grep -P "\t$ds\t" "$SRC/ft.q" || true; done | append behemoth ft.q ft.lock
for ds in $R4090_DS; do grep -P "\t$ds\t" "$SRC/ft.q" || true; done | append rig-4090 ft.q ft.lock

# --- baselines: follow their dataset, so they never wait on replication
for ds in $BEH_DS; do grep -P "\t$ds\t" "$SRC/ev.q.0" || true; done | append behemoth ev.q.0 ev.lock
for ds in $R4090_DS; do grep -P "\t$ds\t" "$SRC/ev.q.0" || true; done | append rig-4090 ev.q.0 ev.lock

# --- transfer batches: receiver-agnostic once checkpoints are replicated, so split
#     by throughput. behemoth is ~2x rig-4090 per cell, hence 2:1.
for p in 1 2 3 4; do
  [ -s "$SRC/ev.q.$p" ] || continue
  awk 'NR % 3 != 0' "$SRC/ev.q.$p" | append behemoth "ev.q.$p" ev.lock
  awk 'NR % 3 == 0' "$SRC/ev.q.$p" | append rig-4090 "ev.q.$p" ev.lock
done

echo "done. $DOWN's queues left untouched on the hub copy for audit."
