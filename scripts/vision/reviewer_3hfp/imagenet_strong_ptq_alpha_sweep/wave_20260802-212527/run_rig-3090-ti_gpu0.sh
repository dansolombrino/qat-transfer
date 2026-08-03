#!/usr/bin/env bash
# wave 20260802-212527; rig-3090-ti GPU 0; user-approved behemoth set 0,2,4,5,6,7
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
failures=0
completed=0
total=$(wc -l < "$SCRIPT_DIR/rig-3090-ti_gpu0.manifest")
while IFS= read -r job; do
  [ -n "$job" ] || continue
  echo "[$(date --iso-8601=seconds)] lane rig-3090-ti_gpu0: $((completed + 1))/$total $job"
  bash "$job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] FAILED rc=$rc: $job" >&2
    failures=$((failures + 1))
  fi
  completed=$((completed + 1))
done < "$SCRIPT_DIR/rig-3090-ti_gpu0.manifest"
echo "[$(date --iso-8601=seconds)] lane rig-3090-ti_gpu0: completed=$completed failures=$failures"
if [ "$failures" -ne 0 ]; then exit 1; fi
