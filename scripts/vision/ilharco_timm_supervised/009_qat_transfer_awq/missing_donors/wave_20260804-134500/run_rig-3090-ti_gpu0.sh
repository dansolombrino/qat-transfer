#!/usr/bin/env bash
# wave 20260804-134500; rig-3090-ti GPU 0; user-approved rig-3090-ti set 0
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
PHYSICAL_GPU=0
AUTHORIZED_GPUS="0"
case ",$AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized rig-3090-ti GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
MANIFEST="$SCRIPT_DIR/rig-3090-ti_gpu0.manifest"
FAILURES="$SCRIPT_DIR/rig-3090-ti_gpu0.failures"
: > "$FAILURES"
failures=0
completed=0
total=8
while IFS= read -r job; do
  [ -n "$job" ] || continue
  echo "[$(date --iso-8601=seconds)] lane rig-3090-ti_gpu0: $((completed + 1))/$total $job"
  bash "$job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] FAILED rc=$rc: $job" >&2
    echo "$job" >> "$FAILURES"
    failures=$((failures + 1))
  fi
  completed=$((completed + 1))
done < "$MANIFEST"
echo "[$(date --iso-8601=seconds)] lane rig-3090-ti_gpu0: completed=$completed failures=$failures"
if [ "$failures" -ne 0 ]; then exit 1; fi
