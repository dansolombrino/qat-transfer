#!/usr/bin/env bash
# wave 20260804-134500; behemoth GPU 4; user-approved behemoth set 0,2,4,5,6,7
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
PHYSICAL_GPU=4
AUTHORIZED_GPUS="0,2,4,5,6,7"
case ",$AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
MANIFEST="$SCRIPT_DIR/behemoth_gpu4.manifest"
FAILURES="$SCRIPT_DIR/behemoth_gpu4.failures"
: > "$FAILURES"
failures=0
completed=0
total=19
while IFS= read -r job; do
  [ -n "$job" ] || continue
  echo "[$(date --iso-8601=seconds)] lane behemoth_gpu4: $((completed + 1))/$total $job"
  bash "$job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] FAILED rc=$rc: $job" >&2
    echo "$job" >> "$FAILURES"
    failures=$((failures + 1))
  fi
  completed=$((completed + 1))
done < "$MANIFEST"
echo "[$(date --iso-8601=seconds)] lane behemoth_gpu4: completed=$completed failures=$failures"
if [ "$failures" -ne 0 ]; then exit 1; fi
