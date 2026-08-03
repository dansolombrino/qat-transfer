#!/usr/bin/env bash
# wave 20260802-212527; behemoth GPU 5; user-approved behemoth set 0,2,4,5,6,7
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
PHYSICAL_GPU=5
BEHEMOTH_AUTHORIZED_GPUS="0,2,4,5,6,7"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
failures=0
completed=0
total=$(wc -l < "$SCRIPT_DIR/behemoth_gpu5.manifest")
while IFS= read -r job; do
  [ -n "$job" ] || continue
  echo "[$(date --iso-8601=seconds)] lane behemoth_gpu5: $((completed + 1))/$total $job"
  bash "$job"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date --iso-8601=seconds)] FAILED rc=$rc: $job" >&2
    failures=$((failures + 1))
  fi
  completed=$((completed + 1))
done < "$SCRIPT_DIR/behemoth_gpu5.manifest"
echo "[$(date --iso-8601=seconds)] lane behemoth_gpu5: completed=$completed failures=$failures"
if [ "$failures" -ne 0 ]; then exit 1; fi
