#!/usr/bin/env bash
set -u
PROJECT_ROOT="/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
PHYSICAL_GPU=5
BEHEMOTH_AUTHORIZED_GPUS="0,2,4,5,6,7"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
failures=0
while IFS= read -r job; do
  [ -n "$job" ] || continue
  bash "$job" || failures=$((failures + 1))
done < "scripts/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/wave_20260802-212527/test_behemoth_gpu5.manifest"
if [ "$failures" -ne 0 ]; then exit 1; fi
