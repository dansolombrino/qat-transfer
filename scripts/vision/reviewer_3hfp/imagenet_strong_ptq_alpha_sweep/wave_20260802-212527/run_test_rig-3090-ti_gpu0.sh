#!/usr/bin/env bash
set -u
PROJECT_ROOT="/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
failures=0
while IFS= read -r job; do
  [ -n "$job" ] || continue
  bash "$job" || failures=$((failures + 1))
done < "scripts/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/wave_20260802-212527/test_rig-3090-ti_gpu0.manifest"
if [ "$failures" -ne 0 ]; then exit 1; fi
