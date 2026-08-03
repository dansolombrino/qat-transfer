#!/usr/bin/env bash
# experiment: 998_rebuttal/005_qv_alignment
# stage: analyze_rowwise_alignment
# wave: 20260803-120627   rig: rig-4090   gpu lane identity: 0 (CPU-only run)
set -uo pipefail
cd "$(dirname "$0")/../../../../.." || exit 1

RUN_ID_FLAT="ptq_bits=3,ptq_granularity=channel,outcome_protocol=full_qv,outcome_split=test,unit_alpha=1.0,analysis_spec=reviewer_3hfp_rowwise_v1,n_permutations=10000,permutation_seed=2038"
PRODUCER_PATH="family=ilharco_timm_supervised/model_name=vit_base_patch16_224.orig_in21k/seed=2038/optim=adamw/lr=1e-05/wd=0.1/ls=0.0/wl=500/max_grad_norm=1.0/batch_size=128/qat_bits=3/qat_granularity=channel/qat_skip_modules=%5B%22head%22%5D/ptq_skip_modules=%5B%22head%22%5D/checkpoint_kind=classifier/epoch_policy=dataset_final/vector_scope=quantized_linear_weight/module_selector=apply_ptq_linear_v1/accumulation_dtype=float64/aggregation_spec=row_cosine_mean_v1"
ANALYZER_PATH="ptq_bits=3/ptq_granularity=channel/outcome_protocol=full_qv/outcome_split=test/unit_alpha=1.0/analysis_spec=reviewer_3hfp_rowwise_v1/n_permutations=10000/permutation_seed=2038"
EVAL_DIR="evaluations/998_rebuttal/005_qv_alignment/rowwise_alignment/$PRODUCER_PATH/analysis/$ANALYZER_PATH"
LOG_DIR="logs/998_rebuttal/005_qv_alignment/$RUN_ID_FLAT/wave_20260803-120627"
ARTIFACT="$EVAL_DIR/rowwise_statistics.json"
export WAVE_ID="20260803-120627"

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"
  exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

SOURCE_TAG="wave--$WAVE_ID"
SOURCE_REVISION=$(git rev-parse "refs/tags/$SOURCE_TAG^{commit}" 2>/dev/null) || {
  echo "[source-drift] missing $SOURCE_TAG" >&2
  exit 86
}
ACTUAL_REVISION=$(git rev-parse HEAD 2>/dev/null) || {
  echo "[source-drift] project is not a Git working tree" >&2
  exit 86
}
if [ "$ACTUAL_REVISION" != "$SOURCE_REVISION" ]; then
  echo "[source-drift] HEAD=$ACTUAL_REVISION expected=$SOURCE_REVISION ($SOURCE_TAG)" >&2
  exit 86
fi
SOURCE_PATHS=(code config scripts visualizations pyproject.toml uv.lock poetry.lock setup.cfg setup.py Pipfile Pipfile.lock requirements*.txt environment*.yml environment*.yaml Dockerfile*)
if ! git diff --quiet -- "${SOURCE_PATHS[@]}" || \
   ! git diff --cached --quiet -- "${SOURCE_PATHS[@]}" || \
   [ -n "$(git ls-files --others --exclude-standard -- "${SOURCE_PATHS[@]}")" ]; then
  echo "[source-drift] execution files differ from $SOURCE_REVISION" >&2
  exit 86
fi
export SOURCE_REVISION SOURCE_TAG
export CUDA_VISIBLE_DEVICES="0"

mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1
HYDRA_ARGS=(
  evaluation_root=evaluations
  family=ilharco_timm_supervised
  model_name=vit_base_patch16_224.orig_in21k
  seed=2038
  optim=adamw
  lr=1e-05
  wd=0.1
  ls=0.0
  wl=500
  max_grad_norm=1.0
  batch_size=128
  qat_bits=3
  qat_granularity=channel
  'qat_skip_modules=["head"]'
  'ptq_skip_modules=["head"]'
  checkpoint_kind=classifier
  epoch_policy=dataset_final
  vector_scope=quantized_linear_weight
  module_selector=apply_ptq_linear_v1
  accumulation_dtype=float64
  aggregation_spec=row_cosine_mean_v1
  ptq_bits=3
  ptq_granularity=channel
  outcome_protocol=full_qv
  outcome_split=test
  unit_alpha=1.0
  analysis_spec=reviewer_3hfp_rowwise_v1
  n_permutations=10000
  permutation_seed=2038
  'outcome_path="evaluations/998_rebuttal/001_zero_shot_reframing/seed=2038/qat=bits=3_gran=channel/ptq=bits=3_gran=channel/split=test/win_loss_ilharco_timm_supervised.json"'
  use_wandb=false
)

.venv/bin/python \
  code/experiments/998_rebuttal/005_qv_alignment/analyze_rowwise_alignment.py \
  "${HYDRA_ARGS[@]}" 2>&1 \
  | tee "$LOG_DIR/wave_rig-4090_gpu0-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
python_rc=${pipeline_rc[0]}
tee_rc=${pipeline_rc[1]}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc; required log is incomplete" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi

if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = json.loads(path.read_text()) if path.exists() else {}
status.update(
    state="failed",
    ended=datetime.datetime.now().isoformat(timespec="seconds"),
    wave_id=os.environ.get("WAVE_ID"),
    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
    source_revision=os.environ.get("SOURCE_REVISION"),
    source_tag=os.environ.get("SOURCE_TAG"),
)
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(status))
tmp.replace(path)
PY
fi
exit "$rc"
