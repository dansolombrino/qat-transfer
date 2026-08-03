#!/usr/bin/env bash
# experiment: 998_rebuttal/005_qv_alignment
# stage: analyze_euclidean_alignment
# wave: 20260802-190820   rig: rig-4090   gpu lane identity: 0 (CPU-only run)
set -uo pipefail
cd "$(dirname "$0")/../../../../.." || exit 1

RUN_ID_FLAT="ptq_bits=3,ptq_granularity=channel,outcome_protocol=full_qv,outcome_split=test,unit_alpha=1.0,analysis_spec=reviewer_3hfp_v1,n_permutations=10000,permutation_seed=2038"
PRODUCER_DIR="evaluations/998_rebuttal/005_qv_alignment/euclidean_alignment/family=ilharco_timm_supervised/model_name=vit_base_patch16_224.orig_in21k/seed=2038/optim=adamw/lr=1e-05/wd=0.1/ls=0.0/wl=500/max_grad_norm=1.0/batch_size=128/qat_bits=3/qat_granularity=channel/qat_skip_modules=%5B%22head%22%5D/ptq_skip_modules=%5B%22head%22%5D/checkpoint_kind=classifier/epoch_policy=dataset_final/vector_scope=quantized_linear_weight/module_selector=apply_ptq_linear_v1/accumulation_dtype=float64"
EVAL_DIR="$PRODUCER_DIR/analysis/ptq_bits=3/ptq_granularity=channel/outcome_protocol=full_qv/outcome_split=test/unit_alpha=1.0/analysis_spec=reviewer_3hfp_v1/n_permutations=10000/permutation_seed=2038"
LOG_DIR="logs/998_rebuttal/005_qv_alignment/$RUN_ID_FLAT/wave_20260802-190820"
ARTIFACT="$EVAL_DIR/euclidean_statistics.json"

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"
  exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES="0"
export WAVE_ID="20260802-190820"
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
  ptq_bits=3
  ptq_granularity=channel
  outcome_protocol=full_qv
  outcome_split=test
  unit_alpha=1.0
  analysis_spec=reviewer_3hfp_v1
  n_permutations=10000
  permutation_seed=2038
  'outcome_path="evaluations/998_rebuttal/001_zero_shot_reframing/seed=2038/qat=bits=3_gran=channel/ptq=bits=3_gran=channel/split=test/win_loss_ilharco_timm_supervised.json"'
  use_wandb=false
)

.venv/bin/python \
  code/experiments/998_rebuttal/005_qv_alignment/analyze_euclidean_alignment.py \
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
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = json.loads(path.read_text()) if path.exists() else {}
status.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"))
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(status))
tmp.replace(path)
PY
fi
exit "$rc"
