#!/usr/bin/env bash
# run: model=vit_base_patch16_224_orig_in21k,src=GTSRB-seed2038,tgt=PCAM-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test   experiment: vision/ilharco_timm_supervised/009_qat_transfer_awq
# wave: 20260803-140339   rig: rig-4090   gpu: 0
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || exit 1
cd "$PROJECT_ROOT" || exit 1

RUN_ID_FLAT=model=vit_base_patch16_224_orig_in21k,src=GTSRB-seed2038,tgt=PCAM-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test
EVAL_DIR=evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k/src=GTSRB-seed2038/tgt=PCAM-seed2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/qat=b3-gchannel-shead/awq=b3-gchannel-shead-n4-grid20-clip1/qv=a1.0/split=test
LOG_DIR=logs/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k,src=GTSRB-seed2038,tgt=PCAM-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test/wave_20260803-140339
ARTIFACT=evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k/src=GTSRB-seed2038/tgt=PCAM-seed2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/qat=b3-gchannel-shead/awq=b3-gchannel-shead-n4-grid20-clip1/qv=a1.0/split=test/eval_results.json
export WAVE_ID=20260803-140339

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

SOURCE_TAG="wave--$WAVE_ID"
SOURCE_REVISION=$(git rev-parse "refs/tags/$SOURCE_TAG^{commit}" 2>/dev/null) || {
  echo "[source-drift] missing $SOURCE_TAG" >&2; exit 86;
}
ACTUAL_REVISION=$(git rev-parse HEAD 2>/dev/null) || {
  echo "[source-drift] project is not a Git working tree" >&2; exit 86;
}
if [ "$ACTUAL_REVISION" != "$SOURCE_REVISION" ]; then
  echo "[source-drift] HEAD=$ACTUAL_REVISION expected=$SOURCE_REVISION ($SOURCE_TAG)" >&2
  exit 86
fi
SOURCE_PATHS=(code config scripts pyproject.toml uv.lock poetry.lock setup.cfg setup.py Pipfile Pipfile.lock requirements*.txt environment*.yml environment*.yaml Dockerfile*)
if ! git diff --quiet -- "${SOURCE_PATHS[@]}" ||    ! git diff --cached --quiet -- "${SOURCE_PATHS[@]}" ||    [ -n "$(git ls-files --others --exclude-standard -- "${SOURCE_PATHS[@]}")" ]; then
  echo "[source-drift] execution files differ from $SOURCE_REVISION" >&2
  exit 86
fi
export SOURCE_REVISION SOURCE_TAG

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

HYDRA_ARGS=('model_name="vit_base_patch16_224.orig_in21k"' batch_size=128 'eval_split="test"' limit_num_batches=null log_to_file=false skip_existing=true lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 'source.dataset_names=["GTSRB"]' source.seed=2038 source.limit_num_epochs=null 'target.dataset_names=["PCAM"]' target.seed=2038 target.limit_num_epochs=null qat.bits=3 'qat.granularity="channel"' 'qat.skip_modules=["head"]' 'qv.alphas=[1.0]' 'awq.skip_modules=["head"]' awq.bits=3 'awq.granularity="channel"' awq.num_calib_batches=4 awq.n_grid=20 awq.clip=true)
.venv/bin/python code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq.py "${HYDRA_ARGS[@]}" 2>&1   | tee "$LOG_DIR/wave_rig-4090_gpu0-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
python_rc=${pipeline_rc[0]}
tee_rc=${pipeline_rc[1]}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc; the required run log is incomplete" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi

if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'EOF'
import datetime, json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text()) if p.exists() else {}
s.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"),
         wave_id=os.environ.get("WAVE_ID"), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
         source_revision=os.environ.get("SOURCE_REVISION"),
         source_tag=os.environ.get("SOURCE_TAG"))
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(s, sort_keys=True) + "\n")
t.replace(p)
EOF
fi
exit "$rc"
