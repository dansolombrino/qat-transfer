#!/usr/bin/env bash
# run: model=vit_base_patch16_224_orig_in21k,src=ImageNet,tgt=FashionMNIST,sseed=2038,tseed=2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,awq=b3-gchannel-shead-n4-grid20-clip1,alpha=1.0,split=test   wave: 20260802-160930   rig: rig-3090-ti   gpu: 0
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)" || exit 1
cd "$PROJECT_ROOT" || exit 1

RUN_ID_FLAT=model=vit_base_patch16_224_orig_in21k,src=ImageNet,tgt=FashionMNIST,sseed=2038,tseed=2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,awq=b3-gchannel-shead-n4-grid20-clip1,alpha=1.0,split=test
EVAL_DIR=evaluations/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv/model=vit_base_patch16_224_orig_in21k/src=ImageNet/tgt=FashionMNIST/sseed=2038/tseed=2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/awq=b3-gchannel-shead-n4-grid20-clip1/alpha=1.0/split=test
LOG_DIR=logs/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv/model=vit_base_patch16_224_orig_in21k,src=ImageNet,tgt=FashionMNIST,sseed=2038,tseed=2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,awq=b3-gchannel-shead-n4-grid20-clip1,alpha=1.0,split=test/wave_20260802-160930
ARTIFACT=evaluations/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv/model=vit_base_patch16_224_orig_in21k/src=ImageNet/tgt=FashionMNIST/sseed=2038/tseed=2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/awq=b3-gchannel-shead-n4-grid20-clip1/alpha=1.0/split=test/eval_results.json

if [ -e "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES=0
export WAVE_ID=20260802-160930
export HYDRA_FULL_ERROR=1
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

HYDRA_ARGS=(model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 eval_split=test source.dataset_name=ImageNet source.seed=2038 target.dataset_name=FashionMNIST target.seed=2038 qv.alpha=1.0 awq.bits=3 awq.granularity=channel 'awq.skip_modules=["head"]' awq.num_calib_batches=4 awq.n_grid=20 awq.clip=true)
.venv/bin/python code/experiments/vision/ilharco_timm_supervised/010_awq_transfer/qv_transfer_awqv.py "${HYDRA_ARGS[@]}" 2>&1 \
  | tee "$LOG_DIR/wave_rig-3090-ti_gpu0-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
python_rc=${pipeline_rc[0]}
tee_rc=${pipeline_rc[1]}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi
if [ "$rc" -ne 0 ] && [ ! -e "$ARTIFACT" ]; then
  .venv/bin/python - "$EVAL_DIR/.status.json" <<'EOF'
import datetime, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text()) if p.exists() else {}
s.update(state="failed", ended=datetime.datetime.now().isoformat(timespec="seconds"))
t = p.with_suffix(".json.tmp")
t.write_text(json.dumps(s, sort_keys=True) + "\n")
t.replace(p)
EOF
fi
exit "$rc"
