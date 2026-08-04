#!/usr/bin/env bash
# run: model=vit_base_patch16_224_orig_in21k,src=PCAM-seed2038,tgt=SVHN-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test   experiment: vision/ilharco_timm_supervised/009_qat_transfer_awq
# wave: 20260804-131438   rig: behemoth   gpu: 4
set -uo pipefail
PROJECT_ROOT="/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1

# Authorization provenance: user approved behemoth GPUs 4,5,6,7 for this
# donor-completion wave; rig-4090 was unreachable at dispatch time.
PHYSICAL_GPU=4
AUTHORIZED_GPUS="4,5,6,7"
case ",$AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] GPU $PHYSICAL_GPU is outside the authorized behemoth set" >&2; exit 64 ;;
esac

RUN_ID_FLAT='model=vit_base_patch16_224_orig_in21k,src=PCAM-seed2038,tgt=SVHN-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test'
EVAL_DIR='evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k/src=PCAM-seed2038/tgt=SVHN-seed2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/qat=b3-gchannel-shead/awq=b3-gchannel-shead-n4-grid20-clip1/qv=a1.0/split=test'
ARTIFACT='evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k/src=PCAM-seed2038/tgt=SVHN-seed2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/qat=b3-gchannel-shead/awq=b3-gchannel-shead-n4-grid20-clip1/qv=a1.0/split=test/eval_results.json'
LOG_DIR='logs/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k,src=PCAM-seed2038,tgt=SVHN-seed2038,optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128,qat=b3-gchannel-shead,awq=b3-gchannel-shead-n4-grid20-clip1,qv=a1.0,split=test/wave_20260804-131438'

if [ -s "$ARTIFACT" ]; then
  echo "[skip] $RUN_ID_FLAT already done (artifact present)"; exit 0
fi
if grep -q '"state": "done"' "$EVAL_DIR/.status.json" 2>/dev/null; then
  echo "[warn] status says done but expected artifact is missing; re-executing $RUN_ID_FLAT" >&2
fi

export CUDA_VISIBLE_DEVICES=4
export WAVE_ID=20260804-131438
export HYDRA_FULL_ERROR=1
mkdir -p "$EVAL_DIR" "$LOG_DIR" || exit 1

.venv/bin/python code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq.py model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 eval_split=test limit_num_batches=null log_to_file=false skip_existing=true 'source.dataset_names=["PCAM"]' source.seed=2038 'target.dataset_names=["SVHN"]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=["head"]' 'qv.alphas=[1.0]' awq.bits=3 awq.granularity=channel 'awq.skip_modules=["head"]' awq.num_calib_batches=4 awq.n_grid=20 awq.clip=true 2>&1 \
  | tee "$LOG_DIR/wave_behemoth_gpu4-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
python_rc=${pipeline_rc[0]}
tee_rc=${pipeline_rc[1]}
rc=$python_rc
if [ "$tee_rc" -ne 0 ]; then
  echo "[error] tee failed with exit code $tee_rc; the required run log is incomplete" >&2
  if [ "$rc" -eq 0 ]; then rc=$tee_rc; fi
fi

if [ ! -s "$ARTIFACT" ]; then
  echo "[error] missing golden artifact: $ARTIFACT" >&2
  if [ "$rc" -eq 0 ]; then rc=70; fi
fi
exit "$rc"
