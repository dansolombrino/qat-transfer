#!/usr/bin/env bash
# frozen validation selection; wave 20260802-212527; lane rig-3090-ti_gpu0
set -uo pipefail
PROJECT_ROOT="/mnt/KS_960GB/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
ARTIFACT='evaluations/vision/ilharco_timm_supervised/009_qat_transfer_awq/vision/qv_transfer_awq/model=vit_base_patch16_224_orig_in21k/src=ImageNet-seed2038/tgt=Food101-seed2038/optim=lr1e-05-wd0.1-ls0.0-wl500-mgn1.0-bs128/qat=b3-gchannel-shead/awq=b3-gchannel-shead-n4-grid20-clip1/qv=a0.3/split=test/eval_results.json'
if [ -s "$ARTIFACT" ]; then echo "[skip] selected test artifact exists: $ARTIFACT"; exit 0; fi
export CUDA_VISIBLE_DEVICES=0
export WAVE_ID=20260802-212527
export HYDRA_FULL_ERROR=1
LOG_DIR='logs/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/method=awq,src=ImageNet,tgt=Food101,qv=a0.3,split=test/wave_20260802-212527'
mkdir -p "$LOG_DIR" || exit 1
.venv/bin/python code/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq/qv_transfer_awq.py model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 eval_split=test 'source.dataset_names=["ImageNet"]' source.seed=2038 'target.dataset_names=["Food101"]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=["head"]' 'qv.alphas=[0.3]' awq.bits=3 awq.granularity=channel 'awq.skip_modules=["head"]' awq.num_calib_batches=4 awq.n_grid=20 awq.clip=true 2>&1 \
  | tee "$LOG_DIR/wave_rig-3090-ti_gpu0-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
rc=${pipeline_rc[0]}
if [ "${pipeline_rc[1]}" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=${pipeline_rc[1]}; fi
if [ ! -s "$ARTIFACT" ] && [ "$rc" -eq 0 ]; then rc=70; fi
exit "$rc"
