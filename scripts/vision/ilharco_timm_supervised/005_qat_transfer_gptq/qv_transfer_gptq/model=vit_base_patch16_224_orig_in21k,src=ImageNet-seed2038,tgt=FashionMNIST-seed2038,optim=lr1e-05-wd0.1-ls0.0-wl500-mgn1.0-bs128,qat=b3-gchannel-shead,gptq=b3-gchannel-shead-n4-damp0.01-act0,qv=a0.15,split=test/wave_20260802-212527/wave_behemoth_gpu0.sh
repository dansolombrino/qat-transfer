#!/usr/bin/env bash
# frozen validation selection; wave 20260802-212527; lane behemoth_gpu0
set -uo pipefail
PROJECT_ROOT="/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1
PHYSICAL_GPU=0
BEHEMOTH_AUTHORIZED_GPUS="0,2,4,5,6,7"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] unauthorized behemoth GPU $PHYSICAL_GPU" >&2; exit 64 ;;
esac
ARTIFACT='evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=FashionMNIST_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.15/split=test/eval_results.json'
if [ -s "$ARTIFACT" ]; then echo "[skip] selected test artifact exists: $ARTIFACT"; exit 0; fi
export CUDA_VISIBLE_DEVICES=0
export WAVE_ID=20260802-212527
export HYDRA_FULL_ERROR=1
LOG_DIR='logs/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/method=gptq,src=ImageNet,tgt=FashionMNIST,qv=a0.15,split=test/wave_20260802-212527'
mkdir -p "$LOG_DIR" || exit 1
.venv/bin/python code/experiments/vision/ilharco_timm_supervised/005_qat_transfer_gptq/qv_transfer_gptq.py model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 eval_split=test 'source.dataset_names=["ImageNet"]' source.seed=2038 'target.dataset_names=["FashionMNIST"]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=["head"]' 'qv.alphas=[0.15]' gptq.bits=3 gptq.granularity=channel 'gptq.skip_modules=["head"]' gptq.num_calib_batches=4 gptq.percdamp=0.01 gptq.actorder=false gptq.block_size=128 2>&1 \
  | tee "$LOG_DIR/wave_behemoth_gpu0-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
rc=${pipeline_rc[0]}
if [ "${pipeline_rc[1]}" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=${pipeline_rc[1]}; fi
if [ ! -s "$ARTIFACT" ] && [ "$rc" -eq 0 ]; then rc=70; fi
exit "$rc"
