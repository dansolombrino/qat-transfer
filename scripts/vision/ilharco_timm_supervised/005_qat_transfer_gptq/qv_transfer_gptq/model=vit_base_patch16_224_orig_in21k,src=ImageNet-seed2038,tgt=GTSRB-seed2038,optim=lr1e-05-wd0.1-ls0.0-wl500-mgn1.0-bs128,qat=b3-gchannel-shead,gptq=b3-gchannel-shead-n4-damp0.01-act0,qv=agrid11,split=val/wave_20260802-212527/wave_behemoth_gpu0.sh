#!/usr/bin/env bash
# run: method=gptq,src=ImageNet,tgt=GTSRB,qv=agrid11,split=val   wave: 20260802-212527   rig: behemoth   gpu: 0
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/dansolombrino/data/PARA/Projects/quantization/qat-transfer"
cd "$PROJECT_ROOT" || exit 1

# Authorization provenance: user explicitly approved behemoth GPUs 0,2,4,5,6,7
# for this ImageNet-only AWQ/GPTQ alpha-sweep wave.
PHYSICAL_GPU=0
BEHEMOTH_AUTHORIZED_GPUS="0,2,4,5,6,7"
case ",$BEHEMOTH_AUTHORIZED_GPUS," in
  *",$PHYSICAL_GPU,"*) ;;
  *) echo "[error] GPU $PHYSICAL_GPU is outside the authorized behemoth set" >&2; exit 64 ;;
esac

ARTIFACTS=(
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.15/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.3/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.45/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.6/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.75/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=0.9/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=1.0/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=1.05/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=1.2/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=1.35/split=val/eval_results.json'
  'evaluations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/vision/qv_transfer_gptq/vit_base_patch16_224_orig_in21k/src=ImageNet_seed=2038/tgt=GTSRB_seed=2038/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/qat=bits=3_gran=channel_skip=head/gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False/qv=alpha=1.5/split=val/eval_results.json'
)
all_done=true
for artifact in "${ARTIFACTS[@]}"; do
  if [ ! -s "$artifact" ]; then all_done=false; break; fi
done
if "$all_done"; then
  echo "[skip] all 11 validation artifacts already exist for gptq/GTSRB"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0
export WAVE_ID=20260802-212527
export HYDRA_FULL_ERROR=1
LOG_DIR='logs/vision/reviewer_3hfp/imagenet_strong_ptq_alpha_sweep/method=gptq,src=ImageNet,tgt=GTSRB,qv=agrid11,split=val/wave_20260802-212527'
mkdir -p "$LOG_DIR" || exit 1

.venv/bin/python code/experiments/vision/ilharco_timm_supervised/005_qat_transfer_gptq/qv_transfer_gptq.py model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-05 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 gpu=0 eval_split=val 'source.dataset_names=["ImageNet"]' source.seed=2038 'target.dataset_names=["GTSRB"]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=["head"]' 'qv.alphas=[0.15,0.3,0.45,0.6,0.75,0.9,1.0,1.05,1.2,1.35,1.5]' gptq.bits=3 gptq.granularity=channel 'gptq.skip_modules=["head"]' gptq.num_calib_batches=4 gptq.percdamp=0.01 gptq.actorder=false gptq.block_size=128 2>&1 \
  | tee "$LOG_DIR/wave_behemoth_gpu0-$(date +%Y%m%d-%H%M%S).log"
pipeline_rc=("${PIPESTATUS[@]}")
python_rc=${pipeline_rc[0]}
tee_rc=${pipeline_rc[1]}
rc=$python_rc
if [ "$tee_rc" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=$tee_rc; fi

missing=0
for artifact in "${ARTIFACTS[@]}"; do
  if [ ! -s "$artifact" ]; then
    echo "[error] missing golden artifact: $artifact" >&2
    missing=$((missing + 1))
  fi
done
if [ "$missing" -ne 0 ] && [ "$rc" -eq 0 ]; then rc=70; fi
exit "$rc"
