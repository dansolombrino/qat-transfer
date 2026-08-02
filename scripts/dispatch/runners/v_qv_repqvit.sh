#!/usr/bin/env bash
# One 006 qv_transfer_repqvit shard (rebuttal WP4, Task 2).
# Usage: v_qv_repqvit.sh <TARGETS_CSV> <GPU> <W_BITS> <A_BITS> <PY> <ROOT>
# All 22 donors are swept against the shard's receivers. Overrides live here,
# not in the dispatcher, so nothing crosses two levels of ssh quoting
# (multi-rig-dispatch rule 6). The script's own skip_existing guard makes this
# safe to re-run after an interruption.
set -u
TGTS="$1"; GPU="$2"; WB="$3"; AB="$4"; PY="$5"; ROOT="$6"
SRCS=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet
cd "$ROOT" || exit 1
"$PY" code/experiments/vision/ilharco_timm_supervised/006_qat_transfer_repqvit/qv_transfer_repqvit.py \
  model_name=vit_base_patch16_224.orig_in21k \
  batch_size=128 \
  lr=1e-5 \
  wd=0.1 \
  ls=0.0 \
  wl=500 \
  max_grad_norm=1.0 \
  "source.dataset_names=[$SRCS]" \
  source.seed=2038 \
  "target.dataset_names=[$TGTS]" \
  target.seed=2038 \
  qat.bits=3 \
  qat.granularity=channel \
  'qat.skip_modules=[head]' \
  'qv.alphas=[0.0,1.0]' \
  'repqvit.skip_modules=[head]' \
  repqvit.w_bits="$WB" \
  repqvit.a_bits="$AB" \
  repqvit.calib_batch_size=32 \
  eval_split=test \
  gpu="$GPU" \
  "hydra.run.dir=logs/dispatch/qv_006_w${WB}a${AB}_gpu${GPU}"
echo "V_QV_REPQVIT_DONE w${WB}a${AB} gpu$GPU"
