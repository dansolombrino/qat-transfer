#!/usr/bin/env bash
# One 008 qv_transfer_pv shard: a single receiver, swept against all 22 donors.
# Usage: v_qv_pv.sh <TARGET> <GPU> <PY> <ROOT> <TAU> <SPLIT> [TORCH_NUM_WORKERS] [SRC_MULT] [TGT_MULT]
#
# SRC_MULT / TGT_MULT (canonical form: 1, 0.25, 4) are independent -- that is the
# point of the epoch_mult axis, since donor and receiver share one optim= fragment
# and would otherwise have nowhere to disagree. Both default to 1.
#
# Overrides live here, not in the dispatcher, so nothing crosses two levels of
# ssh quoting (multi-rig-dispatch rule 6). The script's own skip_existing guard
# makes this safe to re-run after an interruption.
#
# qv.weights=latent is the whole point of the phase: the QV is built from the
# donor's straight-through buffer in pv_state_epoch_N.pt, not from the settled
# q*s checkpoint. Building it from the checkpoint yields a vector 24x larger
# than the QAT QV and orthogonal to it (cosine 0.035), because it is dominated
# by quantization rounding error rather than by anything PV learned.
set -u
TGT="$1"; GPU="$2"; PY="$3"; ROOT="$4"; TAU="$5"; SPLIT="$6"; NW="${7:-}"; SMULT="${8:-1}"; TMULT="${9:-1}"

SRCS=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet

cd "$ROOT" || exit 1
[ -n "$NW" ] && export TORCH_NUM_WORKERS="$NW"

"$PY" code/experiments/vision/ilharco_timm_supervised/008_pv_transfer/qv_transfer_pv.py \
  model_name=vit_base_patch16_224.orig_in21k \
  batch_size=128 \
  lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 \
  "source.dataset_names=[$SRCS]" source.seed=2038 \
  "target.dataset_names=[$TGT]" target.seed=2038 \
  pv.bits=3 pv.granularity=channel 'pv.skip_modules=[head]' \
  pv.delta_decay=0.0 pv.max_code_change_per_step="$TAU" \
  pv.trust_ratio=null pv.p_every=1 pv.temperature=0.0 \
  'qv.alphas=[0.0,1.0]' \
  qv.weights=latent \
  ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' \
  eval_split="$SPLIT" \
  gpu="$GPU" \
  "hydra.run.dir=logs/dispatch/qv008_${TGT}_t${TAU}_${SPLIT}_gpu${GPU}"
RC=$?

# Ground truth is the artifact count, never the exit status: qv_transfer
# returns 0 even when it skipped donors for missing checkpoints (rule 3).
EV="$ROOT/evaluations/vision/ilharco_timm_supervised/008_pv_transfer/vision/qv_transfer_pv"
N=$(find "$EV" -path "*tgt=${TGT}_seed=2038*" -path "*tau=${TAU}_*" -path "*split=${SPLIT}*" -name eval_results.json 2>/dev/null | wc -l)
echo "V_QVPV_DONE $TGT tau=$TAU split=$SPLIT gpu$GPU rc=$RC cells=$N"
