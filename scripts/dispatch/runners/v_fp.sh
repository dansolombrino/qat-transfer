#!/bin/bash
# Vision FP finetune. Usage: v_fp.sh <DATASET> <GPU> <PY> <ROOT> [EPOCH_MULT] [NW]
#
# This runner did not exist before the epoch_mult axis, because every FP
# checkpoint in the repo was produced at the 1x schedule and none ever needed
# re-running. Both budget-controlled experiments need FP at a NON-1 multiplier
# (the QV is QAT - FP, so the two must come from the same budget), hence this.
#
# EPOCH_MULT is given in its canonical form (1, 0.25, 4) -- the same token
# mult_path_frag emits, and accepted verbatim as the Hydra override since
# mult_path_frag(1) and mult_path_frag(1.0) agree. It defaults to 1, which is
# exactly the schedule every existing FP checkpoint used.
#
# Idempotent: finetune_fp.py has no skip_existing of its own, so the guard is
# on the artifact. The mult= component is part of the path, so it must appear
# here too -- otherwise a completed run looks absent and is re-run.
set -u
DS="$1"; GPU="$2"; PY="$3"; ROOT="$4"; MULT="${5:-1}"; NW="${6:-}"

cd "$ROOT" || exit 1

# Exported values win over .env: python-dotenv does not override an existing
# environment variable. Used to raise dataloader workers on the 288-core host,
# where an ImageNet epoch is dataloader-bound, not GPU-bound.
[ -n "$NW" ] && export TORCH_NUM_WORKERS="$NW"

MODEL_SAN=vit_base_patch16_224_orig_in21k
OPTIM="optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128"
CKDIR="$ROOT/storage/checkpoints/vision/ilharco_timm_supervised/fp/$MODEL_SAN/$DS/$OPTIM/mult=$MULT/seed=2038"

if ls "$CKDIR"/classifier_epoch_*.pt >/dev/null 2>&1; then
  echo "V_FP_SKIP ds=$DS mult=$MULT (checkpoint already present)"
  exit 0
fi

"$PY" code/src/vision/ilharco_timm_supervised/finetune_fp.py \
  model_name=vit_base_patch16_224.orig_in21k \
  dataset_name="$DS" \
  seed=2038 \
  gpu="$GPU" \
  batch_size=128 \
  lr=1e-5 \
  wd=0.1 \
  ls=0.0 \
  wl=500 \
  max_grad_norm=1.0 \
  epoch_mult="$MULT" \
  hydra.run.dir="logs/src/vision/ilharco_timm_supervised/finetune_fp/gpu${GPU}/${DS}/mult${MULT}"

# Gate on the artifact, never on an ssh exit status.
if ls "$CKDIR"/classifier_epoch_*.pt >/dev/null 2>&1; then
  echo "V_FP_OK ds=$DS mult=$MULT"
else
  echo "V_FP_FAIL ds=$DS mult=$MULT (no checkpoint at $CKDIR)" >&2
  exit 1
fi
