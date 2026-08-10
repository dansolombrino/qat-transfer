#!/usr/bin/env bash
# One PV-Tuning finetune (phase 008 stage B).
# Usage: v_pv.sh <DATASET> <GPU> <PY> <ROOT> <TORCH_NUM_WORKERS|""> <DELTA> <TAU>
#
# Every Hydra override lives here rather than in the dispatcher so nothing
# crosses two levels of ssh quoting (multi-rig-dispatch rule 6).
#
# On delta_decay: values > 0 pull the straight-through buffer back onto the
# grid every P-step by `delta * (grid - B)`, which at this project's lr=1e-5 is
# ~4500x larger than one AdamW step. The buffer's drift then converges to
# (1-delta)*step/delta, far below one scale step, so no code ever moves and the
# backbone stays pinned to the *pretrained* quantization. Measured: delta=0.9
# gives CIFAR10 0.395 vs QAT's 0.868, and SUN397 below even FP+PTQ. Use
# delta=0 unless the learning rate is raised by orders of magnitude.
#
# Idempotent: finetune_pv.py has no skip_existing of its own, and with
# limit_num_epochs unset it writes exactly one classifier_epoch_<N>.pt at the
# end, so the presence of any such file means this cell is complete. That makes
# the wave safe to re-run after an interruption -- and it must be checked on
# the artifact, never on an ssh exit status (rule 2).
set -u
DS="$1"; GPU="$2"; PY="$3"; ROOT="$4"; NW="${5:-}"; DELTA="${6:-0.0}"; TAU="${7:-0.01}"; MULT="${8:-1}"

cd "$ROOT" || exit 1

# Exported values win over .env: python-dotenv does not override an existing
# environment variable. Used to raise dataloader workers on the 288-core host,
# where an ImageNet epoch is dataloader-bound, not GPU-bound (rule 8).
[ -n "$NW" ] && export TORCH_NUM_WORKERS="$NW"

MODEL_SAN=vit_base_patch16_224_orig_in21k
OPTIM="optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128"
# Training-budget multiplier, in its canonical form (1, 0.25, 4) -- the same
# token mult_path_frag emits, and accepted verbatim as the Hydra override since
# mult_path_frag(1) and mult_path_frag(1.0) agree. It is part of the path since
# the epoch_mult axis, so the idempotency check below must carry it, or a
# completed run looks absent and is re-run -- hours of GPU time, silently.
PVFRAG="pv=bits=3_gran=channel_skip=head_delta=${DELTA}_tau=${TAU}_trust=none_pevery=1_temp=0.0"
CKDIR="$ROOT/storage/checkpoints/vision/ilharco_timm_supervised/pv/$MODEL_SAN/$DS/$OPTIM/mult=$MULT/$PVFRAG/seed=2038"

if ls "$CKDIR"/classifier_epoch_*.pt >/dev/null 2>&1; then
  echo "V_PV_SKIP $DS d=$DELTA t=$TAU gpu$GPU (checkpoint already present)"
  exit 0
fi

"$PY" code/src/vision/ilharco_timm_supervised/finetune_pv.py \
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
  pv.bits=3 \
  pv.granularity=channel \
  'pv.skip_modules=[head]' \
  pv.delta_decay="$DELTA" \
  pv.max_code_change_per_step="$TAU" \
  pv.trust_ratio=null \
  pv.p_every=1 \
  pv.temperature=0.0 \
  "hydra.run.dir=logs/dispatch/pv_ft_${DS}_d${DELTA}_t${TAU}_gpu${GPU}"
RC=$?

if ls "$CKDIR"/classifier_epoch_*.pt >/dev/null 2>&1; then
  echo "V_PV_DONE $DS d=$DELTA t=$TAU gpu$GPU rc=$RC"
else
  echo "V_PV_FAIL $DS d=$DELTA t=$TAU gpu$GPU rc=$RC (no checkpoint written)"
fi
