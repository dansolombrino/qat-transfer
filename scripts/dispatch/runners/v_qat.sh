#!/bin/bash
# Vision QAT finetune at arbitrary bits.
# Usage: v_qat.sh <BITS> <DATASET> <GPU> <PY> <ROOT> [EPOCH_MULT] [NW]
#
# EPOCH_MULT scales this dataset's entry in DATASET_NAME_TO_EPOCHS. It defaults
# to 1.0, which is exactly the schedule every existing run used, so omitting it
# reproduces the previous behaviour byte for byte.
#
# Idempotent, and verified on the artifact rather than on an exit status. Both
# properties matter for a long unattended wave: a worker that is restarted after
# a reboot must skip what already landed, and a dropped ssh returns non-zero
# *after* the remote work finished, so exit status alone would record a complete
# run as failed.
set -u
BITS=$1; DS=$2; GPU=$3; PY=$4; ROOT=$5; MULT=${6:-1.0}; NW=${7:-}

cd "$ROOT" || exit 1

# Exported values win over .env: python-dotenv does not override an existing
# environment variable. Used to raise dataloader workers on the 288-core host.
[ -n "$NW" ] && export TORCH_NUM_WORKERS="$NW"

MODEL_SAN=vit_base_patch16_224_orig_in21k
OPTIM="optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128"

# Resolve the same variable the Python writer resolves -- the checkpoint tree is
# not always under $ROOT (rig-3090-ti keeps it on a separate mount).
CB=$(grep -E '^CHECKPOINT_BASE_PATH=' "$ROOT/.env" 2>/dev/null | tail -1 | cut -d= -f2-)
[ -z "${CB:-}" ] && CB="$ROOT/storage/checkpoints"
QFRAG="qat=bits=${BITS}_gran=channel_skip=head"
CKDIR="$CB/vision/ilharco_timm_supervised/qat/$MODEL_SAN/$DS/$OPTIM/mult=$MULT/$QFRAG/seed=2038"

if ls "$CKDIR"/classifier_epoch_*.pt >/dev/null 2>&1; then
  echo "V_QAT_SKIP bits=$BITS ds=$DS mult=$MULT (checkpoint already present)"
  exit 0
fi

$PY code/src/vision/ilharco_timm_supervised/finetune_qat.py \
  model_name=vit_base_patch16_224.orig_in21k dataset_name="$DS" seed=2038 gpu="$GPU" \
  qat.bits="$BITS" qat.granularity=channel 'qat.skip_modules=[head]' \
  epoch_mult="$MULT" \
  hydra.run.dir="logs/src/vision/ilharco_timm_supervised/finetune_qat/bits${BITS}_gpu${GPU}/${DS}/mult${MULT}"

# Gate on the artifact, never on an ssh exit status.
if ls "$CKDIR"/classifier_epoch_*.pt >/dev/null 2>&1; then
  echo "V_QAT_OK bits=$BITS ds=$DS mult=$MULT"
else
  echo "V_QAT_FAIL bits=$BITS ds=$DS mult=$MULT (no checkpoint at $CKDIR)" >&2
  exit 1
fi
