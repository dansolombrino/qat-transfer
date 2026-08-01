#!/usr/bin/env bash
# One text fp_gptq baseline cell (rebuttal WP2).
# Usage: t_fp_gptq.sh <BITS> <DATASET> <GPU> <PY> <ROOT>
# Model is fixed to google-bert/bert-base-uncased (canonical text model). All
# Hydra overrides live here, not in the dispatcher, to keep them out of nested
# ssh quoting (multi-rig-dispatch rule 6).
set -u
BITS="$1"; DS="$2"; GPU="$3"; PY="$4"; ROOT="$5"
cd "$ROOT" || exit 1
"$PY" code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_gptq.py \
  model_name=google-bert/bert-base-uncased \
  dataset_name="$DS" \
  seed=2038 \
  gpu="$GPU" \
  batch_size=32 \
  lr=1e-05 \
  wd=0.1 \
  ls=0.0 \
  max_grad_norm=1.0 \
  max_length=128 \
  'gptq.skip_modules=[classifier]' \
  gptq.bits="$BITS" \
  gptq.granularity=channel \
  hydra.run.dir="logs/dispatch/fp_gptq_text_wave_b${BITS}/${DS}_gpu${GPU}"
echo "T_FP_GPTQ_DONE bits$BITS $DS gpu$GPU"
