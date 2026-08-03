#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"
export WAVE_ID=20260803-224359
export SOURCE_TAG=wave--20260803-224359
export SOURCE_REVISION="$(git rev-list -n 1 "$SOURCE_TAG")"
test "$(git rev-parse HEAD)" = "$SOURCE_REVISION"
export CUDA_VISIBLE_DEVICES=0,2
mkdir -p "$(dirname "logs/text/google_gemma3_causallm/001_qat_transfer/model=gemma-3-1b-it/task=gsm8k/mode=full/seed=2038/data_spec=equal6449_v1/train_spec=emnlp2025_fullft_v1/qv_source=gemma-3-1b-it-qat-q4_0/alpha=1.0/quantizer=llamacpp-b9637-q4_0/eval_spec=gemma_gen_v1/wave_20260803-224359/behemoth_gpu0_2.log")"
exec ./.venv/bin/torchrun --master-addr=127.0.0.1 --master-port=29501 --nproc_per_node=2 \
  code/experiments/text/google_gemma3_causallm/001_qat_transfer/run_task.py \
  task="gsm8k" mode="full" \
  > "logs/text/google_gemma3_causallm/001_qat_transfer/model=gemma-3-1b-it/task=gsm8k/mode=full/seed=2038/data_spec=equal6449_v1/train_spec=emnlp2025_fullft_v1/qv_source=gemma-3-1b-it-qat-q4_0/alpha=1.0/quantizer=llamacpp-b9637-q4_0/eval_spec=gemma_gen_v1/wave_20260803-224359/behemoth_gpu0_2.log" 2>&1
