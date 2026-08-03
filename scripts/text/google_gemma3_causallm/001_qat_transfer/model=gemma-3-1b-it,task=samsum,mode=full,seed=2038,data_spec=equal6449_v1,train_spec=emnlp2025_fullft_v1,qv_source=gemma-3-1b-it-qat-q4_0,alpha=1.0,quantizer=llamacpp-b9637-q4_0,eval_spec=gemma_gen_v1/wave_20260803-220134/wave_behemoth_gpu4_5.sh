#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"
export WAVE_ID=20260803-220134
export SOURCE_TAG=wave--20260803-220134
export SOURCE_REVISION="$(git rev-list -n 1 "$SOURCE_TAG")"
test "$(git rev-parse HEAD)" = "$SOURCE_REVISION"
export CUDA_VISIBLE_DEVICES=4,5
mkdir -p "$(dirname "logs/text/google_gemma3_causallm/001_qat_transfer/model=gemma-3-1b-it/task=samsum/mode=full/seed=2038/data_spec=equal6449_v1/train_spec=emnlp2025_fullft_v1/qv_source=gemma-3-1b-it-qat-q4_0/alpha=1.0/quantizer=llamacpp-b9637-q4_0/eval_spec=gemma_gen_v1/wave_20260803-220134/behemoth_gpu4_5.log")"
exec ./.venv/bin/torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --nproc_per_node=2 \
  code/experiments/text/google_gemma3_causallm/001_qat_transfer/run_task.py \
  task="samsum" mode="full" \
  > "logs/text/google_gemma3_causallm/001_qat_transfer/model=gemma-3-1b-it/task=samsum/mode=full/seed=2038/data_spec=equal6449_v1/train_spec=emnlp2025_fullft_v1/qv_source=gemma-3-1b-it-qat-q4_0/alpha=1.0/quantizer=llamacpp-b9637-q4_0/eval_spec=gemma_gen_v1/wave_20260803-220134/behemoth_gpu4_5.log" 2>&1
