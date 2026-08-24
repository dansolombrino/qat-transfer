#!/usr/bin/env bash
# One-shot verification: re-derive every number cited in paper/main.tex from the
# parquet ground truth. Emits four reports under plots/, one per (modality, granularity).
#
# Usage: bash paper/scripts/verify_numbers.sh
#
# Outputs:
#   plots/004_input_fragility/paper_verification_channel.md     (vision, per-channel)
#   plots/004_input_fragility/paper_verification_group_128.md   (vision, per-group_128)
#   plots/text/004_input_fragility/qwen3_summary_channel.md     (Qwen3, per-channel)
#   plots/text/004_input_fragility/qwen3_summary_group_128.md   (Qwen3, per-group_128)
#
# Each vision report also diffs generate_paper_tables.py against committed paper/tables/*.tex.

set -euo pipefail
cd "$(dirname "$0")/../.."

VISION=code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/verify_paper_numbers.py
QWEN3=code/visualizations/text/ilharco_automodelforsequenceclassification/004_input_fragility/analyze_qwen3.py

echo "=== Vision: per-channel ==="
uv run --active python "$VISION" \
  --model-name vit_base_patch16_224.orig_in21k --batch-size 128 \
  --also-model-name vit_large_patch16_224.orig_in21k --also-batch-size 64 \
  --granularity channel \
  --out-path plots/004_input_fragility/paper_verification_channel.md

echo "=== Vision: per-group_128 ==="
uv run --active python "$VISION" \
  --model-name vit_base_patch16_224.orig_in21k --batch-size 128 \
  --also-model-name vit_large_patch16_224.orig_in21k --also-batch-size 64 \
  --granularity group_128 \
  --out-path plots/004_input_fragility/paper_verification_group_128.md

echo "=== Qwen3: per-channel ==="
uv run --active python "$QWEN3" \
  --granularity channel \
  --out-path plots/text/004_input_fragility/qwen3_summary_channel.md

echo "=== Qwen3: per-group_128 ==="
uv run --active python "$QWEN3" \
  --granularity group_128 \
  --out-path plots/text/004_input_fragility/qwen3_summary_group_128.md

echo
echo "Done. Reports:"
echo "  plots/004_input_fragility/paper_verification_channel.md"
echo "  plots/004_input_fragility/paper_verification_group_128.md"
echo "  plots/text/004_input_fragility/qwen3_summary_channel.md"
echo "  plots/text/004_input_fragility/qwen3_summary_group_128.md"
