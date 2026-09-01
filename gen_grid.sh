#!/usr/bin/env bash
# Emit the full allocation grid: 6 models x 5 corpora x 7 axis points.
# Stage A = cheap RTN axes (baseline, 2 seeds, 2 budgets)
# Stage B = GPTQ/AWQ stacking on the four small models
# Stage C = GPTQ/AWQ stacking on Qwen 4B/8B (longest pole)
SMALL=("BAAI/bge-large-en-v1.5" "thenlper/gte-large" "intfloat/e5-large-v2" "Qwen/Qwen3-Embedding-0.6B")
BIG=("Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B")
DS=(SciFact NFCorpus FiQA SCIDOCS TRECCOVID)
short(){ case "$1" in *bge*)echo bge;; *gte*)echo gte;; *e5*)echo e5;; *0.6B*)echo qw06;; *4B*)echo qw4;; *8B*)echo qw8;; esac; }

stage_a(){ for m in "${SMALL[@]}" "${BIG[@]}"; do s=$(short "$m"); for d in "${DS[@]}"; do
  echo "A_${s}_${d}_base bash run_alloc2.sh {GPU} A_${s}_${d}_base $m $d rtn 2038 3.5"
  echo "A_${s}_${d}_s101 bash run_alloc2.sh {GPU} A_${s}_${d}_s101 $m $d rtn 101 3.5"
  echo "A_${s}_${d}_s202 bash run_alloc2.sh {GPU} A_${s}_${d}_s202 $m $d rtn 202 3.5"
  echo "A_${s}_${d}_b325 bash run_alloc2.sh {GPU} A_${s}_${d}_b325 $m $d rtn 2038 3.25"
  echo "A_${s}_${d}_b375 bash run_alloc2.sh {GPU} A_${s}_${d}_b375 $m $d rtn 2038 3.75"
done; done; }
stage_b(){ for m in "${SMALL[@]}"; do s=$(short "$m"); for d in "${DS[@]}"; do
  echo "B_${s}_${d}_gptq bash run_alloc2.sh {GPU} B_${s}_${d}_gptq $m $d gptq 2038 3.5"
  echo "B_${s}_${d}_awq  bash run_alloc2.sh {GPU} B_${s}_${d}_awq  $m $d awq  2038 3.5"
done; done; }
stage_c(){ for m in "${BIG[@]}"; do s=$(short "$m"); for d in "${DS[@]}"; do
  echo "C_${s}_${d}_gptq bash run_alloc2.sh {GPU} C_${s}_${d}_gptq $m $d gptq 2038 3.5"
  echo "C_${s}_${d}_awq  bash run_alloc2.sh {GPU} C_${s}_${d}_awq  $m $d awq  2038 3.5"
done; done; }
case "${1:-all}" in a)stage_a;; b)stage_b;; c)stage_c;; all)stage_a;stage_b;stage_c;; esac
