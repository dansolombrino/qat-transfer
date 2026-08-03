# Reviewer 3HFP ImageNet-only strong-PTQ alpha sweep

Wave `20260802-212527` contains 42 validation jobs: 21 receivers times AWQ/GPTQ.
Each job evaluates the paper grid `[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.05, 1.2, 1.35, 1.5]`, giving 462 validation cells.
After validation, choose the best alpha independently for each method/receiver
using the corresponding strong-PTQ FP-head metric, then evaluate only those 42
frozen choices on test. ImageNet is donor-only; the self-pair is excluded.

Placement uses rig-3090-ti GPU 0 and behemoth GPUs `0,2,4,5,6,7`.
Rig-4090 was excluded because its NVIDIA driver was unavailable at preflight.
The user explicitly authorized the six named behemoth GPUs for this wave.
