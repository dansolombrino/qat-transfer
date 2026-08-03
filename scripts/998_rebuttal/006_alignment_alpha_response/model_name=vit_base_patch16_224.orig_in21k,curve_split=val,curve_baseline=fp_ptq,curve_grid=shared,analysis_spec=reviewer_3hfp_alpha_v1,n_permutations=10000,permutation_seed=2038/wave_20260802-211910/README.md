# wave 20260802-211910 — Level-A Euclidean alignment across alpha

Dispatched 2026-08-02 21:19 from rig-4090.

Why this wave: produce the first golden `006_alignment_alpha_response`
statistics by joining the already-complete ViT-B/16 Euclidean QV geometry and
validation alpha curves, with the precommitted shared-permutation QAP analyses.
No model evaluation, checkpoint, CUDA computation, or W&B logging is involved.

This run:
`model_name=vit_base_patch16_224.orig_in21k,curve_split=val,curve_baseline=fp_ptq,curve_grid=shared,analysis_spec=reviewer_3hfp_alpha_v1,n_permutations=10000,permutation_seed=2038`
→ rig-4090, gpu 0 lane identity (CPU-only).

Full wave: one run on rig-4090, gpu 0 lane identity.
