# wave 20260802-162236 — reviewer 3HFP Euclidean QV-alignment pilot

Dispatched 2026-08-02 16:22 from rig-4090.

Why this wave: first real execution after the pre-data contract and the
10-test implementation-verification gate were completed.

This run: `compute_euclidean_alignment` for
`vit_base_patch16_224.orig_in21k` → rig-4090, CPU-only (`gpu0` is the required
lane identity; the code never invokes CUDA).

The canonical producer `run_id_flat` is 415 characters and cannot be a Linux
directory component. For `scripts/` and mirrored `logs/` only, this run uses:

```text
producer_run_id_sha256=da1865269b54e2e48c978428ed1700f22eda648c0102432f3fcda2d3b4122ec9
```

That digest is SHA-256 of the complete canonical flat identity:

```text
family=ilharco_timm_supervised,model_name=vit_base_patch16_224.orig_in21k,seed=2038,optim=adamw,lr=1e-05,wd=0.1,ls=0.0,wl=500,max_grad_norm=1.0,batch_size=128,qat_bits=3,qat_granularity=channel,qat_skip_modules=%5B%22head%22%5D,ptq_skip_modules=%5B%22head%22%5D,checkpoint_kind=classifier,epoch_policy=dataset_final,vector_scope=quantized_linear_weight,module_selector=apply_ptq_linear_v1,accumulation_dtype=float64
```

The scientific nested evaluation identity is unchanged. Full wave: two
scientific stages, sequentially on the rig-4090 CPU lane, followed by three
render-only visualization commands on the same host.
