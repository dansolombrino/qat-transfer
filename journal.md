# Models

## open_clip

| Model | Pretrained |
|---|---|
| ViT-B-16 | laion2b_s34b_b88k |
| ViT-L-14 | laion2b_s32b_b82k |
| ViT-H-14 | laion2b_s32b_b79k |

## timm

| Model | Pretrained |
|---|---|
| deit3_base_patch16_224 | fb_in1k |
| deit3_large_patch16_224 | fb_in1k |
| swin_base_patch4_window7_224 | ms_in22k_ft_in1k |
| swin_large_patch4_window7_224 | ms_in22k_ft_in1k |
| vit_base_patch16_224 | orig_in21k |
| vit_large_patch16_224 | orig_in21k |
| vit_huge_patch14_224 | orig_in21k |

---

# 002 QV Transfer Reversed (timm)

**Status:** Implemented. Reversed QV transfer: computes QV = QAT_src - FP_src (same as 001), but applies it in the opposite direction to the PTQ of the QAT target checkpoint:

```
patched = PTQ(QAT_{S2,Q}^{D2}) - alpha * QV
```

Uses the same `cfg.ptq` config for both the PTQ(QAT_tgt) base construction and the final post-patching PTQ evaluation. Alpha = 1.00. Evaluates with both FP and QAT target heads, with and without PTQ.

## Experiment progress

- [ ] timm — deit3_base_patch16_224.fb_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=deit3_base_patch16_224.fb_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — deit3_large_patch16_224.fb_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=deit3_large_patch16_224.fb_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — swin_base_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=swin_base_patch4_window7_224.ms_in22k_ft_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — swin_large_patch4_window7_224.ms_in22k_ft_in1k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=swin_large_patch4_window7_224.ms_in22k_ft_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — vit_base_patch16_224.orig_in21k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=vit_base_patch16_224.orig_in21k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — vit_large_patch16_224.orig_in21k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=vit_large_patch16_224.orig_in21k batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

- [ ] timm — vit_huge_patch14_224.orig_in21k x 22 datasets (bits=3, channel, skip=head, seed=2038, eval_split=test)
      ```
      uv run --active python code/experiments/vision/ilharco_timm_supervised/002z_qat_transfer_reversed_ptq_after_reverse/qv_transfer.py -m model_name=vit_huge_patch14_224.orig_in21k batch_size=64 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.00 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' eval_split=test gpu=0
      ```

---

# 2026-07-30/31 — bitwidth sweep for the rebuttal: 4-bit and 2-bit

Reviewer question: does QV transfer hold at bitwidths other than 3? Everything in
the paper was `qat.bits=3 / ptq.bits=3, channel`. Decision: **matched-bit**
(train QAT at B bits, deploy PTQ at B bits), α = 1.0, `split=test` only.

Fixed throughout: `seed=2038`, `granularity=channel`.
Vision: `vit_base_patch16_224.orig_in21k`, 22 datasets, `bs=128`, `skip=[head]`.
Text: 4 models × 11 datasets, `bs=32`, `max_length=128`, no warmup.
  `google-bert/bert-base-uncased`, `google-bert/bert-large-uncased` → `skip=[classifier]`
  `google/embeddinggemma-300m`, `Qwen/Qwen3-Embedding-0.6B` → `skip=[score]`
  (AmazonPolarity excluded — commented out of `DATASET_NAME_TO_EPOCHS`.)

## What was RUN

| Modality | Bits | Stage 1 QAT | Stage 2 gate | Stage 3 grid (α=1, test) |
|---|---|---|---|---|
| Vision (timm, 1 backbone) | **4** | 22 runs ✅ | 22 rows ✅ | 22×22 = **484 cells ✅ verified** |
| Text (4 models) | **4** | 44 runs ✅ | 44 rows ✅ | 11×11 ×4 = **484 cells ✅ verified** (121 per model) |
| Vision (timm, 1 backbone) | **2** | 22 runs ✅ | 22 rows ✅ | 22×22 = 484 pairs — **464 on disk**, 20 missing |
| Text (4 models) | **2** | 22 runs ✅ (BERTs only) | 44 rows ✅ | 11×11 ×3 = 363 pairs — **215 on disk**, 148 missing |

Counts above are `find`-verified against the local `evaluations/` tree on 2026-07-31,
not inferred from what was dispatched. Re-verify the same way rather than trusting
the ✅: everything here is written by whichever rig ran the job, so a row can read
"done" locally while a third of it still sits on behemoth or the 3090 Ti.

### 2026-07-31 — text 4-bit grid repair

The text 4-bit grid was **not** the 484 cells this table originally claimed. Two
separate problems, both now fixed:

1. **Never rsynced.** Only 125 of 484 cells were on the 4090; the rest were still
   on behemoth / the 3090 Ti. `scripts/dispatch/gather_results.sh` blocks forever on
   a `RECONCILE COMPLETE` marker in a dead session's scratchpad log, so it never ran
   its two rsync lines. Running those two lines by hand brought the tree to 461.
2. **23 cells genuinely never evaluated.** The gaps were not random — they were one
   receiver per model, truncated mid-sweep (the signature of a runner killed inside
   its inner donor loop):
   - `bert-large-uncased`, `tgt=ToxicConversations` — 9 donors missing
   - `embeddinggemma-300m`, `tgt=IMDB` — 9 donors missing
   - `Qwen3-Embedding-0.6B`, `tgt=ToxicConversations` — 5 donors missing

   All FP and 4-bit QAT checkpoints for those receivers were intact on behemoth, so
   this was 3 `runners/t_qv.sh` invocations (GPUs 4, 5, 2), no retraining. Grid is
   now 121/121 per model = **484**.

Note there is **no `split=val` anywhere in the 4-bit tree** (by design — see "What
was deliberately NOT run"). Any best-α / λ* figure is therefore impossible at 4 bits;
only fixed α=1.0 on `split=test` can be plotted.

## What was REUSED, not re-run

- vision `fp_ptq` + `pretrained_ptq` @ 2 bits — 22 each, pre-existing
- text 2-bit QAT checkpoints for `embeddinggemma-300m` and `Qwen3-Embedding-0.6B` — 11 each
- text matched 2-bit grid for `Qwen3-Embedding-0.6B` — 121 pairs

## What was deliberately NOT run

- **α / λ* sweeps at either bitwidth.** The 3-bit protocol sweeps 11 α values
  (vision) / 40 (text) on `split=val` then reports λ* on test. Skipped: at 4 bits
  Δ_ceiling averages +1.08, so tuning λ to recover a ~1-point ceiling is not worth
  ~2.6× the α=1 cost. Only `qv.alpha=1.0` (the data-free setting) was run.
- 3-bit anything — already on disk.
- Other backbones / families (open_clip, hf_clip, the frozen-ablation timm variant).

## Headline result: the 4-bit Δ_ceiling gate (22/22 vision datasets)

`Δ_ceiling = acc(ptq(QAT)) − acc(ptq(FP))`, the receiver's own QAT gain.

|  | 3-bit | 4-bit |
|---|---|---|
| mean Δ_ceiling | **+45.26** | **+1.08** |
| range | +2.42 … +83.70 | −1.01 … +5.08 |
| datasets with Δ ≤ 0 | 0 | 2 |
| datasets with Δ > 2 pts | 22 | 2 |

At 3 bits PTQ collapses the model and QAT rescues it; at 4 bits PTQ is already
near-lossless so there is almost nothing for a QV to recover. Recovery ratio
`Δ/Δ_ceiling` is ill-conditioned at 4 bits — two denominators are negative.

**Mechanism, visible across all 22:** the only two datasets with real 4-bit
headroom (KMNIST +5.08, EMNIST +4.15) are exactly the two where 4-bit PTQ still
degrades the model (82.67 / 76.41 post-PTQ, vs 89–98 elsewhere). The size of the
QAT benefit tracks how much quantization actually hurts. That reframes the 4-bit
result from "the method fails at 4 bits" to "at 4 bits there is nothing to
recover, and where there is, the method still has room" — a scope boundary with a
mechanism, not a null result.

Suggested framing: the **baseline Δ_ceiling table is the stronger rebuttal
artifact**, with the α=1 grid reported beneath it.

## Infrastructure notes

Ran across behemoth (4× RTX PRO 6000 Blackwell, GPUs 0/2/6/7), rig-4090,
rig-3090-ti. Traps hit and their fixes are captured in
`.claude/skills/multi-rig-dispatch/SKILL.md`; helper scripts in
`scripts/dispatch/` (`rssh.sh`, `reconcile.sh`, `gather_results.sh`,
`final_repair.sh`, `two_bit_pipeline.sh`, `runners/`).

Two worth repeating here:
- behemoth needs **cu129 torch** (`sm_120`); the pinned cu126 build stops at
  `sm_90`. Never run `uv run`/`uv sync` there — use `.venv/bin/python`.
- An ImageNet QAT epoch took **24 h on the 4090** (32 cores, 16 workers) and
  **55 min on behemoth** with `TORCH_NUM_WORKERS=96`. Dataloader-bound, not
  GPU-bound.
