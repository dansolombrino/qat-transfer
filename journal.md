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

# 2026-08-01 — rebuttal WP3: QV transfer under GPTQ (005_qat_transfer_gptq)

New phase `005_qat_transfer_gptq` (Task 2 of `plans/rebuttal_competitor_ptq.md`):
`001_qat_transfer` with the final quantizer swapped from RTN `apply_ptq_` to the
native GPTQ of `code/src/gptq.py` (WP1). Same QV, same patching, same path grammar
with a `gptq=` fragment (bits/gran/skip + ncal/percdamp/actorder; `block_size`
excluded — result-invariant).

## What was DISPATCHED (2026-08-01 11:26, behemoth GPUs 5/6/7)

vit_base_patch16_224.orig_in21k, seed 2038, 3-bit/channel/skip=[head], full 22×22
grid, `qv.alphas=[0.0,1.0]`, `split=test` only, calibration = first 4 train batches
of the receiver, materialized once per receiver and shared by every donor/alpha/head
GPTQ call of that receiver. α=0 runs on the self-pair only (donor-independent): it
IS GPTQ(FP_receiver), the Task-1/Task-2 baseline column, under calibration identical
to the α=1 cells. 506 cells total. tmux sessions `qat_005_full_gpu{5,6,7}`, logs in
`logs/dispatch/005_gptq/`, receivers split by eval cost across the three lanes.
`skip_existing=true` makes relaunch idempotent.

## Smoke result (EuroSAT receiver) — script validated, and an early signal

Pre-quantization accuracies match the 001 RTN cells **bit-for-bit** (patching
pipeline provably identical); only the quantizer differs:

| cell | pre-quant | RTN (001) | GPTQ (005) |
|---|---|---|---|
| DTD→EuroSAT α=1, FP head | 0.9644 | 0.5004 | **0.9604** |
| EuroSAT α=0 (= FP ckpt) | 0.9826 | 0.9707 (`fp_ptq`) | **0.9752** |
| EuroSAT self α=1 (= QAT ckpt) | 0.3356 | 0.9589 | **0.3574** |

Two early observations to test on the full grid:
1. GPTQ rescues the *patched* cross-task model dramatically (+46 pts over RTN on
   DTD→EuroSAT) — but GPTQ(FP) is also strong, so the Task-2 delta
   (α=1 vs α=0 under GPTQ) may be small or negative on easy receivers.
2. GPTQ *hurts* the pure QAT checkpoint (0.357 vs RTN's 0.959). Mechanism: GPTQ's
   objective reconstructs the layer's FP function, and a 3-bit QAT checkpoint's FP
   function is the bad one (FP-forward acc 0.31); RTN instead snaps weights onto
   the grid they were trained for. GPTQ and QAT-style checkpoints are objective-
   mismatched — worth a sentence in the rebuttal.

# 2026-08-01 — rebuttal WP2: GPTQ(FP) baseline (000_baselines/evaluate_fp_gptq)

New baseline scripts (Task 1 of `plans/rebuttal_competitor_ptq.md`):
`evaluate_fp_ptq.py` with the quantizer swapped from RTN `apply_ptq_` to the
native GPTQ of `code/src/gptq.py` (WP1), for BOTH the timm vision family and the
text family (text twin passes a tokenizer-carrying `forward_fn` since text
loaders yield raw `(texts, labels)`). Calibration = first 4 train batches of the
task's own training split; `experiment_type=fp_gptq`; `gptq=` fragment carries
bits/gran/skip + ncal/percdamp/actorder (no `block_size` — result-invariant).

## Smoke (local 4090)

deit3_base/EuroSAT, 3-bit/channel: test_accuracy **0.97963** — matches WP1's
recorded smoke number (0.9796) exactly. RTN `fp_ptq` twin is 0.8237, FP 0.9874.

## DISPATCHED (2026-08-01, behemoth GPUs 0/2/4)

Wave 1: vit_base_patch16_224.orig_in21k × all 22 datasets, seed 2038,
3-bit/channel/skip=[head], defaults ncal=4/percdamp=0.01/actorder=False.
Dispatcher `scripts/dispatch/fp_gptq_wave1.sh` (pull-queue + flock + rssh,
heavy-eval datasets first), runner `runners/v_fp_gptq.sh`. GPUs 5/6/7 are WP3's
005 sweep and were left untouched; output trees are disjoint. Purpose: fill the
000_baselines GPTQ column AND measure per-run wall time to price the full
7-/12-model grid. The wave's cells double as a cross-check of 005's alpha=0
self-pair cells (same recipe, independent code path).

## LANDED (2026-08-01, same day)

22/22 artifacts verified on behemoth and rsynced home (fp_gptq subtree only —
WP3's in-flight trees untouched). Zero failures. Timing: mean 45.6 s/run,
1004 s serial, ~8 min wall on 3 GPUs — a full 22-dataset model sweep costs
~17 min serial. Extrapolated: 7-model canonical grid ≈ 1.5-2 h, all-12-model
grid ≈ 2.5-3 h on GPUs 0/2/4.

Headline: **GPTQ(FP) > RTN(FP) on 22/22 datasets, mean +0.572** on
vit_base_patch16_224.orig_in21k at 3-bit/channel. RTN collapses this backbone
(often near chance); GPTQ restores it to within a few points of FP (e.g.
ImageNet 0.683 vs RTN 0.110, FP 0.744; MNIST 0.984 vs 0.143). Smallest gap:
RenderedSST2 (+0.024), where RTN never collapsed. Cross-check: the EuroSAT
fp_gptq cell (0.9752) equals WP3's independently-computed alpha=0 self-pair
cell bit-for-bit — the two code paths agree.

Implication for the rebuttal: on this backbone the interesting Task-2 question
is entirely "does QV still add gain **under GPTQ**" (005's alpha=1 vs alpha=0),
since GPTQ(FP) alone already closes most of the RTN gap that QV+RTN was
closing. Task-1 framing must lead with complementarity, not competition.

## Text wave LANDED (2026-08-01, same day)

bert-base-uncased x 11 active text datasets (AmazonPolarity is retired from
DATASET_NAME_TO_EPOCHS — stale FP ckpt, no RTN twin; dropped from the queue),
3-bit/channel/skip=[classifier], behemoth GPUs 0/2/4. 11/11 artifacts, zero
failures, mean 38.9 s/run (~4 min wall).

**GPTQ(FP) > RTN(FP) on 11/11, mean +0.100.** Text is far less RTN-fragile
than the vit_base vision backbone (+0.572): BERT under 3-bit RTN loses points,
not orders of magnitude, so GPTQ's headroom is smaller — largest gain
Banking77 (+0.354), smallest AmazonCounterfactual (+0.006); on
ToxicConversations GPTQ even edges out FP (0.9460 vs 0.9428). Same rebuttal
implication as vision, weaker form: GPTQ(FP) is a strong Task-1 column, so the
QV story leans on complementarity (Task 2).

## 2-bit wave LANDED (2026-08-01, same day)

Same two models, same FP checkpoints (bit-independent), gptq.bits=2, mixed
33-run queue on behemoth GPUs 0/2/4; 33/33, zero failures, ~42 s/run (runners
now take BITS as arg 1; dispatcher scripts/dispatch/fp_gptq_wave_b2.sh).

The 3->2 bit cliff is the story:

- **Vision (vit_base orig_in21k):** RTN2 is chance (mean 0.103 vs chance
  0.092). GPTQ2 beats RTN2 on 21/22 and beats chance on 22/22, but its mean
  (0.289) is nowhere near GPTQ3 (0.791). Fine-grained tasks (ImageNet 0.003,
  Cars 0.009, SUN397 0.011) stay destroyed; only coarse/MNIST-like tasks
  survive partially (MNIST 0.794, FashionMNIST 0.678, EuroSAT 0.618). Sole
  GPTQ2<RTN2 loss: PCAM (0.530 vs 0.630, binary task near chance).
- **Text (bert-base):** same shape, softer — GPTQ2 0.363 vs RTN2 0.240 vs
  GPTQ3 0.835; 11/11 wins but most many-class tasks stay collapsed.

Reading: 2-bit one-shot PTQ is beyond GPTQ's error-compensation reach on
these models (consistent with the GPTQ paper's sub-3-bit behavior). This is
the regime where trained robustness (QAT / QV patching) has maximal headroom
over any calibration-only method — the strongest possible Task-2 setting if
005 is extended to 2-bit.

## 2026-08-01 — WP3 sweep landed (506/506)

All three lanes finished clean (11:26–16:5x wall-clock, behemoth GPUs 5/6/7, zero
failures; the only log traceback was a benign multiprocessing temp-dir cleanup
race at a receiver transition). JSONs gathered back to the 4090 tree.

**Full-grid Task-2 headline** (fp head, α=1 cross-task vs the receiver's α=0
GPTQ(FP) baseline, identical calibration): mean Δ = −3.2 pts, median −2.2,
win rate 9.3 % (43/462), best cell +2.8, worst −32.6 (KMNIST receivers hurt
most; RenderedSST2 the only receiver with positive mean). **At λ=1, QV patching
does not add gain on top of GPTQ.** The smoke-run observations held at scale:

- Off-diagonal, QV+GPTQ ≫ QV+RTN (GPTQ rescues patched models RTN collapses),
  and GPTQ(FP) ≫ RTN fp_ptq everywhere (e.g. RESISC45 0.887 vs 0.286) — the
  Task-1 comparison lives in a much stronger regime than the paper's RTN world.
- Diagonal (= the receiver's own QAT checkpoint): GPTQ *destroys* it
  (≈FP-forward accuracy) — objective mismatch, GPTQ(QAT) ≈ QAT-in-FP. The QAT
  ceiling under GPTQ framing must remain RTN(QAT) (= 001's qat_ptq), never
  GPTQ(QAT).

Rebuttal framing note: the honest Task-2 sentence is "under GPTQ there is far
less accuracy left to recover, and λ=1 patching does not recover it" — mirroring
the 4-bit Δ_ceiling scope boundary (see the bitwidth-sweep entry): QV's benefit
tracks how much the quantizer actually hurts. A λ* sweep under GPTQ (val split)
remains the open question if the negative-at-λ=1 result needs softening.

Figures: `code/visualizations/vision/ilharco_timm_supervised/005_qat_transfer_gptq/`
(`qv_transfer_heatmap.py`: raw + Δ vs GPTQ(FP) + Δ vs RTN fp_ptq;
`qv_transfer_rtn_heatmap_minus_gptq_fp.py`: 001 QV+RTN − GPTQ(FP), the Task-1 view).

# 2026-08-01 — phase 007: is GPTQ itself transferrable? (007_gptq_transfer)

Follow-up question after WP3: define `QV_gptq = GPTQ(FP_donor) − FP_donor` — a
pure quantization displacement, no training — patch `FP_tgt + α·QV_gptq`, and
evaluate **raw** (no quantizer after patching; the displacement IS the
quantization under test). Diagonal at α=1 is algebraically GPTQ(FP_tgt).

Two new Hydra scripts under `007_gptq_transfer/` (phase 006 was taken mid-day by
WP4's `006_qat_transfer_repqvit`): `compute_gptq_checkpoints.py` materializes
GPTQ(FP) checkpoints into a new `storage/.../gptq/` subtree paralleling fp/qat
(GPTQ ckpts existed nowhere on disk — every prior consumer quantized in-memory
at eval time), and `qv_transfer_gptqv.py` (001-shaped, single FP-head raw eval,
`qv.alphas` list + resume guard; α=0 deliberately absent — it equals the
recorded fp baseline).

## Dispatched (2026-08-01 18:42 / 18:55, behemoth)

- Step A: 22 GPTQ(FP) checkpoints, GPU 0, 22/22 saved, zero errors.
- Smoke: EuroSAT diagonal = 0.9751852 — **bit-identical** to 005's α=0
  GPTQ(FP) cell (same materialized calibration batches by construction:
  same seed, same loader, same batch count). DTD→EuroSAT raw = 0.9252.
- Step B: full 22×22, α=1, test split, GPUs 0/2/4/7 (WP4's RepQ-ViT sweep
  still held 5/6 at launch), tmux `qat_007_full_gpu{0,2,4,7}`, logs under
  `logs/dispatch/007_gptq/`.

Early signal from the smoke: a *foreign* GPTQ displacement applied raw costs
only ~5 pts vs the receiver's own GPTQ run (0.9252 vs 0.9752 on EuroSAT) —
GPTQ displacement transfer may be surprisingly real. Full grid will tell.

## 2026-08-01 — 007 landed (484/484): GPTQ is NOT transferrable the way QAT is

All lanes clean (GPUs 0/2/4/5/6/7 as WP4 freed them, ~18:55–20:1x). Full-grid
verdict (fp head, raw eval, 462 cross-task cells):

- vs the receiver's OWN GPTQ run: mean gap **−16.7 pts** (median −15.1,
  min −58.5, best +1.6); only 5 % of cells within 2 pts.
- vs the receiver's FP checkpoint (the cost of applying a foreign
  displacement): mean −19.4 pts.
- Worst receivers: EMNIST (−35), KMNIST (−34), OxfordIIITPet (−26);
  most tolerant: STL10/CIFAR10 (−5), RenderedSST2 (−0).

Interpretation: QV_gptq is weight-conditioned, not task-agnostic. GPTQ's error
feedback is computed for the donor's exact weight matrix and calibration
Hessian; added to a different task's weights the compensation lands in the
wrong coordinates, and the patched model is neither on a grid nor
error-compensated. This is a clean *negative control* for the paper's claim:
the QAT QV transfers because QAT learns a task-agnostic robustness direction,
not because any quantization displacement happens to transfer — a
displacement of near-identical norm (GPTQ's) does not. The EuroSAT diagonal
reproduced 005's GPTQ(FP) bit-exactly (0.9751852), validating the pipeline.


# 2026-08-01/02 — 008_pv_transfer: does a stronger finetuner give a better QV?

Every QV in this repo is `QAT_D - FP_D`, and every QAT checkpoint behind it
came from one finetuner: STE. The central claim says nothing about *how* the
QAT optimum was found, so this phase swaps the finetuner and nothing else.
PV-Tuning (Malinovskii et al., NeurIPS'24, arXiv 2405.14852; reference vendored
at `references/AQLM/`) exists precisely because STE is a biased estimator at
extreme compression, which is the 3-bit regime we care about.

Ported to this repo's uniform symmetric grid, P = the integer codes, V = the
straight-through buffer plus every non-quantized parameter (the per-channel
scale stays derived, not learned — that is what keeps `apply_ptq_` a no-op on
the saved checkpoint). At `delta_decay=0, max_code_change_per_step=1` PVLinear
is **bitwise** QATLinear, so existing QAT is a corner of the PV knob grid; that
is a free regression test and was verified end-to-end on a 4-step ViT-B/16 run
(identical straight-through buffer, identical codes on all 48 layers).

## Two corrections the runs forced

**delta_decay > 0 freezes training.** AQLM's LLM regime uses delta=0.9. Here
the pull-to-grid term is O(delta * scale) ~= 0.045 per step against an AdamW
step of ~1e-5 at lr=1e-5 — about 4500x — so the buffer's drift converges to
(1-delta)*step/delta, far below one scale step, no code ever moves, and the
backbone stays pinned to the *pretrained* quantization. Measured before the
sweep was killed at 19/22: CIFAR10 0.395 vs QAT's 0.868, MNIST 0.499 vs 0.980,
and SUN397 0.031 — below even FP+PTQ's 0.054. Rejected. The sweep was redone at
delta=0, where tau is the only knob and tau=1 is exactly QAT. Those delta=0.9
checkpoints are kept as a documented negative control.

**The QV must come from the latent buffer, not the saved checkpoint.** A PV
checkpoint stores settled `q*s` weights, so `PV_ckpt - FP` is dominated by
quantization rounding error rather than by anything PV learned: on MNIST its
norm is 293.8 against the QAT QV's 12.4, with cosine 0.035 — essentially
orthogonal. The control settles a *QAT* checkpoint and reproduces the same
orthogonality with no PV involved, which is what identifies it as an artifact.
The `pv_state_epoch_N.pt` sidecar's straight-through buffer is the exact
analogue of what a QAT checkpoint stores: norm 12.37, cosine 0.909. That is the
comparable object, and `qv.weights: latent` is now the default. Had this gone
unnoticed the phase would have produced a confident, meaningless heatmap.

## Checkpoint contract, verified on hardware

`finetune_pv.py` settles onto the grid before saving, so `apply_ptq_` must
recover the same codes. Across all 22 real ViT-B/16 checkpoints: **0 of ~1.87e9
codes changed**, and `pv` / `pv_ptq` report identical accuracies on 22/22.
`ptq_max_abs_weight_delta` comes back at ~1-2 ulp (2.4e-07) because
`scale = absmax/qmax` round-trips through a CUDA division that is not
bit-identical to the CPU's; it is exactly 0 on CPU. An earlier version of the
check warned on `delta != 0` and fired on every healthy run — the code count is
the invariant, the weight delta is context.

## Result — a null, and it is the useful kind

Full 22x22 grid, 506/506 cells, vit_base_patch16_224.orig_in21k, seed 2038,
3-bit/channel/skip=[head], alpha=1, split=test, delta=0/tau=0.01. Every
receiver landed exactly 23 cells.

    cross-task (transfer)   n=462   mean PV-QAT = -0.0020   PV better 43.1%
    same-task (ceiling)     n=22    mean PV-QAT = -0.0053   PV better 45.5%
    win rate vs fp_ptq              PV 344/462   QAT 354/462

The mean was stable at -0.002..-0.004 across n=25, 91, 108, 210, 226, 420, 462,
and the win rate never left 41-48%. **PV-Tuning does not produce a
better-transferring QV** — marginally worse, on a scale where the QAT-vs-FP
ceiling gap is ~45 points.

That is a negative result for PV and a positive one for the paper's claim: the
transferable content is a property of the quantization grid that any
quantization-aware finetuner recovers, not an artifact of STE's particular
optimum. It is corroborated by the two QVs being 0.909-cosine aligned at equal
norm — measured before the grid ran. The same-task row rules out the obvious
confound: PV's own ceiling is also slightly below QAT's, so the cross-task
deficit tracks a marginally worse optimum rather than a transfer-specific
weakness. tau=0.01 acts as a mild handicap at lr=1e-5, not an improvement.

Caveat on tau: 0.01 and 0.1 differed by 0.01% of codes on MNIST and gave
identical accuracy, so tau is likely not load-bearing at this learning rate.
The tau=0.1 finetunes exist (22/22) but its transfer grid was not run.

## Infrastructure

44 finetunes across behemoth GPUs 0/2/4/5/6/7 + rig-4090 + rig-3090-ti (~2 h);
the 506-cell grid on behemoth's 6 GPUs (~2 h 17 m); 44 baselines (~15 min).
rig-3090-ti had to be provisioned first (no `CHECKPOINT_BASE_PATH`, no
`storage/`, unrelated git history) and all three rigs were rsynced to identical
code, since a version skew across hosts silently produces inconsistent cells.
The transfer grid ran on behemoth alone: a cell needs FP + PV + sidecar for
*every* donor co-located, and `qv_transfer` returns 0 while skipping donors
whose checkpoints are absent (multi-rig-dispatch rule 3), so a host with a
partial donor set emits quietly incomplete rows.
