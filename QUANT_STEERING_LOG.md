# Quant-steering research log

Project-level running journal for the `quant-steering` branch. Claude updates
this file as findings emerge. Future sessions should read it first to see
what's been done and what the current state of the hypothesis is.

**Branch:** `quant-steering` (off `master`)
**Last updated:** 2026-05-28

---

## Hypothesis under test

Weight-only PTQ degrades model accuracy non-uniformly across inputs. Among
inputs the FP model classifies correctly, some are preserved by the quantized
model ("quant-robust") and others get flipped ("quant-fragile"). Adding a
steering vector to the residual stream at one ViT block at inference might
flip the fragile inputs back. The deeper question is whether a **single
shared direction** generalizes across tasks — i.e. whether quant-fragility is
a property of `(model, bits)` or of `(model, bits, task)`.

---

## Key findings (chronological)

### F1 [2026-05-28] — per-task steering at W4-channel is weak across most blocks

After fitting `mean_diff` and `contrastive_svd` steering vectors on 21 finetuned
ViT-B/`orig_in21k` tasks at W4-channel PTQ (skip=[head]), the cross-task
pairwise cosine was weak everywhere: best signed mean cosine ≤ 0.085,
best |cos| ≤ 0.255 at block 0. 3 tasks dropped (Flowers102/OxfordIIITPet/STL10)
because W4-channel was too gentle to produce enough quant-broken inputs.

Verdict at this stage: "sign-flipped weakly shared" — alive but marginal.

### F2 [2026-05-28] — W4-tensor block 6 is a smoking gun

Repeating the fit across the 4-PTQ grid {W3,W4} × {channel,tensor} surfaced a
dramatically stronger configuration:

| Config | Best |cos| (mean_diff) | Block | Tasks loaded |
|---|---|---|---|
| W3-channel | 0.332 | 0 | 21/21 |
| W4-channel | 0.251 | 0 | 18/21 |
| W3-tensor | 0.427 | 0 | 21/21 |
| **W4-tensor** | **0.831** | **6** | **21/21** |

Tensor-wise PTQ is much harsher than channel-wise — produces many bad samples
per task → no skipped tasks. The shared block-6 mid-network axis is where the
universal direction lives at W4-tensor.

### F3 [2026-05-28] — at W4-tensor block 6, sign is task-noise, axis is universal

Direct sign-aware analysis of the 21 W4-tensor block-6 `mean_diff` vectors:
- 20 of 21 tasks lie within |cos| > 0.7 of any other in the cluster.
- Mean within-cluster |cos| = 0.918 (k=2 hierarchical clustering).
- Flowers102 is the only outlier (orthogonal, cos ≈ 0 to everything).
- Sign split is 10 positive / 10 negative vs CIFAR10 reference, with **no**
  obvious semantic structure (MNIST is +, EMNIST is −; CIFAR10 is +, CIFAR100
  is −; KMNIST/FashionMNIST are +, SVHN is −).

Sign is therefore noise from how `v_mean = mean(good) − mean(bad)` happens to
fall per-task. After sign-alignment + average, the universal vector has |cos|
≥ 0.72 with every cluster-1 task (min EuroSAT 0.72, max TinyImageNet 0.997,
mean 0.92).

**This is the regime where Approach A (sign-align then average) should work.**

### F4b [2026-05-28] — universal direction spans blocks 6–9, not just block 6

After running `combine_steering_vectors.py --combiner sign_align_average` over
the W4-tensor `mean_diff` vectors for 20 of 21 tasks (excluding Flowers102 as
the outlier), the per-block mean-|cos|-to-universal table is:

| block | mean \|cos\| | min (worst source task) | max |
|---|---|---|---|
| 0 | 0.43 | 0.001 | 0.88 |
| 1 | 0.49 | 0.15 | 0.76 |
| 2 | 0.40 | 0.07 | 0.71 |
| 3 | 0.36 | 0.06 | 0.66 |
| 4 | 0.50 | 0.06 | 0.85 |
| 5 | **0.86** | 0.40 | 0.99 |
| **6** | **0.96** | **0.75** | **1.00** |
| **7** | **0.96** | **0.76** | **1.00** |
| **8** | **0.96** | **0.77** | **1.00** |
| **9** | **0.95** | **0.76** | **1.00** |
| 10 | 0.94 | 0.72 | 1.00 |
| 11 | 0.89 | 0.67 | 0.98 |

The shared direction isn't a single-block phenomenon — it's a stable mid/late-
network attractor spanning blocks 5–11. Even the lowest source task aligns at
|cos| ≥ 0.75 on blocks 6–9 with the universal mean. This suggests the W4-tensor
quant-fragility direction is a **persistent residual-stream feature** that is
neither destroyed nor created by individual blocks but rather propagated.

Implication for the transfer experiment: sweep `block_sweep=[5,6,7,8,9]` rather
than `[6]` alone — different target tasks may find their best block somewhere
in that range.

### F4 [2026-05-28] — `contrastive_svd` consistently weaker than `mean_diff`

Across all 4 PTQ configs, the contrastive-SVD direction has lower cross-task
cosine than the mean-diff direction at the same block. Per-task `num_bad` is
small (typically 50–200), so the covariance of the bad cloud is rank-deficient
and its top eigenvector is noisy. Pooling activations across tasks (Approach C
in the plan) might rescue cSVD, but mean-diff is the more reliable per-task
estimator at these sample sizes.

---

## Actions taken (chronological)

### 002_quant_steering — per-task fitting + evaluation (complete)
- `fit_steering_vector.py` + yaml — FP pass, in-place PTQ, Q pass with per-block CLS hooks, fit & save `mean_diff` + `contrastive_svd` per block.
- `evaluate_steered_ptq.py` + yaml — sweep (method, block, alpha) on val+test, report FP / plain-PTQ / steered.
- `CROSS_TASK_PLAN.md` — research plan for cross-task extension.

### Diagnostics under 002_quant_steering (complete)
- `phase0_cross_task_cosine.py` — per-block signed + absolute cross-task cosine table + HTML line plots.
- `phase0_cluster_tasks.py` — hierarchical clustering of tasks by |cos|, dendrogram + reordered heatmap HTML.

### Fit sweep complete
- 21/22 tasks per PTQ config at W3-channel, W4-channel, W3-tensor, W4-tensor.
- ImageNet pending HF gating + token (token now set, fetch in progress).

### 003_quant_steering_transfer (in progress)
- `combine_steering_vectors.py` — argparse: sign-align then average (or top-SVD) across tasks → universal vector.
- `evaluate_steered_ptq_transfer.py` + yaml — Hydra: load FP → PTQ → load universal vector → sweep + eval on held-out task.
- `loo_runner.py` — argparse: emit leave-one-out shell commands for the gold-standard transfer test.

---

## Important context for future sessions

- **Backbone**: `vit_base_patch16_224.orig_in21k` (timm). Sanitized on disk as `vit_base_patch16_224_orig_in21k`. NOT the CLIP variant.
- **Finetune hyperparams** (encoded in every storage path): `lr=1e-05, wd=0.1, ls=0.0, wl=500, max_grad_norm=1.0, bs=128, seed=2038`.
- **22 datasets** from `DATASET_NAME_TO_EPOCHS` minus ImageNet (pending).
- **PTQ configs tested so far**: W{3,4} × {channel, tensor}, `skip_modules=[head]`.
- **CHECKPOINT_BASE_PATH**: `/home/lzhou/qat-transfer/storage/checkpoints`.
- **EVALUATION_BASE_PATH**: `/home/lzhou/qat-transfer/evaluations`.
- **HF_TOKEN**: set in `.env` (gated datasets work).
- **Never** execute scripts under `code/src/` or `code/experiments/` (per CLAUDE.md). Diagnostic scripts under `code/visualizations/` are safe to run.

---

## Open questions

- Does the universal vector **actually improve test accuracy** when applied to a held-out task at W4-tensor block 6? (about to test in 003)
- Does **Flowers102** fail dramatically as a control? (orthogonal axis → no help expected, would confirm the steering acts via projection on the task's quant-fragility direction)
- Is there a similar universal direction at higher bit-widths (W2) or other granularities? Hypothesis: the regime where it appears is "harsh enough PTQ to break many inputs the same way".
- What architectural feature of block 6 (mid-network) makes it the universal-direction site? Outlier channels? Specific attention head? Worth a per-channel analysis if 003 confirms transfer works.
- Does the steering generalize across **backbones** (e.g., the CLIP-ViT-B variant) or only within `orig_in21k`?

---

## How to use this file

- **Read it first** every session to know what's been done and what's working.
- **Append a new F# finding** whenever a result lands that changes our picture
  of the problem. Date it.
- **Append a new action item** under "Actions taken" whenever a substantive
  piece of code lands.
- Keep "Open questions" current — move resolved ones into a "Resolved" subsection
  once they're answered, with a back-pointer to the finding that resolved them.
- Do not delete history; only append or reorganize. If a finding is later
  invalidated, add a "(revised by Fk)" annotation rather than removing.
