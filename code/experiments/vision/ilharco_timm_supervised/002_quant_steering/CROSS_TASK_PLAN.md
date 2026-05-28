# Cross-task quantization-steering plan

Companion to [fit_steering_vector.py](fit_steering_vector.py) and
[evaluate_steered_ptq.py](evaluate_steered_ptq.py). The 002 phase fits and
evaluates steering vectors **per-task**. This document plans the cross-task
extension: a single shared direction (per block) fitted on K source tasks that
generalizes to unseen target tasks under the same `(model, bits, granularity)`.

---

## Underlying question

Is quant-fragility a property of **(model, bits)** or of **(model, bits, task)**?

- If `(model, bits)` → there's a universal direction per block; transfer works.
- If `(model, bits, task)` → directions are task-specific; transfer fails.

Prior work on outlier-driven quantization noise (LLM.int8, SmoothQuant, …)
suggests fragility is mostly **architectural** — specific channels are
systematically hard to quantize regardless of downstream task. So the prior
favors "transfer works at least partially." We verify directly via Phase 0.

---

## Phase 0 — diagnostic before any cross-task fitting

Run 002's `fit_steering_vector.py` on **3–5 small datasets** at the same
`(model, bits, granularity)`. Then a pure-analysis script computes, per block:

1. **Cross-task cosine similarity** of `v_mean[i]` between every task pair.
2. **Norm ratio** `||v_mean[i]||` across tasks.
3. **Best-α curves** per task — do they peak at similar α on similar blocks?

Decision rule:
- Mean pairwise cosine **> 0.6** on mid-to-late blocks → strong shared signal → Approach A.
- Mean pairwise cosine **0.2 – 0.5** → partial overlap → Approach B (SVD picks robust shared component).
- Mean pairwise cosine **≈ 0** → cross-task idea is dead; pivot.

No GPU work in Phase 0 — just consumes the per-task `steering_vectors.pt` files
that 002 already produces.

---

## Phase 1 — three combiners, simplest-first

### Approach A — task-averaged vector (QV-transfer analogue)

```
v_shared[i] = mean_k ( v[k][i] / ||v[k][i]|| )
```

Unit-normalize per task before averaging so a single high-magnitude task
doesn't dominate. Cheapest baseline; closest analogue to the existing
[001_qat_transfer](../001_qat_transfer/) line of work.

### Approach B — top right-singular-vector of stacked source vectors

```
V[i] = stack_k ( v[k][i] / ||v[k][i]|| )      # (K, D)
v_shared[i] = top right singular vector of V[i]
```

Picks the rank-1 direction best supported by *most* source tasks; robust to
one or two outlier tasks. Worth using if Phase 0 shows high inter-task
variance.

### Approach C — pool activations across tasks, fit once

```
for each source task k:
    capture cls_good[k], cls_bad[k]  per block  (as today)
    per-task z-score: cls_good[k] ← (cls_good[k] - μ_k) / σ_k
pool good = concat_k cls_good[k];   pool bad = concat_k cls_bad[k]
fit mean-diff / cPCA on pooled
```

Strictly more flexible than A — both the mean and the covariance see K× more
samples. **The only approach that makes `contrastive_svd` viable cross-task**:
per-task ~100 bad samples in 768-D is rank-deficient; K=5 source tasks gives
~500 pooled bad samples → meaningful covariance.

Per-task z-score is non-negotiable; without it the highest-norm task dominates.

### Recommended order
**A first** (headline baseline) → **C** (pooled, strongest variant) → **B**
only if Phase 0 shows wildly varying directions.

---

## Phase 2 — experimental protocol

Three disjoint task pools:

| Pool | Use | Size |
|---|---|---|
| **Source** | Fit the shared vector(s) | K ≥ 6 |
| **Tune** | Pick `block*` and `α*` (cross-task analogue of `pick_best_alpha`) | 1 – 2 |
| **Test** | Report test acc with selected `(block*, α*)` | 2 – 3 |

This is exactly the structure of [001_qat_transfer](../001_qat_transfer/), one
level up: `(source dataset → target dataset)` becomes `(source pool → target
pool)`. Code organization follows that template.

---

## Phase 3 — implementation, if Phases 0–2 show signal

New phase **003_quant_steering_transfer/**, mirroring 001's three-script
shape. Adds **no modifications to 002** — 003 consumes 002's outputs.

```
code/experiments/vision/ilharco_timm_supervised/003_quant_steering_transfer/
  combine_steering_vectors.py        # argparse helper: read per-task .pt files,
                                     # combine via Approach A / B, save .pt. No GPU.
  pool_and_fit_steering_vector.py    # Hydra script: Approach C. Walks source-task
                                     # checkpoints, captures activations across all,
                                     # z-scores per task, fits pooled vector.
  evaluate_steered_ptq_transfer.py   # Hydra script: same shape as 002's eval but
                                     # loads cross-task vector via steering.vectors_path,
                                     # evaluates on a target task.

config/experiments/vision/ilharco_timm_supervised/003_quant_steering_transfer/
  combine_steering_vectors.yaml      # (only if not argparse)
  pool_and_fit_steering_vector.yaml
  evaluate_steered_ptq_transfer.yaml
```

---

## Sharp edges to handle at implementation time

1. **Sign-alignment across tasks.** `v_csvd` is sign-aligned per-task with that
   task's `v_mean`. Across tasks the signs can still disagree if `v_mean`
   itself flips. Before averaging or stacking, re-align all tasks' vectors to a
   reference (e.g. task-0's `v_mean`): flip task-k's vector iff
   `<v_k, v_ref> < 0`.

2. **Same-arch requirement.** Cross-task vectors only compose within a fixed
   `(arch, bits, granularity, skip_modules)`. Different D = different vector
   space; the combiner scripts must assert this on load.

3. **Class-count confound.** ImageNet (1000), DTD (47), CIFAR10 (10) all share
   the backbone residual stream but differ in head shape. Steering acts on the
   backbone, so this *should* not matter — but verify empirically that
   small-class-count tasks don't all contribute a near-identical direction
   that swamps harder tasks.

4. **What "best block" means cross-task.** If task A's best block is 6 and
   task B's best block is 9, the cross-task vector at block 6 is good for A
   but maybe bad for B. Two responses:
   - (a) sweep block on the target — accept block-index as a transferred hyperparameter (tuned on the **tune pool**, not target);
   - (b) apply at *all* blocks simultaneously with a single shared α (tied "residual-stream-delta" intervention). One hyperparameter, possibly weaker effect.

   Compare both.

5. **`contrastive_svd` cross-task likely needs pooling.** Per-task the bad
   group is ~100 samples in 768-D → rank-deficient covariance, noisy
   eigenvectors. Approach A and B on per-task `v_csvd` will mostly amplify
   that noise. Approach C (pooled) is the only place where cross-task cSVD
   becomes statistically meaningful.

---

## Practical next steps

### 1. Finetune more source tasks at the same `(model, lr, wd, …, seed)` as CIFAR10

Recommended initial pool (all under 6 epochs except EuroSAT at 12, all
roughly CIFAR-sized so ~5–15 min each on 2080 Ti):

| Dataset | Epochs | Domain |
|---|---|---|
| CIFAR10 (running) | 6 | natural objects, 10 classes |
| CIFAR100 | 6 | natural objects, 100 classes |
| SVHN | 4 | digits, 10 classes |
| Food101 | 4 | food, 101 classes (fine-grained) |
| EuroSAT | 12 | satellite, 10 classes (OOD domain) |
| GTSRB | 11 | traffic signs, 43 classes (OOD domain) |

Five additional source tasks → six total. Plenty for Phase 0 diagnostic and
Phase 1 Approach A. If signal is there, expand to 10+.

### 2. Run 002 fits across all six tasks at W4 + W8

After finetunes complete, sweep `fit_steering_vector.py` over the
`(dataset_name, ptq.bits)` grid.

### 3. Write the Phase 0 analysis script

Pure-numpy, argparse, no GPU. Loads each task's `steering_vectors.pt`,
computes pairwise cosines per block, prints a heatmap or table. Decision-
making lives there.

### 4. If diagnostic green-lights, build 003

Per the layout above.

---

## Tracking

- **Branch:** `quant-steering`
- **Status as of writing:**
  - 002 (per-task) — done, committed.
  - Phase 0 — not started (waiting on multi-task finetunes).
  - 003 — not implemented yet.
