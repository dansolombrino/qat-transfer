# Input-fragility research log

Project-level running journal for the `input-fragility` branch. Claude updates
this file as findings emerge. Future sessions should read it first to see
what's been done and what the current state of the hypothesis is.

**Branch:** `input-fragility` (off `master`)
**Last updated:** 2026-05-28

---

## Predecessor context (read this first)

This line of work is a **pivot from the abandoned `quant-steering` branch**.
See `QUANT_STEERING_LOG.md` on that branch for the full negative-result trail.
TL;DR: rank-1 residual-stream steering vectors don't usefully recover ViT W4
PTQ accuracy (ceiling ~26% gap on the best task, ~12% mean). During that work
we observed that the same good/bad split (FP-correct ∩ Q-correct vs FP-correct
∩ Q-wrong) exposes **input-level** structure we never directly analyzed —
this branch does that analysis.

The two branches share the same data on disk (the same finetuned checkpoints,
the same `apply_ptq_` pipeline). They diverge in **what** they study about the
good/bad split: `quant-steering` studied activations; this branch studies
inputs and input-derived properties.

---

## Hypothesis under test

**Some inputs are predictably more fragile under PTQ than others, and this
fragility is predictable a-priori from properties of the FP model's
representation of the input — not from raw pixel statistics.**

If true, the practical hook is **input-aware mixed-precision routing**:
forward the input through the FP model once, predict P(bad), route fragile
inputs to FP while serving the rest from PTQ.

Per-task properties measured for each FP-correct val + test sample:

- **FP-derived (model-aware)**: `fp_margin` (top1 − top2 logit),
  `fp_softmax_top1`, `fp_entropy`, `fp_cls_dist_to_class_centroid` (pre-head
  representation distance from this sample's class mean, computed on val,
  reused on test).
- **Image-only (raw pixels)**: `img_brightness`, `img_contrast`,
  `img_edge_density` (Sobel mean), `img_high_freq_ratio` (FFT outer-band
  energy fraction).

---

## Key findings

### F1 [2026-05-28] — fragility is predictable; FP-confidence + class-typicality are the dominant signals

Ran the dump (`004_input_fragility/dump_pred_and_input_props.py`) on the
6 high-drop W4-channel tasks (Cars, EMNIST, KMNIST, SUN397, TinyImageNet,
Food101). Ran the analyzer (`analyze_input_props.py`) on the resulting
parquets.

**Multivariate logistic regression** on all 8 properties, fit on val,
evaluated on held-out test:

| task | CV val AUC | test AUC |
|---|---|---|
| Cars | 0.832 ± 0.047 | 0.784 |
| EMNIST | 0.897 ± 0.019 | 0.904 |
| Food101 | 0.923 ± 0.012 | 0.920 |
| KMNIST | 0.916 ± 0.013 | 0.895 |
| SUN397 | 0.897 ± 0.020 | 0.885 |
| TinyImageNet | 0.920 ± 0.010 | 0.908 |

**Mean test AUC = 0.882 (std 0.045)**. Every task well above chance. Cars
lowest (small val → noisy class centroids).

**Univariate decomposition (mean test AUC across 6 tasks, direction-aware):**

| Property | Discriminative AUC | Direction |
|---|---|---|
| `fp_margin` | ~0.865 | lower → bad |
| `fp_softmax_top1` | ~0.815 | lower → bad |
| `fp_cls_dist_to_class_centroid` | 0.748 | higher → bad |
| `fp_entropy` | 0.644 | higher → bad (variable) |
| `img_brightness` | 0.489 | none |
| `img_contrast` | 0.488 | none |
| `img_edge_density` | 0.492 | none |
| `img_high_freq_ratio` | 0.500 | none |

**Two strong predictors emerge:**
- **FP confidence / margin**: matches the strongest mechanism prior — PTQ
  adds logit noise, so low-margin inputs are closest to the decision
  boundary and flip first.
- **Distance to class centroid in pre-head representation**: outliers within
  a class are more fragile. Effect is strong on tasks with enough samples
  per class to estimate centroids reliably; weak on Cars (~5 val samples
  per class for 196 classes).

**Image pixel statistics carry zero predictive power.** Brightness,
contrast, edges, FFT bands all hover at AUC ≈ 0.50. PTQ-fragility is not a
property of raw pixels — it's a property of how the **model** represents
and is uncertain about the input. This is itself a clean negative result
that sharpens the positive one.

**Multivariate beats best univariate** (+0.02 AUC). FP-confidence and
class-typicality carry partially complementary information.

---

## Actions taken

### 004_input_fragility — fragility-predictor analysis
- `dump_pred_and_input_props.py` + yaml (`code/experiments/.../004_input_fragility/`):
  Hydra. Per-task, forward FP, apply PTQ, forward Q, compute per-sample
  scalars (FP margin/entropy/centroid distance, image stats). Saves one
  parquet per split + JSON metadata.
- `analyze_input_props.py` (`code/visualizations/.../004_input_fragility/`):
  argparse. Univariate AUC per property + multivariate logistic regression
  with 5-fold CV val + held-out test. Markdown + HTML output.

### Data on disk so far
- Dumps at W4-channel for 6 high-drop tasks (Cars, EMNIST, KMNIST, SUN397,
  TinyImageNet, Food101).
- Full 21-task dump at W4-channel: **in progress** (running as of writing).

---

## Open questions / planned next steps

In priority order:

1. **Robustness across 21 tasks**: validate F1's AUC range holds on all
   tasks, not just the 6 high-drop pilot. In progress.
2. **Cross-task predictor generalization**: fit logistic regression on tasks
   A..T, evaluate on task U held out. If transfer works, one trained
   predictor serves all tasks — bigger practical payoff.
3. **End-to-end Pareto curve**: at threshold τ on predicted P(bad), route
   top X% of test inputs to FP and serve the rest from PTQ. Plot test
   accuracy vs FP-compute-fraction. This is the headline figure for the
   paper.
4. **Bit-width sensitivity**: does the same predictor work at W3-channel,
   W8-tensor, etc.? Or is the relationship between FP confidence and
   PTQ fragility bit-width specific? A quick A+B run at one more PTQ config
   would settle this.

---

## Important context for future sessions

- **Backbone**: `vit_base_patch16_224.orig_in21k` (timm). Sanitized on disk
  as `vit_base_patch16_224_orig_in21k`.
- **Finetune hyperparams** (encoded in every path):
  `lr=1e-05, wd=0.1, ls=0.0, wl=500, max_grad_norm=1.0, bs=128, seed=2038`.
- **22 datasets** from `DATASET_NAME_TO_EPOCHS` minus ImageNet (gated;
  HF_TOKEN now set so could be added).
- **CHECKPOINT_BASE_PATH**: `/home/lzhou/qat-transfer/storage/checkpoints`.
- **EVALUATION_BASE_PATH**: `/home/lzhou/qat-transfer/evaluations`.
- Dumps live at:
  `${CHECKPOINT_BASE_PATH}/vision/ilharco_timm_supervised/input_fragility_dumps/...`
- HTML + markdown reports land in `plots/004_input_fragility/`.
- **Never** execute scripts under `code/src/` or `code/experiments/` (per
  CLAUDE.md). Diagnostic scripts under `code/visualizations/` are safe.

---

## How to use this file

- **Read it first** every session to know what's been done and what's working.
- **Append a new F# finding** whenever a result lands that changes our picture
  of the problem. Date it.
- **Append a new action item** under "Actions taken" whenever a substantive
  piece of code lands.
- Keep "Open questions" current — move resolved ones into a "Resolved"
  subsection once they're answered, with a back-pointer to the finding that
  resolved them.
- Do not delete history; only append or reorganize. If a finding is later
  invalidated, add a "(revised by Fk)" annotation rather than removing.
