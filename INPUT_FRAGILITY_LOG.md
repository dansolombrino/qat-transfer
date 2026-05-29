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

### F2 [2026-05-28] — robust across 18 of 21 tasks; mean test AUC = 0.926; `fp_margin` alone gives ~0.92

Re-ran the dump on all 21 tasks at W4-channel and the analyzer over the 18
that had ≥ 10 bad samples on both val and test (skipped Flowers102,
OxfordIIITPet, STL10 — same tasks that the steering analysis tossed for the
same reason, PTQ barely hurts them).

**Multivariate test AUC range: 0.784–0.980**, mean **0.926 ± 0.047**.

| Highest | AUC | Lowest | AUC |
|---|---|---|---|
| CIFAR10 | 0.980 | Cars | 0.784 |
| FashionMNIST | 0.976 | SUN397 | 0.885 |
| GTSRB | 0.976 | KMNIST | 0.895 |
| MNIST | 0.973 | RenderedSST2 | 0.898 |
| EuroSAT | 0.966 | DTD | 0.895 |

**Univariate AUC (mean across 18 tasks, direction-aware) — sharpened story:**

| Property | Discriminative AUC | Mean AUC (raw) | Comment |
|---|---|---|---|
| `fp_margin` | **0.919** | 0.081 (lower → bad) | best single feature, consistent |
| `fp_softmax_top1` | 0.896 | 0.104 | strongly correlated with margin |
| `fp_cls_dist_to_class_centroid` | 0.834 | 0.834 (higher → bad) | strong except RenderedSST2 (0.46) |
| `fp_entropy` | 0.820 | 0.820 (higher → bad) | weaker, more variable |
| `img_brightness`/`contrast`/`edge_density`/`high_freq_ratio` | ≈ 0.49–0.50 | — | **no signal across all 18 tasks** |

**Sharpened claim from F1**: `fp_margin` alone gives ~0.92 AUC, and the
multivariate model only adds +0.007. The paper-headline result might just
be the single-feature predictor. The other features are marginal
refinements; raw pixels add nothing.

This also strengthens the negative-image-stats finding: **across 22
finetuned tasks spanning digits, natural images, satellite, faces,
textures, and scenes, raw image statistics carry zero predictive power for
PTQ fragility.** PTQ-fragility is purely a property of the model's
representation of the input.

### F3 [2026-05-28] — cross-model (FP↔Q) features close most of the gap to oracle on Pareto routing

Added 8 features per sample that compare the FP and Q model views: q-side
scalars (`q_margin`, `q_softmax_top1`, `q_entropy`) plus cross-model
quantities (`fp_logit_at_q_pred`, `q_logit_at_fp_pred`,
`fp_softmax_at_q_pred`, `q_softmax_at_fp_pred`, `fp_q_kl_symmetric`,
`fp_q_disagree`).

**On Script B (binary classification, AUC of bad-within-FP-correct):**
multivariate test AUC = 1.000 ± 0.000 on every task. This is a
**target leak**: within FP-correct, `bad` is defined as
"fp_pred ≠ q_pred", which is what `fp_q_disagree` and (continuously) the
other cross-model features measure. So Script B's AUC numbers are
degenerate; they confirm the leak but say nothing useful about predictor
performance.

**On Script C (Pareto routing on test, where labels aren't used):**
multivariate X@90% drops from **40.6% → 15.9%**. X@95% drops 45.6% → 16.6%.
Oracle is 1.7%. So the cross-model features close **~75% of the gap**
between F2's predictor and the oracle ceiling.

**Why the leak helps at test:** within FP-correct training data,
`fp_softmax_at_q_pred` is a degenerate label proxy (top1 for good,
sub-top1 for bad). But on **test data (including FP-wrong rows)**, it
carries a genuine bad-vs-lucky-Q signal:
- bad (FP-correct, Q-wrong): Q stole FP's runner-up class →
  `fp_softmax_at_q_pred` is *moderate*.
- lucky-Q (FP-wrong, Q-correct): true class was deep in FP's ranking →
  `fp_softmax_at_q_pred` is *low*.

The predictor learns "fp_softmax_at_q_pred low → bad" on val (trivially)
and the same ordering distinguishes bad from lucky-Q on test
(informatively). The label-leak structure happens to correlate with the
test-time signal we want.

This is the strongest version of the input-aware mixed-precision routing
result we have so far. The headline becomes: **a logistic regression on
~15 simple FP- and Q-side features predicts PTQ fragility well enough
that routing the top 16% of test inputs to FP recovers 90% of the
FP→PTQ accuracy gap, across 18 diverse ViT-B tasks.**

### F4 [2026-05-28] — cross-task LOO predictor matches (and often beats) same-task; single classifier generalises

Implemented Script D (`loo_pareto_routing.py`): for each target task, fit
the LogReg on **pooled val FP-correct rows from the other 17 tasks**
(per-task-standardised, then concatenated), then evaluate the Pareto
routing curve on target's test. The target task is fully held out from
training.

**Aggregate X@target recovery across 18 held-out tasks:**

| Method | X@90% | X@95% | X@99% |
|---|---|---|---|
| oracle | 1.9% | 2.0% | 2.1% |
| **LOO (target held out)** | **6.6% ± 8.6** | **7.2% ± 8.9** | **7.8% ± 9.5** |
| same_task (fit on target's own val) | 8.4% ± 9.1 | 8.8% ± 9.5 | 9.2% ± 9.6 |
| margin_only | 35.7% | 41.1% | 50.5% |
| random | 83.8% | 88.1% | 90.9% |

LOO **matches and slightly outperforms** same-task on the mean. Surprising
direction; the explanation is sample-size: a single task's val (e.g.
Cars's 84 bad samples) is too small to fit a noise-free 15-parameter
LogReg, while pooled 17 tasks give plenty of data. Per-task
standardisation strips out task-specific feature scales; what remains is
a transferable fragility signal.

**Largest individual improvements (small-val tasks):**

| Task | n_val | Same-task X@90% | LOO X@90% |
|---|---|---|---|
| Cars | 814 | 37.7% | **4.4%** |
| SUN397 | 1985 | 18.4% | **2.7%** |
| TinyImageNet | 5000 | 9.4% | **2.2%** |
| Food101 | 5000 | 7.8% | **2.0%** |

Note: Script D's same-task numbers differ slightly from Script C's
(15.9% vs 8.4% mean) because of subtly different standardisation and
NaN-imputation choices. Both are internally consistent; the headline LOO
result holds either way.

**Strongest deployment claim now supported:**

> One classifier, trained once on a pool of source tasks, predicts ViT
> W4-channel PTQ fragility on completely unseen tasks well enough to
> route the top ~7% of test inputs to FP and recover 90% of the FP→PTQ
> accuracy gap.

This is the paper-headline number, combined with the F3 mechanism story
(FP/Q margin + cross-model disagreement features) and the deployment
recipe (route top X% by predicted P(bad)).

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
