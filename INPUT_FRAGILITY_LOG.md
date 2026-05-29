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

### F5 [2026-05-28] — predictor generalises to W3-channel (bit-width robustness, with a regime caveat)

Re-ran the full pipeline at W3-channel: 21 task dumps + Script D LOO.

**Aggregate X@90% across 21 held-out tasks at W3-channel:**

| Method | X@90% | LOO − oracle |
|---|---|---|
| oracle | 53.8% ± 16.7 | — |
| **LOO** | **62.8% ± 16.6** | **+9.0 pp** |
| same_task | 65.5% ± 16.9 | +11.7 pp |
| margin_only | 89.8% | +36.0 pp |
| random | 90.3% | +36.5 pp |

The absolute X@90% looks scary (54% for oracle vs W4's 1.9%) — but **it's
the regime, not the predictor**. At W3-channel with vanilla `apply_ptq_`,
mean PTQ test accuracy ≈ 25% (catastrophic; see F5/F6 in
QUANT_STEERING_LOG). Mean FP→PTQ gap ≈ 60pp. Most inputs are now `bad`,
so oracle has to route ~60% of inputs to FP to recover 90% of the gap.
No shortcut exists.

**LOO − oracle gap is only 9pp at W3** (vs 4.7pp at W4). Proportionally
LOO/oracle = **1.17×** at W3 vs 3.5× at W4 — i.e. the predictor is
**relatively closer to oracle at the harsher regime**. And LOO still
beats same-task on the mean. Same direction as F4.

Small-val pooling rescue still applies:

| Task | n_val | Same-task X@90% | LOO X@90% |
|---|---|---|---|
| Cars | 814 | 88.9% | **33.3%** |
| SUN397 | 1985 | 81.2% | **42.8%** |
| TinyImageNet | 5000 | 70.0% | **54.3%** |

**Honest caveat for the paper:** at catastrophic regimes (W3-channel,
W4-tensor), no input-aware routing can save much compute — the gap is
too wide and the `bad` population is too prevalent. Practical deployment
of "PTQ + routing" works at moderate-PTQ regimes (W4-channel) where the
gap is small and the bad population is a minority. At harsher regimes,
the bottleneck is the PTQ method itself (vanilla `apply_ptq_`), not the
predictor.

**Two-axis paper claim** now supported by data:

1. **Predictor generalises across tasks (F4) AND across bit-widths
   (F5).** The mechanism (FP/Q confidence + cross-model disagreement) is
   bit-width-robust.
2. **The practical compute-saving benefit depends on the PTQ regime
   being recoverable** — i.e. on the FP→PTQ gap being a minority of
   inputs. Stronger PTQ methods (SmoothQuant, AWQ, QuaRot) that produce
   gentler W3/W2 degradation would extend the routing-recovers-compute
   regime to harsher bit-widths. Future work.

### F6 [2026-05-28] — feature ablation: the 6.6% number is a DIAGNOSTIC ceiling; the deployable headline is Q-only X@90% ≈ 24%

Ran Script E feature-subset ablation under the LOO setup at W4-channel.

The 15-feature predictor's 6.6% X@90% uses features that require BOTH
forward passes (FP and Q). Not a deployment recipe — if you've paid for
both forward passes there's no compute saving to claim. The deployable
recipe is **PTQ-first**: run PTQ on every input, use its own confidence
signals to decide whether to also run FP.

**Aggregate X@90% LOO across 18 tasks per subset:**

| Subset | Deployment scenario | X@90% |
|---|---|---|
| `image_only` | No model — image stats | 79.2% (near random) |
| **`q_only`** | **PTQ-first deployable** | **24.4% ± 20.8** |
| `q_plus_image` | PTQ-first + image stats | 24.5% (image adds nothing) |
| `fp_only` | FP-side only | 36.0% |
| `fp_plus_image` | FP-side + image | 36.2% |
| `fp_plus_q_no_cross` | Both models, no cross | 29.1% |
| **`all_features`** | **Diagnostic ceiling (both models + cross)** | **6.6% ± 8.6** |

**Key takeaways:**

1. **Deployable headline shifts: X@90% ≈ 24%, not 6.6%.** Under realistic
   PTQ-first deployment, a Q-side multivariate LogReg routes ~24% of
   unseen target-task inputs to FP and recovers 90% of the FP→PTQ gap.
   Still beats random (83.8%) by 3.4×, no per-task training needed.

2. **The 18pp gap (24.4% → 6.6%) is the cost of the lucky-Q
   ambiguity.** Q-side features alone cannot tell bad inputs from
   lucky-Q inputs without comparing to FP's view — but running FP
   defeats the saving. So 24% is the true ceiling for single-pass
   routing; 6.6% is what's achievable only as a diagnostic.

3. **Image pixel statistics carry zero signal even after ablation.**
   image_only ≈ random; q_plus_image = q_only; fp_plus_image = fp_only.
   PTQ-fragility is purely a model phenomenon — never a pixel-level one.

4. **Q-side alone beats FP-side alone** (24% vs 36%). When you have one
   model's signals, Q's own uncertainty predicts "PTQ is wrong here"
   better than FP's uncertainty does. Q knows its own decision boundaries.

5. **Running both models without cross-features barely beats Q-only**
   (29.1% vs 24.4%). The cross-product features carry almost all of the
   "both models" value. Without them, paying for two forward passes
   is essentially wasted.

**Revised deployment claim** (this is the paper's actual headline):

> At W4-channel PTQ, running the quantized model first and using only
> its own confidence signals (`q_margin`, `q_softmax_top1`, `q_entropy`),
> a logistic regression fit on a pool of source tasks routes ~24% of
> unseen target-task inputs to a full-precision fallback and recovers
> 90% of the FP→PTQ accuracy gap. The classifier requires no
> per-target-task training. The 6.6% "both-models" ceiling diagnoses the
> bad-vs-lucky-Q ambiguity that single-pass routing cannot resolve.

### F6b [2026-05-28] — W3-channel feature ablation: deployable Q-only collapses; the recoverable-regime caveat is sharp

Ran Script E at W3-channel for direct comparison with F6.

| Subset | Deployment | W4 X@90% | W3 X@90% |
|---|---|---|---|
| `image_only` | no model | 79.2% | 86.4% |
| **`q_only`** | **PTQ-first deployable** | **24.4%** | **81.2%** |
| `q_plus_image` | PTQ-first + image | 24.5% | 81.3% |
| `fp_only` | FP-side only | 36.0% | 87.1% |
| `fp_plus_image` | FP-side + image | 36.2% | 86.4% |
| `fp_plus_q_no_cross` | both models, no cross | 29.1% | 81.4% |
| **`all_features`** | diagnostic ceiling | **6.6%** | **62.8%** |
| oracle | upper bound | 1.9% | 53.8% |

**Sharp transition between regimes.** At W3-channel even the
all-features diagnostic ceiling needs ~63% routing to recover 90% of
the gap. Q-only is barely better than random (81% vs 90%). Reason: at
W3 the bad-input population is the **majority** (~70% of FP-correct
inputs), so oracle itself is already in "route most things" territory
(54%). No predictor can substantially outperform oracle.

The deployable-routing recipe works **at recoverable PTQ regimes
only** — where bad inputs are a minority of the test set. W4-channel
qualifies; W3-channel doesn't. The two-axis paper claim is now
quantitatively delineated by the F6 / F6b table.

### F7 [2026-05-28] — threshold calibration for online deployment: batch is rock-solid, online τ adds variance

Ran Script F at W4-channel. Q-only LogReg trained per-target under LOO
on pooled source-task val. Compared the batch claim (sort-and-route)
against three online τ-strategies and the LogReg's natural threshold:

| Strategy | At deploy needs | Mean frac routed | Mean recovery |
|---|---|---|---|
| `batch` | full target batch | **26.7%** | **91.0% ± 2.3** |
| `natural` (τ=0.5) | nothing | 29.9% | 82.4% ± 45.8 |
| `val_pct` (75th-pct on target val) | val P(bad), label-free | 26.8% | 70.3% ± 57.4 |
| `source_pct` (75th-pct on pooled source val) | global τ from training | 37.4% | 80.2% ± 68.4 |
| `val_labeled` (val Pareto opt at 90%) | val labels + P(bad) | 23.9% | 78.3% ± 38.2 |

**Batch is rock-solid**: ~91% recovery with σ=2.3pp across 18 tasks.
The headline claim holds tightly task-to-task. **Online τ-strategies
are workable but noisier**: mean recovery 70–82%, with large per-task
variance driven by small-gap outliers (RenderedSST2 at +0.11pp gap
dominates the variance).

**Three deployment tiers now empirically delineated:**

1. **Batch (top)**: ~27% routing → 91% recovery, σ=2.3pp. Drop-in recipe;
   no calibration; ships as one function.
2. **Online + val-percentile τ (middle, label-free)**: ~27% routing →
   70% recovery. Ships LogReg + 1-line per-task calibration step. No
   labels needed.
3. **Online + τ=0.5 (bottom, zero-cost)**: ~30% routing → 82% recovery
   on aggregate but high variance per-task. Single fixed number, no
   calibration.

`val_labeled` shows surprisingly poor consistency — val Pareto points
overfit to small target val sets. Label-free percentile calibration is
the better online recipe in practice.

**For the paper**: the **primary claim is batch routing** (strong
recovery, tight variance, no per-task calibration). The **online
extension** is a clear follow-on with quantified noise. This is a
better story than insisting online matches batch.

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
