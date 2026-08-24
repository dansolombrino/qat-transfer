# PTQ Routing — predicting which inputs post-training quantization breaks

> Research code for **Input-Aware Mixed-Precision Routing for Quantized Transformers**.
> Weight-only PTQ degrades accuracy non-uniformly: most inputs survive, a small minority flips.
> This project predicts *which* inputs flip, cheaply enough to route only those to a
> full-precision fallback at deployment time.
>
> The paper is tracked in [`paper/`](paper/): `short_main.tex` (5-page workshop version) and
> `main.tex` (full version). Every number and figure in them is reproducible from this repo.
>
> If you are reading this without context: read this README front to back. It explains what
> we're studying, why, how the code is laid out, what we found, and how to reproduce it.

---

## TL;DR — what's the project and what did we find?

**Question.** Weight-only post-training quantization (PTQ) degrades accuracy non-uniformly across
inputs. Of the inputs the full-precision (FP) model classifies correctly, the quantized (Q) model
handles the vast majority fine; a small minority gets flipped — a few percent at W4-channel
(4.2% / 3.8% / 4.3% on our three backbones). That minority dominates the accuracy drop.
*Can we predict which inputs PTQ will flip, cheaply enough to be useful at deployment?*

**Setup.** Three backbones spanning two modalities: `vit_base_patch16_224.orig_in21k` and
`vit_large_patch16_224.orig_in21k` on 21 vision tasks each, and `Qwen3-Embedding-0.6B` on 11 MTEB
text tasks — 53 finetuned classifiers in total. Weight-only PTQ by naive round-to-nearest (RTN)
at W4-channel is applied to every linear layer except the head. For each (FP, Q) pair we dump
per-sample features and predictions on val + test and study the relationship.

**Headline result.** The routing signal is a **single scalar from the forward pass the deployment
already runs**: the quantized model's top-1/top-2 logit margin, `q_margin`. Sorting by it and
routing the lowest-margin **22.2% / 18.4% / 20.2%** of inputs (ViT-B/16 / ViT-L/16 / Qwen3)
recovers **90%** of the FP→PTQ accuracy gap. No training, no labels, no fitted weights — just a
threshold. Against a random baseline needing 83.8% / 89.8% / 80.9%, that is a 3.8–4.9× improvement.

**It beats the canonical baseline.** Max-softmax probability (MSP), the selective-prediction
default, needs 30.4% / 22.0% / 21.0% routing — worse by 8.2 / 3.6 / 0.8 pp. MSP measures *absolute
confidence*; the failure mode is *boundary proximity*, which is exactly what the margin measures.
Nothing else we tested beats a single `q_margin` threshold: not entropy, not pairwise
combinations, not a 3-feature Q-side logistic regression, not an FP-side one.

**Theory.** `q_margin` is not merely a good heuristic. A size-ε logit perturbation can flip a
prediction **iff** `q_margin < 2ε` (Prop. 1), and among all scores computed from the quantized
logits, only those preserving `q_margin`'s strict order match the worst-case-flippable set at
*every* perturbation size (Thm. 1). A matching conditional impossibility result (Prop. 2) delimits
when the gap to the two-pass ceiling is structurally irreducible. Proofs in `paper/`, App. C.

**Mechanism finding — the lucky-Q ambiguity.** The deployable ~20% budget versus the ~7% two-pass
diagnostic ceiling is bounded by what we call the **lucky-Q ambiguity**: single-pass features
cannot distinguish (a) inputs PTQ broke (**bad**) from (b) inputs where PTQ happens to be right and
FP is wrong (**lucky-Q**). Both straddle the decision boundary and look identical from the Q side.
Cross-model features resolve it — but they require running *both* models, defeating the saving.

**Negative result, worth keeping.** Raw input statistics carry **zero** signal: image brightness,
contrast, edge density and FFT high-frequency ratio all sit at AUC ≈ 0.53–0.55, and the Qwen3
text statistics reach AUROC 0.494. PTQ fragility is a property of the model's representation of
the input, not of surface input properties.

**Regime caveat.** This works at W4-channel, where the FP→PTQ gap is small and broken inputs are a
minority. At W3-channel under per-channel RTN, PTQ breaks the *majority* of inputs (~62% on
ViT-B/16, ~75% on ViT-L/16) and no input-aware routing recovers compute — even the oracle needs
53.8% / 66.5%. Refining to per-group_128 scales relocates that boundary below W3.

## Table of contents

1. [Background and definitions](#background-and-definitions)
2. [The research question, precisely](#the-research-question-precisely)
3. [Pipeline overview](#pipeline-overview)
4. [Scripts in detail](#scripts-in-detail)
5. [Key findings](#key-findings)
6. [Reproducibility — how to run this from scratch](#reproducibility--how-to-run-this-from-scratch)
7. [Code layout](#code-layout)
8. [Relation to the parent codebase](#relation-to-the-parent-codebase)
9. [Limitations and open questions](#limitations-and-open-questions)

---

## Background and definitions

### Weight-only PTQ for ViTs

Weight-only post-training quantization replaces every `nn.Linear` weight matrix's float32 values with a low-bit-width approximation, rounding-to-nearest, after the model has been trained. Activations stay float32. There's no retraining and no calibration data; just one in-place pass over the weights.

This codebase's PTQ is implemented in [`code/src/quantization.py`](code/src/quantization.py): the function `apply_ptq_(model, bits, granularity, skip_modules)` walks `model.named_children()` and replaces each Linear's weight with its fake-quantized version. After applying it, the model still has float32 weights, but those weights now lie on the quantization grid — running a forward pass produces the same outputs as a true int-N inference engine would.

At inference time on dedicated low-bit-width hardware this would translate to memory and compute savings; on vanilla GPUs without int4 kernels, it's compute-equivalent to FP but quantization-faithful in the outputs.

Two important hyperparameters:
- **`bits`** ∈ {3, 4, 5, …}: target bit-width. We focus on W4 (4 bits) in this work.
- **`granularity`** ∈ {`channel`, `tensor`}:
  - `channel`: one scale + zero-point per output channel of the weight matrix. More flexible; preserves accuracy better.
  - `tensor`: a single scale + zero-point for the whole weight matrix. More aggressive; catastrophic on ViTs without smoothing tricks.

We work primarily at **W4-channel** (the recoverable regime) and use **W3-channel** as a stress test.

### The four-quadrant taxonomy of inputs

For each test input `x` with label `y`, the (FP, Q) pair of models gives us two predictions:

| | Q correct | Q wrong |
|---|---|---|
| **FP correct** | `good`  (PTQ preserved) | `bad`  (PTQ broke this) |
| **FP wrong**   | `lucky_Q`  (Q got it right by coincidence) | both wrong |

The accuracy gap between FP and Q comes from the `bad` quadrant. Routing inputs from `bad` to FP would recover the accuracy at the cost of running FP for those inputs. The challenge is identifying `bad` inputs at inference time without seeing the label.

A subtle complication: `lucky_Q` inputs look almost identical to `bad` inputs from any single-pass perspective. Both have FP and Q disagreeing; both typically have low confidence in both models (they're near a decision boundary). Routing a `lucky_Q` input to FP *hurts* — you go from a correct Q answer to a wrong FP answer. So the prediction task isn't just "is FP and Q going to disagree on this input" — it's the harder "if they disagree, is FP the one that's right." We call this the **lucky-Q ambiguity** and it sets a fundamental ceiling on what single-pass routing can achieve.

---

## The research question, precisely

For a chosen ViT backbone and a chosen PTQ scheme, we ask three nested questions:

1. **Per-task univariate**: do simple features computed from the FP and Q model outputs (margins, entropies, softmax-top1, FP→Q logit-gather features) separate `bad` from `good` within FP-correct test rows?
2. **Per-task multivariate Pareto**: how good is a logistic regression on those features at the actual deployment task — "sort test inputs by P(bad) and route the top X% to FP"? How small can X be while still recovering 90% of the FP→PTQ accuracy gap?
3. **Cross-task transferable Pareto**: if we pool source-task data to train one classifier and apply it to a held-out target task, does the Pareto curve degrade? (LOO transfer.) Does it depend on which features we use (Q-only deployable vs. cross-model academic ceiling)?

Each question is answered by a corresponding script (or pair of scripts). The pipeline below glues them together.

---

## Pipeline overview

```
┌─────────────────────┐                                ┌───────────────────────┐
│  21 FP checkpoints  │                                │  predictions_val.parquet  │
│  (one per dataset)  │   Script A: dump features     │  predictions_test.parquet │
│  see finetune_fp.py │   + apply PTQ in place   ───► │  + dump_metadata.json     │
└─────────────────────┘                                │  (one set per task)       │
                                                       └───────────────┬───────────┘
                                                                       │
              ┌────────────────────────────────────┬───────────────────┼────────────────────────┐
              │                                    │                   │                        │
              ▼                                    ▼                   ▼                        ▼
   ┌──────────────────────┐  ┌──────────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
   │  Script B: univariate │  │ Script C: Pareto routing  │  │ Script D: LOO Pareto │  │ Script E: feature   │
   │  AUC + multivariate   │  │ for single task (sort &   │  │ across tasks         │  │ ablation (image_only │
   │  LogReg AUC per task  │  │ route top X%)             │  │                      │  │  / q_only / fp_only  │
   │                       │  │                           │  │                      │  │  / all_features)     │
   └──────────────────────┘  └──────────────────────────┘  └──────────────────────┘  └──────────────────────┘
                                                                       │
                                                                       ▼
                                                       ┌──────────────────────────┐
                                                       │ Script F: threshold       │
                                                       │ calibration for online    │
                                                       │ (fixed τ on P(bad))       │
                                                       └──────────────────────────┘
                                                                       │
                                                                       ▼
                                                       ┌──────────────────────────┐
                                                       │ generate_paper_figures.py │
                                                       │ → paper/figs/*.pdf        │
                                                       └──────────────────────────┘
```

**Inputs to the whole pipeline:** the 21 finetuned ViT checkpoints already on disk under `${CHECKPOINT_BASE_PATH}/vision/ilharco_timm_supervised/fp/...` (provided externally or produced by `code/src/vision/ilharco_timm_supervised/finetune_fp.py` in this repo). The Qwen3 pipeline is the mirror image under `text/ilharco_automodelforsequenceclassification/`.

**Outputs:** the parquet dumps under `${CHECKPOINT_BASE_PATH}/vision/.../input_fragility_dumps/...` are the canonical record. Everything downstream (AUC tables, Pareto curves, threshold sweeps, paper figures) is derived from those parquets and can be re-run cheaply on CPU.

---

## Scripts in detail

All six scripts live under `code/{experiments,visualizations}/vision/ilharco_timm_supervised/004_input_fragility/`. Script A is the only GPU-heavy one (one forward pass through FP and one through PTQ per task). Scripts B through F and the figure generator are pure-CPU analyses on the saved parquets and take seconds to minutes total.

### Script A — `dump_pred_and_input_props.py` (Hydra, GPU)

[`code/experiments/vision/ilharco_timm_supervised/004_input_fragility/dump_pred_and_input_props.py`](code/experiments/vision/ilharco_timm_supervised/004_input_fragility/dump_pred_and_input_props.py)

This is the only script that touches the GPU. It does the following for each (task, PTQ config) combination:

1. Loads the finetuned FP `ImageClassifier`.
2. Runs the FP model over val and test, recording per-sample:
   - `fp_logits` (held in memory)
   - prediction, top-1/top-2 logit, margin, softmax-top1, entropy
   - pre-head pooled representation (for class-centroid distance)
   - image-pixel statistics: brightness, contrast, Sobel edge density, FFT high-frequency ratio
3. Applies weight-only PTQ in place via `apply_ptq_(model, bits, granularity, skip_modules)`.
4. Runs the (now quantized) model over val and test, recording per-sample:
   - `q_logits`
   - prediction, top-1/top-2 logit, margin, softmax-top1, entropy
5. Computes cross-model scalars from the held FP and Q logits:
   - `fp_logit_at_q_pred`, `q_logit_at_fp_pred` (the other model's logit at this model's predicted class)
   - softmax counterparts
   - symmetric KL between FP and Q softmax distributions
   - `fp_q_disagree` (0/1)
6. Computes class centroid distances on val using the pre-head representations; reuses val centroids for test.
7. Saves two parquets (val and test) of ~30 columns each plus a JSON metadata sidecar.

Storage path:
```
${CHECKPOINT_BASE_PATH}/vision/ilharco_timm_supervised/input_fragility_dumps/
  {sanitized_model}/{dataset}/
  optim=adamw_lr=...wd=...ls=...wl=...mgn=...bs=.../
  ptq=bits={bits}_gran={granularity}_skip={skip_tag}/
  seed={seed}/
    predictions_val.parquet
    predictions_test.parquet
    dump_metadata.json
```

The total wall-clock cost per task is a couple of minutes (two forward passes through val + test); the full 21-task sweep is ~15 min on a single 2080 Ti.

### Script B — `analyze_input_props.py` (argparse, CPU)

[`code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/analyze_input_props.py`](code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/analyze_input_props.py)

For each per-task dump, restricts to FP-correct rows and asks: do the features distinguish `bad` from `good`? Per task, per feature:

- Mean(good), mean(bad), Cohen's d
- AUC on val and test
- Direction (does higher → bad or lower → bad?)

Also fits a multivariate logistic regression (StandardScaler + LogReg, class_weight=balanced, 5-fold CV on val + held-out test). Aggregates AUC across tasks. Emits a markdown table and an HTML report (per-task heatmap + per-property ROC curves).

> **Caveat in Script B's output:** the multivariate AUC under the full feature set is degenerate (1.000) because `fp_q_disagree` and the cross-model features are *exactly* `bad` within FP-correct training data (recall: bad = FP-correct ∩ Q-wrong, which is precisely when fp_pred ≠ q_pred). This is a definitional label leak, *not* a real performance number. The Pareto routing analysis in Scripts C/D/E correctly uses test-set routing instead of this AUC and is unaffected.

### Script C — `pareto_routing.py` (argparse, CPU)

[`code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/pareto_routing.py`](code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/pareto_routing.py)

For each task, fits a same-task multivariate LogReg on val FP-correct rows (positive class = bad), scores test rows, and computes the **batch Pareto curve**: sort test inputs by P(bad), sweep the routing fraction X from 0% to 100%, plot routed test accuracy. Reports `X@90%`, `X@95%`, `X@99%` (smallest X for which routing recovers 90% / 95% / 99% of the FP→PTQ accuracy gap).

Comparison baselines on the same plot:
- **oracle** (`fp_correct − q_correct`): the upper bound at every X.
- **margin_only** (`−fp_margin`): the 1-feature simplification.
- **random** (uniform shuffle): linear-interpolation baseline.

Outputs per-task and aggregate markdown summaries plus an HTML with line plots.

### Script D — `loo_pareto_routing.py` (argparse, CPU)

[`code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/loo_pareto_routing.py`](code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/loo_pareto_routing.py)

The same Pareto analysis, but for each target task the LogReg is trained on the **pooled val FP-correct rows of every other task** (per-task z-scored before pooling). The target task is fully held out; only its unsupervised feature statistics (mean, std) are touched at deployment, for standardisation.

This is the cross-task generalisation experiment. Side-by-side with the same-task baseline from Script C, it tells us whether one classifier transfers across tasks.

### Script E — `feature_ablation_pareto.py` (argparse, CPU)

[`code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/feature_ablation_pareto.py`](code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/feature_ablation_pareto.py)

Runs LOO Pareto for each of seven feature subsets, mapped to deployment scenarios:

| Subset | Deployment scenario |
|---|---|
| `image_only` | no model needed; pure pixel router |
| `q_only` | **PTQ-first, deployable** (only Q model runs in the routing decision) |
| `q_plus_image` | PTQ-first + image |
| `fp_only` | FP-side only (academic, no compute saving) |
| `fp_plus_image` | FP-side + image |
| `fp_plus_q_no_cross` | both models, no cross-product features |
| `all_features` | diagnostic ceiling (both models + cross features) |

The deployable headline is `q_only`. The `all_features` row is an academic ceiling — useful for quantifying the lucky-Q ambiguity gap.

### Script F — `threshold_calibration.py` (argparse, CPU)

[`code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/threshold_calibration.py`](code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/threshold_calibration.py)

Extends the deployable claim to **online (single-sample) deployment** where you can't sort a batch. For each target task it tests four fixed-threshold τ strategies on `P(bad)`:

- `natural` (τ = 0.5): the LogReg's own decision boundary.
- `val_pct`: τ = 75th percentile of P(bad) on target's val. Label-free per-task calibration.
- `source_pct`: τ = 75th percentile of P(bad) on pooled source val. Single global τ.
- `val_labeled`: τ on target's val chosen to reach 90% gap-recovery at minimum routing fraction. Uses target val labels.

Reports per-strategy mean (fraction routed, gap recovery). The batch claim from Script D is the reference. The aggregate finding is that batch is rock-solid (~26% routing → 91% recovery, σ=2.3pp across tasks) and online strategies are workable but noisier.

### `generate_paper_figures.py` (argparse, CPU)

[`code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/generate_paper_figures.py`](code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/generate_paper_figures.py)

Emits four publication-quality matplotlib PDFs into `paper/figs/`:

- `fig_headline_pareto_w4.pdf` — aggregate Pareto: Q-only deployable vs all_features ceiling vs margin_only vs random vs oracle.
- `fig_feature_ablation_w4.pdf` — Pareto curves per feature subset on one panel.
- `fig_regime_comparison.pdf` — W4 vs W3 side by side.
- `fig_loo_vs_same_task.pdf` — per-task scatter of LOO X@90% vs same-task X@90% (Q-only deployable).

Reproduces from the parquets in seconds.

---

## Key findings

All numbers are W4-channel unless stated, averaged over the eligible tasks per backbone
(18 / 16 / 9 for ViT-B/16 / ViT-L/16 / Qwen3; a task is eligible only if PTQ produces ≥10 broken
samples on both val and test, otherwise the per-task metric is too noisy to estimate).
Every figure below is re-derivable with `paper/scripts/verify_numbers.sh`.

### F1 — fragility is predictable from Q-side confidence; input statistics carry none

Univariate discriminative AUC (`max(AUC, 1−AUC)`) for **bad** vs **good** on FP-correct
validation rows, ViT-B/16, mean over 18 tasks:

| Feature | AUC | | Feature | AUC |
|---|---|---|---|---|
| `q_margin` | **0.938** | | `img_brightness` | 0.537 |
| `fp_margin` | 0.928 | | `img_contrast` | 0.544 |
| `fp_softmax_top1` | 0.901 | | `img_edge_density` | 0.550 |
| `q_softmax_top1` | 0.900 | | `img_high_freq_ratio` | 0.533 |
| `fp_entropy` | 0.839 | | *(Qwen3 text stats)* | 0.494 |
| `q_entropy` | 0.825 | | | |

Model-derived margin/softmax/entropy features are strongly discriminative; pixel- and
tokenizer-level statistics sit at chance. PTQ fragility is a representation phenomenon.

### F2 — one feature is the whole recipe

X@90 (smallest FP-compute fraction recovering 90% of the gap; **lower is better**):

| Subset | Deployment | ViT-B/16 | ViT-L/16 | Qwen3 |
|---|---|---|---|---|
| `msp_only` | MSP baseline | 30.4 ± 24.3 | 22.0 ± 16.5 | 21.0 ± 16.5 |
| `q_margin_only` | **margin (proposed)** | **22.2 ± 17.2** | **18.4 ± 12.0** | **20.2 ± 16.5** |
| `q_only` | all 3 Q-side (LogReg) | 24.4 ± 20.8 | 18.2 ± 11.9 | 20.8 ± 16.2 |
| `fp_only` | FP-side only | 34.9 ± 23.1 | 26.9 ± 16.6 | 41.6 ± 21.2 |
| `fp_plus_q_no_cross` | both models | 25.8 ± 20.9 | 18.9 ± 13.4 | 25.1 ± 18.6 |
| `all_features` | ceiling (both + cross) | 7.5 ± 8.8 | 6.0 ± 5.1 | 6.0 ± 3.6 |
| oracle | upper bound | 1.9 ± 1.4 | 2.3 ± 2.2 | 1.8 ± 1.1 |
| random | lower bound | 83.8 ± 21.5 | 89.8 ± 2.8 | 80.9 ± 21.1 |

Adding MSP and entropy to `q_margin` changes nothing beyond task-to-task variability; dropping
`q_margin` is uniformly worse. The Q-side ceiling is reached by one feature with no fitted model.

### F3 — Q-side beats FP-side, even one feature against three

Single-feature `q_margin` (22.2%) beats the *multivariate* FP-side LogReg `fp_only` (34.9%) on
ViT-B/16, and by ~1.5–2× on all three backbones. PTQ has its own decision boundaries, slightly
shifted from FP's; the inputs near *those* boundaries are the ones PTQ flips.

### F4 — cross-model features close the gap to oracle but are not deployable

`all_features` reaches 7.5% / 6.0% / 6.0%, close to oracle's 1.9% / 2.3% / 1.8% — but it runs both
FP and Q on every input, which defeats the purpose. Reported as a **diagnostic ceiling**, not a
recipe. The gap between it and `q_margin` is the lucky-Q ambiguity (see TL;DR, and Prop. 2).

### F5 — cross-task LOO generalises

The single-feature recipe needs no training, so this matters only for the multivariate ablations.
For those, leave-one-out cross-task fitting lands within ~2.5 pp of a same-task fit in aggregate
(`q_only`: 24.4% LOO vs 22.0% same-task on ViT-B/16). One fitted predictor ships across tasks.

### F6 — sharp regime boundary, and it is recipe-dependent

At W3-channel under per-channel RTN, the mean FP→PTQ gap climbs to ~60 pp (ViT-B/16) and ~74 pp
(ViT-L/16), broken inputs become the strict majority (~62% / ~75%), and even the oracle needs
53.8% / 66.5%. The cheap-default-with-fallback pattern stops applying — a property of the regime,
not the predictor. Switching to per-group_128 scales (same naive rounding, finer granularity, no
calibration cost) cuts the W3 gap and the oracle's X@90 by 54–78% and the recipe applies again.

### F7 — batch routing is rock-solid; label-free online routing is deployable but noisier

| Strategy | Routed | Recovery |
|---|---|---|
| `batch` (offline reference) | 22.2 / 18.4 / 20.2% | 91.0 / 90.7 / 90.8% (σ 2.3 / 1.2 / 0.8) |
| `val_pct` (**label-free headline**) | 26.7 / 26.4 / 25.2% | 72.7 / 91.7 / 90.6% (σ 56.4 / 11.3 / 14.5) |
| `val_x90` (label-aware variant) | 21.7 / 16.1 / 15.9% | 77.6 / 80.7 / 89.1% (σ 37.9 / 20.8 / 14.1) |

`val_pct` sets τ at the 25th percentile of `q_margin` on the target's *unlabeled* validation split
— no labels anywhere. The large ViT-B/16 σ is a metric artifact of near-zero-gap tasks admitted by
its eligibility cut (a single misrouted input swings recovery by tens of pp when the gap is
<0.5 pp), not a ranking failure; those tasks need no routing anyway and a label-free
disagreement-rate pre-filter skips them.

### Compute savings

At a typical int4 speedup (S = 2× per input), routing at the batch operating point saves
**~28–32%** of compute versus always-FP on all three backbones, and past 50% at S = 4×.

## Reproducibility — how to run this from scratch

### Prerequisites

- Python 3.11 (`.python-version` pins this).
- `uv` package manager. `uv sync` installs everything from `pyproject.toml` and creates `.venv/`.
- A GPU for Script A only. Everything else is CPU.
- ~10 GB of free disk under `${CHECKPOINT_BASE_PATH}` for the per-task dumps + finetuned-FP checkpoints (the checkpoints themselves are ~350 MB each × 21 tasks ≈ 7 GB).

### Environment

Copy `.env.example` to `.env` and fill in:

```
CHECKPOINT_BASE_PATH=/path/to/storage      # checkpoints + dumps live under here
EVALUATION_BASE_PATH=/path/to/evaluations  # not used by 004_input_fragility but required by shared infra
HEAD_BASE_PATH=...                          # same
TORCH_NUM_WORKERS=4
HF_HOME=...
HF_HUB_CACHE=...
HF_DATASETS_CACHE=...
# HF_TOKEN, HUGGING_FACE_HUB_TOKEN: leave commented unless you need gated HF datasets.
# An empty value here will cause "Authorization: Bearer " to be sent, which the server rejects.
```

`dotenv` loads this file at the start of every Hydra script. Always launch from the repo root, never from inside `code/`.

### Step 0 — get finetuned FP checkpoints

You need 21 finetuned `vit_base_patch16_224.orig_in21k` checkpoints, one per supported vision dataset, at the shared hyperparameter signature (`adamw, lr=1e-5, wd=0.1, ls=0.0, wl=500, mgn=1.0, bs=128, seed=2038`).

Two options:

(a) **Run the finetune** via the finetune script:
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py -m \
  model_name=vit_base_patch16_224.orig_in21k \
  dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet \
  batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0
```
This sweeps 21 finetunes sequentially. Several days of GPU time at the slow-task end (Flowers102, OxfordIIITPet, DTD have 70+ epoch counts).

(b) **Obtain them from someone who already ran (a)**, e.g.\ a collaborator. The checkpoints just need to land at the expected paths:
```
${CHECKPOINT_BASE_PATH}/vision/ilharco_timm_supervised/fp/vit_base_patch16_224_orig_in21k/{dataset}/optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/seed=2038/classifier_epoch_{N}.pt
```
Where `{N}` is the per-dataset epoch count from `code/src/vision/data/common.py`'s `DATASET_NAME_TO_EPOCHS`.

### Step 1 — dump features and predictions at W4-channel

```
uv run --active python code/experiments/vision/ilharco_timm_supervised/004_input_fragility/dump_pred_and_input_props.py -m \
  model_name=vit_base_patch16_224.orig_in21k \
  dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet \
  batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 \
  ptq.bits=4 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Runtime: ~15 min on a 2080 Ti. Output: 21 parquet pairs under `input_fragility_dumps/`.

### Step 2 — analyse

All four downstream scripts share `--bits`, `--granularity`, `--skip-modules`. They take seconds to a minute each.

```
# Per-task univariate + multivariate AUC.
uv run --active python code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/analyze_input_props.py \
  --model-name vit_base_patch16_224.orig_in21k --bits 4 --granularity channel --skip-modules head

# Per-task Pareto routing (same-task LogReg).
uv run --active python code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/pareto_routing.py \
  --model-name vit_base_patch16_224.orig_in21k --bits 4 --granularity channel --skip-modules head

# LOO cross-task Pareto routing. The headline experiment.
uv run --active python code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/loo_pareto_routing.py \
  --model-name vit_base_patch16_224.orig_in21k --bits 4 --granularity channel --skip-modules head

# Feature-subset ablation (image_only / q_only / fp_only / ... / all_features) under LOO.
uv run --active python code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/feature_ablation_pareto.py \
  --model-name vit_base_patch16_224.orig_in21k --bits 4 --granularity channel --skip-modules head

# Online threshold calibration (extends deployable claim to single-sample deployment).
uv run --active python code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/threshold_calibration.py \
  --model-name vit_base_patch16_224.orig_in21k --bits 4 --granularity channel --skip-modules head
```

Each script writes a markdown report and an HTML report to `plots/004_input_fragility/`.

### Step 3 — generate paper figures (optional)

```
uv run --active python code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/generate_paper_figures.py
```

Emits four PDFs to `paper/figs/`. The paper directory is `.gitignore`'d (it's a working draft); the figure generator is committed so anyone can reproduce the figures from the parquets.

### Step 4 — repeat at W3-channel (optional, robustness check)

Re-run steps 1–2 with `ptq.bits=3` and `--bits 3` respectively. This reproduces the regime-boundary finding (F5): the predictor still works qualitatively but the routing problem itself becomes intractable.

---

## Code layout

```
qat-transfer/                        (branch: input-fragility)
├── README.md                        ← this file
├── paper/                           ← TRACKED (build artifacts gitignored)
│   ├── short_main.tex               ← 5-page workshop version
│   ├── main.tex                     ← full version
│   ├── references.bib, neurips_2026.sty
│   ├── figs/                        ← figure PDFs+PNGs (reproducible, see below)
│   ├── tables/                      ← auto-generated LaTeX tables
│   └── scripts/verify_numbers.sh    ← re-derives every number from the parquets
├── code/
│   ├── src/                         ← shared infrastructure
│   │   ├── quantization.py          ← apply_ptq_, fake_quantize_tensor, QATLinear, …
│   │   ├── vision/data/             ← 21 vision dataset loaders + registry
│   │   ├── vision/ilharco_timm_supervised/   ← ImageClassifier, finetune_fp/qat
│   │   ├── text/data/               ← 11 MTEB loaders + registry
│   │   └── {repqvit,ptq4vit,aphq_vit}/       ← alternative PTQ methods (not used by the paper)
│   ├── experiments/
│   │   ├── vision/ilharco_timm_supervised/
│   │   │   ├── 000_baselines/       ← evaluate_fp.py, evaluate_fp_ptq.py, …
│   │   │   ├── 001_qat_transfer/    ← parent project
│   │   │   ├── 002_quant_steering/  ← abandoned line
│   │   │   └── 004_input_fragility/ ← THIS PROJECT
│   │   │       ├── dump_pred_and_input_props.py   ← Script A (GPU)
│   │   │       └── embedding_variance_probe.py    ← patch-embedding probe (negative result)
│   │   └── text/ilharco_automodelforsequenceclassification/004_input_fragility/
│   │       └── dump_pred_and_input_props.py       ← Script A, Qwen3
│   └── visualizations/
│       ├── vision/ilharco_timm_supervised/004_input_fragility/
│       │   ├── analyze_input_props.py     ← Script B      pareto_routing.py        ← Script C
│       │   ├── loo_pareto_routing.py      ← Script D      feature_ablation_pareto.py ← Script E
│       │   ├── threshold_calibration.py   ← Script F
│       │   ├── generate_paper_tables.py   ← LaTeX tables  → paper/tables/
│       │   ├── generate_paper_figures.py  ← figure PDFs   → paper/figs/
│       │   └── verify_paper_numbers.py    ← re-derives every reported number
│       └── text/ilharco_automodelforsequenceclassification/004_input_fragility/
│           └── analyze_qwen3.py           ← Qwen3 summary report
├── config/                          ← Hydra YAMLs mirroring code/ 1:1
└── storage/                         ← (gitignored) checkpoints + parquet dumps
```

Hydra scripts under `code/experiments/` and `code/src/` have matching YAMLs under `config/`.
Argparse scripts under `code/visualizations/` do not.

`storage/`, `plots/`, `evaluations/`, `logs/` and LaTeX build artifacts are gitignored. The paper
*sources*, figures and generated tables are tracked.

---

## Relation to the parent codebase

This branch sits on a larger quantization-aware-transfer codebase, which supplies the shared
infrastructure reused here: model-family wrappers, `apply_ptq_`, dataset loaders, finetune scripts.
Two sibling lines exist and are **not** part of this project:

- **`master`** — the original quantization-aware-transfer research (quantization vectors
  transferred across tasks). Defines most of `code/src/`.
- **`quant-steering`** — an abandoned line: recovering PTQ accuracy with rank-1 residual-stream
  steering vectors. Hit a ~26% gap-recovery ceiling. Do not reopen without a new mechanism reason.

All three consume the same on-disk finetuned-FP checkpoints under
`${CHECKPOINT_BASE_PATH}/…/fp/…`; switching branches never requires re-finetuning.

Conventions — file naming, Hydra config structure, sanitised model names, the `SPLIT_SEED=0` val
carve-out — are inherited and documented in [`CLAUDE.md`](CLAUDE.md). `004_input_fragility`
follows them.

---

## Limitations and open questions

**Limitations:**

- **Both models required at deployment.** The routing decision needs Q's logits, and the FP
  fallback must be available for routed inputs. This suits systems where Q is the cheap default
  and FP is selective — not a recipe for shipping pure-Q deployments.
- **Compute saving is hardware-conditional.** On vanilla GPUs without int4 kernels, weight-only
  PTQ is compute-equivalent to FP and the savings story evaporates. We report savings
  parametrically in the per-input speedup S.
- **Recoverable-regime bound.** The deployable claim is delimited to W4-channel and gentler. At
  W3-channel per-channel RTN, broken inputs are the majority and no routing scheme helps.
- **RTN only.** All results use naive round-to-nearest — deliberately, as the weakest weight-only
  PTQ, to isolate the routing question from the quantization-algorithm question. GPTQ, AWQ,
  SmoothQuant and QuaRot would each shift the recoverable boundary; none is tested here.
- **Prop. 2 is hypothesis-dependent.** It proves single-pass features cannot separate bad from
  lucky-Q *under a symmetry assumption* on the Q-logit distribution that we cannot verify
  empirically. The observed ceiling gap is consistent with it, not proof of it.

**Open questions:**

- **Stronger PTQ methods.** Re-running under GPTQ-class quantization is the natural next step:
  it would extend the recoverable regime and sharpen the deployment story. The implementations
  are already in `code/src/{repqvit,ptq4vit,aphq_vit}/` and `…/baselines/gptq.py`.
- **Breaking the lucky-Q ambiguity cheaply.** Is there a signal cheaper than a second forward pass
  that separates bad from lucky-Q? We tested one candidate and it failed: patch-embedding variance
  (`embedding_variance_probe.py`) is at or below chance (AUROC 0.43–0.50 across CIFAR10, Cars,
  SUN397) — fragility is not visible at the input-embedding stage. The pre-head representation
  remains unexplored and is the more promising place to look.
- **Per-backbone vs universal thresholds.** `q_margin` is scale-robust (X@90 stays in 18–22%
  across three backbones), but the *absolute* τ differs per task. Whether a single normalised
  threshold could ship across backbones is untested.
