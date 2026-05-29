# Input fragility — predicting which inputs PTQ breaks

> **Branch:** `input-fragility`. This branch is a focused experimental project distinct from `master`. The original `master` repo hosts the parent quantization-aware-transfer research codebase; this branch reuses the shared infrastructure (finetuned ViT checkpoints, dataset loaders, PTQ utilities) and adds one new self-contained experimental phase that asks and answers a single research question.
>
> If you are reading this without context: read this README front to back. It explains what we're studying, why, how the code is laid out, what we found, and how to reproduce the results.

---

## TL;DR — what's the project and what did we find?

**Question.** Weight-only post-training quantization (PTQ) on a Vision Transformer degrades accuracy non-uniformly across inputs. The vast majority of inputs the full-precision (FP) model gets right are also handled fine by the quantized (Q) model. A small minority gets flipped. *Can we predict which inputs PTQ will flip, cheaply enough to be useful for deployment?*

**Setup.** 21 vision classification tasks, each with a finetuned `vit_base_patch16_224.orig_in21k` checkpoint. Weight-only PTQ at W4-channel is applied to every linear layer except the classification head, producing a quantized variant of each finetuned model. For each (FP, Q) pair we dump per-sample features and predictions on val + test splits and study the relationship.

**Headline result.** A logistic regression on **three** features — the Q model's top-1/top-2 logit margin, top-1 softmax, and entropy — trained on a pool of 17 source tasks under leave-one-out cross-task transfer, routes **~24%** of unseen target-task test inputs to a full-precision fallback and recovers **~90%** of the FP→PTQ accuracy gap. The same classifier transfers across 18 tasks without per-task retraining.

**Mechanism finding.** The 24% deployable ceiling is bounded by what we call the **lucky-Q ambiguity**: features computable from a single forward pass cannot distinguish (a) inputs PTQ broke (bad) from (b) inputs where PTQ happens to be right and FP is wrong (lucky-Q). They have indistinguishable confidence patterns. Cross-model features that compare what FP thinks of Q's prediction *can* distinguish them — but they require running both models, defeating the compute saving. With cross-model features, X@90% drops to ~7% (the diagnostic ceiling) and the gap from the deployable claim quantifies the lucky-Q ambiguity.

**Negative result, worth keeping.** Raw image pixel statistics (brightness, contrast, edge density, FFT high-frequency ratio) carry **zero** signal for PTQ fragility across all 18 tasks. PTQ-fragility is purely a property of the model's representation of the input, not of pixel-level input properties.

**Regime caveat.** This works at W4-channel where the FP→PTQ gap is small and "bad" inputs are a minority. At catastrophic regimes (W3-channel) where vanilla weight-only PTQ destroys most predictions, no input-aware routing can recover compute — even oracle needs to route ~54% of inputs to FP.

The paper draft is in `paper/main.tex` (gitignored). The detailed finding log is in `INPUT_FRAGILITY_LOG.md`.

---

## Table of contents

1. [Background and definitions](#background-and-definitions)
2. [The research question, precisely](#the-research-question-precisely)
3. [Pipeline overview](#pipeline-overview)
4. [Scripts in detail](#scripts-in-detail)
5. [Key findings](#key-findings)
6. [Reproducibility — how to run this from scratch](#reproducibility--how-to-run-this-from-scratch)
7. [Code layout](#code-layout)
8. [Relation to other branches and shared infrastructure](#relation-to-other-branches-and-shared-infrastructure)
9. [Limitations and open questions](#limitations-and-open-questions)

---

## Background and definitions

### Weight-only PTQ for ViTs

Weight-only post-training quantization replaces every `nn.Linear` weight matrix's float32 values with a low-bit-width approximation, rounding-to-nearest, after the model has been trained. Activations stay float32. There's no retraining and no calibration data; just one in-place pass over the weights.

This codebase's PTQ is implemented in [`code/src/quantization.py`](code/src/quantization.py): the function `apply_ptq_(model, bits, granularity, skip_modules)` walks `model.named_children()` and replaces each Linear's weight with its fake-quantized version. After applying it, the model still has float32 weights, but those weights now lie on the quantization grid — running a forward pass produces the same outputs as a true int-N inference engine would.

At inference time on dedicated low-bit-width hardware this would translate to memory and compute savings; on vanilla GPUs without int4 kernels, it's compute-equivalent to FP but quantization-faithful in the outputs.

Two important hyperparameters:
- **`bits`** ∈ {3, 4, 5, …}: target bit-width. We focus on W4 (4 bits) at this branch.
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
│  shared with master │   + apply PTQ in place   ───► │  + dump_metadata.json     │
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

**Inputs to the whole pipeline:** the 21 finetuned ViT checkpoints already on disk under `${CHECKPOINT_BASE_PATH}/vision/ilharco_timm_supervised/fp/...` (provided externally or produced by `code/src/vision/ilharco_timm_supervised/finetune_fp.py` on master).

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

These are the findings as recorded in [`INPUT_FRAGILITY_LOG.md`](INPUT_FRAGILITY_LOG.md), abbreviated. Each `Fk` is dated in the log.

### F1/F2 — fragility is predictable from FP/Q confidence; image stats carry zero signal

On 18 of 21 finetuned ViT-B tasks at W4-channel (three small-PTQ-gap tasks dropped: Flowers102, OxfordIIITPet, STL10), multivariate logistic regression on 15 features fits to test AUC = **0.926 ± 0.047**. Univariate discriminative AUC per feature, mean across 18 tasks:

| Feature | Discriminative AUC |
|---|---|
| q_margin | 0.937 |
| fp_margin | 0.919 |
| q_softmax_top1 / fp_softmax_top1 | ~0.90 |
| fp_cls_dist_to_class_centroid | 0.834 |
| fp_entropy / q_entropy | ~0.81 |
| img_brightness / contrast / edge_density / high_freq_ratio | ~0.49–0.50 (no signal) |

PTQ fragility is a model-representation phenomenon. Pixel-level statistics don't predict it.

### F3 — cross-model features close most of the gap to oracle but are not deployable

Adding the 6 cross-model features (`fp_logit_at_q_pred`, `q_softmax_at_fp_pred`, etc.) drops Pareto X@90% from 40% → **7%** across 18 tasks. The mechanism is that bad inputs have a specific FP/Q-disagreement signature (Q stole FP's runner-up) distinct from lucky-Q inputs (Q picked something FP ranked deep). The features quantify this.

The catch: computing those features requires running both models, which defeats the compute saving. So this is a diagnostic ceiling, not a deployment recipe.

### F4 — cross-task LOO generalises; one classifier ships across tasks

With `all_features`, LOO X@90% = 6.6% ± 8.6, beating same-task X@90% = 8.4% ± 9.1. LOO is *better* than same-task because the source pool provides 17× the training data, and per-task standardisation strips task-specific feature scales.

With Q-only deployable features (3 features), LOO X@90% = 24.4% vs same-task 22.0% — within 2.5 pp. One classifier transfers across tasks at near-same-task efficiency. See `Table 2` of the paper draft.

### F5 — bit-width robustness; sharp regime boundary at W3

At W3-channel, mean PTQ test accuracy collapses to ~25%, and the bad-input population becomes the majority of test inputs (~70%). Oracle X@90% climbs from 1.9% (W4) to 53.8% (W3). Deployable Q-only X@90% goes from 24% to 81%. The predictor still ranks fragility correctly; it's the routing problem itself that's intractable when bad inputs are the majority.

**Practical implication.** The deployable routing recipe applies at recoverable PTQ regimes only (W4-channel and above). Harsher regimes need a stronger underlying PTQ method (SmoothQuant, AWQ, QuaRot) to recover useful accuracy before any input-aware routing can help.

### F6 — feature ablation; deployable headline is Q-only, not all-features

At W4-channel:

| Subset | Deployment scenario | X@90% |
|---|---|---|
| image_only | no model | 79.2% |
| **q_only** | **PTQ-first deployable** | **24.4%** |
| q_plus_image | PTQ-first + image | 24.5% (image adds 0) |
| fp_only | FP-side only | 36.0% |
| fp_plus_q_no_cross | both models, no cross | 29.1% |
| **all_features** | diagnostic ceiling | **6.6%** |

Image features add nothing; Q-side beats FP-side; cross-features close most of the gap to oracle (1.9%) but at the cost of running both models. The 18-pp gap from Q-only (24%) to the ceiling (7%) is the cost of the lucky-Q ambiguity.

### F7 — batch deployment is rock-solid; online is noisier

Batch routing (sort test set by P(bad), route top X%) on 18 W4-channel tasks: mean 26.7% routing → 91.0% recovery, σ = 2.3 pp. Stable. Drop-in deployment.

Online fixed-threshold routing strategies (a single τ, no batch sorting): mean 24–37% routing → 70–82% recovery, σ in the tens of pp, driven largely by small-PTQ-gap outlier tasks. Workable but noisier.

The paper's primary claim is the batch number; the online table extends to latency-sensitive deployments.

---

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

(a) **Run the finetune** via the script under master:
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
qat-transfer/
├── README.md                           ← this file (input-fragility branch)
├── INPUT_FRAGILITY_LOG.md              ← dated finding log (F1–F7+)
├── CLAUDE.md                           ← project conventions; mostly inherited from master
├── paper/                              ← (gitignored) paper draft and figures
│   ├── main.tex
│   ├── references.bib                  ← still mostly TODO
│   └── figs/*.pdf                      ← reproducible via generate_paper_figures.py
├── code/
│   ├── experiments/vision/ilharco_timm_supervised/
│   │   ├── 000_baselines/              ← inherited from master (evaluate_fp_ptq.py etc.)
│   │   ├── 001_qat_transfer/           ← inherited
│   │   └── 004_input_fragility/        ← THIS BRANCH'S WORK
│   │       ├── dump_pred_and_input_props.py   ← Script A (GPU)
│   │       └── (no other scripts here; analyses are visualization-style argparse)
│   ├── visualizations/vision/ilharco_timm_supervised/
│   │   └── 004_input_fragility/        ← THIS BRANCH'S WORK
│   │       ├── analyze_input_props.py     ← Script B
│   │       ├── pareto_routing.py          ← Script C
│   │       ├── loo_pareto_routing.py      ← Script D
│   │       ├── feature_ablation_pareto.py ← Script E
│   │       ├── threshold_calibration.py   ← Script F
│   │       └── generate_paper_figures.py  ← paper figures
│   └── src/                            ← inherited shared infrastructure
│       ├── quantization.py             ← apply_ptq_, fake_quantize_tensor, …
│       ├── vision/data/                ← dataset loaders, registry, common.py
│       └── vision/ilharco_timm_supervised/modeling.py  ← ImageClassifier
├── config/                             ← Hydra YAMLs mirroring code/
│   └── experiments/vision/ilharco_timm_supervised/
│       └── 004_input_fragility/dump_pred_and_input_props.yaml
└── storage/                            ← (gitignored) checkpoints + dumps live here
```

Hydra scripts under `code/experiments/` and `code/src/` have matching YAML configs under `config/`. Argparse scripts under `code/visualizations/` don't.

`storage/`, `plots/`, `evaluations/`, `logs/`, and `paper/` are gitignored.

---

## Relation to other branches and shared infrastructure

This branch is one of three research projects sharing the same codebase:

- **`master`** — the original quantization-aware-transfer line of research. Defines the shared infrastructure: ViT and CLIP family wrappers, `apply_ptq_`, dataset loaders for 22 vision tasks, finetune scripts.
- **`quant-steering`** — an abandoned line. Tried to recover PTQ accuracy by adding rank-1 steering vectors to the residual stream. Hit a hard ceiling of ~26% gap recovery on the best task; written up as a dead-end in `QUANT_STEERING_LOG.md` on that branch. **Do not reopen** without a new mechanism reason.
- **`input-fragility`** (this branch) — the project this README describes.

The three branches all consume the same on-disk finetuned-FP checkpoints (~7 GB), which live under `${CHECKPOINT_BASE_PATH}/vision/ilharco_timm_supervised/fp/...`. Switching branches doesn't require re-finetuning; only re-running the per-branch analysis scripts.

Conventions (file naming, Hydra config structure, sanitised model names, the SPLIT_SEED=0 val carve-out, etc.) are inherited from master and documented in [`CLAUDE.md`](CLAUDE.md). 004_input_fragility follows them.

---

## Limitations and open questions

**Limitations to flag in any writeup or discussion:**

- **Both models required at deployment**: the routing decision needs the Q model to have already produced its logits (for the Q-only deployable feature set), and the FP fallback to be available for the routed inputs. This is a recipe for systems where both models are co-deployed (Q is the cheap default, FP is selective), not a recipe for shipping pure-Q deployments.
- **Compute-saving requires hardware difference**: on vanilla GPUs without int4 kernels, weight-only PTQ is compute-equivalent to FP and the routing-saves-compute story is hardware-conditional.
- **Recoverable-regime caveat**: the deployable claim is empirically delimited to W4-channel and gentler. At catastrophic regimes (W3-channel, W4-tensor on this codebase), bad inputs are the majority of test inputs and no routing scheme helps.
- **Single backbone tested**: `vit_base_patch16_224.orig_in21k`. The mechanism (FP/Q confidence + cross-model disagreement) is plausibly architecture-agnostic but we have not validated on ViT-S, ViT-L, or non-ViT backbones.

**Open questions worth a follow-up paper or section:**

- **Stronger PTQ methods** (GPTQ, AWQ, SmoothQuant, QuaRot) produce gentler degradation at W3 and below. Re-running this analysis with one of those as the underlying PTQ would extend the recoverable regime and tighten the deployment story to "near-fp accuracy at W3-class memory savings, with X% routing budget."
- **Calibrating online τ better.** Script F's val-percentile calibration is label-free but adds variance. A small target-task labeled set could pick τ more precisely; the labeled-cost vs.\ recovery trade-off is worth quantifying.
- **The lucky-Q ambiguity bound.** Is there a feature *cheaper than running both models* that breaks the ambiguity? Plausibly something derived from intermediate Q activations (not just final logits) carries enough information; we have not explored this.
- **Architecture transfer.** Does a LogReg trained on ViT-B fragility transfer to ViT-L? Or does each backbone need its own classifier? Plausible that the feature distributions differ enough that a one-classifier-per-backbone setup is the natural unit.
