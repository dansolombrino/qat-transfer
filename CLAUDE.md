# CLAUDE.md — qat-transfer

This file describes the conventions, structure, and coding style of this project so that Claude (and any human reader) can produce code that is consistent with the existing codebase.

Read `README.md` alongside this file: `README.md` carries the research narrative, the experiment-phase map, and the per-script command reference. This file carries the conventions you must follow when writing code.

---

## Operating mode — read this before running anything

Large directories are **not in git**. `.gitignore` excludes `evaluations/`, `storage/`, `plots/`, `logs/`, `references/`, `misc/`, `shitpads/`, `outputs/`, and all checkpoint file types (`*.pt`, `*.pth`, `*.safetensors`, ...).

A fresh clone therefore contains **code and configs only**. Before assuming a path exists, check it.

### Never run finetuning or experiment scripts

All scripts under `code/src/` and `code/experiments/{vision,text}/` are GPU-intensive, long-running jobs that additionally require checkpoints under `CHECKPOINT_BASE_PATH`. **Never execute them.** Only read, write, or modify their source code.

This holds for two independent reasons, and either alone is sufficient:

1. They cost GPU-hours to days per sweep.
2. Without `storage/` (the checkpoint tree, not in git) they fail immediately on a missing `.pt` file. A clone that has only the `evaluations/` tree cannot run them at all.

### What *is* runnable

| Path | Runnable? | Needs |
|---|---|---|
| `code/src/**` | **No** — read-only reference | GPU + datasets |
| `code/experiments/{vision,text}/**` | **No** — read-only reference (except the two argparse analysis phases below) | GPU + checkpoints |
| `code/experiments/vision/ilharco_timm_supervised/004_qv_alignment/**` | Yes | `storage/` checkpoints (CPU only) |
| `code/experiments/998_rebuttal/**` | Yes (except `measure_step_time.py`) | the `evaluations/` tree only |
| `code/visualizations/**` | Yes | the `evaluations/` tree only |
| `code/test/**` | Only with datasets present | HF datasets cache |

The one exception inside `998_rebuttal` is `002_cost_amortization/measure_step_time.py`, which is a Hydra script that builds a real model and times QAT steps on a GPU — treat it like `code/src/`.

Two analysis phases read `CHECKPOINT_BASE_PATH` even though the rule below says analysis scripts do not: `998_rebuttal/004_quantization_mechanism` and `vision/ilharco_timm_supervised/004_qv_alignment`. Both still build no model for inference, run no forward pass and need no GPU or dataset, so they keep the property that rule protects — but neither is runnable in a code-only clone, since both need `storage/`. The `pick_best_alpha.py` helpers under each `00N_qat_transfer/` are likewise argparse and CPU-only, but read `evaluations/` rather than checkpoints.

`collect_dataset_sizes.py` reads HuggingFace dataset metadata; pass `--offline` to keep it on the local cache.

### Checking what data is present

```
ls evaluations/                    # is the evaluations tree extracted at all?
find evaluations/<subtree> -name 'eval_results.json' | head
```

If `evaluations/` is missing, the analysis scripts cannot run. Obtain the evaluation handoff through the project's out-of-band channel; do not regenerate the data.

---

## Project overview

Quantization-aware transfer learning research.

Let `PT` be a pre-trained backbone, `FP_D` a full-precision finetune on task `D`, and `QAT_D` a quantization-aware finetune on the same task with quantization configuration `Q`. The **quantization vector** is the displacement between them:

```
QV = QAT_D - FP_D
```

The central claim is that this vector is largely **task-agnostic**: it encodes how to be robust to quantization, not how to do task `D`. So it can be computed once on a **donor** task and added to a *different* **receiver** task's FP checkpoint, buying most of the receiver's QAT benefit without ever running QAT on the receiver:

```
patched = FP_receiver + lambda * QV_donor
acc(ptq(patched))  ~=  acc(ptq(QAT_receiver))
```

The comparison that matters is `ptq(patched)` against `ptq(FP_receiver)` — i.e. against vanilla post-training quantization, which is what you would otherwise ship. `Delta = acc(transfer) - acc(fp_ptq)`, and a "win" is `Delta > 0`.

Supports vision (CLIP, OpenCLIP, timm) and text (AutoModelForSequenceClassification) model families.

### Glossary

| Term | Meaning |
|---|---|
| **QV** | Quantization vector, `QAT_D - FP_D`. A state-dict-shaped delta. |
| **donor** / **source** / `src` | The task the QV was computed on. |
| **receiver** / **target** / `tgt` | The task the QV is applied to. |
| **alpha** / **lambda** / **sf** (scaling factor) | **Synonyms** for the QV scaling coefficient. The code, config keys, and all filesystem paths say `alpha` (and `sf` in some visualization filenames); the paper and the `998_rebuttal` scripts say `lambda`. Prefer `lambda` in prose, keep `alpha` in code and paths. |
| **unit scaling** | `lambda = 1`. Needs no receiver data — this is the **data-free / zero-shot** setting the paper's headline claim rests on. |
| **best alpha** / `lambda_best` / `lambda*` | Scaling selected on the receiver's **validation** split, then reported on **test**. Needs receiver data, so it is *not* zero-shot. |
| **same-task pair** | donor == receiver. At `lambda = 1` this is algebraically just the receiver's own QAT checkpoint, so it is the **QAT ceiling**, not a transfer result. Exclude it from transfer statistics. |
| **cross-task pair** | donor != receiver. The genuine zero-shot transfer setting. |
| **recovery ratio** | `Delta / Delta_ceiling`, the fraction of the receiver's own QAT gain that transfer recovers. The observable side of Proposition 1's `cos^2` law. |
| **FP** | Full-precision finetuned checkpoint. |
| **QAT** | Quantization-aware-trained checkpoint. Saved with wrappers stripped, so its keys match FP exactly. |
| **PTQ** | Post-training quantization, applied at evaluation time via `apply_ptq_`. |

Note that QAT checkpoints are saved as plain `nn.Linear` state dicts (see *Checkpoint saving*), so loading one and running it is an ordinary FP forward pass whose weights merely carry the QAT-trained signature. Differencing two such checkpoints is well-defined, which is what makes the QV a plain state-dict subtraction.

### Canonical experimental configuration

Nearly every result in the repo uses:

```
seed=2038  optim=adamw  lr=1e-5  wd=0.1  ls=0.0  wl=500  max_grad_norm=1.0  bs=128
qat.bits=3  qat.granularity=channel
ptq.bits=3  ptq.granularity=channel
```

with the skipped module being the classification head, named per family: `[head]` for timm, `[classification_head]` for open_clip / hf_clip, `[classifier]` for text.

Backbones:

- **timm:** `deit3_base_patch16_224.fb_in1k`, `deit3_large_patch16_224.fb_in1k`, `swin_base_patch4_window7_224.ms_in22k_ft_in1k`, `swin_large_patch4_window7_224.ms_in22k_ft_in1k`, `vit_base_patch16_224.orig_in21k`, `vit_large_patch16_224.orig_in21k`, `vit_huge_patch14_224.orig_in21k`
- **open_clip:** `ViT-B-16 / laion2b_s34b_b88k`, `ViT-L-14 / laion2b_s32b_b82k`, `ViT-H-14 / laion2b_s32b_b79k`

---

## Environment

- **Python 3.11** (pinned in `.python-version`)
- **Package manager:** `uv` — use `uv sync` to install, `uv add` to add packages. Never use pip.
- **Virtual environment:** `.venv` at project root, managed by uv
- **Environment variables:** all scripts load `.env` via `dotenv` at startup. See `.env.example` for required vars (`CHECKPOINT_BASE_PATH`, `EVALUATION_BASE_PATH`, `HEAD_BASE_PATH`, `TORCH_NUM_WORKERS`, HF cache dirs, etc.)
- **Launch from repo root:** Hydra scripts resolve config search paths from `${oc.env:PWD}`, so always run from the project root — never from inside `code/` or `config/`. Visualization and `998_rebuttal` argparse entrypoints enforce this themselves with `os.chdir(_PROJECT_ROOT)`; legacy `pick_best_alpha.py` scripts do not.
- **Run command:** `uv run --active python <script.py> <overrides...>`

---

## Directory structure

```
qat-transfer/
  code/
    src/                  # Shared library code (models, data loaders, quantization)
    experiments/          # Evaluation, transfer & analysis scripts
    test/                 # Smoke tests for dataloading and modeling
    visualizations/       # Argparse plotting scripts
  config/                 # Hydra YAML configs — mirrors code/ 1:1
  evaluations/            # eval_results.json outputs            (gitignored)
  storage/                # Checkpoints and heads                (gitignored)
  plots/                  # Generated figures and LaTeX tables   (gitignored)
  logs/                   # Hydra run/sweep logs                 (gitignored)
  references/             # Papers, third-party code, submissions (gitignored)
  proofs/                 # Lean formalizations of the propositions
  journal.md              # Running log of baselines & experiment progress
```

### Key shared modules

| Module | Purpose |
|---|---|
| `code/src/quantization.py` | Core quantize/dequantize/fake-quantize functions, QATLinear, enable_qat_/disable_qat_/apply_ptq_ |
| `code/src/task_vectors.py` | TaskVector class for computing and applying checkpoint deltas |
| `code/src/vision/utils.py` | Sanitizers, set_seed, accuracy, LR schedulers, LabelSmoothing, tqdm helpers |
| `code/src/vision/data/common.py` | Vision dataset constants (DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES), split helpers |
| `code/src/vision/data/registry.py` | Vision dataset registry and get_dataset factory |
| `code/src/text/data/common.py` | Text dataset constants and HFTextDataset class |
| `code/src/text/data/registry.py` | Text dataset registry and get_dataset factory |

### Formal proofs

`proofs/` contains Lean formalizations backing the paper's theory:

| File | Content |
|---|---|
| `Proposition1Full.lean` | The `cos^2` recovery law relating transfer gain to donor/receiver QV alignment |
| `Proposition2Full.lean` | — |
| `QuantizationVectorsGenericallyWellDefinedFull.lean` | Well-definedness of the QV construction |
| `FakeQuantizerPreservesFiniteGaugeFull.lean` | Fake-quantizer gauge preservation |
| `AdamWKillsContinuousLinearReparameterizationsFull.lean` | AdamW and continuous linear reparameterizations |

These are not wired into any build; they are checked by hand with Lean.

---

## Code organization conventions

### Model families

Each model family gets its own subdirectory under `code/src/{vision,text}/`, `code/experiments/{vision,text}/`, `config/...`, etc. All follow the same internal structure:

- `code/src/{vision,text}/{family}/` — finetune scripts (finetune_fp.py, finetune_qat.py), modeling.py, heads.py
- `code/experiments/{vision,text}/{family}/` — numbered experiment phases
- `code/visualizations/{vision,text}/{family}/` — plotting scripts
- `config/src/{vision,text}/{family}/` — matching YAML configs
- `config/experiments/{vision,text}/{family}/` — matching YAML configs

Current families:
- **Vision:** `ilharco_hf_clip`, `ilharco_open_clip`, `ilharco_timm_supervised`, `ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head` (ablation: same as `ilharco_timm_supervised` but freezes biases, patch embeddings, norms and the classification head)
- **Text:** `ilharco_automodelforsequenceclassification`

Two directories sit *outside* the family taxonomy because they are cross-family by nature:

- `code/{experiments,visualizations}/998_rebuttal/` — analysis that aggregates across all three families at once
- `code/visualizations/mixed_modalities/{familyA}_{familyB}/` — figures juxtaposing a vision and a text family; the directory name concatenates the two family names with `_`

### Numbered experiment phases

Experiments are organized in numbered directories. New experiments get the next sequential number.

| Phase | Question | Location |
|---|---|---|
| `000_baselines` | What do FP, QAT and PTQ score on their own? | per family |
| `001_qat_transfer` | Does `FP_tgt + alpha * QV_src` recover `QAT_tgt`? The core experiment. | per family |
| `002_qat_transfer_reversed` | Does the QV work in reverse — `ptq(QAT_tgt) - alpha * QV_src`? | `vision/ilharco_timm_supervised` |
| `002z_qat_transfer_reversed_ptq_after_reverse` | Same as 002, but PTQ is applied *after* the subtraction rather than before. | `vision/ilharco_timm_supervised` |
| `003_qat_transfer_activ` | Does the same transfer work in **activation** space instead of weight space? | `vision/ilharco_timm_supervised` |
| `004_qv_alignment` | Does the similarity between donor and receiver QVs predict the observed transfer gain `Delta(D,R)`? The Euclidean, measurable side of Proposition 1's `cos^2` law. | `vision/ilharco_timm_supervised` |
| `998_rebuttal` | Reviewer-driven analyses; cross-family, reads existing evaluations. | top-level |
| `999_paper_stuff` | Camera-ready figures and LaTeX tables. Visualization-only, apart from one FLOPs computation. | per family |

Suffix conventions: a trailing letter (`002z`) marks a variant of the preceding phase rather than a new question. `_dryrun` in an output path marks a smoke-test run (`limit_num_batches` / `limit_num_epochs` set) and must never be mixed with real results.

#### `998_rebuttal` sub-phases

| Sub-phase | Question |
|---|---|
| `001_zero_shot_reframing` | How does the method read when framed as zero-shot? Win rates at `lambda=1` vs `lambda*`, recovery ratios, the `lambda_best` distribution, and how many `lambda=1` failures are merely overshoot of a positively-aligned direction. |
| `002_cost_amortization` | The donor's QAT run is a real one-off cost. Over how many receivers does it amortize before QV transfer undercuts per-task QAT? Also: which donors are worth their cost. |
| `003_lambda_sensitivity` | How sensitive is patching to `lambda`? Safe interval, plateau width, unit retention, unimodality. **In progress** — only the timm curves have been computed; open_clip and text are pending. |
| `004_quantization_mechanism` | *Why* does patching help — does it make the weights easier to quantize (H1), or leave the quantization error the same size but move it somewhere the function ignores (H2)? Weight-space only, with `random` / `shuffle` / `taskvec` nulls. |

Within `998_rebuttal`, the per-family scripts are named `<verb>_<noun>_<family>.py` (`compute_win_loss_timm_supervised.py`, `compute_lambda_curves_open_clip.py`, ...) and share family-independent math via a `*_common.py` module in the same directory. Only path logic differs between families and it stays in the per-family script — follow this split when adding a family.

### config/ mirrors code/ 1:1

Every Hydra script at `code/a/b/script.py` has a matching YAML at `config/a/b/script.yaml`. When adding a new Hydra script, always create both the Python script and its YAML config.

Analysis and visualization scripts are argparse-based and have no YAML.

### Three script families

| Family | Where | CLI | Compute |
|---|---|---|---|
| **Hydra scripts** | `code/src/`, `code/experiments/{vision,text}/`, `998_rebuttal/002_cost_amortization/measure_step_time.py` | `@hydra.main`, `DictConfig`, OmegaConf resolvers | GPU |
| **Analysis scripts** | `code/experiments/998_rebuttal/`, `pick_best_alpha.py` under each `00N_qat_transfer/` | `argparse.ArgumentParser` | CPU; reads and writes JSON under `evaluations/` |
| **Visualization scripts** | `code/visualizations/` | `argparse.ArgumentParser` | CPU; reads `evaluations/`, writes `plots/` |

Analysis and visualization scripts never touch checkpoints, never build a model, and never need a GPU. Keep it that way — if an analysis needs a forward pass, it belongs under `code/experiments/{vision,text}/` as a Hydra script.

---

## Import ordering and boilerplate

Every script follows this exact import order:

```python
# 1. sys.path setup (MUST be first)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[N]))

# 2. dotenv (MUST run before any HF/dataset imports)
from dotenv import load_dotenv
load_dotenv()

# 3. Standard library
import copy
import json
import logging
import os

# 4. Third-party
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm

# 5. Local project imports
from src.quantization import enable_qat_, disable_qat_, apply_ptq_
from src.vision.data.common import DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import set_seed, sanitize_timm_model_name, ...
```

The `parents[N]` value is the number of directory levels from the script to the `code/` directory. For example, `code/src/vision/ilharco_timm_supervised/finetune_fp.py` uses `parents[3]`.

The dotenv comment is important — HF libraries snapshot env vars at import time, so `.env` must be loaded before importing anything from `transformers`, `datasets`, `huggingface_hub`, or our own data modules.

Argparse entrypoints under `code/visualizations/` and `code/experiments/998_rebuttal/` additionally pin the working directory, since every path they build is repo-root-relative:

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[N]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)
```

The legacy `pick_best_alpha.py` helpers do not include this guard and must be invoked from the project root.

---

## Hydra conventions

```python
OmegaConf.register_new_resolver(
    "sanitize_timm", sanitize_timm_model_name, replace=True
)

@hydra.main(
    config_path="../../../../config/src/vision/ilharco_timm_supervised",
    config_name="finetune_fp",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    ...

if __name__ == "__main__":
    main()
```

- `config_path` is always a relative path from the script to its YAML config directory.
- `version_base=None` always.
- OmegaConf resolvers are registered at module level, before `@hydra.main`. Use `replace=True`.
- Config values accessed via dot notation: `cfg.model_name`, `cfg.qat.bits`, etc.
- Required params use `???` in YAML (Hydra's mandatory marker).

### YAML config structure

```yaml
model_name: ???
dataset_name: ???
seed: ???
gpu: ???
batch_size: 128
lr: 1e-5
# ... other hyperparameters ...
limit_num_batches: null
limit_num_epochs: null

# QAT configs (only in finetune_qat.yaml and experiment configs that need it)
qat:
  bits: ???
  granularity: ???
  skip_modules: ???    # list[str], ALWAYS required, no defaults

hydra:
  searchpath:
    - file://config
  run:
    dir: logs/.../${sanitize_timm:${model_name}}/${dataset_name}/seed=${seed}
  sweep:
    dir: logs/...
    subdir: ${sanitize_timm:${model_name}}/${dataset_name}/seed=${seed}
```

Transfer experiments additionally carry `source`/`target` groups (each with `dataset_names` and `seed`), a `qv` group (`alpha`), and a `ptq` group mirroring `qat`.

---

## Path construction

Paths are built with `os.path.join(*parts_list)` — never pathlib for runtime paths. Hyperparameters are encoded directly in directory names.

### Checkpoint paths

**Vision:**
```
{CHECKPOINT_BASE_PATH}/vision/{family}/{fp,qat}/{sanitized_model}/{dataset}/optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={max_grad_norm}_bs={batch_size}/[qat=bits={bits}_gran={granularity}_skip={skip_tag}/]seed={seed}/classifier_epoch_{N}.pt
```

**Text:** same but uses `_ml={max_length}` instead of `_wl={wl}`:
```
{CHECKPOINT_BASE_PATH}/text/{family}/{fp,qat}/{sanitized_model}/{dataset}/optim=adamw_lr={lr}_wd={wd}_ls={ls}_mgn={max_grad_norm}_bs={batch_size}_ml={max_length}/[qat=bits={bits}_gran={granularity}_skip={skip_tag}/]seed={seed}/backbone_epoch_{N}.pt
```

### Evaluation paths

Baselines:
```
{EVALUATION_BASE_PATH}/{vision,text}/{family}/{phase}/{vision,text}/{experiment_type}/{sanitized_model}/{dataset}/[optim=.../][qat=.../][ptq=.../]seed={seed}/eval_results.json
```

`{experiment_type}` is the baseline variant and names the directory directly: `fp`, `fp_ptq`, `fp_gptq`, `qat`, `qat_ptq`, `pretrained`, `pretrained_ptq`, plus `*_dryrun` counterparts. `fp_gptq` paths carry a `gptq=bits=..._gran=..._skip=..._ncal=..._percdamp=..._actorder=...` fragment in place of the `ptq=` fragment (`block_size` is deliberately excluded — result-invariant solver batching).

Transfer:
```
{EVALUATION_BASE_PATH}/{vision,text}/{family}/{phase}/{vision,text}/qv_transfer/{sanitized_model}/src={donor}_seed={s}/tgt={receiver}_seed={s}/optim=.../qat=bits=..._gran=..._skip=.../ptq=bits=..._gran=..._skip=.../qv=alpha={alpha}/split={val,test}/eval_results.json
```

Note the doubled modality segment (`.../ilharco_timm_supervised/001_qat_transfer/vision/qv_transfer/...`) — it is redundant but load-bearing for every existing path, so keep emitting it.

The `split={val,test}` leaf is what makes validation-based alpha selection possible: `pick_best_alpha.py` scans `split=val` across the alpha grid and the reported number is read from `split=test` at the selected alpha.

`998_rebuttal` writes aggregate JSONs rather than per-run `eval_results.json`, under:
```
evaluations/998_rebuttal/{sub_phase}/seed={seed}/qat=.../ptq=.../split={split}/<name>.json
```
with `002_cost_amortization` writing flat files directly under its sub-phase directory.

### Plot paths

Visualization output mirrors the evaluation grammar:
```
plots/{vision,text,mixed_modalities}/{family}/{phase}/{plot_name}/{model_tag}/seed={seed}/[optim_frag/]{qat_frag}/{ptq_frag}/[qv_frag/]{split_frag}/<file>
```

The fragments are built by module-level helpers (`_ptq_frag`, `_split_frag`, ...) shared by shape across the visualization scripts. **The `ptq_frag` segment is mandatory** — it was added so that runs sharing a QAT configuration but differing in evaluation-time PTQ no longer overwrite each other. When adding a plot script, include it.

### Path construction pattern

```python
save_dir_parts = [
    checkpoint_base_path,
    "vision",
    "ilharco_timm_supervised",
    "fp_dryrun" if is_dryrun else "fp",
    sanitize_timm_model_name(cfg.model_name),
    cfg.dataset_name,
    f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
    f"seed={cfg.seed}",
]
save_dir = os.path.join(*save_dir_parts)
os.makedirs(save_dir, exist_ok=True)
```

For QAT paths, the skip_modules tag is built as:
```python
skip_modules_sorted = sorted(cfg.qat.skip_modules)
skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
```

### Model name sanitizers

Each family uses a different sanitizer from `code/src/vision/utils.py`:

| Family | Sanitizer | Rule |
|---|---|---|
| `ilharco_hf_clip`, text families | `sanitize_hf_model_name` | `/` and `-` to `_` |
| `ilharco_timm_supervised` | `sanitize_timm_model_name` | `/`, `-`, `.` to `_` |
| `ilharco_open_clip` | `sanitize_open_clip_model_name` | combines `(model, pretrained)` with `__`, then `/`, `-`, `.` to `_` |

---

## Quantization conventions

### skip_modules is always explicit

**No defaults.** Every function in `quantization.py` that accepts `skip_modules` requires it as a mandatory argument. Every YAML config that involves quantization uses `skip_modules: ???`. Never provide a default value.

### In-place mutation convention

Functions that mutate the model in-place end with `_`:
- `enable_qat_(model, bits, granularity, skip_modules)` — wraps nn.Linear in QATLinear
- `disable_qat_(model)` — unwraps QATLinear back to nn.Linear
- `apply_ptq_(model, bits, granularity, skip_modules)` — fake-quantizes weights in-place

`apply_ptq_` mutates `Linear` weights **without replacing modules**, so any forward hooks registered on those modules survive it. `003_qat_transfer_activ` depends on this: its activation-injection hooks stay live across the PTQ call, giving quantized weights plus activation steering in the same forward pass.

### QAT evaluation pattern (deepcopy to CPU)

```python
eval_model = copy.deepcopy(classifier).to("cpu")
disable_qat_(eval_model)
apply_ptq_(eval_model, bits=bits, granularity=granularity, skip_modules=skip_modules)
eval_model.to(device)
eval_model.eval()
# ... evaluate ...
del eval_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Deepcopy to CPU first to avoid VRAM peaks from holding two full copies simultaneously.

---

## SLURM-aware logging

Every training/evaluation script detects SLURM and adjusts output accordingly:

```python
IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
```

- **Local (interactive):** `print()` / `pprint()` / tqdm progress bars with `random_tqdm_color()`
- **SLURM:** `log.info()` (Python logging), tqdm disabled, periodic logging every `LOG_EVERY` steps

Config printing at script start:
```python
if IS_SLURM:
    log.info("cfg:\n%s", dict(cfg))
else:
    pprint(dict(cfg), expand_all=True)
```

---

## Training loop patterns

### Gradient accumulation

```python
REFERENCE_BATCH_SIZE = 128  # vision (32 for text)
assert REFERENCE_BATCH_SIZE % cfg.batch_size == 0
accum_steps = REFERENCE_BATCH_SIZE // cfg.batch_size

loss = loss_fn(logits, labels) / accum_steps
loss.backward()
if (i + 1) % accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
    optimizer.step()
    scheduler(opt_step)
    optimizer.zero_grad()
```

### Optimizer and scheduler

- **Optimizer:** always `torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)`
- **LR schedule (vision):** `cosine_lr(optimizer, cfg.lr, cfg.wl, total_steps)` — cosine decay with linear warmup
- **LR schedule (text):** `linear_lr(optimizer, cfg.lr, total_steps)` — linear decay, no warmup
- **Loss:** `LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()`

### Epoch structure

```
for epoch in epochs:
    model.train()    → train loop with gradient accumulation
    model.eval()     → per-epoch validation (val_loader)
    save checkpoints (per-epoch if limit_num_epochs set)
model.eval()         → final test evaluation (test_loader)
save final checkpoints
```

### Split constants (shared across all domains)

```python
SPLIT_SEED = 0          # deterministic train/val splits
VAL_FRACTION = 0.1
MAX_VAL_SAMPLES = 5000
```

`SPLIT_SEED` is independent from the per-run `seed` (which controls shuffling, init, etc.). Because it is fixed, the validation slice used for alpha selection is the same slice across every run — which is what makes `pick_best_alpha` comparable across configurations.

---

## Results format

Results are always JSON. Never CSV, never pickle.

```python
results = {
    "model_name": cfg.model_name,
    "dataset_name": cfg.dataset_name,
    # ... all hyperparameters ...
    "test_accuracy": test_accuracy,
    "encoder_path": classifier_path,   # or "backbone_path"
}
os.makedirs(eval_dir, exist_ok=True)
with open(os.path.join(eval_dir, "eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)
```

Transfer runs write four accuracies per donor-receiver pair — the receiver's FP head and its QAT head, each before and after PTQ.

Analysis scripts under `998_rebuttal` follow the same rule for their aggregates: JSON in, JSON out, one file per question, written back under `evaluations/`. A visualization script must never recompute a statistic an analysis script already wrote — read its JSON.

---

## Naming conventions

### Files

- Finetuning scripts: `finetune_fp.py`, `finetune_qat.py`
- Evaluation scripts: `evaluate_{fp,qat,pretrained}{_ptq}.py`
- Transfer scripts: `qv_transfer.py`
- Alpha selection: `pick_best_alpha.py`
- Analysis scripts: `compute_*.py`, `aggregate_*.py`, `collect_*.py`, `measure_*.py`
- Visualization: `qv_transfer_heatmap.py`, `qv_transfer_heatmap_best_sf.py`, `radar_plot.py`, `donor_receiver_table_*.py`, `baseline_bar*.py`, `scatterplot*.py`, `stacked_bar.py`, `win_loss_table.py`, `*_curve.py`
- Model wrappers: `modeling.py`
- Classification heads: `heads.py`

### Variables

- `snake_case` for everything
- Common abbreviations: `cfg` (DictConfig), `sd` (state dict), `bs` (batch size), `ml` (max_length), `wl` (warmup length), `ls` (label smoothing), `wd` (weight decay), `mgn` (max grad norm), `lnb` (limit num batches), `lne` (limit num epochs), `sf` (scaling factor, == alpha == lambda)
- Suffixes: `_dir` (directory path), `_path` (file path), `_loader` (DataLoader), `_dataset` (Dataset), `_bar` (tqdm bar), `_frag` (a path fragment string)

### Functions

- `snake_case`
- Trailing `_` for in-place mutation: `enable_qat_()`, `apply_ptq_()`, `disable_qat_()`
- Leading `_` for private helpers: `_evaluate_ptq()`, `_fp_ckpt_dir()`, `_is_head_key()`, `_ptq_frag()`, `_split_frag()`
- Factories/builders: `get_dataset()`, `build_model_and_tokenizer()`

### Classes

- `PascalCase`: `QATLinear`, `QuantizedLinear`, `ImageClassifier`, `TaskVector`, `LabelSmoothing`

### Constants

- `UPPER_SNAKE_CASE`: `SPLIT_SEED`, `VAL_FRACTION`, `TQDM_KW`, `IS_SLURM`, `SUPPORTED_MODELS`, `REFERENCE_BATCH_SIZE`, `LOG_EVERY`, `EVAL_ROOT`, `PLOT_ROOT`

---

## Section markers in long scripts

Experiment scripts use section markers to delineate logical blocks:

```python
############################################################################
# BEGIN checkpoint loading
############################################################################

... code ...

############################################################################
# END checkpoint loading
############################################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

############################################################################
# BEGIN dataset creation
############################################################################
```

Analysis and visualization scripts use the lighter form instead:

```python
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
```

---

## Docstrings

Experiment and analysis scripts open with a substantial module docstring that states **the question being asked and why the methodology answers it** — not a summary of the code. The `998_rebuttal` scripts are the reference standard: they name the reviewer objection, state the cost or statistical model, and justify each methodological choice ("none is cosmetic"). Match that register when adding a script; a one-line docstring on an experiment script is a defect.

Vision transfer scripts instead open with a boxed `# =====` header defining the notation (`PT`, `FP_{S1}^{D1}`, `QAT_{S1,Q}^{D1}`, `QV`) and then the specific hypothesis. Follow the local convention of the directory you are adding to.

---

## Checkpoint saving

### Vision models

Save the full classifier state dict and the head separately:
```python
classifier.save(classifier_path)                    # classifier_epoch_{N}.pt
torch.save(classifier.model.head, head_path)        # head_epoch_{N}.pt
```

### Text models

Save backbone (everything except head) and head separately:
```python
torch.save({k: v for k, v in sd.items() if not _is_head_key(k, head_module)}, backbone_path)
torch.save({k: v for k, v in sd.items() if _is_head_key(k, head_module)}, head_path)
```

### QAT checkpoint saving

Before saving a QAT-trained model, always strip QAT wrappers so the saved state dict has plain nn.Linear keys (same format as FP checkpoints):
```python
disable_qat_(classifier)
classifier.save(path)
```

This is what makes `QV = QAT - FP` a well-defined state-dict subtraction — the two checkpoints share a key set exactly.

---

## Visualization scripts

Use `argparse` (not Hydra). Structure:

```python
"""Docstring with description."""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[N]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

import plotly.graph_objects as go
# ...
```

- Use plotly for figures.
- **Output format depends on the phase.** Exploratory scripts (`code/visualizations/{vision,text}/{family}/00N_*/`) write **HTML** to `plots/`. Camera-ready scripts (`999_paper_stuff/`) write **PDF**, and the `donor_receiver_table_*` / `win_loss_table` scripts write a **LaTeX** table to `plots/` while printing the same table as plain text to stdout. `998_rebuttal` visualizations write HTML.
- Constants at module level for eval roots and baseline method labels (`EVAL_ROOT`, `PLOT_ROOT`).
- Every hyperparameter that appears in an evaluation path is a **required** CLI argument — there are no defaults for `--seed`, `--qat-bits`, `--ptq-bits`, `--granularity`, `--skip-modules`, etc. This is deliberate: a defaulted hyperparameter silently reads the wrong sweep. Keep it.
- Scripts that compare configurations take plural arguments (`--model-names`, `--batch-sizes`) whose ordering must match.

---

## Device handling

Every GPU-using script requires a mandatory `gpu` parameter (int, no default) specifying the CUDA device index. Analysis and visualization scripts take no `gpu` argument, because they never build a model.

- **Hydra scripts:** `gpu: ???` in YAML config, accessed as `cfg.gpu`
- **Argparse scripts:** `--gpu` required CLI arg (type=int)

```python
device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
model.to(device=device, dtype=torch.float32)
```

Always use `torch.device(...)` explicitly. Move to device after model construction but before creating the optimizer.
