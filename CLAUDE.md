# CLAUDE.md — qat-transfer

This file describes the conventions, structure, and coding style of this project so that Claude (and any human reader) can produce code that is consistent with the existing codebase.

---

## Never run finetuning or experiment scripts

All scripts under `code/src/` and `code/experiments/` are GPU-intensive, long-running jobs. **Never execute them.** Only read, write, or modify their source code.

---

## Project overview

Quantization-aware transfer learning research. The core idea: compute a "quantization vector" (QV = QAT checkpoint - FP checkpoint) on a source task and transfer it to a target task at various scaling factors (alpha). Supports vision (CLIP, OpenCLIP, timm) and text (AutoModelForSequenceClassification) model families.

---

## Environment

- **Python 3.11** (pinned in `.python-version`)
- **Package manager:** `uv` — use `uv sync` to install, `uv add` to add packages. Never use pip.
- **Virtual environment:** `.venv` at project root, managed by uv
- **Environment variables:** all scripts load `.env` via `dotenv` at startup. See `.env.example` for required vars (`CHECKPOINT_BASE_PATH`, `EVALUATION_BASE_PATH`, `HEAD_BASE_PATH`, `TORCH_NUM_WORKERS`, HF cache dirs, etc.)
- **Launch from repo root:** Hydra scripts resolve config search paths from `${oc.env:PWD}`, so always run from the project root — never from inside `code/` or `config/`.
- **Run command:** `uv run --active python <script.py> <overrides...>`

---

## Directory structure

```
qat-transfer/
  code/
    src/                  # Shared library code (models, data loaders, quantization)
    experiments/          # Evaluation & transfer scripts (000_baselines, 001_qat_transfer, ...)
    test/                 # Smoke tests for dataloading and modeling
    visualizations/       # Argparse plotting scripts (heatmaps)
  config/                 # Hydra YAML configs — mirrors code/ 1:1
  evaluations/            # eval_results.json outputs
  storage/                # Checkpoints and heads
  plots/                  # Generated visualization HTML files
  logs/                   # Hydra run/sweep logs
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
- **Vision:** `ilharco_hf_clip`, `ilharco_open_clip`, `ilharco_timm_supervised`
- **Text:** `ilharco_automodelforsequenceclassification`

### Numbered experiment phases

Experiments are organized in numbered directories: `000_baselines`, `001_qat_transfer`, etc. New experiments get the next sequential number.

### config/ mirrors code/ 1:1

Every Hydra script at `code/a/b/script.py` has a matching YAML at `config/a/b/script.yaml`. When adding a new Hydra script, always create both the Python script and its YAML config.

### Two script families

- **Hydra scripts** — everything under `code/src/` and `code/experiments/`. Use `@hydra.main`, `DictConfig`, OmegaConf resolvers.
- **Argparse scripts** — plotting utilities under `code/visualizations/` and `pick_best_alpha` helpers under `code/experiments/.../001_qat_transfer/`. Plain CLI with `argparse.ArgumentParser`.

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

```
{EVALUATION_BASE_PATH}/{vision,text}/{family}/{phase}/{experiment_type}/{sanitized_model}/{dataset}/optim=.../{seed}/eval_results.json
```

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

`SPLIT_SEED` is independent from the per-run `seed` (which controls shuffling, init, etc.).

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

---

## Naming conventions

### Files

- Finetuning scripts: `finetune_fp.py`, `finetune_qat.py`
- Evaluation scripts: `evaluate_{fp,qat,pretrained}{_ptq}.py`
- Transfer scripts: `qv_transfer.py`
- Alpha selection: `pick_best_alpha.py`
- Visualization: `qv_transfer_heatmap.py`, `qv_transfer_heatmap_best_sf.py`
- Model wrappers: `modeling.py`
- Classification heads: `heads.py`

### Variables

- `snake_case` for everything
- Common abbreviations: `cfg` (DictConfig), `sd` (state dict), `bs` (batch size), `ml` (max_length), `wl` (warmup length), `ls` (label smoothing), `wd` (weight decay), `mgn` (max grad norm), `lnb` (limit num batches), `lne` (limit num epochs)
- Suffixes: `_dir` (directory path), `_path` (file path), `_loader` (DataLoader), `_dataset` (Dataset), `_bar` (tqdm bar)

### Functions

- `snake_case`
- Trailing `_` for in-place mutation: `enable_qat_()`, `apply_ptq_()`, `disable_qat_()`
- Leading `_` for private helpers: `_evaluate_ptq()`, `_fp_ckpt_dir()`, `_is_head_key()`
- Factories/builders: `get_dataset()`, `build_model_and_tokenizer()`

### Classes

- `PascalCase`: `QATLinear`, `QuantizedLinear`, `ImageClassifier`, `TaskVector`, `LabelSmoothing`

### Constants

- `UPPER_SNAKE_CASE`: `SPLIT_SEED`, `VAL_FRACTION`, `TQDM_KW`, `IS_SLURM`, `SUPPORTED_MODELS`, `REFERENCE_BATCH_SIZE`, `LOG_EVERY`

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

- Use plotly for visualizations, output as HTML to `plots/`.
- Constants at module level for eval roots and baseline method labels.

---

## Device handling

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device, dtype=torch.float32)
```

Always use `torch.device(...)` explicitly. Move to device after model construction but before creating the optimizer.
