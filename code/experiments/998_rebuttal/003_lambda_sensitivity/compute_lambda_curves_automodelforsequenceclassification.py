"""998 — Lambda sensitivity curves (text, AutoModelForSequenceClassification)

Text counterpart of compute_lambda_curves_timm_supervised.py; see that script's
docstring for what the curve statistics mean and why lambda_star is kept
distinct from the validation-selected alpha_best.

This family carries the models the reviewers singled out: BERT-large has the
worst mean unit-scale transfer in the submission (-29.1%), and reviewer 3HFP
asks whether that is overshoot along a well-aligned direction or a genuinely
anti-aligned one.  The safe interval and the unit retention computed here bear
directly on that question, within the limit that the grid is positive-only.

Note that skip_modules is per-model in this family (classifier for BERT, score
for the embedding models), so the qat/ptq path fragments are rebuilt per model.

Writes a per-family JSON consumed by aggregate_lambda_curves.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_hf_model_name

from lambda_curves_common import (
    GRID,
    curve_key,
    curve_stats,
    summarize_cells,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_automodelforsequenceclassification"
MODALITY = "text"

EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"
EVAL_ROOT_QV        = "evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

EVAL_ROOT_OUT = "evaluations/998_rebuttal/003_lambda_sensitivity"

TEST_METRIC_KEY = "test_accuracy_fp_head_ptq"
TEST_ACC_KEY    = "test_accuracy"

# Validation-selected best scaling, written by pick_best_alpha.py.  Kept for
# reference only: the curve statistics use the test-side grid argmax.
BEST_ALPHA_FILE = "best_alpha_fp_head_ptq.json"
BEST_ALPHA_KEY  = "val_accuracy_fp_head_ptq"

MODEL_DISPLAY_NAMES = {
    "google-bert/bert-base-uncased":   "BERT-Base",
    "google-bert/bert-large-uncased":  "BERT-Large",
    "google/embeddinggemma-300m":      "EmbeddingGemma",
    "Qwen/Qwen3-Embedding-0.6B":       "Qwen3-Embedding",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",     required=True, nargs="+",
                        help="HF model names, e.g. google-bert/bert-base-uncased")
    parser.add_argument("--seed",            required=True, type=int)

    parser.add_argument("--optim",           required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",              required=True, type=float)
    parser.add_argument("--wd",              required=True, type=float)
    parser.add_argument("--ls",              required=True, type=float)
    parser.add_argument("--max-grad-norm",   required=True, type=float)
    parser.add_argument("--max-length",      required=True, type=int)
    parser.add_argument("--batch-sizes",     required=True, type=int, nargs="+",
                        help="Batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",    required=True, nargs="+",
                        help="One skip-module name per model, parallel to --model-names "
                             "(e.g. classifier classifier score score).")

    parser.add_argument("--universal-donor", default=None,
                        help="Donor to report separately as the data-free default. "
                             "Omit to skip.")

    args = parser.parse_args()
    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")
    if len(args.model_names) != len(args.skip_modules):
        parser.error("--skip-modules must have the same length as --model-names")
    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_mgn={args.max_grad_norm}_bs={batch_size}_ml={args.max_length}")


def _qat_frag(args, skip_module):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={skip_module}"


def _ptq_frag(args, skip_module):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={skip_module}"


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------
def _fp_ptq_path(model_dir, dataset, seed, optim_frag, ptq_frag):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _qv_cell_prefix(model_dir, donor, receiver, seed, optim_frag, qat_frag, ptq_frag):
    """Everything above the qv=alpha=... level for one donor-receiver cell."""
    return os.path.join(
        EVAL_ROOT_QV, model_dir,
        f"src={donor}_seed={seed}",
        f"tgt={receiver}_seed={seed}",
        optim_frag, qat_frag, ptq_frag,
    )


def _qv_eval_path(cell_prefix, alpha, split):
    return os.path.join(cell_prefix, f"qv=alpha={alpha}", f"split={split}",
                        "eval_results.json")


def _load_best_alpha(cell_prefix):
    """Validation-selected scaling for one cell, or None if not picked yet."""
    path = os.path.join(cell_prefix, BEST_ALPHA_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            info = json.load(f).get(BEST_ALPHA_KEY)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None
    return info["alpha"] if info is not None else None


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def _load_value(path, key):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get(key)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pairs(model_dir, datasets, args, optim_frag, qat_frag, ptq_frag):
    """Delta(lambda) over the shared grid for every (donor, receiver) cell."""
    pairs = []
    missing = []

    for receiver in datasets:
        baseline_path = _fp_ptq_path(model_dir, receiver, args.seed, optim_frag, ptq_frag)
        baseline = _load_value(baseline_path, TEST_ACC_KEY)
        if baseline is None:
            missing.append(baseline_path)
            continue

        for donor in datasets:
            cell_prefix = _qv_cell_prefix(model_dir, donor, receiver, args.seed,
                                          optim_frag, qat_frag, ptq_frag)

            curve = {}
            for alpha in GRID:
                path = _qv_eval_path(cell_prefix, alpha, "test")
                acc = _load_value(path, TEST_METRIC_KEY)
                if acc is None:
                    missing.append(path)
                    continue
                curve[curve_key(alpha)] = acc - baseline

            if not curve:
                continue

            pairs.append({
                "donor":        donor,
                "receiver":     receiver,
                "baseline_acc": baseline,
                "same_task":    donor == receiver,
                "curve":        curve,
                "stats":        curve_stats(curve),
                # Validation-selected scaling, for reference against 001 only.
                "alpha_best_val": _load_best_alpha(cell_prefix),
            })

    return pairs, missing


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
def _out_dir(args):
    return os.path.join(
        EVAL_ROOT_OUT,
        f"seed={args.seed}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        "split=test",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys())

    models = {}
    for model_name, batch_size, skip_module in zip(
        args.model_names, args.batch_sizes, args.skip_modules
    ):
        model_dir  = sanitize_hf_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)
        qat_frag   = _qat_frag(args, skip_module)
        ptq_frag   = _ptq_frag(args, skip_module)

        print(f"Loading {model_name} ...")
        pairs, missing = load_pairs(model_dir, datasets, args,
                                    optim_frag, qat_frag, ptq_frag)

        cross = [p for p in pairs if not p["same_task"]]
        same  = [p for p in pairs if p["same_task"]]

        entry = {
            "display_name":     MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "model_dir":        model_dir,
            "batch_size":       batch_size,
            "skip_modules":     [skip_module],
            "n_datasets":       len(datasets),
            "n_cells_expected": len(datasets) ** 2,
            "pairs":            pairs,
            "cross_task":       summarize_cells(cross),
            "same_task":        summarize_cells(same),
            "universal_donor":  summarize_cells(
                [p for p in cross if p["donor"] == args.universal_donor]
            ) if args.universal_donor else None,
            "missing":          missing,
        }
        models[model_name] = entry

        summary = entry["cross_task"]
        if summary["n"] == 0:
            print(f"  {entry['display_name']}: no cross-task pairs found")
        else:
            unit = summary["curve"][curve_key(1.0)]
            print(f"  {entry['display_name']}: {summary['n']} cross-task pairs, "
                  f"win_rate={unit['win_rate'] * 100:.1f}% at lambda=1, "
                  f"median safe width={summary['safe_width']['median']:.2f}, "
                  f"median plateau width={summary['plateau_width']['median']:.2f}, "
                  f"median unit retention={summary['unit_retention']['median']:.2f}")
        if missing:
            print(f"  {entry['display_name']}: {len(missing)} missing files",
                  file=sys.stderr)

    results = {
        "family":   FAMILY,
        "modality": MODALITY,
        "config": {
            "seed":            args.seed,
            "optim":           args.optim,
            "lr":              args.lr,
            "wd":              args.wd,
            "ls":              args.ls,
            "max_grad_norm":   args.max_grad_norm,
            "max_length":      args.max_length,
            "qat_bits":        args.qat_bits,
            "ptq_bits":        args.ptq_bits,
            "granularity":     args.granularity,
            "skip_modules":    list(args.skip_modules),
            "grid":            GRID,
            "eval_split":      "test",
            "metric_key":      TEST_METRIC_KEY,
            "baseline":        "fp_ptq",
            "universal_donor": args.universal_donor,
        },
        "datasets": datasets,
        "models":   models,
    }

    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lambda_curves_{FAMILY}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
