"""998 — Lambda sensitivity curves (OpenCLIP)

OpenCLIP counterpart of compute_lambda_curves_timm_supervised.py; see that
script's docstring for what the curve statistics mean and why lambda_star is
kept distinct from the validation-selected alpha_best.

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

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_open_clip_model_name

from lambda_curves_common import (
    GRID,
    curve_key,
    curve_stats,
    summarize_cells,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_open_clip"
MODALITY = "vision"

EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_open_clip/000_baselines/vision"
EVAL_ROOT_QV        = "evaluations/vision/ilharco_open_clip/001_qat_transfer/vision/qv_transfer"

EVAL_ROOT_OUT = "evaluations/998_rebuttal/003_lambda_sensitivity"

TEST_METRIC_KEY = "test_accuracy_patched_qat_ptq"
TEST_ACC_KEY    = "test_accuracy"

# Validation-selected best scaling, written by pick_best_alpha.py.  This family
# writes a single file rather than one per head variant.  Kept for reference
# only: the curve statistics use the test-side grid argmax.
BEST_ALPHA_FILE = "best_alpha.json"
BEST_ALPHA_KEY  = "val_accuracy_patched_qat_ptq"

MODEL_DISPLAY_NAMES = {
    ("ViT-B-16", "laion2b_s34b_b88k"): "ViT-B/16 (LAION)",
    ("ViT-B-16", "openai"):            "ViT-B/16 (OpenAI)",
    ("ViT-L-14", "laion2b_s32b_b82k"): "ViT-L/14 (LAION)",
    ("ViT-H-14", "laion2b_s32b_b79k"): "ViT-H/14 (LAION)",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",     required=True, nargs="+",
                        help="open_clip model names, e.g. ViT-B-16 ViT-L-14")
    parser.add_argument("--pretrained-tags", required=True, nargs="+",
                        help="open_clip pretrained tags (parallel to --model-names)")
    parser.add_argument("--seed",            required=True, type=int)

    parser.add_argument("--optim",           required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",              required=True, type=float)
    parser.add_argument("--wd",              required=True, type=float)
    parser.add_argument("--ls",              required=True, type=float)
    parser.add_argument("--wl",              required=True, type=int)
    parser.add_argument("--max-grad-norm",   required=True, type=float)
    parser.add_argument("--batch-sizes",     required=True, type=int, nargs="+",
                        help="Batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",    required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    parser.add_argument("--universal-donor", default=None,
                        help="Donor to report separately as the data-free default "
                             "(e.g. ImageNet). Omit to skip.")

    args = parser.parse_args()
    if len(args.model_names) != len(args.pretrained_tags):
        parser.error("--model-names and --pretrained-tags must have the same length")
    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")
    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


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
    qat_frag = _qat_frag(args)
    ptq_frag = _ptq_frag(args)

    models = {}
    for model_name, pretrained_tag, batch_size in zip(
        args.model_names, args.pretrained_tags, args.batch_sizes
    ):
        model_dir  = sanitize_open_clip_model_name(model_name, pretrained_tag)
        optim_frag = _optim_frag(args, batch_size)
        model_key  = f"{model_name}__{pretrained_tag}"

        print(f"Loading {model_key} ...")
        pairs, missing = load_pairs(model_dir, datasets, args,
                                    optim_frag, qat_frag, ptq_frag)

        cross = [p for p in pairs if not p["same_task"]]
        same  = [p for p in pairs if p["same_task"]]

        entry = {
            "display_name":     MODEL_DISPLAY_NAMES.get((model_name, pretrained_tag), model_key),
            "model_dir":        model_dir,
            "pretrained_tag":   pretrained_tag,
            "batch_size":       batch_size,
            "skip_modules":     list(args.skip_modules),
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
        models[model_key] = entry

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
            "wl":              args.wl,
            "max_grad_norm":   args.max_grad_norm,
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
