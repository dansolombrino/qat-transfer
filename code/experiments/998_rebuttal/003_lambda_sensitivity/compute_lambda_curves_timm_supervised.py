"""998 — Lambda sensitivity curves (timm supervised)

Reviewer ZEFq asks how sensitive QV patching is to the choice of the scaling
lambda.  The zero-shot reframing analysis in 001 reports only two scalars per
donor-receiver pair, Delta at lambda=1 and Delta at the validation-selected
lambda_best, which says nothing about the shape of the curve between them.
This script reconstructs the full Delta(lambda) curve on the grid shared by all
families and derives, per cell, how wide the winning region is, how flat the
optimum is, and what the data-free default retains.

Delta(lambda) = transfer accuracy at lambda - vanilla PTQ accuracy, so a "win"
is Delta > 0, exactly as in 001.  Pairs are split into:

    same-task  : donor == receiver.  At lambda=1 this is algebraically the
                 receiver's own QAT checkpoint, so it is the QAT ceiling
                 rather than a transfer result.  Reported separately.
    cross-task : donor != receiver.  The genuine transfer setting, and the
                 basis of every headline number.

Note that lambda_star here is the grid argmax on the *test* split, an oracle
quantity describing the shape of the curve.  It is deliberately distinct from
alpha_best, which 001 selects on receiver validation data and which is the only
one usable as a protocol.  Both are stored; conflating them would repeat the
overclaim the reviewers flagged.

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
from src.vision.utils import sanitize_timm_model_name

from lambda_curves_common import (
    GRID,
    curve_key,
    curve_stats,
    summarize_cells,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_timm_supervised"
MODALITY = "vision"

EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_timm_supervised/000_baselines/vision"
EVAL_ROOT_QV        = "evaluations/vision/ilharco_timm_supervised/001_qat_transfer/vision/qv_transfer"

EVAL_ROOT_OUT = "evaluations/998_rebuttal/003_lambda_sensitivity"

TEST_METRIC_KEY = "test_accuracy_fp_head_ptq"
TEST_ACC_KEY    = "test_accuracy"

# Validation-selected best scaling, written by pick_best_alpha.py.  Kept for
# reference only: the curve statistics use the test-side grid argmax.
BEST_ALPHA_FILE = "best_alpha_fp_head_ptq.json"
BEST_ALPHA_KEY  = "val_accuracy_fp_head_ptq"

MODEL_DISPLAY_NAMES = {
    "vit_base_patch16_224.orig_in21k": "ViT-B/16",
    "vit_large_patch16_224.orig_in21k": "ViT-L/16",
    "vit_huge_patch14_224.orig_in21k": "ViT-H/14",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16 (AugReg2)",
    "vit_base_patch16_clip_224.openai_ft_in12k_in1k": "ViT-B/16 (CLIP)",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B/16",
    "deit3_base_patch16_224.fb_in22k_ft_in1k": "DeiT3-B/16 (IN22k)",
    "deit3_large_patch16_224.fb_in1k": "DeiT3-L/16",
    "deit3_large_patch16_224.fb_in22k_ft_in1k": "DeiT3-L/16 (IN22k)",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k": "Swin-L",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",     required=True, nargs="+",
                        help="timm model names, e.g. vit_base_patch16_224.orig_in21k")
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
    for model_name, batch_size in zip(args.model_names, args.batch_sizes):
        model_dir  = sanitize_timm_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)

        print(f"Loading {model_name} ...")
        pairs, missing = load_pairs(model_dir, datasets, args,
                                    optim_frag, qat_frag, ptq_frag)

        cross = [p for p in pairs if not p["same_task"]]
        same  = [p for p in pairs if p["same_task"]]

        entry = {
            "display_name":     MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "model_dir":        model_dir,
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
