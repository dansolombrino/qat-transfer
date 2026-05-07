"""Combined baseline bar chart for all timm supervised models (curated list).

Two subplots per row, shared x-axis (datasets) and a single top legend.
Auto-discovers models and optim frags from the evaluation directory.

Output: single PDF with LaTeX fonts (paper-ready).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
import numpy as np

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_timm_supervised/000_baselines/vision"

TEST_ACC_KEY = "test_accuracy"

DATASET_ORDER_SWAPS = [("DTD", "TinyImageNet"), ("RenderedSST2", "PCAM")]

MODEL_DISPLAY_NAMES = {
    "vit_base_patch16_224.orig_in21k": "ViT-B/16",
    "vit_large_patch16_224.orig_in21k": "ViT-L/16",
    "vit_huge_patch14_224.orig_in21k": "ViT-H/14",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B/16",
    "deit3_large_patch16_224.fb_in1k": "DeiT3-L/16",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k": "Swin-L",
}

MODEL_ORDER = [
    "vit_base_patch16_224.orig_in21k",
    "vit_large_patch16_224.orig_in21k",
    "vit_huge_patch14_224.orig_in21k",
    "deit3_base_patch16_224.fb_in1k",
    "deit3_large_patch16_224.fb_in1k",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
]


def _swapped_dataset_order(datasets_dict):
    ds = sorted(datasets_dict.keys())
    for a, b in DATASET_ORDER_SWAPS:
        if a in ds and b in ds:
            ia, ib = ds.index(a), ds.index(b)
            ds[ia], ds[ib] = ds[ib], ds[ia]
    return ds


def _unsanitize_model_name(sanitized):
    """Best-effort reverse of sanitize_timm_model_name for known models."""
    for orig in MODEL_ORDER:
        if sanitize_timm_model_name(orig) == sanitized:
            return orig
    return sanitized


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--qat-bits", required=True, type=int)
    parser.add_argument("--ptq-bits", required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules", required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


# ---------------------------------------------------------------------------
# Auto-discover models and their optim frags
# ---------------------------------------------------------------------------
def discover_models(seed):
    """Return [(model_name, model_dir, optim_frag), ...] sorted by MODEL_ORDER."""
    fp_root = os.path.join(EVAL_ROOT_BASELINES, "fp")
    if not os.path.isdir(fp_root):
        print(f"[ERROR] fp root not found: {fp_root}", file=sys.stderr)
        return []

    found = []
    for model_dir in sorted(os.listdir(fp_root)):
        model_path = os.path.join(fp_root, model_dir)
        if not os.path.isdir(model_path):
            continue
        first_ds = next((d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))), None)
        if first_ds is None:
            continue
        ds_path = os.path.join(model_path, first_ds)
        optim_frag = next((o for o in os.listdir(ds_path) if o.startswith("optim=")), None)
        if optim_frag is None:
            continue
        seed_dir = os.path.join(ds_path, optim_frag, f"seed={seed}")
        if not os.path.isdir(seed_dir):
            continue
        model_name = _unsanitize_model_name(model_dir)
        if model_name not in MODEL_DISPLAY_NAMES:
            continue
        found.append((model_name, model_dir, optim_frag))

    order_map = {name: i for i, name in enumerate(MODEL_ORDER)}
    found.sort(key=lambda t: order_map.get(t[0], 999))
    return found


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def _load_value(path, key):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get(key)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(model_dir, optim_frag, seed, qat_frag, ptq_frag):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)
    data = {}
    for dataset in datasets:
        ft_acc = _load_value(os.path.join(
            EVAL_ROOT_BASELINES, "fp", model_dir, dataset,
            optim_frag, f"seed={seed}", "eval_results.json"), TEST_ACC_KEY)
        qat_acc = _load_value(os.path.join(
            EVAL_ROOT_BASELINES, "qat", model_dir, dataset,
            optim_frag, qat_frag, f"seed={seed}", "eval_results.json"), TEST_ACC_KEY)
        ptq_acc = _load_value(os.path.join(
            EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
            optim_frag, ptq_frag, f"seed={seed}", "eval_results.json"), TEST_ACC_KEY)
        data[dataset] = {"ft": ft_acc, "qat": qat_acc, "ptq": ptq_acc}
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_all(all_model_data, args):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)
    n_ds = len(datasets)
    n_models = len(all_model_data)

    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)
    bar_width = 0.25
    x_pos = np.arange(n_ds)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 3.2 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (model_name, data) in enumerate(all_model_data):
        ax = axes_flat[idx]
        ft_vals = [data[ds]["ft"] if data[ds]["ft"] is not None else 0 for ds in datasets]
        qat_vals = [data[ds]["qat"] if data[ds]["qat"] is not None else 0 for ds in datasets]
        ptq_vals = [data[ds]["ptq"] if data[ds]["ptq"] is not None else 0 for ds in datasets]

        ax.bar(x_pos - bar_width, ft_vals, bar_width,
               label=r"\textbf{FT}" if idx == 0 else None, color="#e09f3e")
        ax.bar(x_pos, qat_vals, bar_width,
               label=r"\textbf{QAT}" if idx == 0 else None, color="#540b0e")
        ax.bar(x_pos + bar_width, ptq_vals, bar_width,
               label=r"\textbf{PTQ}" if idx == 0 else None, color="#9e2a2b")

        ax.set_ylim(0, 1.0)
        ax.set_ylabel(r"Test Accuracy", fontsize=11)
        display_title = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.set_title(r"\textbf{" + display_title + "}", fontsize=14)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xticks(x_pos)
            ax.set_xticklabels(datasets, fontsize=10, rotation=45, ha="right")
    if n_models % n_cols != 0:
        axes[n_rows - 2][n_cols - 1].set_xticks(x_pos)
        axes[n_rows - 2][n_cols - 1].set_xticklabels(datasets, fontsize=10, rotation=45, ha="right")
        axes[n_rows - 2][n_cols - 1].tick_params(labelbottom=True)

    axes_flat[0].legend(fontsize=12, frameon=False, loc="upper center",
                        bbox_to_anchor=(1.0, 1.35), ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.4 / (3.2 * n_rows))

    out_dir = os.path.join(
        "plots", "vision", "ilharco_timm_supervised", "999_paper_stuff",
        "000_baselines", "baseline_bar_all_models", f"seed={args.seed}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_bar_all_models.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    qf = _qat_frag(args)
    pf = _ptq_frag(args)

    models = discover_models(args.seed)
    if not models:
        print("[ERROR] No models found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} models:")
    all_model_data = []
    for model_name, model_dir, optim_frag in models:
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"  Loading {display} ...")
        data = load_data(model_dir, optim_frag, args.seed, qf, pf)
        all_model_data.append((model_name, data))

    plot_all(all_model_data, args)


if __name__ == "__main__":
    main()
