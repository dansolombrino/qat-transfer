"""Combined baseline bar chart for all text (automodel) models.

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

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_hf_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"

TEST_ACC_KEY = "test_accuracy"

DATASET_ORDER_SWAPS = [("AmazonCounterfactual", "ToxicConversations")]

LABEL_RENAMES = {
    "AmazonCounterfactual": "Counterfactual",
    "TweetSentimentExtraction": "Sentiment",
    "AmazonReviewsClassification": "Reviews",
    "ToxicConversations": "Toxic",
    "MTOPDomain": "MTOP D",
    "MTOPIntent": "MTOP I",
    "MassiveIntent": "Intent",
    "MassiveScenario": "Scenario",
}

# sanitized_dir -> display name
MODEL_DISPLAY_NAMES = {
    "google_bert_bert_base_uncased": "BERT-Base",
    "google_bert_bert_large_uncased": "BERT-Large",
    "google_embeddinggemma_300m": "EmbeddingGemma",
    "Qwen_Qwen3_Embedding_0.6B": "Qwen3-Embedding",
}

MODEL_ORDER = [
    "google_bert_bert_base_uncased",
    "google_bert_bert_large_uncased",
    "google_embeddinggemma_300m",
    "Qwen_Qwen3_Embedding_0.6B",
]


def _swapped_dataset_order(datasets_dict):
    ds = sorted(datasets_dict.keys())
    for a, b in DATASET_ORDER_SWAPS:
        if a in ds and b in ds:
            ia, ib = ds.index(a), ds.index(b)
            ds[ia], ds[ib] = ds[ib], ds[ia]
    return ds


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--qat-bits", required=True, type=int)
    parser.add_argument("--ptq-bits", required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _discover_qat_frag(model_dir, first_ds, optim_frag, bits, granularity):
    """Find the qat=bits=..._gran=..._skip=... directory for this model."""
    prefix = f"qat=bits={bits}_gran={granularity}_skip="
    qat_root = os.path.join(EVAL_ROOT_BASELINES, "qat", model_dir, first_ds, optim_frag)
    if not os.path.isdir(qat_root):
        return None
    for entry in os.listdir(qat_root):
        if entry.startswith(prefix):
            return entry
    return None


def _qat_to_ptq_frag(qat_frag, ptq_bits):
    """Convert 'qat=bits=3_gran=channel_skip=X' to 'ptq=bits=3_gran=channel_skip=X'."""
    return qat_frag.replace("qat=bits=", "ptq=bits=", 1).replace(
        f"bits={qat_frag.split('bits=')[1].split('_')[0]}",
        f"bits={ptq_bits}", 1,
    )


# ---------------------------------------------------------------------------
# Auto-discover models and their optim frags
# ---------------------------------------------------------------------------
def discover_models(seed, qat_bits, ptq_bits, granularity):
    """Return [(model_dir, optim_frag, qat_frag, ptq_frag), ...] sorted by MODEL_ORDER."""
    fp_root = os.path.join(EVAL_ROOT_BASELINES, "fp")
    if not os.path.isdir(fp_root):
        print(f"[ERROR] fp root not found: {fp_root}", file=sys.stderr)
        return []

    found = []
    for model_dir in sorted(os.listdir(fp_root)):
        if model_dir not in MODEL_DISPLAY_NAMES:
            continue
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
        qat_frag = _discover_qat_frag(model_dir, first_ds, optim_frag, qat_bits, granularity)
        if qat_frag is None:
            print(f"  [WARN] No QAT frag for {model_dir}, skipping", file=sys.stderr)
            continue
        skip_tag = qat_frag.split("_skip=")[1]
        ptq_frag = f"ptq=bits={ptq_bits}_gran={granularity}_skip={skip_tag}"
        found.append((model_dir, optim_frag, qat_frag, ptq_frag))

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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.2 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    display_labels = [LABEL_RENAMES.get(ds, ds) for ds in datasets]

    for idx, (model_dir, data) in enumerate(all_model_data):
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
        display_title = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)
        ax.set_title(r"\textbf{" + display_title + "}", fontsize=14)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, fontsize=10, rotation=45, ha="right")
    if n_models % n_cols != 0:
        axes[n_rows - 2][n_cols - 1].set_xticks(x_pos)
        axes[n_rows - 2][n_cols - 1].set_xticklabels(display_labels, fontsize=10, rotation=45, ha="right")
        axes[n_rows - 2][n_cols - 1].tick_params(labelbottom=True)

    axes_flat[0].legend(fontsize=12, frameon=False, loc="upper center",
                        bbox_to_anchor=(1.0, 1.35), ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.4 / (3.2 * n_rows))

    out_dir = os.path.join(
        "plots", "text", "ilharco_automodelforsequenceclassification",
        "999_paper_stuff", "000_baselines", "baseline_bar_all_models",
        f"seed={args.seed}",
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

    models = discover_models(args.seed, args.qat_bits, args.ptq_bits, args.granularity)
    if not models:
        print("[ERROR] No models found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} models:")
    all_model_data = []
    for model_dir, optim_frag, qat_frag, ptq_frag in models:
        display = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)
        print(f"  Loading {display} (skip: {qat_frag.split('_skip=')[1]}) ...")
        data = load_data(model_dir, optim_frag, args.seed, qat_frag, ptq_frag)
        all_model_data.append((model_dir, data))

    plot_all(all_model_data, args)


if __name__ == "__main__":
    main()
