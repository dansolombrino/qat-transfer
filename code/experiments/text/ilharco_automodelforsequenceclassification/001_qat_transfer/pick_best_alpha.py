"""Pick the best QV scaling factor (alpha) per (source, target) pair.

Reads val-split eval_results.json files produced by qv_transfer.py,
finds the alpha that maximises val_accuracy_fp_head_ptq and
val_accuracy_qat_head_ptq for each (source_dataset, target_dataset)
combination, and outputs the result as a table, JSON, or ready-to-run
hydra+submitit commands.

Usage
-----
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py \
    --model-name google-bert/bert-base-uncased --seed 1 \
    --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 \
    --bits 8 --granularity channel --skip-modules classifier \
    --output table
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from dotenv import load_dotenv
load_dotenv()

from src.vision.utils import sanitize_hf_model_name


METRIC_KEYS = {
    "fp_head_ptq":  "val_accuracy_fp_head_ptq",
    "qat_head_ptq": "val_accuracy_qat_head_ptq",
}

EVAL_ROOT_QV = os.path.join(
    os.environ["EVALUATION_BASE_PATH"],
    "text",
    "ilharco_automodelforsequenceclassification",
    "001_qat_transfer",
    "text",
    "qv_transfer",
)

SCRIPT_PATH = "code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name",    required=True)
    p.add_argument("--seed",          required=True, type=int)

    p.add_argument("--lr",            required=True, type=float)
    p.add_argument("--wd",            required=True, type=float)
    p.add_argument("--ls",            required=True, type=float)
    p.add_argument("--max-grad-norm", required=True, type=float)
    p.add_argument("--batch-size",    required=True, type=int)
    p.add_argument("--max-length",    required=True, type=int)

    p.add_argument("--bits",          required=True, type=int)
    p.add_argument("--granularity",   required=True, choices=["tensor", "channel"])
    p.add_argument("--skip-modules",  required=True, nargs="+")

    p.add_argument("--output",        default="table",
                   choices=["table", "json", "commands"],
                   help="Output format (default: table)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers (mirror qv_transfer.py eval_dir layout)
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if skip_modules else "none"


def _optim_frag(lr, wd, ls, mgn, bs, ml):
    return f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_mgn={mgn}_bs={bs}_ml={ml}"


def _qat_frag(bits, gran, skip_modules):
    return f"qat=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _ptq_frag(bits, gran, skip_modules):
    return f"ptq=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def find_best_alphas(args):
    model_dir  = sanitize_hf_model_name(args.model_name)
    optim      = _optim_frag(args.lr, args.wd, args.ls, args.max_grad_norm, args.batch_size, args.max_length)
    qat        = _qat_frag(args.bits, args.granularity, args.skip_modules)
    ptq        = _ptq_frag(args.bits, args.granularity, args.skip_modules)

    # Glob: src=*_seed=*/tgt=*_seed=*/<optim>/<qat>/<ptq>/qv=alpha=*/split=val/eval_results.json
    pattern = os.path.join(
        EVAL_ROOT_QV, model_dir,
        f"src=*_seed={args.seed}",
        f"tgt=*_seed={args.seed}",
        optim, qat, ptq,
        "qv=alpha=*",
        "split=val",
        "eval_results.json",
    )

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No val results found for pattern:\n  {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} val result file(s).", file=sys.stderr)

    # best[metric_label][src_dataset][tgt_dataset] = {"alpha": float, "acc": float}
    best = {label: {} for label in METRIC_KEYS}

    src_re = re.compile(r"^src=(.+?)_seed=\d+$")
    tgt_re = re.compile(r"^tgt=(.+?)_seed=\d+$")
    alpha_re = re.compile(r"^qv=alpha=(.+)$")

    for fpath in files:
        parts = fpath.split(os.sep)

        src_dataset = tgt_dataset = alpha_val = None
        for part in parts:
            m = src_re.match(part)
            if m:
                src_dataset = m.group(1)
                continue
            m = tgt_re.match(part)
            if m:
                tgt_dataset = m.group(1)
                continue
            m = alpha_re.match(part)
            if m:
                try:
                    alpha_val = float(m.group(1))
                except ValueError:
                    pass

        if src_dataset is None or tgt_dataset is None or alpha_val is None:
            print(f"  [SKIP] could not parse: {fpath}", file=sys.stderr)
            continue

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [READ ERROR] {fpath}: {e}", file=sys.stderr)
            continue

        for label, metric_key in METRIC_KEYS.items():
            acc = data.get(metric_key)
            if acc is None:
                print(f"  [MISSING KEY] {metric_key} in {fpath}", file=sys.stderr)
                continue

            if src_dataset not in best[label]:
                best[label][src_dataset] = {}

            prev = best[label][src_dataset].get(tgt_dataset)
            if prev is None or acc > prev["acc"]:
                best[label][src_dataset][tgt_dataset] = {"alpha": alpha_val, "acc": acc}

    return best


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def output_table(best):
    for label, best_for_metric in best.items():
        if not best_for_metric:
            continue

        metric_key = METRIC_KEYS[label]
        print(f"\n=== {label} ({metric_key}) ===\n")

        src_datasets = sorted(best_for_metric.keys(), key=str.lower)

        tgt_set = set()
        for inner in best_for_metric.values():
            tgt_set.update(inner.keys())
        tgt_datasets = sorted(tgt_set, key=str.lower)

        # Header
        src_w = max(len("source"), max((len(s) for s in src_datasets), default=6))
        tgt_w = max(len("target"), max((len(t) for t in tgt_datasets), default=6))
        print(f"{'source':<{src_w}}  {'target':<{tgt_w}}  {'best_alpha':>10}  {'val_acc_ptq':>11}")
        print(f"{'-'*src_w}  {'-'*tgt_w}  {'-'*10}  {'-'*11}")

        for src in src_datasets:
            for tgt in tgt_datasets:
                entry = best_for_metric[src].get(tgt)
                if entry is None:
                    print(f"{src:<{src_w}}  {tgt:<{tgt_w}}  {'N/A':>10}  {'N/A':>11}")
                else:
                    print(f"{src:<{src_w}}  {tgt:<{tgt_w}}  {entry['alpha']:>10.4f}  {entry['acc']:>11.4f}")


def output_json(best):
    print(json.dumps(best, indent=2, sort_keys=True))


def output_commands(best, args):
    skip_list = ",".join(sorted(args.skip_modules))

    # Collect all (label, src, tgt, entry) tuples for global numbering.
    commands = []
    for label, best_for_metric in best.items():
        if not best_for_metric:
            continue
        src_datasets = sorted(best_for_metric.keys(), key=str.lower)
        for src in src_datasets:
            for tgt in sorted(best_for_metric[src].keys(), key=str.lower):
                commands.append((label, src, tgt, best_for_metric[src][tgt]))

    total = len(commands)

    for i, (label, src, tgt, entry) in enumerate(commands):
        alpha = entry["alpha"]
        cmd = (
            f"uv run --active python {SCRIPT_PATH} -m"
            f" hydra/launcher=submitit_slurm"
            f" model_name={args.model_name}"
            f" batch_size={args.batch_size}"
            f" max_length={args.max_length}"
            f" lr={args.lr}"
            f" wd={args.wd}"
            f" ls={args.ls}"
            f" max_grad_norm={args.max_grad_norm}"
            f" 'source.dataset_names=[{src}]'"
            f" source.seed={args.seed}"
            f" 'target.dataset_names=[{tgt}]'"
            f" target.seed={args.seed}"
            f" qat.bits={args.bits}"
            f" qat.granularity={args.granularity}"
            f" 'qat.skip_modules=[{skip_list}]'"
            f" qv.alpha={alpha}"
            f" ptq.bits={args.bits}"
            f" ptq.granularity={args.granularity}"
            f" 'ptq.skip_modules=[{skip_list}]'"
            f" eval_split=test"
        )
        print(f"\necho '{i} / {total} — {label} src={src} tgt={tgt}'\n")
        print(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    best = find_best_alphas(args)

    has_any = any(best_for_metric for best_for_metric in best.values())
    if not has_any:
        print("No best-alpha results found.", file=sys.stderr)
        sys.exit(1)

    if args.output == "table":
        output_table(best)
    elif args.output == "json":
        output_json(best)
    elif args.output == "commands":
        output_commands(best, args)


if __name__ == "__main__":
    main()
