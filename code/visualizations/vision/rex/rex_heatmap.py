"""REx heatmaps (QV-transfer style layout).

Builds two-panel heatmaps from REx evaluation JSON files:

1) Raw accuracy:
   - left panel: REx accuracy for each (bits, sparsity) setting
   - right panel: PTQ baselines per bit + random baseline

2) Delta accuracy:
   - left panel: REx delta versus PTQ baseline
   - right panel: same baseline panel as above

Rows are target datasets. Columns are REx settings (left) and baselines (right).
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Make `from src.vision...` imports work when run from repo root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.vision.utils import sanitize_hf_model_name, sanitize_timm_model_name


HEATMAP_COLORSCALE_SEQUENTIAL = "Viridis"
HEATMAP_COLORSCALE_DIVERGING = "RdYlGn"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=["ilharco_hf_clip", "ilharco_timm_supervised"],
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", required=True, type=int)

    # Run path fragments / filters (mirrors rex eval script metadata).
    parser.add_argument("--optim", required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--wd", required=True, type=float)
    parser.add_argument("--ls", required=True, type=float)
    parser.add_argument("--wl", required=True, type=int)
    parser.add_argument("--max-grad-norm", required=True, type=float)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--order", required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules", required=True, nargs="+")

    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset/order of bits to show. Default: infer from JSON.",
    )
    parser.add_argument(
        "--sparsity",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset/order of sparsity values to show. Default: infer from JSON.",
    )
    parser.add_argument(
        "--delta-mode",
        default="same_bits",
        choices=["same_bits", "equal_budget"],
        help=(
            "same_bits: delta = REx(bits,sparsity) - PTQ(bits). "
            "equal_budget: use precomputed delta_vs_ptq_equal_budget from JSON."
        ),
    )
    parser.add_argument(
        "--evaluation-root",
        default="quantization/qat-transfer/evaluations/vision/rex",
        help="Root directory containing REx eval_results.json files.",
    )
    return parser.parse_args()


def _model_dir(model_family: str, model_name: str) -> str:
    if model_family == "ilharco_hf_clip":
        return sanitize_hf_model_name(model_name)
    if model_family == "ilharco_timm_supervised":
        return sanitize_timm_model_name(model_name)
    raise ValueError(f"Unsupported model_family: {model_family}")


def _optim_frag(args) -> str:
    return (
        f"optim={args.optim}_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
        f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
    )


def _skip_tag(skip_modules) -> str:
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _float_equal(a, b, eps=1e-12):
    return abs(float(a) - float(b)) <= eps


def _matches_payload(payload, args):
    if payload.get("model_family") != args.model_family:
        return False
    if payload.get("model_name") != args.model_name:
        return False
    if int(payload.get("seed")) != int(args.seed):
        return False
    if int(payload.get("order")) != int(args.order):
        return False
    if payload.get("granularity") != args.granularity:
        return False
    if sorted(payload.get("skip_modules", [])) != sorted(args.skip_modules):
        return False

    # Optim settings
    payload_optim = payload.get("optim")
    if payload_optim is not None and payload_optim != args.optim:
        return False
    if not _float_equal(payload.get("lr"), args.lr):
        return False
    if not _float_equal(payload.get("wd"), args.wd):
        return False
    if not _float_equal(payload.get("ls"), args.ls):
        return False
    if int(payload.get("wl")) != int(args.wl):
        return False
    if not _float_equal(payload.get("max_grad_norm"), args.max_grad_norm):
        return False
    if int(payload.get("batch_size")) != int(args.batch_size):
        return False
    return True


def _payload_rank(payload):
    # Prefer non-smoke runs over smoke runs when both exist.
    lnb = payload.get("limit_num_batches")
    lne = payload.get("limit_num_epochs")
    full_run = int(lnb is None and lne is None)
    lnb_score = 10**9 if lnb is None else int(lnb)
    lne_score = 10**9 if lne is None else int(lne)
    result_len = len(payload.get("results", []))
    return (full_run, lne_score, lnb_score, result_len)


def collect_dataset_payloads(args):
    model_dir = _model_dir(args.model_family, args.model_name)
    root = Path(args.evaluation_root) / args.model_family / model_dir
    pattern = str(root / "**" / "eval_results.json")
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"No eval_results.json found under: {root}")

    best_by_dataset = {}
    for path in files:
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue

        if not _matches_payload(payload, args):
            continue

        dataset_name = payload.get("dataset_name")
        if dataset_name is None:
            continue

        rank = _payload_rank(payload)
        previous = best_by_dataset.get(dataset_name)
        if previous is None or rank > previous["rank"]:
            best_by_dataset[dataset_name] = {
                "payload": payload,
                "path": path,
                "rank": rank,
            }

    if len(best_by_dataset) == 0:
        raise ValueError(
            "No REx eval files matched the requested run filters. "
            "Check model/run args and evaluation root."
        )
    return best_by_dataset, model_dir


def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _finite_values(matrix):
    return [v for row in matrix for v in row if v is not None]


def _robust_symmetric_bounds(values, center=0.0, min_span=0.02, q_low=0.05, q_high=0.95):
    if not values:
        return center - min_span, center + min_span
    svals = sorted(values)
    ql = _quantile(svals, q_low)
    qh = _quantile(svals, q_high)
    span = max(abs(center - ql), abs(qh - center), min_span)
    return center - span, center + span


def _extract_maps(payload):
    ptq_by_bits = {}
    rex_by_key = {}

    for item in payload.get("results", []):
        method = item.get("method")
        bits = int(item.get("bits"))
        if method == "ptq":
            ptq_by_bits[bits] = float(item.get("accuracy"))
            continue
        if method != "rex":
            continue
        sparsity = float(item.get("sparsity"))
        rex_by_key[(bits, sparsity)] = item
    return ptq_by_bits, rex_by_key


def build_matrices(best_by_dataset, args):
    datasets = sorted(best_by_dataset.keys(), key=str.lower)

    inferred_bits = set()
    inferred_sparsity = set()
    for entry in best_by_dataset.values():
        _, rex_map = _extract_maps(entry["payload"])
        for bits, sparsity in rex_map.keys():
            inferred_bits.add(int(bits))
            inferred_sparsity.add(float(sparsity))

    bits = args.bits if args.bits is not None else sorted(inferred_bits, reverse=True)
    sparsity = args.sparsity if args.sparsity is not None else sorted(inferred_sparsity)
    bits = [int(b) for b in bits]
    sparsity = [float(s) for s in sparsity]

    rex_keys = [(b, s) for b in bits for s in sparsity]
    rex_labels = [f"b{b}|s={s:g}" for b, s in rex_keys]

    baseline_labels = [f"PTQ b{b}" for b in bits] + ["Random"]
    raw_z, raw_text = [], []
    diff_z, diff_text = [], []
    baseline_z, baseline_text = [], []

    for dataset in datasets:
        payload = best_by_dataset[dataset]["payload"]
        ptq_map, rex_map = _extract_maps(payload)
        random_chance = payload.get("random_chance")

        row_raw_z, row_raw_text = [], []
        row_diff_z, row_diff_text = [], []

        for key in rex_keys:
            item = rex_map.get(key)
            if item is None:
                row_raw_z.append(None)
                row_raw_text.append("")
                row_diff_z.append(None)
                row_diff_text.append("")
                continue

            rex_acc = float(item.get("accuracy"))
            row_raw_z.append(rex_acc)
            row_raw_text.append(f"{rex_acc:.2f}")

            if args.delta_mode == "same_bits":
                base = ptq_map.get(key[0])
                if base is None:
                    row_diff_z.append(None)
                    row_diff_text.append("")
                else:
                    diff = rex_acc - float(base)
                    row_diff_z.append(diff)
                    row_diff_text.append(f"{diff:.2f}")
            else:
                diff = item.get("delta_vs_ptq_equal_budget")
                if diff is None:
                    row_diff_z.append(None)
                    row_diff_text.append("")
                else:
                    diff = float(diff)
                    row_diff_z.append(diff)
                    row_diff_text.append(f"{diff:.2f}")

        row_base_z, row_base_text = [], []
        for bit in bits:
            val = ptq_map.get(bit)
            if val is None:
                row_base_z.append(None)
                row_base_text.append("")
            else:
                row_base_z.append(float(val))
                row_base_text.append(f"{float(val):.2f}")

        if random_chance is None:
            row_base_z.append(None)
            row_base_text.append("")
        else:
            row_base_z.append(float(random_chance))
            row_base_text.append(f"{float(random_chance):.2f}")

        raw_z.append(row_raw_z)
        raw_text.append(row_raw_text)
        diff_z.append(row_diff_z)
        diff_text.append(row_diff_text)
        baseline_z.append(row_base_z)
        baseline_text.append(row_base_text)

    return {
        "datasets": datasets,
        "rex_labels": rex_labels,
        "baseline_labels": baseline_labels,
        "raw_z": raw_z,
        "raw_text": raw_text,
        "diff_z": diff_z,
        "diff_text": diff_text,
        "baseline_z": baseline_z,
        "baseline_text": baseline_text,
        "bits": bits,
        "sparsity": sparsity,
    }


def _common_layout(fig, datasets, rex_labels, baseline_labels):
    fig.update_layout(
        template="plotly_white",
        height=max(400, 60 * len(datasets) + 180),
        width=max(900, 75 * len(rex_labels) + 100 * len(baseline_labels) + 280),
        margin=dict(l=80, r=230, t=120, b=90),
    )
    fig.update_xaxes(
        title_text="REx settings (bits, sparsity)",
        row=1, col=1, side="bottom",
    )
    fig.update_xaxes(title_text="Baselines", row=1, col=2, side="bottom")
    fig.update_yaxes(
        title_text="Target dataset",
        row=1, col=1, autorange="reversed",
    )
    fig.update_yaxes(row=1, col=2, showticklabels=False, autorange="reversed")


def plot_raw_heatmap(matrix, args, model_dir, out_dir):
    datasets = matrix["datasets"]
    rex_labels = matrix["rex_labels"]
    baseline_labels = matrix["baseline_labels"]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[max(1, len(rex_labels)), len(baseline_labels)],
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["raw_z"],
            x=rex_labels,
            y=datasets,
            text=matrix["raw_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>rex=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["baseline_z"],
            x=baseline_labels,
            y=datasets,
            text=matrix["baseline_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis2",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>baseline=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    skip_str = ",".join(sorted(args.skip_modules))
    title = (
        "REx Accuracy Heatmap<br>"
        f"<sup>{args.model_family} | {args.model_name} | seed={args.seed} | "
        f"optim={args.optim} | order={args.order} | granularity={args.granularity} | "
        f"skip={skip_str}</sup>"
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0, cmax=1,
            colorbar=dict(title="REx Acc", x=1.01, y=0.78, len=0.42),
        ),
        coloraxis2=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0, cmax=1,
            colorbar=dict(title="Baseline Acc", x=1.01, y=0.22, len=0.42),
        ),
    )
    _common_layout(fig, datasets, rex_labels, baseline_labels)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "heatmap_rex_raw.png")
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


def plot_delta_heatmap(matrix, args, model_dir, out_dir):
    datasets = matrix["datasets"]
    rex_labels = matrix["rex_labels"]
    baseline_labels = matrix["baseline_labels"]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[max(1, len(rex_labels)), len(baseline_labels)],
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["diff_z"],
            x=rex_labels,
            y=datasets,
            text=matrix["diff_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>rex=%{x}<br>delta=%{z:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["baseline_z"],
            x=baseline_labels,
            y=datasets,
            text=matrix["baseline_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis2",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>baseline=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    diff_cmin, diff_cmax = _robust_symmetric_bounds(
        _finite_values(matrix["diff_z"]), center=0.0, min_span=0.02
    )

    if args.delta_mode == "same_bits":
        delta_title = "Acc Δ (REx − PTQ same bits)"
    else:
        delta_title = "Acc Δ (REx − PTQ equal budget)"

    skip_str = ",".join(sorted(args.skip_modules))
    title = (
        "REx Delta Heatmap<br>"
        f"<sup>{args.model_family} | {args.model_name} | seed={args.seed} | "
        f"optim={args.optim} | order={args.order} | granularity={args.granularity} | "
        f"skip={skip_str} | delta={args.delta_mode}</sup>"
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=HEATMAP_COLORSCALE_DIVERGING,
            cmin=diff_cmin,
            cmax=diff_cmax,
            cmid=0,
            colorbar=dict(title=delta_title, x=1.01, y=0.78, len=0.42),
        ),
        coloraxis2=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0, cmax=1,
            colorbar=dict(title="Baseline Acc", x=1.01, y=0.22, len=0.42),
        ),
    )
    _common_layout(fig, datasets, rex_labels, baseline_labels)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_rex_delta_{args.delta_mode}.png")
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    best_by_dataset, model_dir = collect_dataset_payloads(args)
    matrix = build_matrices(best_by_dataset, args)

    out_dir = os.path.join(
        "plots",
        "vision",
        "rex",
        "rex_heatmap",
        args.model_family,
        model_dir,
        f"seed={args.seed}",
        _optim_frag(args),
        f"order={args.order}",
        f"granularity={args.granularity}",
        f"skip={_skip_tag(args.skip_modules)}",
    )
    plot_raw_heatmap(matrix, args, model_dir, out_dir)
    plot_delta_heatmap(matrix, args, model_dir, out_dir)


if __name__ == "__main__":
    main()
