"""REx transfer heatmaps (QV-transfer style layout) for timm supervised.

Loads REx-transfer results and produces two-panel heatmaps:

1) Raw transfer accuracy:
   - left panel: transfer matrix (rows=target dataset, cols=source dataset)
   - right panel: target baselines (Target FP, Target FP+PTQ, Random)

2) Delta transfer accuracy:
   - left panel: transfer minus target baseline
   - right panel: same target baseline panel

This script generates one heatmap set per alpha value.
"""

import argparse
import glob
import json
import os
import sys

from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.vision.utils import sanitize_timm_model_name


HEATMAP_COLORSCALE_SEQUENTIAL = "Viridis"
HEATMAP_COLORSCALE_DIVERGING = "RdYlGn"

BASELINE_COLUMNS = [
    ("test_accuracy_target_fp", "Target FP"),
    ("test_accuracy_target_fp_ptq", "Target FP+PTQ"),
    ("random_chance", "Random"),
]

METRIC_KEYS = {
    "patched_fp": "test_accuracy_patched_fp",
    "patched_fp_ptq": "test_accuracy_patched_fp_ptq",
}
METRIC_LABELS = {
    "patched_fp": "Patched FP",
    "patched_fp_ptq": "Patched FP+PTQ",
}

SUBTRACTOR_KEYS = {
    "target_fp": "test_accuracy_target_fp",
    "target_fp_ptq": "test_accuracy_target_fp_ptq",
}
SUBTRACTOR_LABELS = {
    "target_fp": "Target FP",
    "target_fp_ptq": "Target FP+PTQ",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", required=True, type=int)

    # Optim path fragment components (kept for qv-style CLI parity/titles).
    parser.add_argument("--optim", required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--wd", required=True, type=float)
    parser.add_argument("--ls", required=True, type=float)
    parser.add_argument("--wl", required=True, type=int)
    parser.add_argument("--max-grad-norm", required=True, type=float)
    parser.add_argument("--batch-size", required=True, type=int)

    parser.add_argument("--rex-bits", required=True, type=int)
    parser.add_argument("--rex-order", required=True, type=int)
    parser.add_argument("--rex-sparsity", required=True, type=float)
    parser.add_argument("--rex-granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--rex-skip-modules", required=True, nargs="+")

    parser.add_argument("--ptq-bits", required=True, type=int)
    parser.add_argument("--ptq-granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--ptq-skip-modules", required=True, nargs="+")

    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset/order of alphas to plot. Default: infer from JSON.",
    )
    parser.add_argument(
        "--evaluation-root",
        default="quantization/qat-transfer/evaluations/vision/rex_transfer/ilharco_timm_supervised",
        help="Root directory containing timm REx-transfer eval_results.json files.",
    )
    return parser.parse_args()


def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(optim, lr, wd, ls, wl, mgn, bs):
    return f"optim={optim}_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={mgn}_bs={bs}"


def _rex_frag(args):
    return (
        f"rex=bits={args.rex_bits}_order={args.rex_order}_sparsity={args.rex_sparsity}"
        f"_gran={args.rex_granularity}_skip={_skip_tag(args.rex_skip_modules)}"
    )


def _ptq_frag(args):
    return (
        f"ptq=bits={args.ptq_bits}_gran={args.ptq_granularity}"
        f"_skip={_skip_tag(args.ptq_skip_modules)}"
    )


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _float_equal(a, b, eps=1e-12):
    return abs(float(a) - float(b)) <= eps


def _payload_rank(payload):
    # Prefer non-smoke runs over smoke runs when both exist.
    lnb = payload.get("limit_num_batches")
    src_lne = payload.get("source", {}).get("limit_num_epochs")
    tgt_lne = payload.get("target", {}).get("limit_num_epochs")
    full_run = int(lnb is None and src_lne is None and tgt_lne is None)
    lnb_score = 10**9 if lnb is None else int(lnb)
    src_lne_score = 10**9 if src_lne is None else int(src_lne)
    tgt_lne_score = 10**9 if tgt_lne is None else int(tgt_lne)
    result_len = len(payload.get("results", []))
    return (full_run, src_lne_score, tgt_lne_score, lnb_score, result_len)


def _matches_payload(payload, args):
    if payload.get("model_family") != "ilharco_timm_supervised":
        return False
    if payload.get("model_name") != args.model_name:
        return False

    source = payload.get("source", {})
    target = payload.get("target", {})
    if int(source.get("seed", -1)) != int(args.seed):
        return False
    if int(target.get("seed", -1)) != int(args.seed):
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

    rex = payload.get("rex", {})
    if int(rex.get("bits", -1)) != int(args.rex_bits):
        return False
    if int(rex.get("order", -1)) != int(args.rex_order):
        return False
    if not _float_equal(rex.get("sparsity"), args.rex_sparsity):
        return False
    if rex.get("granularity") != args.rex_granularity:
        return False
    if sorted(rex.get("skip_modules", [])) != sorted(args.rex_skip_modules):
        return False

    ptq = payload.get("ptq", {})
    if int(ptq.get("bits", -1)) != int(args.ptq_bits):
        return False
    if ptq.get("granularity") != args.ptq_granularity:
        return False
    if sorted(ptq.get("skip_modules", [])) != sorted(args.ptq_skip_modules):
        return False

    return True


def collect_payloads(args):
    model_dir = sanitize_timm_model_name(args.model_name)
    root = Path(args.evaluation_root) / model_dir
    pattern = str(root / "**" / "eval_results.json")
    files = glob.glob(pattern, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(f"No eval_results.json found under: {root}")

    pair_best = {}
    target_best = {}
    inferred_alphas = set()

    for path in files:
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue

        if not _matches_payload(payload, args):
            continue

        source_name = payload.get("source", {}).get("dataset_name")
        target_name = payload.get("target", {}).get("dataset_name")
        if source_name is None or target_name is None:
            continue

        for item in payload.get("results", []):
            alpha = item.get("alpha")
            if alpha is not None:
                inferred_alphas.add(float(alpha))

        rank = _payload_rank(payload)
        pair_key = (source_name, target_name)
        prev_pair = pair_best.get(pair_key)
        if prev_pair is None or rank > prev_pair["rank"]:
            pair_best[pair_key] = {
                "payload": payload,
                "path": path,
                "rank": rank,
            }

        prev_target = target_best.get(target_name)
        if prev_target is None or rank > prev_target["rank"]:
            target_best[target_name] = {
                "payload": payload,
                "path": path,
                "rank": rank,
            }

    if len(pair_best) == 0:
        raise ValueError(
            "No REx-transfer eval files matched the requested filters. "
            "Check args and evaluation root."
        )

    sources = sorted({src for src, _ in pair_best.keys()}, key=str.lower)
    targets = sorted({tgt for _, tgt in pair_best.keys()}, key=str.lower)

    if args.alphas is None:
        alphas = sorted(inferred_alphas)
    else:
        alphas = [float(a) for a in args.alphas]
    if len(alphas) == 0:
        raise ValueError("No alpha values available after filtering.")

    return pair_best, target_best, sources, targets, alphas, model_dir


def _find_alpha_result(payload, alpha, eps=1e-12):
    for item in payload.get("results", []):
        item_alpha = item.get("alpha")
        if item_alpha is None:
            continue
        if abs(float(item_alpha) - float(alpha)) <= eps:
            return item
    return None


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


def _add_diagonal_borders(fig, sources, targets, color="black", width=2, xref="x", yref="y"):
    src_idx = {name: i for i, name in enumerate(sources)}
    tgt_idx = {name: i for i, name in enumerate(targets)}
    for name in sorted(set(sources) & set(targets), key=str.lower):
        x = src_idx[name]
        y = tgt_idx[name]
        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=x - 0.5,
            x1=x + 0.5,
            y0=y - 0.5,
            y1=y + 0.5,
            line=dict(color=color, width=width),
            fillcolor="rgba(0,0,0,0)",
        )


def build_matrices(
    pair_best,
    target_best,
    sources,
    targets,
    alpha,
    metric_key,
    subtractor_key=None,
):
    transfer_z, transfer_text = [], []
    baseline_z, baseline_text = [], []
    baseline_labels = [label for _, label in BASELINE_COLUMNS]

    for target_name in targets:
        target_payload = target_best[target_name]["payload"]
        target_baselines = target_payload.get("target_baselines", {})
        subtractor_val = target_baselines.get(subtractor_key) if subtractor_key else None

        t_row_z, t_row_text = [], []
        b_row_z, b_row_text = [], []

        for source_name in sources:
            pair = pair_best.get((source_name, target_name))
            if pair is None:
                t_row_z.append(None)
                t_row_text.append("")
                continue

            item = _find_alpha_result(pair["payload"], alpha)
            if item is None:
                t_row_z.append(None)
                t_row_text.append("")
                continue

            value = item.get(metric_key)
            if value is None:
                t_row_z.append(None)
                t_row_text.append("")
                continue

            value = float(value)
            if subtractor_key is not None:
                if subtractor_val is None:
                    t_row_z.append(None)
                    t_row_text.append("")
                    continue
                value = value - float(subtractor_val)

            t_row_z.append(value)
            t_row_text.append(f"{value:.2f}")

        for base_key, _ in BASELINE_COLUMNS:
            base_value = target_baselines.get(base_key)
            if base_value is None:
                b_row_z.append(None)
                b_row_text.append("")
            else:
                base_value = float(base_value)
                b_row_z.append(base_value)
                b_row_text.append(f"{base_value:.2f}")

        transfer_z.append(t_row_z)
        transfer_text.append(t_row_text)
        baseline_z.append(b_row_z)
        baseline_text.append(b_row_text)

    return {
        "sources": sources,
        "targets": targets,
        "baseline_labels": baseline_labels,
        "transfer_z": transfer_z,
        "transfer_text": transfer_text,
        "baseline_z": baseline_z,
        "baseline_text": baseline_text,
    }


def _common_layout(fig, targets, sources, baseline_labels):
    fig.update_layout(
        template="plotly_white",
        height=max(400, 60 * len(targets) + 180),
        width=max(900, 55 * len(sources) + 115 * len(baseline_labels) + 260),
        margin=dict(l=80, r=220, t=120, b=90),
    )
    fig.update_xaxes(
        title_text="REx source dataset<br>(dataset displacement is computed from)",
        row=1,
        col=1,
        side="bottom",
    )
    fig.update_xaxes(title_text="Target baselines", row=1, col=2, side="bottom")
    fig.update_yaxes(
        title_text="Target dataset<br>(dataset displacement is applied to)",
        row=1,
        col=1,
        autorange="reversed",
    )
    fig.update_yaxes(row=1, col=2, showticklabels=False, autorange="reversed")


def _output_dir(args, model_dir, optim_frag, rex_frag, ptq_frag, alpha):
    return os.path.join(
        "plots",
        "vision",
        "rex",
        "rex_transfer_heatmap",
        "ilharco_timm_supervised",
        model_dir,
        f"seed={args.seed}",
        optim_frag,
        rex_frag,
        ptq_frag,
        f"alpha={alpha:g}",
    )


def plot_raw_heatmap(
    matrix,
    args,
    model_dir,
    optim_frag,
    rex_frag,
    ptq_frag,
    alpha,
    metric_tag,
):
    sources = matrix["sources"]
    targets = matrix["targets"]
    baseline_labels = matrix["baseline_labels"]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[max(1, len(sources)), len(baseline_labels)],
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["transfer_z"],
            x=sources,
            y=targets,
            text=matrix["transfer_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis",
            xgap=1,
            ygap=1,
            hovertemplate="target=%{y}<br>source=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["baseline_z"],
            x=baseline_labels,
            y=targets,
            text=matrix["baseline_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis2",
            xgap=1,
            ygap=1,
            hovertemplate="target=%{y}<br>baseline=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _add_diagonal_borders(fig, sources=sources, targets=targets, xref="x", yref="y")

    title = (
        f"REx Transfer ({METRIC_LABELS[metric_tag]})<br>"
        f"<sup>{args.model_name} | seed={args.seed} | alpha={alpha:g} | "
        f"optim={args.optim} | rex(bits={args.rex_bits}, order={args.rex_order}, sparsity={args.rex_sparsity}, "
        f"gran={args.rex_granularity}) | ptq(bits={args.ptq_bits}, gran={args.ptq_granularity})</sup>"
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0,
            cmax=1,
            colorbar=dict(title="Transfer Acc", x=1.01, y=0.78, len=0.42),
        ),
        coloraxis2=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0,
            cmax=1,
            colorbar=dict(title="Baseline Acc", x=1.01, y=0.22, len=0.42),
        ),
    )
    _common_layout(fig, targets=targets, sources=sources, baseline_labels=baseline_labels)

    out_dir = _output_dir(args, model_dir, optim_frag, rex_frag, ptq_frag, alpha)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_rex_transfer_{metric_tag}.png")
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


def plot_difference_heatmap(
    matrix,
    args,
    model_dir,
    optim_frag,
    rex_frag,
    ptq_frag,
    alpha,
    metric_tag,
    subtractor_tag,
):
    sources = matrix["sources"]
    targets = matrix["targets"]
    baseline_labels = matrix["baseline_labels"]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[max(1, len(sources)), len(baseline_labels)],
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["transfer_z"],
            x=sources,
            y=targets,
            text=matrix["transfer_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis",
            xgap=1,
            ygap=1,
            hovertemplate="target=%{y}<br>source=%{x}<br>delta=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix["baseline_z"],
            x=baseline_labels,
            y=targets,
            text=matrix["baseline_text"],
            texttemplate="%{text}",
            coloraxis="coloraxis2",
            xgap=1,
            ygap=1,
            hovertemplate="target=%{y}<br>baseline=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _add_diagonal_borders(fig, sources=sources, targets=targets, xref="x", yref="y")

    cmin, cmax = _robust_symmetric_bounds(
        _finite_values(matrix["transfer_z"]),
        center=0.0,
        min_span=0.02,
    )

    title = (
        f"REx Transfer ({METRIC_LABELS[metric_tag]}) − {SUBTRACTOR_LABELS[subtractor_tag]}<br>"
        f"<sup>{args.model_name} | seed={args.seed} | alpha={alpha:g} | "
        f"optim={args.optim} | rex(bits={args.rex_bits}, order={args.rex_order}, sparsity={args.rex_sparsity}, "
        f"gran={args.rex_granularity}) | ptq(bits={args.ptq_bits}, gran={args.ptq_granularity})</sup>"
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=HEATMAP_COLORSCALE_DIVERGING,
            cmin=cmin,
            cmax=cmax,
            cmid=0,
            colorbar=dict(
                title=f"Acc Δ (vs {SUBTRACTOR_LABELS[subtractor_tag]})",
                x=1.01,
                y=0.78,
                len=0.42,
            ),
        ),
        coloraxis2=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0,
            cmax=1,
            colorbar=dict(title="Baseline Acc", x=1.01, y=0.22, len=0.42),
        ),
    )
    _common_layout(fig, targets=targets, sources=sources, baseline_labels=baseline_labels)

    out_dir = _output_dir(args, model_dir, optim_frag, rex_frag, ptq_frag, alpha)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"heatmap_rex_transfer_{metric_tag}_minus_{subtractor_tag}.png",
    )
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()

    pair_best, target_best, sources, targets, alphas, model_dir = collect_payloads(args)
    optim_frag = _optim_frag(
        args.optim,
        args.lr,
        args.wd,
        args.ls,
        args.wl,
        args.max_grad_norm,
        args.batch_size,
    )
    rex_frag = _rex_frag(args)
    ptq_frag = _ptq_frag(args)

    for alpha in alphas:
        for metric_tag, metric_key in METRIC_KEYS.items():
            raw_matrix = build_matrices(
                pair_best=pair_best,
                target_best=target_best,
                sources=sources,
                targets=targets,
                alpha=alpha,
                metric_key=metric_key,
                subtractor_key=None,
            )
            plot_raw_heatmap(
                matrix=raw_matrix,
                args=args,
                model_dir=model_dir,
                optim_frag=optim_frag,
                rex_frag=rex_frag,
                ptq_frag=ptq_frag,
                alpha=alpha,
                metric_tag=metric_tag,
            )

            for subtractor_tag, subtractor_key in SUBTRACTOR_KEYS.items():
                diff_matrix = build_matrices(
                    pair_best=pair_best,
                    target_best=target_best,
                    sources=sources,
                    targets=targets,
                    alpha=alpha,
                    metric_key=metric_key,
                    subtractor_key=subtractor_key,
                )
                plot_difference_heatmap(
                    matrix=diff_matrix,
                    args=args,
                    model_dir=model_dir,
                    optim_frag=optim_frag,
                    rex_frag=rex_frag,
                    ptq_frag=ptq_frag,
                    alpha=alpha,
                    metric_tag=metric_tag,
                    subtractor_tag=subtractor_tag,
                )


if __name__ == "__main__":
    main()
