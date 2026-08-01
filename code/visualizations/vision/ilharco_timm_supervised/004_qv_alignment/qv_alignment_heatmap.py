"""004 — QV similarity heatmap (donor x receiver) — timm supervised

The similarity matrix itself, from the per-model qv_alignment.json written by
compute_qv_alignment.py:

  rows  = receiver datasets (the task the QV is applied to)
  cols  = donor datasets    (the task the QV was computed on)

One figure per requested (metric, granularity, operator) combination, so a single
invocation can render a family of maps and the differences between aggregations
are visible side by side -- which is the point, since the whole sub-phase turns
on those aggregations disagreeing.

The diagonal is drawn but excluded from the colour bounds
---------------------------------------------------------
Where donor == receiver the similarity is 1 (or 0, for the distance metrics) by
algebra rather than by measurement: it is the same vector compared with itself.
Left in the colour range it would set one end of the scale and compress every
real cell into a narrow band of colour.  So the diagonal is outlined in black,
kept visible, and dropped from the bounds computation.

Bounds are otherwise robust, from the 5th and 95th percentiles of the
off-diagonal cells, because a single strongly-aligned pair would otherwise flatten
the map.  Signed metrics get a diverging scale centred on zero, so
"anti-aligned" and "unaligned" are distinguishable at a glance; the metrics
bounded below by zero get a sequential scale, where a diverging one would imply
a meaningless midpoint.

Writes HTML, PDF and PNG to plots/vision/ilharco_timm_supervised/004_qv_alignment/.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import os

os.chdir(_PROJECT_ROOT)

import argparse
import json

import plotly.graph_objects as go

from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/vision/ilharco_timm_supervised/004_qv_alignment/vision/qv_alignment"
PLOT_ROOT = "plots/vision/ilharco_timm_supervised/004_qv_alignment/qv_alignment_heatmap"

IN_FILE = "qv_alignment.json"

METRICS       = ("cosine", "correlation", "dot", "normalized_l2",
                 "sign_agreement", "angular", "cka")
GRANULARITIES = ("neuron", "layer", "model")
NEURON_MODES  = ("two_stage", "pooled")
OPERATORS     = ("mean", "wmean_params", "wmean_norm", "median",
                 "std", "min", "max", "frac_positive")

# Metrics that take both signs, and therefore want a diverging scale centred on
# zero.  The rest are bounded below by zero and get a sequential one.
SIGNED_METRICS = frozenset({"cosine", "correlation", "dot"})

HEATMAP_COLORSCALE_SEQUENTIAL = "Viridis"
HEATMAP_COLORSCALE_DIVERGING  = "RdBu"

# Value each metric takes on the diagonal, used only to label it in hover.
IDENTITY_VALUE = {
    "cosine": 1.0, "correlation": 1.0, "sign_agreement": 1.0, "cka": 1.0,
    "normalized_l2": 0.0, "angular": 0.0, "dot": None,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name",    required=True,
                        help="timm model name, e.g. vit_base_patch16_224.orig_in21k")
    parser.add_argument("--seed",          required=True, type=int)

    # optim path-fragment components
    parser.add_argument("--optim",         required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",            required=True, type=float)
    parser.add_argument("--wd",            required=True, type=float)
    parser.add_argument("--ls",            required=True, type=float)
    parser.add_argument("--wl",            required=True, type=int)
    parser.add_argument("--max-grad-norm", required=True, type=float)
    parser.add_argument("--batch-size",    required=True, type=int)

    # quantization path-fragment components
    parser.add_argument("--qat-bits",      required=True, type=int)
    parser.add_argument("--ptq-bits",      required=True, type=int)
    parser.add_argument("--granularity",   required=True, choices=["tensor", "channel"],
                        help="The QUANTIZER's granularity, part of the path. Not to "
                             "be confused with --granularities below.")
    parser.add_argument("--skip-modules",  required=True, nargs="+",
                        help="One or more module names skipped during quantization "
                             "(no default: must be specified explicitly).")

    # which figures to render
    parser.add_argument("--metrics",       required=True, nargs="+", choices=list(METRICS))
    parser.add_argument("--granularities", required=True, nargs="+", choices=list(GRANULARITIES),
                        help="AGGREGATION levels, not the quantizer's --granularity.")
    parser.add_argument("--neuron-modes",  default=list(NEURON_MODES), nargs="+",
                        choices=list(NEURON_MODES))
    parser.add_argument("--operators",     default=["mean"], nargs="+", choices=list(OPERATORS),
                        help="Ignored at model granularity, which has no aggregation step.")

    parser.add_argument("--robust",        action="store_true",
                        help="Clip the colour scale to the 5th/95th percentile of the "
                             "off-diagonal cells. Off by default: the diagonal is "
                             "already excluded, so clipping mostly saturates the "
                             "strongly-aligned pairs the map exists to show.")

    parser.add_argument("--in-name",       default=IN_FILE)
    parser.add_argument("--width",         type=int, default=1000)
    parser.add_argument("--height",        type=int, default=900)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers (must mirror what compute_qv_alignment.py writes)
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args):
    # The experiment configs hardcode "optim=adamw" in the path even though they
    # accept other optimizers. We mirror that exactly.
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={args.batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _run_frag(args, model_dir):
    """The tail shared by the input and output trees.

    `alignment` sits where the sibling figures have `split=`: a weight-space
    similarity has no evaluation split to belong to.
    """
    return os.path.join(
        model_dir, f"seed={args.seed}", _optim_frag(args),
        _qat_frag(args), _ptq_frag(args), "alignment",
    )


def _in_dir(args, model_dir):
    return os.path.join(EVAL_ROOT, _run_frag(args, model_dir))


def _out_dir(args, model_dir):
    return os.path.join(PLOT_ROOT, _run_frag(args, model_dir))


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------
def cell_value(pair, metric, granularity, mode, operator):
    """One similarity out of a pair record, or None if it was not emitted.

    Mirrors similarity_series() in aggregate_qv_alignment.py.  Duplicated rather
    than imported because visualization scripts in this repo are self-contained
    and import only from code/src.
    """
    block = pair["metrics"].get(metric)
    if block is None:
        return None

    if granularity == "model":
        return block.get("model_wise")

    if granularity == "layer":
        layer = block.get("layer_wise")
        return layer["agg"].get(operator) if layer else None

    neuron = block.get("neuron_wise")
    if not neuron or mode not in neuron:
        return None
    return neuron[mode].get(operator)


def figure_specs(args):
    """Every (metric, granularity, mode, operator) the CLI asked for."""
    specs = []
    for metric in args.metrics:
        for granularity in args.granularities:
            if granularity == "model":
                specs.append((metric, granularity, None, None))
            elif granularity == "layer":
                specs.extend((metric, granularity, None, op) for op in args.operators)
            else:
                specs.extend(
                    (metric, granularity, mode, op)
                    for mode in args.neuron_modes
                    for op in args.operators
                )
    return specs


def spec_label(metric, granularity, mode, operator):
    parts = [metric, granularity]
    if mode is not None:
        parts.append(mode)
    if operator is not None:
        parts.append(operator)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Colour bounds
# ---------------------------------------------------------------------------
def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def colour_bounds(values, signed, robust, min_span=0.02, q_low=0.05, q_high=0.95):
    """Colour bounds over the off-diagonal cells.

    Full range by default.  The 001 heatmap clips to the 5th/95th percentile to
    stop one outlier flattening the map, but that reasoning does not transfer
    here: the outlier it was defending against is the donor == receiver
    diagonal, and the diagonal is already excluded.  On the measured ViT-B/16
    matrix the off-diagonal cosines run to 0.233 while the 95th percentile is
    0.050, so clipping would saturate every strongly-aligned pair to one colour
    -- and which pairs those are is the entire question the map is asked to
    answer.  `--robust` restores the clipping for a matrix where the tail really
    is a distraction.

    A signed metric is bounded symmetrically about zero, so the midpoint of the
    diverging scale is zero and a negative cell reads as negative rather than
    merely low.
    """
    finite = sorted(v for v in values if v is not None)
    if not finite:
        return 0.0, 1.0

    if robust:
        lo, hi = _quantile(finite, q_low), _quantile(finite, q_high)
    else:
        lo, hi = finite[0], finite[-1]

    if signed:
        half = max(abs(lo), abs(hi), min_span / 2)
        return -half, half

    if hi - lo < min_span:
        mid = (hi + lo) / 2
        return mid - min_span / 2, mid + min_span / 2
    return lo, hi


def _add_diagonal_borders(fig, datasets, color="black", width=2):
    """Outline the donor == receiver cells.

    They are an algebraic identity rather than a measurement, and marking them
    keeps a reader from reading the bright diagonal as a result.
    """
    for i in range(len(datasets)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
            line=dict(color=color, width=width),
            fillcolor="rgba(0,0,0,0)",
        )


def _hover(donor, receiver, pair, value, spec):
    """Hover text for one cell.

    The norm ratio rides along because a similarity alone cannot distinguish
    "right direction, wrong magnitude" from "right direction and comparable
    magnitude" -- the overshoot-versus-anti-alignment question 001 raises.
    """
    head = f"{donor} → {receiver}"

    if pair is None:
        return f"{head}<br>no checkpoint pair"
    if value is None:
        return f"{head}<br>not emitted for {spec_label(*spec)}"

    ratio = pair["norm_ratio"]
    ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"

    text = (f"{head}<br>{spec_label(*spec)} = {value:.4f}"
            f"<br>‖ρ_donor‖/‖ρ_receiver‖ = {ratio_str}")
    if donor == receiver:
        text += "<br><b>diagonal: an identity, not a measurement</b>"
    return text


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args, spec):
    metric, granularity, mode, operator = spec
    datasets = results["datasets"]
    index = {(p["donor"], p["receiver"]): p for p in results["pairs"]}

    z, hover, off_diagonal = [], [], []
    for receiver in datasets:
        row_z, row_t = [], []
        for donor in datasets:
            pair = index.get((donor, receiver))
            value = cell_value(pair, *spec) if pair else None
            row_z.append(value)

            row_t.append(_hover(donor, receiver, pair, value, spec))

            if pair is not None and value is not None and donor != receiver:
                off_diagonal.append(value)

        z.append(row_z)
        hover.append(row_t)

    assert off_diagonal, (
        f"no off-diagonal cell carries {spec_label(*spec)}; "
        f"re-run compute_qv_alignment.py including it"
    )

    signed = metric in SIGNED_METRICS
    cmin, cmax = colour_bounds(off_diagonal, signed, args.robust)

    fig = go.Figure(
        go.Heatmap(
            z=z, x=datasets, y=datasets,
            text=hover, hovertemplate="%{text}<extra></extra>",
            colorscale=(HEATMAP_COLORSCALE_DIVERGING if signed
                        else HEATMAP_COLORSCALE_SEQUENTIAL),
            reversescale=signed,
            zmin=cmin, zmax=cmax,
            zmid=0.0 if signed else None,
            xgap=1, ygap=1,
            colorbar=dict(title=metric, len=0.85),
        )
    )

    _add_diagonal_borders(fig, datasets)

    fig.update_xaxes(
        title_text="Donor dataset<br>(task the QV was computed on)",
        side="bottom", tickangle=-45,
    )
    fig.update_yaxes(
        title_text="Receiver dataset<br>(task the QV is applied to)",
        autorange="reversed",
    )

    cfg = results["config"]
    skip_str = ",".join(sorted(cfg["skip_modules"]))
    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=140, r=40, t=110, b=150),
        title=dict(
            text=(f"QV similarity — {spec_label(*spec)}<br>"
                  f"<sup>{results['display_name']} | seed={cfg['seed']} | "
                  f"qat_bits={cfg['qat_bits']} | ptq_bits={cfg['ptq_bits']} | "
                  f"granularity={cfg['granularity']} | skip={skip_str} | "
                  f"colour bounds [{cmin:+.3f}, {cmax:+.3f}] from off-diagonal cells"
                  f"{', 5–95th pct' if args.robust else ''}</sup>"),
            x=0.5, xanchor="center", font=dict(size=13),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    model_dir = sanitize_timm_model_name(args.model_name)

    results_path = os.path.join(_in_dir(args, model_dir), args.in_name)
    assert os.path.exists(results_path), (
        f"{results_path} not found. Run compute_qv_alignment.py first with "
        f"matching --seed/--qat-bits/--ptq-bits/--granularity/--skip-modules."
    )
    with open(results_path) as f:
        results = json.load(f)

    out_dir = _out_dir(args, model_dir)
    os.makedirs(out_dir, exist_ok=True)

    for spec in figure_specs(args):
        fig = build_figure(results, args, spec)
        stem = f"heatmap_{spec_label(*spec).replace('/', '_')}"

        for ext in ("html", "pdf", "png"):
            path = os.path.join(out_dir, f"{stem}.{ext}")
            if ext == "html":
                fig.write_html(path, include_plotlyjs="cdn")
            elif ext == "pdf":
                fig.write_image(path)
            else:
                fig.write_image(path, scale=300 / 96)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
