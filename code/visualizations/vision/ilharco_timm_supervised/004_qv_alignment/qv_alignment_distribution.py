"""004 — Distribution of off-diagonal QV similarities, per aggregation — timm supervised

The heatmap says which donor-receiver pairs align.  It does not say what the
*population* of alignments looks like, and that is the question the sub-phase
actually turns on: whether the 22x22 matrix is a broad cloud of weak alignment
with a handful of strongly-aligned pairs, or a tight band with no structure to
predict anything with.  A correlation of 0 against Delta means something quite
different in those two cases.

One box per aggregation, all on one axis
----------------------------------------
The boxes in a figure are the aggregations of a single metric --
`cosine/model`, `cosine/layer/mean`, `cosine/neuron/pooled/mean`, ... -- drawn
side by side over the same 462 pairs.  They are not repeated measurements of one
quantity: on ViT-B/16 the same pair reads 0.074 flattened model-wise and 0.167 as
a mean over layers, because flattening weights each layer by its QV energy while
a mean over layers does not.  Putting them on one axis makes the size of that
disagreement the thing you see first, rather than something you reconstruct by
opening two heatmaps.

Metrics get separate figures, because they do not share a scale: `dot` is
unbounded and `cosine` lives in [-1, 1], so a shared axis would compress every
cosine box to a sliver.

The diagonal is excluded, not merely marked
-------------------------------------------
Where donor == receiver the similarity is 1 by algebra rather than by
measurement.  The heatmap can keep those cells visible and outline them; a
distribution cannot, because a box is a summary and an identity contributes to a
summary silently.  Every statistic here is over cross-task pairs only -- the same
exclusion aggregate_qv_alignment.py applies before correlating.

Outliers are drawn, not summarized
----------------------------------
A point further than `--sigma` standard deviations from the mean is drawn
individually and left out of the box body: the quartiles, the median and both
whiskers are computed over the survivors alone.  This is deliberate and it is not
plotly's default 1.5-IQR rule.  The tail here is the interesting part -- a few
strongly-aligned donor-receiver pairs are what a transfer method would exploit --
so those pairs are worth naming on hover rather than folding into a whisker, and
worth keeping out of the quartiles they would otherwise drag.

The mean and sigma that define "outlier" are computed once, over the full
off-diagonal population including the eventual outliers.  Recomputing them on the
survivors would let the threshold chase the data it is filtering.

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
import glob
import json
import math

import plotly.graph_objects as go

from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/vision/ilharco_timm_supervised/004_qv_alignment/vision/qv_alignment"
PLOT_ROOT = "plots/vision/ilharco_timm_supervised/004_qv_alignment/qv_alignment_distribution"

IN_FILE = "qv_alignment.json"

METRICS       = ("cosine", "correlation", "dot", "normalized_l2",
                 "sign_agreement", "angular", "cka")
GRANULARITIES = ("neuron", "layer", "model")
NEURON_MODES  = ("two_stage", "pooled")
OPERATORS     = ("mean", "wmean_params", "wmean_norm", "median",
                 "std", "min", "max", "frac_positive")

# Metrics that take both signs, and for which the zero line is therefore a
# meaningful reference rather than an arbitrary tick.
SIGNED_METRICS = frozenset({"cosine", "correlation", "dot"})

BOX_FILL    = "rgba(31, 119, 180, 0.35)"
BOX_LINE    = "rgb(31, 119, 180)"
OUTLIER_HI  = "rgb(214, 39, 40)"
OUTLIER_LO  = "rgb(148, 103, 189)"

# Half-width of the band the outlier markers are spread across, in x-axis units
# where a box is 1 wide.  Wide enough to separate a dozen markers, narrow enough
# that no marker can be misread as belonging to the neighbouring box.
JITTER_HALF_WIDTH = 0.18

# A filename long enough to break a filesystem is not a filename.  Past this the
# stem is truncated at a variant boundary and the full list goes to stdout.
MAX_STEM_CHARS = 150


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

    # which boxes to draw
    parser.add_argument("--metrics",       required=True, nargs="+", choices=list(METRICS),
                        help="One figure per metric: they do not share a y scale.")
    parser.add_argument("--granularities", required=True, nargs="+", choices=list(GRANULARITIES),
                        help="AGGREGATION levels, not the quantizer's --granularity.")
    parser.add_argument("--neuron-modes",  default=list(NEURON_MODES), nargs="+",
                        choices=list(NEURON_MODES))
    parser.add_argument("--operators",     default=["mean"], nargs="+", choices=list(OPERATORS),
                        help="Ignored at model granularity, which has no aggregation step.")

    parser.add_argument("--sigma",         default=3.0, type=float,
                        help="Outlier threshold in standard deviations from the "
                             "mean of the off-diagonal population. Points beyond "
                             "it are drawn individually and excluded from the box.")

    parser.add_argument("--in-name",       default=IN_FILE)
    parser.add_argument("--in-agg",        default=None,
                        help="The agg= fragment of the alignment JSON to read, "
                             "with or without the prefix. Only needed when more "
                             "than one aggregation set was computed for this "
                             "checkpoint configuration; the error lists the "
                             "candidates when it is.")
    parser.add_argument("--width",         type=int, default=1100)
    parser.add_argument("--height",        type=int, default=700)

    args = parser.parse_args()

    if args.sigma <= 0:
        parser.error("--sigma must be positive: a non-positive threshold makes "
                     "every point an outlier and leaves no box to draw")

    return args


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
    """The tail shared by the input and output trees, above the `agg=` level.

    `alignment` sits where the sibling figures have `split=`: a weight-space
    similarity has no evaluation split to belong to.
    """
    return os.path.join(
        model_dir, f"seed={args.seed}", _optim_frag(args),
        _qat_frag(args), _ptq_frag(args),
    )


def _in_path(args, model_dir):
    """The alignment JSON, found by globbing the `agg=` level.

    Duplicated rather than imported from the 004 experiment directory's
    qv_alignment_common.py, for the same reason cell_value() below is duplicated:
    visualization scripts in this repo are self-contained and import only from
    code/src.
    """
    base = os.path.join(EVAL_ROOT, _run_frag(args, model_dir))

    if args.in_agg is not None:
        frag = args.in_agg if args.in_agg.startswith("agg=") else f"agg={args.in_agg}"
        path = os.path.join(base, frag, "alignment", args.in_name)
        assert os.path.exists(path), f"{path} not found (from --in-agg)"
        return path

    pattern = os.path.join(base, "agg=*", "alignment", args.in_name)
    matches = sorted(glob.glob(pattern))

    assert matches, (
        f"nothing matching {pattern}. Run compute_qv_alignment.py first with "
        f"matching --seed/--qat-bits/--ptq-bits/--granularity/--skip-modules."
    )
    if len(matches) > 1:
        listing = "\n  ".join(
            os.path.basename(os.path.dirname(os.path.dirname(m))) for m in matches
        )
        raise SystemExit(
            f"{len(matches)} aggregation sets under {base}; pass --in-agg with "
            f"one of:\n  {listing}"
        )
    return matches[0]


def _out_dir(args, model_dir):
    """No `agg=` level here, unlike the heatmap.

    A box figure spans several aggregations by construction -- that is the
    comparison it exists to make -- so there is no single fragment to name the
    directory with. The variants go in the stem instead.
    """
    return os.path.join(PLOT_ROOT, _run_frag(args, model_dir), "alignment")


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


def figure_specs(args, metric):
    """Every (metric, granularity, mode, operator) the CLI asked for, for one metric."""
    specs = []
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


def spec_tail(spec):
    """The label without its metric prefix, which the figure already carries."""
    return "/".join(spec_label(*spec).split("/")[1:])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _quantile(sorted_values, q):
    """Linear-interpolation quantile.

    Hand-rolled rather than taken from numpy or scipy, matching the sibling
    scripts: this sub-phase computes its statistics in plain Python throughout so
    that what a figure shows is readable from the figure's own source.
    """
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def off_diagonal_points(results, spec):
    """Every cross-task cell carrying this variant, as (value, donor, receiver).

    Same-task pairs are dropped here and nowhere else, so there is one place to
    read to know what the population is.
    """
    points = []
    for pair in results["pairs"]:
        if pair["donor"] == pair["receiver"]:
            continue
        value = cell_value(pair, *spec)
        if value is None:
            continue
        points.append((value, pair["donor"], pair["receiver"]))
    return points


def box_statistics(points, sigma):
    """Split a population into a box body and its outliers.

    The mean and standard deviation are over the *full* population, outliers
    included, so the threshold does not move as points are removed by it.  The
    deviation is the population form (dividing by n), matching the `std` operator
    in qv_alignment_common.py rather than introducing a second convention.

    Everything drawn as a box -- quartiles, median, both fences -- is then
    computed over the survivors alone.  A sigma of exactly 0 admits no outliers:
    with every value identical there is no tail to separate.
    """
    values = [v for v, _, _ in points]
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)

    threshold = sigma * std
    if std == 0.0:
        inliers, outliers = list(points), []
    else:
        inliers  = [p for p in points if abs(p[0] - mean) <= threshold]
        outliers = [p for p in points if abs(p[0] - mean) >  threshold]

    body = sorted(v for v, _, _ in inliers)

    return {
        "n":          n,
        "mean":       mean,
        "std":        std,
        "threshold":  threshold,
        "n_inliers":  len(body),
        "q1":         _quantile(body, 0.25),
        "median":     _quantile(body, 0.50),
        "q3":         _quantile(body, 0.75),
        "lowerfence": body[0]  if body else None,
        "upperfence": body[-1] if body else None,
        "outliers":   sorted(outliers, key=lambda p: p[0]),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _jitter_offsets(k):
    """Deterministic x offsets spreading k markers across one box.

    Deterministic rather than random so that re-rendering a figure produces the
    same picture -- a figure that moves between runs cannot be diffed.
    """
    if k <= 1:
        return [0.0] * k
    step = 2 * JITTER_HALF_WIDTH / (k - 1)
    return [-JITTER_HALF_WIDTH + i * step for i in range(k)]


def _tick_label(spec, stats):
    """The aggregation, plus how much of its population the box actually covers.

    Printed on the axis rather than left to hover, so a box drawn over 455 of 462
    pairs cannot be read as one drawn over all of them.
    """
    n_out = len(stats["outliers"])
    suffix = f", {n_out} out" if n_out else ""
    return (f"{spec_tail(spec)}<br>"
            f"<sup>n={stats['n_inliers']} of {stats['n']}{suffix}</sup>")


def _add_box(fig, xpos, stats, label):
    fig.add_trace(go.Box(
        x=[xpos],
        q1=[stats["q1"]], median=[stats["median"]], q3=[stats["q3"]],
        lowerfence=[stats["lowerfence"]], upperfence=[stats["upperfence"]],
        mean=[stats["mean"]],
        boxmean=True,
        # Precomputed statistics only: plotly must not re-derive quartiles, and
        # must not re-apply its own 1.5-IQR outlier rule on top of the sigma one.
        boxpoints=False,
        width=0.5,
        fillcolor=BOX_FILL,
        line=dict(color=BOX_LINE, width=1.6),
        name=label,
        showlegend=False,
        hovertemplate=(
            f"<b>{label}</b><br>"
            f"median = {stats['median']:.4f}<br>"
            f"q1 = {stats['q1']:.4f}, q3 = {stats['q3']:.4f}<br>"
            f"whiskers = [{stats['lowerfence']:.4f}, {stats['upperfence']:.4f}]<br>"
            f"mean (dashed) = {stats['mean']:.4f}, sd = {stats['std']:.4f}<br>"
            f"box over {stats['n_inliers']} of {stats['n']} cross-task pairs"
            "<extra></extra>"
        ),
    ))


def _add_outliers(fig, xpos, stats, label, sigma):
    if not stats["outliers"]:
        return

    offsets = _jitter_offsets(len(stats["outliers"]))
    xs, ys, colors, hover = [], [], [], []

    for (value, donor, receiver), dx in zip(stats["outliers"], offsets):
        deviation = (value - stats["mean"]) / stats["std"]
        xs.append(xpos + dx)
        ys.append(value)
        colors.append(OUTLIER_HI if deviation > 0 else OUTLIER_LO)
        hover.append(
            f"<b>{donor} → {receiver}</b><br>{label} = {value:.4f}<br>"
            f"{deviation:+.2f} sd from the mean (threshold ±{sigma:g})"
        )

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=7, color=colors, line=dict(color="white", width=1)),
        text=hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))


def build_figure(results, args, metric, specs):
    stats_by_spec = []
    for spec in specs:
        points = off_diagonal_points(results, spec)
        assert points, (
            f"no off-diagonal cell carries {spec_label(*spec)}; "
            f"re-run compute_qv_alignment.py including it"
        )
        stats_by_spec.append((spec, box_statistics(points, args.sigma)))

    fig = go.Figure()
    for xpos, (spec, stats) in enumerate(stats_by_spec):
        label = spec_label(*spec)
        _add_box(fig, xpos, stats, label)
        _add_outliers(fig, xpos, stats, label, args.sigma)

    if metric in SIGNED_METRICS:
        # Where the metric is signed, zero separates aligned from anti-aligned,
        # and whether a box straddles it is the first thing worth seeing.
        fig.add_hline(y=0.0, line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))

    fig.update_xaxes(
        title_text="Aggregation",
        tickmode="array",
        tickvals=list(range(len(stats_by_spec))),
        ticktext=[_tick_label(spec, stats) for spec, stats in stats_by_spec],
        range=[-0.6, len(stats_by_spec) - 0.4],
    )
    fig.update_yaxes(title_text=metric)

    cfg = results["config"]
    skip_str = ",".join(sorted(cfg["skip_modules"]))
    n_out = sum(len(stats["outliers"]) for _, stats in stats_by_spec)

    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=90, r=40, t=115, b=130),
        # Kept short on purpose: kaleido derives a fallback output filename from
        # the title text and stat()s it, so a title much past ~250 characters
        # fails PDF export with ENAMETOOLONG. The reasoning the subtitle would
        # otherwise spell out lives in the module docstring.
        title=dict(
            text=(f"QV similarity distribution — {metric}, by aggregation<br>"
                  f"<sup>{results['display_name']} | seed={cfg['seed']} | "
                  f"qat={cfg['qat_bits']}b | ptq={cfg['ptq_bits']}b | "
                  f"{cfg['granularity']} | skip={skip_str}<br>"
                  f"cross-task pairs only | {n_out} point"
                  f"{'' if n_out == 1 else 's'} beyond ±{args.sigma:g} sd drawn "
                  f"separately, excluded from the boxes</sup>"),
            x=0.5, xanchor="center", font=dict(size=13),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------
def figure_stem(metric, specs):
    """`distribution_<metric>_<variant>__<variant>...`, truncated if absurd.

    The variants are in the stem rather than in a directory because a figure
    spans several of them; the metric leads because it is the one thing every box
    in the figure shares.
    """
    tails = [spec_tail(spec).replace("/", "-") for spec in specs]
    stem = f"distribution_{metric}_" + "__".join(tails)

    if len(stem) <= MAX_STEM_CHARS:
        return stem

    kept = []
    head = f"distribution_{metric}_"
    for tail in tails:
        candidate = head + "__".join(kept + [tail])
        # Leave room for the `_andNmore` suffix rather than discovering it does
        # not fit after the fact.
        if len(candidate) + 12 > MAX_STEM_CHARS:
            break
        kept.append(tail)

    return head + "__".join(kept) + f"_and{len(tails) - len(kept)}more"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    model_dir = sanitize_timm_model_name(args.model_name)

    results_path = _in_path(args, model_dir)
    with open(results_path) as f:
        results = json.load(f)

    out_dir = _out_dir(args, model_dir)
    os.makedirs(out_dir, exist_ok=True)

    for metric in args.metrics:
        specs = figure_specs(args, metric)
        fig = build_figure(results, args, metric, specs)

        stem = figure_stem(metric, specs)
        if stem.endswith("more"):
            # Truncation must not be silent: the file name would then understate
            # what the figure shows.
            print(f"{metric}: stem truncated, full variant list is "
                  f"{[spec_label(*s) for s in specs]}")

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
