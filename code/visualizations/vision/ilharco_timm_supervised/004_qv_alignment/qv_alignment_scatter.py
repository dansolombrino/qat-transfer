"""004 — QV similarity against transfer outcome — timm supervised

The figure reviewer 3HFP's question reduces to.  One marker per cross-task
donor-receiver pair:

    x   how similar the donor's quantization vector is to the receiver's, under
        the variant named by --variant (metric / granularity / operator).
    y   what the transfer actually bought, selected by --outcome:

        delta          Top-1 gain over vanilla PTQ at lambda = 1, the data-free
                       setting the paper's headline claim rests on.
        delta_best     the same at the receiver-validation-selected lambda.
        recovery       delta / delta_ceiling, the fraction of the receiver's own
                       QAT gain that the donor patch recovers.  This is the
                       observable side of Proposition 1's cos^2 law and is the
                       y most figures here should use.
        recovery_best  the same at lambda_best.

With --squared the x axis is the squared similarity, because Proposition 1
predicts recovery tracks cos^2 rather than cos.

Same-task pairs are absent, not hidden: the aggregate excludes them, since on the
diagonal the similarity is 1 by algebra and the gain is the QAT ceiling rather
than a transfer result.  Including them would put a perfect-similarity,
maximal-gain point in the corner of every scatter and bend any line drawn
through it.

What is read and what is drawn
------------------------------
Every point and every coefficient comes from the `points` and `pooled` /
`per_model` blocks of qv_alignment_delta.json.  The correlation in the
annotation is *read*, never recomputed, so the number on the figure is by
construction the number aggregate_qv_alignment.py reported for the population
being displayed.

The one thing computed here is the least-squares line, which is drawing geometry
rather than a reported statistic -- it is a visual aid for the coefficient
already quoted, and nothing downstream consumes it.

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

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/vision/ilharco_timm_supervised/004_qv_alignment/vision/qv_alignment"
PLOT_ROOT = "plots/vision/ilharco_timm_supervised/004_qv_alignment/qv_alignment_scatter"

IN_FILE = "qv_alignment_delta.json"

OUTCOMES = ("delta", "delta_best", "recovery", "recovery_best")

# Outcomes that can legitimately be negative, and therefore want a zero line: a
# donor whose vector hurts its receiver is a real result, not a clipping
# artefact.
SIGNED_OUTCOMES = frozenset(OUTCOMES)

OUTCOME_TITLES = {
    "delta":         "Top-1 gain over vanilla PTQ at λ = 1 (p.p.)",
    "delta_best":    "Top-1 gain over vanilla PTQ at λ* (p.p.)",
    "recovery":      "recovery ratio Δ / Δ_ceiling at λ = 1",
    "recovery_best": "recovery ratio Δ / Δ_ceiling at λ*",
}

# Deltas are stored as fractions and reported as accuracy points; the recovery
# ratios are already dimensionless and must not be rescaled.
OUTCOME_SCALE = {
    "delta":         100.0,
    "delta_best":    100.0,
    "recovery":      1.0,
    "recovery_best": 1.0,
}

# One colour per backbone, so a pooled scatter still shows which model a point
# came from.  Same palette as the 998 lambda_curve figure.
MODEL_COLORS = [
    "#B24C3F", "#C4703E", "#D19A4A", "#8A6E4B", "#6E7F5C",
    "#2F6E8F", "#4A8FA8", "#6FA8BF",
    "#5C6E9E", "#7A5C9E", "#9E5C8A", "#4F9E7A",
]

COLOR_ZERO = "#8A8A8A"      # the outcome = 0 line
COLOR_FIT  = "#2F2F2F"      # the least-squares line


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",         required=True, type=int)
    parser.add_argument("--qat-bits",     required=True, type=int)
    parser.add_argument("--ptq-bits",     required=True, type=int)
    parser.add_argument("--granularity",  required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules", required=True, nargs="+",
                        help="One or more module names skipped during quantization "
                             "(no default: must be specified explicitly).")

    parser.add_argument("--variant",      required=True,
                        help="Similarity variant to plot, as labelled in the aggregate: "
                             "'cosine/model', 'cosine/layer/max', "
                             "'cosine/neuron/two_stage/mean', ...")
    parser.add_argument("--outcome",      required=True, choices=OUTCOMES)
    parser.add_argument("--squared",      action="store_true",
                        help="Plot the squared similarity. Proposition 1 predicts "
                             "recovery tracks cos^2, so this is the theory-faithful "
                             "form for the recovery outcomes.")

    parser.add_argument("--model-names",  default=None, nargs="+",
                        help="Restrict to these backbones. Default: every model in "
                             "the aggregate. With exactly one, the annotation quotes "
                             "that model's coefficient rather than the pooled one.")

    parser.add_argument("--in-name",      default=IN_FILE)
    parser.add_argument("--in-agg",       default=None,
                        help="The agg= fragment of the aggregate JSON to read, "
                             "with or without the prefix. Only needed when more "
                             "than one aggregation set was computed for this "
                             "checkpoint configuration.")
    parser.add_argument("--width",        type=int, default=920)
    parser.add_argument("--height",       type=int, default=620)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers (must mirror what the aggregate writes to disk)
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _run_frag(args):
    """The `seed=/qat=/ptq=` tail shared by the input and output trees.

    Below it sits the `agg=` level naming the aggregation set the correlations
    were computed over, then `alignment` -- which sits where the sibling figures
    have `split=`, because a weight-space similarity has no evaluation split to
    belong to.
    """
    return os.path.join(f"seed={args.seed}", _qat_frag(args), _ptq_frag(args))


def _in_path(args):
    """The aggregate JSON, found by globbing the `agg=` level.

    A reader does not know which aggregation set the compute script was asked
    for, and rebuilding the fragment would mean repeating four list arguments
    that this script has no other use for.  Several matches is an error rather
    than a choice: the coefficients differ between aggregation sets, and a
    scatter captioned with the wrong rho is not detectable by looking at it.

    Duplicated rather than imported from the 004 experiment directory, per the
    repo convention that visualization scripts import only from code/src.
    """
    base = os.path.join(EVAL_ROOT, _run_frag(args))

    if args.in_agg is not None:
        frag = args.in_agg if args.in_agg.startswith("agg=") else f"agg={args.in_agg}"
        path = os.path.join(base, frag, "alignment", args.in_name)
        assert os.path.exists(path), f"{path} not found (from --in-agg)"
        return frag, path

    pattern = os.path.join(base, "agg=*", "alignment", args.in_name)
    matches = sorted(glob.glob(pattern))

    assert matches, (
        f"nothing matching {pattern}. Run aggregate_qv_alignment.py first with "
        f"matching --seed/--qat-bits/--ptq-bits/--granularity/--skip-modules."
    )
    frags = [os.path.basename(os.path.dirname(os.path.dirname(m))) for m in matches]
    if len(matches) > 1:
        listing = "\n  ".join(frags)
        raise SystemExit(
            f"{len(matches)} aggregation sets under {base}; pass --in-agg with "
            f"one of:\n  {listing}"
        )
    return frags[0], matches[0]


def _out_dir(args, agg_frag):
    return os.path.join(PLOT_ROOT, _run_frag(args), agg_frag, "alignment")


# ---------------------------------------------------------------------------
# Statistics: read, never recomputed
# ---------------------------------------------------------------------------
def _stats_block(results, args):
    """The correlation block matching exactly the population being plotted.

    With one backbone selected the per-model block is quoted; otherwise the
    pooled one.  Quoting the pooled coefficient over a single-model scatter would
    caption the figure with a number computed from points it does not show.
    """
    transform = "squared" if args.squared else "raw"

    if args.model_names is not None and len(args.model_names) == 1:
        model = args.model_names[0]
        block = results["per_model"].get(model)
        assert block is not None, (
            f"{model} is not in the aggregate; it has "
            f"{sorted(results['per_model'])}"
        )
        return block[transform][args.variant][args.outcome], block["display_name"]

    return results["pooled"][transform][args.variant][args.outcome], None


def _fit_line(xs, ys):
    """Ordinary least squares, in closed form.

    Drawing geometry, not a reported statistic: the coefficient in the
    annotation is read from the aggregate.  scipy is not a dependency of this
    project and a two-parameter fit does not justify adding one.
    """
    n = len(xs)
    if n < 2:
        return None

    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None

    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx


def _fmt_stat(stat):
    if stat["r"] is None:
        return f"n/a (n={stat['n']})"
    ci = ""
    if stat["ci_lo"] is not None:
        ci = f" [{stat['ci_lo']:+.3f}, {stat['ci_hi']:+.3f}]"
    return f"{stat['r']:+.4f}{ci}"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args):
    stats, single_model = _stats_block(results, args)
    scale = OUTCOME_SCALE[args.outcome]

    keep = set(args.model_names) if args.model_names else None
    by_model = {}

    for p in results["points"]:
        if keep is not None and p["model"] not in keep:
            continue
        x = p["sims"].get(args.variant)
        y = p[args.outcome]
        # A pair with no similarity under this variant, or no usable outcome
        # (recovery is undefined where the receiver's own QAT gain is ~0), is
        # dropped here exactly as the correlation dropped it.
        if x is None or y is None:
            continue
        by_model.setdefault(p["model"], {"display": p["display_name"],
                                         "x": [], "y": [], "label": []})
        by_model[p["model"]]["x"].append(x * x if args.squared else x)
        by_model[p["model"]]["y"].append(y * scale)
        by_model[p["model"]]["label"].append(f"{p['donor']} → {p['receiver']}")

    assert by_model, (
        f"no point has both '{args.variant}' and '{args.outcome}'. "
        f"Available variants: {results['variants'][:6]} ..."
    )

    n_drawn = sum(len(b["x"]) for b in by_model.values())

    fig = go.Figure()

    for i, (model, block) in enumerate(sorted(by_model.items())):
        fig.add_trace(
            go.Scatter(
                x=block["x"], y=block["y"],
                mode="markers",
                name=f"{block['display']} (n={len(block['x'])})",
                marker=dict(
                    color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    size=7, opacity=0.75,
                    line=dict(color="white", width=0.5),
                ),
                text=block["label"],
                hovertemplate="%{text}<br>x=%{x:.4f}<br>y=%{y:.3f}<extra></extra>",
            )
        )

    all_x = [x for b in by_model.values() for x in b["x"]]
    all_y = [y for b in by_model.values() for y in b["y"]]

    fit = _fit_line(all_x, all_y)
    if fit is not None:
        slope, intercept = fit
        x_lo, x_hi = min(all_x), max(all_x)
        fig.add_trace(
            go.Scatter(
                x=[x_lo, x_hi],
                y=[slope * x_lo + intercept, slope * x_hi + intercept],
                mode="lines",
                name="least squares",
                line=dict(color=COLOR_FIT, width=1.6, dash="solid"),
                hoverinfo="skip",
            )
        )

    if args.outcome in SIGNED_OUTCOMES:
        fig.add_hline(y=0.0, line=dict(color=COLOR_ZERO, width=1, dash="dot"))

    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)

    fig.update_xaxes(
        title_text=_x_title(args),
        range=[min(all_x) - 0.06 * x_span, max(all_x) + 0.06 * x_span],
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=OUTCOME_TITLES[args.outcome],
        range=[min(all_y) - 0.08 * y_span, max(all_y) + 0.10 * y_span],
        zeroline=False,
    )

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.02,
        xanchor="left", yanchor="bottom",
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.82)", bordercolor="#CCCCCC", borderwidth=1,
        font=dict(size=11),
        text=(f"pearson  {_fmt_stat(stats['pearson'])}<br>"
              f"spearman {_fmt_stat(stats['spearman'])}<br>"
              f"n = {stats['pearson']['n']}"),
    )

    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=76, r=30, t=96, b=64),
        legend=dict(orientation="h", yanchor="top", y=-0.12,
                    xanchor="left", x=0, font=dict(size=10)),
        title=dict(text=_headline(results, args, single_model, n_drawn),
                   x=0.5, xanchor="center", font=dict(size=13)),
    )

    # The figure must show the population its coefficient was computed over.
    assert n_drawn == stats["pearson"]["n"], (
        f"drawing {n_drawn} points but the quoted correlation has "
        f"n={stats['pearson']['n']}; the figure and its caption disagree"
    )

    return fig


def _x_title(args):
    base = f"QV similarity — {args.variant}"
    return f"{base}, squared" if args.squared else base


def _headline(results, args, single_model, n_drawn):
    scope = single_model if single_model else f"{len(results['per_model'])} backbones"
    cfg = results["config"]
    return (
        f"{OUTCOME_TITLES[args.outcome]} against QV similarity<br>"
        f"<sup>{scope} | {n_drawn} cross-task donor–receiver pairs | "
        f"qat_bits={cfg['qat_bits']} | ptq_bits={cfg['ptq_bits']} | "
        f"granularity={cfg['granularity']} | seed={cfg['seed']}</sup>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    agg_frag, results_path = _in_path(args)
    with open(results_path) as f:
        results = json.load(f)

    assert args.variant in results["variants"], (
        f"'{args.variant}' is not in the aggregate. It has "
        f"{len(results['variants'])} variants, e.g. {results['variants'][:6]}"
    )

    fig = build_figure(results, args)

    out_dir = _out_dir(args, agg_frag)
    os.makedirs(out_dir, exist_ok=True)

    stem = f"scatter_{args.variant.replace('/', '_')}_{args.outcome}"
    if args.squared:
        stem = f"{stem}_squared"
    if args.model_names is not None and len(args.model_names) == 1:
        stem = f"{stem}_{args.model_names[0].replace('.', '_')}"

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
