"""004 — Does the answer depend on how the similarity was aggregated?

The result of this sub-phase is not a single coefficient but a *disagreement
between aggregations*.  The flattened model-wise similarity is uncorrelated with
what transfer actually buys, while the best-aligned single layer is not.  Those
are two different claims about the same pair of vectors, and which one gets
quoted is a choice made by an aggregation operator.

This figure makes that choice visible.  One row per (metric, granularity,
operator) variant, sorted by |Spearman| against the chosen outcome, with the
Fisher-z 95% interval drawn as an error bar so a coefficient that does not clear
zero is visibly not clearing zero.  Per-backbone markers sit on the pooled bar,
because a variant that looks strong pooled but disagrees between backbones is a
variant that found the pooling, not the signal.

Read this as a robustness display, not a menu.  A correlation that survives only
one of fifty aggregations is a correlation found by searching fifty
aggregations, and the figure is laid out so that reading it that way is the
natural one: if the top of the chart is a handful of variants barely clear of an
interval that touches zero, that is the finding.

Reads the aggregate JSON and recomputes no statistic.  Writes HTML, PDF and PNG
to plots/vision/ilharco_timm_supervised/004_qv_alignment/.
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
PLOT_ROOT = "plots/vision/ilharco_timm_supervised/004_qv_alignment/qv_alignment_operator_sensitivity"

IN_FILE = "qv_alignment_delta.json"

OUTCOMES = ("delta", "delta_best", "recovery", "recovery_best")

OUTCOME_TITLES = {
    "delta":         "Top-1 gain over vanilla PTQ at λ = 1",
    "delta_best":    "Top-1 gain over vanilla PTQ at λ*",
    "recovery":      "recovery ratio Δ / Δ_ceiling at λ = 1",
    "recovery_best": "recovery ratio Δ / Δ_ceiling at λ*",
}

COEFFICIENTS = ("spearman", "pearson")

COLOR_BAR   = "#2F6E8F"
COLOR_ZERO  = "#8A8A8A"

# One colour per backbone, matching the scatter and the 998 lambda_curve figure.
MODEL_COLORS = [
    "#B24C3F", "#C4703E", "#D19A4A", "#8A6E4B", "#6E7F5C",
    "#2F6E8F", "#4A8FA8", "#6FA8BF",
    "#5C6E9E", "#7A5C9E", "#9E5C8A", "#4F9E7A",
]


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

    parser.add_argument("--outcome",      required=True, choices=OUTCOMES)
    parser.add_argument("--transform",    required=True, choices=["raw", "squared"],
                        help="squared is the theory-faithful form: Proposition 1 "
                             "predicts recovery tracks cos^2.")
    parser.add_argument("--coefficient",  default="spearman", choices=list(COEFFICIENTS),
                        help="Spearman by default: an ordering claim survives the "
                             "local-quadratic assumption failing, a linear fit does not.")

    parser.add_argument("--metrics",      default=None, nargs="+",
                        help="Restrict to variants of these metrics. Default: all.")
    parser.add_argument("--top",          default=25, type=int,
                        help="How many variants to draw, by |coefficient|. 0 draws all.")
    parser.add_argument("--no-per-model", action="store_true",
                        help="Omit the per-backbone markers.")

    parser.add_argument("--in-name",      default=IN_FILE)
    parser.add_argument("--in-agg",       default=None,
                        help="The agg= fragment of the aggregate JSON to read, "
                             "with or without the prefix. Only needed when more "
                             "than one aggregation set was computed for this "
                             "checkpoint configuration.")
    parser.add_argument("--width",        type=int, default=1040)
    parser.add_argument("--height",       type=int, default=None,
                        help="Default: sized to the number of rows drawn.")
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
    """Everything above the `agg=` level, which is globbed rather than rebuilt."""
    return os.path.join(f"seed={args.seed}", _qat_frag(args), _ptq_frag(args))


def _in_path(args):
    """The aggregate JSON, found by globbing the `agg=` level.

    Duplicated rather than imported from the 004 experiment directory, per the
    repo convention that visualization scripts import only from code/src.
    Several matches is an error: this figure exists to compare aggregations
    against each other, so which set it drew from is not a detail it may guess.
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
# Row selection
# ---------------------------------------------------------------------------
def select_rows(results, args):
    """Variants present in the aggregate, sorted by |coefficient|, descending."""
    pooled = results["pooled"][args.transform]

    rows = []
    for variant in results["variants"]:
        if args.metrics is not None and variant.split("/")[0] not in args.metrics:
            continue

        block = pooled.get(variant, {}).get(args.outcome)
        if block is None:
            continue

        stat = block[args.coefficient]
        if stat["r"] is None:
            continue

        rows.append({"variant": variant, "stat": stat})

    assert rows, (
        f"no variant carries {args.coefficient} against {args.outcome} "
        f"({args.transform}); the aggregate has {len(results['variants'])} variants"
    )

    rows.sort(key=lambda r: -abs(r["stat"]["r"]))
    return rows if args.top <= 0 else rows[:args.top]


def _per_model_values(results, args, variant):
    """(display_name, r) for each backbone, for the same variant and outcome."""
    out = []
    for model, block in results["per_model"].items():
        stat = block[args.transform].get(variant, {}).get(args.outcome)
        if stat is None:
            continue
        r = stat[args.coefficient]["r"]
        if r is not None:
            out.append((block["display_name"], r))
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args):
    rows = select_rows(results, args)

    # Drawn bottom-up so the strongest variant lands at the top of the chart.
    rows = list(reversed(rows))
    labels = [r["variant"] for r in rows]
    values = [r["stat"]["r"] for r in rows]

    # A missing interval (n < 4, or |r| == 1) becomes a zero-length bar rather
    # than a silently absent one.
    err_plus  = [(r["stat"]["ci_hi"] - r["stat"]["r"]) if r["stat"]["ci_hi"] is not None else 0.0
                 for r in rows]
    err_minus = [(r["stat"]["r"] - r["stat"]["ci_lo"]) if r["stat"]["ci_lo"] is not None else 0.0
                 for r in rows]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values, y=labels,
            orientation="h",
            marker=dict(color=COLOR_BAR),
            error_x=dict(type="data", symmetric=False,
                         array=err_plus, arrayminus=err_minus,
                         color="#444444", thickness=1.1, width=3),
            name=f"pooled ({args.coefficient})",
            hovertemplate="%{y}<br>r=%{x:+.4f}<extra></extra>",
        )
    )

    if not args.no_per_model:
        models = sorted({m for m in results["per_model"]})
        for i, model in enumerate(models):
            display = results["per_model"][model]["display_name"]
            xs, ys = [], []
            for r in rows:
                for name, value in _per_model_values(results, args, r["variant"]):
                    if name == display:
                        xs.append(value)
                        ys.append(r["variant"])
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers",
                    name=display,
                    marker=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)],
                                size=7, symbol="diamond",
                                line=dict(color="white", width=0.6)),
                    hovertemplate=f"{display}<br>%{{y}}<br>r=%{{x:+.4f}}<extra></extra>",
                )
            )

    fig.add_vline(x=0.0, line=dict(color=COLOR_ZERO, width=1, dash="dot"))

    fig.update_xaxes(title_text=f"{args.coefficient} correlation with "
                                f"{OUTCOME_TITLES[args.outcome]}",
                     zeroline=False)
    fig.update_yaxes(title_text="", automargin=True, tickfont=dict(size=9))

    cfg = results["config"]
    n_pooled = results["pooled"]["n_pairs"]
    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height or max(420, 22 * len(rows) + 190),
        margin=dict(l=20, r=40, t=104, b=64),
        bargap=0.28,
        legend=dict(orientation="h", yanchor="top", y=-0.06,
                    xanchor="left", x=0, font=dict(size=10)),
        title=dict(
            text=(f"Does the answer depend on the aggregation? "
                  f"({'squared' if args.transform == 'squared' else 'raw'} similarity)<br>"
                  f"<sup>{len(rows)} of {len(results['variants'])} variants | "
                  f"{len(results['per_model'])} backbones | up to {n_pooled} "
                  f"cross-task pairs | error bars are Fisher-z 95% intervals | "
                  f"qat_bits={cfg['qat_bits']} ptq_bits={cfg['ptq_bits']} "
                  f"seed={cfg['seed']}</sup>"),
            x=0.5, xanchor="center", font=dict(size=13),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    agg_frag, results_path = _in_path(args)
    with open(results_path) as f:
        results = json.load(f)

    fig = build_figure(results, args)

    out_dir = _out_dir(args, agg_frag)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"operator_sensitivity_{args.outcome}_{args.transform}_{args.coefficient}"

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
