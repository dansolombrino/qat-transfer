"""998 — Donor benefit against donor cost

One marker per donor dataset, from the donor_benefit.json written by
compute_donor_benefit.py:

    x   the compute spent producing that donor's quantization vector, i.e. one
        QAT run over the donor's training set, in millions of training samples.
    y   how many of the other tasks that single QV helps, averaged over the
        backbones of the family, with the across-backbone range as error bars.

Good donors sit top-left: cheap to produce, useful to many receivers.  The slope
of the line from the origin to a marker is the amortized price of one successful
deployment, so dashed iso-lines of constant cost-per-win are drawn across the
plot and the Pareto frontier (no other donor is both cheaper and better) is
highlighted.

Writes an HTML figure to plots/998_rebuttal/002_cost_amortization/.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import os

os.chdir(_PROJECT_ROOT)

import argparse
import json

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/002_cost_amortization"
PLOT_ROOT = "plots/998_rebuttal/002_cost_amortization"

COLOR_PARETO = "#2F6E8F"     # Pareto-optimal donors
COLOR_OTHER = "#9AA5AC"      # dominated donors
COLOR_OTHER_ERR = "rgba(154, 165, 172, 0.40)"
COLOR_ISO = "#B24C3F"        # iso-cost-per-win lines
COLOR_FRONT = "rgba(47, 110, 143, 0.45)"

SAMPLES_PER_UNIT = 1e6       # axes are reported in millions of samples
UNIT_LABEL = "M training samples"

ISO_LINES = [0.01, 0.02, 0.04, 0.08]   # M training samples per receiver helped


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-name",
        default="donor_benefit.json",
        help="File read from evaluations/998_rebuttal/002_cost_amortization/.",
    )
    parser.add_argument(
        "--label-all",
        action="store_true",
        help="Label every donor, not just ImageNet and the Pareto-optimal ones.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Log-scale the cost axis (the donors bunch up under ImageNet otherwise).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF next to the HTML (needs kaleido).",
    )
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=560)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scaled(value):
    return value / SAMPLES_PER_UNIT


def _hover(d):
    return (
        f"<b>{d['donor']}</b><br>"
        f"QAT cost: {_scaled(d['cost_samples']):.3f} {UNIT_LABEL}"
        f" ({d['epochs']} epochs x {d['train_size']:,})<br>"
        f"helps {d['mean_wins']:.1f} / {d['n_receivers']} receivers"
        f" (range {d['min_wins']}-{d['max_wins']} across backbones)<br>"
        f"win rate: {d['win_rate'] * 100:.1f}%<br>"
        f"mean gain: {d['mean_delta'] * 100:+.2f} p.p.<br>"
        f"cost per win: {_scaled(d['cost_per_win']):.3f} {UNIT_LABEL}"
        "<extra></extra>"
    )


def _label_positions(group, donors):
    """Nudge labels apart: donors that crowd on the cost axis alternate above/below."""
    x_span = max(o["cost_samples"] for o in donors)
    threshold = 0.08 * x_span
    ordered = sorted(group, key=lambda d: d["cost_samples"])
    xs = [d["cost_samples"] for d in ordered]

    positions = {}
    flip = False
    for i, d in enumerate(ordered):
        crowded = (i > 0 and xs[i] - xs[i - 1] < threshold) or (
            i + 1 < len(xs) and xs[i + 1] - xs[i] < threshold
        )
        if d["cost_samples"] >= 0.6 * x_span:
            positions[d["donor"]] = "middle left"
        elif crowded:
            positions[d["donor"]] = "bottom center" if flip else "top center"
            flip = not flip
        else:
            positions[d["donor"]] = "top center"
    return [positions[d["donor"]] for d in group]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args):
    donors = [d for d in results["donors"] if d["cost_samples"] is not None]
    assert donors, "no donor has a cost entry; check dataset_sizes.json"

    pareto = sorted(
        (d for d in donors if d["pareto_optimal"]), key=lambda d: d["cost_samples"]
    )
    others = [d for d in donors if not d["pareto_optimal"]]

    x_max = _scaled(max(d["cost_samples"] for d in donors)) * 1.18
    y_max = max(d["max_wins"] for d in donors) * 1.10
    n_receivers = donors[0]["n_receivers"]

    fig = go.Figure()

    ############################################################################
    # BEGIN iso-cost-per-win lines
    ############################################################################
    for cost_per_win in ISO_LINES:
        # wins = cost / cost_per_win, a ray through the origin.
        x_end = min(x_max, y_max * cost_per_win)
        y_end = x_end / cost_per_win
        fig.add_trace(
            go.Scatter(
                x=[0, x_end],
                y=[0, y_end],
                mode="lines",
                line=dict(color=COLOR_ISO, width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=x_end,
            y=y_end,
            text=f"{cost_per_win * 1000:.0f}k / win",
            showarrow=False,
            xanchor="left" if x_end < x_max * 0.98 else "right",
            yanchor="bottom",
            font=dict(size=10, color=COLOR_ISO),
        )
    # One legend entry standing in for the whole iso-line family.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=COLOR_ISO, width=1, dash="dot"),
            name="constant cost per receiver helped",
        )
    )
    ############################################################################
    # END iso-cost-per-win lines
    ############################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Pareto frontier
    ############################################################################
    fig.add_trace(
        go.Scatter(
            x=[_scaled(d["cost_samples"]) for d in pareto],
            y=[d["mean_wins"] for d in pareto],
            mode="lines",
            line=dict(color=COLOR_FRONT, width=2, shape="hv"),
            hoverinfo="skip",
            name="Pareto frontier",
        )
    )
    ############################################################################
    # END Pareto frontier
    ############################################################################

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN donor markers
    ############################################################################
    for group, color, err_color, name, size in (
        (others, COLOR_OTHER, COLOR_OTHER_ERR, "dominated donors", 9),
        (pareto, COLOR_PARETO, COLOR_PARETO, "Pareto-optimal donors", 13),
    ):
        if not group:
            continue
        labelled = [
            d["donor"] if (args.label_all or d["pareto_optimal"] or d["donor"] == "ImageNet")
            else ""
            for d in group
        ]
        fig.add_trace(
            go.Scatter(
                x=[_scaled(d["cost_samples"]) for d in group],
                y=[d["mean_wins"] for d in group],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[d["max_wins"] - d["mean_wins"] for d in group],
                    arrayminus=[d["mean_wins"] - d["min_wins"] for d in group],
                    color=err_color,
                    thickness=1,
                    width=3,
                ),
                mode="markers+text",
                marker=dict(color=color, size=size, line=dict(color="white", width=1)),
                text=labelled,
                textposition=_label_positions(group, donors),
                textfont=dict(size=10, color=color),
                hovertemplate=[_hover(d) for d in group],
                name=name,
            )
        )
    ############################################################################
    # END donor markers
    ############################################################################

    fig.update_xaxes(
        title_text=f"cost of the donor's QAT run ({UNIT_LABEL})",
        range=None if args.log_x else [0, x_max],
        type="log" if args.log_x else "linear",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=f"receivers helped (of {n_receivers}), mean over backbones",
        range=[0, y_max],
        zeroline=False,
    )

    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=80, r=40, t=100, b=110),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
        title=dict(text=_headline(results, donors), x=0.5, xanchor="center",
                   font=dict(size=14)),
    )
    fig.add_annotation(
        text=(
            "Up and to the left is better. The slope from the origin to a marker is the "
            "amortized cost of one successful deployment."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.30,
        showarrow=False,
        xanchor="left",
        align="left",
        font=dict(size=11, color="#5A5A5A"),
    )

    return fig


def _headline(results, donors):
    """State the finding in words: what the extra ImageNet compute actually buys."""
    by_cost = sorted(donors, key=lambda d: d["cost_samples"])
    cheapest = by_cost[0]
    priciest = by_cost[-1]
    best = max(donors, key=lambda d: d["mean_wins"])
    ratio = priciest["cost_samples"] / cheapest["cost_samples"]
    cpw_ratio = priciest["cost_per_win"] / cheapest["cost_per_win"]
    return (
        f"What one donor QAT run buys, per receiver it helps<br>"
        f"<sup>{best['donor']} helps the most receivers "
        f"({best['mean_wins']:.1f}/{best['n_receivers']}), but costs "
        f"{cpw_ratio:.1f}x more per win than {cheapest['donor']}, "
        f"which is {ratio:.1f}x cheaper to produce and still helps "
        f"{cheapest['mean_wins']:.1f}/{cheapest['n_receivers']}</sup>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    results_path = os.path.join(EVAL_ROOT, args.in_name)
    assert os.path.exists(results_path), (
        f"{results_path} not found. Run compute_donor_benefit.py first."
    )
    with open(results_path) as f:
        results = json.load(f)

    fig = build_figure(results, args)

    os.makedirs(PLOT_ROOT, exist_ok=True)
    stem = f"donor_benefit_scatter_{results['family']}_{results['settings']['alpha_regime']}"
    html_path = os.path.join(PLOT_ROOT, f"{stem}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path}")

    if args.pdf:
        pdf_path = os.path.join(PLOT_ROOT, f"{stem}.pdf")
        fig.write_image(pdf_path)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
