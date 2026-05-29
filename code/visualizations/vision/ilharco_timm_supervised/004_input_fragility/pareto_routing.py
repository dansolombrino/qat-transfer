"""Script C — input-aware mixed-precision routing Pareto curves.

For each task, fit four routing scores on the parquets produced by Script A,
then sweep the routing fraction X from 0% (all PTQ) to 100% (all FP). At each
X, top-X% of test samples by score are served from FP and the remainder from
PTQ; the resulting test accuracy is the routed accuracy at that fraction.

Routing scores compared:
  - **oracle**       : `fp_correct − q_correct`. Upper bound at every X.
  - **multivariate** : `P(bad)` from a StandardScaler + LogisticRegression
                       on val (FP-correct rows, positive=bad), evaluated on
                       all test rows. The Script B headline predictor.
  - **margin_only**  : −`fp_margin`. The 1-feature simplification.
  - **random**       : uniform random shuffle (seed=0). Linear-interpolation
                       baseline; bounds how much value the predictor adds.

Headline question: at what FP-compute-fraction X does the predictor recover
~all of the FP→PTQ gap? Answer is the central figure of the paper.

Outputs:
  - Markdown summary (per-task + aggregate) to stdout and to plots/.
  - HTML report: per-task Pareto curves + an aggregate plot showing
    mean ± 25th/75th-percentile bands per routing method.

Argparse, no Hydra, no GPU. ~1 min on the 18 W4-channel dumps we have.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
os.chdir(_PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.vision.utils import sanitize_timm_model_name


PROPERTY_COLUMNS = [
    # FP-pass difficulty proxies
    "fp_margin",
    "fp_softmax_top1",
    "fp_entropy",
    "fp_cls_dist_to_class_centroid",
    # Q-pass difficulty proxies
    "q_margin",
    "q_softmax_top1",
    "q_entropy",
    # Cross-model (FP <-> Q) features
    "fp_logit_at_q_pred",
    "q_logit_at_fp_pred",
    "fp_softmax_at_q_pred",
    "q_softmax_at_fp_pred",
    "fp_q_kl_symmetric",
    "fp_q_disagree",
    # Raw image statistics
    "img_brightness",
    "img_contrast",
    "img_edge_density",
    "img_high_freq_ratio",
]

METHOD_NAMES = ["oracle", "multivariate", "margin_only", "random"]
METHOD_COLORS = {
    "oracle":       "#000000",
    "multivariate": "#1f77b4",
    "margin_only":  "#ff7f0e",
    "random":       "#7f7f7f",
}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

CANONICAL_FRACTIONS = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", required=True)
    p.add_argument("--lr", default="1e-05")
    p.add_argument("--wd", default="0.1")
    p.add_argument("--ls", default="0.0")
    p.add_argument("--wl", default="500")
    p.add_argument("--max-grad-norm", default="1.0")
    p.add_argument("--batch-size", default="128")
    p.add_argument("--seed", default="2038")
    p.add_argument("--bits", required=True, type=int)
    p.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    p.add_argument("--skip-modules", nargs="+", default=["head"])
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--min-bad-test", type=int, default=10,
                   help="Skip tasks with fewer than this many bad test samples.")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _build_paths(args):
    checkpoint_base = Path(os.environ["CHECKPOINT_BASE_PATH"])
    sanitized = sanitize_timm_model_name(args.model_name)
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    optim_tag = (
        f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
        f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
    )
    ptq_tag = f"ptq=bits={args.bits}_gran={args.granularity}_skip={skip_tag}"
    seed_tag = f"seed={args.seed}"
    return checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag


def _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag) -> list[str]:
    base = (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "input_fragility_dumps" / sanitized
    )
    if not base.exists():
        return []
    out = []
    for ds_dir in sorted(base.iterdir()):
        if not ds_dir.is_dir():
            continue
        f = ds_dir / optim_tag / ptq_tag / seed_tag / "predictions_test.parquet"
        if f.exists():
            out.append(ds_dir.name)
    return out


def _load_task(checkpoint_base, sanitized, dataset, optim_tag, ptq_tag, seed_tag):
    d = (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "input_fragility_dumps" / sanitized / dataset
        / optim_tag / ptq_tag / seed_tag
    )
    return (
        pd.read_parquet(d / "predictions_val.parquet"),
        pd.read_parquet(d / "predictions_test.parquet"),
    )


def _fit_multivariate(df_val: pd.DataFrame):
    """Fit on FP-correct val rows; positive class = bad. Returns (scaler, clf, val_medians)."""
    vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
    cols = [c for c in PROPERTY_COLUMNS if c in vf.columns]
    X = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
    # Impute NaN with column-median computed on val (mostly affects
    # fp_cls_dist_to_class_centroid for tasks with empty val classes).
    val_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(val_medians, inds[1])
    y = vf["bad"].astype(int).to_numpy()

    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None, None, None, cols

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(scaler.transform(X), y)
    return scaler, clf, val_medians, cols


def _scores_for_test(df_test: pd.DataFrame, scaler, clf, val_medians, cols, seed: int = 0) -> dict:
    n = len(df_test)
    fp_correct = df_test["fp_correct"].to_numpy(dtype=bool)
    q_correct = df_test["q_correct"].to_numpy(dtype=bool)

    # Oracle: benefit of routing to FP = fp_correct - q_correct  (+1 / 0 / -1)
    oracle = fp_correct.astype(int) - q_correct.astype(int)

    # Multivariate
    if scaler is None:
        multivariate = np.zeros(n)
    else:
        X = np.array(df_test[cols].to_numpy(dtype=np.float64), copy=True)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(val_medians, inds[1])
        multivariate = clf.predict_proba(scaler.transform(X))[:, 1]

    # Margin-only: lower margin → more "bad" → higher routing score
    margin_only = -df_test["fp_margin"].to_numpy(dtype=np.float64)

    # Random (deterministic seed for reproducibility)
    rng = np.random.default_rng(seed)
    random_score = rng.random(n)

    return {
        "oracle": oracle.astype(np.float64),
        "multivariate": multivariate,
        "margin_only": margin_only,
        "random": random_score,
    }


def _routed_curve(scores: np.ndarray, fp_correct: np.ndarray, q_correct: np.ndarray, rng_tie=None):
    """Return (fractions, routed_accuracies) of length N+1 where fractions[k] = k/N
    and accuracies[k] is the test accuracy if top-k by score are routed to FP."""
    n = len(scores)
    # Deterministic tie-breaking: break ties with a fixed random order so different
    # scores with the same value get a stable ordering.
    if rng_tie is None:
        rng_tie = np.random.default_rng(12345)
    tie = rng_tie.random(n)
    order = np.lexsort((tie, -scores))  # primary key descending scores; ties break by tie

    fp_sorted = fp_correct[order].astype(np.int64)
    q_sorted = q_correct[order].astype(np.int64)

    fp_cum = np.concatenate(([0], np.cumsum(fp_sorted)))  # length n+1
    q_cum = np.concatenate(([0], np.cumsum(q_sorted)))
    total_q = q_cum[-1]
    # At k routed to FP: correct = fp_cum[k] + (total_q - q_cum[k])
    correct = fp_cum + (total_q - q_cum)
    accuracies = correct.astype(np.float64) / n
    fractions = np.arange(n + 1) / n
    return fractions, accuracies


def _sample_at(fractions: np.ndarray, accuracies: np.ndarray, target_fracs):
    """Linear-interpolate the curve at target fractions in [0, 1]."""
    return np.interp(target_fracs, fractions, accuracies)


def _per_task_block(task: str, plain_ptq: float, plain_fp: float,
                    method_curves: dict, n_bad_test: int) -> str:
    """Markdown summary block for a single task."""
    drop = plain_fp - plain_ptq
    out = [f"### {task}", ""]
    out.append(f"- Test set: plain PTQ = {plain_ptq*100:.2f}%, plain FP = {plain_fp*100:.2f}%, gap = {drop*100:+.2f}pp, n_bad_test = {n_bad_test}")
    out.append("")
    header = "| X (FP fraction) | " + " | ".join(METHOD_NAMES) + " |"
    sep = "|" + "---|" * (1 + len(METHOD_NAMES))
    out.append(header)
    out.append(sep)
    for X in CANONICAL_FRACTIONS:
        cells = []
        for m in METHOD_NAMES:
            frac, acc = method_curves[m]
            a = float(_sample_at(frac, acc, [X])[0])
            if drop > 1e-9:
                gain_pct_of_gap = (a - plain_ptq) / drop * 100
                cells.append(f"{a*100:.2f}% ({gain_pct_of_gap:+.0f}% of gap)")
            else:
                cells.append(f"{a*100:.2f}%")
        out.append(f"| {X*100:>5.0f}% | " + " | ".join(cells) + " |")
    out.append("")
    # Find X needed to recover 90% / 95% / 99% of the gap with multivariate
    if drop > 1e-9:
        frac_m, acc_m = method_curves["multivariate"]
        gap_recovered_at = (acc_m - plain_ptq) / drop
        for target in (0.9, 0.95, 0.99):
            idx = np.searchsorted(gap_recovered_at, target)
            if idx < len(frac_m):
                out.append(f"- multivariate reaches {int(target*100)}% gap recovery at X = {frac_m[idx]*100:.1f}%")
            else:
                out.append(f"- multivariate does not reach {int(target*100)}% gap recovery (max = {gap_recovered_at[-1]*100:.1f}%)")
        out.append("")
    return "\n".join(out)


def _aggregate_block(per_task: dict) -> str:
    """Aggregate summary across tasks: mean recovery at canonical fractions + per-method
    fraction needed for 90% / 95% / 99% gap recovery."""
    out = ["", "## Aggregate across tasks", "", "### Mean gap-recovery (% of FP→PTQ gap) at canonical FP-compute fractions", ""]
    out.append("| X (FP fraction) | " + " | ".join(METHOD_NAMES) + " |")
    out.append("|" + "---|" * (1 + len(METHOD_NAMES)))
    for X in CANONICAL_FRACTIONS:
        cells = []
        for m in METHOD_NAMES:
            recovery_pcs = []
            for task, info in per_task.items():
                drop = info["fp_test"] - info["q_test"]
                if drop <= 1e-9:
                    continue
                frac, acc = info["curves"][m]
                a = float(_sample_at(frac, acc, [X])[0])
                recovery_pcs.append((a - info["q_test"]) / drop * 100)
            if recovery_pcs:
                cells.append(f"{np.mean(recovery_pcs):+.1f}% ± {np.std(recovery_pcs):.1f}")
            else:
                cells.append("—")
        out.append(f"| {X*100:>5.0f}% | " + " | ".join(cells) + " |")
    out.append("")

    out.append("### FP-compute fraction X needed to recover 90% / 95% / 99% of the gap (per method, mean across tasks)")
    out.append("")
    out.append("| method | mean X@90% | mean X@95% | mean X@99% |")
    out.append("|---|---|---|---|")
    for m in METHOD_NAMES:
        rows_for_targets = {0.9: [], 0.95: [], 0.99: []}
        for task, info in per_task.items():
            drop = info["fp_test"] - info["q_test"]
            if drop <= 1e-9:
                continue
            frac, acc = info["curves"][m]
            rec = (acc - info["q_test"]) / drop
            for tgt in rows_for_targets:
                idx = np.searchsorted(rec, tgt)
                if idx < len(frac):
                    rows_for_targets[tgt].append(frac[idx])
        cells = []
        for tgt in (0.9, 0.95, 0.99):
            vals = rows_for_targets[tgt]
            if vals:
                cells.append(f"{np.mean(vals)*100:.1f}% ± {np.std(vals)*100:.1f}")
            else:
                cells.append("—")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def _render_html(per_task: dict, out_html: Path, title: str):
    tasks = list(per_task.keys())
    n_tasks = len(tasks)

    # 1) Aggregate panel: mean ± IQR per method across tasks
    grid_fracs = np.linspace(0, 1, 201)
    per_method_curves = {m: [] for m in METHOD_NAMES}
    for task, info in per_task.items():
        drop = info["fp_test"] - info["q_test"]
        if drop <= 1e-9:
            continue
        for m in METHOD_NAMES:
            frac, acc = info["curves"][m]
            rec = (acc - info["q_test"]) / drop
            per_method_curves[m].append(_sample_at(frac, rec, grid_fracs))
    per_method_curves = {m: np.stack(c, axis=0) for m, c in per_method_curves.items() if c}

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Aggregate Pareto: mean gap-recovery vs FP-compute fraction (band = 25th/75th percentile across tasks)",
            "Per-task multivariate routing curves",
        ),
        row_heights=[0.55, 0.45],
        vertical_spacing=0.13,
    )

    for m in METHOD_NAMES:
        if m not in per_method_curves:
            continue
        curves = per_method_curves[m]
        mean = curves.mean(axis=0)
        lo = np.percentile(curves, 25, axis=0)
        hi = np.percentile(curves, 75, axis=0)
        color = METHOD_COLORS[m]
        # IQR band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid_fracs, grid_fracs[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False, name=f"{m} IQR",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=grid_fracs, y=mean, mode="lines",
                       name=m, line=dict(color=color, width=2.5)),
            row=1, col=1,
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="grey", row=1, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="grey", row=1, col=1)
    fig.update_xaxes(title_text="FP-compute fraction X", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="gap recovery (1.0 = full FP)", row=1, col=1, range=[-0.1, 1.1])

    # 2) Per-task multivariate curves (gap-recovery, faded)
    for task, info in per_task.items():
        drop = info["fp_test"] - info["q_test"]
        if drop <= 1e-9:
            continue
        frac, acc = info["curves"]["multivariate"]
        rec = (acc - info["q_test"]) / drop
        fig.add_trace(
            go.Scatter(x=frac, y=rec, mode="lines", name=task,
                       line=dict(width=1)),
            row=2, col=1,
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="grey", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dot", line_color="grey", row=2, col=1)
    fig.update_xaxes(title_text="FP-compute fraction X", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="gap recovery", row=2, col=1, range=[-0.2, 1.1])

    fig.update_layout(title=title, height=950, showlegend=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))


def main() -> None:
    args = parse_args()
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = _build_paths(args)

    datasets = args.datasets or _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
    if not datasets:
        print(f"No dumps under {checkpoint_base}/.../input_fragility_dumps/{sanitized}/", file=sys.stderr)
        sys.exit(1)

    print(f"Building Pareto curves for {len(datasets)} task(s) at W{args.bits}-{args.granularity}\n")

    per_task: dict[str, dict] = {}
    md = [f"# Input-aware mixed-precision routing — {args.model_name} | W{args.bits} {args.granularity}\n"]
    md.append(f"For each task, fit a logistic-regression predictor on val ({len(PROPERTY_COLUMNS)} features), "
              f"sort test inputs by predicted P(bad), and sweep the FP-compute fraction X. "
              f"At each X, the top X% by score go to FP; the rest are served from PTQ.\n")
    md.append("Recovery percentages are relative to the FP→PTQ gap: 0% = plain PTQ, 100% = plain FP.\n")

    for ds in datasets:
        df_val, df_test = _load_task(checkpoint_base, sanitized, ds, optim_tag, ptq_tag, seed_tag)
        n_bad_test = int(df_test["bad"].sum())
        if n_bad_test < args.min_bad_test:
            print(f"  skip {ds}: n_bad_test = {n_bad_test} < {args.min_bad_test}")
            continue

        scaler, clf, val_medians, cols = _fit_multivariate(df_val)
        scores = _scores_for_test(df_test, scaler, clf, val_medians, cols)
        fp_correct = df_test["fp_correct"].to_numpy(dtype=bool)
        q_correct = df_test["q_correct"].to_numpy(dtype=bool)

        # Compute the four Pareto curves
        curves = {}
        for m in METHOD_NAMES:
            curves[m] = _routed_curve(scores[m], fp_correct, q_correct,
                                      rng_tie=np.random.default_rng(0))

        fp_test_acc = float(fp_correct.mean())
        q_test_acc = float(q_correct.mean())
        per_task[ds] = {
            "fp_test": fp_test_acc,
            "q_test": q_test_acc,
            "curves": curves,
            "n_bad_test": n_bad_test,
        }

        md.append(_per_task_block(ds, q_test_acc, fp_test_acc, curves, n_bad_test))

    md.append(_aggregate_block(per_task))
    markdown = "\n".join(md)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "004_input_fragility"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"pareto_routing_{sanitized}_bits{args.bits}_{args.granularity}.md"
    html_path = out_dir / f"pareto_routing_{sanitized}_bits{args.bits}_{args.granularity}.html"
    md_path.write_text(markdown)
    _render_html(
        per_task, html_path,
        title=f"Pareto routing curves — {args.model_name} W{args.bits} {args.granularity} ({len(per_task)} tasks)",
    )

    print(markdown)
    print()
    print(f"Markdown saved: {md_path}")
    print(f"HTML saved:     {html_path}")


if __name__ == "__main__":
    main()
