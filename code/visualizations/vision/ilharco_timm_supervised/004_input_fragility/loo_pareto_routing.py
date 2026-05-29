"""Script D — leave-one-out cross-task Pareto routing.

Tests whether the F3 predictor — trained on a single task's val split with
the 15-feature LogReg — generalises across tasks. Setup:

  For each candidate target task t:
    1. Train the LogReg on the val FP-correct rows of every OTHER task
       (pooled, per-task-standardised, leak-features included).
    2. Score t's test rows with that LOO predictor.
    3. Compute the Pareto routing curve (oracle, LOO, same-task, margin_only,
       random). The same-task predictor is the F3 baseline — it was trained
       on t's own val.
    4. Report X@90% / X@95% / X@99% under each method.

Aggregate: how does the LOO Pareto compare to the same-task Pareto? If
LOO X@90% ≈ same-task X@90%, a single classifier transfers across tasks —
the strongest practical claim for the paper. If LOO X@90% degrades
significantly (say > 25%), the predictor is task-specific and the
deployment story tightens to "per-task fragility predictor."

Argparse, no Hydra, no GPU. ~1–2 min on the 18 W4-channel dumps.
"""

import argparse
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

from src.vision.utils import sanitize_timm_model_name


PROPERTY_COLUMNS = [
    # FP-pass
    "fp_margin",
    "fp_softmax_top1",
    "fp_entropy",
    "fp_cls_dist_to_class_centroid",
    # Q-pass
    "q_margin",
    "q_softmax_top1",
    "q_entropy",
    # Cross-model (target-leak within FP-correct; informative at test time)
    "fp_logit_at_q_pred",
    "q_logit_at_fp_pred",
    "fp_softmax_at_q_pred",
    "q_softmax_at_fp_pred",
    "fp_q_kl_symmetric",
    "fp_q_disagree",
    # Image
    "img_brightness",
    "img_contrast",
    "img_edge_density",
    "img_high_freq_ratio",
]

CANONICAL_FRACTIONS = [0.00, 0.05, 0.10, 0.16, 0.25, 0.50, 1.00]
METHODS_REPORTED = ["oracle", "loo", "same_task", "margin_only", "random"]
METHOD_COLORS = {
    "oracle":       "#000000",
    "loo":          "#d62728",
    "same_task":    "#1f77b4",
    "margin_only":  "#ff7f0e",
    "random":       "#7f7f7f",
}


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
    p.add_argument("--min-bad-test", type=int, default=10)
    p.add_argument("--min-bad-val", type=int, default=10)
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
        if (ds_dir / optim_tag / ptq_tag / seed_tag / "predictions_test.parquet").exists():
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


def _per_task_standardise(X: np.ndarray):
    """Per-feature z-score, NaN-safe. Returns (X_norm, mu, sigma) where
    NaN cells in X become 0 post-standardisation."""
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    X_norm = (X - mu) / sigma_safe
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    return X_norm.astype(np.float64), mu, sigma_safe


def _apply_standardise(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X_norm = (X - mu) / sigma
    return np.where(np.isnan(X_norm), 0.0, X_norm).astype(np.float64)


def _routed_curve(scores: np.ndarray, fp_correct: np.ndarray, q_correct: np.ndarray):
    n = len(scores)
    rng_tie = np.random.default_rng(12345)
    tie = rng_tie.random(n)
    order = np.lexsort((tie, -scores))

    fp_sorted = fp_correct[order].astype(np.int64)
    q_sorted = q_correct[order].astype(np.int64)
    fp_cum = np.concatenate(([0], np.cumsum(fp_sorted)))
    q_cum = np.concatenate(([0], np.cumsum(q_sorted)))
    total_q = q_cum[-1]
    correct = fp_cum + (total_q - q_cum)
    accuracies = correct.astype(np.float64) / n
    fractions = np.arange(n + 1) / n
    return fractions, accuracies


def _sample_at(fractions, accuracies, target_fracs):
    return np.interp(target_fracs, fractions, accuracies)


def _x_for_target_recovery(fractions, recoveries, target: float):
    """Smallest X at which recovery reaches `target` (1.0 = full FP).
    Returns nan if the curve never reaches the target."""
    above = np.where(recoveries >= target)[0]
    if len(above) == 0:
        return float("nan")
    return float(fractions[above[0]])


def main() -> None:
    args = parse_args()
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = _build_paths(args)
    datasets = args.datasets or _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
    if not datasets:
        print(f"No dumps under {checkpoint_base}/.../input_fragility_dumps/{sanitized}/", file=sys.stderr)
        sys.exit(1)

    # 1. Load every task's parquets once
    print(f"Loading parquets for {len(datasets)} task(s) …")
    task_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for ds in datasets:
        task_data[ds] = _load_task(checkpoint_base, sanitized, ds, optim_tag, ptq_tag, seed_tag)
    cols = [c for c in PROPERTY_COLUMNS if c in task_data[datasets[0]][0].columns]
    print(f"Using {len(cols)} features\n")

    # 2. Pre-compute, per task, the per-task-standardised val FP-correct
    # arrays (for pooling into LOO training sets) and the val statistics
    # (for standardising that task's own test at apply time).
    per_task_train: dict[str, dict] = {}
    for ds, (df_val, df_test) in task_data.items():
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        n_bad_val = int(vf["bad"].sum())
        n_bad_test = int(df_test["bad"].sum())
        if n_bad_val < args.min_bad_val or n_bad_test < args.min_bad_test:
            print(f"  skip {ds}: n_bad_val={n_bad_val}, n_bad_test={n_bad_test}")
            continue
        X_val = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        y_val = vf["bad"].astype(int).to_numpy()
        X_val_norm, mu, sigma = _per_task_standardise(X_val)
        per_task_train[ds] = {
            "X_norm": X_val_norm,
            "y": y_val,
            "mu": mu,
            "sigma": sigma,
        }

    eligible_tasks = list(per_task_train.keys())
    print(f"{len(eligible_tasks)} eligible target tasks\n")

    # 3. For each target task: pool the others' standardised val rows,
    # fit a LogReg, score target's test, build curves.
    per_task_results: dict[str, dict] = {}
    for target in eligible_tasks:
        df_val_tgt, df_test_tgt = task_data[target]
        fp_correct_test = df_test_tgt["fp_correct"].to_numpy(dtype=bool)
        q_correct_test = df_test_tgt["q_correct"].to_numpy(dtype=bool)

        # ----- LOO predictor: pool source-task train sets -----
        Xs = [per_task_train[t]["X_norm"] for t in eligible_tasks if t != target]
        ys = [per_task_train[t]["y"] for t in eligible_tasks if t != target]
        X_loo = np.concatenate(Xs, axis=0)
        y_loo = np.concatenate(ys, axis=0)
        if y_loo.sum() < 5 or (len(y_loo) - y_loo.sum()) < 5:
            print(f"  skip {target}: pooled source pool degenerate")
            continue

        clf_loo = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)
        clf_loo.fit(X_loo, y_loo)

        # Score target's test rows using the target's OWN val statistics (the
        # realistic deployment normalisation — the target task has its own
        # val set we can compute mu/sigma from).
        X_test_tgt = np.array(df_test_tgt[cols].to_numpy(dtype=np.float64), copy=True)
        mu_tgt = per_task_train[target]["mu"]
        sigma_tgt = per_task_train[target]["sigma"]
        X_test_tgt_norm = _apply_standardise(X_test_tgt, mu_tgt, sigma_tgt)
        loo_scores = clf_loo.predict_proba(X_test_tgt_norm)[:, 1]

        # ----- Same-task predictor (F3 baseline): fit on target's own val -----
        clf_same = LogisticRegression(max_iter=500, class_weight="balanced")
        clf_same.fit(per_task_train[target]["X_norm"], per_task_train[target]["y"])
        same_task_scores = clf_same.predict_proba(X_test_tgt_norm)[:, 1]

        # ----- Other baselines -----
        oracle_scores = fp_correct_test.astype(int) - q_correct_test.astype(int)
        margin_scores = -df_test_tgt["fp_margin"].to_numpy(dtype=np.float64)
        random_scores = np.random.default_rng(0).random(len(df_test_tgt))

        scores_by_method = {
            "oracle": oracle_scores.astype(np.float64),
            "loo": loo_scores,
            "same_task": same_task_scores,
            "margin_only": margin_scores,
            "random": random_scores,
        }
        curves = {
            m: _routed_curve(s, fp_correct_test, q_correct_test)
            for m, s in scores_by_method.items()
        }

        per_task_results[target] = {
            "fp_test": float(fp_correct_test.mean()),
            "q_test": float(q_correct_test.mean()),
            "curves": curves,
        }
        print(f"  done: {target} "
              f"(fp_test={per_task_results[target]['fp_test']*100:.1f}% "
              f"q_test={per_task_results[target]['q_test']*100:.1f}%)")

    # 4. Per-task summary + aggregate
    md = [f"# LOO cross-task Pareto routing — {args.model_name} | W{args.bits} {args.granularity}\n"]
    md.append(f"For each target task, the LogReg is fit on the **pooled val FP-correct rows of the other "
              f"{len(per_task_results) - 1} tasks**, per-task-standardised. The `same_task` row is the F3 "
              f"baseline (fit on target's own val).\n")

    # Per-task X@target tables
    md.append("## Per-task X@target recovery (% FP-compute needed to reach the target gap-recovery)\n")
    md.append("| target | gap (pp) | FP test% | PTQ test% | X@90% (loo) | X@90% (same) | X@90% (oracle) | X@90% (margin) | X@90% (random) |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    rows_table = []
    for target, info in per_task_results.items():
        gap = info["fp_test"] - info["q_test"]
        cells = [target, f"{gap*100:+.2f}", f"{info['fp_test']*100:.2f}", f"{info['q_test']*100:.2f}"]
        x90 = {}
        for m in METHODS_REPORTED:
            frac, acc = info["curves"][m]
            if gap > 1e-9:
                rec = (acc - info["q_test"]) / gap
                x = _x_for_target_recovery(frac, rec, 0.9)
            else:
                x = float("nan")
            x90[m] = x
            cells.append(f"{x*100:.1f}%" if not np.isnan(x) else "—")
        md.append("| " + " | ".join(cells) + " |")
        rows_table.append((target, x90, gap))

    # Aggregate X@target across tasks
    md.append("\n## Aggregate X@target (mean ± std across tasks)\n")
    md.append("| method | X@90% | X@95% | X@99% |")
    md.append("|---|---|---|---|")
    for m in METHODS_REPORTED:
        cells = [m]
        for tgt in (0.9, 0.95, 0.99):
            xs = []
            for target, info in per_task_results.items():
                gap = info["fp_test"] - info["q_test"]
                if gap <= 1e-9:
                    continue
                frac, acc = info["curves"][m]
                rec = (acc - info["q_test"]) / gap
                x = _x_for_target_recovery(frac, rec, tgt)
                if not np.isnan(x):
                    xs.append(x)
            if xs:
                cells.append(f"{np.mean(xs)*100:.1f}% ± {np.std(xs)*100:.1f}")
            else:
                cells.append("—")
        md.append("| " + " | ".join(cells) + " |")

    # Aggregate gap recovery at canonical X
    md.append("\n## Mean gap-recovery at canonical FP-compute fractions (%)\n")
    md.append("| X | " + " | ".join(METHODS_REPORTED) + " |")
    md.append("|" + "---|" * (1 + len(METHODS_REPORTED)))
    for X in CANONICAL_FRACTIONS:
        cells = [f"{X*100:.0f}%"]
        for m in METHODS_REPORTED:
            recs = []
            for target, info in per_task_results.items():
                gap = info["fp_test"] - info["q_test"]
                if gap <= 1e-9:
                    continue
                frac, acc = info["curves"][m]
                rec = (acc - info["q_test"]) / gap
                v = float(_sample_at(frac, rec, [X])[0])
                recs.append(v * 100)
            if recs:
                cells.append(f"{np.mean(recs):+.1f}% ± {np.std(recs):.1f}")
            else:
                cells.append("—")
        md.append("| " + " | ".join(cells) + " |")

    markdown = "\n".join(md)

    # HTML: per-task LOO vs same-task curves
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(
            "LOO (red, target task held out) vs same-task (blue) gap-recovery, aggregate across tasks",
        ),
    )
    grid_fracs = np.linspace(0, 1, 201)
    for m in METHODS_REPORTED:
        curves_arr = []
        for target, info in per_task_results.items():
            gap = info["fp_test"] - info["q_test"]
            if gap <= 1e-9:
                continue
            frac, acc = info["curves"][m]
            rec = (acc - info["q_test"]) / gap
            curves_arr.append(_sample_at(frac, rec, grid_fracs))
        if not curves_arr:
            continue
        arr = np.stack(curves_arr, axis=0)
        mean = arr.mean(axis=0)
        lo = np.percentile(arr, 25, axis=0)
        hi = np.percentile(arr, 75, axis=0)
        c = METHOD_COLORS[m]
        h = c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        fillcolor = f"rgba({r},{g},{b},0.12)"
        fig.add_trace(
            go.Scatter(x=np.concatenate([grid_fracs, grid_fracs[::-1]]),
                       y=np.concatenate([hi, lo[::-1]]),
                       fill="toself", fillcolor=fillcolor, line=dict(color="rgba(0,0,0,0)"),
                       hoverinfo="skip", showlegend=False, name=f"{m} IQR"),
        )
        fig.add_trace(
            go.Scatter(x=grid_fracs, y=mean, mode="lines",
                       name=m, line=dict(color=c, width=2.5)),
        )
    fig.add_hline(y=1.0, line_dash="dot", line_color="grey")
    fig.add_hline(y=0.0, line_dash="dot", line_color="grey")
    fig.update_xaxes(title_text="FP-compute fraction X", range=[0, 1])
    fig.update_yaxes(title_text="gap recovery (1.0 = full FP)", range=[-0.1, 1.1])
    fig.update_layout(title=f"LOO Pareto — {args.model_name} W{args.bits} {args.granularity}", height=600)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "004_input_fragility"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"loo_pareto_{sanitized}_bits{args.bits}_{args.granularity}.md"
    html_path = out_dir / f"loo_pareto_{sanitized}_bits{args.bits}_{args.granularity}.html"
    md_path.write_text(markdown)
    fig.write_html(str(html_path))

    print(markdown)
    print()
    print(f"Markdown saved: {md_path}")
    print(f"HTML saved:     {html_path}")


if __name__ == "__main__":
    main()
