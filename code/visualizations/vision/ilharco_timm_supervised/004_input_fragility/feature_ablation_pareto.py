"""Script E — feature-subset ablation for LOO cross-task Pareto routing.

For each deployment scenario, refits the LogReg using only the features
available under that scenario, runs leave-one-out cross-task routing, and
reports aggregate X@90/95/99 across held-out tasks.

Deployment scenarios:
  image_only         — no model forward; pure pixel router. Floor.
  q_only             — features computable from q_logits alone.
  q_plus_image       — Q-first deployable: PTQ runs first, decide then.
  fp_only            — FP-side scalars only (academic; not deployable).
  fp_plus_image      — FP-side + image stats (academic).
  fp_plus_q_no_cross — both models, no logit-gather cross features.
  all_features       — everything we have. Ceiling.

For each scenario, prints per-task X@90/95/99 + aggregate mean ± std.
HTML report compares Pareto curves across subsets.

Argparse, no Hydra, no GPU.
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
from sklearn.linear_model import LogisticRegression

from src.vision.utils import sanitize_timm_model_name


IMAGE_FEATURES = ["img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio"]
Q_SIDE_FEATURES = ["q_margin", "q_softmax_top1", "q_entropy"]
FP_SIDE_FEATURES = ["fp_margin", "fp_softmax_top1", "fp_entropy", "fp_cls_dist_to_class_centroid"]
CROSS_FEATURES = [
    "fp_logit_at_q_pred",
    "q_logit_at_fp_pred",
    "fp_softmax_at_q_pred",
    "q_softmax_at_fp_pred",
    "fp_q_kl_symmetric",
    "fp_q_disagree",
]

FEATURE_SUBSETS: dict[str, list[str]] = {
    "image_only":         IMAGE_FEATURES,
    "q_only":             Q_SIDE_FEATURES,
    "q_plus_image":       Q_SIDE_FEATURES + IMAGE_FEATURES,
    "fp_only":            FP_SIDE_FEATURES,
    "fp_plus_image":      FP_SIDE_FEATURES + IMAGE_FEATURES,
    "fp_plus_q_no_cross": FP_SIDE_FEATURES + Q_SIDE_FEATURES + IMAGE_FEATURES,
    "all_features":       FP_SIDE_FEATURES + Q_SIDE_FEATURES + CROSS_FEATURES + IMAGE_FEATURES,
}

DEPLOYMENT_LABELS = {
    "image_only":         "no model — image stats",
    "q_only":             "PTQ-first (no image)",
    "q_plus_image":       "PTQ-first deployable",
    "fp_only":             "FP-side only",
    "fp_plus_image":      "FP-side + image",
    "fp_plus_q_no_cross": "both models, no cross",
    "all_features":       "both models + cross (ceiling)",
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
    base = checkpoint_base / "vision" / "ilharco_timm_supervised" / "input_fragility_dumps" / sanitized
    if not base.exists():
        return []
    out = []
    for ds_dir in sorted(base.iterdir()):
        if ds_dir.is_dir() and (ds_dir / optim_tag / ptq_tag / seed_tag / "predictions_test.parquet").exists():
            out.append(ds_dir.name)
    return out


def _load_task(checkpoint_base, sanitized, dataset, optim_tag, ptq_tag, seed_tag):
    d = checkpoint_base / "vision" / "ilharco_timm_supervised" / "input_fragility_dumps" / sanitized / dataset / optim_tag / ptq_tag / seed_tag
    return pd.read_parquet(d / "predictions_val.parquet"), pd.read_parquet(d / "predictions_test.parquet")


def _per_task_standardise(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    X_norm = (X - mu) / sigma_safe
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    return X_norm.astype(np.float64), mu, sigma_safe


def _apply_standardise(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X_norm = (X - mu) / sigma
    return np.where(np.isnan(X_norm), 0.0, X_norm).astype(np.float64)


def _routed_curve(scores, fp_correct, q_correct):
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
    return np.arange(n + 1) / n, correct.astype(np.float64) / n


def _sample_at(fractions, accuracies, target_fracs):
    return np.interp(target_fracs, fractions, accuracies)


def _x_for_target_recovery(fractions, recoveries, target: float):
    above = np.where(recoveries >= target)[0]
    return float("nan") if len(above) == 0 else float(fractions[above[0]])


def _run_one_subset(subset_name, cols, task_data, eligible_tasks, min_bad_val=10):
    """Per-task LOO with only `cols`. Returns dict task -> {fp_test, q_test, curve_loo}.
    Per-task standardisation, target's own val stats applied to target's test."""

    # 1. Pre-build standardised val FP-correct rows + per-task statistics
    per_task_train = {}
    for ds in eligible_tasks:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < min_bad_val:
            continue
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xv_norm, mu, sigma = _per_task_standardise(Xv)
        per_task_train[ds] = {"X_norm": Xv_norm, "y": yv, "mu": mu, "sigma": sigma}

    eligible = list(per_task_train.keys())
    if len(eligible) < 3:
        return {}

    # 2. LOO per target
    per_task = {}
    for target in eligible:
        df_val_tgt, df_test_tgt = task_data[target]
        Xs = [per_task_train[t]["X_norm"] for t in eligible if t != target]
        ys = [per_task_train[t]["y"] for t in eligible if t != target]
        X_loo = np.concatenate(Xs, axis=0)
        y_loo = np.concatenate(ys, axis=0)
        if y_loo.sum() < 5 or (len(y_loo) - y_loo.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_loo, y_loo)

        X_test = np.array(df_test_tgt[cols].to_numpy(dtype=np.float64), copy=True)
        X_test_norm = _apply_standardise(X_test, per_task_train[target]["mu"], per_task_train[target]["sigma"])
        scores = clf.predict_proba(X_test_norm)[:, 1]

        fp_correct_test = df_test_tgt["fp_correct"].to_numpy(dtype=bool)
        q_correct_test = df_test_tgt["q_correct"].to_numpy(dtype=bool)
        frac, acc = _routed_curve(scores, fp_correct_test, q_correct_test)
        per_task[target] = {
            "fp_test": float(fp_correct_test.mean()),
            "q_test": float(q_correct_test.mean()),
            "curve_loo": (frac, acc),
        }
    return per_task


def _aggregate_x_at(per_task, target_recovery=0.9):
    xs = []
    for task, info in per_task.items():
        gap = info["fp_test"] - info["q_test"]
        if gap <= 1e-9:
            continue
        frac, acc = info["curve_loo"]
        rec = (acc - info["q_test"]) / gap
        x = _x_for_target_recovery(frac, rec, target_recovery)
        if not np.isnan(x):
            xs.append(x)
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (float("nan"), float("nan"), 0)


def main() -> None:
    args = parse_args()
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = _build_paths(args)
    datasets = args.datasets or _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
    if not datasets:
        print(f"No dumps under {checkpoint_base}/.../input_fragility_dumps/{sanitized}/", file=sys.stderr)
        sys.exit(1)

    print(f"Loading parquets for {len(datasets)} task(s)…")
    task_data = {ds: _load_task(checkpoint_base, sanitized, ds, optim_tag, ptq_tag, seed_tag) for ds in datasets}

    eligible_tasks = []
    for ds, (df_val, df_test) in task_data.items():
        n_bad_val = int(df_val["bad"].sum())
        n_bad_test = int(df_test["bad"].sum())
        if n_bad_val >= args.min_bad_val and n_bad_test >= args.min_bad_test:
            eligible_tasks.append(ds)
    print(f"{len(eligible_tasks)} eligible tasks: {eligible_tasks}\n")

    # Run all subsets
    subset_results: dict[str, dict] = {}
    md = [f"# Feature ablation — LOO Pareto routing — {args.model_name} | W{args.bits} {args.granularity}\n"]
    md.append(f"For each feature subset, LogReg fit on pooled val FP-correct rows from other tasks "
              f"(per-task-standardised), evaluated on held-out target's test. "
              f"`x@K%` = FP-compute fraction needed to recover K% of the FP→PTQ gap.\n")

    md.append("\n## Aggregate across tasks (LOO)\n")
    md.append("| subset | deployment | X@90% | X@95% | X@99% | n_tasks |")
    md.append("|---|---|---|---|---|---|")
    rows = []
    for subset_name, cols in FEATURE_SUBSETS.items():
        # Only keep columns actually present in the parquet
        any_df = task_data[eligible_tasks[0]][0]
        valid_cols = [c for c in cols if c in any_df.columns]
        if len(valid_cols) < len(cols):
            print(f"  warning: subset {subset_name} is missing columns "
                  f"{set(cols) - set(valid_cols)} — using {valid_cols}")
        per_task = _run_one_subset(subset_name, valid_cols, task_data, eligible_tasks, args.min_bad_val)
        subset_results[subset_name] = per_task
        x90 = _aggregate_x_at(per_task, 0.90)
        x95 = _aggregate_x_at(per_task, 0.95)
        x99 = _aggregate_x_at(per_task, 0.99)
        deploy_label = DEPLOYMENT_LABELS.get(subset_name, "")
        rows.append((subset_name, deploy_label, x90, x95, x99, len(per_task)))
        md.append(
            f"| `{subset_name}` | {deploy_label} | "
            f"{x90[0]*100:.1f}% ± {x90[1]*100:.1f} | "
            f"{x95[0]*100:.1f}% ± {x95[1]*100:.1f} | "
            f"{x99[0]*100:.1f}% ± {x99[1]*100:.1f} | "
            f"{x90[2]} |"
        )

    # Per-task per-subset X@90%
    md.append("\n## Per-task X@90% LOO\n")
    md.append("| task | " + " | ".join(FEATURE_SUBSETS.keys()) + " |")
    md.append("|" + "---|" * (1 + len(FEATURE_SUBSETS)))
    common_tasks = set.intersection(*[set(subset_results[s].keys()) for s in FEATURE_SUBSETS]) if all(subset_results.values()) else set()
    for task in sorted(common_tasks):
        cells = [task]
        for subset_name in FEATURE_SUBSETS:
            info = subset_results[subset_name][task]
            gap = info["fp_test"] - info["q_test"]
            if gap <= 1e-9:
                cells.append("—")
                continue
            frac, acc = info["curve_loo"]
            rec = (acc - info["q_test"]) / gap
            x = _x_for_target_recovery(frac, rec, 0.9)
            cells.append(f"{x*100:.1f}%" if not np.isnan(x) else "—")
        md.append("| " + " | ".join(cells) + " |")

    markdown = "\n".join(md)

    # HTML: per-subset aggregate Pareto curves
    grid = np.linspace(0, 1, 201)
    fig = go.Figure()
    colors = ["#7f7f7f", "#e377c2", "#bcbd22", "#17becf", "#ff7f0e", "#1f77b4", "#d62728"]
    for color, (subset_name, per_task) in zip(colors, subset_results.items()):
        curves = []
        for task, info in per_task.items():
            gap = info["fp_test"] - info["q_test"]
            if gap <= 1e-9:
                continue
            frac, acc = info["curve_loo"]
            rec = (acc - info["q_test"]) / gap
            curves.append(_sample_at(frac, rec, grid))
        if not curves:
            continue
        mean = np.stack(curves, axis=0).mean(axis=0)
        fig.add_trace(go.Scatter(x=grid, y=mean, mode="lines", name=f"{subset_name} ({DEPLOYMENT_LABELS[subset_name]})",
                                 line=dict(color=color, width=2.5)))
    fig.add_hline(y=1.0, line_dash="dot", line_color="grey")
    fig.add_hline(y=0.9, line_dash="dot", line_color="green",
                  annotation_text="90% gap", annotation_position="right")
    fig.update_xaxes(title_text="FP-compute fraction X", range=[0, 1])
    fig.update_yaxes(title_text="mean gap recovery", range=[-0.05, 1.1])
    fig.update_layout(title=f"Feature-subset LOO Pareto — W{args.bits} {args.granularity}", height=600)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "004_input_fragility"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"feature_ablation_{sanitized}_bits{args.bits}_{args.granularity}.md"
    html_path = out_dir / f"feature_ablation_{sanitized}_bits{args.bits}_{args.granularity}.html"
    md_path.write_text(markdown)
    fig.write_html(str(html_path))

    print(markdown)
    print()
    print(f"Markdown saved: {md_path}")
    print(f"HTML saved:     {html_path}")


if __name__ == "__main__":
    main()
