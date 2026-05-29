"""Script B — analyzer for input-fragility dumps produced by 004's Script A.

For each per-task pair of `predictions_{val,test}.parquet` files, restricts to
FP-correct samples and asks: do any input or FP-derived properties predict
"bad" (FP-correct ∩ Q-wrong) vs "good" (FP-correct ∩ Q-correct)?

Per task, per property, reports:
  - Mean(good), Mean(bad)
  - Cohen's d
  - AUC of `property -> P(bad)`, computed on val and on test
  - Direction (higher = more bad?  or lower = more bad?)

Aggregates AUC across tasks (mean, std, per-task table).

Also fits a multivariate logistic regression on the val parquet using ALL
properties, and evaluates AUC both 5-fold-CV-on-val and on test. This is the
upper bound of "what a simple linear model can extract from these features."

Argparse, no Hydra, no GPU. Outputs:
  - Markdown summary to stdout (per-task, per-property, plus aggregate).
  - HTML report with per-property distributions and ROC curves.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.vision.utils import sanitize_timm_model_name


PROPERTY_COLUMNS = [
    # FP-pass difficulty proxies
    "fp_margin",
    "fp_softmax_top1",
    "fp_entropy",
    "fp_cls_dist_to_class_centroid",
    # Raw image statistics
    "img_brightness",
    "img_contrast",
    "img_edge_density",
    "img_high_freq_ratio",
]

# Pretty groupings for the report
PROPERTY_GROUPS = {
    "FP-difficulty": ["fp_margin", "fp_softmax_top1", "fp_entropy", "fp_cls_dist_to_class_centroid"],
    "Image-stats":   ["img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio"],
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
    p.add_argument("--datasets", nargs="*", default=None,
                   help="If omitted, auto-discover all tasks dumped at this config.")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="K-fold splits for the multivariate logistic regression on val.")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to plots/004_input_fragility/")
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
        f = ds_dir / optim_tag / ptq_tag / seed_tag / "predictions_val.parquet"
        if f.exists():
            out.append(ds_dir.name)
    return out


def _load_task(checkpoint_base, sanitized, dataset, optim_tag, ptq_tag, seed_tag):
    d = (
        checkpoint_base / "vision" / "ilharco_timm_supervised"
        / "input_fragility_dumps" / sanitized / dataset
        / optim_tag / ptq_tag / seed_tag
    )
    df_val = pd.read_parquet(d / "predictions_val.parquet")
    df_test = pd.read_parquet(d / "predictions_test.parquet")
    return df_val, df_test


def _filter_fp_correct(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to FP-correct rows. Within these, `bad` is the positive class."""
    return df[df["fp_correct"]].reset_index(drop=True)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC with guards for degenerate cases and NaN scores."""
    mask = ~np.isnan(scores)
    if mask.sum() < 2:
        return float("nan")
    y_true = np.asarray(y_true)[mask]
    scores = np.asarray(scores)[mask]
    if len(np.unique(y_true)) < 2 or np.std(scores) < 1e-12:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if s < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / s)


def _univariate_per_property(df_val: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per property with val + test AUC, effect size, means."""
    vf = _filter_fp_correct(df_val)
    tf = _filter_fp_correct(df_test)
    yv = vf["bad"].astype(int).to_numpy()
    yt = tf["bad"].astype(int).to_numpy()

    rows = []
    for col in PROPERTY_COLUMNS:
        if col not in vf.columns:
            continue
        xv = vf[col].to_numpy(dtype=np.float64)
        xt = tf[col].to_numpy(dtype=np.float64)
        mask_good_v = (yv == 0)
        mask_bad_v = (yv == 1)
        auc_v = _safe_auc(yv, xv)
        auc_t = _safe_auc(yt, xt)
        direction = "higher → bad" if (not np.isnan(auc_v) and auc_v > 0.5) else "lower → bad"
        rows.append({
            "property": col,
            "mean_good_val": float(xv[mask_good_v].mean()) if mask_good_v.any() else float("nan"),
            "mean_bad_val":  float(xv[mask_bad_v].mean())  if mask_bad_v.any()  else float("nan"),
            "cohens_d_val":  _cohens_d(xv[mask_bad_v], xv[mask_good_v]),  # positive if bad > good
            "auc_val":       auc_v,
            "auc_test":      auc_t,
            "direction":     direction,
        })
    return pd.DataFrame(rows)


def _multivariate_logreg(df_val: pd.DataFrame, df_test: pd.DataFrame, n_folds: int):
    """Fit StandardScaler + LogisticRegression on FP-correct val with all
    properties; return (mean cv-auc on val, auc on test, fitted full-data model
    + scaler so callers can inspect coefficients)."""
    vf = _filter_fp_correct(df_val)
    tf = _filter_fp_correct(df_test)

    cols = [c for c in PROPERTY_COLUMNS if c in vf.columns]
    Xv = vf[cols].to_numpy(dtype=np.float64)
    Xt = tf[cols].to_numpy(dtype=np.float64)
    yv = vf["bad"].astype(int).to_numpy()
    yt = tf["bad"].astype(int).to_numpy()

    # Drop rows with any NaN feature (notably fp_cls_dist_to_class_centroid
    # when a test sample's class had zero val samples).
    v_keep = ~np.isnan(Xv).any(axis=1)
    t_keep = ~np.isnan(Xt).any(axis=1)
    Xv, yv = Xv[v_keep], yv[v_keep]
    Xt, yt = Xt[t_keep], yt[t_keep]

    if yv.sum() < 5 or (len(yv) - yv.sum()) < 5:
        return {"cv_auc_val_mean": float("nan"), "cv_auc_val_std": float("nan"),
                "auc_test": float("nan"), "coefs": {}, "intercept": float("nan")}

    n_folds_eff = min(n_folds, int(yv.sum()), int((len(yv) - yv.sum())))
    n_folds_eff = max(2, n_folds_eff)

    # CV-AUC on val
    cv_aucs = []
    skf = StratifiedKFold(n_splits=n_folds_eff, shuffle=True, random_state=0)
    for tr_idx, te_idx in skf.split(Xv, yv):
        scaler = StandardScaler().fit(Xv[tr_idx])
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(scaler.transform(Xv[tr_idx]), yv[tr_idx])
        scores = clf.predict_proba(scaler.transform(Xv[te_idx]))[:, 1]
        cv_aucs.append(_safe_auc(yv[te_idx], scores))
    cv_aucs = np.asarray(cv_aucs)

    # Fit on full val, evaluate on test
    scaler = StandardScaler().fit(Xv)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(scaler.transform(Xv), yv)
    test_scores = clf.predict_proba(scaler.transform(Xt))[:, 1]
    auc_test = _safe_auc(yt, test_scores)

    coefs = {c: float(w) for c, w in zip(cols, clf.coef_.flatten())}
    return {
        "cv_auc_val_mean": float(np.nanmean(cv_aucs)),
        "cv_auc_val_std":  float(np.nanstd(cv_aucs)),
        "auc_test":        float(auc_test),
        "coefs":           coefs,
        "intercept":       float(clf.intercept_[0]),
    }


def _per_task_markdown(task: str, uni: pd.DataFrame, multi: dict, n_good_val, n_bad_val, n_good_test, n_bad_test) -> str:
    out = [f"## {task}", ""]
    out.append(f"- val:  good={n_good_val}  bad={n_bad_val}")
    out.append(f"- test: good={n_good_test}  bad={n_bad_test}")
    out.append("")
    out.append("### Univariate (within FP-correct, positive class = `bad`)")
    out.append("")
    out.append("| property | mean(good) val | mean(bad) val | Cohen's d | AUC val | AUC test | direction |")
    out.append("|---|---|---|---|---|---|---|")
    for _, r in uni.iterrows():
        out.append(
            f"| `{r['property']}` "
            f"| {r['mean_good_val']:.4f} | {r['mean_bad_val']:.4f} "
            f"| {r['cohens_d_val']:+.3f} "
            f"| {r['auc_val']:.3f} | {r['auc_test']:.3f} "
            f"| {r['direction']} |"
        )
    out.append("")
    out.append("### Multivariate logistic regression on all properties")
    out.append(
        f"- 5-fold CV AUC on val: **{multi['cv_auc_val_mean']:.3f} ± {multi['cv_auc_val_std']:.3f}**"
    )
    out.append(f"- AUC on held-out test: **{multi['auc_test']:.3f}**")
    if multi["coefs"]:
        out.append("- Standardised coefficients (positive = higher value → more likely bad):")
        sorted_coefs = sorted(multi["coefs"].items(), key=lambda kv: -abs(kv[1]))
        for c, w in sorted_coefs:
            out.append(f"  - `{c}`: {w:+.3f}")
    out.append("")
    return "\n".join(out)


def _aggregate_markdown(task_uni: dict[str, pd.DataFrame], task_multi: dict[str, dict]) -> str:
    out = ["", "## Aggregate across tasks", ""]
    out.append("### Univariate AUC (test) — per property")
    out.append("")
    out.append("| property | mean AUC | std AUC | per-task | direction (mode) |")
    out.append("|---|---|---|---|---|")
    by_prop: dict[str, list] = {p: [] for p in PROPERTY_COLUMNS}
    by_prop_dir: dict[str, list] = {p: [] for p in PROPERTY_COLUMNS}
    for task, df in task_uni.items():
        for _, r in df.iterrows():
            by_prop[r["property"]].append(r["auc_test"])
            by_prop_dir[r["property"]].append(r["direction"])
    for prop in PROPERTY_COLUMNS:
        vals = [v for v in by_prop[prop] if not np.isnan(v)]
        if not vals:
            continue
        mean_auc = float(np.mean(vals))
        std_auc = float(np.std(vals))
        per_task_pieces = []
        for task, df in task_uni.items():
            r = df[df["property"] == prop]
            if len(r) == 0:
                continue
            v = float(r["auc_test"].iloc[0])
            per_task_pieces.append(f"{task}={v:.2f}")
        dir_strs = by_prop_dir[prop]
        mode_dir = max(set(dir_strs), key=dir_strs.count) if dir_strs else "—"
        out.append(
            f"| `{prop}` | **{mean_auc:.3f}** | {std_auc:.3f} | {', '.join(per_task_pieces)} | {mode_dir} |"
        )
    out.append("")
    out.append("### Multivariate logistic regression (test AUC) — per task")
    out.append("")
    out.append("| task | CV val AUC | test AUC |")
    out.append("|---|---|---|")
    test_aucs = []
    for task, m in task_multi.items():
        out.append(
            f"| {task} | {m['cv_auc_val_mean']:.3f} ± {m['cv_auc_val_std']:.3f} | **{m['auc_test']:.3f}** |"
        )
        if not np.isnan(m["auc_test"]):
            test_aucs.append(m["auc_test"])
    if test_aucs:
        out.append(f"\nMean multivariate test AUC across {len(test_aucs)} tasks: **{np.mean(test_aucs):.3f}** "
                   f"(std {np.std(test_aucs):.3f}).")
    out.append("")
    return "\n".join(out)


def _render_html(task_uni: dict[str, pd.DataFrame], task_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]], out_html: Path, title: str):
    """One HTML with:
       (a) heatmap of per-task × per-property test AUC,
       (b) ROC curves per property, one panel per property, lines per task."""
    tasks = list(task_uni.keys())
    props = PROPERTY_COLUMNS

    # AUC heatmap
    auc_mat = np.full((len(tasks), len(props)), np.nan)
    for i, t in enumerate(tasks):
        df = task_uni[t]
        for j, p in enumerate(props):
            row = df[df["property"] == p]
            if len(row):
                auc_mat[i, j] = row["auc_test"].iloc[0]

    fig = make_subplots(
        rows=1 + len(props), cols=1,
        subplot_titles=["Test AUC heatmap (rows=tasks, cols=properties)"] + [f"ROC: {p}" for p in props],
        row_heights=[0.25] + [0.75 / len(props)] * len(props),
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Heatmap(
            z=auc_mat, x=props, y=tasks,
            zmin=0.4, zmax=0.9, colorscale="Viridis",
            colorbar=dict(title="AUC"),
            hovertemplate="task=%{y}<br>prop=%{x}<br>AUC=%{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Per-property ROC curves, one trace per task
    for ridx, prop in enumerate(props, start=2):
        for t in tasks:
            df_val, df_test = task_dfs[t]
            tf = _filter_fp_correct(df_test)
            if prop not in tf.columns or len(tf) == 0:
                continue
            yt = tf["bad"].astype(int).to_numpy()
            xt = tf[prop].to_numpy(dtype=np.float64)
            mask = ~np.isnan(xt)
            yt = yt[mask]
            xt = xt[mask]
            if len(np.unique(yt)) < 2 or len(xt) < 2:
                continue
            row = task_uni[t]
            r = row[row["property"] == prop]
            if len(r) == 0:
                continue
            auc = float(r["auc_test"].iloc[0])
            # If AUC < 0.5, the property is inversely related — flip the score so ROC is canonical.
            scores = xt if auc >= 0.5 else -xt
            displayed = max(auc, 1 - auc)
            fpr, tpr, _ = roc_curve(yt, scores)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines",
                           name=f"{t} (AUC={displayed:.2f})",
                           legendgroup=t, showlegend=(ridx == 2)),
                row=ridx, col=1,
            )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                       line=dict(color="grey", dash="dash"), showlegend=False),
            row=ridx, col=1,
        )
        fig.update_xaxes(title_text="FPR", row=ridx, col=1, range=[0, 1])
        fig.update_yaxes(title_text="TPR", row=ridx, col=1, range=[0, 1])

    fig.update_layout(title=title, height=350 + 280 * len(props))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))


def main() -> None:
    args = parse_args()
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = _build_paths(args)

    datasets = args.datasets or _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
    if not datasets:
        print(f"No dumps under {checkpoint_base}/.../input_fragility_dumps/{sanitized}/", file=sys.stderr)
        sys.exit(1)
    print(f"Analyzing {len(datasets)} task(s) at W{args.bits}-{args.granularity}: {datasets}\n")

    task_uni: dict[str, pd.DataFrame] = {}
    task_multi: dict[str, dict] = {}
    task_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    md_chunks = []
    md_chunks.append(
        f"# Input-fragility analysis — {args.model_name} | W{args.bits} {args.granularity}\n"
    )

    for ds in datasets:
        df_val, df_test = _load_task(checkpoint_base, sanitized, ds, optim_tag, ptq_tag, seed_tag)
        n_good_val = int(df_val["good"].sum()); n_bad_val = int(df_val["bad"].sum())
        n_good_test = int(df_test["good"].sum()); n_bad_test = int(df_test["bad"].sum())
        if n_bad_val < 10 or n_bad_test < 10:
            print(f"  skipping {ds}: too few bad samples (val={n_bad_val}, test={n_bad_test})")
            continue
        uni = _univariate_per_property(df_val, df_test)
        multi = _multivariate_logreg(df_val, df_test, args.cv_folds)
        task_uni[ds] = uni
        task_multi[ds] = multi
        task_dfs[ds] = (df_val, df_test)
        md_chunks.append(_per_task_markdown(ds, uni, multi, n_good_val, n_bad_val, n_good_test, n_bad_test))

    md_chunks.append(_aggregate_markdown(task_uni, task_multi))
    markdown = "\n".join(md_chunks)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "004_input_fragility"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"input_fragility_{sanitized}_bits{args.bits}_{args.granularity}.md"
    md_path.write_text(markdown)
    html_path = out_dir / f"input_fragility_{sanitized}_bits{args.bits}_{args.granularity}.html"
    _render_html(
        task_uni, task_dfs, html_path,
        title=f"Input-fragility ROCs — {args.model_name} W{args.bits} {args.granularity} ({len(task_uni)} tasks)",
    )

    print(markdown)
    print()
    print(f"Markdown saved: {md_path}")
    print(f"HTML saved:     {html_path}")


if __name__ == "__main__":
    main()
