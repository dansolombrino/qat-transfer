"""Generate publication-quality figures for the paper from the existing dumps.

Outputs to paper/figs/:
  fig_headline_pareto_w4.pdf    — Aggregate Pareto: Q-only LOO vs all_features
                                  vs margin_only vs random, plus oracle line.
  fig_feature_ablation_w4.pdf    — Pareto curves per feature subset.
  fig_regime_comparison.pdf      — W4 vs W3 aggregate curves side by side.
  fig_loo_vs_same_task.pdf       — Per-task scatter of LOO vs same-task X@90%.

Matplotlib (vector PDF), no GPU, ~5 s.
"""

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.vision.utils import sanitize_timm_model_name


# Set publication-quality defaults.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

MODEL_NAME = "vit_base_patch16_224.orig_in21k"
SANITIZED = sanitize_timm_model_name(MODEL_NAME)
LR, WD, LS, WL, MGN, BS, SEED = "1e-05", "0.1", "0.0", "500", "1.0", "128", "2038"
SKIP_TAG = "head"
FIG_DIR = _PROJECT_ROOT / "paper" / "figs"

# Feature sets — must match Script E.
IMAGE_FEATS = ["img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio"]
Q_FEATS = ["q_margin", "q_softmax_top1", "q_entropy"]
FP_FEATS = ["fp_margin", "fp_softmax_top1", "fp_entropy", "fp_cls_dist_to_class_centroid"]
CROSS_FEATS = ["fp_logit_at_q_pred", "q_logit_at_fp_pred", "fp_softmax_at_q_pred",
               "q_softmax_at_fp_pred", "fp_q_kl_symmetric", "fp_q_disagree"]
ALL_FEATS = FP_FEATS + Q_FEATS + CROSS_FEATS + IMAGE_FEATS

SUBSETS = {
    "image_only": IMAGE_FEATS,
    "q_only": Q_FEATS,
    "fp_only": FP_FEATS,
    "fp_plus_q_no_cross": FP_FEATS + Q_FEATS + IMAGE_FEATS,
    "all_features": ALL_FEATS,
}

SUBSET_COLORS = {
    "image_only": "#7f7f7f",
    "q_only": "#1f77b4",
    "fp_only": "#ff7f0e",
    "fp_plus_q_no_cross": "#2ca02c",
    "all_features": "#d62728",
}
SUBSET_LABELS = {
    "image_only": "image only",
    "q_only": "Q only (deployable)",
    "fp_only": "FP only",
    "fp_plus_q_no_cross": "FP+Q, no cross",
    "all_features": "all features (ceiling)",
}


def _dump_dir(dataset: str, bits: int, granularity: str) -> Path:
    base = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision" / "ilharco_timm_supervised" / "input_fragility_dumps" / SANITIZED
    return (
        base / dataset
        / f"optim=adamw_lr={LR}_wd={WD}_ls={LS}_wl={WL}_mgn={MGN}_bs={BS}"
        / f"ptq=bits={bits}_gran={granularity}_skip={SKIP_TAG}"
        / f"seed={SEED}"
    )


def _load_all(bits: int, granularity: str):
    base = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision" / "ilharco_timm_supervised" / "input_fragility_dumps" / SANITIZED
    out = {}
    for ds_dir in sorted(base.iterdir()):
        if not ds_dir.is_dir():
            continue
        d = _dump_dir(ds_dir.name, bits, granularity)
        if not (d / "predictions_test.parquet").exists():
            continue
        df_val = pd.read_parquet(d / "predictions_val.parquet")
        df_test = pd.read_parquet(d / "predictions_test.parquet")
        if int(df_val["bad"].sum()) >= 10 and int(df_test["bad"].sum()) >= 10:
            out[ds_dir.name] = (df_val, df_test)
    return out


def _per_task_z(X):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    X_norm = (X - mu) / sigma_safe
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    return X_norm.astype(np.float64), mu, sigma_safe


def _apply_z(X, mu, sigma):
    X_norm = (X - mu) / sigma
    return np.where(np.isnan(X_norm), 0.0, X_norm).astype(np.float64)


def _routed_curve(scores, fp_correct, q_correct):
    n = len(scores)
    rng = np.random.default_rng(12345)
    tie = rng.random(n)
    order = np.lexsort((tie, -scores))
    fp_sorted = fp_correct[order].astype(np.int64)
    q_sorted = q_correct[order].astype(np.int64)
    fp_cum = np.concatenate(([0], np.cumsum(fp_sorted)))
    q_cum = np.concatenate(([0], np.cumsum(q_sorted)))
    total_q = q_cum[-1]
    correct = fp_cum + (total_q - q_cum)
    return np.arange(n + 1) / n, correct.astype(np.float64) / n


def _loo_curves(task_data, cols, eligible):
    """For each target task, fit LogReg on pooled OTHER-tasks val FP-correct rows
    (per-task z-scored), score target's test, return curve dict per task."""
    per_task_train = {}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xv_norm, mu, sigma = _per_task_z(Xv)
        per_task_train[ds] = {"X_norm": Xv_norm, "y": yv, "mu": mu, "sigma": sigma}

    curves = {}
    for target in eligible:
        df_val_tgt, df_test_tgt = task_data[target]
        Xs = [per_task_train[t]["X_norm"] for t in eligible if t != target]
        ys = [per_task_train[t]["y"] for t in eligible if t != target]
        X_pool = np.concatenate(Xs, axis=0)
        y_pool = np.concatenate(ys, axis=0)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_pool, y_pool)
        X_test = np.array(df_test_tgt[cols].to_numpy(dtype=np.float64), copy=True)
        X_test_norm = _apply_z(X_test, per_task_train[target]["mu"], per_task_train[target]["sigma"])
        scores = clf.predict_proba(X_test_norm)[:, 1]
        fp_c = df_test_tgt["fp_correct"].to_numpy(dtype=bool)
        q_c = df_test_tgt["q_correct"].to_numpy(dtype=bool)
        frac, acc = _routed_curve(scores, fp_c, q_c)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        rec = (acc - float(q_c.mean())) / gap
        curves[target] = (frac, rec)
    return curves


def _baseline_curves(task_data, eligible):
    """Oracle, margin_only, random — one curve per task."""
    oracle = {}
    margin = {}
    rand = {}
    for target in eligible:
        df_val_tgt, df_test_tgt = task_data[target]
        fp_c = df_test_tgt["fp_correct"].to_numpy(dtype=bool)
        q_c = df_test_tgt["q_correct"].to_numpy(dtype=bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue

        s_or = fp_c.astype(int) - q_c.astype(int)
        s_m = -df_test_tgt["fp_margin"].to_numpy(dtype=np.float64)
        s_r = np.random.default_rng(0).random(len(df_test_tgt))

        for name, s, store in [("oracle", s_or, oracle), ("margin", s_m, margin), ("random", s_r, rand)]:
            frac, acc = _routed_curve(s, fp_c, q_c)
            rec = (acc - float(q_c.mean())) / gap
            store[target] = (frac, rec)
    return oracle, margin, rand


def _aggregate_to_grid(curves, grid):
    out = []
    for task, (frac, rec) in curves.items():
        out.append(np.interp(grid, frac, rec))
    return np.stack(out, axis=0) if out else None


# ============================================================================
# Figure 1: headline Pareto at W4-channel — Q-only deployable vs all_features ceiling
# ============================================================================
def fig_headline_pareto_w4():
    print("Generating Figure 1 (headline Pareto, W4-channel) ...")
    task_data = _load_all(4, "channel")
    eligible = list(task_data.keys())
    print(f"  {len(eligible)} eligible tasks")
    grid = np.linspace(0, 1, 201)

    oracle, margin, rand = _baseline_curves(task_data, eligible)
    curves_q = _loo_curves(task_data, Q_FEATS, eligible)
    curves_all = _loo_curves(task_data, ALL_FEATS, eligible)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    series = [
        ("Oracle (upper bound)", oracle, "#000000", "-"),
        ("All features (ceiling, both models)", curves_all, "#d62728", "-"),
        ("Q only (PTQ-first deployable)", curves_q, "#1f77b4", "-"),
        ("FP margin only", margin, "#ff7f0e", "--"),
        ("Random", rand, "#7f7f7f", ":"),
    ]
    for label, curves, color, ls in series:
        arr = _aggregate_to_grid(curves, grid)
        if arr is None:
            continue
        mean = arr.mean(axis=0)
        lo = np.percentile(arr, 25, axis=0)
        hi = np.percentile(arr, 75, axis=0)
        ax.fill_between(grid * 100, lo * 100, hi * 100, color=color, alpha=0.10, linewidth=0)
        ax.plot(grid * 100, mean * 100, color=color, linestyle=ls, label=label)

    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(98, 91.5, "90% gap recovered", ha="right", va="bottom",
            fontsize=8, color="green")
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("FP-compute fraction (\\%)")
    ax.set_ylabel("Mean FP$\\to$PTQ gap recovery (\\%)")
    ax.set_title("LOO cross-task routing at W4-channel (18 ViT-B tasks)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    out = FIG_DIR / "fig_headline_pareto_w4.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ============================================================================
# Figure 2: feature ablation at W4-channel
# ============================================================================
def fig_feature_ablation_w4():
    print("Generating Figure 2 (feature ablation, W4-channel) ...")
    task_data = _load_all(4, "channel")
    eligible = list(task_data.keys())
    grid = np.linspace(0, 1, 201)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, cols in SUBSETS.items():
        curves = _loo_curves(task_data, cols, eligible)
        arr = _aggregate_to_grid(curves, grid)
        if arr is None:
            continue
        mean = arr.mean(axis=0)
        ax.plot(grid * 100, mean * 100,
                color=SUBSET_COLORS[name], label=SUBSET_LABELS[name])

    oracle, _, rand = _baseline_curves(task_data, eligible)
    for name, curves, color, ls in [("oracle", oracle, "#000000", "-"),
                                     ("random", rand, "#bbbbbb", ":")]:
        arr = _aggregate_to_grid(curves, grid)
        ax.plot(grid * 100, arr.mean(axis=0) * 100,
                color=color, linestyle=ls, linewidth=1.2,
                label="oracle" if name == "oracle" else "random")

    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("FP-compute fraction (\\%)")
    ax.set_ylabel("Mean gap recovery (\\%)")
    ax.set_title("Feature-subset ablation, W4-channel LOO (18 tasks)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    out = FIG_DIR / "fig_feature_ablation_w4.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ============================================================================
# Figure 3: regime comparison W4 vs W3
# ============================================================================
def fig_regime_comparison():
    print("Generating Figure 3 (regime comparison W4 vs W3) ...")
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    for ax, bits, granularity, title in [
        (axes[0], 4, "channel", "W4-channel (recoverable regime)"),
        (axes[1], 3, "channel", "W3-channel (catastrophic regime)"),
    ]:
        task_data = _load_all(bits, granularity)
        eligible = list(task_data.keys())
        oracle, _, rand = _baseline_curves(task_data, eligible)
        curves_q = _loo_curves(task_data, Q_FEATS, eligible)
        curves_all = _loo_curves(task_data, ALL_FEATS, eligible)

        for label, curves, color, ls in [
            ("Oracle", oracle, "#000000", "-"),
            ("All features (ceiling)", curves_all, "#d62728", "-"),
            ("Q only (deployable)", curves_q, "#1f77b4", "-"),
            ("Random", rand, "#7f7f7f", ":"),
        ]:
            arr = _aggregate_to_grid(curves, grid)
            if arr is None:
                continue
            ax.plot(grid * 100, arr.mean(axis=0) * 100,
                    color=color, linestyle=ls, label=label)

        ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(-5, 105)
        ax.set_xlabel("FP-compute fraction (\\%)")
        ax.set_title(title + f" ({len(eligible)} tasks)")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    axes[0].set_ylabel("Mean gap recovery (\\%)")
    axes[0].legend(loc="lower right", framealpha=0.95, fontsize=8)

    out = FIG_DIR / "fig_regime_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ============================================================================
# Figure 4: per-task LOO vs same-task scatter
# ============================================================================
def fig_loo_vs_same_task():
    """Q-only deployable: LOO vs same-task per task. Matches paper Table:loo."""
    print("Generating Figure 4 (LOO vs same-task X@90%, Q-only deployable) ...")
    task_data = _load_all(4, "channel")
    eligible = list(task_data.keys())

    loo_curves = _loo_curves(task_data, Q_FEATS, eligible)

    # Same-task: fit LogReg on target's own val (Q-only features).
    same_curves = {}
    for target in eligible:
        df_val_tgt, df_test_tgt = task_data[target]
        vf = df_val_tgt[df_val_tgt["fp_correct"]].reset_index(drop=True)
        Xv = np.array(vf[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xv_norm, mu, sigma = _per_task_z(Xv)
        yv = vf["bad"].astype(int).to_numpy()
        if yv.sum() < 5 or (len(yv) - yv.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xv_norm, yv)
        X_test = np.array(df_test_tgt[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        X_test_norm = _apply_z(X_test, mu, sigma)
        scores = clf.predict_proba(X_test_norm)[:, 1]
        fp_c = df_test_tgt["fp_correct"].to_numpy(dtype=bool)
        q_c = df_test_tgt["q_correct"].to_numpy(dtype=bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        frac, acc = _routed_curve(scores, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        same_curves[target] = (frac, rec)

    def x_at_90(curves):
        out = {}
        for t, (frac, rec) in curves.items():
            above = np.where(rec >= 0.9)[0]
            out[t] = float("nan") if len(above) == 0 else float(frac[above[0]])
        return out

    loo_x = x_at_90(loo_curves)
    same_x = x_at_90(same_curves)
    common = sorted(set(loo_x.keys()) & set(same_x.keys()))

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    xs = [same_x[t] * 100 for t in common]
    ys = [loo_x[t] * 100 for t in common]
    ax.scatter(xs, ys, s=44, color="#1f77b4", edgecolor="white", zorder=3)
    for t, x, y in zip(common, xs, ys):
        ax.annotate(t, (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=7, alpha=0.75)
    lim = max(max(xs, default=0), max(ys, default=0), 5) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.8,
            label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Same-task X@90\\% (\\%)")
    ax.set_ylabel("LOO cross-task X@90\\% (\\%)")
    ax.set_title("Q-only deployable: LOO vs same-task X@90\\% (W4-channel, 18 tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.95)

    out = FIG_DIR / "fig_loo_vs_same_task.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ============================================================================

if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_headline_pareto_w4()
    fig_feature_ablation_w4()
    fig_regime_comparison()
    fig_loo_vs_same_task()
    print("\nAll figures generated.")
