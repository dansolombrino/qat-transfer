"""Generate publication-quality figures for the paper from the existing dumps.

Outputs to paper/figs/, one set per (model, PTQ-config) combination:
  fig_headline_pareto_{sanitized}_bits{bits}_{granularity}.pdf
  fig_feature_ablation_{sanitized}_bits{bits}_{granularity}.pdf
  fig_regime_comparison_{sanitized}_W{p}vsW{s}_{granularity}.pdf
  fig_loo_vs_same_task_{sanitized}_bits{bits}_{granularity}.pdf

Argparse: --model-name, --batch-size, --primary-bits, --secondary-bits,
--granularity, plus the standard finetuning hyperparameters used in the
checkpoint path. Defaults reproduce the ViT-B paper baseline.

Matplotlib (vector PDF), no GPU, ~5 s end-to-end.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

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


# Publication-quality matplotlib defaults.
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


# ============================================================================
# argparse + cfg helpers
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default="vit_base_patch16_224.orig_in21k")
    p.add_argument("--batch-size", default="128")
    p.add_argument("--lr", default="1e-05")
    p.add_argument("--wd", default="0.1")
    p.add_argument("--ls", default="0.0")
    p.add_argument("--wl", default="500")
    p.add_argument("--max-grad-norm", default="1.0")
    p.add_argument("--seed", default="2038")
    p.add_argument("--skip-modules", nargs="+", default=["head"])
    p.add_argument("--primary-bits", type=int, default=4,
                   help="The headline PTQ bit-width (e.g. 4).")
    p.add_argument("--secondary-bits", type=int, default=3,
                   help="The stress-test PTQ bit-width used in the regime-comparison panel.")
    p.add_argument("--granularity", default="channel",
                   help="PTQ granularity (channel|tensor); shared between primary and secondary.")
    # Optional second backbone for dual-panel figures.
    p.add_argument("--also-model-name", default=None,
                   help="If set, emit DUAL-backbone figures (side-by-side with --model-name). "
                        "Common choice: --model-name vit_base_patch16_224.orig_in21k "
                        "--also-model-name vit_large_patch16_224.orig_in21k.")
    p.add_argument("--also-batch-size", default=None,
                   help="--batch-size for the second backbone (defaults to --batch-size).")
    return p.parse_args()


def _build_cfg(args: argparse.Namespace) -> SimpleNamespace:
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    return SimpleNamespace(
        model_name=args.model_name,
        sanitized=sanitize_timm_model_name(args.model_name),
        lr=args.lr, wd=args.wd, ls=args.ls, wl=args.wl,
        mgn=args.max_grad_norm, bs=args.batch_size, seed=args.seed,
        skip_tag=skip_tag,
        granularity=args.granularity,
    )


def _short_model(model_name: str) -> str:
    """ViT-{S,B,L}/{patch} short tag for figure titles."""
    n = model_name.lower()
    if "base" in n:
        size = "ViT-B"
    elif "large" in n:
        size = "ViT-L"
    elif "small" in n:
        size = "ViT-S"
    else:
        size = "ViT"
    m = re.search(r"patch(\d+)", model_name)
    return f"{size}/{m.group(1)}" if m else size


# ============================================================================
# parquet loading + curve helpers (cfg-parameterised)
# ============================================================================

def _dump_dir(cfg: SimpleNamespace, dataset: str, bits: int, granularity: str) -> Path:
    base = (
        Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision"
        / "ilharco_timm_supervised" / "input_fragility_dumps" / cfg.sanitized
    )
    return (
        base / dataset
        / f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.mgn}_bs={cfg.bs}"
        / f"ptq=bits={bits}_gran={granularity}_skip={cfg.skip_tag}"
        / f"seed={cfg.seed}"
    )


def _load_all(cfg: SimpleNamespace, bits: int, granularity: str) -> dict:
    base = (
        Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision"
        / "ilharco_timm_supervised" / "input_fragility_dumps" / cfg.sanitized
    )
    if not base.exists():
        return {}
    out = {}
    for ds_dir in sorted(base.iterdir()):
        if not ds_dir.is_dir():
            continue
        d = _dump_dir(cfg, ds_dir.name, bits, granularity)
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
# Figure 1: headline Pareto
# ============================================================================
def fig_headline_pareto(cfg, bits, granularity, model_short, out_path):
    print(f"Generating Figure 1 (headline Pareto, W{bits}-{granularity}, {cfg.sanitized}) ...")
    task_data = _load_all(cfg, bits, granularity)
    eligible = list(task_data.keys())
    print(f"  {len(eligible)} eligible tasks")
    if not eligible:
        print("  no eligible tasks; skipping.")
        return
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
    ax.text(98, 91.5, "90\\% gap recovered", ha="right", va="bottom",
            fontsize=8, color="green")
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("FP-compute fraction (\\%)")
    ax.set_ylabel("Mean FP$\\to$PTQ gap recovery (\\%)")
    ax.set_title(f"LOO cross-task routing at W{bits}-{granularity} ({len(eligible)} {model_short} tasks)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Figure 2: feature ablation
# ============================================================================
def fig_feature_ablation(cfg, bits, granularity, model_short, out_path):
    print(f"Generating Figure 2 (feature ablation, W{bits}-{granularity}, {cfg.sanitized}) ...")
    task_data = _load_all(cfg, bits, granularity)
    eligible = list(task_data.keys())
    if not eligible:
        print("  no eligible tasks; skipping.")
        return
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
        if arr is None:
            continue
        ax.plot(grid * 100, arr.mean(axis=0) * 100,
                color=color, linestyle=ls, linewidth=1.2,
                label="oracle" if name == "oracle" else "random")

    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("FP-compute fraction (\\%)")
    ax.set_ylabel("Mean gap recovery (\\%)")
    ax.set_title(f"Feature-subset ablation, W{bits}-{granularity} LOO ({len(eligible)} {model_short} tasks)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Figure 3: regime comparison (primary vs secondary bits)
# ============================================================================
def fig_regime_comparison(cfg, primary_bits, secondary_bits, granularity, model_short, out_path):
    print(f"Generating Figure 3 (regime comparison W{primary_bits} vs W{secondary_bits}, {cfg.sanitized}) ...")
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    for ax, bits, label_suffix in [
        (axes[0], primary_bits, "(recoverable regime)" if primary_bits >= 4 else ""),
        (axes[1], secondary_bits, "(catastrophic regime)" if secondary_bits <= 3 else ""),
    ]:
        task_data = _load_all(cfg, bits, granularity)
        eligible = list(task_data.keys())
        if not eligible:
            ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"W{bits}-{granularity}")
            continue
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
        title = f"W{bits}-{granularity} {label_suffix}".strip()
        ax.set_title(f"{title} ({len(eligible)} tasks)")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    axes[0].set_ylabel("Mean gap recovery (\\%)")
    axes[0].legend(loc="lower right", framealpha=0.95, fontsize=8)

    fig.suptitle(f"Regime comparison — {model_short}", fontsize=11, y=1.02)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Figure 4: per-task LOO vs same-task scatter (Q-only deployable)
# ============================================================================
def fig_loo_vs_same_task(cfg, bits, granularity, model_short, out_path):
    print(f"Generating Figure 4 (LOO vs same-task X@90\\%, Q-only, W{bits}-{granularity}, {cfg.sanitized}) ...")
    task_data = _load_all(cfg, bits, granularity)
    eligible = list(task_data.keys())
    if not eligible:
        print("  no eligible tasks; skipping.")
        return

    loo_curves = _loo_curves(task_data, Q_FEATS, eligible)

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
    ax.set_title(f"Q-only deployable LOO vs same-task X@90\\%\n"
                 f"(W{bits}-{granularity}, {len(common)} {model_short} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.95)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Dual-backbone variants: two backbones side-by-side in one figure.
# ============================================================================

def _annotate_x_at_90(ax, grid, mean, color, y_offset_pp):
    """Mark the X@90 point on a Pareto curve: scatter dot at (X@90, 90) plus
    a small text label showing the X@90 value. Stagger labels with y_offset_pp
    so multiple curves don't overlap labels at the 90% line."""
    above = np.where(mean >= 0.9)[0]
    if len(above) == 0:
        return
    x90 = float(grid[above[0]]) * 100
    ax.scatter([x90], [90], s=22, color=color, zorder=5, edgecolor="white", linewidth=0.6)
    ax.annotate(
        f"{x90:.1f}\\%",
        xy=(x90, 90),
        xytext=(x90 + 2.0, 90 + y_offset_pp),
        fontsize=7, color=color, ha="left", va="center",
        annotation_clip=True,
    )


def _plot_headline_on_ax(ax, cfg, bits, granularity, model_short, show_legend):
    """Render the headline Pareto into a pre-existing axis. Returns n_tasks."""
    task_data = _load_all(cfg, bits, granularity)
    eligible = list(task_data.keys())
    if not eligible:
        ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{model_short} — W{bits}-{granularity}")
        return 0
    grid = np.linspace(0, 1, 201)
    oracle, margin, rand = _baseline_curves(task_data, eligible)
    curves_q = _loo_curves(task_data, Q_FEATS, eligible)
    curves_all = _loo_curves(task_data, ALL_FEATS, eligible)
    series = [
        ("Oracle", oracle, "#000000", "-"),
        ("All features (ceiling)", curves_all, "#d62728", "-"),
        ("Q only (deployable)", curves_q, "#1f77b4", "-"),
        ("FP margin only", margin, "#ff7f0e", "--"),
        ("Random", rand, "#7f7f7f", ":"),
    ]
    # Vertical stagger so X@90 labels don't collide.
    annotate_offsets = {
        "Oracle": -6.0,
        "All features (ceiling)": -3.0,
        "Q only (deployable)": +3.0,
        "FP margin only": +6.0,
    }
    for label, curves, color, ls in series:
        arr = _aggregate_to_grid(curves, grid)
        if arr is None:
            continue
        mean = arr.mean(axis=0)
        lo = np.percentile(arr, 25, axis=0)
        hi = np.percentile(arr, 75, axis=0)
        ax.fill_between(grid * 100, lo * 100, hi * 100, color=color, alpha=0.10, linewidth=0)
        ax.plot(grid * 100, mean * 100, color=color, linestyle=ls, label=label)
        if label in annotate_offsets:
            _annotate_x_at_90(ax, grid, mean, color, annotate_offsets[label])
    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("FP-compute fraction (\\%)")
    ax.set_title(f"{model_short} --- W{bits}-{granularity} ({len(eligible)} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    return len(eligible)


def _plot_feature_ablation_on_ax(ax, cfg, bits, granularity, model_short, show_legend):
    task_data = _load_all(cfg, bits, granularity)
    eligible = list(task_data.keys())
    if not eligible:
        ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{model_short} --- W{bits}-{granularity}")
        return 0
    grid = np.linspace(0, 1, 201)
    # Annotate only the three subsets the paper actually highlights: q_only,
    # all_features, and oracle. Others would crowd the figure.
    annotate_offsets = {"q_only": +3.0, "all_features": -3.0}
    for name, cols in SUBSETS.items():
        curves = _loo_curves(task_data, cols, eligible)
        arr = _aggregate_to_grid(curves, grid)
        if arr is None:
            continue
        mean = arr.mean(axis=0)
        ax.plot(grid * 100, mean * 100,
                color=SUBSET_COLORS[name], label=SUBSET_LABELS[name])
        if name in annotate_offsets:
            _annotate_x_at_90(ax, grid, mean, SUBSET_COLORS[name], annotate_offsets[name])
    oracle, _, rand = _baseline_curves(task_data, eligible)
    for name, curves, color, ls in [("oracle", oracle, "#000000", "-"),
                                     ("random", rand, "#bbbbbb", ":")]:
        arr = _aggregate_to_grid(curves, grid)
        if arr is None:
            continue
        mean = arr.mean(axis=0)
        ax.plot(grid * 100, mean * 100,
                color=color, linestyle=ls, linewidth=1.2,
                label="oracle" if name == "oracle" else "random")
        if name == "oracle":
            _annotate_x_at_90(ax, grid, mean, color, -6.0)
    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("FP-compute fraction (\\%)")
    ax.set_title(f"{model_short} --- W{bits}-{granularity} ({len(eligible)} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.95, fontsize=7)
    return len(eligible)


def fig_headline_pareto_dual(cfg_a, cfg_b, bits, granularity, short_a, short_b, out_path):
    print(f"Generating dual headline Pareto: {cfg_a.sanitized} vs {cfg_b.sanitized}, "
          f"W{bits}-{granularity} ...")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    _plot_headline_on_ax(axes[0], cfg_a, bits, granularity, short_a, show_legend=False)
    _plot_headline_on_ax(axes[1], cfg_b, bits, granularity, short_b, show_legend=True)
    axes[0].set_ylabel("Mean FP$\\to$PTQ gap recovery (\\%)")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_feature_ablation_dual(cfg_a, cfg_b, bits, granularity, short_a, short_b, out_path):
    print(f"Generating dual feature ablation: {cfg_a.sanitized} vs {cfg_b.sanitized}, "
          f"W{bits}-{granularity} ...")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    _plot_feature_ablation_on_ax(axes[0], cfg_a, bits, granularity, short_a, show_legend=False)
    _plot_feature_ablation_on_ax(axes[1], cfg_b, bits, granularity, short_b, show_legend=True)
    axes[0].set_ylabel("Mean gap recovery (\\%)")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def fig_regime_comparison_dual(cfg_a, cfg_b, primary_bits, secondary_bits, granularity,
                                short_a, short_b, out_path):
    """2x2 grid: rows = backbones, cols = regimes (primary, secondary)."""
    print(f"Generating dual regime comparison: {cfg_a.sanitized} vs {cfg_b.sanitized}, "
          f"W{primary_bits} vs W{secondary_bits} ...")
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), sharex=True, sharey=True)
    grid = np.linspace(0, 1, 201)

    rows = [(cfg_a, short_a), (cfg_b, short_b)]
    cols = [(primary_bits, "(recoverable)" if primary_bits >= 4 else ""),
            (secondary_bits, "(catastrophic)" if secondary_bits <= 3 else "")]
    for r, (cfg, short) in enumerate(rows):
        for c, (bits, suffix) in enumerate(cols):
            ax = axes[r, c]
            task_data = _load_all(cfg, bits, granularity)
            eligible = list(task_data.keys())
            if not eligible:
                ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"{short} --- W{bits}-{granularity}")
                continue
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
            ax.set_title(f"{short} --- W{bits}-{granularity} {suffix} ({len(eligible)} tasks)".strip())
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
            if r == 1:
                ax.set_xlabel("FP-compute fraction (\\%)")
            if c == 0:
                ax.set_ylabel("Mean gap recovery (\\%)")
    axes[0, 1].legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def _scatter_loo_vs_same_on_ax(ax, cfg, bits, granularity, model_short):
    task_data = _load_all(cfg, bits, granularity)
    eligible = list(task_data.keys())
    if not eligible:
        ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{model_short} --- W{bits}-{granularity}")
        return
    loo_curves = _loo_curves(task_data, Q_FEATS, eligible)
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
    xs = [same_x[t] * 100 for t in common]
    ys = [loo_x[t] * 100 for t in common]
    ax.scatter(xs, ys, s=40, color="#1f77b4", edgecolor="white", zorder=3)
    for t, x, y in zip(common, xs, ys):
        ax.annotate(t, (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=6, alpha=0.75)
    lim = max(max(xs, default=0), max(ys, default=0), 5) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Same-task X@90\\% (\\%)")
    ax.set_title(f"{model_short} --- W{bits}-{granularity} ({len(common)} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)


def fig_loo_vs_same_task_dual(cfg_a, cfg_b, bits, granularity, short_a, short_b, out_path):
    print(f"Generating dual LOO-vs-same scatter: {cfg_a.sanitized} vs {cfg_b.sanitized} ...")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    _scatter_loo_vs_same_on_ax(axes[0], cfg_a, bits, granularity, short_a)
    _scatter_loo_vs_same_on_ax(axes[1], cfg_b, bits, granularity, short_b)
    axes[0].set_ylabel("LOO cross-task X@90\\% (\\%)")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    args = parse_args()
    cfg_a = _build_cfg(args)
    short_a = _short_model(args.model_name)

    primary_tag = f"{cfg_a.sanitized}_bits{args.primary_bits}_{args.granularity}"
    regime_tag = (
        f"{cfg_a.sanitized}_W{args.primary_bits}vsW{args.secondary_bits}_{args.granularity}"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Always emit the single-backbone figures (default behaviour, backward-compatible).
    fig_headline_pareto(
        cfg_a, args.primary_bits, args.granularity, short_a,
        FIG_DIR / f"fig_headline_pareto_{primary_tag}.pdf",
    )
    fig_feature_ablation(
        cfg_a, args.primary_bits, args.granularity, short_a,
        FIG_DIR / f"fig_feature_ablation_{primary_tag}.pdf",
    )
    fig_regime_comparison(
        cfg_a, args.primary_bits, args.secondary_bits, args.granularity, short_a,
        FIG_DIR / f"fig_regime_comparison_{regime_tag}.pdf",
    )
    fig_loo_vs_same_task(
        cfg_a, args.primary_bits, args.granularity, short_a,
        FIG_DIR / f"fig_loo_vs_same_task_{primary_tag}.pdf",
    )

    # Optional: dual-backbone figures (the paper's actual headline plots).
    if args.also_model_name is not None:
        args_b_ns = argparse.Namespace(**{**vars(args), "model_name": args.also_model_name})
        if args.also_batch_size is not None:
            args_b_ns.batch_size = args.also_batch_size
        cfg_b = _build_cfg(args_b_ns)
        short_b = _short_model(args.also_model_name)
        pair_tag = (
            f"{cfg_a.sanitized}_vs_{cfg_b.sanitized}"
            f"_bits{args.primary_bits}_{args.granularity}"
        )
        regime_pair_tag = (
            f"{cfg_a.sanitized}_vs_{cfg_b.sanitized}"
            f"_W{args.primary_bits}vsW{args.secondary_bits}_{args.granularity}"
        )
        fig_headline_pareto_dual(
            cfg_a, cfg_b, args.primary_bits, args.granularity, short_a, short_b,
            FIG_DIR / f"fig_headline_pareto_dual_{pair_tag}.pdf",
        )
        fig_feature_ablation_dual(
            cfg_a, cfg_b, args.primary_bits, args.granularity, short_a, short_b,
            FIG_DIR / f"fig_feature_ablation_dual_{pair_tag}.pdf",
        )
        fig_regime_comparison_dual(
            cfg_a, cfg_b, args.primary_bits, args.secondary_bits, args.granularity,
            short_a, short_b,
            FIG_DIR / f"fig_regime_comparison_dual_{regime_pair_tag}.pdf",
        )
        fig_loo_vs_same_task_dual(
            cfg_a, cfg_b, args.primary_bits, args.granularity, short_a, short_b,
            FIG_DIR / f"fig_loo_vs_same_task_dual_{pair_tag}.pdf",
        )

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
