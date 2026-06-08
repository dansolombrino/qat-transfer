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
import json
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


def _save_fig(fig, out_path):
    """Write the figure to `out_path` (typically .pdf) and a sibling .png."""
    out_path = Path(out_path)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=200)


# Feature sets — must match Script E.
IMAGE_FEATS = ["img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio"]
TEXT_FEATS  = ["txt_n_tokens", "txt_n_unique_tokens", "txt_type_token_ratio", "txt_punct_ratio"]
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
    "image_only": "#9467bd",  # input-domain only — distinct from Random's gray
    "q_only": "#1f77b4",
    "fp_only": "#ff7f0e",
    "fp_plus_q_no_cross": "#2ca02c",
    "all_features": "#d62728",
}
SUBSET_LABELS = {
    "image_only": "input-domain only",
    "q_only": "Q-only LogReg",
    "fp_only": "FP only",
    "fp_plus_q_no_cross": "FP+Q, no cross",
    "all_features": "all features (ceiling)",
}


def _input_feats_for(task_data) -> list:
    """Return the input-domain feature list (image-pixel for ViT, tokenizer-text for Qwen3)
    by inspecting the columns of the first task's val dataframe."""
    sample_df = next(iter(task_data.values()))[0]
    if "img_brightness" in sample_df.columns:
        return IMAGE_FEATS
    if "txt_n_tokens" in sample_df.columns:
        return TEXT_FEATS
    return []  # neither family present


def _subsets_for(task_data) -> dict:
    """Modality-aware SUBSETS: substitutes IMAGE_FEATS with TEXT_FEATS on Qwen3 dumps."""
    inp = _input_feats_for(task_data)
    return {
        "image_only": inp,
        "q_only": Q_FEATS,
        "fp_only": FP_FEATS,
        "fp_plus_q_no_cross": FP_FEATS + Q_FEATS + inp,
        "all_features": FP_FEATS + Q_FEATS + CROSS_FEATS + inp,
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
    # Optional Qwen3 NLP backbone for tri-panel headline / ablation figures.
    p.add_argument("--qwen3-model-name", default="Qwen/Qwen3-Embedding-0.6B",
                   help="HF model id for the NLP backbone (text/.../input_fragility_dumps/).")
    p.add_argument("--qwen3-batch-size", default="32")
    p.add_argument("--qwen3-max-length", default="128")
    p.add_argument("--qwen3-skip-modules", nargs="+", default=["score"])
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

def _load_all_qwen3(args, bits: int) -> dict:
    """Loader for Qwen3 NLP dumps. Returns the same (df_val, df_test, meta) tuple
    schema as `_load_all` so the rest of the pipeline does not care about modality.
    Returns {} if the Qwen3 sweep dump base directory does not exist.
    """
    try:
        from src.text.utils import sanitize_hf_model_name
    except Exception:
        # Fallback: lightweight inline sanitizer matching the HF convention.
        def sanitize_hf_model_name(name: str) -> str:
            return name.replace("/", "_").replace("-", "_")
    sanitized = sanitize_hf_model_name(args.qwen3_model_name)
    skip_tag = "-".join(sorted(args.qwen3_skip_modules)) if args.qwen3_skip_modules else "none"
    base = (
        Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text"
        / "ilharco_automodelforsequenceclassification" / "input_fragility_dumps" / sanitized
    )
    if not base.exists():
        return {}
    out = {}
    for ds_dir in sorted(base.iterdir()):
        if not ds_dir.is_dir():
            continue
        d = (
            ds_dir
            / f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_mgn={args.max_grad_norm}_bs={args.qwen3_batch_size}_ml={args.qwen3_max_length}"
            / f"ptq=bits={bits}_gran={args.granularity}_skip={skip_tag}"
            / f"seed={args.seed}"
        )
        if not (d / "predictions_test.parquet").exists():
            continue
        df_val = pd.read_parquet(d / "predictions_val.parquet")
        df_test = pd.read_parquet(d / "predictions_test.parquet")
        if int(df_val["bad"].sum()) >= 10 and int(df_test["bad"].sum()) >= 10:
            out[ds_dir.name] = (df_val, df_test)
    return out


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
    curves_q = _loo_curves(task_data, ["q_margin"], eligible)
    subsets = _subsets_for(task_data)
    curves_all = _loo_curves(task_data, subsets["all_features"], eligible)
    curves_input = _loo_curves(task_data, subsets["image_only"], eligible) if subsets["image_only"] else None

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    series = [
        ("oracle", oracle, "#000000", "-"),
        ("all features (ceiling)", curves_all, "#d62728", "-"),
        ("q_margin (deployable)", curves_q, "#1f77b4", "-"),
        ("FP margin (no LogReg)", margin, "#ff7f0e", "--"),
    ]
    if curves_input is not None:
        series.append(("input-domain only", curves_input, SUBSET_COLORS["image_only"], "--"))
    series.append(("random", rand, "#7f7f7f", ":"))
    annotate_labels = {"q_margin (deployable)", "FP margin (no LogReg)",
                       "input-domain only", "random"}
    for label, curves, color, ls in series:
        if not curves:
            continue
        xs, ys = _headline_curve(curves)
        if xs.size:
            ax.plot(xs, ys, color=color, linestyle=ls, label=label)
        if label in annotate_labels:
            _annotate_mean_per_task_x90(ax, curves, color, 0.0)

    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(98, 91.5, "90% gap recovered", ha="right", va="bottom",
            fontsize=8, color="green")
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"Average FP-compute fraction $\mathrm{mean}_t[X_{@r}]$ (%)")
    ax.set_ylabel("Gap recovery threshold r (%)")
    ax.set_title(f"LOO cross-task routing at W{bits}-{granularity} ({len(eligible)} {model_short} tasks)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    _save_fig(fig, out_path)
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
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name, cols in _subsets_for(task_data).items():
        if not cols:
            continue
        curves = _loo_curves(task_data, cols, eligible)
        if not curves:
            continue
        xs, ys = _headline_curve(curves)
        if xs.size:
            ax.plot(xs, ys, color=SUBSET_COLORS[name], label=SUBSET_LABELS[name])

    oracle, _, rand = _baseline_curves(task_data, eligible)
    for name, curves, color, ls in [("oracle", oracle, "#000000", "-"),
                                     ("random", rand, "#bbbbbb", ":")]:
        if not curves:
            continue
        xs, ys = _headline_curve(curves)
        if xs.size:
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.2, label=name)

    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"Average FP-compute fraction $\mathrm{mean}_t[X_{@r}]$ (%)")
    ax.set_ylabel("Gap recovery threshold r (%)")
    ax.set_title(f"Feature-subset ablation, W{bits}-{granularity} LOO ({len(eligible)} {model_short} tasks)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    _save_fig(fig, out_path)
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
        curves_q = _loo_curves(task_data, ["q_margin"], eligible)
        curves_all = _loo_curves(task_data, ALL_FEATS, eligible)

        for label, curves, color, ls in [
            ("oracle", oracle, "#000000", "-"),
            ("all features (ceiling)", curves_all, "#d62728", "-"),
            ("q_margin (deployable)", curves_q, "#1f77b4", "-"),
            ("random", rand, "#7f7f7f", ":"),
        ]:
            if not curves:
                continue
            xs, ys = _headline_curve(curves)
            if xs.size:
                ax.plot(xs, ys, color=color, linestyle=ls, label=label)

        ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(-5, 105)
        ax.set_xlabel(r"Average FP-compute fraction $\mathrm{mean}_t[X_{@r}]$ (%)")
        title = f"W{bits}-{granularity} {label_suffix}".strip()
        ax.set_title(f"{title} ({len(eligible)} tasks)")
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)

    axes[0].set_ylabel("Gap recovery threshold r (%)")
    axes[0].legend(loc="lower right", framealpha=0.95, fontsize=8)

    fig.suptitle(f"Regime comparison — {model_short}", fontsize=11, y=1.02)
    _save_fig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Figure 4: per-task LOO vs same-task scatter (Q-only deployable)
# ============================================================================
def fig_loo_vs_same_task(cfg, bits, granularity, model_short, out_path):
    print(f"Generating Figure 4 (LOO vs same-task X@90%, Q-only, W{bits}-{granularity}, {cfg.sanitized}) ...")
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
    ax.set_xlabel("Same-task X@90% (%)")
    ax.set_ylabel("LOO cross-task X@90% (%)")
    ax.set_title(f"Q-only deployable LOO vs same-task X@90%\n"
                 f"(W{bits}-{granularity}, {len(common)} {model_short} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    ax.legend(loc="upper left", framealpha=0.95)

    _save_fig(fig, out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


# ============================================================================
# Dual-backbone variants: two backbones side-by-side in one figure.
# ============================================================================

def _per_task_x90s(curves):
    """Per-task X@90 = smallest f at which task t's recovery curve hits 0.9.
    Returns a list of values in percent. Tasks that never reach 0.9 are dropped."""
    xs = []
    for _task, (frac, rec) in curves.items():
        above = np.where(rec >= 0.9)[0]
        if len(above) == 0:
            continue
        xs.append(float(frac[above[0]]) * 100)
    return xs


def _headline_curve(curves, r_grid=None):
    """Curve (mean_t [X@r(t)], r) for r in r_grid.

    At each recovery threshold r, take each task's per-task X@r (smallest f
    at which that task's recovery hits r), then average across tasks. The
    resulting curve passes through (mean_t [X@90(t)], 0.9) by construction.

    Returns (x_array, y_array) in % units. Tasks where X@r is undefined at
    a given r (curve never reaches r) are dropped at that r only."""
    if r_grid is None:
        r_grid = np.linspace(0.0, 1.0, 101)
    xs, ys = [], []
    for r in r_grid:
        x_at_r = []
        for _task, (frac, rec) in curves.items():
            above = np.where(rec >= r)[0]
            if len(above) == 0:
                continue
            x_at_r.append(float(frac[above[0]]))
        if x_at_r:
            xs.append(float(np.mean(x_at_r)) * 100.0)
            ys.append(r * 100.0)
    return np.array(xs), np.array(ys)


def _annotate_mean_per_task_x90(ax, curves, color, y_offset_pp):
    """Drop a vertical dashed line at mean_t X@90(t) and label its value.
    This is the headline metric the paper reports: per-task X@90 averaged
    across tasks (each task gets its own routing budget). The label is the
    X@90 value, rotated and placed to the left of the dashed line; the
    caption explains the color-coding."""
    del y_offset_pp  # legacy stagger no longer needed with rotated labels
    xs = _per_task_x90s(curves)
    if not xs:
        return
    mean_x = float(np.mean(xs))
    ax.axvline(mean_x, color=color, linestyle="--", alpha=0.55, linewidth=0.9)
    ax.annotate(
        f"{mean_x:.1f}%",
        xy=(mean_x, 15),
        xytext=(mean_x - 1.2, 15),
        fontsize=8, color=color, ha="right", va="center",
        rotation=90, rotation_mode="anchor",
        annotation_clip=True,
    )


def _plot_headline_on_ax(ax, task_data, bits, granularity, model_short, show_legend):
    """Render the headline Pareto into a pre-existing axis. Returns n_tasks.
    `task_data` is a pre-loaded dict from _load_all or _load_all_qwen3."""
    eligible = list(task_data.keys())
    if not eligible:
        ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{model_short} — W{bits}-{granularity}")
        return 0
    grid = np.linspace(0, 1, 201)
    oracle, margin, rand = _baseline_curves(task_data, eligible)
    curves_q = _loo_curves(task_data, ["q_margin"], eligible)
    subsets = _subsets_for(task_data)
    curves_all = _loo_curves(task_data, subsets["all_features"], eligible)
    curves_input = _loo_curves(task_data, subsets["image_only"], eligible) if subsets["image_only"] else None
    series = [
        ("oracle", oracle, "#000000", "-"),
        ("all features (ceiling)", curves_all, "#d62728", "-"),
        ("q_margin (deployable)", curves_q, "#1f77b4", "-"),
        ("FP margin (no LogReg)", margin, "#ff7f0e", "--"),
    ]
    if curves_input is not None:
        series.append(("input-domain only", curves_input, SUBSET_COLORS["image_only"], "--"))
    series.append(("random", rand, "#7f7f7f", ":"))
    # Vertical stagger so X@90 labels don't collide.
    annotate_offsets = {
        "q_margin (deployable)": +3.0,
        "FP margin (no LogReg)": +6.0,
        "input-domain only": +9.0,
    }
    # Headline aggregation: at each recovery threshold r, take each task's
    # per-task X@r (smallest f at which that task hits r), then average across
    # tasks. The resulting (mean_X@r, r) curve passes through the headline
    # mean_t[X@90(t)] = 22 / 18 / 20 at r=0.9 by construction.
    for label, curves, color, ls in series:
        if not curves:
            continue
        xs, ys = _headline_curve(curves)
        if xs.size == 0:
            continue
        ax.plot(xs, ys, color=color, linestyle=ls, label=label)
        if label in annotate_offsets:
            _annotate_mean_per_task_x90(ax, curves, color, annotate_offsets[label])
    legend_handles = None  # use the labels passed to ax.plot above
    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"Average FP-compute fraction $\mathrm{mean}_t[X_{@r}]$ (%)")
    ax.set_ylabel("Gap recovery threshold r (%)")
    ax.set_title(f"{model_short} --- W{bits}-{granularity} ({len(eligible)} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    if show_legend:
        if legend_handles is None:
            ax.legend(loc="lower right", framealpha=0.95, fontsize=8)
        else:
            ax.legend(handles=legend_handles, loc="lower right",
                      framealpha=0.95, fontsize=8)
    return len(eligible)


def _plot_feature_ablation_on_ax(ax, task_data, bits, granularity, model_short, show_legend):
    """`task_data` is pre-loaded; use _subsets_for to pick the modality-appropriate
    input-domain feature list."""
    eligible = list(task_data.keys())
    if not eligible:
        ax.text(0.5, 0.5, f"no dumps at W{bits}-{granularity}",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{model_short} --- W{bits}-{granularity}")
        return 0
    annotate_offsets = {"q_only": +3.0,
                        "fp_only": +6.0, "fp_plus_q_no_cross": -6.0,
                        "image_only": +9.0}
    subsets = _subsets_for(task_data)
    for name, cols in subsets.items():
        if not cols:
            continue
        curves = _loo_curves(task_data, cols, eligible)
        if not curves:
            continue
        color = SUBSET_COLORS[name]
        xs, ys = _headline_curve(curves)
        if xs.size:
            ax.plot(xs, ys, color=color, label=SUBSET_LABELS[name])
        if name in annotate_offsets:
            _annotate_mean_per_task_x90(ax, curves, color, annotate_offsets[name])
    oracle, _, rand = _baseline_curves(task_data, eligible)
    for name, curves, color, ls in [("oracle", oracle, "#000000", "-"),
                                     ("random", rand, "#bbbbbb", ":")]:
        if not curves:
            continue
        xs, ys = _headline_curve(curves)
        if xs.size:
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.2, label=name)
        if name in annotate_offsets:
            _annotate_mean_per_task_x90(ax, curves, color, annotate_offsets[name])
    ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"Average FP-compute fraction $\mathrm{mean}_t[X_{@r}]$ (%)")
    ax.set_ylabel("Gap recovery threshold r (%)")
    ax.set_title(f"{model_short} --- W{bits}-{granularity} ({len(eligible)} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    if show_legend:
        ax.legend(loc="lower right", framealpha=0.95, fontsize=7)
    return len(eligible)


_HEADLINE_FONT_RC = {
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


def fig_headline_pareto_dual(cfg_a, cfg_b, bits, granularity, short_a, short_b, out_path,
                             qwen3_data=None, qwen3_short=None):
    """Side-by-side headline Pareto. If `qwen3_data` is provided, emits a 3-panel
    figure (two ViTs + Qwen3); otherwise 2-panel."""
    n_panels = 3 if qwen3_data else 2
    width = 5.2 * n_panels
    print(f"Generating {n_panels}-panel headline Pareto: {cfg_a.sanitized} vs {cfg_b.sanitized}"
          f"{' vs Qwen3' if qwen3_data else ''}, W{bits}-{granularity} ...")
    td_a = _load_all(cfg_a, bits, granularity)
    td_b = _load_all(cfg_b, bits, granularity)
    with plt.rc_context(_HEADLINE_FONT_RC):
        fig, axes = plt.subplots(1, n_panels, figsize=(width, 3.8), sharey=True)
        _plot_headline_on_ax(axes[0], td_a, bits, granularity, short_a, show_legend=False)
        _plot_headline_on_ax(axes[1], td_b, bits, granularity, short_b, show_legend=False)
        if qwen3_data:
            _plot_headline_on_ax(axes[2], qwen3_data, bits, granularity, qwen3_short, show_legend=False)
        axes[0].set_ylabel("Gap recovery threshold r (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(bottom=0.24)
        _save_fig(fig, out_path)
        plt.close(fig)
    print(f"  saved {out_path}")


def fig_feature_ablation_dual(cfg_a, cfg_b, bits, granularity, short_a, short_b, out_path,
                              qwen3_data=None, qwen3_short=None):
    """Side-by-side feature ablation. If `qwen3_data` is provided, emits a 3-panel figure."""
    n_panels = 3 if qwen3_data else 2
    width = 5.2 * n_panels
    print(f"Generating {n_panels}-panel feature ablation: {cfg_a.sanitized} vs {cfg_b.sanitized}"
          f"{' vs Qwen3' if qwen3_data else ''}, W{bits}-{granularity} ...")
    td_a = _load_all(cfg_a, bits, granularity)
    td_b = _load_all(cfg_b, bits, granularity)
    with plt.rc_context(_HEADLINE_FONT_RC):
        fig, axes = plt.subplots(1, n_panels, figsize=(width, 3.8), sharey=True)
        _plot_feature_ablation_on_ax(axes[0], td_a, bits, granularity, short_a, show_legend=False)
        _plot_feature_ablation_on_ax(axes[1], td_b, bits, granularity, short_b, show_legend=False)
        if qwen3_data:
            _plot_feature_ablation_on_ax(axes[2], qwen3_data, bits, granularity, qwen3_short, show_legend=False)
        axes[0].set_ylabel("Gap recovery threshold r (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   framealpha=0.95, bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(bottom=0.24)
        _save_fig(fig, out_path)
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
            curves_q = _loo_curves(task_data, ["q_margin"], eligible)
            curves_all = _loo_curves(task_data, ALL_FEATS, eligible)
            for label, curves, color, ls in [
                ("oracle", oracle, "#000000", "-"),
                ("all features (ceiling)", curves_all, "#d62728", "-"),
                ("q_margin (deployable)", curves_q, "#1f77b4", "-"),
                ("random", rand, "#7f7f7f", ":"),
            ]:
                if not curves:
                    continue
                xs, ys = _headline_curve(curves)
                if xs.size:
                    ax.plot(xs, ys, color=color, linestyle=ls, label=label)
            ax.axhline(90, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.set_xlim(0, 100)
            ax.set_ylim(-5, 105)
            ax.set_title(f"{short} --- W{bits}-{granularity} {suffix} ({len(eligible)} tasks)".strip())
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
            if r == 1:
                ax.set_xlabel(r"Average FP-compute fraction $\mathrm{mean}_t[X_{@r}]$ (%)")
            if c == 0:
                ax.set_ylabel("Gap recovery threshold r (%)")
    axes[0, 1].legend(loc="lower right", framealpha=0.95, fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_path)
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
    ax.set_xlabel("Same-task X@90% (%)")
    ax.set_title(f"{model_short} --- W{bits}-{granularity} ({len(common)} tasks)")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)


def fig_loo_vs_same_task_dual(cfg_a, cfg_b, bits, granularity, short_a, short_b, out_path):
    print(f"Generating dual LOO-vs-same scatter: {cfg_a.sanitized} vs {cfg_b.sanitized} ...")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    _scatter_loo_vs_same_on_ax(axes[0], cfg_a, bits, granularity, short_a)
    _scatter_loo_vs_same_on_ax(axes[1], cfg_b, bits, granularity, short_b)
    axes[0].set_ylabel("LOO cross-task X@90% (%)")
    _save_fig(fig, out_path)
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
        # Optionally load Qwen3 dumps for tri-panel headline / ablation figures.
        qwen3_data = _load_all_qwen3(args, args.primary_bits)
        qwen3_short = "Qwen3-Emb-0.6B" if qwen3_data else None
        if qwen3_data:
            print(f"  including Qwen3 panel ({len(qwen3_data)} tasks loaded)")
        fig_headline_pareto_dual(
            cfg_a, cfg_b, args.primary_bits, args.granularity, short_a, short_b,
            FIG_DIR / f"fig_headline_pareto_dual_{pair_tag}.pdf",
            qwen3_data=qwen3_data, qwen3_short=qwen3_short,
        )
        fig_feature_ablation_dual(
            cfg_a, cfg_b, args.primary_bits, args.granularity, short_a, short_b,
            FIG_DIR / f"fig_feature_ablation_dual_{pair_tag}.pdf",
            qwen3_data=qwen3_data, qwen3_short=qwen3_short,
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
