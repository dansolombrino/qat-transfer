"""Verify paper/main.tex numeric claims against the parquet source of truth.

Designed to be invokable on demand:
  uv run --active python code/visualizations/.../004_input_fragility/verify_paper_numbers.py \\
    --model-name vit_base_patch16_224.orig_in21k --batch-size 128 \\
    --also-model-name vit_large_patch16_224.orig_in21k --also-batch-size 64

It performs three checks and writes one consolidated report to
`plots/004_input_fragility/paper_verification.md`:

  1. **Auto-generated tables.** Re-runs `generate_paper_tables.py` with the
     same arguments and diffs the freshly emitted .tex files against the
     ones currently in `paper/tables/`. Any non-empty diff is a stale
     committed table, flagged explicitly.

  2. **Aggregate analyzer numbers.** Re-derives every "canonical" aggregate
     from the parquets directly (not from the analyzer .md cache), so the
     report cannot drift even if the .md files are stale. Output tables
     cover: eligibility per (backbone, regime); per-task FP / PTQ test
     accuracy and gap; mean gap and mean bad-fraction over eligible tasks;
     univariate discriminative AUC per feature; LOO Pareto $X_{@90}$ per
     feature subset; per-task Q-only LOO vs same-task $X_{@90}$ on the
     primary backbone; and threshold-calibration aggregates.

  3. **Derived multipliers.** Lists the speed-up ratios (random / Q-only,
     fp_only / Q-only, etc.) that appear in the abstract and §1 prose,
     computed from the same parquet-derived aggregates.

Output: `plots/004_input_fragility/paper_verification.md`. Grep this
against `paper/main.tex` to spot hand-typed mismatches.

Argparse, no GPU. ~30 s end-to-end."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from types import SimpleNamespace

from src.vision.utils import sanitize_timm_model_name


PROPERTY_COLUMNS = [
    "fp_margin", "fp_softmax_top1", "fp_entropy",
    "q_margin", "q_softmax_top1", "q_entropy",
    "fp_logit_at_q_pred", "q_logit_at_fp_pred", "fp_softmax_at_q_pred",
    "q_softmax_at_fp_pred", "fp_q_kl_symmetric", "fp_q_disagree",
    "img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio",
]
IMAGE_FEATS = ["img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio"]
Q_FEATS = ["q_margin", "q_softmax_top1", "q_entropy"]
# `fp_cls_dist_to_class_centroid` was removed: it required selecting the
# centroid for the test sample's true class, which is not available at
# inference time. Still present in the parquet dumps; no longer consumed.
FP_FEATS = ["fp_margin", "fp_softmax_top1", "fp_entropy"]
CROSS_FEATS = ["fp_logit_at_q_pred", "q_logit_at_fp_pred", "fp_softmax_at_q_pred",
               "q_softmax_at_fp_pred", "fp_q_kl_symmetric", "fp_q_disagree"]
ALL_FEATS = FP_FEATS + Q_FEATS + CROSS_FEATS + IMAGE_FEATS
SUBSETS = {
    "image_only": IMAGE_FEATS,
    # Univariate Q-side ablations (one feature each).
    "msp_only":         ["q_softmax_top1"],
    "q_margin_only":    ["q_margin"],
    "q_entropy_only":   ["q_entropy"],
    # Pairwise Q-side ablations (drop one of the three features).
    "q_margin_msp":     ["q_margin", "q_softmax_top1"],
    "q_margin_entropy": ["q_margin", "q_entropy"],
    "q_msp_entropy":    ["q_softmax_top1", "q_entropy"],
    "q_only": Q_FEATS,
    "q_plus_image": Q_FEATS + IMAGE_FEATS,
    "fp_only": FP_FEATS,
    "fp_plus_image": FP_FEATS + IMAGE_FEATS,
    "fp_plus_q_no_cross": FP_FEATS + Q_FEATS + IMAGE_FEATS,
    "all_features": ALL_FEATS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default="vit_base_patch16_224.orig_in21k")
    p.add_argument("--batch-size", default="128")
    p.add_argument("--also-model-name", default="vit_large_patch16_224.orig_in21k")
    p.add_argument("--also-batch-size", default="64")
    p.add_argument("--lr", default="1e-05")
    p.add_argument("--wd", default="0.1")
    p.add_argument("--ls", default="0.0")
    p.add_argument("--wl", default="500")
    p.add_argument("--max-grad-norm", default="1.0")
    p.add_argument("--seed", default="2038")
    p.add_argument("--skip-modules", nargs="+", default=["head"])
    p.add_argument("--bits-primary", type=int, default=4)
    p.add_argument("--bits-secondary", type=int, default=3)
    p.add_argument("--granularity", default="channel")
    p.add_argument("--min-bad", type=int, default=10)
    p.add_argument("--out-path", default=None)
    return p.parse_args()


def _build_cfg(args, model_name, batch_size) -> SimpleNamespace:
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    return SimpleNamespace(
        model_name=model_name,
        sanitized=sanitize_timm_model_name(model_name),
        lr=args.lr, wd=args.wd, ls=args.ls, wl=args.wl,
        mgn=args.max_grad_norm, bs=batch_size, seed=args.seed,
        skip_tag=skip_tag,
    )


def _dump_dir(cfg, dataset, bits, granularity):
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


def _load_all(cfg, bits, granularity):
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
        out[ds_dir.name] = (df_val, df_test)
    return out


def _eligibility(task_data, min_bad):
    eligible, excluded = [], []
    for ds, (dv, dt) in task_data.items():
        if int(dv["bad"].sum()) >= min_bad and int(dt["bad"].sum()) >= min_bad:
            eligible.append(ds)
        else:
            excluded.append((ds, int(dv["bad"].sum()), int(dt["bad"].sum())))
    return eligible, excluded


def _per_task_z(X):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    X_norm = (X - mu) / sigma_safe
    return np.where(np.isnan(X_norm), 0.0, X_norm).astype(np.float64), mu, sigma_safe


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
    return np.arange(n + 1) / n, (fp_cum + (q_cum[-1] - q_cum)).astype(np.float64) / n


def _x_at_90(frac, rec):
    above = np.where(rec >= 0.9)[0]
    return float("nan") if len(above) == 0 else float(frac[above[0]])


def _loo_x90(task_data, cols, eligible):
    pt = {}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sg = _per_task_z(Xv)
        pt[ds] = {"X": Xn, "y": yv, "mu": mu, "sg": sg}
    out = {}
    for target in eligible:
        Xs = [pt[t]["X"] for t in eligible if t != target]
        ys = [pt[t]["y"] for t in eligible if t != target]
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
        dt = task_data[target][1]
        Xt = np.array(dt[cols].to_numpy(dtype=np.float64), copy=True)
        Xtn = _apply_z(Xt, pt[target]["mu"], pt[target]["sg"])
        scores = clf.predict_proba(Xtn)[:, 1]
        fp_c = dt["fp_correct"].to_numpy(bool)
        q_c = dt["q_correct"].to_numpy(bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        frac, acc = _routed_curve(scores, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        out[target] = _x_at_90(frac, rec)
    return out


def _same_task_x90(task_data, cols, eligible):
    out = {}
    for target in eligible:
        df_val, df_test = task_data[target]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        if yv.sum() < 5 or (len(yv) - yv.sum()) < 5:
            continue
        Xn, mu, sg = _per_task_z(Xv)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xn, yv)
        Xt = np.array(df_test[cols].to_numpy(dtype=np.float64), copy=True)
        Xtn = _apply_z(Xt, mu, sg)
        scores = clf.predict_proba(Xtn)[:, 1]
        fp_c = df_test["fp_correct"].to_numpy(bool)
        q_c = df_test["q_correct"].to_numpy(bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        frac, acc = _routed_curve(scores, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        out[target] = _x_at_90(frac, rec)
    return out


def _baseline_x90(task_data, eligible, kind):
    out = {}
    for target in eligible:
        dt = task_data[target][1]
        fp_c = dt["fp_correct"].to_numpy(bool)
        q_c = dt["q_correct"].to_numpy(bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        if kind == "oracle":
            s = fp_c.astype(int) - q_c.astype(int)
        elif kind == "random":
            s = np.random.default_rng(0).random(len(dt))
        elif kind == "margin_only":
            s = -dt["fp_margin"].to_numpy(dtype=np.float64)
        else:
            raise ValueError(kind)
        frac, acc = _routed_curve(s, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        out[target] = _x_at_90(frac, rec)
    return out


def _agg(xs):
    vals = [v for v in xs if not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(vals)) * 100, float(np.std(vals)) * 100, len(vals))


def _univariate_auc(task_data, eligible):
    """Discriminative AUC = max(AUC, 1-AUC). Returns {feature: (mean, std, n)}.
    Computed on FP-correct val rows (positive class = bad)."""
    rows = {c: [] for c in PROPERTY_COLUMNS}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < 5 or len(vf) - int(vf["bad"].sum()) < 5:
            continue
        y = vf["bad"].astype(int).to_numpy()
        for c in PROPERTY_COLUMNS:
            if c not in vf.columns:
                continue
            x = vf[c].to_numpy(dtype=np.float64)
            mask = ~np.isnan(x)
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2 or np.std(x[mask]) < 1e-12:
                continue
            try:
                auc = roc_auc_score(y[mask], x[mask])
                rows[c].append(max(auc, 1 - auc))
            except Exception:
                pass
    return {c: (float(np.mean(v)) if v else float("nan"),
                float(np.std(v)) if v else float("nan"), len(v))
            for c, v in rows.items()}


def _threshold_calibration(task_data, cfg, eligible, percentile=75.0):
    """Compute the five threshold-strategy aggregates that appear in tab:threshold.
    Q-only features only; primary backbone, primary regime."""
    pt = {}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < 5:
            continue
        Xv = np.array(vf[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sg = _per_task_z(Xv)
        pt[ds] = {"X": Xn, "y": yv, "mu": mu, "sg": sg, "df_val": df_val}

    res = {s: {"frac": [], "rec": []} for s in
           ["batch", "natural", "val_pct", "source_pct", "val_labeled"]}

    for target in eligible:
        if target not in pt:
            continue
        Xs = [pt[t]["X"] for t in eligible if t != target and t in pt]
        ys = [pt[t]["y"] for t in eligible if t != target and t in pt]
        X_pool = np.concatenate(Xs, axis=0)
        y_pool = np.concatenate(ys, axis=0)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_pool, y_pool)
        source_pool_scores = clf.predict_proba(X_pool)[:, 1]

        df_val_tgt, df_test_tgt = task_data[target]
        Xv_full = np.array(df_val_tgt[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xv_full_n = _apply_z(Xv_full, pt[target]["mu"], pt[target]["sg"])
        Xt = np.array(df_test_tgt[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xtn = _apply_z(Xt, pt[target]["mu"], pt[target]["sg"])
        val_scores = clf.predict_proba(Xv_full_n)[:, 1]
        test_scores = clf.predict_proba(Xtn)[:, 1]

        fp_t = df_test_tgt["fp_correct"].to_numpy(bool)
        q_t = df_test_tgt["q_correct"].to_numpy(bool)
        gap = float(fp_t.mean()) - float(q_t.mean())
        if gap <= 1e-9:
            continue
        q_acc = float(q_t.mean())

        # batch — smallest fraction at which the routed curve crosses 0.90 recovery.
        # Use `np.where` (not `np.searchsorted`) because `rec` is not strictly
        # monotonic and searchsorted on a non-sorted array is ill-defined; the
        # `np.where` form matches `_loo_x90_for_subset` in `generate_paper_tables.py`.
        frac, acc = _routed_curve(test_scores, fp_t, q_t)
        rec = (acc - q_acc) / gap
        idx_above = np.where(rec >= 0.90)[0]
        if len(idx_above) > 0:
            idx = int(idx_above[0])
            res["batch"]["frac"].append(float(frac[idx]))
            res["batch"]["rec"].append(float(rec[idx]))

        # natural τ=0.5
        for label, tau in [
            ("natural", 0.5),
            ("val_pct", float(np.percentile(val_scores, percentile))),
            ("source_pct", float(np.percentile(source_pool_scores, percentile))),
        ]:
            routed = test_scores > tau
            correct = np.where(routed, fp_t, q_t)
            res[label]["frac"].append(float(routed.mean()))
            res[label]["rec"].append((float(correct.mean()) - q_acc) / gap)

        # val_labeled: choose tau on val to reach 0.90 recovery with smallest fraction
        fp_v = df_val_tgt["fp_correct"].to_numpy(bool)
        q_v = df_val_tgt["q_correct"].to_numpy(bool)
        gap_v = float(fp_v.mean()) - float(q_v.mean())
        if gap_v > 1e-9:
            cand = np.unique(val_scores)[::-1]
            cand = np.concatenate((cand, [-np.inf]))
            best_tau = None
            for tau in cand:
                r = val_scores > tau
                c = np.where(r, fp_v, q_v)
                if (float(c.mean()) - float(q_v.mean())) / gap_v >= 0.90:
                    best_tau = float(tau)
                    break
            if best_tau is not None:
                r_t = test_scores > best_tau
                c_t = np.where(r_t, fp_t, q_t)
                res["val_labeled"]["frac"].append(float(r_t.mean()))
                res["val_labeled"]["rec"].append((float(c_t.mean()) - q_acc) / gap)

    summary = {}
    for s in res:
        if not res[s]["frac"]:
            summary[s] = {"frac_mean": float("nan"), "frac_std": float("nan"),
                          "rec_mean": float("nan"), "rec_std": float("nan"), "n": 0}
        else:
            summary[s] = {
                "frac_mean": float(np.mean(res[s]["frac"])) * 100,
                "frac_std": float(np.std(res[s]["frac"])) * 100,
                "rec_mean": float(np.mean(res[s]["rec"])) * 100,
                "rec_std": float(np.std(res[s]["rec"])) * 100,
                "n": len(res[s]["frac"]),
            }
    return summary


def _check_table_diffs(args):
    """Re-run generate_paper_tables.py with the same arguments and diff."""
    out = []
    out.append("## 1. Auto-generated tables: diff against committed `paper/tables/`")
    out.append("")
    tmp = Path("/tmp/qat_verify_tables")
    tmp.mkdir(parents=True, exist_ok=True)
    tables = [
        f"appendix_task_stats_{sanitize_timm_model_name(args.model_name)}.tex",
        f"appendix_pareto_ablation_{sanitize_timm_model_name(args.model_name)}_bits{args.bits_primary}_{args.granularity}.tex",
        f"appendix_task_stats_{sanitize_timm_model_name(args.also_model_name)}.tex",
        f"appendix_pareto_ablation_{sanitize_timm_model_name(args.also_model_name)}_bits{args.bits_primary}_{args.granularity}.tex",
        f"ablation_dual_{sanitize_timm_model_name(args.model_name)}_vs_{sanitize_timm_model_name(args.also_model_name)}_W{args.bits_primary}vsW{args.bits_secondary}_{args.granularity}.tex",
    ]
    # snapshot existing
    for t in tables:
        src = Path("paper/tables") / t
        if src.exists():
            (tmp / t).write_bytes(src.read_bytes())
    # re-run
    script = "code/visualizations/vision/ilharco_timm_supervised/004_input_fragility/generate_paper_tables.py"
    cmd = [
        "uv", "run", "--active", "python", script,
        "--model-name", args.model_name, "--batch-size", args.batch_size,
        "--also-model-name", args.also_model_name, "--also-batch-size", args.also_batch_size,
        "--lr", args.lr, "--wd", args.wd, "--ls", args.ls, "--wl", args.wl,
        "--max-grad-norm", args.max_grad_norm, "--seed", args.seed,
        "--skip-modules", *args.skip_modules,
        "--bits-primary", str(args.bits_primary),
        "--bits-secondary", str(args.bits_secondary),
        "--granularity", args.granularity,
    ]
    res = subprocess.run(cmd, env={**os.environ, "VIRTUAL_ENV": str(_PROJECT_ROOT / ".venv")},
                         capture_output=True, text=True)
    if res.returncode != 0:
        out.append("```")
        out.append("FAILED to re-run generate_paper_tables.py:")
        out.append(res.stderr or res.stdout)
        out.append("```")
        return "\n".join(out)
    # diff
    out.append("| table | status |")
    out.append("|---|---|")
    for t in tables:
        before = tmp / t
        after = Path("paper/tables") / t
        if not before.exists() and after.exists():
            out.append(f"| `{t}` | newly created |")
        elif before.exists() and after.exists():
            if before.read_bytes() == after.read_bytes():
                out.append(f"| `{t}` | ✅ identical |")
            else:
                out.append(f"| `{t}` | ❌ DIFFERS from re-run |")
        else:
            out.append(f"| `{t}` | ⚠ missing |")
    out.append("")
    return "\n".join(out)


def _report_backbone(cfg, label, args, task_data_p, task_data_s, eligible_p, excluded_p, eligible_s, excluded_s):
    rows = []
    rows.append(f"### {label} (`{cfg.sanitized}`, bs={cfg.bs})")
    rows.append("")
    rows.append(f"- W{args.bits_primary}-{args.granularity}: **{len(eligible_p)}** eligible / {len(task_data_p)} total.")
    if excluded_p:
        rows.append(f"  - Excluded (n_bad<{args.min_bad}): "
                    + ", ".join(f"{ds} (val {vb}, test {tb})" for ds, vb, tb in excluded_p))
    rows.append(f"- W{args.bits_secondary}-{args.granularity}: **{len(eligible_s)}** eligible / {len(task_data_s)} total.")
    if excluded_s:
        rows.append(f"  - Excluded (n_bad<{args.min_bad}): "
                    + ", ".join(f"{ds} (val {vb}, test {tb})" for ds, vb, tb in excluded_s))
    rows.append("")
    return "\n".join(rows)


def _gap_and_bad(task_data, eligible):
    """Mean FP→PTQ test-acc gap (pp) and mean test bad-fraction (%) across eligible tasks."""
    gaps, bads = [], []
    for ds in eligible:
        dt = task_data[ds][1]
        fp = float(dt["fp_correct"].mean())
        q = float(dt["q_correct"].mean())
        b = float(dt["bad"].mean())
        gaps.append((fp - q) * 100)
        bads.append(b * 100)
    if not gaps:
        return (float("nan"), float("nan"))
    return (float(np.mean(gaps)), float(np.mean(bads)))


def main():
    args = parse_args()
    out_lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_lines.append(f"# Paper-numbers verification report — {now}")
    out_lines.append("")
    out_lines.append(f"Source of truth: parquets in "
                     f"`storage/checkpoints/.../input_fragility_dumps/` re-derived in place.")
    out_lines.append(f"This report is the canonical reference; grep it against `paper/main.tex` claims.")
    out_lines.append("")

    # 1) auto-generated table diffs
    out_lines.append(_check_table_diffs(args))

    # build cfgs
    cfg_p = _build_cfg(args, args.model_name, args.batch_size)
    cfg_s = _build_cfg(args, args.also_model_name, args.also_batch_size)
    primary_label = "Primary backbone"
    secondary_label = "Secondary backbone"

    # 2) Load parquets for both backbones at both regimes
    print(f"Loading parquets for {cfg_p.sanitized} and {cfg_s.sanitized} ...")
    data = {}
    for cfg, lbl in [(cfg_p, "p"), (cfg_s, "s")]:
        for bits in [args.bits_primary, args.bits_secondary]:
            data[(lbl, bits)] = _load_all(cfg, bits, args.granularity)

    elig = {}
    excl = {}
    for k, td in data.items():
        elig[k], excl[k] = _eligibility(td, args.min_bad)

    out_lines.append("## 2. Eligibility")
    out_lines.append("")
    out_lines.append(_report_backbone(cfg_p, primary_label, args,
                                       data[("p", args.bits_primary)], data[("p", args.bits_secondary)],
                                       elig[("p", args.bits_primary)], excl[("p", args.bits_primary)],
                                       elig[("p", args.bits_secondary)], excl[("p", args.bits_secondary)]))
    out_lines.append(_report_backbone(cfg_s, secondary_label, args,
                                       data[("s", args.bits_primary)], data[("s", args.bits_secondary)],
                                       elig[("s", args.bits_primary)], excl[("s", args.bits_primary)],
                                       elig[("s", args.bits_secondary)], excl[("s", args.bits_secondary)]))

    # 3) Mean gap and bad fraction across eligible tasks
    out_lines.append("## 3. Mean FP$\\to$PTQ test-accuracy gap and mean test bad-fraction across eligible tasks")
    out_lines.append("")
    out_lines.append("| Backbone | Regime | n_eligible | Mean gap (pp) | Mean bad fraction (%) |")
    out_lines.append("|---|---|---|---|---|")
    for cfg, k_prefix, name in [(cfg_p, "p", "primary"), (cfg_s, "s", "secondary")]:
        for bits in [args.bits_primary, args.bits_secondary]:
            td = data[(k_prefix, bits)]
            el = elig[(k_prefix, bits)]
            g, b = _gap_and_bad(td, el)
            out_lines.append(f"| `{cfg.sanitized}` | W{bits}-{args.granularity} | {len(el)} | {g:.2f} | {b:.2f} |")
    out_lines.append("")

    # 4) Univariate AUC (primary backbone, primary regime)
    out_lines.append(f"## 4. Univariate discriminative AUC (primary backbone, W{args.bits_primary}-{args.granularity})")
    out_lines.append("")
    out_lines.append("`Discriminative AUC = max(AUC, 1-AUC)`, computed on FP-correct val rows. "
                     "Mean and std across eligible tasks.")
    out_lines.append("")
    aucs = _univariate_auc(data[("p", args.bits_primary)], elig[("p", args.bits_primary)])
    out_lines.append("| Feature | Discriminative AUC | Std |")
    out_lines.append("|---|---|---|")
    for c in PROPERTY_COLUMNS:
        m, s, n = aucs.get(c, (float("nan"), float("nan"), 0))
        out_lines.append(f"| `{c}` | {m:.3f} | {s:.3f} |")
    out_lines.append("")

    # 5) Pareto X@90 ablation (all four backbone × regime combos)
    out_lines.append("## 5. LOO Pareto $X_{@90}$ per feature subset (mean ± std, %)")
    out_lines.append("")
    header = (
        "| Subset | "
        f"{cfg_p.sanitized} W{args.bits_primary} "
        f"| {cfg_s.sanitized} W{args.bits_primary} "
        f"| {cfg_p.sanitized} W{args.bits_secondary} "
        f"| {cfg_s.sanitized} W{args.bits_secondary} |"
    )
    out_lines.append(header)
    out_lines.append("|---|---|---|---|---|")
    for sname, cols in SUBSETS.items():
        cells = [f"`{sname}`"]
        for k_prefix in ("p", "s"):
            for bits in [args.bits_primary, args.bits_secondary]:
                td = data[(k_prefix, bits)]
                el = elig[(k_prefix, bits)]
                if not el:
                    cells.append("—")
                    continue
                x = list(_loo_x90(td, cols, el).values())
                m, s, _ = _agg(x)
                cells.append(f"{m:.1f} ± {s:.1f}")
        # reorder cells: subset, p×primary, s×primary, p×secondary, s×secondary
        # (cells already in order: p-primary, p-secondary, s-primary, s-secondary; rearrange)
        reordered = [cells[0], cells[1], cells[3], cells[2], cells[4]]
        out_lines.append(" | ".join(reordered) + " |")
    # baselines: oracle, margin_only, random
    for kind in ("oracle", "margin_only", "random"):
        cells = [kind]
        for k_prefix in ("p", "s"):
            for bits in [args.bits_primary, args.bits_secondary]:
                td = data[(k_prefix, bits)]
                el = elig[(k_prefix, bits)]
                if not el:
                    cells.append("—")
                    continue
                x = list(_baseline_x90(td, el, kind).values())
                m, s, _ = _agg(x)
                cells.append(f"{m:.1f} ± {s:.1f}")
        reordered = [cells[0], cells[1], cells[3], cells[2], cells[4]]
        out_lines.append(" | ".join(reordered) + " |")
    out_lines.append("")

    # 6) Per-task Q-only LOO vs same-task (primary backbone, primary regime)
    out_lines.append(f"## 6. Per-task Q-only LOO vs same-task $X_{{@90}}$ "
                     f"({cfg_p.sanitized} W{args.bits_primary}-{args.granularity})")
    out_lines.append("")
    el_pp = elig[("p", args.bits_primary)]
    loo_q = _loo_x90(data[("p", args.bits_primary)], Q_FEATS, el_pp)
    same_q = _same_task_x90(data[("p", args.bits_primary)], Q_FEATS, el_pp)
    out_lines.append("| Task | same-task | LOO | |diff| |")
    out_lines.append("|---|---|---|---|")
    diffs = []
    for ds in sorted(set(loo_q) & set(same_q)):
        s = same_q[ds] * 100
        l = loo_q[ds] * 100
        d = abs(l - s)
        diffs.append(d)
        out_lines.append(f"| {ds} | {s:.1f}% | {l:.1f}% | {d:.1f} pp |")
    if diffs:
        loo_mean = float(np.mean([loo_q[t] for t in same_q if t in loo_q])) * 100
        same_mean = float(np.mean([same_q[t] for t in same_q if t in loo_q])) * 100
        out_lines.append(f"| **mean** | **{same_mean:.1f}%** | **{loo_mean:.1f}%** | "
                         f"max |diff| = **{max(diffs):.1f} pp** |")
    out_lines.append("")

    # 7) Threshold calibration aggregates (primary backbone, primary regime)
    out_lines.append(f"## 7. Threshold calibration (Q-only, primary backbone, W{args.bits_primary}-{args.granularity})")
    out_lines.append("")
    th = _threshold_calibration(data[("p", args.bits_primary)], cfg_p, el_pp)
    out_lines.append("| Strategy | Frac routed (%) | Recovery (%) |")
    out_lines.append("|---|---|---|")
    for s in ["batch", "natural", "val_pct", "source_pct", "val_labeled"]:
        r = th[s]
        if r["n"] == 0:
            out_lines.append(f"| `{s}` | — | — |")
        else:
            out_lines.append(f"| `{s}` | {r['frac_mean']:.1f} ± {r['frac_std']:.1f} | "
                             f"{r['rec_mean']:.1f} ± {r['rec_std']:.1f} |")
    out_lines.append("")

    # 8) Derived multipliers used in §1 and abstract
    out_lines.append("## 8. Derived multipliers (used in §1 / abstract prose)")
    out_lines.append("")
    bp = th["batch"]
    margin_p = list(_baseline_x90(data[("p", args.bits_primary)], el_pp, "margin_only").values())
    random_p = list(_baseline_x90(data[("p", args.bits_primary)], el_pp, "random").values())
    margin_mean = float(np.mean(margin_p)) * 100 if margin_p else float("nan")
    random_mean = float(np.mean(random_p)) * 100 if random_p else float("nan")
    if bp["n"]:
        out_lines.append(f"- Batch Q-only routing fraction: **{bp['frac_mean']:.1f}%** "
                         f"(recovery {bp['rec_mean']:.1f}% ± {bp['rec_std']:.1f}%)")
    out_lines.append(f"- `margin_only` LOO $X_{{@90}}$ mean: **{margin_mean:.1f}%**")
    out_lines.append(f"- `random` LOO $X_{{@90}}$ mean: **{random_mean:.1f}%**")
    if bp["n"]:
        out_lines.append(f"- margin_only / batch ratio: **{margin_mean / bp['frac_mean']:.2f}×**")
        out_lines.append(f"- random / batch ratio: **{random_mean / bp['frac_mean']:.2f}×**")
    out_lines.append("")

    # save
    out_path = Path(args.out_path) if args.out_path else (
        _PROJECT_ROOT / "plots" / "004_input_fragility" / "paper_verification.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")
    print(f"\nReport saved: {out_path}")
    # also dump a JSON sidecar for programmatic consumption
    json_path = out_path.with_suffix(".json")
    snapshot = {
        "generated_at": now,
        "univariate_auc": {k: {"mean": float(v[0]), "std": float(v[1]), "n": v[2]}
                            for k, v in aucs.items()},
        "eligibility": {f"{prefix}_W{bits}": elig[(prefix, bits)]
                         for prefix in ("p", "s") for bits in (args.bits_primary, args.bits_secondary)},
        "threshold_calibration": th,
        "margin_mean_pct": margin_mean,
        "random_mean_pct": random_mean,
    }
    json_path.write_text(json.dumps(snapshot, indent=2))
    print(f"JSON sidecar saved: {json_path}")


if __name__ == "__main__":
    main()
