"""Script F — threshold calibration for online (single-sample) deployment.

The batch claim (route top X% by P(bad), recover 90% of gap at X = 24%) is
the paper's main deployment story; this script tests the *online* variant
where each input must be routed immediately using a fixed threshold τ on
P(bad). The question:

  How close can a τ-based decision rule get to the batch claim, given that
  τ has to be picked in advance from val (with or without labels) or set
  globally across tasks?

Strategies tested (per held-out target task, LOO LogReg fit on pooled
source-task val FP-correct rows, Q-side features only):

  natural      — τ = 0.5, the LogReg's own decision boundary.
  val_pct      — τ = 75th percentile of P(bad) on target's val,
                 LABEL-FREE per-task calibration.
  source_pct   — τ = 75th percentile of P(bad) on the POOLED source-task
                 val, LABEL-FREE single global τ.
  val_labeled  — τ on target's val chosen to reach gap-recovery ≥ 0.90
                 at minimum routing fraction. Uses target val labels;
                 represents the "what if we had a small labeled
                 calibration set per task" option.
  batch        — the F6 batch baseline: sort target's test by P(bad), take
                 top fraction, find smallest routing fraction X for which
                 gap-recovery ≥ 0.90.  This is the reference.

Per task and per strategy we report (routing fraction on test,
gap-recovery on test); aggregate gives mean ± std.

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


Q_SIDE_FEATURES = ["q_margin", "q_softmax_top1", "q_entropy"]
TARGET_BATCH_RECOVERY = 0.90  # the gap-recovery level the batch claim targets


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
    p.add_argument("--min-bad-val", type=int, default=10)
    p.add_argument("--min-bad-test", type=int, default=10)
    p.add_argument("--global-percentile", type=float, default=75.0,
                   help="Percentile (on POOLED source val P(bad)) for the source_pct strategy.")
    p.add_argument("--target-percentile", type=float, default=75.0,
                   help="Percentile (on target val P(bad)) for the val_pct strategy.")
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
    return sorted(
        ds_dir.name for ds_dir in base.iterdir()
        if ds_dir.is_dir() and (ds_dir / optim_tag / ptq_tag / seed_tag / "predictions_test.parquet").exists()
    )


def _load_task(checkpoint_base, sanitized, dataset, optim_tag, ptq_tag, seed_tag):
    d = (checkpoint_base / "vision" / "ilharco_timm_supervised"
         / "input_fragility_dumps" / sanitized / dataset / optim_tag / ptq_tag / seed_tag)
    return pd.read_parquet(d / "predictions_val.parquet"), pd.read_parquet(d / "predictions_test.parquet")


def _per_task_standardise(X):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    X_norm = (X - mu) / sigma_safe
    X_norm = np.where(np.isnan(X_norm), 0.0, X_norm)
    return X_norm.astype(np.float64), mu, sigma_safe


def _apply_standardise(X, mu, sigma):
    X_norm = (X - mu) / sigma
    return np.where(np.isnan(X_norm), 0.0, X_norm).astype(np.float64)


def _route_with_tau(scores, tau, fp_correct, q_correct):
    """Apply fixed-τ routing: route inputs with P(bad) > tau to FP."""
    routed = scores > tau
    correct = np.where(routed, fp_correct, q_correct)
    fraction_routed = float(routed.mean())
    accuracy = float(correct.mean())
    return fraction_routed, accuracy


def _batch_x_at_recovery(scores, fp_correct, q_correct, q_acc, fp_acc, target_recovery):
    """Smallest sort-and-route X for which gap-recovery >= target. Returns (X, accuracy)."""
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
    accs = correct.astype(np.float64) / n
    fracs = np.arange(n + 1) / n
    gap = fp_acc - q_acc
    if gap <= 1e-9:
        return float("nan"), float("nan")
    recs = (accs - q_acc) / gap
    idx = np.searchsorted(recs, target_recovery)
    if idx >= len(fracs):
        return float("nan"), float(accs[-1])
    return float(fracs[idx]), float(accs[idx])


def _val_labeled_tau(val_scores, val_fp_correct, val_q_correct, target_recovery):
    """On val, pick the smallest τ such that gap-recovery on val ≥ target."""
    n = len(val_scores)
    fp_acc = float(val_fp_correct.mean())
    q_acc = float(val_q_correct.mean())
    gap = fp_acc - q_acc
    if gap <= 1e-9:
        return None
    # Sweep candidate thresholds = the unique val scores (plus +inf)
    candidates = np.unique(val_scores)[::-1]  # descending
    candidates = np.concatenate((candidates, [-np.inf]))
    for tau in candidates:
        routed = val_scores > tau
        correct = np.where(routed, val_fp_correct, val_q_correct)
        acc = correct.mean()
        if (acc - q_acc) / gap >= target_recovery:
            return float(tau)
    return None


def _train_q_only_loo(target, task_data, eligible, min_bad_val):
    """Fit Q-only LogReg on pooled val FP-correct rows from every eligible task ≠ target.
    Returns (clf, source_val_scores) where source_val_scores is the LogReg's predicted
    P(bad) on the pooled source-task val rows (used for the global-τ strategy)."""
    cols = Q_SIDE_FEATURES
    pieces, ys = [], []
    source_val_pieces = []
    for ds in eligible:
        if ds == target:
            continue
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < min_bad_val:
            continue
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xv_norm, mu, sigma = _per_task_standardise(Xv)
        pieces.append(Xv_norm)
        ys.append(yv)
        source_val_pieces.append((Xv_norm, mu, sigma))
    X_pool = np.concatenate(pieces, axis=0)
    y_pool = np.concatenate(ys, axis=0)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_pool, y_pool)
    pool_scores = clf.predict_proba(X_pool)[:, 1]
    return clf, pool_scores


def main():
    args = parse_args()
    checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag = _build_paths(args)
    datasets = args.datasets or _discover_tasks(checkpoint_base, sanitized, optim_tag, ptq_tag, seed_tag)
    if not datasets:
        print(f"No dumps under {checkpoint_base}/.../input_fragility_dumps/{sanitized}/", file=sys.stderr)
        sys.exit(1)

    print(f"Loading parquets for {len(datasets)} task(s) …")
    task_data = {ds: _load_task(checkpoint_base, sanitized, ds, optim_tag, ptq_tag, seed_tag) for ds in datasets}

    eligible = []
    for ds, (df_val, df_test) in task_data.items():
        if int(df_val["bad"].sum()) >= args.min_bad_val and int(df_test["bad"].sum()) >= args.min_bad_test:
            eligible.append(ds)
    print(f"{len(eligible)} eligible tasks: {eligible}\n")

    strategies = ["batch", "natural", "val_pct", "source_pct", "val_labeled"]
    results = {s: {"frac": [], "rec": []} for s in strategies}
    per_task_rows = []

    for target in eligible:
        df_val_tgt, df_test_tgt = task_data[target]
        # Standardise target val + test using target val statistics (label-free).
        cols = Q_SIDE_FEATURES
        Xv_tgt = np.array(df_val_tgt[cols].to_numpy(dtype=np.float64), copy=True)
        # Standardisation should use ALL target val (not just FP-correct); the LogReg
        # was trained on per-task-standardised FP-correct rows of OTHER tasks. To match
        # that statistical regime at scoring time, standardise the target's full val
        # using stats from the FP-correct subset (the natural match).
        vf_tgt = df_val_tgt[df_val_tgt["fp_correct"]].reset_index(drop=True)
        Xv_fpc = np.array(vf_tgt[cols].to_numpy(dtype=np.float64), copy=True)
        _, mu_tgt, sigma_tgt = _per_task_standardise(Xv_fpc)

        Xv_tgt_norm = _apply_standardise(Xv_tgt, mu_tgt, sigma_tgt)
        Xt_tgt = np.array(df_test_tgt[cols].to_numpy(dtype=np.float64), copy=True)
        Xt_tgt_norm = _apply_standardise(Xt_tgt, mu_tgt, sigma_tgt)

        clf, source_pool_scores = _train_q_only_loo(target, task_data, eligible, args.min_bad_val)

        val_scores_full = clf.predict_proba(Xv_tgt_norm)[:, 1]
        test_scores = clf.predict_proba(Xt_tgt_norm)[:, 1]

        fp_test_correct = df_test_tgt["fp_correct"].to_numpy(dtype=bool)
        q_test_correct = df_test_tgt["q_correct"].to_numpy(dtype=bool)
        fp_test_acc = float(fp_test_correct.mean())
        q_test_acc = float(q_test_correct.mean())
        gap = fp_test_acc - q_test_acc

        fp_val_correct = df_val_tgt["fp_correct"].to_numpy(dtype=bool)
        q_val_correct = df_val_tgt["q_correct"].to_numpy(dtype=bool)

        # ----- batch: smallest sort-routing X for which test gap-recovery >= 0.90 -----
        x_batch, acc_batch = _batch_x_at_recovery(
            test_scores, fp_test_correct, q_test_correct, q_test_acc, fp_test_acc, TARGET_BATCH_RECOVERY,
        )
        rec_batch = ((acc_batch - q_test_acc) / gap) if gap > 1e-9 else float("nan")

        # ----- natural: τ = 0.5 -----
        f_nat, a_nat = _route_with_tau(test_scores, 0.5, fp_test_correct, q_test_correct)
        rec_nat = (a_nat - q_test_acc) / gap if gap > 1e-9 else float("nan")

        # ----- val_pct: label-free per-task τ from target's full val -----
        tau_vp = float(np.percentile(val_scores_full, args.target_percentile))
        f_vp, a_vp = _route_with_tau(test_scores, tau_vp, fp_test_correct, q_test_correct)
        rec_vp = (a_vp - q_test_acc) / gap if gap > 1e-9 else float("nan")

        # ----- source_pct: label-free single global τ from POOLED source val -----
        tau_sp = float(np.percentile(source_pool_scores, args.global_percentile))
        f_sp, a_sp = _route_with_tau(test_scores, tau_sp, fp_test_correct, q_test_correct)
        rec_sp = (a_sp - q_test_acc) / gap if gap > 1e-9 else float("nan")

        # ----- val_labeled: per-task τ from val with labels (min routing fraction for ≥0.90 recovery on val) -----
        tau_lab = _val_labeled_tau(val_scores_full, fp_val_correct, q_val_correct, TARGET_BATCH_RECOVERY)
        if tau_lab is None:
            f_lab, a_lab, rec_lab = float("nan"), float("nan"), float("nan")
        else:
            f_lab, a_lab = _route_with_tau(test_scores, tau_lab, fp_test_correct, q_test_correct)
            rec_lab = (a_lab - q_test_acc) / gap if gap > 1e-9 else float("nan")

        results["batch"]["frac"].append(x_batch)
        results["batch"]["rec"].append(rec_batch)
        results["natural"]["frac"].append(f_nat)
        results["natural"]["rec"].append(rec_nat)
        results["val_pct"]["frac"].append(f_vp)
        results["val_pct"]["rec"].append(rec_vp)
        results["source_pct"]["frac"].append(f_sp)
        results["source_pct"]["rec"].append(rec_sp)
        results["val_labeled"]["frac"].append(f_lab)
        results["val_labeled"]["rec"].append(rec_lab)

        per_task_rows.append({
            "task": target,
            "gap": gap,
            "batch_frac": x_batch, "batch_rec": rec_batch,
            "natural_frac": f_nat, "natural_rec": rec_nat,
            "val_pct_frac": f_vp, "val_pct_rec": rec_vp,
            "source_pct_frac": f_sp, "source_pct_rec": rec_sp,
            "val_labeled_frac": f_lab, "val_labeled_rec": rec_lab,
        })

    # Markdown report
    md = [f"# Threshold calibration for online routing — {args.model_name} | W{args.bits} {args.granularity}\n"]
    md.append(f"Q-only LogReg (3 features). Target batch recovery = {TARGET_BATCH_RECOVERY*100:.0f}%.\n")
    md.append(f"`batch` = sort-and-take X strategy (paper headline). Other rows are online τ-based variants.\n")

    md.append("\n## Aggregate across tasks (mean ± std)\n")
    md.append("| strategy | what it needs at deploy time | mean fraction routed | mean gap recovery |")
    md.append("|---|---|---|---|")
    desc = {
        "batch":       "full target batch (sort it)",
        "natural":     "nothing (τ = 0.5)",
        "val_pct":     "target val P(bad) (label-free)",
        "source_pct":  "source pool P(bad) (one global τ)",
        "val_labeled": "target val labels + P(bad) (labeled calibration)",
    }
    for s in strategies:
        fr = np.array(results[s]["frac"], dtype=float)
        rc = np.array(results[s]["rec"], dtype=float)
        fr = fr[~np.isnan(fr)]
        rc = rc[~np.isnan(rc)]
        if len(fr) == 0:
            md.append(f"| {s} | {desc[s]} | — | — |")
        else:
            md.append(f"| `{s}` | {desc[s]} | {fr.mean()*100:.1f}% ± {fr.std()*100:.1f} | {rc.mean()*100:.1f}% ± {rc.std()*100:.1f} |")

    md.append("\n## Per-task detail\n")
    md.append("| task | gap (pp) | batch (frac, rec) | natural | val_pct | source_pct | val_labeled |")
    md.append("|---|---|---|---|---|---|---|")
    for r in per_task_rows:
        def pair(f, rc):
            return "—" if (np.isnan(f) or np.isnan(rc)) else f"{f*100:.1f}%, {rc*100:.1f}%"
        md.append(
            f"| {r['task']} | {r['gap']*100:+.2f} | {pair(r['batch_frac'], r['batch_rec'])} "
            f"| {pair(r['natural_frac'], r['natural_rec'])} "
            f"| {pair(r['val_pct_frac'], r['val_pct_rec'])} "
            f"| {pair(r['source_pct_frac'], r['source_pct_rec'])} "
            f"| {pair(r['val_labeled_frac'], r['val_labeled_rec'])} |"
        )

    markdown = "\n".join(md)

    # HTML: scatter of (fraction routed, gap recovery) per task per strategy
    fig = go.Figure()
    colors = {"batch": "#000000", "natural": "#9467bd", "val_pct": "#1f77b4",
              "source_pct": "#ff7f0e", "val_labeled": "#2ca02c"}
    for s in strategies:
        fr = np.array(results[s]["frac"], dtype=float) * 100
        rc = np.array(results[s]["rec"], dtype=float) * 100
        fig.add_trace(go.Scatter(x=fr, y=rc, mode="markers", name=s,
                                  marker=dict(size=10, color=colors[s])))
    fig.add_hline(y=90, line_dash="dot", line_color="green",
                  annotation_text="target recovery 90%", annotation_position="right")
    fig.update_xaxes(title_text="fraction of test routed to FP (%)", range=[0, 100])
    fig.update_yaxes(title_text="gap recovery (%)", range=[-10, 110])
    fig.update_layout(title=f"Threshold strategies — W{args.bits} {args.granularity} ({len(per_task_rows)} tasks)",
                      height=600)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "plots" / "004_input_fragility"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"threshold_calibration_{sanitized}_bits{args.bits}_{args.granularity}.md"
    html_path = out_dir / f"threshold_calibration_{sanitized}_bits{args.bits}_{args.granularity}.html"
    md_path.write_text(markdown)
    fig.write_html(str(html_path))

    print(markdown)
    print()
    print(f"Markdown saved: {md_path}")
    print(f"HTML saved:     {html_path}")


if __name__ == "__main__":
    main()
