"""Generate publication-quality LaTeX tables for the paper appendix.

Reads the same parquets that fed the figures and emits two .tex tables under
paper/tables/:

  appendix_task_stats.tex     — per-task counts (n_val, n_test, n_bad on val/test),
                                FP test accuracy, PTQ test accuracy, gap. One row
                                per task; both W4-channel and W3-channel reported.

  appendix_pareto_ablation.tex — per-task LOO X@90% under each feature subset
                                (image_only, q_only, fp_only, fp_plus_q_no_cross,
                                 all_features, oracle, random). One row per task;
                                W4-channel.

Both tables are emitted as standalone \\begin{table}...\\end{table} blocks so
the main paper can \\input{} them directly. The TODO macros in main.tex are
replaced with these \\input{} calls.

Argparse, no Hydra, no GPU. ~30 s end-to-end.
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
from sklearn.linear_model import LogisticRegression

from types import SimpleNamespace

from src.vision.utils import sanitize_timm_model_name


OUT_DIR = _PROJECT_ROOT / "paper" / "tables"


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

SUBSET_HEADERS = {
    "image_only":         "image",
    "q_only":             "Q-only",
    "fp_only":            "FP-only",
    "fp_plus_q_no_cross": "FP+Q",
    "all_features":       "all",
}


def parse_args():
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
    p.add_argument("--bits-primary", type=int, default=4)
    p.add_argument("--bits-secondary", type=int, default=3)
    p.add_argument("--granularity", default="channel")
    p.add_argument("--min-bad-test", type=int, default=10)
    p.add_argument("--min-bad-val", type=int, default=10)
    return p.parse_args()


def _build_cfg(args):
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    return SimpleNamespace(
        model_name=args.model_name,
        sanitized=sanitize_timm_model_name(args.model_name),
        lr=args.lr, wd=args.wd, ls=args.ls, wl=args.wl,
        mgn=args.max_grad_norm, bs=args.batch_size, seed=args.seed,
        skip_tag=skip_tag,
    )


def _dump_dir(cfg, dataset: str, bits: int, granularity: str) -> Path:
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


def _load_all(cfg, bits: int, granularity: str):
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
        meta_path = d / "dump_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        out[ds_dir.name] = (df_val, df_test, meta)
    return out


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
    total_q = q_cum[-1]
    correct = fp_cum + (total_q - q_cum)
    return np.arange(n + 1) / n, correct.astype(np.float64) / n


def _x_at_recovery(frac, rec, target):
    above = np.where(rec >= target)[0]
    return float("nan") if len(above) == 0 else float(frac[above[0]])


def _loo_x90_for_subset(task_data, cols, eligible, min_bad_val):
    """Returns {task: X@90} under LOO with the given feature subset."""
    per_task_train = {}
    for ds in eligible:
        df_val, _, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < min_bad_val:
            continue
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sigma = _per_task_z(Xv)
        per_task_train[ds] = {"Xn": Xn, "y": yv, "mu": mu, "sigma": sigma}

    out = {}
    for target in per_task_train:
        Xs = [per_task_train[t]["Xn"] for t in per_task_train if t != target]
        ys = [per_task_train[t]["y"] for t in per_task_train if t != target]
        X_pool = np.concatenate(Xs, axis=0)
        y_pool = np.concatenate(ys, axis=0)
        if y_pool.sum() < 5 or (len(y_pool) - y_pool.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_pool, y_pool)
        df_test = task_data[target][1]
        X_test = np.array(df_test[cols].to_numpy(dtype=np.float64), copy=True)
        X_test_norm = _apply_z(X_test, per_task_train[target]["mu"], per_task_train[target]["sigma"])
        scores = clf.predict_proba(X_test_norm)[:, 1]
        fp_c = df_test["fp_correct"].to_numpy(dtype=bool)
        q_c = df_test["q_correct"].to_numpy(dtype=bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        frac, acc = _routed_curve(scores, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        out[target] = _x_at_recovery(frac, rec, 0.9)
    return out


def _baseline_x90(task_data, eligible, kind: str):
    """kind in {oracle, random}."""
    out = {}
    for target in eligible:
        df_test = task_data[target][1]
        fp_c = df_test["fp_correct"].to_numpy(dtype=bool)
        q_c = df_test["q_correct"].to_numpy(dtype=bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        if kind == "oracle":
            s = fp_c.astype(int) - q_c.astype(int)
        elif kind == "random":
            s = np.random.default_rng(0).random(len(df_test))
        else:
            raise ValueError(kind)
        frac, acc = _routed_curve(s, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        out[target] = _x_at_recovery(frac, rec, 0.9)
    return out


def _fmt_pct(x, prec=1):
    if x is None or (isinstance(x, float) and (np.isnan(x) or x is float("nan"))):
        return "---"
    if isinstance(x, float) and np.isnan(x):
        return "---"
    return f"{x * 100:.{prec}f}\\%"


def _fmt_count(x):
    if x is None:
        return "---"
    return f"{int(x):,}"


# ============================================================================
# Table A1: per-task statistics
# ============================================================================
def emit_task_stats_table(cfg, args):
    print("Building per-task statistics table ...")
    primary = _load_all(cfg, args.bits_primary, args.granularity)
    secondary = _load_all(cfg, args.bits_secondary, args.granularity)
    all_tasks = sorted(set(primary.keys()) | set(secondary.keys()))

    rows = []
    for ds in all_tasks:
        rec = {"task": ds}
        # W4 stats
        if ds in primary:
            dv, dt, _ = primary[ds]
            rec["n_val"]      = len(dv)
            rec["n_test"]     = len(dt)
            rec["n_bad_w4"]   = int(dt["bad"].sum())
            rec["fp_w4"]      = float(dt["fp_correct"].mean())
            rec["q_w4"]       = float(dt["q_correct"].mean())
            rec["eligible_w4"] = (int(dv["bad"].sum()) >= args.min_bad_val
                                  and rec["n_bad_w4"] >= args.min_bad_test)
        # W3 stats
        if ds in secondary:
            dv2, dt2, _ = secondary[ds]
            rec["n_bad_w3"]   = int(dt2["bad"].sum())
            rec["fp_w3"]      = float(dt2["fp_correct"].mean())
            rec["q_w3"]       = float(dt2["q_correct"].mean())
            rec["eligible_w3"] = (int(dv2["bad"].sum()) >= args.min_bad_val
                                  and rec["n_bad_w3"] >= args.min_bad_test)
        rows.append(rec)

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append("% Per-task statistics at W4-channel and W3-channel.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Per-task dataset statistics and FP/PTQ test accuracy at "
        f"W{args.bits_primary}-{args.granularity} and W{args.bits_secondary}-{args.granularity}. "
        "$n_{\\text{bad}}$ is the count of test inputs that FP gets right and PTQ flips. "
        "$\\Delta$ is the FP$\\to$PTQ test-accuracy gap (positive $=$ PTQ drop). "
        "Tasks marked $\\dagger$ are excluded from the analysis at that regime because "
        f"$n_{{\\text{{bad}}}} < {args.min_bad_test}$ (PTQ barely degrades them).}}"
    )
    lines.append("\\label{tab:task_stats}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{lrrrrrrrrrr}")
    lines.append("\\toprule")
    lines.append(f"& & & \\multicolumn{{4}}{{c}}{{W{args.bits_primary}-{args.granularity}}} "
                 f"& \\multicolumn{{4}}{{c}}{{W{args.bits_secondary}-{args.granularity}}} \\\\")
    lines.append("\\cmidrule(lr){4-7} \\cmidrule(lr){8-11}")
    lines.append("Task & $n_{\\text{val}}$ & $n_{\\text{test}}$ "
                 "& $n_{\\text{bad}}$ & FP & PTQ & $\\Delta$ "
                 "& $n_{\\text{bad}}$ & FP & PTQ & $\\Delta$ \\\\")
    lines.append("\\midrule")
    for r in rows:
        task = r["task"]
        nv = _fmt_count(r.get("n_val"))
        nt = _fmt_count(r.get("n_test"))
        # W4 group
        if "n_bad_w4" in r:
            nb4 = f"{r['n_bad_w4']:,}" + ("" if r["eligible_w4"] else "$^{\\dagger}$")
            fp4 = _fmt_pct(r["fp_w4"], 1)
            q4  = _fmt_pct(r["q_w4"], 1)
            d4  = f"{(r['fp_w4'] - r['q_w4']) * 100:+.2f}"
        else:
            nb4, fp4, q4, d4 = "---", "---", "---", "---"
        # W3 group
        if "n_bad_w3" in r:
            nb3 = f"{r['n_bad_w3']:,}" + ("" if r["eligible_w3"] else "$^{\\dagger}$")
            fp3 = _fmt_pct(r["fp_w3"], 1)
            q3  = _fmt_pct(r["q_w3"], 1)
            d3  = f"{(r['fp_w3'] - r['q_w3']) * 100:+.2f}"
        else:
            nb3, fp3, q3, d3 = "---", "---", "---", "---"
        lines.append(f"{task} & {nv} & {nt} & {nb4} & {fp4} & {q4} & {d4} & {nb3} & {fp3} & {q3} & {d3} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path = OUT_DIR / f"appendix_task_stats_{cfg.sanitized}.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


# ============================================================================
# Table A2: per-task X@90 across feature subsets
# ============================================================================
def emit_pareto_ablation_table(cfg, args):
    print("Building per-task LOO X@90 ablation table ...")
    task_data = _load_all(cfg, args.bits_primary, args.granularity)

    eligible = []
    for ds, (dv, dt, _) in task_data.items():
        if int(dv["bad"].sum()) >= args.min_bad_val and int(dt["bad"].sum()) >= args.min_bad_test:
            eligible.append(ds)

    print(f"  {len(eligible)} eligible tasks")

    # Compute per-task X@90 for each subset, plus oracle + random
    results: dict[str, dict[str, float]] = {ds: {} for ds in eligible}
    for subset_name, cols in SUBSETS.items():
        x90 = _loo_x90_for_subset(task_data, cols, eligible, args.min_bad_val)
        for ds, v in x90.items():
            results[ds][subset_name] = v
    oracle_x = _baseline_x90(task_data, eligible, "oracle")
    random_x = _baseline_x90(task_data, eligible, "random")
    for ds, v in oracle_x.items():
        results[ds]["oracle"] = v
    for ds, v in random_x.items():
        results[ds]["random"] = v

    # Mean across tasks per column
    columns = list(SUBSETS.keys()) + ["oracle", "random"]
    means = {}
    for col in columns:
        vals = [results[ds][col] for ds in eligible if col in results[ds] and not np.isnan(results[ds][col])]
        means[col] = float(np.mean(vals)) if vals else float("nan")
        means[col + "_std"] = float(np.std(vals)) if vals else float("nan")

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% Per-task LOO X@90 at W{args.bits_primary}-{args.granularity}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Per-task LOO $X_{{@90\\%}}$ at W{args.bits_primary}-{args.granularity}, "
        "across every feature subset plus oracle and random baselines. Lower is better. "
        "\\texttt{q\\_only} is the deployable (PTQ-first) recipe; \\texttt{all\\_features} is the "
        "diagnostic ceiling requiring both \\fp{} and \\ptq{} forward passes. "
        "The discrepancy between \\texttt{q\\_only} and \\texttt{all\\_features} "
        "quantifies the \\luckyq{} ambiguity per task.}"
    )
    lines.append("\\label{tab:pareto_per_task}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + "r" * len(columns) + "}")
    lines.append("\\toprule")
    header = "Task"
    for col in columns:
        header += f" & {SUBSET_HEADERS.get(col, col)}"
    lines.append(header + " \\\\")
    lines.append("\\midrule")
    for ds in sorted(eligible):
        row = [ds]
        for col in columns:
            v = results[ds].get(col, float("nan"))
            row.append(_fmt_pct(v, 1) if not np.isnan(v) else "---")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    mean_row = ["\\textbf{mean}"]
    for col in columns:
        m = means[col]
        s = means[col + "_std"]
        mean_row.append(f"{m * 100:.1f} $\\pm$ {s * 100:.1f}\\%" if not np.isnan(m) else "---")
    lines.append(" & ".join(mean_row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tag = f"{cfg.sanitized}_bits{args.bits_primary}_{args.granularity}"
    out_path = OUT_DIR / f"appendix_pareto_ablation_{tag}.tex"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    cfg = _build_cfg(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emit_task_stats_table(cfg, args)
    emit_pareto_ablation_table(cfg, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
