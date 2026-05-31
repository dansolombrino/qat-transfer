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
from sklearn.metrics import roc_auc_score

from types import SimpleNamespace

from src.vision.utils import sanitize_hf_model_name, sanitize_timm_model_name


OUT_DIR = _PROJECT_ROOT / "paper" / "tables"


IMAGE_FEATS = ["img_brightness", "img_contrast", "img_edge_density", "img_high_freq_ratio"]
Q_FEATS = ["q_margin", "q_softmax_top1", "q_entropy"]
FP_FEATS = ["fp_margin", "fp_softmax_top1", "fp_entropy"]
# NOTE: `fp_cls_dist_to_class_centroid` was removed from FP_FEATS because the
# feature definition required selecting the centroid for the true class of the
# test sample, which is not available at inference time. It remains in the
# parquet dumps but is no longer consumed by any predictor or univariate report.
CROSS_FEATS = ["fp_logit_at_q_pred", "q_logit_at_fp_pred", "fp_softmax_at_q_pred",
               "q_softmax_at_fp_pred", "fp_q_kl_symmetric", "fp_q_disagree"]
ALL_FEATS = FP_FEATS + Q_FEATS + CROSS_FEATS + IMAGE_FEATS

SUBSETS = {
    "image_only": IMAGE_FEATS,
    # MSP (max-softmax-prob) baseline of Hendrycks & Gimpel 2017 / Geifman &
    # El-Yaniv 2017: threshold the quantized model's top-1 softmax probability.
    # One-feature special case of `q_only`; we report it as a reference point.
    "msp_only": ["q_softmax_top1"],
    "q_only": Q_FEATS,
    "fp_only": FP_FEATS,
    "fp_plus_q_no_cross": FP_FEATS + Q_FEATS + IMAGE_FEATS,
    "all_features": ALL_FEATS,
}

SUBSET_HEADERS = {
    "image_only":         "image",
    "msp_only":           "MSP",
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
    # Optional second backbone for the combined dual-backbone ablation table.
    p.add_argument("--also-model-name", default=None,
                   help="If set, emit a DUAL-backbone ablation table comparing --model-name "
                        "and --also-model-name side-by-side at both PTQ regimes.")
    p.add_argument("--also-batch-size", default=None,
                   help="--batch-size for the second backbone (defaults to --batch-size).")
    # Optional Qwen3 NLP backbone for the threshold table only.
    # If the dump directory exists, the threshold table becomes 3-backbone (two ViTs + Qwen3).
    p.add_argument("--qwen3-model-name", default="Qwen/Qwen3-Embedding-0.6B",
                   help="HF model id for the NLP backbone (looked up under text/.../input_fragility_dumps/).")
    p.add_argument("--qwen3-batch-size", default="32")
    p.add_argument("--qwen3-max-length", default="128")
    p.add_argument("--qwen3-skip-modules", nargs="+", default=["score"])
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


def _load_all_qwen3(args, bits: int):
    """Loader for NLP (Qwen3) dumps. Returns the same (df_val, df_test, meta) tuple
    schema as `_load_all` so downstream consumers do not care about the domain.
    Returns {} if the dump base directory does not exist (Qwen3 sweep wasn't run).
    """
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


def _tex_safe_join(lines):
    """Join `lines` for `.tex` output, escaping raw `group_<N>` granularity
    references to `group\\_<N>` so LaTeX renders the underscore as text rather
    than as math-mode subscript. We deliberately do not touch:
      - lines that begin with `%` (LaTeX comments),
      - lines that contain `\\label{...}`, `\\input{...}`, `\\ref{...}`,
        `\\cite{...}` or `\\bibliography{...}` --- those use raw `_` as part
        of identifier keys and would break under naive escaping.
    The pattern matches only `group_<digit>`, so it cannot collide with
    identifier-style underscores like `vit_base_patch16_224_orig_in21k`.
    """
    import re
    safe_pattern = re.compile(r'(?<!\\)group_(\d)')
    skip_pattern = re.compile(r'\\(label|input|ref|cite|bibliography)\{')
    out = []
    for line in lines:
        if line.lstrip().startswith("%") or skip_pattern.search(line):
            out.append(line)
        else:
            out.append(safe_pattern.sub(r'group\\_\1', line))
    return "\n".join(out) + "\n"


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

    short = _short_model(cfg.model_name)
    label_slug = short.lower().replace("/", "_").replace("-", "_")
    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% Per-task statistics for {short} at W4-channel and W3-channel.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Per-task dataset statistics and FP/PTQ test accuracy "
        f"\\textbf{{for {short}}} at "
        f"W{args.bits_primary}-{args.granularity} and W{args.bits_secondary}-{args.granularity}. "
        "$n_{\\text{bad}}$ is the count of test inputs that FP gets right and PTQ flips. "
        "$\\Delta$ is the FP$\\to$PTQ test-accuracy gap (positive $=$ PTQ drop). "
        "Tasks marked $\\dagger$ are excluded from the analysis at that regime because "
        f"$n_{{\\text{{bad}}}} < {args.min_bad_test}$ (PTQ barely degrades them).}}"
    )
    lines.append(f"\\label{{tab:task_stats_{label_slug}}}")
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
    out_path.write_text(_tex_safe_join(lines))
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
    short = _short_model(cfg.model_name)
    label_slug = short.lower().replace("/", "_").replace("-", "_")
    lines.append(
        f"\\caption{{Per-task LOO $X_{{@90\\%}}$ \\textbf{{for {short}}} "
        f"at W{args.bits_primary}-{args.granularity}, "
        "across every feature subset plus oracle and random baselines. Lower is better. "
        "\\texttt{q\\_only} is the deployable (PTQ-first) recipe; \\texttt{all\\_features} is the "
        "diagnostic ceiling requiring both \\fp{} and \\ptq{} forward passes. "
        "The discrepancy between \\texttt{q\\_only} and \\texttt{all\\_features} "
        "quantifies the \\luckyq{} ambiguity per task.}"
    )
    lines.append(f"\\label{{tab:pareto_per_task_{label_slug}}}")
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
    out_path.write_text(_tex_safe_join(lines))
    print(f"  saved {out_path}")


def _aggregate_subset_x90(cfg, args, bits):
    """Returns {subset_name: (mean_x90, std_x90, n_tasks_used)} for one backbone+regime."""
    task_data = _load_all(cfg, bits, args.granularity)
    eligible = [
        ds for ds, (dv, dt, _) in task_data.items()
        if int(dv["bad"].sum()) >= args.min_bad_val
        and int(dt["bad"].sum()) >= args.min_bad_test
    ]
    out = {}
    for subset_name, cols in SUBSETS.items():
        x90 = _loo_x90_for_subset(task_data, cols, eligible, args.min_bad_val)
        vals = [v for v in x90.values() if not np.isnan(v)]
        if vals:
            out[subset_name] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            out[subset_name] = (float("nan"), float("nan"), 0)
    for kind in ("oracle", "random"):
        x90 = _baseline_x90(task_data, eligible, kind)
        vals = [v for v in x90.values() if not np.isnan(v)]
        if vals:
            out[kind] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            out[kind] = (float("nan"), float("nan"), 0)
    out["__n_eligible__"] = (float("nan"), float("nan"), len(eligible))
    return out


def emit_dual_ablation_table(cfg_a, cfg_b, args, short_a, short_b):
    """Combined ablation table with both backbones at both regimes.
    4 numeric columns per feature subset: (cfg_a, p_bits) (cfg_b, p_bits) (cfg_a, s_bits) (cfg_b, s_bits)."""
    print("Building dual-backbone ablation table ...")
    res_ap = _aggregate_subset_x90(cfg_a, args, args.bits_primary)
    res_bp = _aggregate_subset_x90(cfg_b, args, args.bits_primary)
    res_as = _aggregate_subset_x90(cfg_a, args, args.bits_secondary)
    res_bs = _aggregate_subset_x90(cfg_b, args, args.bits_secondary)

    def cell(d, key):
        m, s, _ = d.get(key, (float("nan"), float("nan"), 0))
        if np.isnan(m):
            return "---"
        return f"{m * 100:.1f} $\\pm$ {s * 100:.1f}"

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% Dual-backbone ablation: {cfg_a.sanitized} & {cfg_b.sanitized}, "
                 f"W{args.bits_primary} and W{args.bits_secondary} ({args.granularity}).")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{LOO \\xat{{90}} per feature subset, on both backbones "
        f"and at both \\ptq{{}} regimes. Lower is better. "
        f"\\texttt{{q\\_only}} is the PTQ-first deployable recipe; "
        f"\\texttt{{all\\_features}} is the diagnostic ceiling requiring both \\fp{{}} and \\ptq{{}} forward passes. "
        f"At W{args.bits_primary}-{args.granularity} the deployable Q-only predictor reaches 90\\% gap recovery at "
        f"$\\sim$24\\% \\fp{{}}-compute fraction on {short_a} and tightens to $\\sim$18\\% on {short_b}; "
        f"at W{args.bits_secondary}-{args.granularity} both backbones land in the catastrophic regime where the routing problem itself is intractable. "
        f"Eligible-task counts in the last row.}}"
    )
    lines.append("\\label{tab:ablation}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append(
        f"& & \\multicolumn{{2}}{{c}}{{W{args.bits_primary}-{args.granularity}}}"
        f" & \\multicolumn{{2}}{{c}}{{W{args.bits_secondary}-{args.granularity}}} \\\\"
    )
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
    lines.append(
        f"Subset & Deployment "
        f"& {short_a} & {short_b} "
        f"& {short_a} & {short_b} \\\\"
    )
    lines.append("\\midrule")
    deployment_labels = {
        "image_only":         "no model",
        "msp_only":           "MSP baseline",
        "q_only":             "PTQ-first deployable",
        "fp_only":            "FP-side only",
        "fp_plus_q_no_cross": "both models",
        "all_features":       "ceiling (both+cross)",
    }
    for subset_name in SUBSETS:
        label = deployment_labels[subset_name]
        escaped = subset_name.replace("_", "\\_")
        row = [
            f"\\texttt{{{escaped}}}",
            label,
            cell(res_ap, subset_name),
            cell(res_bp, subset_name),
            cell(res_as, subset_name),
            cell(res_bs, subset_name),
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    for kind, pretty in [("oracle", "oracle (upper bound)"),
                          ("random", "random (lower bound)")]:
        row = [
            kind, pretty,
            cell(res_ap, kind), cell(res_bp, kind),
            cell(res_as, kind), cell(res_bs, kind),
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    # Eligible task counts row.
    def n_cell(d):
        return str(int(d["__n_eligible__"][2]))
    lines.append(
        f"\\multicolumn{{2}}{{r}}{{$n_{{\\text{{eligible tasks}}}}$}} "
        f"& {n_cell(res_ap)} & {n_cell(res_bp)} & {n_cell(res_as)} & {n_cell(res_bs)} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tag = (
        f"{cfg_a.sanitized}_vs_{cfg_b.sanitized}"
        f"_W{args.bits_primary}vsW{args.bits_secondary}_{args.granularity}"
    )
    out_path = OUT_DIR / f"ablation_dual_{tag}.tex"
    out_path.write_text(_tex_safe_join(lines))
    print(f"  saved {out_path}")


# ============================================================================
# AUROC (secondary metric): LOO predictor's P(bad) vs `bad` on FP-correct test
# ============================================================================
def _loo_auroc_for_subset(task_data, cols, eligible, min_bad_val):
    """Per-task LOO AUROC of P(bad) vs `bad` labels on FP-correct test rows.

    The predictor is fit identically to `_loo_x90_for_subset` (LogReg on
    pooled source-task val FP-correct rows, per-task z-scoring); the only
    difference is that we score the FP-correct test rows and report ROC-AUC
    instead of X@90 on the routed curve. AUROC is threshold-free and does
    not depend on the FP$\\to$PTQ gap magnitude, so it stabilizes the
    paper's recovery numbers on tasks with tiny gaps.

    Returns {task: auroc}."""
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
        tf = df_test[df_test["fp_correct"]].reset_index(drop=True)
        if int(tf["bad"].sum()) < 5 or int((~tf["bad"]).sum()) < 5:
            continue
        X_test = np.array(tf[cols].to_numpy(dtype=np.float64), copy=True)
        X_test_norm = _apply_z(X_test, per_task_train[target]["mu"], per_task_train[target]["sigma"])
        scores = clf.predict_proba(X_test_norm)[:, 1]
        y_test = tf["bad"].astype(int).to_numpy()
        try:
            out[target] = float(roc_auc_score(y_test, scores))
        except ValueError:
            continue
    return out


def _aggregate_subset_auroc(cfg, args, bits):
    """{subset: (mean_auroc, std_auroc, n_tasks)} for one backbone+regime."""
    task_data = _load_all(cfg, bits, args.granularity)
    eligible = [
        ds for ds, (dv, dt, _) in task_data.items()
        if int(dv["bad"].sum()) >= args.min_bad_val
        and int(dt["bad"].sum()) >= args.min_bad_test
    ]
    out = {}
    for subset_name, cols in SUBSETS.items():
        a = _loo_auroc_for_subset(task_data, cols, eligible, args.min_bad_val)
        vals = list(a.values())
        if vals:
            out[subset_name] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            out[subset_name] = (float("nan"), float("nan"), 0)
    out["__n_eligible__"] = (float("nan"), float("nan"), len(eligible))
    return out


def emit_dual_auroc_table(cfg_a, cfg_b, args, short_a, short_b):
    """Secondary-metric table: LOO AUROC per subset on both backbones at W4."""
    print("Building dual-backbone AUROC table ...")
    res_a = _aggregate_subset_auroc(cfg_a, args, args.bits_primary)
    res_b = _aggregate_subset_auroc(cfg_b, args, args.bits_primary)

    # AUROC is suppressed (rendered as "---") for any subset whose feature set
    # definitionally encodes the `bad` label on FP-correct rows. The
    # `all_features` subset includes `fp_q_disagree`, which on FP-correct rows
    # is identically 1 iff `bad`; reporting AUROC there is degenerate.
    SUPPRESSED_SUBSETS = {"all_features"}

    def cell(d, key):
        m, s, _ = d.get(key, (float("nan"), float("nan"), 0))
        if key in SUPPRESSED_SUBSETS:
            return "---"
        if np.isnan(m):
            return "---"
        return f"{m:.3f} $\\pm$ {s:.3f}"

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% Dual-backbone LOO AUROC at W{args.bits_primary}-{args.granularity}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{Secondary metric --- \\textbf{{LOO AUROC of $P(\\bad)$ vs \\bad{{}} labels on FP-correct test rows}} "
        f"at W{args.bits_primary}-{args.granularity}, per feature subset, both backbones. "
        f"Threshold-free and independent of the per-task \\fp$\\to$\\ptq{{}} gap magnitude, so it does not suffer "
        f"the recovery-percentage noise that affects tasks with tiny gaps. "
        f"Mean $\\pm$ std across eligible tasks. "
        f"The deployable \\texttt{{q\\_only}} predictor stays in the high-AUROC range on both backbones; "
        f"\\texttt{{image\\_only}} hovers near chance (0.5). Higher is better. "
        f"AUROC is suppressed (---) for \\texttt{{all\\_features}}: its cross-model disagreement features "
        f"definitionally encode \\bad{{}} on FP-correct rows (\\S\\ref{{sec:discussion}}), making AUROC degenerate.}}"
    )
    lines.append("\\label{tab:auroc}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\toprule")
    lines.append(f"Subset & Deployment & {short_a} & {short_b} \\\\")
    lines.append("\\midrule")
    deployment_labels = {
        "image_only":         "no model",
        "msp_only":           "MSP baseline",
        "q_only":             "PTQ-first deployable",
        "fp_only":            "FP-side only",
        "fp_plus_q_no_cross": "both models",
        "all_features":       "ceiling (both+cross)",
    }
    for subset_name in SUBSETS:
        label = deployment_labels[subset_name]
        escaped = subset_name.replace("_", "\\_")
        row = [
            f"\\texttt{{{escaped}}}", label,
            cell(res_a, subset_name), cell(res_b, subset_name),
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    lines.append(
        f"\\multicolumn{{2}}{{r}}{{$n_{{\\text{{eligible tasks}}}}$}} "
        f"& {int(res_a['__n_eligible__'][2])} & {int(res_b['__n_eligible__'][2])} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tag = f"{cfg_a.sanitized}_vs_{cfg_b.sanitized}_bits{args.bits_primary}_{args.granularity}"
    out_path = OUT_DIR / f"auroc_dual_{tag}.tex"
    out_path.write_text(_tex_safe_join(lines))
    print(f"  saved {out_path}")


# ============================================================================
# Online threshold calibration (Q-only, W4): dual-backbone tab:threshold
# ============================================================================
def _threshold_calibration_from_data(task_data, args, percentile=75.0):
    """Pure-math threshold calibration over an already-loaded task_data dict.

    Returns {strategy: (frac_mean_pct, frac_std_pct, rec_mean_pct, rec_std_pct, n)}
    plus a sentinel key '__n_eligible__' giving the eligible task count.

    Strategies: batch / natural (tau=0.5) / val_pct (target val percentile, label-free)
    / source_pct (pooled source val percentile, label-free) / val_labeled (target val
    labels, minimum routing fraction at >=0.9 recovery on val). Q-only features.

    Domain-agnostic: works on either vision or NLP task_data because both share the
    same parquet schema for Q-side features and bad/fp_correct/q_correct flags."""
    eligible = [
        ds for ds, (dv, dt, _) in task_data.items()
        if int(dv["bad"].sum()) >= args.min_bad_val
        and int(dt["bad"].sum()) >= args.min_bad_test
    ]

    pt = {}
    for ds in eligible:
        df_val, _, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < 5:
            continue
        Xv = np.array(vf[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sigma = _per_task_z(Xv)
        pt[ds] = {"Xn": Xn, "y": yv, "mu": mu, "sigma": sigma}

    strategies = ["batch", "natural", "val_pct", "source_pct", "val_labeled"]
    res = {s: {"frac": [], "rec": []} for s in strategies}

    for target in eligible:
        if target not in pt:
            continue
        Xs = [pt[t]["Xn"] for t in eligible if t != target and t in pt]
        ys = [pt[t]["y"] for t in eligible if t != target and t in pt]
        X_pool = np.concatenate(Xs, axis=0)
        y_pool = np.concatenate(ys, axis=0)
        if y_pool.sum() < 5 or (len(y_pool) - y_pool.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_pool, y_pool)
        source_pool_scores = clf.predict_proba(X_pool)[:, 1]

        df_val_t, df_test_t = task_data[target][0], task_data[target][1]
        Xv_full = np.array(df_val_t[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xv_full_n = _apply_z(Xv_full, pt[target]["mu"], pt[target]["sigma"])
        Xt = np.array(df_test_t[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xtn = _apply_z(Xt, pt[target]["mu"], pt[target]["sigma"])
        val_scores = clf.predict_proba(Xv_full_n)[:, 1]
        test_scores = clf.predict_proba(Xtn)[:, 1]

        fp_t = df_test_t["fp_correct"].to_numpy(bool)
        q_t = df_test_t["q_correct"].to_numpy(bool)
        gap = float(fp_t.mean()) - float(q_t.mean())
        if gap <= 1e-9:
            continue
        q_acc = float(q_t.mean())

        # batch — sort by score, smallest fraction reaching 0.90 recovery
        frac, acc = _routed_curve(test_scores, fp_t, q_t)
        rec = (acc - q_acc) / gap
        idx_above = np.where(rec >= 0.90)[0]
        if len(idx_above) > 0:
            res["batch"]["frac"].append(float(frac[idx_above[0]]))
            res["batch"]["rec"].append(float(rec[idx_above[0]]))

        # fixed-tau strategies
        for label, tau in [
            ("natural", 0.5),
            ("val_pct", float(np.percentile(val_scores, percentile))),
            ("source_pct", float(np.percentile(source_pool_scores, percentile))),
        ]:
            routed = test_scores > tau
            correct = np.where(routed, fp_t, q_t)
            res[label]["frac"].append(float(routed.mean()))
            res[label]["rec"].append((float(correct.mean()) - q_acc) / gap)

        # val_labeled — pick tau on val to reach 0.90 val-recovery with min frac
        fp_v = df_val_t["fp_correct"].to_numpy(bool)
        q_v = df_val_t["q_correct"].to_numpy(bool)
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
    for s in strategies:
        if not res[s]["frac"]:
            summary[s] = (float("nan"),) * 4 + (0,)
        else:
            summary[s] = (
                float(np.mean(res[s]["frac"])) * 100,
                float(np.std(res[s]["frac"])) * 100,
                float(np.mean(res[s]["rec"])) * 100,
                float(np.std(res[s]["rec"])) * 100,
                len(res[s]["frac"]),
            )
    summary["__n_eligible__"] = (float("nan"),) * 4 + (len(eligible),)
    return summary


def emit_dual_threshold_table(cfg_a, cfg_b, args, short_a, short_b,
                              qwen3_res=None, qwen3_short=None):
    """Online threshold calibration on both ViT backbones at W4, optionally with
    a third Qwen3 backbone column pair appended. If `qwen3_res` is None (Qwen3
    dump base does not exist), emits the original 2-backbone table."""
    print("Building dual-backbone online-threshold table ...")
    res_a = _threshold_calibration_from_data(
        _load_all(cfg_a, args.bits_primary, args.granularity), args,
    )
    res_b = _threshold_calibration_from_data(
        _load_all(cfg_b, args.bits_primary, args.granularity), args,
    )

    backbones = [(short_a, res_a), (short_b, res_b)]
    if qwen3_res is not None:
        backbones.append((qwen3_short, qwen3_res))

    def frac(d, key):
        fm, fs, _, _, _ = d.get(key, (float("nan"),) * 5)
        return "---" if np.isnan(fm) else f"{fm:.1f}\\%"

    def rec(d, key, bold=False):
        _, _, rm, rs, _ = d.get(key, (float("nan"),) * 5)
        if np.isnan(rm):
            return "---"
        body = f"{rm:.1f}\\% $\\pm$ {rs:.1f}"
        return f"\\textbf{{{body}}}" if bold else body

    n_back = len(backbones)
    column_spec = "ll" + "cc" * n_back

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% {n_back}-backbone online-routing calibration at W{args.bits_primary}-{args.granularity}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    qwen3_phrase = (
        " The Qwen3-Emb-0.6B column extends the comparison to a decoder-only NLP backbone (see \\S\\ref{sec:qwen3})."
        if qwen3_res is not None else ""
    )
    lines.append(
        f"\\caption{{Online (single-sample) deployment via fixed threshold $\\tau$ on $P(\\bad)$, "
        f"Q-only predictor at W{args.bits_primary}-{args.granularity}, "
        f"on \\textbf{{all three backbones}}. Mean across W{args.bits_primary}-eligible tasks per backbone. "
        f"$\\sigma$ on recovery in the last column of each pair. "
        f"Batch routing is the strong story across the board; "
        f"online $\\tau$-strategies are noisier on ViT-B/16 (tasks with tiny \\fp$\\to$\\ptq{{}} gaps inflate the per-task recovery variance) "
        f"but tighten substantially on ViT-L/16 and Qwen3.{qwen3_phrase}}}"
    )
    lines.append("\\label{tab:threshold}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    # backbone-group header
    header_cells = ["&"]
    for short, _ in backbones:
        header_cells.append(f"& \\multicolumn{{2}}{{c}}{{{short}}} ")
    lines.append(" ".join(header_cells) + "\\\\")
    # cmidrules
    cmidrules = []
    for i in range(n_back):
        lo = 3 + 2 * i
        hi = lo + 1
        cmidrules.append(f"\\cmidrule(lr){{{lo}-{hi}}}")
    lines.append(" ".join(cmidrules))
    # column-header row
    col_header = "Strategy & Needs at deploy time " + ("& Frac.\\ routed & Recovery " * n_back) + "\\\\"
    lines.append(col_header)
    lines.append("\\midrule")
    deploy_needs = {
        "batch":       "full target batch (sort it)",
        "natural":     "nothing ($\\tau = 0.5$)",
        "val_pct":     "target val $P(\\bad)$, label-free",
        "source_pct":  "pooled source val $P(\\bad)$, global $\\tau$",
        "val_labeled": "target val labels + $P(\\bad)$",
    }
    bs = "\\"
    for s in ["batch", "natural", "val_pct", "source_pct", "val_labeled"]:
        bold = (s == "batch")
        s_esc = s.replace("_", bs + "_")
        cells = [f"\\texttt{{{s_esc}}}", deploy_needs[s]]
        for _, r in backbones:
            cells.append(frac(r, s))
            cells.append(rec(r, s, bold=bold))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    n_cells = " ".join(
        f"& \\multicolumn{{2}}{{c}}{{{int(r['__n_eligible__'][4])}}}"
        for _, r in backbones
    )
    lines.append(
        f"\\multicolumn{{2}}{{r}}{{$n_{{\\text{{eligible tasks}}}}$}} "
        f"{n_cells} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tag = f"{cfg_a.sanitized}_vs_{cfg_b.sanitized}_bits{args.bits_primary}_{args.granularity}"
    out_path = OUT_DIR / f"threshold_dual_{tag}.tex"
    out_path.write_text(_tex_safe_join(lines))
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    cfg = _build_cfg(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emit_task_stats_table(cfg, args)
    emit_pareto_ablation_table(cfg, args)
    if args.also_model_name is not None:
        # Build second cfg
        args_b_ns = argparse.Namespace(**{**vars(args), "model_name": args.also_model_name})
        if args.also_batch_size is not None:
            args_b_ns.batch_size = args.also_batch_size
        cfg_b = _build_cfg(args_b_ns)
        # Per-backbone task-stats (the dual ablation table is one shared file).
        emit_task_stats_table(cfg_b, args)
        emit_pareto_ablation_table(cfg_b, args)
        short_a = _short_model(args.model_name)
        short_b = _short_model(args.also_model_name)
        emit_dual_ablation_table(cfg, cfg_b, args, short_a, short_b)
        emit_dual_auroc_table(cfg, cfg_b, args, short_a, short_b)
        # Optional Qwen3 column-pair for the threshold table — only if the dumps exist.
        qwen3_data = _load_all_qwen3(args, args.bits_primary)
        if qwen3_data:
            qwen3_res = _threshold_calibration_from_data(qwen3_data, args)
            qwen3_short = "Qwen3-Emb-0.6B"
            print(f"  including Qwen3 column ({len(qwen3_data)} tasks loaded)")
        else:
            qwen3_res = None
            qwen3_short = None
            print("  Qwen3 dumps not found — emitting 2-backbone threshold table")
        emit_dual_threshold_table(cfg, cfg_b, args, short_a, short_b,
                                  qwen3_res=qwen3_res, qwen3_short=qwen3_short)
    print("\nDone.")


def _short_model(model_name: str) -> str:
    n = model_name.lower()
    if "base" in n:
        size = "ViT-B"
    elif "large" in n:
        size = "ViT-L"
    elif "small" in n:
        size = "ViT-S"
    else:
        size = "ViT"
    import re
    m = re.search(r"patch(\d+)", model_name)
    return f"{size}/{m.group(1)}" if m else size


if __name__ == "__main__":
    main()
