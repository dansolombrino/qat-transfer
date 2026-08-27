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
TXT_FEATS = ["txt_n_tokens", "txt_n_unique_tokens", "txt_type_token_ratio", "txt_punct_ratio"]
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
    # Univariate Q-side ablations: each isolates one Q-side scalar.
    # `msp_only` is the canonical max-softmax-prob baseline (Hendrycks & Gimpel 2017,
    # Geifman & El-Yaniv 2017). `q_margin_only` and `q_entropy_only` test whether
    # any single Q-side feature already matches the 3-feature `q_only` predictor.
    "msp_only":         ["q_softmax_top1"],
    "q_margin_only":    ["q_margin"],
    "q_entropy_only":   ["q_entropy"],
    # Pairwise Q-side ablations: each drops one of the three features.
    "q_margin_msp":     ["q_margin", "q_softmax_top1"],
    "q_margin_entropy": ["q_margin", "q_entropy"],
    "q_msp_entropy":    ["q_softmax_top1", "q_entropy"],
    "q_only": Q_FEATS,
    "fp_only": FP_FEATS,
    "fp_plus_q_no_cross": FP_FEATS + Q_FEATS + IMAGE_FEATS,
    "all_features": ALL_FEATS,
}

SUBSET_HEADERS = {
    "image_only":         "image",
    "msp_only":           "MSP",
    "q_margin_only":      "margin",
    "q_entropy_only":     "entropy",
    "q_margin_msp":       "\\shortstack{margin\\\\+MSP}",
    "q_margin_entropy":   "\\shortstack{margin\\\\+ent}",
    "q_msp_entropy":      "\\shortstack{MSP\\\\+ent}",
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
            rec["n_val"]        = len(dv)
            rec["n_test"]       = len(dt)
            rec["n_bad_val_w4"] = int(dv["bad"].sum())
            rec["n_bad_w4"]     = int(dt["bad"].sum())
            rec["fp_w4"]        = float(dt["fp_correct"].mean())
            rec["q_w4"]         = float(dt["q_correct"].mean())
            rec["eligible_w4"]  = (rec["n_bad_val_w4"] >= args.min_bad_val
                                   and rec["n_bad_w4"] >= args.min_bad_test)
        # W3 stats
        if ds in secondary:
            dv2, dt2, _ = secondary[ds]
            rec["n_bad_val_w3"] = int(dv2["bad"].sum())
            rec["n_bad_w3"]     = int(dt2["bad"].sum())
            rec["fp_w3"]        = float(dt2["fp_correct"].mean())
            rec["q_w3"]         = float(dt2["q_correct"].mean())
            rec["eligible_w3"]  = (rec["n_bad_val_w3"] >= args.min_bad_val
                                   and rec["n_bad_w3"] >= args.min_bad_test)
        rows.append(rec)

    short = _short_model(cfg.model_name)
    label_slug = short.lower().replace("/", "_").replace("-", "_")

    def _regime_block(bits, suffix):
        """One 8-column table for a single bit-width. Split per regime so the
        table fits the column width (13 columns overflows)."""
        out = []
        out.append("\\begin{table}[t]")
        out.append("\\centering")
        out.append(
            f"\\caption{{Per-task statistics \\textbf{{for {short}}} at "
            f"W{bits}-{args.granularity}. "
            "$n_{\\text{bad}}^{\\text{val/test}}$ count val/test inputs FP gets right and PTQ flips; "
            "$\\Delta$ is the FP$\\to$PTQ test-accuracy gap (positive $=$ PTQ drop). "
            f"$\\dagger$ marks rows excluded at this regime by the eligibility filter "
            f"($n_{{\\text{{bad}}}} < {args.min_bad_val}$ on val or $< {args.min_bad_test}$ on test).}}"
        )
        out.append(f"\\label{{tab:task_stats_{label_slug}_w{bits}}}")
        out.append("\\small")
        out.append("\\setlength{\\tabcolsep}{3pt}")
        out.append("\\begin{tabular}{lrrrrrrr}")
        out.append("\\toprule")
        out.append("Task & $n_{\\text{val}}$ & $n_{\\text{test}}$ "
                   "& $n_{\\text{bad}}^{\\text{val}}$ & $n_{\\text{bad}}^{\\text{test}}$ "
                   "& FP & PTQ & $\\Delta$ \\\\")
        out.append("\\midrule")
        for r in rows:
            nv, nt = _fmt_count(r.get("n_val")), _fmt_count(r.get("n_test"))
            if f"n_bad_w{suffix}" in r:
                mark = "" if r[f"eligible_w{suffix}"] else "$^{\\dagger}$"
                nbv = f"{r[f'n_bad_val_w{suffix}']:,}"
                nbt = f"{r[f'n_bad_w{suffix}']:,}{mark}"
                fp  = _fmt_pct(r[f"fp_w{suffix}"], 1)
                q   = _fmt_pct(r[f"q_w{suffix}"], 1)
                d   = f"{(r[f'fp_w{suffix}'] - r[f'q_w{suffix}']) * 100:+.2f}"
            else:
                nbv = nbt = fp = q = d = "---"
            out.append(f"{r['task']} & {nv} & {nt} & {nbv} & {nbt} & {fp} & {q} & {d} \\\\")
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")
        return out

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% Per-task statistics for {short}, one table per regime "
                 f"(W{args.bits_primary}-{args.granularity}, W{args.bits_secondary}-{args.granularity}).")
    lines += _regime_block(args.bits_primary, 4)
    lines.append("")
    lines += _regime_block(args.bits_secondary, 3)

    out_path = OUT_DIR / f"appendix_task_stats_{cfg.sanitized}_{args.granularity}.tex"
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
        f"\\caption{{Per-task $X_{{@90\\%}}$ \\textbf{{for {short}}} "
        f"at W{args.bits_primary}-{args.granularity}, per feature subset plus oracle and random baselines. "
        "Lower is better. Same row taxonomy as Table~\\ref{tab:ablation}: singleton/pairwise rows are "
        "per-task sort-by-score; multivariate rows use LOO cross-task LogReg.}"
    )
    lines.append(f"\\label{{tab:pareto_per_task_{label_slug}}}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
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
        mean_row.append(f"{m * 100:.1f}\\%" if not np.isnan(m) else "---")
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
        f"\\caption{{\\xat{{90}} per feature subset on {short_a} and {short_b} "
        f"at W{args.bits_primary}-{args.granularity} and W{args.bits_secondary}-{args.granularity}. "
        f"Lower is better. Singleton and pairwise rows are per-task sort-by-score (no fitted classifier); "
        f"the multivariate rows (\\texttt{{q\\_only}}, \\texttt{{fp\\_only}}, "
        f"\\texttt{{fp\\_plus\\_q\\_no\\_cross}}, \\texttt{{all\\_features}}) use a leave-one-out (LOO) cross-task LogReg. "
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
        "q_margin_only":      "margin (proposed)",
        "q_entropy_only":     "entropy alone",
        "q_margin_msp":       "margin + MSP",
        "q_margin_entropy":   "margin + entropy",
        "q_msp_entropy":      "MSP + entropy",
        "q_only":             "all 3 Q-side (LogReg)",
        "fp_only":            "FP-side only",
        "fp_plus_q_no_cross": "both models",
        "all_features":       "ceiling (both+cross)",
    }
    for subset_name in SUBSETS:
        label = deployment_labels.get(subset_name, subset_name)
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


def _qwen3_subsets():
    """Qwen3-version of SUBSETS: swap IMAGE_FEATS → TXT_FEATS in every subset that
    references input-domain features. Q-side / FP-side / cross-model rows are unchanged.
    The first row is renamed `text_only` so the AUROC table can render a single column
    that covers either modality."""
    s = dict(SUBSETS)
    s.pop("image_only")
    out = {"text_only": TXT_FEATS}
    for k, v in s.items():
        out[k] = [c for c in v if c not in IMAGE_FEATS] + (
            TXT_FEATS if any(c in IMAGE_FEATS for c in v) else []
        )
    return out


def _aggregate_subset_auroc_from_data(task_data, args, subsets):
    """{subset: (mean_auroc, std_auroc, n_tasks)} from already-loaded task_data."""
    eligible = [
        ds for ds, (dv, dt, _) in task_data.items()
        if int(dv["bad"].sum()) >= args.min_bad_val
        and int(dt["bad"].sum()) >= args.min_bad_test
    ]
    out = {}
    for subset_name, cols in subsets.items():
        a = _loo_auroc_for_subset(task_data, cols, eligible, args.min_bad_val)
        vals = list(a.values())
        if vals:
            out[subset_name] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        else:
            out[subset_name] = (float("nan"), float("nan"), 0)
    out["__n_eligible__"] = (float("nan"), float("nan"), len(eligible))
    return out


def _aggregate_subset_auroc(cfg, args, bits):
    """{subset: (mean_auroc, std_auroc, n_tasks)} for one backbone+regime (vision)."""
    return _aggregate_subset_auroc_from_data(
        _load_all(cfg, bits, args.granularity), args, SUBSETS
    )


def emit_dual_auroc_table(cfg_a, cfg_b, args, short_a, short_b,
                          qwen3_res=None, qwen3_short=None):
    """Secondary-metric table: AUROC per subset on both ViT backbones at W4,
    optionally with a third Qwen3 column."""
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

    has_qwen3 = qwen3_res is not None
    n_backbones = 3 if has_qwen3 else 2

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% {n_backbones}-backbone AUROC at W{args.bits_primary}-{args.granularity}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    backbone_phrase = f"{short_a}, {short_b}" + (f", and {qwen3_short}" if has_qwen3 else "")
    lines.append(
        f"\\caption{{\\textbf{{AUROC of $P(\\bad)$ vs \\bad{{}} labels on FP-correct test rows}} "
        f"at W{args.bits_primary}-{args.granularity}, per feature subset, on {backbone_phrase}. "
        f"Higher is better. Mean $\\pm$ std across eligible tasks per backbone. "
        f"Singleton/pairwise rows are per-task sort-by-score; multivariate rows "
        f"(\\texttt{{q\\_only}}, \\texttt{{fp\\_only}}, \\texttt{{fp\\_plus\\_q\\_no\\_cross}}, "
        f"\\texttt{{all\\_features}}) use LOO cross-task LogReg. "
        f"\\texttt{{all\\_features}}'s AUROC is suppressed (---); see App.~\\ref{{app:auroc_leak}}.}}"
    )
    lines.append("\\label{tab:auroc}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    col_spec = "ll" + "c" * n_backbones
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header_backbones = f"{short_a} & {short_b}" + (f" & {qwen3_short}" if has_qwen3 else "")
    lines.append(f"Subset & Deployment & {header_backbones} \\\\")
    lines.append("\\midrule")
    deployment_labels = {
        "image_only":         "no model",
        "text_only":          "no model",
        "msp_only":           "MSP baseline",
        "q_margin_only":      "margin (proposed)",
        "q_entropy_only":     "entropy alone",
        "q_margin_msp":       "margin + MSP",
        "q_margin_entropy":   "margin + entropy",
        "q_msp_entropy":      "MSP + entropy",
        "q_only":             "all 3 Q-side (LogReg)",
        "fp_only":            "FP-side only",
        "fp_plus_q_no_cross": "both models",
        "all_features":       "ceiling (both+cross)",
    }
    for subset_name in SUBSETS:
        label = deployment_labels.get(subset_name, subset_name)
        if subset_name == "image_only" and has_qwen3:
            # Modality-spanning first row: vision is image_only, Qwen3 is text_only.
            display_name = "image\\_only / text\\_only"
            qwen3_key = "text_only"
        else:
            display_name = subset_name.replace("_", "\\_")
            qwen3_key = subset_name
        row = [
            f"\\texttt{{{display_name}}}", label,
            cell(res_a, subset_name), cell(res_b, subset_name),
        ]
        if has_qwen3:
            row.append(cell(qwen3_res, qwen3_key))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")
    n_cells = f"& {int(res_a['__n_eligible__'][2])} & {int(res_b['__n_eligible__'][2])}"
    if has_qwen3:
        n_cells += f" & {int(qwen3_res['__n_eligible__'][2])}"
    lines.append(
        f"\\multicolumn{{2}}{{r}}{{$n_{{\\text{{eligible tasks}}}}$}} {n_cells} \\\\"
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

    Uses the single-feature q_margin recipe (the paper's deployable headline).
    LogReg of one feature is monotone in that feature, so the routing decisions
    here are equivalent to thresholding q_margin directly; the batch column
    therefore equals the q_margin_only X@90 from tab:ablation.

    Returns {strategy: (frac_mean_pct, frac_std_pct, rec_mean_pct, rec_std_pct, n)}
    plus a sentinel key '__n_eligible__' giving the eligible task count.

    Strategies: batch (offline sort-and-cut reference) / val_pct (target val percentile,
    label-free) / val_x90 (target val labels, minimum routing fraction at >=0.9 recovery
    on val).

    Domain-agnostic: works on either vision or NLP task_data because both share the
    same parquet schema for Q-side features and bad/fp_correct/q_correct flags."""
    feats = ["q_margin"]
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
        Xv = np.array(vf[feats].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sigma = _per_task_z(Xv)
        pt[ds] = {"Xn": Xn, "y": yv, "mu": mu, "sigma": sigma}

    strategies = ["batch", "val_pct", "val_x90"]
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

        df_val_t, df_test_t = task_data[target][0], task_data[target][1]
        Xv_full = np.array(df_val_t[feats].to_numpy(dtype=np.float64), copy=True)
        Xv_full_n = _apply_z(Xv_full, pt[target]["mu"], pt[target]["sigma"])
        Xt = np.array(df_test_t[feats].to_numpy(dtype=np.float64), copy=True)
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

        # fixed-tau strategies (only val_pct is in the table; val_x90 is computed below
        # because it needs the val routed curve).
        for label, tau in [
            ("val_pct", float(np.percentile(val_scores, percentile))),
        ]:
            routed = test_scores > tau
            correct = np.where(routed, fp_t, q_t)
            res[label]["frac"].append(float(routed.mean()))
            res[label]["rec"].append((float(correct.mean()) - q_acc) / gap)

        fp_v = df_val_t["fp_correct"].to_numpy(bool)
        q_v = df_val_t["q_correct"].to_numpy(bool)
        gap_v = float(fp_v.mean()) - float(q_v.mean())

        # val_x90 — per-task percentile derived from val: find smallest f at which
        # sorting val by score recovers 0.90 on val, then set tau as the score
        # percentile on val that corresponds to routing fraction f. No fixed global
        # percentile, no test-side information used in the choice.
        if gap_v > 1e-9:
            frac_v, acc_v = _routed_curve(val_scores, fp_v, q_v)
            rec_v = (acc_v - float(q_v.mean())) / gap_v
            idx_v = np.where(rec_v >= 0.90)[0]
            if len(idx_v) > 0:
                f_v = float(frac_v[idx_v[0]])  # val X@90 fraction
                # Set tau as the (1 - f_v)*100-th percentile of val_scores so that
                # routing val_score > tau routes the top f_v fraction of val.
                tau_x90 = float(np.percentile(val_scores, (1.0 - f_v) * 100))
                r_t = test_scores > tau_x90
                c_t = np.where(r_t, fp_t, q_t)
                res["val_x90"]["frac"].append(float(r_t.mean()))
                res["val_x90"]["rec"].append((float(c_t.mean()) - q_acc) / gap)

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

    lines = []
    lines.append("% Auto-generated by code/visualizations/.../004_input_fragility/generate_paper_tables.py")
    lines.append(f"% {n_back}-backbone online-routing calibration at W{args.bits_primary}-{args.granularity}.")
    lines.append("% Transposed (row-grouped by backbone) to fit \\linewidth;")
    lines.append("% strategy definitions live in \\S\\ref{sec:online}.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    del qwen3_res  # caption no longer branches on Qwen3 presence
    lines.append(
        f"\\caption{{Online deployment via a fixed threshold $\\tau$ on the single-feature "
        f"\\texttt{{q\\_margin}} score at W{args.bits_primary}-{args.granularity}, on all three backbones. "
        f"Mean across the W{args.bits_primary}-eligible tasks per backbone; $\\sigma$ on recovery in the third column. "
        f"\\texttt{{batch}} is the offline sort-and-route reference (X@90 fraction on the test set) "
        f"and upper-bounds the two proposed online $\\tau$-strategies, which differ only in whether the calibration uses labels "
        f"(defined in \\S\\ref{{sec:online}}).}}"
    )
    lines.append("\\label{tab:threshold}")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Strategy & Frac.\\ routed & Recovery \\\\")
    lines.append("\\midrule")
    bs = "\\"
    strategies = ["batch", "val_pct", "val_x90"]
    for i, (short, r) in enumerate(backbones):
        n_eligible = int(r["__n_eligible__"][4])
        lines.append(
            f"\\multicolumn{{3}}{{l}}{{\\textit{{{short}}} "
            f"({n_eligible} W{args.bits_primary}-eligible tasks)}} \\\\"
        )
        for s in strategies:
            bold = (s == "batch")
            s_esc = s.replace("_", bs + "_")
            cells = [f"\\texttt{{{s_esc}}}", frac(r, s), rec(r, s, bold=bold)]
            lines.append(" & ".join(cells) + " \\\\")
        if i < n_back - 1:
            lines.append("\\midrule")
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
        # Optional Qwen3 column for the AUROC and threshold tables — only if the dumps exist.
        qwen3_data = _load_all_qwen3(args, args.bits_primary)
        if qwen3_data:
            qwen3_thr_res = _threshold_calibration_from_data(qwen3_data, args)
            qwen3_auroc_res = _aggregate_subset_auroc_from_data(
                qwen3_data, args, _qwen3_subsets()
            )
            qwen3_short = "Qwen3-Emb-0.6B"
            print(f"  including Qwen3 column ({len(qwen3_data)} tasks loaded)")
        else:
            qwen3_thr_res = None
            qwen3_auroc_res = None
            qwen3_short = None
            print("  Qwen3 dumps not found — emitting 2-backbone tables")
        emit_dual_auroc_table(cfg, cfg_b, args, short_a, short_b,
                              qwen3_res=qwen3_auroc_res, qwen3_short=qwen3_short)
        emit_dual_threshold_table(cfg, cfg_b, args, short_a, short_b,
                                  qwen3_res=qwen3_thr_res, qwen3_short=qwen3_short)
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
