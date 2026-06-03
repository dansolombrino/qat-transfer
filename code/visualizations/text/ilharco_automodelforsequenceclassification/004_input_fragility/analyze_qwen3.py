"""One-shot analyzer for Qwen3 Embedding input-fragility dumps.

Mirrors the vision-side `verify_paper_numbers.py`, adapted to:
  - Read text-side dumps under
    `${CHECKPOINT_BASE_PATH}/text/ilharco_automodelforsequenceclassification/input_fragility_dumps/`.
  - Use the NLP feature schema (`txt_*` instead of `img_*`).

Computes everything we'd need to slot Qwen3 into the paper:
  - Eligibility per regime
  - Per-task FP, Q test accuracy + gap, bad-fraction
  - Per-subset LOO X@90 at W4 (deployable + ceiling)
  - Per-subset LOO AUROC at W4 (secondary metric)
  - Online threshold calibration at W4 (5 strategies)
  - W3 oracle ceiling, mean gap, mean bad-fraction (for the regime-polarization story)
  - Univariate discriminative AUC of every feature at W4

Writes a markdown report under `plots/text/004_input_fragility/qwen3_summary.md`.

Argparse, no GPU, CPU-only.
"""

import argparse
import json
import os
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

from src.vision.utils import sanitize_hf_model_name


# Vision dumps use img_*; text dumps use txt_*. Everything else is identical.
TXT_FEATS = ["txt_n_tokens", "txt_n_unique_tokens", "txt_type_token_ratio", "txt_punct_ratio"]
Q_FEATS = ["q_margin", "q_softmax_top1", "q_entropy"]
FP_FEATS = ["fp_margin", "fp_softmax_top1", "fp_entropy"]
# `fp_cls_dist_to_class_centroid` was removed: it required the test sample's
# true class to pick which centroid to use, so it was not inference-deployable.
# Still present in the parquet dumps; no longer consumed.
CROSS_FEATS = ["fp_logit_at_q_pred", "q_logit_at_fp_pred", "fp_softmax_at_q_pred",
               "q_softmax_at_fp_pred", "fp_q_kl_symmetric", "fp_q_disagree"]
ALL_FEATS = FP_FEATS + Q_FEATS + CROSS_FEATS + TXT_FEATS

ALL_FEATURE_NAMES = Q_FEATS + FP_FEATS + CROSS_FEATS + TXT_FEATS

SUBSETS = {
    "text_only":          TXT_FEATS,
    # Univariate Q-side ablations (one feature each).
    "msp_only":           ["q_softmax_top1"],
    "q_margin_only":      ["q_margin"],
    "q_entropy_only":     ["q_entropy"],
    # Pairwise Q-side ablations (drop one of the three features).
    "q_margin_msp":       ["q_margin", "q_softmax_top1"],
    "q_margin_entropy":   ["q_margin", "q_entropy"],
    "q_msp_entropy":      ["q_softmax_top1", "q_entropy"],
    "q_only":             Q_FEATS,
    "q_plus_text":        Q_FEATS + TXT_FEATS,
    "fp_only":            FP_FEATS,
    "fp_plus_text":       FP_FEATS + TXT_FEATS,
    "fp_plus_q_no_cross": FP_FEATS + Q_FEATS + TXT_FEATS,
    "all_features":       ALL_FEATS,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--batch-size", default="32")
    p.add_argument("--lr", default="1e-05")
    p.add_argument("--wd", default="0.1")
    p.add_argument("--ls", default="0.0")
    p.add_argument("--max-grad-norm", default="1.0")
    p.add_argument("--max-length", default="128")
    p.add_argument("--seed", default="2038")
    p.add_argument("--skip-modules", nargs="+", default=["score"])
    p.add_argument("--bits-primary", type=int, default=4)
    p.add_argument("--bits-secondary", type=int, default=3)
    p.add_argument("--granularity", default="channel")
    p.add_argument("--min-bad", type=int, default=10)
    p.add_argument("--out-path", default=None)
    return p.parse_args()


def _dump_dir(args, dataset, bits):
    sanitized = sanitize_hf_model_name(args.model_name)
    skip_tag = "-".join(sorted(args.skip_modules)) if args.skip_modules else "none"
    base = (
        Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text"
        / "ilharco_automodelforsequenceclassification" / "input_fragility_dumps" / sanitized
    )
    return (
        base / dataset
        / f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_mgn={args.max_grad_norm}_bs={args.batch_size}_ml={args.max_length}"
        / f"ptq=bits={bits}_gran={args.granularity}_skip={skip_tag}"
        / f"seed={args.seed}"
    )


def _load_all(args, bits):
    sanitized = sanitize_hf_model_name(args.model_name)
    base = (
        Path(os.environ["CHECKPOINT_BASE_PATH"]) / "text"
        / "ilharco_automodelforsequenceclassification" / "input_fragility_dumps" / sanitized
    )
    out = {}
    if not base.exists():
        return out
    for ds_dir in sorted(base.iterdir()):
        d = _dump_dir(args, ds_dir.name, bits)
        if not (d / "predictions_test.parquet").exists():
            continue
        df_val = pd.read_parquet(d / "predictions_val.parquet")
        df_test = pd.read_parquet(d / "predictions_test.parquet")
        out[ds_dir.name] = (df_val, df_test)
    return out


def _per_task_z(X):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-9, 1.0, sigma)
    Xn = (X - mu) / sigma_safe
    return np.where(np.isnan(Xn), 0.0, Xn).astype(np.float64), mu, sigma_safe


def _apply_z(X, mu, sigma):
    Xn = (X - mu) / sigma
    return np.where(np.isnan(Xn), 0.0, Xn).astype(np.float64)


def _routed_curve(scores, fp_c, q_c):
    n = len(scores)
    rng = np.random.default_rng(12345)
    tie = rng.random(n)
    order = np.lexsort((tie, -scores))
    fp_s = fp_c[order].astype(np.int64)
    q_s = q_c[order].astype(np.int64)
    fp_cum = np.concatenate(([0], np.cumsum(fp_s)))
    q_cum = np.concatenate(([0], np.cumsum(q_s)))
    return np.arange(n + 1) / n, (fp_cum + (q_cum[-1] - q_cum)).astype(np.float64) / n


def _x_at_90(frac, rec):
    above = np.where(rec >= 0.9)[0]
    return float("nan") if len(above) == 0 else float(frac[above[0]])


def _eligibility(task_data, min_bad):
    return [
        ds for ds, (dv, dt) in task_data.items()
        if int(dv["bad"].sum()) >= min_bad and int(dt["bad"].sum()) >= min_bad
    ]


def _loo(task_data, cols, eligible):
    """Return per-task (X@90, AUROC) under LOO."""
    pt = {}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < 5:
            continue
        Xv = np.array(vf[cols].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sg = _per_task_z(Xv)
        pt[ds] = {"X": Xn, "y": yv, "mu": mu, "sg": sg}
    out = {}
    for target in pt:
        Xs = [pt[t]["X"] for t in pt if t != target]
        ys = [pt[t]["y"] for t in pt if t != target]
        X_pool = np.concatenate(Xs, axis=0)
        y_pool = np.concatenate(ys, axis=0)
        if y_pool.sum() < 5 or (len(y_pool) - y_pool.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_pool, y_pool)
        dt = task_data[target][1]
        Xt = np.array(dt[cols].to_numpy(dtype=np.float64), copy=True)
        Xtn = _apply_z(Xt, pt[target]["mu"], pt[target]["sg"])
        scores_full = clf.predict_proba(Xtn)[:, 1]
        fp_c = dt["fp_correct"].to_numpy(bool)
        q_c = dt["q_correct"].to_numpy(bool)
        gap = float(fp_c.mean()) - float(q_c.mean())
        if gap <= 1e-9:
            continue
        frac, acc = _routed_curve(scores_full, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        x90 = _x_at_90(frac, rec)
        # AUROC on FP-correct test rows
        tf = dt[dt["fp_correct"]].reset_index(drop=True)
        if int(tf["bad"].sum()) < 5 or int((~tf["bad"]).sum()) < 5:
            auroc = float("nan")
        else:
            Xtf = np.array(tf[cols].to_numpy(dtype=np.float64), copy=True)
            Xtfn = _apply_z(Xtf, pt[target]["mu"], pt[target]["sg"])
            s = clf.predict_proba(Xtfn)[:, 1]
            try:
                auroc = float(roc_auc_score(tf["bad"].astype(int).to_numpy(), s))
            except ValueError:
                auroc = float("nan")
        out[target] = (x90, auroc)
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
        else:
            raise ValueError(kind)
        frac, acc = _routed_curve(s, fp_c, q_c)
        rec = (acc - float(q_c.mean())) / gap
        out[target] = _x_at_90(frac, rec)
    return out


def _threshold_calibration(task_data, eligible, percentile=75.0):
    pt = {}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < 5:
            continue
        Xv = np.array(vf[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        yv = vf["bad"].astype(int).to_numpy()
        Xn, mu, sg = _per_task_z(Xv)
        pt[ds] = {"X": Xn, "y": yv, "mu": mu, "sg": sg}

    res = {s: {"frac": [], "rec": []} for s in
           ["batch", "natural", "val_pct", "source_pct", "val_labeled"]}

    for target in eligible:
        if target not in pt:
            continue
        Xs = [pt[t]["X"] for t in eligible if t != target and t in pt]
        ys = [pt[t]["y"] for t in eligible if t != target and t in pt]
        if not Xs:
            continue
        X_pool = np.concatenate(Xs, axis=0)
        y_pool = np.concatenate(ys, axis=0)
        if y_pool.sum() < 5 or (len(y_pool) - y_pool.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_pool, y_pool)
        src_scores = clf.predict_proba(X_pool)[:, 1]

        df_val_t, df_test_t = task_data[target]
        Xv_full = np.array(df_val_t[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xv_full_n = _apply_z(Xv_full, pt[target]["mu"], pt[target]["sg"])
        Xt = np.array(df_test_t[Q_FEATS].to_numpy(dtype=np.float64), copy=True)
        Xtn = _apply_z(Xt, pt[target]["mu"], pt[target]["sg"])
        val_scores = clf.predict_proba(Xv_full_n)[:, 1]
        test_scores = clf.predict_proba(Xtn)[:, 1]

        fp_t = df_test_t["fp_correct"].to_numpy(bool)
        q_t = df_test_t["q_correct"].to_numpy(bool)
        gap = float(fp_t.mean()) - float(q_t.mean())
        if gap <= 1e-9:
            continue
        q_acc = float(q_t.mean())

        frac, acc = _routed_curve(test_scores, fp_t, q_t)
        rec = (acc - q_acc) / gap
        idx_above = np.where(rec >= 0.90)[0]
        if len(idx_above) > 0:
            res["batch"]["frac"].append(float(frac[idx_above[0]]))
            res["batch"]["rec"].append(float(rec[idx_above[0]]))

        for label, tau in [
            ("natural", 0.5),
            ("val_pct", float(np.percentile(val_scores, percentile))),
            ("source_pct", float(np.percentile(src_scores, percentile))),
        ]:
            routed = test_scores > tau
            correct = np.where(routed, fp_t, q_t)
            res[label]["frac"].append(float(routed.mean()))
            res[label]["rec"].append((float(correct.mean()) - q_acc) / gap)

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


def _univariate_auc(task_data, eligible):
    rows = {c: [] for c in ALL_FEATURE_NAMES}
    for ds in eligible:
        df_val, _ = task_data[ds]
        vf = df_val[df_val["fp_correct"]].reset_index(drop=True)
        if int(vf["bad"].sum()) < 5 or len(vf) - int(vf["bad"].sum()) < 5:
            continue
        y = vf["bad"].astype(int).to_numpy()
        for c in ALL_FEATURE_NAMES:
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
                float(np.std(v)) if v else float("nan"),
                len(v)) for c, v in rows.items()}


def main():
    args = parse_args()
    out = [f"# Qwen3 Embedding 0.6B input-fragility summary — {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    out.append("")

    print(f"Loading W{args.bits_primary} ...")
    td_p = _load_all(args, args.bits_primary)
    print(f"Loading W{args.bits_secondary} ...")
    td_s = _load_all(args, args.bits_secondary)

    el_p = _eligibility(td_p, args.min_bad)
    el_s = _eligibility(td_s, args.min_bad)
    excl_p = sorted(set(td_p) - set(el_p))
    excl_s = sorted(set(td_s) - set(el_s))

    out.append("## 1. Eligibility")
    out.append(f"- W{args.bits_primary}-{args.granularity}: **{len(el_p)} of {len(td_p)} eligible**. Excluded: {excl_p}")
    out.append(f"- W{args.bits_secondary}-{args.granularity}: **{len(el_s)} of {len(td_s)} eligible**. Excluded: {excl_s}")
    out.append("")

    out.append(f"## 2. Mean FP→PTQ gap and bad-fraction")
    out.append("| Regime | n_eligible | Mean gap (pp) | Mean bad fraction (%) |")
    out.append("|---|---|---|---|")
    for td, el, lbl in [(td_p, el_p, f"W{args.bits_primary}"), (td_s, el_s, f"W{args.bits_secondary}")]:
        if not el:
            out.append(f"| {lbl} | 0 | — | — |")
            continue
        gaps = []
        bads = []
        for ds in el:
            dt = td[ds][1]
            fp = float(dt["fp_correct"].mean())
            q = float(dt["q_correct"].mean())
            gaps.append((fp - q) * 100)
            bads.append(float(dt["bad"].mean()) * 100)
        out.append(f"| {lbl} | {len(el)} | {np.mean(gaps):.2f} | {np.mean(bads):.2f} |")
    out.append("")

    print("Univariate AUCs ...")
    aucs = _univariate_auc(td_p, el_p)
    out.append(f"## 3. Univariate discriminative AUC at W{args.bits_primary}")
    out.append("| Feature | mean AUC | std |")
    out.append("|---|---|---|")
    for c in ALL_FEATURE_NAMES:
        m, s, n = aucs.get(c, (float("nan"), float("nan"), 0))
        if np.isnan(m):
            continue
        out.append(f"| `{c}` | {m:.3f} | {s:.3f} |")
    out.append("")

    print("LOO X@90 + AUROC per subset ...")
    out.append(f"## 4. LOO X@90 and AUROC per feature subset at W{args.bits_primary} (Qwen3)")
    out.append("| Subset | mean X@90 | std X@90 | mean AUROC | std AUROC |")
    out.append("|---|---|---|---|---|")
    for sname, cols in SUBSETS.items():
        r = _loo(td_p, cols, el_p)
        if not r:
            out.append(f"| `{sname}` | — | — | — | — |")
            continue
        x = [v[0] for v in r.values() if not np.isnan(v[0])]
        a = [v[1] for v in r.values() if not np.isnan(v[1])]
        out.append(
            f"| `{sname}` "
            f"| {np.mean(x)*100:.1f}% | {np.std(x)*100:.1f} "
            f"| {np.mean(a):.3f} | {np.std(a):.3f} |"
            if x and a else f"| `{sname}` | — | — | — | — |"
        )
    for kind in ("oracle", "random"):
        b = _baseline_x90(td_p, el_p, kind)
        if not b:
            out.append(f"| {kind} | — | — | — | — |")
            continue
        v = [vv for vv in b.values() if not np.isnan(vv)]
        out.append(f"| {kind} | {np.mean(v)*100:.1f}% | {np.std(v)*100:.1f} | — | — |")
    out.append("")

    print(f"Online threshold calibration at W{args.bits_primary} ...")
    th = _threshold_calibration(td_p, el_p)
    out.append(f"## 5. Online threshold calibration at W{args.bits_primary} (Q-only)")
    out.append("| Strategy | Frac routed (%) | Recovery (%) |")
    out.append("|---|---|---|")
    for s in ["batch", "natural", "val_pct", "source_pct", "val_labeled"]:
        r = th[s]
        if r["n"] == 0:
            out.append(f"| `{s}` | — | — |")
        else:
            out.append(f"| `{s}` | {r['frac_mean']:.1f} ± {r['frac_std']:.1f} | {r['rec_mean']:.1f} ± {r['rec_std']:.1f} |")
    out.append("")

    print(f"W{args.bits_secondary} LOO X@90 per subset + oracle/random (regime comparison) ...")
    out.append(f"## 6. W{args.bits_secondary} LOO X@90 per feature subset and baselines (Qwen3)")
    out.append("| Subset | mean X@90 | std X@90 | mean AUROC | std AUROC |")
    out.append("|---|---|---|---|---|")
    for sname, cols in SUBSETS.items():
        r = _loo(td_s, cols, el_s)
        if not r:
            out.append(f"| `{sname}` | — | — | — | — |")
            continue
        x = [v[0] for v in r.values() if not np.isnan(v[0])]
        a = [v[1] for v in r.values() if not np.isnan(v[1])]
        out.append(
            f"| `{sname}` "
            f"| {np.mean(x)*100:.1f}% | {np.std(x)*100:.1f} "
            f"| {np.mean(a):.3f} | {np.std(a):.3f} |"
            if x and a else f"| `{sname}` | — | — | — | — |"
        )
    for kind in ("oracle", "random"):
        b = _baseline_x90(td_s, el_s, kind)
        v = [vv for vv in b.values() if not np.isnan(vv)]
        if v:
            out.append(f"| {kind} | {np.mean(v)*100:.1f}% | {np.std(v)*100:.1f} | — | — |")
        else:
            out.append(f"| {kind} | — | — | — | — |")
    out.append("")

    out_path = Path(args.out_path) if args.out_path else (
        _PROJECT_ROOT / "plots" / "text" / "004_input_fragility" / "qwen3_summary.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out) + "\n")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
