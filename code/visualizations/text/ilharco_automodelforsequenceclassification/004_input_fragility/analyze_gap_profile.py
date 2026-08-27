"""Qwen3 gap-profile analysis + top-k set stability (C2), with the vision configs alongside.

Two jobs:

  1. Replicate on a third backbone with a different readout path (decoder-only,
     `score` head, pooled at last non-pad token) the results established on two ViTs:
     the eps/margin regime parameter, the three de-risking questions, and the
     conformal certificate.

  2. Test C2 -- top-k SET stability -- which is the first result here that is about
     the ranking rather than about the argmax. C2 says the top-k set survives every
     eps-perturbation iff  z_(k) - z_(k+1) >= 2 eps,  i.e.  gap_{k+1} - gap_k >= 2 eps.
     For an embedding model that is the retrieval-stability criterion.
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
load_dotenv(_PROJECT_ROOT / ".env")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src.vision.utils import sanitize_hf_model_name, sanitize_timm_model_name

_CBP = Path(os.environ["CHECKPOINT_BASE_PATH"])

# (domain, family, model, hp-dir, bits, label)
CONFIGS = [
    ("text", "ilharco_automodelforsequenceclassification", "Qwen/Qwen3-Embedding-0.6B",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128", 4, "score", "Qwen3 W4"),
    ("text", "ilharco_automodelforsequenceclassification", "Qwen/Qwen3-Embedding-0.6B",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128", 3, "score", "Qwen3 W3"),
    ("vision", "ilharco_timm_supervised", "vit_base_patch16_224.orig_in21k",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128", 4, "head", "ViT-B W4"),
    ("vision", "ilharco_timm_supervised", "vit_large_patch16_224.orig_in21k",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=64", 4, "head", "ViT-L W4"),
]


def load(domain, family, model, hp, bits, skip, gran="channel"):
    san = sanitize_hf_model_name if domain == "text" else sanitize_timm_model_name
    base = _CBP / domain / family / "gap_profile" / san(model)
    out = {}
    if not base.exists():
        return out
    for ds in sorted(base.iterdir()):
        p = (ds / hp / f"ptq=bits={bits}_gran={gran}_skip={skip}" / "seed=2038"
             / "gap_profile_test.parquet")
        if p.exists():
            out[ds.name] = pd.read_parquet(p)
    return out


def _cv_auc(X, y):
    if len(set(y)) < 2 or min(np.bincount(y.astype(int))) < 5:
        return np.nan
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                          Xs, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def _pair_score(d):
    """|Delta(z_a - z_b)| on (a,b) = the quantised model's own top-2. Threshold is 1x."""
    K = len([c for c in d.columns if c.startswith("fp_cls_")])
    fpc = np.stack([d[f"fp_cls_{j}"].values for j in range(1, K + 1)], 1)
    fpg = np.stack([d[f"fp_gap_{j}"].values for j in range(1, K + 1)], 1)
    a, b = d.q_cls_1.values, d.q_cls_2.values
    pa, pb = (fpc == a[:, None]), (fpc == b[:, None])
    ok = pa.any(1) & pb.any(1)
    with np.errstate(invalid="ignore"):
        gF = (np.nanmin(np.where(pb, fpg, np.nan), 1)
              - np.nanmin(np.where(pa, fpg, np.nan), 1))
    return np.abs(d.q_gap_2.values - gF), ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-bad", type=int, default=10)
    p.add_argument("--alpha", type=float, nargs="+", default=[0.10, 0.05, 0.01])
    p.add_argument("--kmax", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(0)

    print("=" * 88)
    print("A. REGIME PARAMETER + THE THREE QUESTIONS  (Qwen3 vs the two ViTs)")
    print("=" * 88)
    for domain, fam, model, hp, bits, skip, label in CONFIGS:
        tasks = load(domain, fam, model, hp, bits, skip)
        if not tasks:
            print(f"\n  {label}: no data")
            continue
        REL, W1, C2r, Q2m, Q2f, Q3m, NB = [], [], [], [], [], [], []
        for ds, d in tasks.items():
            if int(d["bad"].sum()) < args.min_bad:
                continue
            NB.append(int(d["bad"].sum()))
            e = d["eps_linf"].values
            REL.append(np.median(e) / max(np.median(d["fp_gap_2"].values), 1e-9))
            K = len([c for c in d.columns if c.startswith("q_gap_")])
            G = np.stack([d[f"q_gap_{k}"].values for k in range(1, K + 1)], 1)
            w = (G < 2 * e[:, None]).sum(1)
            W1.append((w == 1).mean())
            tau = np.quantile(d["q_gap_2"].values, 0.25)
            cont = (G < tau).sum(1)[d["q_gap_2"].values < tau]
            C2r.append((cont == 2).mean())
            fpc = d[d.fp_correct]
            y = fpc["bad"].values.astype(int)
            gp = np.stack([fpc[f"q_gap_{k}"].values for k in range(2, K + 1)], 1)
            Q2m.append(roc_auc_score(y, -gp[:, 0]) if len(set(y)) > 1 else np.nan)
            Q2f.append(_cv_auc(gp, y))
            sub = d[d.bad | d.lucky_q]
            y2 = sub["bad"].values.astype(int)
            Q3m.append(roc_auc_score(y2, -sub["q_gap_2"].values) if len(set(y2)) > 1 else np.nan)
        m = np.nanmean
        print(f"\n  {label}  ({len(NB)} tasks, {sum(NB)} bad)")
        print(f"    eps / median FP margin   : {m(REL):.3f}")
        print(f"    |C_eps| = 1 (certifiable): {m(W1)*100:.1f}% of all inputs")
        print(f"    Q1 two-way among routed  : {m(C2r)*100:.1f}%")
        print(f"    Q2 margin {m(Q2m):.3f} -> full profile {m(Q2f):.3f}  (delta {m(Q2f)-m(Q2m):+.3f})")
        print(f"    Q3 bad vs lucky-Q        : {m(Q3m):.3f}   (0.5 = Prop. 2 holds)")

    print()
    print("=" * 88)
    print("B. THE CERTIFICATE ON A THIRD BACKBONE")
    print("=" * 88)
    for domain, fam, model, hp, bits, skip, label in CONFIGS:
        tasks = load(domain, fam, model, hp, bits, skip)
        if not tasks:
            continue
        print(f"\n  {label}")
        for alpha in args.alpha:
            A, Ac, Bb, Bf = [], [], [], []
            for ds, d in tasks.items():
                if len(d) < 200:
                    continue
                idx = rng.permutation(len(d))
                h = len(d) // 2
                cal, tst = d.iloc[idx[:h]], d.iloc[idx[h:]]
                flip = tst.fp_top1.values != tst.q_top1.values
                n = len(cal)
                q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
                c1 = tst.q_gap_2.values >= 2 * np.quantile(cal.eps_linf.values, q)
                A.append(c1.mean()); Ac.append(flip[c1].mean() if c1.sum() else 0.0)
                ps, ok = _pair_score(cal)
                if ok.sum() < 20:
                    continue
                mm = ok.sum()
                q2 = min(np.ceil((mm + 1) * (1 - alpha)) / mm, 1.0)
                c2 = tst.q_gap_2.values >= np.quantile(ps[ok], q2)
                Bb.append(c2.mean()); Bf.append(flip[c2].mean() if c2.sum() else 0.0)
            f = lambda a: np.mean(a) * 100 if len(a) else float("nan")
            print(f"    alpha={alpha:.2f}   eps_inf: cert {f(A):5.1f}% / flips {f(Ac):.3f}%"
                  f"   |   contender-only: cert {f(Bb):5.1f}% / flips {f(Bf):.3f}%")

    print()
    print("=" * 88)
    print("C. TOP-k SET STABILITY (C2) — the ranking result, not the argmax result")
    print("=" * 88)
    print("  C2: the top-k SET survives every eps-perturbation iff  gap_{k+1} - gap_k >= 2 eps.")
    print("  'observed' = FP top-k set != PTQ top-k set.  'recall' = share of actual set changes")
    print("  the criterion flags (must be 100% -- it is a theorem).  'certified' = the conformal")
    print("  form: share of inputs whose top-k set is provably stable, at alpha=0.10.\n")
    for domain, fam, model, hp, bits, skip, label in CONFIGS:
        tasks = load(domain, fam, model, hp, bits, skip)
        if not tasks:
            continue
        print(f"  {label}")
        for k in range(1, args.kmax + 1):
            CH, RC, CE, CF = [], [], [], []
            for ds, d in tasks.items():
                K = len([c for c in d.columns if c.startswith("q_cls_")])
                if K < k + 1 or len(d) < 200:
                    continue
                fs = np.sort(np.stack([d[f"fp_cls_{j}"].values for j in range(1, k + 1)], 1), 1)
                qs = np.sort(np.stack([d[f"q_cls_{j}"].values for j in range(1, k + 1)], 1), 1)
                changed = (fs != qs).any(1)
                sep = d[f"q_gap_{k+1}"].values - d[f"q_gap_{k}"].values
                e = d["eps_linf"].values
                CH.append(changed.mean())
                flagged = sep < 2 * e
                RC.append((flagged & changed).sum() / max(changed.sum(), 1))
                idx = rng.permutation(len(d))
                h = len(d) // 2
                cal, tst = d.iloc[idx[:h]], d.iloc[idx[h:]]
                n = len(cal)
                qq = min(np.ceil((n + 1) * 0.90) / n, 1.0)
                eh = np.quantile(cal.eps_linf.values, qq)
                st = (tst[f"q_gap_{k+1}"].values - tst[f"q_gap_{k}"].values) >= 2 * eh
                fs_t = np.sort(np.stack([tst[f"fp_cls_{j}"].values for j in range(1, k + 1)], 1), 1)
                qs_t = np.sort(np.stack([tst[f"q_cls_{j}"].values for j in range(1, k + 1)], 1), 1)
                ch_t = (fs_t != qs_t).any(1)
                CE.append(st.mean()); CF.append(ch_t[st].mean() if st.sum() else 0.0)
            if not CH:
                continue
            print(f"    k={k}  observed set change {np.mean(CH)*100:5.2f}%   "
                  f"criterion recall {np.mean(RC)*100:6.1f}%   "
                  f"certified stable {np.mean(CE)*100:5.1f}%   "
                  f"violations among certified {np.mean(CF)*100:.3f}%")
        print()


if __name__ == "__main__":
    main()
