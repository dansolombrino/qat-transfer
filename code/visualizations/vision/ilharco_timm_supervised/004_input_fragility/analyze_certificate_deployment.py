"""Two deployment questions about the certificate.

G. How much calibration data does it need? (reviewers always ask)
H. How does the certified set relate to the routing recipe already in the paper?
   Both are thresholds on q_margin, so the certificate does not change the policy --
   it puts a guarantee on part of it. Quantify which part.

Produces F16 / MATERIAL.md 2.4c: how much calibration data the certificate needs (50 inputs
is enough) and what share of already-Q-served traffic it certifies.
"""
import os
import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[5]   # repo root, derived not hardcoded
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
from dotenv import load_dotenv
load_dotenv(_R / ".env")

import numpy as np
import pandas as pd
from src.vision.utils import sanitize_timm_model_name

B = Path(os.environ["CHECKPOINT_BASE_PATH"]) / "vision" / "ilharco_timm_supervised"


def load(model, bs, bits, gran="channel"):
    base = B / "gap_profile" / sanitize_timm_model_name(model)
    out = {}
    for ds in sorted(base.iterdir()):
        p = (ds / f"optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs={bs}"
             / f"ptq=bits={bits}_gran={gran}_skip=head" / "seed=2038"
             / "gap_profile_test.parquet")
        if p.exists():
            out[ds.name] = pd.read_parquet(p)
    return out


def pair_score(d):
    K = len([c for c in d.columns if c.startswith("fp_cls_")])
    fpc = np.stack([d[f"fp_cls_{j}"].values for j in range(1, K + 1)], 1)
    fpg = np.stack([d[f"fp_gap_{j}"].values for j in range(1, K + 1)], 1)
    a, b = d.q_cls_1.values, d.q_cls_2.values
    pa, pb = (fpc == a[:, None]), (fpc == b[:, None])
    ok = pa.any(1) & pb.any(1)
    gF = np.nanmin(np.where(pb, fpg, np.nan), 1) - np.nanmin(np.where(pa, fpg, np.nan), 1)
    return np.abs(d.q_gap_2.values - gF), ok


CFG = [("vit_base_patch16_224.orig_in21k", "128", 4, "ViT-B W4"),
       ("vit_large_patch16_224.orig_in21k", "64", 4, "ViT-L W4")]

print("=" * 84)
print("G. CALIBRATION-SET SIZE: how many inputs do you need before the certificate is usable?")
print("=" * 84)
print("  alpha=0.10, contender-only score. n = calibration inputs drawn per task,")
print("  20 random draws each; the rest of the task is the test half.\n")
for model, bs, bits, label in CFG:
    tasks = load(model, bs, bits)
    print(f"  {label}")
    for n in (50, 100, 200, 500, 1000):
        cov, flip_r = [], []
        for ds, d in tasks.items():
            if len(d) < 2 * n + 200:
                continue
            for r in range(20):
                rng = np.random.default_rng(1000 * r + len(ds))
                idx = rng.permutation(len(d))
                cal, tst = d.iloc[idx[:n]], d.iloc[idx[n:]]
                ps, ok = pair_score(cal)
                if ok.sum() < 10:
                    continue
                m = ok.sum()
                q = min(np.ceil((m + 1) * 0.90) / m, 1.0)
                c = tst.q_gap_2.values >= np.quantile(ps[ok], q)
                fl = tst.fp_top1.values != tst.q_top1.values
                cov.append(c.mean())
                flip_r.append(fl[c].mean() if c.sum() else 0.0)
        if not cov:
            print(f"    n={n:<5d} (no task large enough)")
            continue
        print(f"    n={n:<5d} certified {np.mean(cov)*100:5.1f}% (sd {np.std(cov)*100:4.1f})"
              f"   flips {np.mean(flip_r)*100:.3f}%   worst draw {np.max(flip_r)*100:.2f}%")
    print()

print("=" * 84)
print("H. CERTIFIED vs ROUTED: the certificate does not change the policy, it guarantees part of it")
print("=" * 84)
print("  Both are thresholds on q_margin. Q-served = 100 - X@90 from the paper.")
print("  Question: what share of what we already serve from Q is now *provably* correct?\n")
for model, bs, bits, label in CFG:
    tasks = load(model, bs, bits)
    rng = np.random.default_rng(0)
    for alpha in (0.10, 0.05, 0.01):
        C, S, ACC = [], [], []
        for ds, d in tasks.items():
            if len(d) < 200:
                continue
            idx = rng.permutation(len(d))
            h = len(d) // 2
            cal, tst = d.iloc[idx[:h]], d.iloc[idx[h:]]
            ps, ok = pair_score(cal)
            if ok.sum() < 20:
                continue
            m = ok.sum()
            q = min(np.ceil((m + 1) * (1 - alpha)) / m, 1.0)
            cert = tst.q_gap_2.values >= np.quantile(ps[ok], q)
            # the paper's deployed rule: send the lowest-margin 25% to FP
            served = tst.q_gap_2.values >= np.quantile(tst.q_gap_2.values, 0.25)
            C.append(cert.mean()); S.append(served.mean())
            ACC.append((cert & ~served).mean())      # certified but we route it anyway
        c, s = np.mean(C) * 100, np.mean(S) * 100
        print(f"  {label}  alpha={alpha:.2f}:  certified {c:5.1f}%   Q-served by the "
              f"current rule {s:.1f}%   -> {min(c/s,1.0)*100:5.1f}% of Q-served traffic is certified"
              f"   (wasted FP on certified inputs: {np.mean(ACC)*100:.1f}%)")
    print()
