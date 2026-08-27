"""How much of the RANKING does quantization preserve, versus how much of the argmax?

C2's certificate almost never fires for k>=2. The reason is worth quantifying: the
argmax is protected by the largest gap in the profile, and z_(k) - z_(k+1) shrinks fast,
so 2*eps swamps every deeper cut. If that is right then a model whose accuracy survives
quantization can still have its rankings scrambled -- which is a warning for anyone
serving top-k from a quantized embedding model.

Measures, per config and k: top-k set overlap (Jaccard), exact-set-match rate,
Kendall tau on the FP top-k order, and the separation gap that C2 depends on.

Produces F17 / MATERIAL.md 2.7 -- SECOND HEADLINE: quantization preserves the argmax far
better than the ranking. The top-1 gap is ~10x larger relative to the perturbation than every
deeper gap, so the exact top-5 set survives for only 18-45% of inputs.
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
from src.vision.utils import sanitize_hf_model_name, sanitize_timm_model_name

_CBP = Path(os.environ["CHECKPOINT_BASE_PATH"])
CONFIGS = [
    ("text", "ilharco_automodelforsequenceclassification", "Qwen/Qwen3-Embedding-0.6B",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128", 4, "score", "Qwen3 W4"),
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


def kendall_on_topk(fp, qp, k):
    """Kendall tau between the FP top-k order and where those same classes sit in Q.
    Classes absent from Q's stored top-K are placed after everything present."""
    n, K = qp.shape
    pos = np.full((n, k), K + 1, dtype=np.int32)
    for j in range(k):
        hit = (qp == fp[:, j][:, None])
        has = hit.any(1)
        pos[has, j] = hit[has].argmax(1)
    conc = np.zeros(n); disc = np.zeros(n)
    for a in range(k):
        for b in range(a + 1, k):
            d = pos[:, b] - pos[:, a]
            conc += (d > 0); disc += (d < 0)
    tot = conc + disc
    return np.divide(conc - disc, np.maximum(tot, 1))


print("=" * 92)
print("TOP-k RANKING FRAGILITY: quantization preserves the argmax far better than the ranking")
print("=" * 92)
print("  exact = FP top-k set equals PTQ top-k set.  overlap = |intersection| / k.")
print("  tau = Kendall tau on the FP top-k classes' relative order under PTQ.")
print("  sep/2eps = median (z_(k) - z_(k+1)) / 2 eps -- C2's criterion; >= 1 means certifiable.\n")
for domain, fam, model, hp, bits, skip, label in CONFIGS:
    tasks = load(domain, fam, model, hp, bits, skip)
    if not tasks:
        print(f"  {label}: no data"); continue
    print(f"  {label}")
    for k in (1, 2, 3, 5, 10):
        EX, OV, TA, SEP = [], [], [], []
        for ds, d in tasks.items():
            K = len([c for c in d.columns if c.startswith("q_cls_")])
            if K < k + 1 or len(d) < 200:
                continue
            fp = np.stack([d[f"fp_cls_{j}"].values for j in range(1, k + 1)], 1)
            qp_k = np.stack([d[f"q_cls_{j}"].values for j in range(1, k + 1)], 1)
            qp_full = np.stack([d[f"q_cls_{j}"].values for j in range(1, K + 1)], 1)
            EX.append((np.sort(fp, 1) == np.sort(qp_k, 1)).all(1).mean())
            inter = np.array([len(np.intersect1d(fp[i], qp_k[i])) for i in range(len(d))])
            OV.append((inter / k).mean())
            if k >= 2:
                TA.append(np.mean(kendall_on_topk(fp, qp_full, k)))
            sep = d[f"q_gap_{k+1}"].values - d[f"q_gap_{k}"].values
            SEP.append(np.median(sep / (2 * np.maximum(d["eps_linf"].values, 1e-9))))
        if not EX:
            continue
        t = f"{np.mean(TA):.3f}" if TA else "  -- "
        print(f"    k={k:<3d} exact-set {np.mean(EX)*100:5.1f}%   overlap {np.mean(OV)*100:5.1f}%"
              f"   tau {t}   median sep/2eps {np.mean(SEP):.3f}")
    print()
