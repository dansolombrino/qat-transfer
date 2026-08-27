"""Does ranking fragility scale with the number of candidates? Free test on existing data.

The 21 vision tasks span label spaces from 10 to 397 classes, and the Qwen3 tasks from 2 to
77. If the mechanism behind section 2.7 is "more candidates packed into the same score range
=> smaller gaps => more flips", then top-k set survival should fall with the candidate count
*within* the classification data we already have -- no retrieval needed.

This is the bridge between the classification result and the retrieval result.

Produces F21 / MATERIAL.md 2.9 -- FRAMES 2.7: the driver is candidate count, not modality.
The separation ratio falls monotonically as the label space grows, on existing data, across
all three backbones.
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
CFG = [
    ("vision", "ilharco_timm_supervised", "vit_base_patch16_224.orig_in21k",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128", "head", "ViT-B"),
    ("vision", "ilharco_timm_supervised", "vit_large_patch16_224.orig_in21k",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=64", "head", "ViT-L"),
    ("text", "ilharco_automodelforsequenceclassification", "Qwen/Qwen3-Embedding-0.6B",
     "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128", "score", "Qwen3"),
]


def load(domain, fam, model, hp, skip, bits, gran):
    san = sanitize_hf_model_name if domain == "text" else sanitize_timm_model_name
    base = _CBP / domain / fam / "gap_profile" / san(model)
    out = {}
    if not base.exists():
        return out
    for ds in sorted(base.iterdir()):
        p = (ds / hp / f"ptq=bits={bits}_gran={gran}_skip={skip}" / "seed=2038"
             / "gap_profile_test.parquet")
        if p.exists():
            out[ds.name] = pd.read_parquet(p)
    return out


rows = []
for domain, fam, model, hp, skip, label in CFG:
    for bits, gran in ((4, "group_128"), (4, "channel")):
        for ds, d in load(domain, fam, model, hp, skip, bits, gran).items():
            n_cls = int(max(d.label.max(), d.fp_top1.max(), d.q_top1.max())) + 1
            K = len([c for c in d.columns if c.startswith("q_cls_")])
            if K < 5 or len(d) < 200:
                continue
            e = d.eps_linf.values
            r = {"backbone": label, "cfg": f"W{bits}-{gran}", "task": ds,
                 "n_classes": n_cls, "n": len(d),
                 "flip1": (d.fp_top1.values != d.q_top1.values).mean()}
            for k in (1, 3, 5):
                fs = np.sort(np.stack([d[f"fp_cls_{j}"].values for j in range(1, k+1)], 1), 1)
                qs = np.sort(np.stack([d[f"q_cls_{j}"].values for j in range(1, k+1)], 1), 1)
                r[f"exact{k}"] = (fs == qs).all(1).mean()
            sep2 = d["q_gap_3"].values - d["q_gap_2"].values
            r["sep2_over_2eps"] = np.median(sep2 / (2 * np.maximum(e, 1e-9)))
            rows.append(r)

df = pd.DataFrame(rows)
if df.empty:
    print("no data"); sys.exit()

for cfg in sorted(df.cfg.unique()):
    sub = df[df.cfg == cfg]
    print("=" * 92)
    print(f"{cfg}   ({len(sub)} task-rows across {sub.backbone.nunique()} backbones)")
    print("=" * 92)
    bins = [(0, 12, "<=12"), (12, 50, "13-50"), (50, 150, "51-150"), (150, 10000, ">150")]
    print(f"  {'label space':<12} {'tasks':>5} {'med #cls':>9} {'top-1 flip':>11} "
          f"{'exact top-3':>12} {'exact top-5':>12} {'sep2/2eps':>10}")
    for lo, hi, name in bins:
        b = sub[(sub.n_classes > lo) & (sub.n_classes <= hi)]
        if b.empty:
            continue
        print(f"  {name:<12} {len(b):>5} {int(b.n_classes.median()):>9} "
              f"{b.flip1.mean()*100:>10.1f}% {b.exact3.mean()*100:>11.1f}% "
              f"{b.exact5.mean()*100:>11.1f}% {b.sep2_over_2eps.mean():>10.3f}")
    if len(sub) > 3:
        lc = np.log10(sub.n_classes.values)
        for col in ("exact5", "exact3", "sep2_over_2eps"):
            r = np.corrcoef(lc, sub[col].values)[0, 1]
            print(f"    corr(log10 #classes, {col:<15}) = {r:+.3f}")
    print()

print("Per-task detail at W4-group_128, sorted by label-space size:")
s = df[df.cfg == "W4-group_128"].sort_values("n_classes")
if s.empty:
    s = df[df.cfg == "W4-channel"].sort_values("n_classes")
    print("  (group_128 unavailable; showing channel)")
for _, r in s.iterrows():
    print(f"  {r.backbone:<6} {r.task:<22} {r.n_classes:>4} cls   "
          f"top-1 flip {r.flip1*100:5.1f}%   exact top-5 {r.exact5*100:5.1f}%   "
          f"sep2/2eps {r.sep2_over_2eps:.3f}")
