"""Does group_128 restore certifiability? And does a precision ladder actually save compute?

F11 showed group_128 restores the *representation* at W3. The operational question is
whether it restores the *certificate*, which channel-granularity W3 fails completely.

Then the ladder. Graded routing only pays if a cheaper rung can settle inputs the cheapest
rung cannot certify. Rungs by real cost are W3 < W4 < FP (channel vs group_128 at equal bits
costs the same MACs -- only the dequant scales differ -- so group_128 is not a rung, it is
strictly better at equal cost). So the ladder is W3-g128 -> W4-g128 -> FP, and the question is
what expected cost it achieves against always-W4 plus routing.

Produces F18 / MATERIAL.md 2.8: group_128 is the practical recommendation (certified coverage
~79% at W4 for free), and the precision ladder is dead (35-36% MORE expensive than
always-W4 plus routing).
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
VB = ("vision", "ilharco_timm_supervised", "vit_base_patch16_224.orig_in21k",
      "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128", "head")
QW = ("text", "ilharco_automodelforsequenceclassification", "Qwen/Qwen3-Embedding-0.6B",
      "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_mgn=1.0_bs=32_ml=128", "score")


def load(cfg, bits, gran):
    domain, fam, model, hp, skip = cfg
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


def pair_score(d):
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


def certify(cal, tst, alpha):
    ps, ok = pair_score(cal)
    if ok.sum() < 20:
        return None
    m = ok.sum()
    q = min(np.ceil((m + 1) * (1 - alpha)) / m, 1.0)
    return tst.q_gap_2.values >= np.quantile(ps[ok], q)


print("=" * 94)
print("A. DOES group_128 RESTORE CERTIFIABILITY?  (channel vs group_128, same bits)")
print("=" * 94)
rng = np.random.default_rng(0)
for name, cfg in (("ViT-B", VB), ("Qwen3", QW)):
    for bits in (4, 3):
        for gran in ("channel", "group_128"):
            tasks = load(cfg, bits, gran)
            if not tasks:
                print(f"  {name} W{bits}-{gran:9s}: no data")
                continue
            REL, W1, ACC, CERT, FL = [], [], [], [], []
            for ds, d in tasks.items():
                e = d["eps_linf"].values
                REL.append(np.median(e) / max(np.median(d["fp_gap_2"].values), 1e-9))
                K = len([c for c in d.columns if c.startswith("q_gap_")])
                G = np.stack([d[f"q_gap_{k}"].values for k in range(1, K + 1)], 1)
                W1.append(((G < 2 * e[:, None]).sum(1) == 1).mean())
                ACC.append(d["q_correct"].mean())
                if len(d) < 200:
                    continue
                idx = rng.permutation(len(d))
                h = len(d) // 2
                c = certify(d.iloc[idx[:h]], d.iloc[idx[h:]], 0.10)
                if c is None:
                    continue
                t = d.iloc[idx[h:]]
                CERT.append(c.mean())
                FL.append((t.fp_top1.values != t.q_top1.values)[c].mean() if c.sum() else 0.0)
            m = np.nanmean
            print(f"  {name} W{bits}-{gran:9s}  eps/margin {m(REL):5.3f}   "
                  f"|C|=1 {m(W1)*100:5.1f}%   PTQ acc {m(ACC)*100:5.1f}%   "
                  f"certified@0.10 {m(CERT)*100:5.1f}%   flips {m(FL)*100:.3f}%")
    print()

print("=" * 94)
print("B. THE ~10x TOP-1/TOP-2 SEPARATION RATIO — does it survive group_128?")
print("=" * 94)
print("  median (z_(k) - z_(k+1)) / 2 eps.  >= 1 means the k-cut is certifiable.\n")
for name, cfg in (("ViT-B", VB), ("Qwen3", QW)):
    for bits in (4, 3):
        for gran in ("channel", "group_128"):
            tasks = load(cfg, bits, gran)
            if not tasks:
                continue
            row = []
            for k in (1, 2, 3, 5):
                S, E = [], []
                for ds, d in tasks.items():
                    K = len([c for c in d.columns if c.startswith("q_gap_")])
                    if K < k + 1:
                        continue
                    sep = d[f"q_gap_{k+1}"].values - d[f"q_gap_{k}"].values
                    S.append(np.median(sep / (2 * np.maximum(d.eps_linf.values, 1e-9))))
                    fs = np.sort(np.stack([d[f"fp_cls_{j}"].values for j in range(1, k+1)], 1), 1)
                    qs = np.sort(np.stack([d[f"q_cls_{j}"].values for j in range(1, k+1)], 1), 1)
                    E.append((fs == qs).all(1).mean())
                row.append((np.nanmean(S), np.nanmean(E)))
            r = "  ".join(f"k={k}: {s:5.3f} ({e*100:4.1f}%)" for k, (s, e) in zip((1,2,3,5), row))
            print(f"  {name} W{bits}-{gran:9s}  {r}")
    print()

print("=" * 94)
print("C. PRECISION LADDER: W3-g128 -> W4-g128 -> FP, priced against always-W4 + routing")
print("=" * 94)
print("  Rung cost proxy = bits/16 of an FP pass (weight-bound), FP = 1.0.")
print("  A rung settles an input if the certificate fires there at alpha=0.10.")
print("  'errors' = top-1 differs from FP on inputs the ladder settled early.\n")
COST = {3: 3/16, 4: 4/16, "fp": 1.0}
for name, cfg in (("ViT-B", VB), ("Qwen3", QW)):
    t3, t4 = load(cfg, 3, "group_128"), load(cfg, 4, "group_128")
    common = sorted(set(t3) & set(t4))
    if not common:
        print(f"  {name}: no common tasks"); continue
    LAD, LADE, BASE, BASEE = [], [], [], []
    for ds in common:
        d3, d4 = t3[ds], t4[ds]
        if len(d3) != len(d4) or not (d3.label.values == d4.label.values).all():
            continue
        if len(d3) < 200:
            continue
        idx = rng.permutation(len(d3))
        h = len(d3) // 2
        c3 = certify(d3.iloc[idx[:h]], d3.iloc[idx[h:]], 0.10)
        c4 = certify(d4.iloc[idx[:h]], d4.iloc[idx[h:]], 0.10)
        if c3 is None or c4 is None:
            continue
        a3, a4 = d3.iloc[idx[h:]], d4.iloc[idx[h:]]
        f3 = a3.fp_top1.values != a3.q_top1.values
        f4 = a4.fp_top1.values != a4.q_top1.values
        # ladder: settle at W3 if certified; else run W4 and settle if certified; else FP
        at3 = c3
        at4 = ~c3 & c4
        atf = ~c3 & ~c4
        cost = at3.mean()*COST[3] + at4.mean()*(COST[3]+COST[4]) + atf.mean()*(COST[3]+COST[4]+1.0)
        err = (at3 & f3).mean() + (at4 & f4).mean()
        LAD.append(cost); LADE.append(err)
        # baseline: always W4, certify what you can, FP the rest
        bcost = c4.mean()*COST[4] + (~c4).mean()*(COST[4]+1.0)
        BASE.append(bcost); BASEE.append((c4 & f4).mean())
    if not LAD:
        print(f"  {name}: no alignable tasks"); continue
    print(f"  {name}  ladder  cost {np.mean(LAD):.3f} FP-equiv   early-settle errors {np.mean(LADE)*100:.3f}%")
    print(f"  {name}  always-W4 + route  cost {np.mean(BASE):.3f} FP-equiv   errors {np.mean(BASEE)*100:.3f}%")
    ratio = np.mean(LAD) / np.mean(BASE)
    verdict = "MORE expensive" if ratio > 1 else "cheaper"
    print(f"         -> ladder costs {abs(ratio-1)*100:.1f}% {verdict} than the baseline\n")
