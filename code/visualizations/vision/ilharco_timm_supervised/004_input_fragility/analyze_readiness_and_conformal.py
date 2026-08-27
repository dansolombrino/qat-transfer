"""Produces F15 / MATERIAL.md 2.6 (does the fragility signal track depth or FP readiness --
inconclusive) and the first split-conformal certificate numbers behind F14 / 2.4b."""

import os,sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]   # repo root, derived not hardcoded; sys.path.insert(0,str(_R/"code")); os.chdir(_R)
from dotenv import load_dotenv; load_dotenv(_R/".env")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.vision.utils import sanitize_timm_model_name
B=Path(os.environ["CHECKPOINT_BASE_PATH"])/"vision"/"ilharco_timm_supervised"
def load(kind,model,bs,bits,gran,fn):
    base=B/kind/sanitize_timm_model_name(model); out={}
    if not base.exists(): return out
    for ds in sorted(base.iterdir()):
        p=(ds/f"optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs={bs}"
           /f"ptq=bits={bits}_gran={gran}_skip=head"/"seed=2038"/fn)
        if p.exists(): out[ds.name]=pd.read_parquet(p)
    return out
CFG=[("vit_base_patch16_224.orig_in21k","128",4,"channel","ViT-B W4"),
     ("vit_large_patch16_224.orig_in21k","64",4,"channel","ViT-L W4"),
     ("vit_base_patch16_224.orig_in21k","128",3,"channel","ViT-B W3")]

print("="*80)
print("C. WHAT DOES THE FRAGILITY SIGNAL TRACK -- DEPTH, OR FP DECISION-READINESS?")
print("="*80)
print("  For each AUROC level, report (rel. depth at crossing, FP-acc as %% of final at crossing).")
print("  If the signal tracks READINESS, the second column matches across architectures.\n")
rows={}
for model,bs,bits,gran,label in CFG:
    tasks=load("layerwise_logit_lens",model,bs,bits,gran,"layerwise_test.parquet")
    if not tasks: continue
    nb=len([c for c in next(iter(tasks.values())).columns if c.startswith("fp_top1_l")])
    ACC,AUR=[],[]
    for ds,d in tasks.items():
        if int(d["bad"].sum())<10: continue
        lab=d["label"].values
        ACC.append([(d[f"fp_top1_l{i}"].values==lab).mean() for i in range(nb)])
        fpc=d[d.fp_correct]; y=fpc["bad"].values.astype(int)
        if len(set(y))>1: AUR.append([roc_auc_score(y,-fpc[f"q_margin_l{i}"].values) for i in range(nb)])
    acc=np.nanmean(ACC,0); aur=np.nanmean(AUR,0); rows[label]=(nb,acc,aur)
    print(f"  {label}:")
    for lvl in (0.55,0.60,0.65,0.70,0.75,0.80):
        i=next((j for j in range(nb) if aur[j]>=lvl),None)
        if i is None: print(f"    AUROC>={lvl:.2f}: never reached"); continue
        print(f"    AUROC>={lvl:.2f}: block b{i:<2d}  rel.depth {i/(nb-1)*100:5.1f}%   "
              f"FP-acc {acc[i]*100:4.1f}% = {acc[i]/acc[-1]*100:5.1f}% of final")
    print()
if "ViT-B W4" in rows and "ViT-L W4" in rows:
    print("  Cross-architecture spread (ViT-B W4 vs ViT-L W4), lower = better invariant:")
    for lvl in (0.55,0.60,0.65,0.70,0.75,0.80):
        v=[]
        for lab in ("ViT-B W4","ViT-L W4"):
            nb,acc,aur=rows[lab]; i=next((j for j in range(nb) if aur[j]>=lvl),None)
            v.append((i/(nb-1),acc[i]/acc[-1]) if i is not None else (np.nan,np.nan))
        print(f"    AUROC>={lvl:.2f}  |d_depth| {abs(v[0][0]-v[1][0])*100:5.1f} pts   "
              f"|d_readiness| {abs(v[0][1]-v[1][1])*100:5.1f} pts")

print(); print("="*80)
print("D. SPLIT-CONFORMAL CERTIFICATE: calibrate eps-hat, then certify with ZERO FP compute")
print("="*80)
print("  Prop.1: q_margin >= 2*eps  =>  q_top1 == fp_top1.  eps is unknown at test time, so")
print("  take eps_hat = the (1-alpha) quantile of eps on a held-out calibration split.")
print("  Guarantee: P(certified AND flipped) <= alpha.  Measured below on the test half.\n")
rng=np.random.default_rng(0)
for model,bs,bits,gran,label in CFG:
    tasks=load("gap_profile",model,bs,bits,gran,"gap_profile_test.parquet")
    print(f"  {label}")
    for alpha in (0.10,0.05,0.01):
        COV,EFF,ACCG=[],[],[]
        for ds,d in tasks.items():
            if len(d)<200: continue
            idx=rng.permutation(len(d)); h=len(d)//2
            cal,tst=d.iloc[idx[:h]],d.iloc[idx[h:]]
            n=len(cal); q=min(np.ceil((n+1)*(1-alpha))/n,1.0)
            eh=np.quantile(cal["eps_linf"].values,q)
            cert=tst["q_gap_2"].values>=2*eh
            flip=tst["fp_top1"].values!=tst["q_top1"].values
            if cert.sum()==0: COV.append(0.0); EFF.append(0.0); continue
            COV.append(flip[cert].mean()); EFF.append(cert.mean())
            ACCG.append((tst["q_top1"].values==tst["label"].values)[cert].mean())
        print(f"    alpha={alpha:.2f}  certified {np.mean(EFF)*100:5.1f}% of inputs   "
              f"observed flip-rate among certified {np.mean(COV)*100:.3f}%   "
              f"(bound {alpha*100:.0f}%)   acc on certified {np.mean(ACCG)*100:.1f}%")
    print()
