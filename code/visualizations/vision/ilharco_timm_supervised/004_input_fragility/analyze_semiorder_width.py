"""Produces F13 / MATERIAL.md 2.3 (semiorder width |C_eps| at the theory threshold 2*eps) and
F15 / 2.6 (the negative result on whether fragility onset tracks FP decision-readiness)."""

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

print("="*78); print("A. SEMIORDER WIDTH at the THEORY threshold 2*eps (not the routing percentile)")
print("="*78)
for model,bs,bits,gran,label in CFG:
    W=[]; 
    for ds,d in load("gap_profile",model,bs,bits,gran,"gap_profile_test.parquet").items():
        if int(d["bad"].sum())<10: continue
        K=len([c for c in d.columns if c.startswith("q_gap_")])
        G=np.stack([d[f"q_gap_{k}"].values for k in range(1,K+1)],1)
        w=(G<2*d["eps_linf"].values[:,None]).sum(1)          # |C_eps| per input, capped at K
        W.append([(w==1).mean(),(w==2).mean(),(w==3).mean(),(w>=4).mean(),
                  (w>=K).mean(),w.mean()])
    W=np.array(W).mean(0)
    print(f"  {label:9s} |C|=1 {W[0]*100:5.1f}%  |C|=2 {W[1]*100:5.1f}%  |C|=3 {W[2]*100:5.1f}%  "
          f"|C|>=4 {W[3]*100:5.1f}%  (censored at K: {W[4]*100:4.1f}%)  mean {W[5]:.2f}")

print(); print("="*78)
print("B. IS THE ONSET WHERE THE **FP** MODEL BECOMES DECISION-READY?  (no PTQ needed)")
print("="*78)
for model,bs,bits,gran,label in CFG:
    tasks=load("layerwise_logit_lens",model,bs,bits,gran,"layerwise_test.parquet")
    if not tasks: print(f"  {label}: no lens data"); continue
    nb=len([c for c in next(iter(tasks.values())).columns if c.startswith("fp_top1_l")])
    ACC,AUR=[],[]
    for ds,d in tasks.items():
        if int(d["bad"].sum())<10: continue
        lab=d["label"].values
        ACC.append([(d[f"fp_top1_l{i}"].values==lab).mean() for i in range(nb)])
        fpc=d[d.fp_correct]; y=fpc["bad"].values.astype(int)
        if len(set(y))>1:
            AUR.append([roc_auc_score(y,-fpc[f"q_margin_l{i}"].values) for i in range(nb)])
    acc=np.nanmean(ACC,0); aur=np.nanmean(AUR,0)
    onset=next((i for i in range(nb) if aur[i]>0.55),None)
    ready95=next((i for i in range(nb) if acc[i]>=0.95*acc[-1]),None)
    ready90=next((i for i in range(nb) if acc[i]>=0.90*acc[-1]),None)
    print(f"\n  {label}  ({nb} blocks, {len(ACC)} tasks)")
    print(f"    FP lens acc : "+" ".join(f"{a*100:.0f}" for a in acc))
    print(f"    frag AUROC  : "+" ".join(f"{a*100:.0f}" for a in aur))
    f=lambda i:f"b{i} ({i/(nb-1)*100:.0f}%)" if i is not None else "never"
    print(f"    FP ready@90% {f(ready90)}   FP ready@95% {f(ready95)}   fragility onset {f(onset)}")
