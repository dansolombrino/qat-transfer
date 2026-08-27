"""Produces F13 / MATERIAL.md 2.3: eps/margin as the regime parameter, how tight the Prop. 1
bound is, and whether AUROC improves with gap-profile depth."""

import os,sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]   # repo root, derived not hardcoded; sys.path.insert(0,str(_R/"code")); os.chdir(_R)
from dotenv import load_dotenv; load_dotenv(_R/".env")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from src.vision.utils import sanitize_timm_model_name

def load(model,bs,bits,gran="channel"):
    base=(Path(os.environ["CHECKPOINT_BASE_PATH"])/"vision"/"ilharco_timm_supervised"
          /"gap_profile"/sanitize_timm_model_name(model)); out={}
    for ds in sorted(base.iterdir()):
        p=(ds/f"optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs={bs}"
           /f"ptq=bits={bits}_gran={gran}_skip=head"/"seed=2038"/"gap_profile_test.parquet")
        if p.exists(): out[ds.name]=pd.read_parquet(p)
    return out

def cv(X,y):
    if len(set(y))<2 or min(np.bincount(y.astype(int)))<5: return np.nan
    Xs=(X-X.mean(0))/(X.std(0)+1e-9)
    return roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=3000,class_weight="balanced"),
        Xs,y,cv=5,method="predict_proba")[:,1])

for model,bs,bits,label in [("vit_base_patch16_224.orig_in21k","128",4,"ViT-B W4"),
                            ("vit_large_patch16_224.orig_in21k","64",4,"ViT-L W4"),
                            ("vit_base_patch16_224.orig_in21k","128",3,"ViT-B W3")]:
    tasks=load(model,bs,bits); REL,PREC,FIRE,KS=[],[],[],[]
    for ds,d in tasks.items():
        if int(d["bad"].sum())<10: continue
        e=d["eps_linf"].values; g2=d["q_gap_2"].values
        # eps relative to the logit SCALE of the task (median FP top1-top2 gap)
        REL.append(np.median(e)/max(np.median(d["fp_gap_2"].values),1e-9))
        # how CONSERVATIVE is Prop.1: of inputs the bound calls "at risk", how many flip?
        at_risk=g2<2*e; flip=d["fp_top1"].values!=d["q_top1"].values
        PREC.append(flip[at_risk].mean() if at_risk.sum() else np.nan); FIRE.append(at_risk.mean())
        # where does the W3 ranking gain live? incremental AUROC as k grows
        fpc=d[d.fp_correct]; y=fpc["bad"].values.astype(int)
        kmax=len([c for c in d.columns if c.startswith("q_gap_")])
        KS.append([cv(np.stack([fpc[f"q_gap_{j}"].values for j in range(2,min(K,kmax)+1)],1),y)
                   if kmax>=2 else np.nan for K in (2,3,5,10)])
    m=lambda a:np.nanmean(a)
    print(f"\n### {label}")
    print(f"  eps / median FP margin      : {m(REL):.3f}   (scale-free: is eps really bigger, or are logits?)")
    print(f"  Prop.1 bound fires on        : {m(FIRE)*100:.1f}% of inputs")
    print(f"  ... of those, actually flip  : {m(PREC)*100:.1f}%   (bound tightness)")
    K=np.array(KS); print(f"  AUROC vs profile depth k<=2/3/5/10 : "+"  ".join(f"{v:.3f}" for v in np.nanmean(K,0)))
