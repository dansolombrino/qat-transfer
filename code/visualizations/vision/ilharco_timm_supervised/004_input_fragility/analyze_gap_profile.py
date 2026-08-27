"""Settle the three de-risking questions and characterise eps, at 21-task scale.

Q1 graded routing : distribution of |C_eps| among routed inputs
Q2 ranking vs margin : does the full gap profile beat gap_2 at predicting `bad`?
Q3 lucky-Q ceiling : can any gap feature separate `bad` from `lucky-Q`? (Prop. 2)
eps : ||z_PTQ - z_FP||_inf, and Prop. 1's recall as a per-input bound
"""
import argparse, json, os, sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_R/"code")); os.chdir(_R)
from dotenv import load_dotenv; load_dotenv(_R/".env")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from src.vision.utils import sanitize_timm_model_name

def load(model, bs, bits, gran="channel"):
    base=(Path(os.environ["CHECKPOINT_BASE_PATH"])/"vision"/"ilharco_timm_supervised"
          /"gap_profile"/sanitize_timm_model_name(model))
    out={}
    if not base.exists(): return out
    for ds in sorted(base.iterdir()):
        p=(ds/f"optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs={bs}"
           /f"ptq=bits={bits}_gran={gran}_skip=head"/"seed=2038"/"gap_profile_test.parquet")
        if p.exists(): out[ds.name]=pd.read_parquet(p)
    return out

def auc(y, s):
    return roc_auc_score(y, s) if len(set(y))>1 else np.nan

def cv_auc(X, y):
    if len(set(y))<2 or min(np.bincount(y.astype(int)))<5: return np.nan
    Xs=(X-X.mean(0))/(X.std(0)+1e-9)
    p=cross_val_predict(LogisticRegression(max_iter=3000,class_weight="balanced"),
                        Xs,y,cv=5,method="predict_proba")[:,1]
    return roc_auc_score(y,p)

for model,bs,bits,label in [("vit_base_patch16_224.orig_in21k","128",4,"ViT-B W4-channel"),
                            ("vit_large_patch16_224.orig_in21k","64",4,"ViT-L W4-channel"),
                            ("vit_base_patch16_224.orig_in21k","128",3,"ViT-B W3-channel")]:
    tasks=load(model,bs,bits)
    if not tasks: print(f"\n### {label}: no data"); continue
    E,REC,C2,C3,Cbig,Q2m,Q2f,Q3m,Q3f,NB=[],[],[],[],[],[],[],[],[],[]
    for ds,d in tasks.items():
        if int(d["bad"].sum())<10: continue
        NB.append(int(d["bad"].sum()))
        e=d["eps_linf"].values; E.append(np.median(e))
        pred=d["q_gap_2"].values < 2*e
        act=d["fp_top1"].values!=d["q_top1"].values
        REC.append((pred&act).sum()/max(act.sum(),1))
        K=[c for c in d.columns if c.startswith("q_gap_")]
        G=np.stack([d[f"q_gap_{k}"].values for k in range(1,len(K)+1)],1)
        tau=np.quantile(d["q_gap_2"].values,0.25); routed=d["q_gap_2"].values<tau
        cont=(G<tau).sum(1)[routed]
        C2.append((cont==2).mean()); C3.append((cont==3).mean()); Cbig.append((cont>=4).mean())
        fpc=d[d.fp_correct]; y=fpc["bad"].values.astype(int)
        gp=np.stack([fpc[f"q_gap_{k}"].values for k in range(2,len(K)+1)],1)
        Q2m.append(auc(y,-gp[:,0])); Q2f.append(cv_auc(gp,y))
        sub=d[d.bad|d.lucky_q]; y2=sub["bad"].values.astype(int)
        g2=np.stack([sub[f"q_gap_{k}"].values for k in range(2,len(K)+1)],1)
        Q3m.append(auc(y2,-g2[:,0])); Q3f.append(cv_auc(g2,y2))
    m=lambda a: (np.nanmean(a), np.nanstd(a))
    print(f"\n### {label}  ({len(NB)} eligible tasks, {sum(NB)} bad samples)")
    print(f"  eps_linf (median per task): {m(E)[0]:.3f} +- {m(E)[1]:.3f}")
    print(f"  Prop.1 recall (flips with margin < 2 eps): {m(REC)[0]*100:.1f}% +- {m(REC)[1]*100:.1f}")
    print(f"  Q1 contender count among routed: |C|=2 {m(C2)[0]*100:.1f}%  |C|=3 {m(C3)[0]*100:.1f}%  |C|>=4 {m(Cbig)[0]*100:.1f}%")
    print(f"  Q2 predict 'bad'   : margin {m(Q2m)[0]:.3f}   full profile {m(Q2f)[0]:.3f}   delta {m(Q2f)[0]-m(Q2m)[0]:+.3f}")
    print(f"  Q3 bad vs lucky-Q  : margin {m(Q3m)[0]:.3f}   full profile {m(Q3f)[0]:.3f}   (0.5 = Prop.2 holds)")
