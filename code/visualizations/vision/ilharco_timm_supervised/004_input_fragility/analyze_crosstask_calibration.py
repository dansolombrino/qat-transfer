"""Produces F14 / MATERIAL.md 2.4b: leave-one-task-out calibration of eps-hat, i.e. whether a
new task needs any FP access at all. NOTE its section E uses a 2x threshold for the pairwise
score, which is WRONG -- superseded by analyze_contender_certificate.py. Kept for the LOO table."""

import os,sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]   # repo root, derived not hardcoded; sys.path.insert(0,str(_R/"code")); os.chdir(_R)
from dotenv import load_dotenv; load_dotenv(_R/".env")
import numpy as np, pandas as pd
from src.vision.utils import sanitize_timm_model_name
B=Path(os.environ["CHECKPOINT_BASE_PATH"])/"vision"/"ilharco_timm_supervised"
def load(model,bs,bits,gran="channel"):
    base=B/"gap_profile"/sanitize_timm_model_name(model); out={}
    for ds in sorted(base.iterdir()):
        p=(ds/f"optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs={bs}"
           /f"ptq=bits={bits}_gran={gran}_skip=head"/"seed=2038"/"gap_profile_test.parquet")
        if p.exists(): out[ds.name]=pd.read_parquet(p)
    return out
def eps_pair(d):
    """Perturbation of the TOP-2 GAP itself, not of all D logits.
    Defined where both models agree on the top-2 SET (order may differ)."""
    same=((d.fp_cls_1.values==d.q_cls_1.values)&(d.fp_cls_2.values==d.q_cls_2.values))
    e=np.abs(d.fp_gap_2.values-d.q_gap_2.values)
    return e,same
CFG=[("vit_base_patch16_224.orig_in21k","128",4,"ViT-B W4"),
     ("vit_large_patch16_224.orig_in21k","64",4,"ViT-L W4"),
     ("vit_base_patch16_224.orig_in21k","128",3,"ViT-B W3")]
rng=np.random.default_rng(0)
print("="*82)
print("E. TIGHTER CERTIFICATE: calibrate on the TOP-2 GAP perturbation, not ||dz||_inf")
print("="*82)
print("  Only the contender coordinates can flip the top-1, so eps_inf over all D classes")
print("  is loose. Score: eps_pair = |(zF1-zF2) - (zQ1-zQ2)|.  Same conformal structure.\n")
for model,bs,bits,label in CFG:
    tasks=load(model,bs,bits); print(f"  {label}")
    for alpha in (0.10,0.05,0.01):
        R={"inf":[[],[]],"pair":[[],[]]}
        for ds,d in tasks.items():
            if len(d)<200: continue
            idx=rng.permutation(len(d)); h=len(d)//2
            cal,tst=d.iloc[idx[:h]],d.iloc[idx[h:]]
            flip=tst.fp_top1.values!=tst.q_top1.values
            ep_c,sm_c=eps_pair(cal)
            for k,ec in (("inf",cal.eps_linf.values),("pair",ep_c[sm_c])):
                if len(ec)<20: continue
                n=len(ec); q=min(np.ceil((n+1)*(1-alpha))/n,1.0)
                cert=tst.q_gap_2.values>=2*np.quantile(ec,q)
                R[k][0].append(cert.mean()); R[k][1].append(flip[cert].mean() if cert.sum() else 0.0)
        m=lambda a:np.mean(a)*100 if len(a) else float("nan")
        print(f"    alpha={alpha:.2f}   eps_inf: cert {m(R['inf'][0]):5.1f}% / flips {m(R['inf'][1]):.3f}%"
              f"   |   eps_pair: cert {m(R['pair'][0]):5.1f}% / flips {m(R['pair'][1]):.3f}%")
    print()

print("="*82)
print("F. CROSS-TASK CALIBRATION (leave-one-task-out): does eps-hat transfer between tasks?")
print("="*82)
print("  Calibrate eps-hat on the 20 OTHER tasks, certify the held-out task. If this holds,")
print("  a new task needs no FP access at all.\n")
for model,bs,bits,label in CFG:
    tasks=load(model,bs,bits); ks=[k for k,v in tasks.items() if len(v)>=200]
    print(f"  {label}")
    for alpha in (0.10,0.05,0.01):
        E,C,W=[],[],0
        for ho in ks:
            pool=np.concatenate([tasks[k].eps_linf.values for k in ks if k!=ho])
            n=len(pool); q=min(np.ceil((n+1)*(1-alpha))/n,1.0)
            eh=np.quantile(pool,q); t=tasks[ho]
            cert=t.q_gap_2.values>=2*eh; flip=t.fp_top1.values!=t.q_top1.values
            E.append(cert.mean()); fr=flip[cert].mean() if cert.sum() else 0.0
            C.append(fr); W=max(W,fr)
        print(f"    alpha={alpha:.2f}  certified {np.mean(E)*100:5.1f}%   mean flip {np.mean(C)*100:.3f}%"
              f"   WORST-TASK flip {W*100:.3f}%   (bound {alpha*100:.0f}%)")
    print()
