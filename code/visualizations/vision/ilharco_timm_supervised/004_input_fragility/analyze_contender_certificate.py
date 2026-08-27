"""Produces F14 / MATERIAL.md 2.4b: the contender-only certificate. The threshold is 1x the
pairwise score, NOT 2x -- the score is already a difference of two coordinates. This nearly
doubles certified coverage at alpha=0.01 versus the ||.||_inf form."""

import os,sys
from pathlib import Path
_R = Path(__file__).resolve().parents[5]   # repo root, derived not hardcoded
sys.path.insert(0, str(_R / "code"))
os.chdir(_R)
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

def pair_score(d):
    """|Delta(z_a - z_b)| on the pair (a,b) = the QUANTIZED model's own top-2.
    Always defined when both classes are locatable in the stored FP top-K.
    Flip of that pair requires q_gap_2 < this score -- a factor of 1, not 2."""
    K=len([c for c in d.columns if c.startswith("fp_cls_")])
    fpc=np.stack([d[f"fp_cls_{j}"].values for j in range(1,K+1)],1)
    fpg=np.stack([d[f"fp_gap_{j}"].values for j in range(1,K+1)],1)
    a,b=d.q_cls_1.values,d.q_cls_2.values
    pa=(fpc==a[:,None]); pb=(fpc==b[:,None])
    ok=pa.any(1)&pb.any(1)
    ga=np.where(pa,fpg,np.nan); gb=np.where(pb,fpg,np.nan)
    gF=np.nanmin(gb,axis=1)-np.nanmin(ga,axis=1)      # zF[a]-zF[b]
    return np.abs(d.q_gap_2.values-gF),ok

CFG=[("vit_base_patch16_224.orig_in21k","128",4,"ViT-B W4"),
     ("vit_large_patch16_224.orig_in21k","64",4,"ViT-L W4"),
     ("vit_base_patch16_224.orig_in21k","128",3,"ViT-B W3")]
rng=np.random.default_rng(0)
print("="*84)
print("E'. CONTENDER-ONLY CERTIFICATE (corrected threshold: 1x eps_pair, not 2x)")
print("="*84)
for model,bs,bits,label in CFG:
    tasks=load(model,bs,bits); print(f"  {label}")
    for alpha in (0.10,0.05,0.01):
        A,Bc,Cc,Dd,COV=[],[],[],[],[]
        for ds,d in tasks.items():
            if len(d)<200: continue
            idx=rng.permutation(len(d)); h=len(d)//2
            cal,tst=d.iloc[idx[:h]],d.iloc[idx[h:]]
            flip=tst.fp_top1.values!=tst.q_top1.values
            n=len(cal); q=min(np.ceil((n+1)*(1-alpha))/n,1.0)
            c1=tst.q_gap_2.values>=2*np.quantile(cal.eps_linf.values,q)
            A.append(c1.mean()); Bc.append(flip[c1].mean() if c1.sum() else 0.0)
            ps,ok=pair_score(cal)
            if ok.sum()<20: continue
            m=ok.sum(); q2=min(np.ceil((m+1)*(1-alpha))/m,1.0)
            c2=tst.q_gap_2.values>=np.quantile(ps[ok],q2)
            Cc.append(c2.mean()); Dd.append(flip[c2].mean() if c2.sum() else 0.0)
            COV.append(ok.mean())
        f=lambda a:np.mean(a)*100 if len(a) else float("nan")
        print(f"    alpha={alpha:.2f}   eps_inf: cert {f(A):5.1f}% / flips {f(Bc):.3f}%"
              f"   |   contender-only: cert {f(Cc):5.1f}% / flips {f(Dd):.3f}%"
              f"   [score defined on {f(COV):.0f}% of cal]")
    print()
