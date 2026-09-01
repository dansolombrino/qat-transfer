"""The two-sided table: capture per policy on classification, both modalities, RTN@3.5."""
import json, glob, os
import numpy as np

ROOTS=[("vision","storage/checkpoints/vision/ilharco_timm_supervised/gap_allocation_clsf"),
       ("text","storage/checkpoints/text/ilharco_automodelforsequenceclassification/gap_allocation_clsf")]
POLS=[("random","Random"),("mse","MSE-driven"),("gap","Gap-driven")]

def pooled(rows,pol,key,higher):
    g=h=0.0;n=0
    for r in rows:
        pols=[p for p in r if p.startswith("random")] if pol=="random" else [pol]
        for p in pols:
            if p not in r or key not in r[p]: continue
            lo,hi,v=r["floor"][key],r["ceil"][key],r[p][key]
            if not higher: lo,hi,v=-lo,-hi,-v
            if hi-lo<=1e-6: continue
            g+=v-lo; h+=hi-lo; n+=1
    return (g/h if h else float("nan")), n

def main():
    data={}
    for tag,root in ROOTS:
        rows=[]
        for f in glob.glob(os.path.expanduser(f"~/qat-transfer/{root}/**/gap_allocation_clsf.json"),recursive=True):
            j=json.load(open(f))
            if j["method"]!="rtn" or j["seed"]!=2038: continue
            if j["bits"]!=3 or j["granularity"]!="group_128" or abs(j["avg_bits_target"]-3.5)>1e-6: continue
            rows.append(j["results"])
        data[tag]=rows
    L=[r"\begin{table}[t]",r"\centering",r"\small",
       r"\caption{\textbf{The rule inverts on classification.} Capture of the W3$\to$W4 benefit "
       r"at a 3.5-bit average on classification (RTN; 2 ViTs $\times$ 21 datasets, "
       r"Qwen3-0.6B $\times$ 11 tasks). The criteria that win retrieval lose here and vice "
       r"versa: curvature of the task loss is the right signal where failures are magnitude "
       r"failures, and the gap criterion measures a quantity that rarely decides the outcome.}",
       r"\label{tab:twosided}",r"\begin{tabular}{lcc}",r"\toprule",
       r"Allocation & Vision clsf.\ (flip) & Text clsf.\ (flip) \\",r"\midrule"]
    for pol,lab in POLS:
        cells=[]
        for tag in ("vision","text"):
            c,n=pooled(data[tag],pol,"flip",False)
            cells.append("--" if np.isnan(c) else f"{c*100:.1f}\\%")
        L.append(f"{lab} & "+" & ".join(cells)+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(os.path.expanduser("~/qat-transfer/paper/tables/new_twosided.tex"),"w").write("\n".join(L)+"\n")
    print("wrote new_twosided.tex")

if __name__=="__main__":
    main()
