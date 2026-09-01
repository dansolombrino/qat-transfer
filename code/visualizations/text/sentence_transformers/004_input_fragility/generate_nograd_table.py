#!/usr/bin/env python3
"""The no-gradient allocation bake-off: every forward-only criterion, every rounding method."""
import json, glob, os, collections
BASE = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/gap_allocation")
OUT  = os.path.expanduser("~/qat-transfer/paper/tables/new_alloc_nograd.tex")
CRIT = [("gap","Gap sensitivity (ours)"), ("actnorm","Activation norm"), ("mse","Reconstruction error"),
        ("relerr","Relative error (SQNR)"), ("position","Depth heuristic"),
        ("hess","Activation-weighted error"), ("awqsal","Activation salience"), ("random0","Random split")]
ROUND = [("rtn","RTN"), ("gptq","\\gptq{}"), ("awq","\\awq{}")]

rows=[j for f in glob.glob(os.path.join(BASE,"**","gap_allocation.json"),recursive=True)
      for j in [json.load(open(f))] if "n_eval_queries" in j and "hess" in j.get("results",{})]
by=collections.defaultdict(list)
for j in rows: by[j.get("method","rtn")].append(j)

def capture(js, c):
    g=h=0.0
    for j in js:
        r=j["results"]
        if c not in r: continue
        lo,hi,v=-r["floor"]["flip"],-r["ceil"]["flip"],-r[c]["flip"]
        if hi-lo>1e-6: g+=v-lo; h+=hi-lo
    return g/h*100 if h else float("nan")

L=[r"\begin{table}[t]", r"\centering", r"\small",
   r"\caption{\textbf{Among allocation criteria that need only forward passes, gap sensitivity "
   r"is first under every rounding method.} Fraction of the W3$\to$W4 reduction in top-1 change "
   r"recovered at a $3.5$-bit average, pooled over models and corpora. Every criterion here is "
   r"computable inside a serving deployment: no backward pass, no autograd, no calibration loss "
   r"to choose. Reconstruction error, the objective existing allocators minimise, is "
   r"indistinguishable from a random split at the same budget. The two criteria implied by the "
   r"rounding algorithms themselves fall \emph{below} random, and are worst under the very "
   r"quantizer they derive from: allocating bits by the objective the rounder already optimises "
   r"is redundant.}",
   r"\label{tab:alloc-nograd}",
   r"\begin{tabular}{l" + "r"*len(ROUND) + "}", r"\toprule",
   r"Allocation criterion & " + " & ".join(n for _,n in ROUND) + r" \\", r"\midrule"]
for c,name in CRIT:
    cells=[]
    for m,_ in ROUND:
        v=capture(by[m], c)
        s = "--" if v!=v else f"{v:.1f}\\%"
        cells.append(f"\\textbf{{{s}}}" if c=="gap" else s)
    L.append(f"{name} & " + " & ".join(cells) + r" \\")
    if c=="gap": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(OUT,"w").write("\n".join(L)+"\n")
print(f"wrote {OUT}")
for m,n in ROUND:
    print(f"  {n:<8} n={len(by[m])}  gap={capture(by[m],'gap'):.1f}%")
