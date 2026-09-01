#!/usr/bin/env python3
"""Do the criteria disagree about WHICH layers matter, or only about how to rank them?"""
import json, glob, os, statistics
SRC = "/data02/users/lzhou/qat-transfer/sens_only"
OUT = os.path.expanduser("~/qat-transfer/paper/tables/new_alloc_agreement.tex")
NAMES = [("mse","Reconstruction error"), ("relerr","Relative error (SQNR)"),
         ("position","Depth heuristic"), ("random0","Random split"),
         ("actnorm","Activation norm"), ("hess","Activation-weighted error"),
         ("awqsal","Activation salience")]
cells=[json.load(open(f)).get("allocs",{}) for f in
       glob.glob(os.path.join(SRC,"**","gap_allocation.json"),recursive=True)]
cells=[a for a in cells if a and "gap" in a]
def agree(a,b):
    ks=set(a)&set(b); return sum(a[k]==b[k] for k in ks)/len(ks)*100
rand=statistics.mean(agree(x["gap"],x["random0"]) for x in cells)
L=[r"\begin{table}[t]", r"\centering", r"\small",
   r"\caption{\textbf{The criteria that do worst disagree with gap sensitivity about \emph{which} "
   r"layers matter, not merely about how to rank them.} Percentage of layers assigned the same "
   r"width as the gap-driven allocation at a $3.5$-bit average, mean over "
   f"{len(cells)}" r" model and corpus pairs. A random split at the same budget agrees "
   f"{rand:.0f}" r"\% of the time, which is the level to read against: allocations constrained to "
   r"the same average necessarily overlap. Reconstruction error and the weight-only and structural "
   r"heuristics sit at that level, so they carry no information about layer identity. The two "
   r"criteria derived from the rounding algorithms sit \emph{below} it, agreeing with gap less "
   r"often than chance, which is why they also score below a random split in "
   r"Table~\ref{tab:alloc-nograd}: they are not noisy about which layers decide the ranking, they "
   r"are systematically wrong about it.}",
   r"\label{tab:alloc-agreement}", r"\begin{tabular}{lr}", r"\toprule",
   r"Allocation criterion & Agreement with gap \\", r"\midrule"]
rows=sorted(((statistics.mean(agree(x["gap"],x[c]) for x in cells),c,n) for c,n in NAMES),reverse=True)
for mu,c,n in rows:
    mark = r"\;$\dagger$" if c=="random0" else ""
    L.append(f"{n}{mark} & {mu:.1f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}",
      r"\\[2pt]\footnotesize $\dagger$ chance level at this budget.", r"\end{table}"]
open(OUT,"w").write("\n".join(L)+"\n")
print(f"wrote {OUT} ({len(cells)} cells, chance {rand:.1f}%)")
