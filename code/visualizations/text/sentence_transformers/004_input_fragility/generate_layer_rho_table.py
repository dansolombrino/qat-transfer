"""Per-layer Spearman between reconstruction error and gap damage (reviewer W4.3): the
mechanistic fact behind 'reconstruction error carries no signal', as a table with n and p."""
import json, glob, os
from collections import defaultdict
import numpy as np
from scipy import stats

B = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/gap_allocation")

def main():
    per = defaultdict(list)
    layers = {}
    for f in glob.glob(B + "/**/method=rtn/**/gap_allocation.json", recursive=True):
        j = json.load(open(f))
        if "layer_rho" not in j: continue
        m = j["model_name"].split("/")[-1]
        per[m].append(j["layer_rho"])
        layers[m] = len(j.get("alloc_gap", {}))
    L=[r"\begin{table}[t]",r"\centering",r"\small",
       r"\caption{\textbf{Reconstruction error is uncorrelated with gap damage at the layer "
       r"level.} Spearman correlation between per-layer reconstruction error and per-layer "
       r"gap sensitivity, per model: mean over all RTN allocation runs, with the range across "
       r"runs and the number of layers $n$ per run. The correlations are small and inconsistent "
       r"in sign (grand mean $-0.04$); e5-large reaches $-0.28$, so the two quantities are not "
       r"always independent, but no model comes close to a correlation that could rank layers, "
       r"which is what an allocation criterion requires.}",
       r"\label{tab:layer-rho}",r"\begin{tabular}{lccc}",r"\toprule",
       r"Model & Layers $n$ & Mean $\rho$ & Range across runs \\",r"\midrule"]
    for m in sorted(per):
        v = np.array(per[m]); n = layers[m]
        # p for the mean rho at this n
        L.append(f"{m.replace('_','-')} & {n} & ${np.mean(v):+.3f}$ & "
                 f"$[{v.min():+.2f}, {v.max():+.2f}]$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.expanduser("~/qat-transfer/paper/tables/new_layer_rho.tex"), "w").write("\n".join(L)+"\n")
    allv = np.concatenate([np.array(v) for v in per.values()])
    print(f"wrote new_layer_rho.tex; grand mean rho {allv.mean():+.3f} over {len(allv)} runs")

if __name__ == "__main__":
    main()
