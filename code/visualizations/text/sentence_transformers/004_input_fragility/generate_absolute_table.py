"""Absolute metrics for the allocation comparison (reviewer W4.1): what a 3.5-bit model
actually scores, not only the captured fraction of headroom. Pooled over the 30-config
headline grid (6 models x 5 corpora, RTN, seed 2038) where all policies incl. HAWQ exist."""
import json, glob, os
import numpy as np

B = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/gap_allocation")
ROWS = [("floor", "All W3"), ("random", "Random split"), ("mse", "MSE-driven"), ("gap", "Gap-driven"), ("ceil", "All W4")]

def main():
    acc = {k: {"flip": [], "recall1": [], "gold_lost": []} for k, _ in ROWS}
    n = 0
    for f in glob.glob(B + "/**/method=rtn/avg=3.5/seed=2038/gap_allocation.json", recursive=True):
        j = json.load(open(f)); r = j["results"]
        if "hawq" not in r:
            continue
        n += 1
        for k, _ in ROWS:
            if k == "random":
                for m in ("flip", "recall1", "gold_lost"):
                    acc[k][m].append(np.mean([r[p][m] for p in r if p.startswith("random")]))
            else:
                for m in ("flip", "recall1", "gold_lost"):
                    acc[k][m].append(r[k][m])
    L=[r"\begin{table}[t]",r"\centering",r"\small",
       r"\caption{\textbf{Absolute retrieval metrics under each allocation.} Mean over the "
       r"24-configuration headline grid (six models $\times$ four corpora, RTN, average 3.5 "
       r"bits for the middle rows). Flip is disagreement with the full-precision top-1; "
       r"Recall@1 and gold retention are against relevance judgements. The capture ratios of "
       r"Table~\ref{tab:alloc-nograd} are computed from these absolute values.}",
       r"\label{tab:alloc-absolute}",r"\begin{tabular}{lccc}",r"\toprule",
       r"Configuration & Flip $\downarrow$ & R@1 $\uparrow$ & Gold lost $\downarrow$ \\",r"\midrule"]
    for k, lab in ROWS:
        f_, r_, g_ = (100*np.mean(acc[k][m]) for m in ("flip","recall1","gold_lost"))
        row = f"{lab} & {f_:.1f}\\% & {r_:.1f}\\% & {g_:.1f}\\%"
        if k == "gap":
            row = f"\\textbf{{{lab}}} & \\textbf{{{f_:.1f}\\%}} & \\textbf{{{r_:.1f}\\%}} & \\textbf{{{g_:.1f}\\%}}"
        L.append(row + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.expanduser("~/qat-transfer/paper/tables/new_alloc_absolute.tex"), "w").write("\n".join(L)+"\n")
    print(f"wrote new_alloc_absolute.tex over n={n} configs")

if __name__ == "__main__":
    main()
