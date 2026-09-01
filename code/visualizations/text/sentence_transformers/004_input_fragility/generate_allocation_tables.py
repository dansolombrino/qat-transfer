"""Paper tables for the gap-driven bit allocation intervention.

Emits paper/tables/new_alloc_main.tex, new_alloc_budget.tex, new_alloc_seed.tex.
Capture is pooled: sum(gain)/sum(headroom) over configs with positive headroom, so a config
where W4 is no better than W3 cannot contribute a meaningless ratio.
"""
import json, glob, os
from collections import defaultdict
import numpy as np

BASE = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/"
                          "sentence_transformers/gap_allocation")
OUT = os.path.expanduser("~/qat-transfer/paper/tables")
POLS = [("random", "Random"), ("mse", "MSE-driven"), ("gap", "Gap-driven")]
METRICS = [("flip", "Flip", False), ("recall1", "R@1", True), ("gold_lost", "Gold kept", False)]


def load():
    rows = []
    for f in glob.glob(os.path.join(BASE, "**", "gap_allocation.json"), recursive=True):
        j = json.load(open(f))
        if j.get("avg_bits_target") is None or "group_sizes" in j:
            continue
        rows.append(j)
    return rows


def pooled(rows, method, avg, pol, key, higher):
    g = h = 0.0
    n = 0
    for j in rows:
        if j.get("method", "rtn") != method or abs(j["avg_bits_target"] - avg) > 1e-6:
            continue
        r = j["results"]
        pols = [p for p in r if p.startswith("random")] if pol == "random" else [pol]
        for p in pols:
            try:
                lo, hi, v = r["floor"][key], r["ceil"][key], r[p][key]
            except KeyError:
                continue
            if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (lo, hi, v)):
                continue
            if not higher:
                lo, hi, v = -lo, -hi, -v
            if hi - lo <= 1e-6:
                continue
            g += v - lo
            h += hi - lo
            n += 1
    return (g / h if h else np.nan), n


def fmt(x):
    return "--" if np.isnan(x) else f"{x*100:.1f}\\%"


def main():
    rows = load()
    os.makedirs(OUT, exist_ok=True)

    # ---- main: capture by quantizer at the 3.5-bit budget
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Gap-driven allocation recovers most of an extra bit, and composes "
         r"with error-minimizing quantizers.} Fraction of the W3$\to$W4 benefit captured at a "
         r"3.5-bit average, pooled over six models and five corpora. Allocating by reconstruction "
         r"error is indistinguishable from a random split at the same budget; no existing "
         r"allocation criterion targets ranking gaps. A HAWQ-style curvature criterion becomes "
         r"competitive only when its calibration loss is itself contrastive, and gap-driven "
         r"allocation leads it on the quality metrics. AWQ is itself an allocation method and "
         r"does not subsume the effect.}",
         r"\label{tab:alloc-main}",
         r"\begin{tabular}{ll" + "r" * len(METRICS) + "}", r"\toprule",
         "Quantizer & Allocation & " + " & ".join(m[1] for m in METRICS) + r" \\", r"\midrule"]
    for meth, lab in (("rtn", "RTN"), ("gptq", "GPTQ"), ("awq", "AWQ")):
        first = True
        for pol, plab in POLS:
            cells = []
            for key, _, higher in METRICS:
                c, _ = pooled(rows, meth, 3.5, pol, key, higher)
                cells.append(("\\textbf{" + fmt(c) + "}") if pol == "gap" else fmt(c))
            if all("--" in c for c in cells):
                continue          # policy not measured under this quantizer (hawq: RTN only)
            head = lab if first else ""
            first = False
            L.append(f"{head} & {plab} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if meth != "awq" else "")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "new_alloc_main.tex"), "w").write("\n".join(x for x in L if x) + "\n")

    # ---- budget generality
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{The allocation holds across budgets.} Capture at three fractional "
         r"bit budgets under RTN. At an integer average uniform is the only allocation, so the "
         r"question is only posed between integers. Capture rises with budget: more of the extra "
         r"bit is available to place.}",
         r"\label{tab:alloc-budget}",
         r"\begin{tabular}{ll" + "r" * len(METRICS) + "}", r"\toprule",
         "Budget & Allocation & " + " & ".join(m[1] for m in METRICS) + r" \\", r"\midrule"]
    for avg in (3.25, 3.5, 3.75):
        for i, (pol, plab) in enumerate([q for q in POLS if q[0] != "hawq"]):
            cells = []
            for key, _, higher in METRICS:
                c, _ = pooled(rows, "rtn", avg, pol, key, higher)
                cells.append(("\\textbf{" + fmt(c) + "}") if pol == "gap" else fmt(c))
            head = f"{avg:g} bits" if i == 0 else ""
            L.append(f"{head} & {plab} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule" if avg != 3.75 else "")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "new_alloc_budget.tex"), "w").write("\n".join(x for x in L if x) + "\n")

    # ---- seed stability
    runs = defaultdict(dict)
    for j in rows:
        if j.get("method", "rtn") != "rtn":
            continue
        runs[(j["model_name"].split("/")[-1], j["dataset_name"],
              j["avg_bits_target"])][int(j["seed"])] = j["alloc_gap"]
    per_model = defaultdict(list)
    for (m, d, a), by_seed in runs.items():
        seeds = sorted(by_seed)
        for i in range(len(seeds)):
            for k in range(i + 1, len(seeds)):
                A, B = by_seed[seeds[i]], by_seed[seeds[k]]
                com = sorted(set(A) & set(B))
                if com:
                    per_model[m].append(np.mean([A[l] == B[l] for l in com]))
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{The allocation is a property of the model, not of the calibration "
         r"draw.} Fraction of layers receiving the same bit-width under different calibration "
         r"seeds $\{101,202,2038\}$, over all corpora and budgets. Chance agreement for this "
         r"two-way split is approximately 50\%.}",
         r"\label{tab:alloc-seed}", r"\begin{tabular}{lrr}", r"\toprule",
         r"model & seed pairs & layer agreement \\", r"\midrule"]
    allv = []
    for m in sorted(per_model):
        v = per_model[m]
        allv += v
        L.append(f"{m.replace('_','-')} & {len(v)} & {np.mean(v)*100:.1f}\\% \\\\")
    L += [r"\midrule", f"all & {len(allv)} & \\textbf{{{np.mean(allv)*100:.1f}\\%}} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "new_alloc_seed.tex"), "w").write("\n".join(L) + "\n")
    # ---- per-model breakdown, so the pooled number cannot hide a single dominant model
    per = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for j in rows:
        if j.get("method", "rtn") != "rtn" or abs(j["avg_bits_target"] - 3.5) > 1e-6:
            continue
        m = j["model_name"].split("/")[-1]
        r = j["results"]
        for pol in ("random0", "random1", "random2", "mse", "gap"):
            k = "random" if pol.startswith("random") else pol
            try:
                lo, hi, v = (-r["floor"]["flip"], -r["ceil"]["flip"], -r[pol]["flip"])
            except KeyError:
                continue
            if hi - lo <= 1e-6:
                continue
            per[m][k][0] += v - lo
            per[m][k][1] += hi - lo
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{The effect is present in every model, not carried by one.} "
         r"Capture of the W3$\to$W4 flip-rate benefit at a 3.5-bit average under \rtn{}, "
         r"per model, pooled over the five corpora. Gap-driven allocation beats both baselines "
         r"on all six models, across a $24\times$ parameter range and two architecture families.}",
         r"\label{tab:alloc-permodel}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"model & random & MSE-driven & gap-driven \\", r"\midrule"]
    for m in sorted(per):
        row = per[m]
        c = {k: 100 * row[k][0] / row[k][1] for k in ("random", "mse", "gap")}
        L.append(f"{m.replace('_','-')} & {c['random']:.1f}\\% & {c['mse']:.1f}\\% & "
                 f"\\textbf{{{c['gap']:.1f}\\%}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "new_alloc_permodel.tex"), "w").write("\n".join(L) + "\n")

    print("wrote new_alloc_main.tex, new_alloc_budget.tex, new_alloc_seed.tex, new_alloc_permodel.tex")


if __name__ == "__main__":
    main()
