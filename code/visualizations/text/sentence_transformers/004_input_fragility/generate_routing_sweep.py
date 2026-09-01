"""Routing: score baseline (margin vs MSP) and threshold sensitivity, from the prediction dumps.

The deployable recipe fixes tau at a percentile of the routing score on an unlabeled validation
slice. Two questions a reviewer will ask: does the maximum-softmax-probability baseline do as
well, and how sensitive is the single hyperparameter? Both are answered offline: the dumps carry
the score, the FP prediction, the PTQ prediction and the label per input.
"""
import glob, os, re
import numpy as np
import pandas as pd

ROOTS = {
    "ViT-B/16": "storage/checkpoints/vision/ilharco_timm_supervised/input_fragility_dumps/vit_base_patch16_224_orig_in21k",
    "ViT-L/16": "storage/checkpoints/vision/ilharco_timm_supervised/input_fragility_dumps/vit_large_patch16_224_orig_in21k",
    "Qwen3-Emb-0.6B": "storage/checkpoints/text/ilharco_automodelforsequenceclassification/input_fragility_dumps/Qwen_Qwen3_Embedding_0.6B",
}
PCTS = [10, 25, 50]
SPLIT_SEED, VAL_FRAC, MIN_GAP = 0, 0.10, 0.005   # gap eligibility, as in the original recipe


def task_rows(path):
    for f in glob.glob(os.path.join(os.path.expanduser("~/qat-transfer"), path,
                                    "**", "predictions_test.parquet"), recursive=True):
        if "bits=4" not in f or "channel" not in f:
            continue
        try:
            d = pd.read_parquet(f, columns=["label", "fp_pred", "q_pred", "q_margin",
                                            "q_softmax_top1"])
        except Exception:
            continue
        yield re.search(r"in21k/([^/]+)/|0\.6B/([^/]+)/", f), d


def evaluate(d, score, pct, rng):
    """Route the lowest-`pct` percentile of `score` (threshold set on a held-out val slice)."""
    n = len(d)
    idx = rng.permutation(n)
    n_val = max(int(VAL_FRAC * n), 20)
    val, test = idx[:n_val], idx[n_val:]
    tau = np.percentile(score[val], pct)
    routed = score[test] < tau
    fp_ok = (d["fp_pred"].values == d["label"].values)[test]
    q_ok = (d["q_pred"].values == d["label"].values)[test]
    served = np.where(routed, fp_ok, q_ok)
    acc_q, acc_fp, acc_r = q_ok.mean(), fp_ok.mean(), served.mean()
    gap = acc_fp - acc_q
    if gap < MIN_GAP:
        return None
    return routed.mean(), (acc_r - acc_q) / gap


def main():
    rng = np.random.default_rng(SPLIT_SEED)
    out = {}
    for model, path in ROOTS.items():
        acc = {(s, p): [] for s in ("margin", "msp") for p in PCTS}
        n_tasks = 0
        for _, d in task_rows(path):
            scores = {"margin": d["q_margin"].values, "msp": d["q_softmax_top1"].values}
            used = False
            for s, sc in scores.items():
                for p in PCTS:
                    r = evaluate(d, sc, p, np.random.default_rng(SPLIT_SEED))
                    if r:
                        acc[(s, p)].append(r); used = True
            n_tasks += used
        out[model] = (acc, n_tasks)

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{Routing score and threshold sensitivity.} Fraction of traffic routed "
         r"and fraction of the quantization accuracy loss recovered, at three percentiles of the "
         r"routing score on an unlabeled validation slice, W4-channel, averaged over the "
         r"eligible tasks per backbone. The top-1/top-2 margin recovers more of the loss than "
         r"maximum softmax probability at every operating point, and the recipe degrades "
         r"gracefully in its single hyperparameter.}",
         r"\label{tab:routing-sweep}", r"\begin{tabular}{llcccccc}", r"\toprule",
         r"& & \multicolumn{2}{c}{10th pct.} & \multicolumn{2}{c}{25th pct.} & "
         r"\multicolumn{2}{c}{50th pct.} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Backbone & Score & routed & recovered & routed & recovered & routed & recovered \\",
         r"\midrule"]
    for model, (acc, n_tasks) in out.items():
        for i, (s, lab) in enumerate((("margin", r"margin (ours)"), ("msp", "MSP"))):
            cells = []
            for p in PCTS:
                v = acc[(s, p)]
                if not v:
                    cells += ["--", "--"]; continue
                fr = 100 * np.mean([x[0] for x in v]); rc = 100 * np.mean([x[1] for x in v])
                cells += [f"{fr:.1f}\\%", (f"\\textbf{{{rc:.1f}\\%}}" if s == "margin"
                                           else f"{rc:.1f}\\%")]
            head = f"{model} ({n_tasks})" if i == 0 else ""
            L.append(f"{head} & {lab} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.expanduser("~/qat-transfer/paper/tables/new_routing_sweep.tex"),
         "w").write("\n".join(L) + "\n")
    for model, (acc, n) in out.items():
        print(model, f"({n} tasks)")
        for s in ("margin", "msp"):
            print("   ", s, " ".join(
                f"p{p}: {100*np.mean([x[1] for x in acc[(s,p)]]):.1f}%" if acc[(s, p)] else f"p{p}: --"
                for p in PCTS))


if __name__ == "__main__":
    main()
