"""Aggregate the classification allocation runs (vision + text).

Same pooled estimator as the retrieval aggregator: sum(gain)/sum(headroom) over configs with
positive headroom, so a config where W4 is no better than W3 cannot contribute an unstable ratio.
Metrics differ because the task does: flip / accuracy / correct-answers-lost, rather than
flip / Recall@1 / gold-retention.
"""
import json, glob, os
from collections import defaultdict
import numpy as np

ROOTS = [
    (os.path.expanduser("~/qat-transfer/storage/checkpoints/vision/"
                        "ilharco_timm_supervised/gap_allocation_clsf"), "vision"),
    (os.path.expanduser("~/qat-transfer/storage/checkpoints/text/"
                        "ilharco_automodelforsequenceclassification/gap_allocation_clsf"), "text"),
]
POLS = [("random", "random"), ("mse", "MSE-driven"), ("gap", "gap-driven")]
METRICS = [("flip", "flip", False), ("acc", "accuracy", True), ("correct_lost", "kept correct", False)]


def load():
    rows = []
    for root, modality in ROOTS:
        for f in glob.glob(os.path.join(root, "**", "gap_allocation_clsf.json"), recursive=True):
            j = json.load(open(f))
            j["_modality"] = modality
            rows.append(j)
    return rows


def pooled(rows, pol, key, higher, modality=None, model=None):
    g = h = 0.0
    n = 0
    for j in rows:
        if modality and j["_modality"] != modality:
            continue
        if model and j["model_name"] != model:
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
    return "   n/a" if np.isnan(x) else f"{x:6.1%}"


def main():
    rows = load()
    if not rows:
        print("no classification results yet")
        return

    print(f"{'scope':<34} {'policy':<11} " + " ".join(f"{m[1]:>12}" for m in METRICS) + f" {'n':>5}")
    scopes = [("ALL classification", dict())]
    for mod in sorted({j["_modality"] for j in rows}):
        scopes.append((f"  {mod}", dict(modality=mod)))
        for mdl in sorted({j["model_name"] for j in rows if j["_modality"] == mod}):
            scopes.append((f"    {mdl.split('/')[-1]}", dict(modality=mod, model=mdl)))
    for label, sel in scopes:
        for pol, plab in POLS:
            cells, nn = [], 0
            for key, _, higher in METRICS:
                c, n = pooled(rows, pol, key, higher, **sel)
                cells.append(fmt(c))
                nn = max(nn, n)
            print(f"{label:<34} {plab:<11} " + " ".join(f"{c:>12}" for c in cells) + f" {nn:>5}")
        print()

    # How much damage was there to recover? This is the quantity that decides whether the
    # allocation has anything to do, and it is what separates retrieval from classification.
    fl = [j["results"]["floor"]["flip"] for j in rows]
    ce = [j["results"]["ceil"]["flip"] for j in rows]
    print(f"configs: {len(rows)}   W3 flip mean {np.mean(fl):.1%}   W4 flip mean {np.mean(ce):.1%}"
          f"   mean headroom {np.mean(np.array(fl)-np.array(ce)):.1%}")
    print(f"datasets: {len(sorted({(j['_modality'], j['dataset_name']) for j in rows}))}")


if __name__ == "__main__":
    main()
