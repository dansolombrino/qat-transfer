"""Is the gap-driven allocation itself stable, or is it fitting calibration noise?

Compares the per-layer bit assignment chosen under different calibration seeds for the same
(model, corpus, budget). If a different 128-query sample picks a different set of layers to
protect, the 'sensitivity' is noise and the capture result was luck on one draw.
"""
import json, glob, os
from collections import defaultdict
import numpy as np

BASE = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/"
                          "sentence_transformers/gap_allocation")

runs = defaultdict(dict)          # (model, ds, avg) -> seed -> alloc
for f in glob.glob(os.path.join(BASE, "**", "gap_allocation.json"), recursive=True):
    j = json.load(open(f))
    if j.get("avg_bits_target") is None or "group_sizes" in j:
        continue
    if j.get("method", "rtn") != "rtn":
        continue
    key = (j["model_name"].split("/")[-1], j["dataset_name"], j["avg_bits_target"])
    runs[key][int(j["seed"])] = j["alloc_gap"]

agree, jac, rows = [], [], []
for key, by_seed in sorted(runs.items()):
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        continue
    for i in range(len(seeds)):
        for k in range(i + 1, len(seeds)):
            a, b = by_seed[seeds[i]], by_seed[seeds[k]]
            common = sorted(set(a) & set(b))
            if not common:
                continue
            same = np.mean([a[l] == b[l] for l in common])
            hi_a = {l for l in common if a[l] == max(a.values())}
            hi_b = {l for l in common if b[l] == max(b.values())}
            j_ = len(hi_a & hi_b) / max(len(hi_a | hi_b), 1)
            agree.append(same); jac.append(j_)
            rows.append((key, seeds[i], seeds[k], same, j_, len(common)))

if not rows:
    print("no multi-seed pairs yet")
else:
    print(f"{'model':<24} {'corpus':<11} {'seeds':<10} {'layer agree':>12} {'top-bit Jaccard':>16}")
    for (m, d, _), s1, s2, same, j_, n in rows:
        print(f"{m:<24} {d:<11} {s1}/{s2:<5} {same:>11.1%} {j_:>15.1%}")
    print(f"\npairs={len(rows)}  mean layer agreement={np.mean(agree):.1%}  "
          f"mean top-bit Jaccard={np.mean(jac):.1%}")
    print(f"chance agreement for a 2-way split at this budget ~ 50%")
