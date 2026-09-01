#!/usr/bin/env python3
"""Compare the disjoint-calibration rerun against the superseded overlapping-calibration
results, over whatever has landed so far. CPU-only, safe to run at any time."""
import json, glob, os, re, collections
NEW = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/gap_allocation")
OLD = "/data02/users/lzhou/qat-transfer/superseded_overlapping_calib_20260830"
POLS = ["floor", "ceil", "random0", "mse", "hawq", "gap"]

def load(base):
    out = {}
    for f in glob.glob(os.path.join(base, "**", "gap_allocation.json"), recursive=True):
        try: d = json.load(open(f))
        except Exception: continue
        key = tuple(os.path.relpath(f, base).split(os.sep)[:-1])
        out[key] = d
    return out

new, old = load(NEW), load(OLD)
# only configs the rerun has actually refreshed (new file carries the split fields)
done = {k: v for k, v in new.items() if "n_eval_queries" in v}
print(f"refreshed configs: {len(done)} / {len(old)}")
if not done: raise SystemExit(0)

delta = collections.defaultdict(list)
wins = collections.Counter(); pairs = 0
for k, dn in done.items():
    do = old.get(k)
    for p in POLS:
        if p in dn["results"] and do and p in do["results"]:
            delta[p].append(dn["results"][p]["flip"] - do["results"][p]["flip"])
    r = dn["results"]
    if "gap" in r:
        pairs += 1
        for p in ("mse", "random0", "hawq", "floor"):
            if p in r and r["gap"]["flip"] < r[p]["flip"]: wins[p] += 1
            elif p in r: wins[p + "_LOSS"] += 1

print("\nflip-rate shift, new (held-out) minus old (overlapping), in points:")
for p in POLS:
    if delta[p]:
        v = sorted(delta[p]); n = len(v)
        print(f"  {p:<9} n={n:<4} median {v[n//2]*100:+.2f}   mean {sum(v)/n*100:+.2f}")

print(f"\ngap beats baseline, on the refreshed configs (n={pairs}):")
for p in ("mse", "random0", "hawq", "floor"):
    w, l = wins[p], wins[p + "_LOSS"]
    if w + l: print(f"  vs {p:<9} {w}/{w+l}  ({w/(w+l):.0%})")

margins = []
for k, dn in done.items():
    r = dn["results"]
    if "gap" in r and "mse" in r: margins.append(r["mse"]["flip"] - r["gap"]["flip"])
if margins:
    m = sorted(margins)
    print(f"\ngap-vs-MSE margin (points): median {m[len(m)//2]*100:+.2f}  "
          f"min {m[0]*100:+.2f}  max {m[-1]*100:+.2f}")
