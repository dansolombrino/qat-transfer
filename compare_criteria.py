#!/usr/bin/env python3
"""Rank every allocation criterion against every rounding method. CPU-only, run any time."""
import json, glob, os, collections, math
B = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/gap_allocation")
NAMES = {"gap":"Gap (ours)","hawq":"HAWQ Fisher","hess":"GPTQ-Hessian","awqsal":"AWQ-salience",
         "mse":"Weight MSE","relerr":"SQNR","position":"Depth rule","actnorm":"Activation norm",
         "fisheronly":"Fisher (no noise)","random0":"Random","floor":"Uniform W3","ceil":"Uniform W4"}
rows=[]
for f in glob.glob(os.path.join(B,"**","gap_allocation.json"), recursive=True):
    j=json.load(open(f))
    if "n_eval_queries" not in j or "hess" not in j.get("results",{}): continue
    rows.append(j)
if not rows:
    print("no full-criterion runs yet"); raise SystemExit
print(f"configs with the full criterion set: {len(rows)}")
bym=collections.defaultdict(list)
for j in rows: bym[j.get("method","rtn")].append(j)

for meth in sorted(bym):
    js=bym[meth]
    crit=[c for c in NAMES if c not in ("floor","ceil") and all(c in j["results"] for j in js)]
    # mean rank across configs (1 = best), and pooled capture of the extra bit
    ranks=collections.defaultdict(list); cap={}
    for j in js:
        r=j["results"]
        order=sorted(crit, key=lambda c: r[c]["flip"])
        for i,c in enumerate(order): ranks[c].append(i+1)
    for c in crit:
        g=h=0.0
        for j in js:
            r=j["results"]
            lo,hi,v=-r["floor"]["flip"],-r["ceil"]["flip"],-r[c]["flip"]
            if hi-lo>1e-6: g+=v-lo; h+=hi-lo
        cap[c]=g/h if h else float("nan")
    print(f"\n--- rounding = {meth.upper()}  ({len(js)} configs) ---")
    print(f"  {'criterion':<20} {'mean rank':>9} {'capture':>8}   {'beats Gap':>9}")
    gapflip={id(j):j["results"]["gap"]["flip"] for j in js}
    for c in sorted(crit, key=lambda c: sum(ranks[c])/len(ranks[c])):
        mr=sum(ranks[c])/len(ranks[c])
        bw=sum(1 for j in js if j["results"][c]["flip"] < gapflip[id(j)])
        tag="" if c!="gap" else "  <-- ours"
        print(f"  {NAMES[c]:<20} {mr:>9.2f} {cap[c]*100:>7.1f}%   {bw:>4}/{len(js)}{tag}")
