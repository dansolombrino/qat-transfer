"""The regime-switch figure: which allocation criterion wins, as a function of damage.

X axis: the run's W3->W4 floor flip rate (how far below the certificate boundary the setting
sits -- the damage the certificate quantity predicts). Y axis: gap capture minus MSE capture on
flip. Prediction: positive at high damage (retrieval regime), negative near zero damage
(classification regime), crossing in between.
"""
import json, glob, os
import numpy as np

ROOTS = [
    (os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/"
                        "gap_allocation"), "gap_allocation.json", "text-retrieval"),
    (os.path.expanduser("~/qat-transfer/storage/checkpoints/vision/ilharco_hf_clip/"
                        "gap_allocation_clip"), "gap_allocation_clip.json", "clip-retrieval"),
    (os.path.expanduser("~/qat-transfer/storage/checkpoints/vision/ilharco_timm_supervised/"
                        "gap_allocation_clsf"), "gap_allocation_clsf.json", "vision-clsf"),
    (os.path.expanduser("~/qat-transfer/storage/checkpoints/text/"
                        "ilharco_automodelforsequenceclassification/gap_allocation_clsf"),
     "gap_allocation_clsf.json", "text-clsf"),
]


def collect():
    pts = []
    for root, fname, fam in ROOTS:
        for f in glob.glob(os.path.join(root, "**", fname), recursive=True):
            j = json.load(open(f))
            if j.get("avg_bits_target") is None or "group_sizes" in j:
                continue
            r = j["results"]
            lo, hi = r["floor"]["flip"], r["ceil"]["flip"]
            hr = lo - hi
            if hr <= 1e-6:
                continue
            cap = lambda v: (lo - v) / hr
            rnd = np.mean([cap(r[k]["flip"]) for k in r if k.startswith("random")])
            pts.append(dict(family=fam, damage=lo, headroom=hr,
                            d_gap_mse=cap(r["gap"]["flip"]) - cap(r["mse"]["flip"]),
                            d_gap_rnd=cap(r["gap"]["flip"]) - rnd,
                            model=j["model_name"].split("/")[-1],
                            ds=j.get("dataset_name", "?")))
    return pts


def main():
    pts = collect()
    print(f"{len(pts)} runs across {len(set(p['family'] for p in pts))} families\n")
    bins = [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .45), (.45, .6), (.6, 1.0)]
    print(f"{'damage bin':<12} {'n':>4} {'gap-MSE':>9} {'gap-rand':>9}  families")
    for a, b in bins:
        sel = [p for p in pts if a <= p["damage"] < b]
        if not sel:
            continue
        gm = np.mean([p["d_gap_mse"] for p in sel])
        gr = np.mean([p["d_gap_rnd"] for p in sel])
        fams = ",".join(sorted({p["family"] for p in sel}))
        print(f"{a:.2f}-{b:.2f}   {len(sel):>4} {gm:>+8.1%} {gr:>+8.1%}  {fams}")
    lo_d = [p for p in pts if p["damage"] < 0.10]
    hi_d = [p for p in pts if p["damage"] >= 0.30]
    print(f"\nlow damage (<10%):  gap-MSE {np.mean([p['d_gap_mse'] for p in lo_d]):+.1%} (n={len(lo_d)})")
    print(f"high damage (>=30%): gap-MSE {np.mean([p['d_gap_mse'] for p in hi_d]):+.1%} (n={len(hi_d)})")


if __name__ == "__main__":
    main()
