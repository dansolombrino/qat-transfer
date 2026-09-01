"""Near-threshold mass: what fraction of inputs can the gap-sensitivity criterion even see?

Gap sensitivity aggregates |delta gap| over calibration inputs. Only inputs near the flip
threshold (fp_gap_2 < 2*eps) carry information about *failures*; inputs far above it move the
median without ever being at risk. Hypothesis (F60 follow-up): the criterion wins exactly where
the near-threshold mass is large. Computed from the existing W3/group_128 gap-profile parquets,
per family — no new GPU work.
"""
import glob, os, re
import numpy as np
import pandas as pd

SPECS = [
    ("text-retrieval", "storage/checkpoints/text/sentence_transformers/retrieval_gap_profile/**/*.parquet"),
    ("clip-retrieval", "storage/checkpoints/vision/clip_crossmodal/**/*.parquet"),
    ("vision-retrieval", "storage/checkpoints/vision/ilharco_timm_supervised/retrieval_gap_profile/**/*.parquet"),
    ("vision-clsf", "storage/checkpoints/vision/ilharco_timm_supervised/gap_profile/**/*.parquet"),
    ("text-clsf", "storage/checkpoints/text/ilharco_automodelforsequenceclassification/gap_profile/**/*.parquet"),
]
W3 = re.compile(r"bits=3[_/].*group_128|bits=3_gran=group_128")


def main():
    rows = []
    for fam, pat in SPECS:
        for f in glob.glob(os.path.expanduser(os.path.join("~/qat-transfer", pat)), recursive=True):
            if not W3.search(f):
                continue
            try:
                df = pd.read_parquet(f, columns=["fp_gap_2", "eps_linf"])
            except Exception:
                continue
            if not len(df):
                continue
            frac = float((df["fp_gap_2"] < 2 * df["eps_linf"]).mean())
            rows.append(dict(family=fam, frac=frac, n=len(df), f=f))
    if not rows:
        print("nothing matched")
        return
    d = pd.DataFrame(rows)
    print(f"{'family':<18} {'files':>6} {'inputs':>9} {'near-threshold mass':>20}")
    for fam, g in d.groupby("family"):
        tot = g["n"].sum()
        w = (g["frac"] * g["n"]).sum() / tot
        print(f"{fam:<18} {len(g):>6} {tot:>9} {w:>19.1%}")
    print("\n(mass = fraction of inputs with fp_gap_2 < 2*eps_linf at W3/group_128 —")
    print(" the inputs whose fate an allocation decision can actually change)")


if __name__ == "__main__":
    main()
