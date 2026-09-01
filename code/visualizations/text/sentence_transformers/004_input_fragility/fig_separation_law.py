"""Figure: one quantity organises both task families.

Each point is a (model, task, bit-width, corpus-size) cell: median separation ratio
g_2 / 2*eps against the measured top-1 change rate. Rebuilt from the gap-profile parquets
(the original generator predates the server migration).
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import use_paper_style
use_paper_style(base=15)
import matplotlib.pyplot as plt
from scipy import stats

ROOTS = [
    ("classification", "vision", "storage/checkpoints/vision/ilharco_timm_supervised/gap_profile"),
    ("classification", "text", "storage/checkpoints/text/ilharco_automodelforsequenceclassification/gap_profile"),
    ("retrieval", "vision", "storage/checkpoints/vision/ilharco_timm_supervised/retrieval_gap_profile"),
    ("retrieval", "text", "storage/checkpoints/text/sentence_transformers/retrieval_gap_profile"),
]

def cells():
    rows = []
    for family, modality, root in ROOTS:
        for f in glob.glob(os.path.expanduser(f"~/qat-transfer/{root}/**/*.parquet"), recursive=True):
            bits = 3 if "bits=3" in f else (4 if "bits=4" in f else None)
            if bits is None:
                continue
            try:
                df = pd.read_parquet(f, columns=["fp_gap_2", "eps_linf", "fp_top1", "q_top1"])
            except Exception:
                continue
            if len(df) < 50:
                continue
            sep = float(np.median(df["fp_gap_2"] / np.maximum(2 * df["eps_linf"], 1e-12)))
            flip = float((df["fp_top1"] != df["q_top1"]).mean())
            if sep <= 0:
                continue
            rows.append(dict(family=family, modality=modality, bits=bits, sep=sep, flip=flip))
    return pd.DataFrame(rows)

d = cells()
print(f"{len(d)} cells: " + ", ".join(f"{k}={v}" for k, v in d.family.value_counts().items()))

fig, ax = plt.subplots(figsize=(9.0, 4.6))
STYLE = {("classification", "vision"): ("#d62728", "s", "Classification (vision)"),
         ("classification", "text"):   ("#ff7f0e", "^", "Classification (text)"),
         ("retrieval", "vision"):      ("#9467bd", "v", "Retrieval (vision)"),
         ("retrieval", "text"):        ("#1f77b4", "o", "Retrieval (text)")}
for key, g in d.groupby(["family", "modality"]):
    c, m, lab = STYLE[key]
    ax.scatter(g.sep, 100 * g.flip, s=30, alpha=0.55, c=c, marker=m,
               edgecolors="none", label=f"{lab} ($n$={len(g)})")

rho, p = stats.spearmanr(d.sep, d.flip)
ax.axvline(1.0, color="0.4", ls=":", lw=1.4)
# annotate in the empty upper-right region, clear of every point
ax.text(0.985, 0.93, "Certifiable regime\n" + r"($g_2 \geq 2\varepsilon$)",
        transform=ax.transAxes, fontsize=15, color="0.35", va="top", ha="right",
        linespacing=1.35)
ax.set_xscale("log")
ax.set_xlabel(r"Median separation ratio $g_2 / 2\varepsilon$", fontsize=16, labelpad=10)
ax.set_ylabel(r"Top-1 change rate (\%)", fontsize=16)
ax.set_title(f"One quantity organises both task families (Spearman ${rho:.2f}$)", fontsize=16)
ax.tick_params(labelsize=15)
ax.legend(frameon=False, fontsize=13.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.34), ncol=2, columnspacing=2.4, handletextpad=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.expanduser("~/qat-transfer/paper/figs/fig_separation_law.pdf")
fig.savefig(out)
print("saved", out, f"Spearman {rho:.3f} (p={p:.2g})")
