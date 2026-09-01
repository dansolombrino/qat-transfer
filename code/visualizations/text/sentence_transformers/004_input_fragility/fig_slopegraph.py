"""Slopegraph: one encoder, two readouts (the same-checkpoint contrast)."""
import glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import use_paper_style
use_paper_style(base=13)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.expanduser("~/qat-transfer/paper/figs")
os.makedirs(OUT, exist_ok=True)
C = dict(clsf="#d62728", retr="#1f77b4", gap="#d62728", hawq="#ff7f0e", mse="#8c8c8c",
         rnd="#bbbbbb", ok="#2ca02c")


# ---------------------------------------------------------------- 1. slopegraph
def fig_slope():
    B = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/"
                           "same_checkpoint_contrast")
    rows = [json.load(open(f)) for f in
            glob.glob(os.path.join(B, "**", "contrast.json"), recursive=True)]
    rows.sort(key=lambda j: (j["dataset_name"], -j["bits"]))
    DS = sorted({j["dataset_name"] for j in rows})
    col = dict(zip(DS, ["#1f77b4", "#d62728", "#2ca02c"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 2.6), gridspec_kw={"wspace": 0.22})
    for ax, keys, ylab, ttl, logy in (
            (a1, ("clsf_flip", "retr_flip"), r"Top-1 change rate (\%)",
             "The same vectors, read two ways", False),
            (a2, ("clsf_sep", "retr_sep"), r"Median separation $g_2/2\varepsilon$",
             "and the quantity that explains it", True)):
        for j in rows:
            y0, y1 = j[keys[0]], j[keys[1]]
            if not logy:
                y0, y1 = 100 * y0, 100 * y1
            ax.plot([0, 1], [y0, y1], color=col[j["dataset_name"]], lw=2.2,
                    ls="-" if j["bits"] == 4 else "--", alpha=0.85, marker="o", ms=6, zorder=2)
        ax.set_xlim(-0.12, 1.12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Classification", "Retrieval"], fontsize=12)
        ax.tick_params(axis="x", pad=4)
        ax.set_ylabel(ylab, fontsize=13)
        ax.set_title(ttl, fontsize=13)
        ax.spines[["top", "right"]].set_visible(False)
        if logy:
            ax.set_yscale("log")
            ax.axhline(1.0, color="0.45", ls=":", lw=1.3)
            ax.annotate("certifiable", (1.10, 1.06), fontsize=10.5, color="0.45",
                        ha="right", va="bottom")
    handles = [Line2D([], [], color=col[d], lw=2.2, label=d) for d in DS]
    handles += [Line2D([], [], color="0.35", lw=2.2, ls="-", label="W4"),
                Line2D([], [], color="0.35", lw=2.2, ls="--", label="W3")]
    fig.legend(handles=handles, frameon=False, fontsize=11.5, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.02))
    fig.savefig(f"{OUT}/fig_slopegraph.pdf", bbox_inches="tight")
    print("p1 slopegraph")



fig_slope()
