"""Page-1 figure: the hook (CLIP contrast) and the method (capture bars) in one glance."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import use_paper_style
use_paper_style(base=12)
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 2.5), gridspec_kw={"wspace": 0.30})

# Panel A: what the benchmark sees vs what breaks (CLIP ViT-L/14, W4, Flickr30k)
vals = [1.3, 24.6, 10.9]
labels = ["Recall@1\nlost", "Top-1 results\nchanged", "Queries losing\ntheir gold image"]
colors = ["#7f7f7f", "#1f77b4", "#d62728"]
bars = ax1.bar(range(3), vals, color=colors, width=0.62)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v}\\%", ha="center", fontsize=12)
ax1.set_xticks(range(3)); ax1.set_xticklabels(labels, fontsize=10.5)
ax1.set_ylabel(r"\% of queries", fontsize=13)
ax1.set_ylim(0, 29)
ax1.set_title("What the benchmark sees vs.\\ what breaks\n(CLIP ViT-L/14, 4-bit, Flickr30k)", fontsize=12.5)
ax1.spines[["top", "right"]].set_visible(False)

# Panel B: allocation capture (RTN @ 3.5 bits, pooled, Table alloc-main)
pols = ["Random", "MSE", "HAWQ", "Gap\n(ours)"]
caps = [50.3, 49.3, 70.6, 74.8]
cols = ["#bbbbbb", "#bbbbbb", "#ff7f0e", "#d62728"]
bars = ax2.bar(range(4), caps, color=cols, width=0.62)
for b, v in zip(bars, caps):
    ax2.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v}\\%", ha="center", fontsize=12)
ax2.set_xticks(range(4)); ax2.set_xticklabels(pols, fontsize=11.5)
ax2.set_ylabel("\\% of an extra bit's\nbenefit captured", fontsize=13)
ax2.set_ylim(0, 88)
ax2.set_title("Where to spend bits on retrieval\n(6 models $\\times$ 5 corpora, 3.5-bit budget)", fontsize=12.5)
ax2.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = os.path.expanduser("~/qat-transfer/paper/figs/fig_hook.pdf")
fig.savefig(out, bbox_inches="tight")
print("saved", out)
