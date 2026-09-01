"""Regime figure: gap-capture minus MSE-capture vs damage, drawn PER FAMILY.

Pooling families manufactures a clean monotone trend (F59); the honest figure shows each
family's own trend, making both the cross-family pattern and the within-family exception
visible. Stage I points (fixed dataset, epsilon ladder) can be overlaid when they land.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/qat-transfer/code/analysis"))
from boundary_figure import collect
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import use_paper_style
use_paper_style(base=13)
import matplotlib.pyplot as plt

FAM = {
    "text-retrieval": dict(c="#1f77b4", m="o", lab="text retrieval (BEIR)"),
    "clip-retrieval": dict(c="#17becf", m="D", lab="CLIP text$\\to$image"),
    "vision-clsf":    dict(c="#d62728", m="s", lab="vision classification"),
    "text-clsf":      dict(c="#ff7f0e", m="^", lab="text classification"),
}

pts = collect()
# capture is a ratio over headroom; below ~5pp of headroom it is numerically meaningless and
# the collapsed-regime runs (2-bit / tensor) would fabricate a spurious band at y~0, x~1
pts = [p for p in pts if p["headroom"] >= 0.05]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
YLIM = 0.85          # a handful of near-zero-headroom configs produce |y| > 1; clip and count
clipped = 0
for fam, st in FAM.items():
    sel = [p for p in pts if p["family"] == fam]
    if not sel:
        continue
    x = np.array([p["damage"] for p in sel])
    y = np.array([p["d_gap_mse"] for p in sel])
    clipped += int((np.abs(y) > YLIM).sum())
    y = np.clip(y, -YLIM, YLIM)
    ax.scatter(x, y, s=18, alpha=0.45, c=st["c"], marker=st["m"], label=f'{st["lab"]} ($n$={len(sel)})',
               edgecolors="none")
    # per-family running median so the trend is per-family, never pooled
    if len(sel) >= 8:
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        w = max(len(xs) // 6, 5)
        starts = list(range(0, len(xs) - w + 1, max(w // 2, 1)))
        if starts[-1] != len(xs) - w:
            starts.append(len(xs) - w)      # cover the family's full damage range
        med_x = [xs[i:i+w].mean() for i in starts]
        med_y = [np.median(ys[i:i+w]) for i in starts]
        ax.plot(med_x, med_y, c=st["c"], lw=2.6)
ax.axhline(0, c="k", lw=0.8, ls="--")
ax.set_xlabel("Damage: top-1 flip rate at the low bit-width", fontsize=14)
ax.set_ylabel(r"Capture(gap) $-$ capture(MSE)", fontsize=14)
ax.set_ylim(-YLIM - 0.06, YLIM + 0.06)
if clipped:
    ax.text(0.995, 0.01, f"{clipped} points clipped to $\\pm${YLIM}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="0.45")
ax.legend(frameon=True, framealpha=0.93, edgecolor="0.85", fontsize=12, loc="upper left")
ax.set_title("Which allocation criterion wins, by setting", fontsize=14)
ax.tick_params(labelsize=13)
fig.tight_layout()
out = os.path.expanduser("~/qat-transfer/paper/figs/fig_boundary.pdf")
fig.savefig(out)
print("saved", out, "with", len(pts), "runs")
