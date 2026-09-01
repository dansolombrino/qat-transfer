"""Shared figure styling: LaTeX fonts, larger type. Imported by every paper figure script."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

def use_paper_style(base=12):
    """LaTeX text rendering when a TeX toolchain is present, mathtext fallback otherwise."""
    has_tex = shutil.which("latex") is not None
    plt.rcParams.update({
        "text.usetex": has_tex,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "figure.titlesize": base + 2,
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
    })
    if has_tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{times}\usepackage{amsmath}"
    return has_tex
