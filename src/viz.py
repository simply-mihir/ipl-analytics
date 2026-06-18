"""
src.viz
=======
Shared plotting style + a handful of reusable plot helpers.

Importing this module sets the matplotlib/seaborn defaults so every notebook
and the README chart pack share one visual identity.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- house style -------------------------------------------------------------
PRIMARY   = "#1f4e79"   # deep blue
ACCENT    = "#c0392b"   # crimson
MUTED     = "#7f8c8d"   # slate
HIGHLIGHT = "#27ae60"   # green
PALETTE   = ["#1f4e79", "#c0392b", "#27ae60", "#f39c12", "#8e44ad", "#16a085"]

def apply_style():
    sns.set_style("whitegrid", {
        "grid.color":      "#eaeaea",
        "axes.edgecolor":  "#cccccc",
    })
    plt.rcParams.update({
        "figure.figsize":   (10, 6),
        "figure.dpi":       110,
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
    })

apply_style()


# --- helpers -----------------------------------------------------------------
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str, fig=None):
    """Save current figure to reports/figures/<name>.png and close it."""
    path = FIG_DIR / name
    (fig or plt.gcf()).savefig(path)
    plt.close(fig or plt.gcf())
    return path


def annotate_bars(ax, fmt="{:.0f}", offset=3, fontsize=10):
    """Write the value above each bar."""
    for p in ax.patches:
        h = p.get_height()
        if h == h:  # not nan
            ax.text(p.get_x() + p.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", va="bottom", fontsize=fontsize)
