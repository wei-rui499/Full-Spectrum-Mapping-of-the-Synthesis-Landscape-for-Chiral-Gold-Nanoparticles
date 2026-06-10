"""
journal_style.py
================
Nature-family journal unified visualization style.
Call apply_style() at the top of any plotting script.

Conventions:
  - No figure titles (set_title) — captions belong in the manuscript
  - All figure text in English
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import numpy as np

# ── Palettes ──────────────────────────────────────────────
PALETTE_8 = [
    "#D1DFE6",  # light grey-blue
    "#3F74A3",  # deep blue
    "#699BC5",  # mid blue
    "#794E47",  # brown
    "#9C7B57",  # brown-gold
    "#A78F95",  # grey-pink
    "#B54646",  # brick red
    "#D18F90",  # light rose
]

# 30-color extended palette (grouped by hue: blue → cyan → green → brown → gold → grey-pink → red → rose)
PALETTE_30 = [
    # ── Blue series (deep→light) ──
    "#1D3F5E",  #  0  ink blue
    "#2B5278",  #  1  deep sea blue
    "#3F74A3",  #  2  deep blue ★
    "#4E88B4",  #  3  steel blue
    "#699BC5",  #  4  mid blue ★
    "#89B4D4",  #  5  sky blue
    "#B8D4E8",  #  6  powder blue
    "#D1DFE6",  #  7  light grey-blue ★
    # ── Cyan series ──
    "#3A7A8C",  #  8  teal
    "#5AACB8",  #  9  light cyan
    # ── Green series (complementary) ──
    "#3A9C7A",  # 10  emerald green
    # ── Brown series (deep→light) ──
    "#5C362E",  # 11  deep brown
    "#794E47",  # 12  brown ★
    "#6B5A3E",  # 13  olive brown
    "#A08068",  # 14  light brown
    "#9C7B57",  # 15  brown-gold ★
    "#BFA070",  # 16  golden brown
    "#D4C4A0",  # 17  sand
    # ── Amber (complementary) ──
    "#C49A3A",  # 18  amber gold
    # ── Grey-pink series ──
    "#8C7580",  # 19  deep grey-pink
    "#A78F95",  # 20  grey-pink ★
    "#C4ADB5",  # 21  light grey-pink
    "#9088A0",  # 22  purple-grey
    # ── Red series (deep→light) ──
    "#8C2E2E",  # 23  deep brick red
    "#A33F5C",  # 24  rose red
    "#B54646",  # 25  brick red ★
    "#C46A3A",  # 26  burnt orange
    "#D06060",  # 27  mid red
    "#D18F90",  # 28  light rose ★
    "#E4B3B4",  # 29  light pink
]

# 4-Cluster colors (maximum contrast)
CLUSTER_COLORS_4 = [
    "#E4B3B4",  # Cluster 0 — light pink
    "#3F74A3",  # Cluster 1 — deep blue
    "#8C2E2E",  # Cluster 2 — deep brick red
    "#8C7580",  # Cluster 3 — deep grey-pink
]

# 6-Cluster colors (4+3 hierarchical)
CLUSTER_COLORS_6 = [
    "#3F74A3",  # C0a
    "#699BC5",  # C0b
    "#D1DFE6",  # C0c
    "#B54646",  # C1
    "#9C7B57",  # C2
    "#794E47",  # C3
]

# ── Size constants (Nature standard) ─────────────────────
SINGLE_COL = 89 / 25.4      # single column width (inches)
DOUBLE_COL = 183 / 25.4     # double column width (inches)
DPI = 600
FIG_FORMAT = "pdf"

# ── Font sizes ───────────────────────────────────────────
FONTSIZE_TICK = 8
FONTSIZE_LABEL = 10
FONTSIZE_TITLE = 10
FONTSIZE_PANEL = 14        # panel label (a, b, c...)
FONTSIZE_LEGEND = 8
FONTSIZE_CBAR = 8

# ── Other constants ──────────────────────────────────────
WAVELENGTH_RANGE = (400, 1100)
OUTPUT_DIR = Path("figures")


def apply_style():
    """Apply global matplotlib style. Call once at script entry."""
    mpl.rcParams.update({
        # fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": FONTSIZE_TICK,

        # axes
        "axes.linewidth": 0.8,
        "axes.labelsize": FONTSIZE_LABEL,
        "axes.titlesize": FONTSIZE_TITLE,
        "axes.titleweight": "bold",
        "axes.spines.top": True,
        "axes.spines.right": True,

        # ticks
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "xtick.direction": "out",
        "xtick.labelsize": FONTSIZE_TICK,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.5,
        "ytick.major.size": 4,
        "ytick.minor.size": 2,
        "ytick.direction": "out",
        "ytick.labelsize": FONTSIZE_TICK,

        # legend
        "legend.frameon": False,
        "legend.fontsize": FONTSIZE_LEGEND,

        # save
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,

        # lines
        "lines.linewidth": 1.2,
        "lines.markersize": 5,

        # background
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # PDF/PS text as TrueType (editable in Adobe Illustrator)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_cluster_colors(n=4):
    """Return a list of n cluster colors."""
    if n <= 4:
        return CLUSTER_COLORS_4[:n]
    elif n <= 6:
        return CLUSTER_COLORS_6[:n]
    else:
        return (PALETTE_30 * ((n // 30) + 1))[:n]


def get_cluster_cmap(n=4):
    """Return a ListedColormap for n clusters."""
    return mcolors.ListedColormap(get_cluster_colors(n))


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add a panel label (a, b, c...) at the top-left of axes, 14 pt bold."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONTSIZE_PANEL, fontweight="bold",
            va="bottom", ha="left")


def save_fig(fig, name, output_dir=None):
    """Save figure as PDF (600 DPI)."""
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"{name}.{FIG_FORMAT}"
    fig.savefig(filepath, format=FIG_FORMAT, dpi=DPI)
    plt.close(fig)
    print(f"Saved -> {filepath}")
    return filepath


def cbar_style(cbar):
    """Apply uniform colorbar appearance."""
    cbar.ax.tick_params(labelsize=FONTSIZE_TICK, width=0.6, length=3)
    cbar.outline.set_linewidth(0.6)


# ── Self-test: run this file to display palettes ─────────
if __name__ == "__main__":
    apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 8))

    # Left: 30-color palette
    ax = axes[0]
    dark_indices = {0, 1, 2, 8, 10, 11, 12, 13, 19, 23, 24, 25}
    for i, c in enumerate(PALETTE_30):
        ax.barh(i, 1, color=c, edgecolor="white", linewidth=0.5)
        ax.text(0.5, i, f"{i:2d}  {c}", ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white" if i in dark_indices else "#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(PALETTE_30) - 0.5)
    ax.invert_yaxis()
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("30-Color Palette")
    add_panel_label(ax, "a")

    # Right: 4-Cluster assignment
    ax = axes[1]
    for i, c in enumerate(CLUSTER_COLORS_4):
        ax.barh(i, 1, color=c, edgecolor="white", linewidth=0.5)
        ax.text(0.5, i, f"Cluster {i}  {c}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("4-Cluster Assignment")
    add_panel_label(ax, "b")

    fig.tight_layout()
    save_fig(fig, "palette_preview", output_dir="figures")
