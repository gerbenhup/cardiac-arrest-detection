from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Gill Sans", "Myriad Pro", "Arial", "Helvetica"]

SINGLE_COL_IN = 81.5  / 25.4
DOUBLE_COL_IN = 175.7 / 25.4
S2_WIDTH_IN   = 150.0 / 25.4
DPI           = 1200
FONT_AXIS     = 9      # pt at the printed width
FONT_TICK     = 7
FONT_LEGEND   = 6.5
FONT_PANEL    = 10


def panel_letter(i: int, case: str) -> str:
    return f"({chr((65 if case == 'upper' else 97) + i)})"


def save(fig: Any, stem: Path) -> None:
    # Saved at the figure's own size (no cropping), so it prints at the width it was laid out for. No
    # CreationDate in the PDF, so regenerating a figure gives an identical file.
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.parent / f"{stem.name}.png")
    fig.savefig(stem.parent / f"{stem.name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def draw_axis_break(ax: Any, pos: float, axis: str = "x") -> None:
    # Two diagonal slashes at the middle of a symlog axis's linear span (0..pos). Sized in inches,
    # so the marker has the same on-page size and a true 45 degree angle on axes of any aspect.
    fig = ax.figure
    fig.canvas.draw()
    bbox   = ax.get_window_extent()
    size   = 0.035 * fig.dpi
    dx, dy = size / bbox.width, size / bbox.height
    gap_px = 0.03 * fig.dpi
    lw     = matplotlib.rcParams["axes.linewidth"]
    if axis == "x":
        gap  = gap_px / bbox.width
        frac = ax.transAxes.inverted().transform(ax.transData.transform((pos / 2.0, 0)))[0]
        for c in (frac - gap, frac + gap):
            ax.plot([c - dx, c + dx], [-dy, dy], transform=ax.transAxes,
                    color="k", lw=lw, clip_on=False, zorder=10)
    else:
        gap  = gap_px / bbox.height
        frac = ax.transAxes.inverted().transform(ax.transData.transform((0, pos / 2.0)))[1]
        for c in (frac - gap, frac + gap):
            ax.plot([-dx, dx], [c - dy, c + dy], transform=ax.transAxes,
                    color="k", lw=lw, clip_on=False, zorder=10)
