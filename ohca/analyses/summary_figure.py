from __future__ import annotations
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .. import plotting
from .characteristics import plot_effects
from .froc import FROCAnalysis
from .roc_pr import PRCAnalysis, ROCAnalysis

# ROC (a) and PR (b) side by side, FROC (c) centred below them; axes placed by hand so that every panel has
# the same size, the panel width following from the figure width.

PANEL_H_IN       = 1.93
GAP_AB_IN        = 0.75
GAP_AC_IN        = 0.70
MARGIN_LEFT_IN   = 0.67
MARGIN_BOTTOM_IN = 0.54
MARGIN_TOP_IN    = 0.34
MARGIN_RIGHT_IN  = 0.20

FROC_STYLES = (("tab:blue", "-"), ("tab:orange", "--"), ("tab:green", "-"), ("tab:red", "-"))


def panel_width(width_in: float) -> float:
    return (width_in - MARGIN_LEFT_IN - MARGIN_RIGHT_IN - GAP_AB_IN) / 2


def create(roc: ROCAnalysis, prc: PRCAnalysis, frocs: List[Tuple[FROCAnalysis, str]], classifier_label: str,
           panel_label_case: str, width: float) -> Any:

    panel_w = panel_width(width)
    height  = MARGIN_BOTTOM_IN + 2 * PANEL_H_IN + GAP_AC_IN + MARGIN_TOP_IN
    fig     = plt.figure(figsize=(width, height), dpi=plotting.DPI)

    x_a, y_c = MARGIN_LEFT_IN, MARGIN_BOTTOM_IN
    x_b, y_a = x_a + panel_w + GAP_AB_IN, y_c + PANEL_H_IN + GAP_AC_IN
    x_c      = (x_a + x_b) / 2

    def box(x: float, y: float) -> List[float]:
        return [x / width, y / height, panel_w / width, PANEL_H_IN / height]

    ax_a = fig.add_axes(box(x_a, y_a))
    ax_b = fig.add_axes(box(x_b, y_a))
    ax_c = fig.add_axes(box(x_c, y_c))

    roc.plot(ax_a, classifier_label)
    prc.plot(ax_b, classifier_label)
    for (froc, label), (color, ls) in zip(frocs, FROC_STYLES):
        froc.plot(ax_c, color=color, ls=ls, label=label)

    # The PR legend is lifted clear of the near-zero prevalence line.
    for i, (ax, loc) in enumerate(((ax_a, "lower right"), (ax_b, "lower left"), (ax_c, "lower right"))):
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(handles, legend_labels, loc=loc, fontsize=plotting.FONT_LEGEND, framealpha=0.9,
                  borderaxespad=0.3, handlelength=1.4, labelspacing=0.3,
                  bbox_to_anchor=(0.0, 0.10) if ax is ax_b else None)
        ax.set_xlabel(ax.get_xlabel(), fontsize=plotting.FONT_AXIS)
        ax.set_ylabel(ax.get_ylabel(), fontsize=plotting.FONT_AXIS)
        ax.tick_params(labelsize=plotting.FONT_TICK)
        ax.set_title(plotting.panel_letter(i, panel_label_case), loc="left",
                     fontsize=plotting.FONT_PANEL, fontweight="bold")
    return fig


def characteristics(panels: List[Tuple[pd.DataFrame, str]], panel_label_case: str, width: float) -> Any:
    # Effect panels side by side, one per (coefficients, x label), on shared covariate rows: the first
    # panel's covariates, then any other panel's; a covariate a panel lacks leaves its row empty.
    rows: List[str] = []
    for coef, _ in panels:
        rows += [c for c in coef.index if c not in rows]
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 3.0),
                             dpi=plotting.DPI, sharey=True, squeeze=False)
    for i, (ax, (coef, xlabel)) in enumerate(zip(axes[0], panels)):
        plot_effects(ax, coef, xlabel, rows=rows)
        ax.set_title(plotting.panel_letter(i, panel_label_case), loc="left", fontsize=plotting.FONT_PANEL,
                     fontweight="bold")
        if i:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    return fig
