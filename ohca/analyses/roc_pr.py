from __future__ import annotations
from typing import Any, Dict, List, Tuple

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .. import plotting, resampling, scoring
from ..dataset import Segment
from .cached import Cached


def _macro_bootstrap(curves: Dict[str, np.ndarray], aucs: Dict[str, float], n_bootstrap: int, desc: str
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[float, float]]:
    # Macro-averaged curve and AUC with percentile CIs from the participant bootstrap.
    pids = list(curves)

    def _one_iter(sampled: List[str]) -> Tuple[np.ndarray, float]:
        return (scoring.macro_average(curves, sampled),
                float(scoring.macro_average({p: np.array([aucs[p]]) for p in sampled}, sampled)[0]))

    mean  = _one_iter(pids)
    boot  = resampling.participants(pids, _one_iter, n_bootstrap, desc=desc)
    mat   = np.stack([c for c, _ in boot])
    boots = np.array([a for _, a in boot])
    return (mean[0], np.percentile(mat, 2.5, axis=0), np.percentile(mat, 97.5, axis=0), mean[1],
            (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))


def _style(ax: Any, xlabel: str, ylabel: str, legend_loc: str) -> None:
    ax.set_xlabel(xlabel, fontsize=plotting.FONT_AXIS)
    ax.set_ylabel(ylabel, fontsize=plotting.FONT_AXIS)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(labelsize=plotting.FONT_TICK)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc=legend_loc, framealpha=0.9, fontsize=plotting.FONT_LEGEND)


class ROCAnalysis(Cached):

    # Segment-level ROC per unit (participant, or a fold's pooled scores), macro-averaged, with a
    # unit-bootstrap CI.

    def __init__(self, pids: List[str], fpr_grid: np.ndarray, tpr_per_participant: Dict[str, np.ndarray],
                 auc_per_participant: Dict[str, float], tpr_mean: np.ndarray, tpr_ci_lo: np.ndarray,
                 tpr_ci_hi: np.ndarray, auc_mean: float, auc_ci: Tuple[float, float]) -> None:
        self.pids                = pids
        self.fpr_grid            = fpr_grid
        self.tpr_per_participant = tpr_per_participant
        self.auc_per_participant = auc_per_participant
        self.tpr_mean            = tpr_mean
        self.tpr_ci_lo           = tpr_ci_lo
        self.tpr_ci_hi           = tpr_ci_hi
        self.auc_mean            = auc_mean
        self.auc_ci              = auc_ci

    @staticmethod
    def create(units: Dict[str, List[Segment]], n_bootstrap: int) -> ROCAnalysis:
        tpr_pp: Dict[str, np.ndarray] = {}
        auc_pp: Dict[str, float]      = {}
        for pid, segments in units.items():
            tpr_pp[pid], auc_pp[pid] = scoring.pooled_roc(np.concatenate([s[2] for s in segments]),
                                                          np.concatenate([s[3] for s in segments]), scoring.PROB_GRID)
        return ROCAnalysis(list(tpr_pp), scoring.PROB_GRID, tpr_pp, auc_pp,
                           *_macro_bootstrap(tpr_pp, auc_pp, n_bootstrap, "Bootstrap ROC"))

    def curve_table(self) -> pd.DataFrame:
        return pd.DataFrame({"fpr": self.fpr_grid, "tpr_mean": self.tpr_mean,
                             "tpr_ci_lo": self.tpr_ci_lo, "tpr_ci_hi": self.tpr_ci_hi})

    def per_participant_table(self) -> pd.DataFrame:
        return pd.DataFrame({"auc": self.auc_per_participant}).rename_axis("participant_id")

    def plot(self, ax: Any, label: str, color: str = "tab:blue") -> None:
        ax.fill_between(self.fpr_grid, self.tpr_ci_lo, self.tpr_ci_hi, alpha=0.20, color=color)
        ax.plot(self.fpr_grid, self.tpr_mean, color=color, lw=1.4,
                label=f"{label}\nAUC-ROC {100 * self.auc_mean:.1f}% [{100 * self.auc_ci[0]:.1f}–{100 * self.auc_ci[1]:.1f}%]")
        ax.plot([0, 1], [0, 1], "k--", lw=0.5, label="Random classifier")
        _style(ax, "False Positive Rate", "Sensitivity", "lower right")


class PRCAnalysis(Cached):

    # Segment-level precision-recall per participant, macro-averaged, with a participant-bootstrap CI.

    def __init__(self, pids: List[str], recall_grid: np.ndarray, prec_per_participant: Dict[str, np.ndarray],
                 prauc_per_participant: Dict[str, float], prec_mean: np.ndarray, prec_ci_lo: np.ndarray,
                 prec_ci_hi: np.ndarray, prauc_mean: float, prauc_ci: Tuple[float, float],
                 prevalence: float) -> None:
        self.pids                  = pids
        self.recall_grid           = recall_grid
        self.prec_per_participant  = prec_per_participant
        self.prauc_per_participant = prauc_per_participant
        self.prec_mean             = prec_mean
        self.prec_ci_lo            = prec_ci_lo
        self.prec_ci_hi            = prec_ci_hi
        self.prauc_mean            = prauc_mean
        self.prauc_ci              = prauc_ci
        self.prevalence            = prevalence

    @staticmethod
    def create(units: Dict[str, List[Segment]], n_bootstrap: int) -> PRCAnalysis:
        prec_pp:  Dict[str, np.ndarray] = {}
        prauc_pp: Dict[str, float]      = {}
        n_pos, n_obs = 0, 0
        for pid, segments in units.items():
            scores = np.concatenate([s[2] for s in segments])
            lab    = np.concatenate([s[3] for s in segments])
            prec_pp[pid], prauc_pp[pid] = scoring.pooled_prc(scores, lab, scoring.PROB_GRID)
            n_pos += int((lab == 1).sum())
            n_obs += len(lab)
        return PRCAnalysis(list(prec_pp), scoring.PROB_GRID, prec_pp, prauc_pp,
                           *_macro_bootstrap(prec_pp, prauc_pp, n_bootstrap, "Bootstrap PRC"),
                           n_pos / n_obs if n_obs > 0 else 0.0)

    def curve_table(self) -> pd.DataFrame:
        return pd.DataFrame({"recall": self.recall_grid, "prec_mean": self.prec_mean,
                             "prec_ci_lo": self.prec_ci_lo, "prec_ci_hi": self.prec_ci_hi})

    def per_participant_table(self) -> pd.DataFrame:
        return pd.DataFrame({"prauc": self.prauc_per_participant}).rename_axis("participant_id")

    def plot(self, ax: Any, label: str, color: str = "tab:blue") -> None:
        ax.fill_between(self.recall_grid, self.prec_ci_lo, self.prec_ci_hi, alpha=0.20, color=color)
        ax.plot(self.recall_grid, self.prec_mean, color=color, lw=1.4,
                label=f"{label}\nPR-AUC {100 * self.prauc_mean:.1f}% [{100 * self.prauc_ci[0]:.1f}–{100 * self.prauc_ci[1]:.1f}%]")
        ax.axhline(self.prevalence, color="k", ls="--", lw=0.5, label="Random classifier")
        _style(ax, "Recall (Sensitivity)", "Precision", "upper right")
