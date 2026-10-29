from __future__ import annotations
import warnings
from typing import Any, Dict, List, Optional

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .. import labels, plotting, resampling, scoring
from ..dataset import Segment
from .cached import Cached


class FROCAnalysis(Cached):

    # Event-level sensitivity vs FAR at each fold's debounce length, one curve per fold. km_corrected: Kaplan-
    # Meier sensitivity. far_units: a fold's FAR from those segments, on the thresholds of both together.

    def __init__(self, km_corrected: bool, far_grid: np.ndarray, froc_per_participant: Dict[str, np.ndarray],
                 weights_per_participant: Dict[str, float], froc_mean: np.ndarray, froc_ci_lo: np.ndarray,
                 froc_ci_hi: np.ndarray) -> None:
        self.km_corrected            = km_corrected
        self.far_grid                = far_grid
        self.froc_per_participant    = froc_per_participant
        self.weights_per_participant = weights_per_participant
        self.froc_mean               = froc_mean
        self.froc_ci_lo              = froc_ci_lo
        self.froc_ci_hi              = froc_ci_hi

    @property
    def label(self) -> str:
        return "Censoring-corrected" if self.km_corrected else "Censoring-naive"

    @staticmethod
    def create(units: Dict[Any, List[Segment]], debounce: Dict[Any, int], km_corrected: bool,
               far_grid: np.ndarray, micro: bool, n_bootstrap: int,
               far_units: Optional[Dict[Any, List[Segment]]] = None) -> FROCAnalysis:

        froc_pp:     Dict[Any, np.ndarray] = {}
        n_events_pp: Dict[Any, int]        = {}
        for pid, segments in units.items():
            min_dur = debounce[pid]
            posts   = [s[2] for s in segments]
            anns    = [s[3] for s in segments]
            n_events_pp[pid] = sum(1 for a in anns if labels.has_event(a))
            flat, offsets, has_ev, ev_st, thrs = scoring.prepare(posts, anns)
            if far_units is None:
                sens, far = scoring.score_fold_prepared(flat, offsets, has_ev, ev_st, thrs, min_dur)
            else:
                f_posts = [s[2] for s in far_units[pid]]
                thrs    = scoring.prepare(posts + f_posts, anns + [s[3] for s in far_units[pid]])[4]
                sens, _ = scoring.score_fold_prepared(flat, offsets, has_ev, ev_st, thrs, min_dur)
                _, far  = scoring.score_fold_prepared(*scoring.prepare(f_posts, [s[3] for s in far_units[pid]])[:4],
                                                      thrs, min_dur)
            order = np.argsort(far)
            if km_corrected:
                run_len, censored = scoring.event_run_survival(posts, anns, thrs)
                if run_len.shape[0] > 0:
                    sens = scoring.km_survival(run_len, censored, np.array([min_dur]))[:, 0]
            froc_pp[pid] = scoring.step_hold(far_grid, far[order], np.maximum.accumulate(sens[order]))

        pids    = list(froc_pp)
        weights = {p: float(n_events_pp[p]) if micro else 1.0 for p in pids}
        boot    = FROCAnalysis._boot(froc_pp, weights, n_bootstrap)
        return FROCAnalysis(
            km_corrected, far_grid, froc_pp, weights,
            scoring.macro_average(froc_pp, pids, np.array([weights[p] for p in pids])),
            np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0))

    @staticmethod
    def _boot(froc_pp: Dict[Any, np.ndarray], weights: Dict[Any, float], n_bootstrap: int) -> np.ndarray:
        # The weighted macro-average curve of each participant-bootstrap resample.
        pids = list(froc_pp)
        w    = np.array([weights[p] for p in pids])
        return np.stack([scoring.macro_average(froc_pp, [pids[j] for j in row], w[row])
                         for row in resampling.indices(len(pids), n_bootstrap)])

    def curve_table(self) -> pd.DataFrame:
        return pd.DataFrame({"false_alarms_per_hour": self.far_grid, "sensitivity_mean": self.froc_mean,
                             "sensitivity_ci_lo": self.froc_ci_lo, "sensitivity_ci_hi": self.froc_ci_hi})

    def far_at_targets(self, targets: np.ndarray) -> np.ndarray:
        return scoring.invert_at_targets(self.froc_mean, self.far_grid, targets)

    def far_table(self, n_bootstrap: int) -> pd.DataFrame:
        # FAR needed per sensitivity, CI from the inverted bootstrap curves. A resample that never reaches
        # the target counts as +inf, so a bound is "not reached" once over 2.5% of resamples miss it.
        targets  = np.arange(0.0, 1.001, 0.05)
        boot     = np.stack([scoring.invert_at_targets(c, self.far_grid, targets)
                             for c in self._boot(self.froc_per_participant, self.weights_per_participant, n_bootstrap)])
        censored = np.where(np.isnan(boot), np.inf, boot)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ci_lo = np.percentile(censored, 2.5,  axis=0)
            ci_hi = np.percentile(censored, 97.5, axis=0)
        return pd.DataFrame({"sensitivity":                 targets,
                             "false_alarms_per_hour_mean":  self.far_at_targets(targets),
                             "false_alarms_per_hour_ci_lo": np.where(np.isnan(ci_lo), np.inf, ci_lo),
                             "false_alarms_per_hour_ci_hi": np.where(np.isnan(ci_hi), np.inf, ci_hi)})

    def plot(self, ax: Any, color: str = "tab:blue", ls: str = "-", label: Optional[str] = None) -> None:
        linthresh = 1e-3
        ax.fill_between(self.far_grid, self.froc_ci_lo, self.froc_ci_hi, step="post", alpha=0.18, color=color)
        ax.step(self.far_grid, self.froc_mean, where="post", color=color, lw=1.4, ls=ls,
                label=self.label if label is None else label)
        # symlog places the FAR = 0 point; the log ticks start at 1e-3 so they clear the 0 tick.
        ax.set_xscale("symlog", linthresh=linthresh, linscale=0.5)
        ax.set_xticks([0.0] + [10.0 ** k for k in range(-3, 3)])
        ax.set_xlim(0.0, 1e2)
        plotting.draw_axis_break(ax, linthresh, axis="x")
        ax.set_xlabel("False alarm rate (alarms/hour)", fontsize=plotting.FONT_AXIS)
        ax.set_ylabel("Sensitivity", fontsize=plotting.FONT_AXIS)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.tick_params(labelsize=plotting.FONT_TICK)
        ax.grid(True, alpha=0.3, which="both", linewidth=0.4)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=plotting.FONT_LEGEND)

