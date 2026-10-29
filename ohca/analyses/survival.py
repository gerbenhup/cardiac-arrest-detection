from __future__ import annotations
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .. import debounce, plotting, resampling, scoring
from ..dataset import Segment
from .cached import Cached

FOLD_WORKERS = 4   # threads computing the per-fold curves


class DebounceSurvivalAnalysis(Cached):

    # FAR needed per sensitivity target vs debounce length. Recordings end shortly after an event, so run
    # lengths are right-censored (Kaplan-Meier); a fold abstains where none of its events has room.
    # Curves are averaged over folds, then inverted.

    MIN_RELIABLE_FOLDS = 30

    def __init__(self, debounce_grid: List[int], sens_targets: List[float], far_mean: Dict[float, np.ndarray],
                 far_ci_lo: Dict[float, np.ndarray], far_ci_hi: Dict[float, np.ndarray], n_folds: np.ndarray,
                 km_events: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]]) -> None:
        self.debounce_grid = debounce_grid
        self.sens_targets  = sens_targets
        self.far_mean      = far_mean
        self.far_ci_lo     = far_ci_lo
        self.far_ci_hi     = far_ci_hi
        self.n_folds       = n_folds
        self.km_events     = km_events

    @staticmethod
    def _fold_curve(segments: List[Segment], d_arr: np.ndarray, far_grid: np.ndarray, targets: Tuple[float, ...]
                    ) -> Tuple[np.ndarray, Dict[float, Tuple[np.ndarray, np.ndarray]]]:

        posts  = [s[2] for s in segments]
        anns   = [s[3] for s in segments]
        curve = np.full((len(d_arr), len(far_grid)), np.nan)
        room  = scoring.event_room(posts, anns)[0]
        if not len(room):
            return curve, {}

        flat, offsets, has_ev, ev_st, thrs = scoring.prepare(posts, anns)
        run_len, censored = scoring.event_run_survival(posts, anns, thrs)
        km = scoring.km_survival(run_len, censored, d_arr)
        for di, d in enumerate(d_arr):
            if int((room >= d).sum()) < 1:
                continue
            _, far = scoring.score_fold_prepared(flat, offsets, has_ev, ev_st, thrs, int(d))
            order  = np.argsort(far)
            curve[di] = scoring.step_hold(far_grid, far[order], np.maximum.accumulate(km[order, di]))

        # Life-table cohort per target: the runs at the highest threshold whose survival at the
        # shortest debounce reaches the target (thresholds are descending).
        env    = np.maximum.accumulate(km[:, 0])
        cohort = {}
        for target in targets:
            hit = np.nonzero(env >= target)[0]
            if hit.size:
                cohort[target] = (run_len[:, int(hit[0])].copy(), censored[:, int(hit[0])].copy())
        return curve, cohort

    @staticmethod
    def create(units: Dict[Any, List[Segment]], far_grid: np.ndarray, targets: Tuple[float, ...],
               n_bootstrap: int) -> DebounceSurvivalAnalysis:

        grid  = debounce.DEBOUNCE_GRID
        d_arr = np.array(grid, dtype=np.int64)
        pids  = list(units)
        folds = Parallel(n_jobs=FOLD_WORKERS, prefer="threads")(
            delayed(DebounceSurvivalAnalysis._fold_curve)(units[p], d_arr, far_grid, targets)
            for p in tqdm(pids, desc="Debounce survival (Kaplan-Meier)"))
        by_pid  = dict(zip(pids, folds))
        stacked = np.stack([by_pid[p][0] for p in pids])
        n_folds = (~np.isnan(stacked[:, :, 0])).sum(axis=0)
        t_arr   = np.asarray(targets, dtype=float)

        def _invert(mean_curve: np.ndarray) -> np.ndarray:
            out = np.full((len(t_arr), len(grid)), np.nan)
            for di in range(len(grid)):
                if not np.isnan(mean_curve[di]).all():
                    out[:, di] = scoring.invert_at_targets(mean_curve[di], far_grid, t_arr)
            return out

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            point   = _invert(np.nanmean(stacked, axis=0))
            idx_mat = resampling.indices(len(pids), n_bootstrap)

            # Warning filters are process-wide, so this block also covers the worker threads.
            def _one_replicate(b: int) -> np.ndarray:
                return _invert(np.nanmean(stacked[idx_mat[b]], axis=0))

            boot = np.stack(list(tqdm(Parallel(n_jobs=resampling.WORKERS, prefer="threads", return_as="generator")(
                delayed(_one_replicate)(b) for b in range(n_bootstrap)),
                total=n_bootstrap, desc="Bootstrap debounce survival", leave=False)))
            return DebounceSurvivalAnalysis(
                list(grid), list(targets),
                {t: point[ti] for ti, t in enumerate(targets)},
                {t: np.nanpercentile(boot[:, ti], 2.5,  axis=0) for ti, t in enumerate(targets)},
                {t: np.nanpercentile(boot[:, ti], 97.5, axis=0) for ti, t in enumerate(targets)},
                n_folds,
                {t: {pid: c[t] for pid, (_, c) in by_pid.items() if t in c} for t in targets})

    def km_table(self, target: float, n_bootstrap: int) -> pd.DataFrame:
        # Kaplan-Meier life table at one operating point: participant-averaged survival with a
        # participant-bootstrap CI (events within a participant are not independent).
        events = self.km_events[target]
        pids   = list(events)
        max_t  = int(max(self.debounce_grid))
        run_len  = np.concatenate([events[p][0] for p in pids])
        censored = np.concatenate([events[p][1] for p in pids])
        lengths  = np.arange(1, max_t + 1)
        km_pp    = np.stack([scoring.km_survival(events[p][0][:, None], events[p][1][:, None], lengths)[0]
                             for p in pids])
        idx_mat = resampling.indices(len(pids), n_bootstrap)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            point = np.nanmean(km_pp, axis=0)
            boot  = np.stack(Parallel(n_jobs=resampling.WORKERS, prefer="threads")(
                delayed(lambda b: np.nanmean(km_pp[idx_mat[b]], axis=0))(b)
                for b in tqdm(range(n_bootstrap), leave=False, desc=f"Bootstrap KM life table ({int(target * 100)}%)")))
        ci_lo, ci_hi, se = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0), boot.std(axis=0, ddof=1)

        rows, at_risk = [], int(sum(len(events[p][0]) for p in pids))
        for t in range(max_t):
            eq     = (run_len == t)
            deaths = int(np.sum(eq & ~censored))
            cens_t = int(np.sum(eq &  censored))
            if t + 1 in self.debounce_grid:
                rows.append({"debounce_s": t + 1, "n_at_risk": at_risk, "n_events": deaths, "n_censored": cens_t,
                             "survival": point[t], "se": se[t],
                             "ci_lo": float(np.clip(ci_lo[t], 0.0, 1.0)), "ci_hi": float(np.clip(ci_hi[t], 0.0, 1.0))})
            at_risk = max(0, at_risk - deaths - cens_t)
        return pd.DataFrame(rows)

    def table(self) -> pd.DataFrame:
        data: Dict[str, np.ndarray] = {"debounce_s": np.array(self.debounce_grid, dtype=float)}
        for target in self.sens_targets:
            pct = int(target * 100)
            data[f"far@{pct}_mean"]  = self.far_mean[target]
            data[f"far@{pct}_ci_lo"] = self.far_ci_lo[target]
            data[f"far@{pct}_ci_hi"] = self.far_ci_hi[target]
        data["n_folds_contributing"] = self.n_folds
        return pd.DataFrame(data)

    def plot(self, width_in: float) -> Any:
        fig, ax = plt.subplots(figsize=(width_in, 3.2), dpi=plotting.DPI)
        nonzero, highs = [], []
        for target in self.sens_targets:
            mean, lo, hi = self.far_mean[target], self.far_ci_lo[target], self.far_ci_hi[target]
            line, = ax.plot(self.debounce_grid, mean, marker="o", ms=3, lw=1.1,
                            label=f"FAR @ {int(target * 100)}% sensitivity")
            ax.fill_between(self.debounce_grid, lo, hi, alpha=0.15, color=line.get_color())
            for arr in (mean, lo, hi):
                pos = arr[np.isfinite(arr) & (arr > 0)]
                if pos.size:
                    nonzero.append(float(pos.min()))
                    highs.append(float(pos.max()))

        ax.set_xlabel("Debounce length (s)", fontsize=plotting.FONT_AXIS)
        ax.set_ylabel("False alarm rate (alarms/hour)", fontsize=plotting.FONT_AXIS)
        ax.tick_params(labelsize=plotting.FONT_TICK)
        linthresh = min(nonzero) / 2.0 if nonzero else 1e-3
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
        ax.set_ylim(-linthresh * 0.3, max(highs) * 1.3 if highs else 1.0)
        ax.set_xlim(self.debounce_grid[0], self.debounce_grid[-1])
        plotting.draw_axis_break(ax, 1e-3, axis="y")

        # Dashed lines where folds start to abstain and where fewer than MIN_RELIABLE_FOLDS remain:
        # a bootstrap CI over a handful of folds narrows instead of widening.
        n_folds  = np.asarray(self.n_folds)
        full     = int(n_folds[0])
        cut_full = self.debounce_grid[int(np.argmax(n_folds < full))] if (n_folds < full).any() else None
        cut_rel  = (self.debounce_grid[int(np.argmax(n_folds < self.MIN_RELIABLE_FOLDS))]
                    if full >= self.MIN_RELIABLE_FOLDS and (n_folds < self.MIN_RELIABLE_FOLDS).any() else None)
        for cut in (cut_full, cut_rel):
            if cut is not None:
                ax.axvline(cut, color="0.3", lw=1.0, ls="--", zorder=6)
        cuts   = [c for c in (cut_full, cut_rel) if c is not None]
        bounds = [self.debounce_grid[0]] + cuts + [self.debounce_grid[-1]]
        names  = {cut_full: f"<{full} folds", cut_rel: f"<{self.MIN_RELIABLE_FOLDS} folds"}
        for lo, hi, name in zip(bounds[:-1], bounds[1:], [f"{full} folds"] + [names[c] for c in cuts]):
            if hi > lo:
                ax.text((lo + hi) / 2, 0.96, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                        fontsize=plotting.FONT_TICK, color="0.3", zorder=7,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
        ax.grid(True, alpha=0.3, which="both", linewidth=0.4)
        ax.legend(fontsize=plotting.FONT_LEGEND, loc="upper right", bbox_to_anchor=(0.995, 0.86),
                  framealpha=0.9, borderaxespad=0.4)
        fig.tight_layout()
        return fig
