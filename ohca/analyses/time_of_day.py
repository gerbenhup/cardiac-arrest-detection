from __future__ import annotations
import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smt
from joblib import Parallel, delayed

from .. import dataset, environment, labels, plotting, resampling, scoring
from .cached import Cached

HOURS_PER_BLOCK = 6
N_BLOCKS        = 24 // HOURS_PER_BLOCK


class TimeBlockFARAnalysis(Cached):

    # FAR by local hour and 6-hour block at each target's cohort FAR budget (from the FROC), +-15 min around
    # events excluded. Blocks: Poisson GEE (exchangeable, Mancl-DeRouen covariance), an omnibus Wald test
    # and Bonferroni-corrected pairwise IRRs.

    BLOCK_LABELS = {0: "00-06", 1: "06-12", 2: "12-18", 3: "18-00"}
    MARGIN_S     = 15 * 60

    def __init__(self, sensitivity_targets: Tuple[float, ...], hour_rate: Dict[float, np.ndarray],
                 hour_ci_lo: Dict[float, np.ndarray], hour_ci_hi: Dict[float, np.ndarray],
                 block_rate: Dict[float, np.ndarray], block_ci_lo: Dict[float, np.ndarray],
                 block_ci_hi: Dict[float, np.ndarray], pairwise_table: pd.DataFrame,
                 omnibus: Dict[float, Dict[str, float]]) -> None:
        self.sensitivity_targets = sensitivity_targets
        self.hour_rate           = hour_rate
        self.hour_ci_lo          = hour_ci_lo
        self.hour_ci_hi          = hour_ci_hi
        self.block_rate          = block_rate
        self.block_ci_lo         = block_ci_lo
        self.block_ci_hi         = block_ci_hi
        self.pairwise_table      = pairwise_table
        self.omnibus             = omnibus

    def omnibus_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"sensitivity_target": t, **self.omnibus[t]}
                             for t in self.sensitivity_targets if t in self.omnibus])

    @staticmethod
    def _contrast(b: int, block_idx: Dict[int, int], ref_block: Optional[int], n_params: int) -> Optional[np.ndarray]:
        # Sum coding: the reference block is minus the sum of the other block terms.
        c = np.zeros(n_params)
        if b in block_idx:
            c[block_idx[b]] = 1.0
        elif b == ref_block:
            for idx in block_idx.values():
                c[idx] = -1.0
        else:
            return None
        return c

    @staticmethod
    def _fold_counts(r: Dict[str, Any], h5: dataset.H5Dataset, debounce: int, cohort_far: Dict[float, float],
                     targets: Tuple[float, ...]
                     ) -> Tuple[Dict[Tuple[str, int], float], Dict[float, Dict[Tuple[str, int], float]]]:

        L     = dataset.STEP
        tz    = dataset.TIMEZONE.key
        posts = [s[2] for s in r["test_scores"]]
        flat, offsets, has_ev, ev_st, thrs = scoring.prepare(posts, [s[3] for s in r["test_scores"]])
        _, far = scoring.score_fold_prepared(flat, offsets, has_ev, ev_st, thrs, debounce)

        # Within budget, the threshold that spends most of it; FAR is not monotone in the index.
        t_star: Dict[float, float] = {}
        for target in targets:
            budget = cohort_far.get(target, np.inf)
            if not np.isfinite(budget):
                t_star[target] = float(thrs[-1])
                continue
            ok = np.where(far <= budget)[0]
            t_star[target] = float(thrs[ok[int(np.argmax(far[ok]))]]) if ok.size else float(thrs[0])

        raw = {pid: h5.labels_and_times(pid) for pid in dict.fromkeys(s[0] for s in r["test_scores"])}
        valid_chunks: Dict[str, List[np.ndarray]] = {pid: [] for pid in raw}
        fa_chunks: Dict[float, Dict[str, List[np.ndarray]]] = {t: {pid: [] for pid in raw} for t in targets}
        for pid, si, scores, _ in r["test_scores"]:
            raw_ann, t_arr = raw[pid][si]
            n_sc     = len(scores)
            t_scores = t_arr[np.arange(n_sc, dtype=np.intp) * L].astype(np.float64)
            in_win   = np.zeros(n_sc, dtype=bool)
            for on, end in labels.regions(labels.positive(raw_ann)):
                lo = float(t_arr[on]) - TimeBlockFARAnalysis.MARGIN_S
                hi = float(t_arr[min(end - 1, len(t_arr) - 1)]) + TimeBlockFARAnalysis.MARGIN_S
                in_win |= (t_scores >= lo) & (t_scores <= hi)
            if not (~in_win).any():
                continue
            valid_chunks[pid].append(t_scores[~in_win])
            scores = np.asarray(scores)
            for target in targets:
                fa = np.array([o for o in scoring.debounce_onsets(scores > t_star[target], debounce)
                               if not in_win[o]], dtype=np.intp)
                if fa.size > 0:
                    fa_chunks[target][pid].append(t_scores[fa])

        at_risk: Dict[Tuple[str, int], float] = {}
        for pid, chunks in valid_chunks.items():
            if chunks:
                hours = pd.to_datetime(np.concatenate(chunks), unit="s", utc=True).tz_convert(tz).hour.values
                for h in range(24):
                    sec = float((hours == h).sum()) * scoring.STEP_S
                    if sec > 0.0:
                        at_risk[(pid, h)] = sec
        fa_count: Dict[float, Dict[Tuple[str, int], float]] = {}
        for target in targets:
            counts: Dict[Tuple[str, int], float] = {}
            for pid, chunks in fa_chunks[target].items():
                if chunks:
                    for h in pd.to_datetime(np.concatenate(chunks), unit="s", utc=True).tz_convert(tz).hour.values:
                        counts[(pid, int(h))] = counts.get((pid, int(h)), 0.0) + 1.0
            fa_count[target] = counts
        return at_risk, fa_count

    @staticmethod
    def _hourly(fa_count: Dict[Tuple[str, int], float], at_risk: Dict[Tuple[str, int], float], n_bootstrap: int,
                workers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Per-participant hourly rates, macro-averaged; an hour without at-risk time is NaN, not zero.
        pids    = sorted({pid for pid, _ in at_risk})
        rate_pp = np.full((len(pids), 24), np.nan)
        for i, pid in enumerate(pids):
            for h in range(24):
                sec = at_risk.get((pid, h))
                if sec is not None and sec > 0.0:
                    rate_pp[i, h] = fa_count.get((pid, h), 0.0) / sec * scoring.HOUR_S
        with warnings.catch_warnings():
            # Warning filters are process-wide, so this block also covers the worker threads.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            idx_mat = resampling.indices(len(pids), n_bootstrap)
            boot    = np.stack(Parallel(n_jobs=workers, prefer="threads")(
                delayed(lambda b: np.nanmean(rate_pp[idx_mat[b]], axis=0))(b) for b in range(n_bootstrap)))
            return (np.nanmean(rate_pp, axis=0), np.nanpercentile(boot, 2.5, axis=0),
                    np.nanpercentile(boot, 97.5, axis=0))

    @staticmethod
    def _block_table(fa_count: Dict[Tuple[str, int], float], at_risk: Dict[Tuple[str, int], float]) -> pd.DataFrame:
        block_fa: Dict[Tuple[str, int], float] = {}
        block_at: Dict[Tuple[str, int], float] = {}
        for (pid, h), sec in at_risk.items():
            block_at[(pid, h // HOURS_PER_BLOCK)] = block_at.get((pid, h // HOURS_PER_BLOCK), 0.0) + sec
        for (pid, h), n in fa_count.items():
            block_fa[(pid, h // HOURS_PER_BLOCK)] = block_fa.get((pid, h // HOURS_PER_BLOCK), 0.0) + n
        df = pd.DataFrame([{"participant_id": pid, "block": b, "fa_count": block_fa.get((pid, b), 0.0),
                            "time_at_risk": sec} for (pid, b), sec in block_at.items()])
        df["block"] = df["block"].astype("category")
        return df

    @staticmethod
    def _no_alarms(df: pd.DataFrame, present: List[int], target: float
                   ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], pd.DataFrame]:
        # No false alarms at all: rate 0 with an exact one-sided upper bound; not testable.
        cls  = TimeBlockFARAnalysis
        rate = (np.zeros(N_BLOCKS), np.zeros(N_BLOCKS), np.full(N_BLOCKS, np.nan))
        time_b = df.groupby("block", observed=True)["time_at_risk"].sum()
        for b in present:
            if float(time_b.get(b, 0.0)) > 0.0:
                rate[2][b] = float(scipy.stats.chi2.ppf(0.975, 2)) / (2.0 * float(time_b.get(b, 0.0))) * scoring.HOUR_S
        pairwise = pd.DataFrame([{"sensitivity_target": target, "block_i": i, "block_j": j,
                                  "label": f"{cls.BLOCK_LABELS[i]} vs {cls.BLOCK_LABELS[j]}",
                                  "irr": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p_raw": np.nan}
                                 for i, j in itertools.combinations(present, 2)])
        pairwise["p_bonf"] = np.nan
        return rate, {"wald_chi2": float("nan"), "df": 0, "p": float("nan")}, pairwise

    @staticmethod
    def _gee(df: pd.DataFrame, present: List[int], target: float
             ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any], pd.DataFrame]:
        # Poisson GEE of the block counts: rates with t-based CIs, the omnibus Wald test, pairwise IRRs.
        cls = TimeBlockFARAnalysis
        gee = smf.gee("fa_count ~ C(block, Sum)", groups="participant_id", data=df,
                      offset=np.log(df["time_at_risk"].values), family=sm.families.Poisson(),
                      cov_struct=sm.cov_struct.Exchangeable())
        try:
            fit = gee.fit(maxiter=1000, cov_type="bias_reduced")
        except Exception as e:   # statsmodels raises several types here
            environment.log(f"Time of day: Mancl-DeRouen covariance failed ({e!r}); robust sandwich used")
            fit = gee.fit(maxiter=1000)
        names    = list(fit.params.index)
        beta     = np.asarray(fit.params.values, dtype=np.float64)
        cov      = np.asarray(fit.cov_params())
        df_resid = max(int(df["participant_id"].nunique()) - len(names), 1)
        tcrit    = float(scipy.stats.t.ppf(0.975, df_resid))

        block_terms = [i for i, nm in enumerate(names) if nm.startswith("C(block, Sum)")]
        R = np.zeros((len(block_terms), len(names)))
        for k, i in enumerate(block_terms):
            R[k, i] = 1.0
        try:
            wald = fit.wald_test(R, scalar=True)
        except TypeError:
            wald = fit.wald_test(R)
        omnibus = {"wald_chi2": float(np.squeeze(np.asarray(wald.statistic))), "df": int(len(block_terms)),
                   "p": float(np.squeeze(np.asarray(wald.pvalue)))}

        block_idx = {b: names.index(f"C(block, Sum)[S.{b}]") for b in present if f"C(block, Sum)[S.{b}]" in names}
        refs      = [b for b in present if b not in block_idx]
        ref_block = refs[0] if refs else None

        rate = (np.full(N_BLOCKS, np.nan), np.full(N_BLOCKS, np.nan), np.full(N_BLOCKS, np.nan))
        for b in present:
            c = cls._contrast(b, block_idx, ref_block, len(names))
            if c is None:
                continue
            c[names.index("Intercept")] = 1.0
            eta, se = float(c @ beta), float(np.sqrt(c @ cov @ c))
            rate[0][b] = np.exp(eta) * scoring.HOUR_S
            rate[1][b] = np.exp(eta - tcrit * se) * scoring.HOUR_S
            rate[2][b] = np.exp(eta + tcrit * se) * scoring.HOUR_S

        rows = []
        for i, j in itertools.combinations(present, 2):
            c_i = cls._contrast(i, block_idx, ref_block, len(names))
            c_j = cls._contrast(j, block_idx, ref_block, len(names))
            if c_i is None or c_j is None:
                continue
            c    = c_i - c_j
            b_ij = float(c @ beta)
            se   = float(np.sqrt(c @ cov @ c))
            rows.append({"sensitivity_target": target, "block_i": i, "block_j": j,
                         "label": f"{cls.BLOCK_LABELS[i]} vs {cls.BLOCK_LABELS[j]}",
                         "irr": float(np.exp(b_ij)), "ci_lo": float(np.exp(b_ij - tcrit * se)),
                         "ci_hi": float(np.exp(b_ij + tcrit * se)),
                         "p_raw": float(2.0 * scipy.stats.t.sf(abs(b_ij / se if se > 0 else 0.0), df_resid))})
        pairwise = pd.DataFrame(rows)
        pairwise["p_bonf"] = (smt.multipletests(pairwise["p_raw"].values, method="bonferroni")[1]
                              if len(pairwise) else pd.Series(dtype=float))
        return rate, omnibus, pairwise

    @staticmethod
    def create(results: List[Dict[str, Any]], h5: dataset.H5Dataset, debounce: Dict[int, int],
               cohort_far: Dict[float, float], targets: Tuple[float, ...], n_bootstrap: int,
               workers: int) -> TimeBlockFARAnalysis:

        cls   = TimeBlockFARAnalysis
        folds = Parallel(n_jobs=workers, prefer="threads")(
            delayed(cls._fold_counts)(r, h5, debounce[r["fold"]], cohort_far, targets) for r in results)

        out: Dict[str, Dict[float, Any]] = {k: {} for k in ("hour_rate", "hour_ci_lo", "hour_ci_hi", "block_rate",
                                                            "block_ci_lo", "block_ci_hi", "omnibus")}
        pairwise_frames: List[pd.DataFrame] = []
        for target in targets:
            fa_count: Dict[Tuple[str, int], float] = {}
            at_risk:  Dict[Tuple[str, int], float] = {}
            for fold_at, fold_fa in folds:
                for k, v in fold_fa[target].items():
                    fa_count[k] = fa_count.get(k, 0.0) + v
                for k, v in fold_at.items():
                    at_risk[k] = at_risk.get(k, 0.0) + v
            out["hour_rate"][target], out["hour_ci_lo"][target], out["hour_ci_hi"][target] = \
                cls._hourly(fa_count, at_risk, n_bootstrap, workers)

            df      = cls._block_table(fa_count, at_risk)
            present = sorted(int(b) for b in df["block"].cat.categories)
            model   = cls._no_alarms if float(df["fa_count"].sum()) == 0.0 else cls._gee
            (rate, lo, hi), omnibus, pairwise = model(df, present, target)
            out["block_rate"][target], out["block_ci_lo"][target], out["block_ci_hi"][target] = rate, lo, hi
            out["omnibus"][target] = omnibus
            pairwise_frames.append(pairwise)

        return cls(tuple(targets), pairwise_table=pd.concat(pairwise_frames, ignore_index=True), **out)

    def plot(self, width_in: float) -> Any:
        targets   = self.sensitivity_targets
        fig, axes = plt.subplots(len(targets), 1, figsize=(width_in, 6.0), dpi=plotting.DPI,
                                 sharex=True, sharey=False)
        axes  = [axes] if len(targets) == 1 else axes
        hours = np.arange(24)

        for i, (ax, target) in enumerate(zip(axes, targets)):
            for b in range(0, N_BLOCKS, 2):
                ax.axvspan(b * HOURS_PER_BLOCK, (b + 1) * HOURS_PER_BLOCK, color="0.5", alpha=0.06, zorder=0)

            rate, lo, hi = self.hour_rate[target], self.hour_ci_lo[target], self.hour_ci_hi[target]
            ax.errorbar(hours + 0.5, rate, yerr=[rate - lo, hi - rate], fmt="o", color="tab:blue",
                        ecolor="tab:blue", capsize=2.2, markersize=5, elinewidth=1.1, capthick=1.1, zorder=2,
                        label="Hourly estimate (bootstrap, 95% CI)")
            for h in hours[~np.isnan(rate)]:
                ax.hlines([lo[h], hi[h]], h, h + 1, color="tab:blue", linewidth=1.0, zorder=2)

            for b in range(N_BLOCKS):
                ax.fill_between([b * HOURS_PER_BLOCK, (b + 1) * HOURS_PER_BLOCK], self.block_ci_lo[target][b],
                                self.block_ci_hi[target][b],
                                color="firebrick", alpha=0.18, linewidth=0, zorder=3)
                ax.hlines(self.block_rate[target][b], b * HOURS_PER_BLOCK, (b + 1) * HOURS_PER_BLOCK, color="firebrick",
                          linewidth=1.4,
                          zorder=4, label="6-hour block estimate (GEE Poisson, 95% CI)" if b == 0 else None)

            # symlog places zero CI bounds; the linear span sits just under the smallest positive
            # bound and at least two powers of ten stay in view.
            def finite_positive(arr: np.ndarray) -> np.ndarray:
                return arr[(~np.isnan(arr)) & (arr > 0)]

            lows      = [finite_positive(a).min() for a in (lo, self.block_ci_lo[target]) if finite_positive(a).size]
            highs     = [finite_positive(a).max() for a in (hi, self.block_ci_hi[target]) if finite_positive(a).size]
            linthresh = min(lows) / 2.0 if lows else 1e-3
            y_hi      = max(max(highs) * 1.15 if highs else 1.0, 10.0 ** np.ceil(np.log10(linthresh)) * 10.0)
            y_lo      = -linthresh * 0.25
            ax.set_yscale("symlog", linthresh=linthresh, linscale=0.25)
            ax.set_xlim(0, 24)
            ax.set_ylim(y_lo, y_hi)
            plotting.draw_axis_break(ax, linthresh, axis="y")

            # Only 0 and powers of ten outside the linear span, and only ticks inside the view (an
            # out-of-view tick passed to set_yticks re-expands the axis).
            def in_view(t: float) -> bool:
                return (t == 0.0 or t >= linthresh) and y_lo <= t <= y_hi

            major   = [t for t in ax.yaxis.get_majorticklocs() if in_view(t)]
            ax.set_yticks(major)
            ax.set_yticklabels(["0" if t == 0.0 else
                                (f"$\\mathdefault{{10^{{{round(np.log10(t))}}}}}$"
                                 if np.isclose(t, 10.0 ** round(np.log10(t)), rtol=1e-9) else f"{t:g}")
                                for t in major])
            ax.set_yticks([t for t in ax.yaxis.get_minorticklocs() if in_view(t)], minor=True)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.3, which="both", linewidth=0.4)
            ax.set_title(plotting.panel_letter(i, "upper"), loc="left", fontsize=plotting.FONT_PANEL, fontweight="bold")
            ax.set_ylabel(f"FAR @ {int(target * 100)}% sensitivity", fontsize=plotting.FONT_AXIS)
            ax.tick_params(labelsize=plotting.FONT_TICK)
            om = self.omnibus.get(target)
            if om is not None:
                p_txt = "<0.001" if om["p"] < 1e-3 else f"={om['p']:.3f}"
                ax.set_title(f"omnibus $\\chi^2_{{{om['df']}}}$={om['wald_chi2']:.1f}, $p${p_txt}",
                             loc="right", fontsize=plotting.FONT_TICK)

        axes[-1].set_xticks(hours)
        axes[-1].set_xticklabels([f"{h:02d}" for h in hours], fontsize=plotting.FONT_TICK, rotation=90)
        axes[-1].set_xlabel("Hour of day (local timezone)", fontsize=plotting.FONT_AXIS)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, fontsize=plotting.FONT_LEGEND, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0), ncol=2, framealpha=0.9)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        fig.subplots_adjust(hspace=0.30)
        return fig
