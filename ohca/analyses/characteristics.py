from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smt
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from .. import environment, plotting
from .cached import Cached


MAX_REDRAWS = 50   # resamples tried per bootstrap replicate
FIT_ERRORS  = (ValueError, np.linalg.LinAlgError, PerfectSeparationError)   # a resample that cannot be fitted


def _bootstrap_table(df: pd.DataFrame, binary: List[str], estimate: Callable[[pd.DataFrame], np.ndarray],
                     point: np.ndarray, null: float, coef_names: List[str], n_bootstrap: int, seed: int,
                     cluster_of: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    # Bootstrap over participants, or over the clusters of cluster_of (all of a cluster's rows together):
    # percentile CIs and Bonferroni-corrected two-sided p-values against `null`. A resample that lacks a
    # category of a binary predictor, or whose fit fails, is redrawn.
    pids    = df["participant_id"].tolist()
    indexed = df.set_index("participant_id")
    units   = pids if cluster_of is None else sorted({cluster_of[p] for p in pids})
    members = ({p: [p] for p in pids} if cluster_of is None else
               {u: [p for p in pids if cluster_of[p] == u] for u in units})
    rng     = np.random.default_rng(seed)
    rows, n_dropped, n_redrawn = [], 0, 0
    for _ in range(n_bootstrap):
        for _attempt in range(MAX_REDRAWS):
            drawn   = [p for u in rng.choice(units, size=len(units), replace=True) for p in members[u]]
            boot_df = indexed.loc[drawn].reset_index()
            n_redrawn += _attempt > 0
            if any(boot_df[c].nunique() < 2 for c in binary):
                continue
            try:
                rows.append(estimate(boot_df))
                break
            except FIT_ERRORS:
                continue
        else:
            n_dropped += 1
    if n_dropped or n_redrawn:
        environment.log(f"Characteristics: {n_redrawn} resamples redrawn, {n_dropped}/{n_bootstrap} bootstrap "
                        f"replicates dropped")

    if not rows:
        raise ValueError("no bootstrap replicate could be fitted")
    boot = np.stack(rows)
    ci   = np.percentile(boot, [2.5, 97.5], axis=0).T
    praw = np.array([2.0 * min(float((boot[:, j] > null).mean()), float((boot[:, j] < null).mean()))
                     for j in range(boot.shape[1])])
    return pd.DataFrame({"coef": point, "ci_lo": ci[:, 0], "ci_hi": ci[:, 1], "p_raw": praw,
                         "p_bonf": smt.multipletests(praw, method="bonferroni")[1]}, index=coef_names)


def _design(metric: Dict[str, float], metadata: pd.DataFrame, covariates: Tuple[str, ...], outcome: str
            ) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = CharacteristicsAnalysis.covariate_table(metadata.loc[metadata.index.isin(metric)])
    df = df.reset_index(drop=False)
    df[outcome] = [metric[p] for p in df["participant_id"]]
    # A constant column would make the design singular.
    coef_names = [c for c in covariates if c in df and df[c].notna().any() and df[c].std() > 0]
    df     = df.dropna(subset=coef_names + [outcome]).reset_index(drop=True)
    binary = [c for c in coef_names if set(df[c].unique()) <= {0.0, 1.0}]
    return df, coef_names, binary


class CharacteristicsAnalysis(Cached):

    # Fractional-logit GLM (Papke & Wooldridge 1996) of a per-participant metric in [0, 1]: average marginal
    # effects in percentage points, bootstrap CIs and Bonferroni-corrected p-values, generalized VIFs.

    LABELS = {
        "sex_m":                  "Sex (M vs F)",
        "age_decades":            "Age (decades)",
        "bmi":                    "BMI (kg/m²)",
        "skin_tone_num":          "Skin tone (I→VI)",
        "wrist_circumference_cm": "Wrist circumference (cm)",
        "icd_test":               "Procedure type\n(ICD-test vs. ablation)",
    }
    XLABEL       = "Average marginal effect\n(percentage points)"
    PANEL_XLABEL = "Average marginal effect\n(PR-AUC percentage points)"   # next to another cohort's panel

    def __init__(self, coefficients: pd.DataFrame, vif: Dict[str, float]) -> None:
        self.coefficients = coefficients
        self.vif          = vif

    @staticmethod
    def covariate_table(metadata: pd.DataFrame) -> pd.DataFrame:
        # Natural units: binary sex, age in decades, BMI, ordinal skin tone (NaN when unrecorded),
        # wrist circumference in cm. A study derives only the covariates it records.
        df = metadata.copy()
        if "sex" in df:
            df["sex_m"] = (df["sex"].astype(str).str.upper().str[0] == "M").astype(float)
        if "skin_tone" in df:
            codes               = df["skin_tone"].cat.codes
            df["skin_tone_num"] = np.where(codes >= 0, codes + 1, np.nan)
        if "age" in df:
            df["age_decades"] = df["age"] / 10.0
        if "wrist_circumference_mm" in df:
            df["wrist_circumference_cm"] = df["wrist_circumference_mm"] / 10.0
        return df

    @staticmethod
    def create(metric: Dict[str, float], metadata: pd.DataFrame, covariates: Tuple[str, ...],
               n_bootstrap: int, seed: int = 0) -> CharacteristicsAnalysis:

        df, coef_names, binary = _design(metric, metadata, covariates, "metric")
        formula = "metric ~ " + " + ".join(coef_names)

        def _fit(data: pd.DataFrame) -> Any:
            return smf.glm(formula, data=data, family=sm.families.Binomial()).fit()

        def _effects(fit: Any, data: pd.DataFrame) -> np.ndarray:
            return _ame(fit.predict, data, fit.params, coef_names, binary, lambda mu: mu * (1.0 - mu)) * 100.0

        def _boot_ame(data: pd.DataFrame) -> np.ndarray:
            fit = _fit(data)
            if not fit.converged:
                raise ValueError("not converged")
            return _effects(fit, data)

        fit = _fit(df)
        return CharacteristicsAnalysis(
            _bootstrap_table(df, binary, _boot_ame, _effects(fit, df), 0.0, coef_names, n_bootstrap, seed),
            _gvif(fit, coef_names))

    def plot(self, ax: Any) -> None:
        plot_effects(ax, self.coefficients, self.XLABEL)


class FARCharacteristicsAnalysis(Cached):

    # The rate counterpart of the fractional logit: a Poisson quasi-likelihood GLM (log link, offset log
    # monitored hours) of a participant's false-alarm count. Average marginal effects in false alarms per
    # hour; the bootstrap resamples clusters (the CV folds, whose members share a threshold).

    XLABEL = "Average marginal effect\n(false alarms/h)"

    def __init__(self, coefficients: pd.DataFrame, vif: Dict[str, float], n_participants: int) -> None:
        self.coefficients   = coefficients
        self.vif            = vif
        self.n_participants = n_participants

    @staticmethod
    def create(counts: Dict[str, float], hours: Dict[str, float], cluster_of: Dict[str, int], metadata: pd.DataFrame,
               covariates: Tuple[str, ...], n_bootstrap: int, seed: int = 0) -> FARCharacteristicsAnalysis:

        df, coef_names, binary = _design(counts, metadata, covariates, "alarms")
        df["log_hours"] = np.log([hours[p] for p in df["participant_id"]])
        formula = "alarms ~ " + " + ".join(coef_names)

        def _fit(data: pd.DataFrame) -> Any:
            return smf.glm(formula, data=data, family=sm.families.Poisson(), offset=data["log_hours"]).fit()

        def _effects(fit: Any, data: pd.DataFrame) -> np.ndarray:
            def per_hour(d: pd.DataFrame) -> pd.Series:
                return fit.predict(d, offset=np.zeros(len(d)))
            return _ame(per_hour, data, fit.params, coef_names, binary, lambda mu: mu)

        def _boot_ame(data: pd.DataFrame) -> np.ndarray:
            fit = _fit(data)
            if not fit.converged:
                raise ValueError("not converged")
            return _effects(fit, data)

        fit = _fit(df)
        return FARCharacteristicsAnalysis(
            _bootstrap_table(df, binary, _boot_ame, _effects(fit, df), 0.0, coef_names, n_bootstrap, seed,
                             cluster_of),
            _gvif(fit, coef_names), len(df))


def _ame(predict: Callable[[pd.DataFrame], pd.Series], data: pd.DataFrame, params: pd.Series, coef_names: List[str],
         binary: List[str], slope: Callable[[pd.Series], pd.Series]) -> np.ndarray:
    # A binary covariate: the mean predicted difference between 1 and 0. A continuous one: its coefficient
    # times the mean slope of the inverse link.
    w   = float(slope(predict(data)).mean())
    out = []
    for name in coef_names:
        if name in binary:
            d1 = data.copy(); d1[name] = 1.0
            d0 = data.copy(); d0[name] = 0.0
            out.append(float(predict(d1).mean() - predict(d0).mean()))
        else:
            out.append(w * float(params[name]))
    return np.array(out)


def _gvif(fit: Any, coef_names: List[str]) -> Dict[str, float]:
    cov = fit.cov_params().loc[coef_names, coef_names].values
    sd  = np.sqrt(np.diag(cov))
    vif = np.diag(np.linalg.inv(cov / np.outer(sd, sd)))
    return {c: vif[i] for i, c in enumerate(coef_names)}


def plot_effects(ax: Any, coef: pd.DataFrame, xlabel: str, rows: Optional[Sequence[str]] = None) -> None:
    # Effects significant after Bonferroni correction get a star. rows: the covariates on the y axis, in
    # order (default: coef's), so that panels can share rows.
    shared = rows is not None
    rows   = list(coef.index) if rows is None else list(rows)
    ys     = np.array([rows.index(c) for c in coef.index])
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.errorbar(coef["coef"], ys, xerr=[coef["coef"] - coef["ci_lo"], coef["ci_hi"] - coef["coef"]],
                fmt="o", capsize=3, markersize=5, elinewidth=0.8, capthick=0.8,
                color="firebrick", ecolor="firebrick")
    for i, (y, significant) in enumerate(zip(ys, coef["p_bonf"] < 0.05)):
        if significant:
            ax.plot(coef["coef"].iloc[i], y, "*", color="gold", ms=7, zorder=5)
    if shared:
        ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([CharacteristicsAnalysis.LABELS.get(c, c) for c in rows], fontsize=plotting.FONT_TICK)
    ax.set_xlabel(xlabel, fontsize=plotting.FONT_AXIS, labelpad=10)
    ax.tick_params(axis="x", labelsize=plotting.FONT_TICK)
    ax.grid(True, alpha=0.3, linewidth=0.4)
