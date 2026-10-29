from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import resampling, scoring
from ..dataset import Segment, test_by_participant
from .cached import Cached

# Detection at each fold's validated operating point (threshold and debounce length) on its held-out
# participants: event-level sensitivity (Kaplan-Meier when censoring-corrected) and per-participant FAR,
# averaged per cohort; each cohort bootstrapped on its own. person_of pools a person's records first.

METHODS = ("censoring_naive", "censoring_corrected")


class OperatingPointAnalysis(Cached):

    def __init__(self, participants: pd.DataFrame, table: pd.DataFrame) -> None:
        self.participants = participants
        self.table        = table

    @staticmethod
    def _participant(segments: List[Segment], thr: float, d: int) -> Dict[str, Any]:
        c = scoring.detection_counts([s[2] for s in segments], [s[3] for s in segments], np.array([thr]), d)
        return dict(n_events=c["n_events"], detected=float(c["detected"][0]), hours=c["hours"],
                    alarms=float(c["alarms"][0]), scaled_run=c["scaled_run"][:, 0], censored=c["censored"][:, 0])

    @staticmethod
    def create(results: List[Dict[str, Any]], debounce: pd.DataFrame, cohort_of: Dict[str, str],
               n_bootstrap: int, person_of: Optional[Dict[str, str]] = None) -> OperatingPointAnalysis:

        rows = []
        for r in results:
            row = debounce.iloc[r["fold"]]
            for pid, segments in test_by_participant([r]).items():
                for method in METHODS:
                    if f"threshold_{method}" in row and not pd.isna(row[f"threshold_{method}"]):
                        rows.append(dict(participant_id=pid, cohort=cohort_of[pid], fold=r["fold"], method=method,
                                         threshold=float(row[f"threshold_{method}"]),
                                         debounce_s=int(row[f"debounce_{method}"]),
                                         **OperatingPointAnalysis._participant(
                                             segments, float(row[f"threshold_{method}"]),
                                             int(row[f"debounce_{method}"]))))
        grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for row in rows:
            key = ((person_of or {}).get(row["participant_id"], row["participant_id"]), row["method"], row["cohort"])
            if key not in grouped:
                grouped[key] = {**row, "participant_id": key[0]}
                continue
            g = grouped[key]
            for name in ("n_events", "detected", "hours", "alarms"):
                g[name] += row[name]
            for name in ("scaled_run", "censored"):
                g[name] = np.concatenate([g[name], row[name]])
        pp = pd.DataFrame(list(grouped.values()))

        def sensitivity(df: pd.DataFrame, method: str) -> float:
            if method == "censoring_corrected":
                return float(scoring.km_reach(np.concatenate(df["scaled_run"].values),
                                             np.concatenate(df["censored"].values)))
            return float(df["detected"].sum() / df["n_events"].sum())

        out = []
        for method, by_method in pp.groupby("method", sort=False):
            for cohort, df in by_method.groupby("cohort", sort=True):
                df      = df.reset_index(drop=True)
                metrics = {"far_mean":   lambda d: float((d["alarms"] / d["hours"]).mean()),
                           "far_pooled": lambda d: float(d["alarms"].sum() / d["hours"].sum())}
                if df["n_events"].sum() > 0:
                    metrics["sensitivity"] = lambda d, m=method: sensitivity(d[d["n_events"] > 0], m)
                idx = resampling.indices(len(df), n_bootstrap)
                for name, fn in metrics.items():
                    boot = np.array([fn(df.iloc[i]) for i in idx])
                    out.append(dict(method=method, cohort=cohort, metric=name, estimate=fn(df),
                                    ci_lo=float(np.nanpercentile(boot, 2.5)), ci_hi=float(np.nanpercentile(boot, 97.5)),
                                    n_participants=len(df), n_events=int(df["n_events"].sum()),
                                    hours=float(df["hours"].sum())))
        return OperatingPointAnalysis(pp.drop(columns=["scaled_run", "censored"]), pd.DataFrame(out))

    def counts(self, cohort: str, method: str = "censoring_naive"
               ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        # False alarms, monitored hours and test fold per participant (per person with person_of).
        df  = self.participants[(self.participants["cohort"] == cohort) & (self.participants["method"] == method)]
        ids = df["participant_id"]
        return (dict(zip(ids, np.round(df["alarms"]))), dict(zip(ids, df["hours"])),
                dict(zip(ids, df["fold"].astype(int))))
