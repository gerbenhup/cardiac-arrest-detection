from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import environment, labels, resampling, scoring
from .dataset import Segment

# Per fold, on its validation units: the debounce length minimising the bootstrap 97.5th percentile of the
# FAR needed for SEL_TARGET sensitivity, and the threshold reaching it. Censoring-naive counts an event cut
# off by the recording's end as a miss; censoring-corrected treats run lengths as right-censored.

SEL_TARGET    = 0.80
DEBOUNCE_GRID = list(range(1, 61))
N_BOOTSTRAP   = 1000
MIN_EVALUABLE = 1


def _by_unit(val_scores: List[Segment], unit_of: Optional[Dict[str, str]]
             ) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], List[str]]:
    by_pid: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for pid, _, z, lab in val_scores:
        by_pid.setdefault(pid if unit_of is None else unit_of[pid], []).append((z, lab))
    return by_pid, sorted(by_pid.keys())


def _viable(posts: List[np.ndarray], anns: List[np.ndarray], min_evaluable_fraction: float) -> List[int]:
    # Debounce lengths that at least MIN_EVALUABLE events have room for after the earliest onset, and that
    # min_evaluable_fraction of the events last (beyond that, Kaplan-Meier extrapolates a near-empty risk set).
    room, duration = scoring.event_room(posts, anns)
    if not len(room):
        return [DEBOUNCE_GRID[0]]
    viable = [d for d in DEBOUNCE_GRID if int((room >= d).sum()) >= MIN_EVALUABLE
              and float((duration >= d).mean()) >= min_evaluable_fraction]
    return viable if viable else [DEBOUNCE_GRID[0]]


def _candidates(val_scores: List[Segment], unit_of: Optional[Dict[str, str]],
                min_evaluable_fraction: float
                ) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]], List[str], List[int], np.ndarray,
                           np.ndarray]:
    # One threshold grid from the pooled validation data, so per-unit curves can be averaged.
    by_pid, pids = _by_unit(val_scores, unit_of)
    posts  = [z for p in pids for z, _ in by_pid[p]]
    anns   = [lab for p in pids for _, lab in by_pid[p]]
    viable = _viable(posts, anns, min_evaluable_fraction)
    thrs   = scoring.prepare(posts, anns)[4]
    return by_pid, pids, viable, thrs, np.array(viable, dtype=np.int64)


def _participant_curves(by_pid: Dict[str, List[Tuple[np.ndarray, np.ndarray]]], pids: List[str], thrs: np.ndarray,
                        d_arr: np.ndarray, far_by: Optional[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    # (sens, far), each (n_pid, n_debounce, n_thr); sensitivity is NaN without events. With far_by, a unit's
    # FAR comes from those segments (NaN for a unit without any).
    sens, far = [], []
    for pid in pids:
        flat, offsets, has_ev, ev_st, _ = scoring.prepare([z for z, _ in by_pid[pid]], [lab for _, lab in by_pid[pid]])
        s_p, f_p = scoring.multi_debounce_sweep(flat, offsets, has_ev, ev_st, thrs, d_arr)
        if far_by is not None:
            segs = far_by.get(pid, [])
            f_p  = (scoring.multi_debounce_sweep(*scoring.prepare([z for z, _ in segs], [lab for _, lab in segs])[:4],
                                                 thrs, d_arr)[1] if segs else np.full_like(f_p, np.nan))
        sens.append(s_p)
        far.append(f_p)
    return np.stack(sens), np.stack(far)


def _far_units(val_scores: List[Segment], unit_of: Optional[Dict[str, str]],
               far_pids: Optional[Set[str]]) -> Optional[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    return None if far_pids is None else _by_unit([s for s in val_scores if s[0] in far_pids], unit_of)[0]


def _macro(curves: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # NaN-aware weighted mean for many weightings at once: curves (n_pid, n_d, n_thr), weights
    # (n_rep, n_pid) holding each replicate's participant multiplicities.
    n_pid  = curves.shape[0]
    valid  = ~np.isnan(curves)
    num    = weights @ np.where(valid, curves, 0.0).reshape(n_pid, -1)
    den    = weights @ valid.reshape(n_pid, -1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out.reshape((weights.shape[0],) + curves.shape[1:])


def _select(sens: np.ndarray, far: np.ndarray, by_pid: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
            pids: List[str], viable: List[int], thrs: np.ndarray, seed: int, far_grid: np.ndarray, micro: bool
            ) -> Tuple[float, int, float]:

    idx_mat = resampling.indices(len(pids), N_BOOTSTRAP, seed)
    weights = np.zeros((N_BOOTSTRAP, len(pids)), dtype=np.float64)
    np.add.at(weights, (np.repeat(np.arange(N_BOOTSTRAP), len(pids)), idx_mat.ravel()), 1.0)
    unit_w  = np.ones(len(pids), dtype=np.float64)
    if micro:
        # Event-count weights: selection uses the same estimator as the reported results.
        unit_w = np.array([float(sum(1 for _, lab in by_pid[p] if labels.has_event(lab))) for p in pids],
                          dtype=np.float64)
        weights = weights * unit_w[None, :]
    macro_far = _macro(far, weights)
    macro_sen = _macro(sens, weights)

    boot_far = np.empty((N_BOOTSTRAP, len(viable)))
    for b in range(N_BOOTSTRAP):
        for di in range(len(viable)):
            boot_far[b, di] = scoring.far_at(macro_sen[b, di], macro_far[b, di], SEL_TARGET, far_grid)

    # An unreached target is +inf; replace it with a finite sentinel and take an observed order
    # statistic ('higher'), so interpolation can neither produce inf - inf nor blend in the sentinel.
    boot_far = np.where(np.isposinf(boot_far), np.finfo(np.float64).max, boot_far)
    with np.errstate(invalid="ignore"):
        upper = np.nanpercentile(boot_far, 97.5, axis=0, method="higher")
    try:
        best = int(np.nanargmin(upper))
    except ValueError:
        environment.log(f"  fold {seed}: no debounce length reaches the target; the first one is used")
        best = 0
    best_far = float(upper[best])
    full_sen = _macro(sens, unit_w[None, :])[0, best]
    full_far = _macro(far, unit_w[None, :])[0, best]
    reached  = np.flatnonzero(full_sen >= SEL_TARGET)
    thr_idx  = int(reached[np.argmin(full_far[reached])]) if reached.size else len(thrs) - 1
    return (float("inf") if best_far >= np.finfo(np.float64).max else best_far), int(viable[best]), float(thrs[thr_idx])


def censoring_naive(val_scores: List[Segment], seed: int, far_grid: np.ndarray,
                    micro: bool, unit_of: Optional[Dict[str, str]] = None,
                    min_evaluable_fraction: float = 0.0, far_pids: Optional[Set[str]] = None
                    ) -> Tuple[float, int, float]:
    by_pid, pids, viable, thrs, d_arr = _candidates(val_scores, unit_of, min_evaluable_fraction)
    sens, far = _participant_curves(by_pid, pids, thrs, d_arr, _far_units(val_scores, unit_of, far_pids))
    return _select(sens, far, by_pid, pids, viable, thrs, seed, far_grid, micro)


def censoring_corrected(val_scores: List[Segment], seed: int, far_grid: np.ndarray,
                        micro: bool, unit_of: Optional[Dict[str, str]] = None,
                        min_evaluable_fraction: float = 0.0, far_pids: Optional[Set[str]] = None
                        ) -> Tuple[float, int, float]:
    by_pid, pids, viable, thrs, d_arr = _candidates(val_scores, unit_of, min_evaluable_fraction)
    _, far = _participant_curves(by_pid, pids, thrs, d_arr, _far_units(val_scores, unit_of, far_pids))
    km = np.empty((len(pids), len(viable), len(thrs)))
    for i, pid in enumerate(pids):
        run_len, censored = scoring.event_run_survival([z for z, _ in by_pid[pid]], [lab for _, lab in by_pid[pid]],
                                                       thrs)
        km[i] = np.nan if run_len.shape[0] == 0 else scoring.km_survival(run_len, censored, d_arr).T
    return _select(km, far, by_pid, pids, viable, thrs, seed, far_grid, micro)
