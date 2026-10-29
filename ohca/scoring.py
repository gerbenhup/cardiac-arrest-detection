from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np

from . import dataset, labels

# Event-level detection: per-second score -> threshold sweep -> min-duration debounce. An alarm is a
# true positive within ALARM_MARGIN seconds of the event onset.

STEP_S       = dataset.STEP / dataset.FS
ALARM_MARGIN = (-10.0, 30.0)
N_THRESHOLDS = 2000
PROB_GRID    = np.linspace(0.0, 1.0, 500)   # shared ROC false-positive-rate and PR recall axis
HOUR_S       = 3600.0
NO_WINDOW    = 1e18                          # detection-window bound of a segment without an event


@numba.njit(cache=True, parallel=True)
def _multi_debounce_sweep_jit(
    flat_post:   np.ndarray,
    offsets:     np.ndarray,
    has_event:   np.ndarray,
    event_start: np.ndarray,
    thresholds:  np.ndarray,
    delta_t:     float,
    wlo_off:     float,
    whi_off:     float,
    min_durs:    np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    # Threshold sweep (thresholds in parallel) with a min-duration debounce, for several lengths in one pass:
    # a run raises an alarm at each length it reaches, dated to its start. Sensitivity is NaN without events.
    n_thr = len(thresholds)
    n_d   = len(min_durs)
    sens  = np.zeros((n_d, n_thr))
    far   = np.zeros((n_d, n_thr))
    n_seg = len(offsets) - 1

    for ti in numba.prange(n_thr):
        thr = thresholds[ti]
        TP  = np.zeros(n_d)
        FN  = np.zeros(n_d)
        FP  = np.zeros(n_d)
        mon = 0.0

        for s in range(n_seg):
            lo = offsets[s]
            hi = offsets[s + 1]
            n  = hi - lo

            in_run  = False
            run_len = 0
            n_ons   = np.zeros(n_d, dtype=np.int64)
            found   = np.zeros(n_d, dtype=np.bool_)
            before  = np.zeros(n_d, dtype=np.int64)

            if has_event[s]:
                wlo = event_start[s] + wlo_off / delta_t
                whi = event_start[s] + whi_off / delta_t
            else:
                wlo = NO_WINDOW
                whi = -NO_WINDOW

            for k in range(n):
                v = flat_post[lo + k]
                if not in_run:
                    if v > thr:
                        in_run  = True
                        run_len = 1
                    else:
                        run_len = 0
                else:
                    if v > thr:
                        run_len += 1
                    else:
                        in_run  = False
                        run_len = 0

                if in_run:
                    for di in range(n_d):
                        if run_len == min_durs[di]:
                            onset_k    = k - min_durs[di] + 1
                            n_ons[di] += 1
                            if has_event[s]:
                                if wlo <= onset_k <= whi:
                                    found[di] = True
                                elif onset_k < wlo:
                                    before[di] += 1

            for di in range(n_d):
                if has_event[s]:
                    if found[di]:
                        TP[di] += 1
                        FP[di] += before[di]
                    else:
                        FN[di] += 1
                        FP[di] += n_ons[di]
                else:
                    FP[di] += n_ons[di]
            if has_event[s]:
                mon += event_start[s] * delta_t
            else:
                mon += n * delta_t

        for di in range(n_d):
            sens[di, ti] = TP[di] / (TP[di] + FN[di]) if (TP[di] + FN[di]) > 0 else np.nan
            far[di, ti]  = FP[di] / mon * HOUR_S if mon > 0 else 0.0

    return sens, far


@numba.njit(cache=True, parallel=True)
def _event_run_survival_jit(
    flat_scores: np.ndarray,
    offsets:     np.ndarray,
    event_start: np.ndarray,
    thresholds:  np.ndarray,
    delta_t:     float,
    wlo_off:     float,
    whi_off:     float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Per (event, threshold): the longest above-threshold run starting inside the detection window, and
    # whether the recording ended while it was still running (right-censored).
    n_events = len(offsets) - 1
    n_thr    = len(thresholds)
    run_len  = np.zeros((n_events, n_thr))
    censored = np.zeros((n_events, n_thr), dtype=np.bool_)

    for e in range(n_events):
        lo  = offsets[e]
        hi  = offsets[e + 1]
        n   = hi - lo
        wlo = event_start[e] + wlo_off / delta_t
        whi = event_start[e] + whi_off / delta_t

        for ti in numba.prange(n_thr):
            thr           = thresholds[ti]
            best_len      = 0.0
            best_censored = False
            in_run        = False
            run_start     = -1

            for k in range(n):
                above = flat_scores[lo + k] > thr
                if above and not in_run:
                    in_run    = True
                    run_start = k
                elif (not above) and in_run:
                    if run_start >= wlo and run_start <= whi:
                        length = float(k - run_start)
                        if length > best_len:
                            best_len      = length
                            best_censored = False
                    in_run = False

            if in_run and run_start >= wlo and run_start <= whi:
                length = float(n - run_start)
                if length > best_len:
                    best_len      = length
                    best_censored = True

            run_len[e, ti]  = best_len
            censored[e, ti] = best_censored

    return run_len, censored


def far_grid(far_min: float) -> np.ndarray:
    # Alarms per hour; an explicit 0 point so a zero-false-alarm operating point is representable.
    return np.concatenate([[0.0], np.geomspace(far_min, 100.0, 1000)])


def prepare(posts: List[np.ndarray], anns: List[np.ndarray]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Flattened scores, segment offsets, event flags/onsets and the thresholds: unique scores subsampled by
    # rank, so the extreme scores where FAR reaches zero keep the same resolution.
    lengths = np.array([len(p) for p in posts], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat    = np.concatenate(posts)
    has_ev  = np.array([labels.has_event(a) for a in anns], dtype=np.bool_)
    ev_st   = np.array([float(np.argmax(a)) if e else -1.0 for a, e in zip(anns, has_ev)], dtype=np.float64)
    uniq    = np.unique(flat)
    if len(uniq) > N_THRESHOLDS:
        thrs = uniq[np.round(np.linspace(0, len(uniq) - 1, N_THRESHOLDS)).astype(np.int64)][::-1]
    else:
        thrs = uniq[::-1]
    return flat, offsets, has_ev, ev_st, thrs


def multi_debounce_sweep(flat: np.ndarray, offsets: np.ndarray, has_ev: np.ndarray, ev_st: np.ndarray,
                         thrs: np.ndarray, min_durs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _multi_debounce_sweep_jit(flat, offsets, has_ev, ev_st, thrs, STEP_S, ALARM_MARGIN[0], ALARM_MARGIN[1],
                                     np.asarray(min_durs, dtype=np.int64))


def score_fold_prepared(flat: np.ndarray, offsets: np.ndarray, has_ev: np.ndarray, ev_st: np.ndarray,
                        thrs: np.ndarray, min_dur: int) -> Tuple[np.ndarray, np.ndarray]:
    sens, far = multi_debounce_sweep(flat, offsets, has_ev, ev_st, thrs, np.array([min_dur]))
    return sens[0], far[0]


def step_hold(grid: np.ndarray, sorted_x: np.ndarray, sorted_y: np.ndarray, default: float = 0.0) -> np.ndarray:
    # Right-continuous step lookup: at each grid point, the value at the largest x <= that point.
    idx = np.searchsorted(sorted_x, grid, side="right") - 1
    return np.where(idx >= 0, sorted_y[np.clip(idx, 0, len(sorted_y) - 1)], default)


def far_at(sens: np.ndarray, far: np.ndarray, target: float, far_grid: np.ndarray) -> float:
    # Smallest FAR on the monotone FROC envelope reaching `target` sensitivity; inf if never reached.
    order = np.argsort(far)
    froc  = step_hold(far_grid, far[order], np.maximum.accumulate(sens[order]))
    idx   = np.searchsorted(froc, target)
    return far_grid[idx] if idx < len(far_grid) else float("inf")


def macro_average(per_participant: Dict[str, np.ndarray], pids: List[str],
                  weights: Optional[np.ndarray] = None) -> np.ndarray:
    # NaN-aware weighted mean of per-participant curves. Equal weights give the participant (macro)
    # average; event-count weights give the pooled event-level (micro) estimate.
    mat = np.stack([per_participant[p] for p in pids])
    if weights is None:
        weights = np.ones(mat.shape[0], dtype=np.float64)
    w     = np.asarray(weights, dtype=np.float64).reshape((-1,) + (1,) * (mat.ndim - 1))
    valid = ~np.isnan(mat)
    num   = np.sum(np.where(valid, mat, 0.0) * w, axis=0)
    den   = np.sum(valid * w, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def event_onsets(anns: List[np.ndarray]) -> Tuple[List[int], np.ndarray]:
    # The segments with an event, and each one's onset (steps).
    ev_idx = [i for i in range(len(anns)) if labels.has_event(anns[i])]
    return ev_idx, np.array([float(np.argmax(anns[i])) for i in ev_idx], dtype=np.float64)


def event_room(posts: List[np.ndarray], anns: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    # Per event: the steps left after the earliest alarm onset its window allows, and after its onset.
    ev_idx, ev_starts = event_onsets(anns)
    lengths = np.array([len(posts[i]) for i in ev_idx], dtype=np.int64)
    return lengths - np.maximum(0.0, ev_starts + ALARM_MARGIN[0] / STEP_S), lengths - ev_starts


def event_run_survival(posts: List[np.ndarray], anns: List[np.ndarray], thrs: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    ev_idx, ev_starts = event_onsets(anns)
    if not ev_idx:
        return np.zeros((0, len(thrs))), np.zeros((0, len(thrs)), dtype=np.bool_)
    ev_posts   = [posts[i] for i in ev_idx]
    ev_offsets = np.concatenate([[0], np.cumsum(np.array([len(p) for p in ev_posts], dtype=np.int64))])
    return _event_run_survival_jit(np.concatenate(ev_posts), ev_offsets, ev_starts, thrs, STEP_S,
                                   ALARM_MARGIN[0], ALARM_MARGIN[1])


def km_survival(run_len: np.ndarray, censored: np.ndarray, d_arr: np.ndarray) -> np.ndarray:
    # Kaplan-Meier estimate of P(run length >= d) for all thresholds at once. run_len/censored:
    # (n_events, n_thr). Returns (n_thr, len(d_arr)).
    n_ev, n_thr = run_len.shape
    max_t   = int(d_arr.max())
    surv    = np.ones((n_thr, max_t + 1))
    at_risk = np.full(n_thr, float(n_ev))
    s       = np.ones(n_thr)
    for t in range(max_t):
        eq         = (run_len == t)
        deaths     = (eq & ~censored).sum(axis=0).astype(float)
        censored_t = (eq & censored).sum(axis=0).astype(float)
        s          = s * np.where(at_risk > 0, 1.0 - deaths / np.maximum(at_risk, 1e-12), 1.0)
        surv[:, t + 1] = s
        at_risk    = np.maximum(0.0, at_risk - deaths - censored_t)
    return surv[:, d_arr]


def km_reach(scaled_run: np.ndarray, censored: np.ndarray) -> np.ndarray:
    # Product-limit P(scaled run >= 1) per column of (n_events, n) runs: runs ending before 1 are misses,
    # censored ones leave the risk set (deaths before censorings at ties).
    order   = np.lexsort((censored, scaled_run), axis=0)
    run     = np.take_along_axis(scaled_run, order, axis=0)
    cens    = np.take_along_axis(censored, order, axis=0)
    at_risk = (run.shape[0] - np.arange(run.shape[0], dtype=np.float64)).reshape((-1,) + (1,) * (run.ndim - 1))
    return np.prod(np.where(~cens & (run < 1.0), 1.0 - 1.0 / at_risk, 1.0), axis=0)


def detection_counts(posts: List[np.ndarray], anns: List[np.ndarray], thrs: np.ndarray, min_dur: int
                     ) -> Dict[str, Any]:
    # One participant's counts per threshold at a debounce length: events, detected events, monitored
    # hours, false alarms, and per event the detection run in debounce lengths (and if censored).
    flat, offsets, has_ev, ev_st, _ = prepare(posts, anns)
    sens, far = score_fold_prepared(flat, offsets, has_ev, ev_st, thrs, min_dur)
    n_events  = int(has_ev.sum())
    hours     = float(np.where(has_ev, ev_st, np.diff(offsets)).sum()) * STEP_S / HOUR_S
    run_len, censored = event_run_survival(posts, anns, thrs)
    return dict(n_events=n_events, detected=np.nan_to_num(sens) * n_events, hours=hours, alarms=far * hours,
                scaled_run=run_len / min_dur, censored=censored)


def invert_at_targets(curve: np.ndarray, x_grid: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # Smallest x where a monotone curve reaches each target, NaN if never.
    idx = np.searchsorted(curve, targets)
    return np.where(idx < len(x_grid), x_grid[np.clip(idx, 0, len(x_grid) - 1)], np.nan)


def _ranked_counts(scores: np.ndarray, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    # Cumulative true and false positives at each distinct score, descending (tied scores are one
    # threshold), and the number of positives.
    order     = np.argsort(-np.asarray(scores), kind="stable")
    sorted_sc = np.asarray(scores)[order]
    sorted_lb = np.asarray(lab)[order] == 1
    last      = np.flatnonzero(np.append(sorted_sc[1:] != sorted_sc[:-1], True))
    return np.cumsum(sorted_lb)[last].astype(float), np.cumsum(~sorted_lb)[last].astype(float), int(sorted_lb.sum())


def pooled_roc(scores: np.ndarray, lab: np.ndarray, fpr_grid: np.ndarray) -> Tuple[np.ndarray, float]:
    # Segment-level ROC on fpr_grid. NaN for a participant without positives (no ranking evidence).
    cum_tp, cum_fp, n_pos = _ranked_counts(scores, lab)
    n_neg = len(lab) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full_like(fpr_grid, np.nan, dtype=float), np.nan
    tpr        = np.concatenate([[0.0], cum_tp / n_pos, [1.0]])
    fpr        = np.concatenate([[0.0], cum_fp / n_neg, [1.0]])
    tpr_interp = np.interp(fpr_grid, fpr, tpr)
    # np.interp takes the last of repeated x values; every ROC curve passes through (0, 0).
    if fpr_grid[0] == 0.0:
        tpr_interp[0] = 0.0
    return tpr_interp, float(np.trapz(tpr_interp, fpr_grid))


def pooled_prc(scores: np.ndarray, lab: np.ndarray, recall_grid: np.ndarray) -> Tuple[np.ndarray, float]:
    # Segment-level precision-recall on recall_grid, step-held rather than linearly interpolated:
    # interpolation in PR space is optimistically biased (Davis & Goadrich, 2006).
    cum_tp, cum_fp, n_pos = _ranked_counts(scores, lab)
    if n_pos == 0:
        return np.full_like(recall_grid, np.nan, dtype=float), np.nan
    recall    = np.concatenate([[0.0], cum_tp / n_pos])
    precision = np.concatenate([[1.0], cum_tp / (cum_tp + cum_fp)])
    prec_step = step_hold(recall_grid, recall, precision, default=1.0)
    return prec_step, float(np.trapz(prec_step, recall_grid))


def debounce_onsets(active: np.ndarray, min_dur: int) -> List[int]:
    # Alarm onsets at one threshold: min_dur consecutive active steps, dated to the run start (the
    # kernel's rule).
    onsets: List[int] = []
    run_len = 0
    for i in range(len(active)):
        run_len = run_len + 1 if active[i] else 0
        if run_len == min_dur:
            onsets.append(i - min_dur + 1)
    return onsets
