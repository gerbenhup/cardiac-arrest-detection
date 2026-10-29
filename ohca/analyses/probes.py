from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from .. import dataset, labels, scoring
from ..cv import NestedCrossValidationManager, armed_from
from ..dataset import Segment
from ..models import RNNModel

# Gap-leakage probes on the fold models: the score over WINDOW_S after a synthetic gap minus without it.
# Arrest-free: gaps in monitored time EXCLUSION_MARGIN_S from any event or real gap. Arrest: a gap ending at
# an event onset with EVENT_CLEAN_BEFORE_S of monitored lead-in. Gaps are copies of real gaps; positions
# depend only on (seed, participant, segment) and the labels, so every weight is probed at the same places.

WINDOW_S             = 30.0
EXCLUSION_MARGIN_S   = 300.0
EVENT_CLEAN_BEFORE_S = 60.0
N_PROBES             = 10
N_EVENT_DRAWS        = 10
SEED                 = 0


def _gap_pool(data: List[Segment], max_len: float = np.inf
              ) -> List[Tuple[int, np.ndarray]]:
    # The real gaps (length, feature row) of at most max_len samples, to copy into the probes.
    pool = [(int(hi - lo), sig[lo].copy()) for _, _, sig, ann in data
            for lo, hi in labels.regions(np.asarray(ann) == labels.GAPPED) if hi - lo <= max_len]
    if not pool:
        raise ValueError("no real gaps to copy into the probes")
    return pool


def _armed_heads(manager: NestedCrossValidationManager, data: List[Segment]
                 ) -> List[int]:
    if not manager.armed:
        return [0] * len(data)
    return [int(h) for h in armed_from(data, int(round(manager.arming_s * dataset.FS)),
                                       int(round(manager.warmup_s * dataset.FS)))]


def _eligible_runs(ann: np.ndarray, margin: int) -> List[Tuple[int, int]]:
    n       = len(ann)
    blocked = np.zeros(n, dtype=bool)
    for lo, hi in labels.regions(np.asarray(ann) != labels.MONITOR):
        blocked[max(0, lo - margin):min(n, hi + margin)] = True
    return labels.regions(~blocked)


def _clean_lead_in_events(ann: np.ndarray, clean: int) -> List[Tuple[int, int]]:
    ann = np.asarray(ann)
    return [(lo, hi) for lo, hi in labels.regions(labels.positive(ann))
            if lo >= clean and (ann[lo - clean:lo] == labels.MONITOR).all()]


def arrest_free(manager: NestedCrossValidationManager) -> pd.DataFrame:

    fs, L  = dataset.FS, dataset.STEP
    margin = int(round(EXCLUSION_MARGIN_S * fs))
    window = int(round(WINDOW_S / scoring.STEP_S))
    pool   = _gap_pool(manager.load(manager.ids))
    rows: List[Dict[str, Any]] = []

    for r in tqdm(manager.results, desc="Gap probe, arrest-free context"):
        model = RNNModel.load(manager.fold_dir(r["fold"]) / "model")
        data = manager.load(manager.groups[r["fold"]])
        for (pid, si, sig, ann), head in zip(data, _armed_heads(manager, data)):
            runs     = [(max(lo, head), hi) for lo, hi in _eligible_runs(ann, margin) if hi > head]
            fittable = [(lo, hi) for lo, hi in runs if (hi - lo) >= min(d for d, _ in pool)]
            if not fittable:
                continue
            z_orig, _ = model.apply_segment(sig, ann)
            if len(z_orig) == 0:
                continue
            rng = np.random.default_rng([SEED, manager.ids.index(pid), si])
            for _ in range(N_PROBES):
                lo, hi     = fittable[rng.integers(len(fittable))]
                candidates = [(k, d, t) for k, (d, t) in enumerate(pool) if d <= (hi - lo)]
                if not candidates:
                    continue
                pool_index, dur, template = candidates[rng.integers(len(candidates))]
                start = lo + int(rng.integers(0, hi - lo - dur + 1))
                sig_g, ann_g = sig.copy(), ann.copy()
                sig_g[start:start + dur] = template
                ann_g[start:start + dur] = labels.GAPPED
                z_gap, _ = model.apply_segment(sig_g, ann_g)
                step0 = (start + dur) // L
                step1 = min(step0 + window, len(z_orig), len(z_gap))
                if step0 >= step1:
                    continue
                rows.append(dict(participant_id=pid, segment=si, start_s=start / fs, pool_index=pool_index,
                                 gap_duration_s=dur / fs, baseline_score=float(np.mean(z_orig[step0:step1])),
                                 gap_score=float(np.mean(z_gap[step0:step1]))))
        del model
        tf.keras.backend.clear_session()

    if not rows:
        raise ValueError("no probe positions in the held-out data")
    probes = pd.DataFrame(rows)
    probes["diff"] = probes["gap_score"] - probes["baseline_score"]
    return probes


def arrest(manager: NestedCrossValidationManager) -> pd.DataFrame:

    fs, L  = dataset.FS, dataset.STEP
    clean  = int(round(EVENT_CLEAN_BEFORE_S * fs))
    window = int(round(WINDOW_S / scoring.STEP_S))
    pool   = _gap_pool(manager.load(manager.ids), clean)
    rows: List[Dict[str, Any]] = []

    for r in tqdm(manager.results, desc="Gap probe, arrest context"):
        model = None
        data = manager.load(manager.groups[r["fold"]])
        for (pid, si, sig, ann), head in zip(data, _armed_heads(manager, data)):
            events = [(lo, hi) for lo, hi in _clean_lead_in_events(ann, clean) if lo - clean >= head]
            if not events:
                continue
            if model is None:
                model = RNNModel.load(manager.fold_dir(r["fold"]) / "model")
            z_orig, l_orig = model.apply_segment(sig, ann)
            if len(z_orig) == 0:
                continue
            rng = np.random.default_rng([SEED, manager.ids.index(pid), si])
            for lo, hi in events:
                step0    = lo // L
                step1    = min(step0 + window, len(z_orig))
                ev_steps = np.flatnonzero(np.asarray(l_orig)[step0:step1] == labels.POSITIVE) + step0
                if len(ev_steps) == 0:
                    continue
                gap_scores, durations = [], []
                for _ in range(N_EVENT_DRAWS):
                    dur, template = pool[rng.integers(len(pool))]
                    sig_g, ann_g  = sig.copy(), ann.copy()
                    sig_g[lo - dur:lo] = template
                    ann_g[lo - dur:lo] = labels.GAPPED
                    z_gap, _ = model.apply_segment(sig_g, ann_g)
                    gap_scores.append(float(np.mean(z_gap[ev_steps])))
                    durations.append(dur / fs)
                rows.append(dict(participant_id=pid, segment=si, onset_s=lo / fs, n_event_steps=len(ev_steps),
                                 mean_gap_duration_s=float(np.mean(durations)), probe="insertion",
                                 baseline_score=float(np.mean(z_orig[ev_steps])),
                                 gap_score=float(np.mean(gap_scores))))
        if model is not None:
            del model
            tf.keras.backend.clear_session()

    if not rows:
        raise ValueError("no probe positions in the held-out data")
    probes = pd.DataFrame(rows)
    probes["diff"] = probes["gap_score"] - probes["baseline_score"]
    return probes
