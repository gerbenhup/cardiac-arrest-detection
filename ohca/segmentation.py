from __future__ import annotations
from typing import List, NamedTuple, Tuple

import numpy as np

from . import labels

# A continuous recording plus an episode list becomes segments that start with event-free signal and
# either end at an event or contain none.


class Episode(NamedTuple):
    onset_s: float
    end_s:   float
    kind:    str


class SegmentationSpec(NamedTuple):
    positive_types:   Tuple[str, ...]   # episode kinds that are the event to detect
    post_event_s:     float             # excluded after each event
    min_event_s:      float             # shorter events are discarded
    min_lead_in_s:    float             # minimum event-free signal before an event
    anchor_match_s:   float             # an anchor without an event this close is unmatched
    anchor_exclude_s: float             # excluded around an unmatched anchor


def mask_from(episodes: List[Episode], kinds: Tuple[str, ...], n: int, fs: float) -> np.ndarray:
    # Sample i (t = i/fs) is covered when onset <= t <= end, inclusive at both ends.
    mask = np.zeros(n, dtype=bool)
    for ep in episodes:
        if ep.kind in kinds:
            lo = int(np.clip(np.ceil (ep.onset_s * fs),     0, n))
            hi = int(np.clip(np.floor(ep.end_s   * fs) + 1, 0, n))
            mask[lo:max(hi, lo)] = True
    return mask


def label(pos_mask: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    return np.where(pos_mask, labels.POSITIVE, np.where(gap_mask, labels.GAPPED, labels.MONITOR)).astype(np.int8)


def segment(n: int, fs: float, episodes: List[Episode], anchors: List[Episode],
            spec: SegmentationSpec) -> List[Tuple[int, int, np.ndarray]]:

    pos_mask = mask_from(episodes, spec.positive_types, n, fs)
    anc_mask = mask_from(anchors, tuple({e.kind for e in anchors}), n, fs)

    min_event   = int(spec.min_event_s      * fs)
    min_lead    = int(spec.min_lead_in_s    * fs)
    post_event  = int(spec.post_event_s     * fs)
    match_marg  = int(spec.anchor_match_s   * fs)
    anc_exclude = int(spec.anchor_exclude_s * fs)

    queue: List[Tuple[int, int]] = [(0, n)]
    final: List[Tuple[int, int]] = []
    while queue:
        lo, hi = queue.pop(0)
        if hi - lo <= 0:
            continue
        pos_r = [(a + lo, b + lo) for a, b in labels.regions(pos_mask[lo:hi])]

        # An anchor without a matching event: the participant reported an event that was not found,
        # so the surrounding time cannot be trusted as negative either.
        unmatched = None
        for a_s, a_e in [(a + lo, b + lo) for a, b in labels.regions(anc_mask[lo:hi])]:
            dist = min(max(0, c_s - a_e, a_s - c_e) for c_s, c_e in pos_r) if pos_r else np.inf
            if dist > match_marg:
                unmatched = (a_s, a_e)
                break
        if unmatched is not None:
            before, after = unmatched[0] - anc_exclude, unmatched[1] + anc_exclude
            if before > lo:
                queue.append((lo, before))
            if after < hi:
                queue.append((after, hi))
            continue

        # An event ending inside the span closes the segment; the rest resumes after post_event.
        if pos_r and pos_r[0][1] < hi:
            cut = pos_r[0][1]
            queue.append((lo, cut))
            if cut + post_event < hi:
                queue.append((cut + post_event, hi))
            continue

        if pos_r and (pos_r[0][1] - pos_r[0][0]) < min_event:
            queue.append((lo, pos_r[0][0]))
            continue

        if (pos_r[0][0] - lo if pos_r else hi - lo) <= min_lead:
            continue
        final.append((lo, hi))

    no_gaps = np.zeros(n, dtype=bool)
    return sorted(((lo, hi, label(pos_mask[lo:hi], no_gaps[lo:hi])) for lo, hi in final), key=lambda s: s[0])
