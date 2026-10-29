from __future__ import annotations
from typing import List, Tuple

import numpy as np

# Per-sample label contract: +1 the event to detect, 0 monitored negative time, -1 gapped (a masked
# artifact; scored as negative time).
POSITIVE = 1
MONITOR  = 0
GAPPED   = -1


def positive(a: np.ndarray) -> np.ndarray:
    return np.asarray(a) == POSITIVE


def has_event(a: np.ndarray) -> bool:
    return bool(positive(a).any())


def regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    # Contiguous True runs as [start, end) index pairs.
    if len(mask) == 0:
        return []
    d = np.diff(np.concatenate([[0], np.asarray(mask, dtype=bool).view(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
