from __future__ import annotations
from typing import Any, Callable, List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# The participant-cluster bootstrap behind every confidence interval.

WORKERS = 16   # threads per bootstrap


def indices(n_pids: int, n_boot: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_pids, size=(n_boot, n_pids))


def participants(pids: List[str], one_iter_fn: Callable[[List[str]], Any], n_boot: int, desc: str, seed: int = 0,
                 workers: int = WORKERS) -> List[Any]:
    idx_mat = indices(len(pids), n_boot, seed)
    return list(tqdm(Parallel(n_jobs=workers, prefer="threads", return_as="generator")(
                         delayed(one_iter_fn)([pids[j] for j in idx]) for idx in idx_mat),
                     total=n_boot, desc=desc, leave=False))
