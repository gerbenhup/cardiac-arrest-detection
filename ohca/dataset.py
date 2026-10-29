from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd

from . import environment, labels

FS        = 32.0                          # samples per second
STEP      = 32                            # samples per model step (1 s)
TIMEZONE  = ZoneInfo("Europe/Amsterdam")  # recording clock, for the time-of-day analysis
SKIN_TONE = ["I", "II", "III", "IV", "V", "VI"]

Segment = Tuple[str, int, np.ndarray, np.ndarray]   # (participant, segment index, signals or scores, labels)


def test_by_participant(results: List[Dict[str, Any]]) -> Dict[str, List[Segment]]:
    units: Dict[str, List[Segment]] = {}
    for r in results:
        for s in r["test_scores"]:
            units.setdefault(s[0], []).append(s)
    return units


def test_by_fold(results: List[Dict[str, Any]], keep: Optional[Callable[[str], bool]] = None
                 ) -> Dict[int, List[Segment]]:
    return {r["fold"]: [s for s in r["test_scores"] if keep is None or keep(s[0])] for r in results}


class H5Dataset:

    # Shared dataset.h5 schema, one group per participant and segment:
    #   /{pid}/{segment:05d}/X     float32 [T, 7]   ppg gain, led power, ambient, ppg, acc x, y, z
    #   /{pid}/{segment:05d}/ANN   int8    [T]      labels contract
    #   /{pid}/{segment:05d}/time  float64 [T]      epoch seconds
    # Accelerometry is in counts at 512 lsb/g. Subclasses build the file from raw data and read
    # records.xlsx into `metadata`, indexed by participant id.

    def __init__(self, base_dir: Path, min_episodes: int, workers: int) -> None:

        self.base_dir = Path(base_dir)
        self.path     = self.base_dir / "dataset.h5"
        if not self.path.exists():
            self.build(self.path, workers)

        with h5py.File(self.path, "r") as hf:
            pids            = sorted(hf.keys())
            self.n_episodes = {p: sum(labels.has_event(hf[p][s]["ANN"][:]) for s in hf[p]) for p in pids}
            self.start_time = {p: float(hf[p][min(hf[p].keys())]["time"][0]) for p in pids}

        self.participant_ids = [p for p in pids if self.n_episodes[p] >= min_episodes]
        self.excluded       = [p for p in pids if self.n_episodes[p] <  min_episodes]
        self.metadata       = self.read_metadata()

    def build(self, path: Path, workers: int) -> None:
        raise NotImplementedError

    def read_metadata(self) -> pd.DataFrame:
        raise NotImplementedError

    def fold_groups(self) -> List[List[str]]:
        # The participants held out together per CV fold; one each is leave-one-participant-out.
        return [[p] for p in self.participant_ids]

    def cohort(self, pid: str) -> str:
        return ""

    def person(self, pid: str) -> str:
        # The resampling unit: one person can contribute several records (both wrists).
        return pid

    @staticmethod
    def write(path: Path, participants: Iterable[Tuple[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]) -> None:
        # participants yields (pid, [(X, ANN, time), ...]); written via a temporary file.
        tmp = path.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as hf:
            for pid, segments in participants:
                if not segments:
                    environment.log(f"  {pid}: no usable segments, not included")
                    continue
                group = hf.create_group(pid)
                for si, (x, ann, t) in enumerate(segments):
                    sg = group.create_group(f"{si:05d}")
                    sg.create_dataset("X",    data=np.asarray(x,   dtype=np.float32), compression="lzf")
                    sg.create_dataset("ANN",  data=np.asarray(ann, dtype=np.int8),    compression="lzf")
                    sg.create_dataset("time", data=np.asarray(t,   dtype=np.float64), compression="lzf")
                environment.log(f"  {pid}: {len(segments)} segments, "
                                f"{sum(labels.has_event(a) for _, a, _ in segments)} with an event")
        os.replace(tmp, path)

    def get(self, pids: List[str]) -> List[Segment]:
        out = []
        with h5py.File(self.path, "r") as hf:
            for pid in pids:
                for si, s in enumerate(sorted(hf[pid].keys())):
                    out.append((pid, si, hf[pid][s]["X"][:], hf[pid][s]["ANN"][:]))
        return out

    def labels_and_times(self, pid: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        with h5py.File(self.path, "r") as hf:
            return [(hf[pid][s]["ANN"][:], hf[pid][s]["time"][:]) for s in sorted(hf[pid].keys())]
