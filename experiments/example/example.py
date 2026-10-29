from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ohca import dataset, environment, segmentation
from ohca.cv import SearchSpec
from ohca.pipeline import RunSpec, run_phases


class SyntheticDataset(dataset.H5Dataset):

    # A generated study in the dataset.h5 format, from a seeded random generator (no real recordings).
    # Per participant one recording: a pulse wave with respiratory modulation, LED power steps and motion
    # bursts; events where the pulse fades; shocks (gapped), some right before an event.

    N_PARTICIPANTS   = 8
    RECORDING_H      = 1.5
    EVENTS_PER_HOUR  = 4.0
    EVENT_S          = (10.0, 40.0)
    SHOCKED_FRACTION = 0.5
    START            = datetime.datetime(2026, 1, 5, 8, 0)
    SEGMENTATION     = segmentation.SegmentationSpec(
        positive_types   = ("loss of pulse",),
        post_event_s     = 5 * 60.0,
        min_event_s      = 5.0,
        min_lead_in_s    = 120.0,
        anchor_match_s   = 0.0,
        anchor_exclude_s = 0.0,
    )

    @staticmethod
    def _recording(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, List[segmentation.Episode]]:
        fs = dataset.FS
        t  = np.arange(n) / fs

        # Event onsets on a jittered grid, so events never overlap and keep room for a lead-in.
        n_events = max(1, int(round(SyntheticDataset.EVENTS_PER_HOUR * n / fs / 3600.0)))
        slot     = n / fs / n_events
        episodes: List[segmentation.Episode] = []
        pulse    = np.ones(n)
        for k in range(n_events):
            onset = k * slot + rng.uniform(0.4, 0.7) * slot
            dur   = rng.uniform(*SyntheticDataset.EVENT_S)
            episodes.append(segmentation.Episode(onset, onset + dur, "loss of pulse"))
            i0, i1 = int(onset * fs), int(min(onset + dur, n / fs) * fs)
            fade   = np.exp(-np.arange(i1 - i0) / (rng.uniform(1.0, 3.0) * fs))
            pulse[i0:i1] = fade
            if rng.random() < SyntheticDataset.SHOCKED_FRACTION:
                episodes.append(segmentation.Episode(onset - rng.uniform(0.5, 2.0), onset, "shock"))
        for _ in range(rng.integers(1, 3)):
            at = rng.uniform(0.1, 0.9) * n / fs
            if all(abs(at - e.onset_s) > 120.0 for e in episodes):
                episodes.append(segmentation.Episode(at, at + rng.uniform(0.5, 2.0), "shock"))

        # PPG: a pulse wave (fundamental plus a dicrotic harmonic) on a drifting baseline.
        hr      = rng.uniform(55, 85) + 6.0 * np.sin(2 * np.pi * t / rng.uniform(300, 900))
        phase   = 2 * np.pi * np.cumsum(hr / 60.0) / fs
        resp    = 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.2, 0.3) * t)
        wave    = np.sin(phase) + 0.35 * np.sin(2 * phase + 1.0)
        dc      = 2e5 * (1.0 + 0.05 * np.sin(2 * np.pi * t / rng.uniform(1800, 3600)))
        ppg     = dc * (1.0 + rng.uniform(0.005, 0.02) * resp * wave * pulse)

        # Accelerometry in counts at 512 lsb/g: gravity on a slowly turning wrist plus motion bursts.
        tilt    = 0.6 * np.sin(2 * np.pi * t / rng.uniform(600, 1800))
        acc     = 512.0 * np.column_stack([np.sin(tilt), np.zeros(n), np.cos(tilt)]) + rng.normal(0, 2.0, (n, 3))
        for _ in range(int(rng.poisson(6 * n / fs / 3600.0))):
            i0  = int(rng.uniform(0, n - 60 * fs))
            i1  = i0 + int(rng.uniform(5, 60) * fs)
            amp = rng.uniform(50, 300)
            acc[i0:i1] += rng.normal(0, amp, (i1 - i0, 3))
            ppg[i0:i1] += dc[i0:i1] * 0.02 * amp / 300.0 * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t[i0:i1])

        # The device's gain control steps the LED power; the features divide it out.
        led = np.full(n, 100.0)
        for i in np.sort(rng.integers(0, n, 3)):
            led[i:] = rng.choice([60.0, 80.0, 100.0, 120.0])
        ppg = ppg * led / 100.0 + rng.normal(0, 30.0, n)

        for e in episodes:
            if e.kind == "shock":
                i0, i1 = int(e.onset_s * fs), int(e.end_s * fs) + 1
                ppg[i0:i1]  = 4e5 * led[i0:i1] / 100.0
                acc[i0:i1] += rng.normal(0, 2000.0, (i1 - i0, 3))

        x = np.column_stack([np.ones(n), led, rng.normal(1000, 20, n), ppg, acc]).astype(np.float32)
        return x, sorted(episodes, key=lambda e: e.onset_s)

    @staticmethod
    def participant(index: int) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        fs          = dataset.FS
        rng         = np.random.default_rng(index)
        n           = int(SyntheticDataset.RECORDING_H * 3600 * fs)
        x, episodes = SyntheticDataset._recording(rng, n)
        # Recordings start at different hours, so the time-of-day analysis sees every block.
        start       = SyntheticDataset.START + datetime.timedelta(days=index, hours=3 * index)
        t           = start.replace(tzinfo=dataset.TIMEZONE).timestamp() + np.arange(n) / fs
        gap         = segmentation.mask_from(episodes, ("shock",), n, fs)
        pos         = segmentation.mask_from(episodes, SyntheticDataset.SEGMENTATION.positive_types, n, fs)
        # Segments are cut around the events; shocks are labelled gapped wherever they fall.
        return f"S{index + 1:03d}", [(x[lo:hi], segmentation.label(pos[lo:hi], gap[lo:hi]), t[lo:hi])
                                     for lo, hi, _ in segmentation.segment(n, fs, episodes, [],
                                                                           SyntheticDataset.SEGMENTATION)]

    def build(self, path: Path, workers: int) -> None:
        environment.log(f"Generating {path} ...")
        dataset.H5Dataset.write(path, (self.participant(i) for i in range(self.N_PARTICIPANTS)))
        rng = np.random.default_rng(self.N_PARTICIPANTS)
        pd.DataFrame({
            "participant_id": [f"S{i + 1:03d}" for i in range(self.N_PARTICIPANTS)],
            "sex":            rng.choice(["M", "F"], self.N_PARTICIPANTS),
            "age":            rng.uniform(40, 80, self.N_PARTICIPANTS).round(),
        }).to_excel(self.base_dir / "records.xlsx", index=False)

    def read_metadata(self) -> pd.DataFrame:
        df = pd.read_excel(self.base_dir / "records.xlsx").set_index("participant_id")
        return df[df.index.isin(self.participant_ids)]


EXAMPLE_SPEC = RunSpec(
    name                = "example",
    base_dir            = Path(__file__).resolve().parent,
    dataset_cls         = SyntheticDataset,
    min_episodes        = 1,
    far_min             = 1e-2,
    micro_average       = True,
    covariates          = ("sex_m", "age_decades"),
    classifier_label    = "Loss-of-pulse classifier",
    panel_label_case    = "lower",
    # A search small enough for a CPU: one screening rung of 2 candidates, 1 finalist, short epochs.
    search              = SearchSpec(screen_epochs=1, n_screen=2, n_finalists=1, rung_eta=2, max_epochs=2, patience=1,
                                     val_size=2, steps_per_epoch=5),
    gapped              = True,
    time_of_day         = True,
    combined_figure     = True,
    gap_penalties       = (0, 1000),
    pp_workers          = 4,
    n_bootstrap         = 200,
)

if __name__ == "__main__":
    run_phases(EXAMPLE_SPEC)
