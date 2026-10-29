from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ohca import dataset, environment, segmentation
from ohca.cv import SearchSpec
from ohca.pipeline import RunSpec, run_phases


class BecaS1Dataset(dataset.H5Dataset):

    # Healthy participants wore the device for several days and repeatedly occluded the arm with a cuff.
    # Raw data, under raw_data/ (the participants are those with a consensus annotation):
    #   PDL/<pid>/FT_PPG_<YYYYmmdd-HHMMSS>.txt, FT_ACC_<...>.txt   device exports per recording
    #   Annotations/annotations/consensus/<pid>.json             consensus occlusions (epoch seconds)
    #   soft labels.xlsx                                         occlusions reported by participants

    SEGMENTATION = segmentation.SegmentationSpec(
        positive_types   = ("ohca",),
        post_event_s     = 15 * 60.0,
        min_event_s      = 10.0,
        min_lead_in_s    = 60.0,
        anchor_match_s   = 5 * 60.0,
        anchor_exclude_s = 15 * 60.0,
    )

    @staticmethod
    def _read_table(path: Path, header_rows: int) -> np.ndarray:
        return pd.read_csv(path, sep="\t", header=None, skiprows=header_rows, dtype=np.int64, engine="c").values

    @staticmethod
    def _read_ppg(path: Path) -> np.ndarray:
        # Per row of 32 samples: quality, 4 ADC gains, 4 LED powers (8 samples each), ambient, 32 PPG values.
        raw = BecaS1Dataset._read_table(path, 9)
        return np.stack([np.repeat(raw[:, 1:5].ravel(), 8), np.repeat(raw[:, 5:9].ravel(), 8),
                         np.repeat(raw[:, 9], 32), raw[:, 10:].ravel()], axis=-1)

    @staticmethod
    def _read_acc(path: Path) -> np.ndarray:
        # Per row of 32 samples: quality, then interleaved x, y, z.
        raw = BecaS1Dataset._read_table(path, 11)
        return np.stack([raw[:, 1::3].ravel(), raw[:, 2::3].ravel(), raw[:, 3::3].ravel()], axis=-1)

    @staticmethod
    def _episodes(starts: List[float], ends: List[float], t0: float, n: int, kind: str) -> List[segmentation.Episode]:
        # Epoch seconds -> seconds into this recording, clipped to it.
        fs, out = dataset.FS, []
        for s, e in zip(starts, ends):
            a, b = s - t0, e - t0
            if b > 0 and a < n / fs:
                out.append(segmentation.Episode(max(a, 0.0), min(b, n / fs), kind))
        return out

    @staticmethod
    def read_participant(pid: str, raw_dir: Path, anchors: pd.DataFrame
                         ) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:

        fs   = dataset.FS
        tz   = dataset.TIMEZONE
        rows = anchors.loc[anchors["participantID"] == pid]
        a_s  = [x.tz_localize(tz.key).timestamp() for x in rows["start"]]
        a_e  = [x.tz_localize(tz.key).timestamp() for x in rows["end"]]
        cons = json.loads((raw_dir / "Annotations" / "annotations" / "consensus" / f"{pid}.json").read_text()
                          )["annotations"]
        c_s  = [x["start"] for x in cons]
        c_e  = [x["end"]   for x in cons]

        segments = []
        for ppg_path in sorted((raw_dir / "PDL" / pid).glob("FT_PPG_*.txt")):
            acc_path = ppg_path.with_name(ppg_path.name.replace("FT_PPG", "FT_ACC"))
            if not acc_path.exists():
                continue
            # The file name holds the local (Europe/Amsterdam) recording start.
            t0  = datetime.datetime.strptime(ppg_path.stem.split("_")[2], "%Y%m%d-%H%M%S").replace(tzinfo=tz)
            t0  = t0.timestamp()
            ppg = BecaS1Dataset._read_ppg(ppg_path)
            acc = BecaS1Dataset._read_acc(acc_path)
            t   = np.linspace(t0, t0 + len(ppg) / fs, len(ppg), endpoint=False)
            n   = min(len(t), len(ppg), len(acc))
            x   = np.column_stack([ppg[:n], acc[:n]]).astype(np.float32)
            for lo, hi, lab in segmentation.segment(n, fs, BecaS1Dataset._episodes(c_s, c_e, t0, n, "ohca"),
                                                      BecaS1Dataset._episodes(a_s, a_e, t0, n, "anchor"),
                                                      BecaS1Dataset.SEGMENTATION):
                segments.append((x[lo:hi], lab, t[lo:hi]))
        segments.sort(key=lambda s: s[2][0])
        return pid, segments

    def build(self, path: Path, workers: int) -> None:
        raw_dir = self.base_dir / "raw_data"
        anchors = pd.read_excel(raw_dir / "soft labels.xlsx")
        pids    = sorted(f.stem for f in (raw_dir / "Annotations" / "annotations" / "consensus").glob("*.json"))
        environment.log(f"Building {path} ...")
        dataset.H5Dataset.write(path, Parallel(n_jobs=workers, return_as="generator")(
            delayed(BecaS1Dataset.read_participant)(pid, raw_dir, anchors)
            for pid in pids if (raw_dir / "PDL" / pid).exists()))

    def read_metadata(self) -> pd.DataFrame:
        df = pd.read_excel(self.base_dir / "records.xlsx").rename(columns={
            "Participant ID":           "participant_id",
            "Sex":                      "sex",
            "Birthdate":                "birthdate",
            "Skin tone":                "skin_tone",
            "Weight [kg]":              "weight_kg",
            "Height [cm]":              "height_cm",
            "Wrist circumference [mm]": "wrist_circumference_mm",
        }).set_index("participant_id")
        df = df[df.index.isin(self.participant_ids)].copy()
        df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
        # Age on the first day of recording.
        df["age"] = [(datetime.datetime.fromtimestamp(self.start_time[pid], tz=dataset.TIMEZONE).date()
                      - (b.date() if hasattr(b, "date") else b)).days / 365.25 for pid, b in df["birthdate"].items()]
        df["skin_tone"] = pd.Categorical(df["skin_tone"], categories=dataset.SKIN_TONE, ordered=True)
        return df[["sex", "age", "skin_tone", "bmi", "wrist_circumference_mm"]]


BECA_S1_SPEC = RunSpec(
    name                = "BECA-S1",
    base_dir            = Path(__file__).resolve().parent,
    dataset_cls         = BecaS1Dataset,
    min_episodes        = 5,
    far_min             = 1e-4,
    micro_average       = False,
    covariates          = ("sex_m", "age_decades", "bmi", "skin_tone_num", "wrist_circumference_cm"),
    classifier_label    = "Occlusion classifier",
    panel_label_case    = "upper",
    search              = SearchSpec(screen_epochs=1),
    gapped              = False,
    time_of_day         = True,
    combined_figure     = False,
)

if __name__ == "__main__":
    run_phases(BECA_S1_SPEC)
