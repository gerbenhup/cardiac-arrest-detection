from __future__ import annotations
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from joblib import Parallel, delayed

from ohca import dataset, environment, features, labels, plotting, segmentation
from ohca.analyses import summary_figure
from ohca.cv import SearchSpec
from ohca.pipeline import RunSpec, run_phases

BASE_DIR = Path(__file__).resolve().parent

Records       = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
ACC_LSB_PER_G = 512.0     # accelerometry counts per g in dataset.h5
G             = 9.80665   # m/s^2


class BecaS2Raw:

    # BECA-S2 (cathlab ICD testing and ablation), under raw_data/BECA-S2/:
    #   1_raw_and_alignment/data/records.xlsx, elan_side_selection.xlsx, segment_offsets.csv
    #   1_raw_and_alignment/data/<pid>/Elan_<L|R>/FT_PPG_*.txt, FT_ACC_*.txt   wrist device exports
    #   1_raw_and_alignment/data/<pid>/Cath/metadata.inf, data.txt, segment_timestamps_corrected.csv
    #   2_annotation_and_packaging/annotations/gerben/<pid>.json                annotated episodes
    #   records.xlsx                                                            participant characteristics
    # Accelerometry is 128 Hz at 256 lsb/g, decimated to 32 Hz and scaled to 512 lsb/g.

    ACC_FS         = 128.0
    ACC_GAIN       = 2.0
    ICD_TEST_TYPES = {"s-icd implantation", "dft test"}

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir  = raw_dir
        self.data_dir = raw_dir / "1_raw_and_alignment" / "data"
        self.ann_dir  = raw_dir / "2_annotation_and_packaging" / "annotations" / "gerben"

    @staticmethod
    def is_icd_test(procedure_type: Any) -> bool:
        # ICD testing induces ventricular fibrillation; ablation induces tachycardias that are often perfused.
        return str(procedure_type).strip().lower() in BecaS2Raw.ICD_TEST_TYPES

    @staticmethod
    def decimate_acc(axis: np.ndarray) -> np.ndarray:
        # Zero-phase, so the accelerometry stays sample-aligned with the PPG.
        return scipy.signal.decimate(axis, int(round(BecaS2Raw.ACC_FS / dataset.FS)), ftype="fir", zero_phase=True)

    @staticmethod
    def _read_table(path: Path, header_rows: int) -> np.ndarray:
        # The device occasionally truncates a row on write: rows with an unusual column count are dropped.
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[header_rows:]
        split = [line.split() for line in lines if line.strip()]
        counts: Dict[int, int] = {}
        for s in split:
            counts[len(s)] = counts.get(len(s), 0) + 1
        modal = max(counts, key=counts.get)
        kept  = [s for s in split if len(s) == modal]
        if len(kept) < len(split):
            environment.log(f"  {path.name}: dropped {len(split) - len(kept)} malformed row(s)")
        return np.array(kept, dtype=np.float64)

    def read_signals(self, pid: str, side: str) -> np.ndarray:
        pid_dir   = self.data_dir / pid
        ppg_files = sorted(pid_dir.glob(f"Elan_{side}/FT_PPG_*.txt"))
        acc_files = sorted(pid_dir.glob(f"Elan_{side}/FT_ACC_*.txt"))
        if not ppg_files or not acc_files:
            raise FileNotFoundError(f"{pid}: no Elan_{side} PPG/ACC export")
        return self.read_ft(ppg_files, acc_files)

    @staticmethod
    def read_ft(ppg_files: List[Path], acc_files: List[Path]) -> np.ndarray:
        # One recording (consecutive exports concatenated) as X [T, 7] at 32 Hz.
        p = np.concatenate([BecaS2Raw._read_table(f, 9)  for f in ppg_files], axis=0)
        a = np.concatenate([BecaS2Raw._read_table(f, 11) for f in acc_files], axis=0)

        n_per_row    = (p.shape[1] - 3) // 2
        gain, led    = np.repeat(p[:, 1], n_per_row), np.repeat(p[:, 2], n_per_row)
        ppg, ambient = p[:, 3::2].ravel(), p[:, 4::2].ravel()
        axes = [BecaS2Raw.decimate_acc(a[:, k::3].ravel() * BecaS2Raw.ACC_GAIN) for k in (1, 2, 3)]
        n    = min(len(ppg), len(gain), len(led), len(ambient), *[len(x) for x in axes])
        return np.column_stack([gain[:n], led[:n], ambient[:n], ppg[:n],
                                axes[0][:n], axes[1][:n], axes[2][:n]]).astype(np.float32)

    def raw_episodes(self, pid: str) -> List[Tuple[datetime.datetime, datetime.datetime, str]]:
        # (onset, end, kind) on the naive local clock; the kind is the annotation text.
        eps = json.loads((self.ann_dir / f"{pid}.json").read_text(encoding="utf-8"))["episodes"]
        return [(datetime.datetime.fromisoformat(e["onset"]).replace(tzinfo=None),
                 datetime.datetime.fromisoformat(e["end"]).replace(tzinfo=None),
                 str(e["text"]).strip().lower()) for e in eps]

    def read_episodes(self, pid: str, t0: datetime.datetime) -> List[segmentation.Episode]:
        return sorted((segmentation.Episode((a - t0).total_seconds(), (b - t0).total_seconds(), kind)
                       for a, b, kind in self.raw_episodes(pid)), key=lambda e: e.onset_s)

    def records(self) -> pd.DataFrame:
        r = pd.read_excel(self.data_dir / "records.xlsx")
        r["Subject ID"] = r["Subject ID"].astype(str)
        return r.set_index("Subject ID")

    def sides(self) -> Dict[str, str]:
        s = pd.read_excel(self.data_dir / "elan_side_selection.xlsx")
        return dict(zip(s["participant"].astype(str), s["side"].astype(str)))

    @staticmethod
    def t0_of(row: pd.Series) -> datetime.datetime:
        # The staff-logged measurement start, which the annotations are aligned to (the device clock is not).
        ms = row["Measurement start time"]
        return datetime.datetime.combine(pd.to_datetime(row["Procedure date & time"]).date(),
                                         ms if isinstance(ms, datetime.time) else pd.to_datetime(str(ms)).time())

    def cath_meta(self, pid: str) -> Tuple[Optional[float], Optional[datetime.datetime]]:
        # Sampling rate and start time from metadata.inf (None where missing).
        fs, start_time = None, None
        for line in (self.data_dir / pid / "Cath" / "metadata.inf").read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^Data Sampling Rate\s*=\s*(\d+)", line.strip())
            if m:
                fs = float(m.group(1))
            m = re.match(r"^Start Time\s*=\s*(.+)$", line.strip())
            if m:
                text = m.group(1).strip()
                for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
                    try:
                        start_time = datetime.datetime.strptime(text, fmt)
                        break
                    except ValueError:
                        continue
        return fs, start_time

    def cath_sampling_rate(self, pid: str) -> float:
        fs = self.cath_meta(pid)[0]
        if fs is None:
            raise ValueError(f"{pid}: no cath sampling rate")
        return fs

    def cath_offsets(self, pid: str) -> Dict[Tuple[int, int], float]:
        # Per cath segment (sample start, end index): seconds from cath time to wrist time.
        df = pd.read_csv(self.data_dir / "segment_offsets.csv", encoding="utf-8-sig", dtype={"participant": str},
                         float_precision="round_trip")
        df = df[df["participant"] == pid]
        return {(int(s), int(e)): float(o) for s, e, o in zip(df["s_idx"], df["e_idx"], df["offset_sec"])}

    @staticmethod
    def _cath_seconds_of_day(text: Optional[str]) -> Optional[float]:
        # HH:MM:SS:F or HH:MM:SS.F (tenths of a second); 'X' marks an unreadable row.
        if text is None or text.strip() in ("", "X"):
            return None
        parts = text.strip().replace(".", ":").split(":")
        if len(parts) < 3:
            return None
        tenths = int(parts[3]) / 10.0 if len(parts) >= 4 else 0.0
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + int(parts[2]) + tenths

    def read_cath_coverage(self, pid: str, t0: datetime.datetime) -> List[segmentation.Episode]:
        # Spans of cathlab ECG/ABP recording on the wrist timeline. Each cath segment has its own clock offset
        # (the clocks drift apart); the segment start, not the contiguous corrected start, keeps real gaps.
        meta_f = self.data_dir / pid / "Cath" / "metadata.inf"
        csv_f  = self.data_dir / pid / "Cath" / "segment_timestamps_corrected.csv"
        if not meta_f.exists() or not csv_f.exists() or csv_f.stat().st_size == 0:
            return []
        fs, start_time = self.cath_meta(pid)
        if fs is None or start_time is None:
            return []
        lines = csv_f.read_text(encoding="utf-8-sig").strip().splitlines()
        if len(lines) < 2:
            return []
        header  = lines[0].split(",")
        offsets = self.cath_offsets(pid)

        out, day_offset, prev_sec = [], 0, None
        for line in lines[1:]:
            d   = dict(zip(header, line.split(",")))
            key = (int(d["Sample start index"]), int(d["Sample end index"]))
            sec = self._cath_seconds_of_day(d["Segment start"])
            if sec is None:
                seg_start = datetime.datetime.strptime(d["Corrected start time"], "%Y-%m-%d %H:%M:%S.%f")
            else:
                if prev_sec is not None and sec < prev_sec - 60:
                    day_offset += 1
                prev_sec  = sec
                seg_start = (datetime.datetime.combine(start_time.date() + datetime.timedelta(days=day_offset),
                                                       datetime.time()) + datetime.timedelta(seconds=sec))
            if key not in offsets:
                raise ValueError(f"{pid}: no segment_offsets.csv row for cath samples {key}")
            onset_s = (seg_start - t0).total_seconds()
            end_s   = onset_s + max(key[1] - key[0], 0) / fs
            out.append(segmentation.Episode(onset_s + offsets[key], end_s + offsets[key], "ecg_covered"))
        return sorted(out, key=lambda e: e.onset_s)

    def read_participant(self, pid: str, row: pd.Series, side: str) -> Tuple[str, Records]:
        fs       = dataset.FS
        t0       = self.t0_of(row)
        signals  = self.read_signals(pid, side)
        icd_test = self.is_icd_test(row["Procedure type"])
        coverage = [] if icd_test else self.read_cath_coverage(pid, t0)
        t0_epoch = t0.replace(tzinfo=dataset.TIMEZONE).timestamp()
        return pid, [(signals[lo:hi], lab, t0_epoch + float(lo) / fs + np.arange(hi - lo) / fs)
                     for lo, hi, lab in segment_procedure(len(signals), fs, self.read_episodes(pid, t0),
                                                                        icd_test, coverage, pid)]

    def build(self, path: Path, workers: int) -> None:
        records = self.records()
        sides   = self.sides()
        pids    = [p.stem for p in sorted(self.ann_dir.glob("P*.json")) if p.stem in records.index and p.stem in sides]
        environment.log(f"Building {path} ...")
        dataset.H5Dataset.write(path, Parallel(n_jobs=workers, return_as="generator")(
            delayed(self.read_participant)(pid, records.loc[pid], sides[pid]) for pid in pids))

    def read_metadata(self, pids: List[str]) -> pd.DataFrame:
        # records.xlsx: sheets 'records' and 'wrist_side_selection' (the worn wrist and its circumference).
        path = self.raw_dir / "records.xlsx"
        df = pd.read_excel(path, sheet_name="records").rename(columns={
            "Subject ID":     "participant_id",
            "Sex [m/f]":      "sex",
            "Age [years]":    "age",
            "Skin tone":      "skin_tone",
            "BMI [kg/m2]":    "bmi",
            "Procedure type": "procedure_type",
        })
        df["participant_id"] = df["participant_id"].astype(str)
        df    = df.set_index("participant_id")
        wrist = pd.read_excel(path, sheet_name="wrist_side_selection")
        wrist = wrist.set_index(wrist["Subject ID"].astype(str))
        df["wrist_circumference_mm"] = wrist["Selected wrist circumference [mm]"]
        df = df[df.index.isin(pids)].copy()
        for c in ("age", "bmi", "wrist_circumference_mm"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["skin_tone"] = pd.Categorical(df["skin_tone"], categories=dataset.SKIN_TONE, ordered=True)
        df["icd_test"]  = df["procedure_type"].map(self.is_icd_test).astype(float)
        return df[["sex", "age", "skin_tone", "bmi", "wrist_circumference_mm", "icd_test"]]


# Segment cutting for the cathlab procedures:
# 1. Usable time is [end of pre-op, start of post-op); for ablation only where cathlab ECG/ABP covers it.
# 2. Touching loss-of-pulse rows form one event. An event shorter than MIN_LOP_S is cut out with
#    HARD_CUT_MARGIN_S around it; a real event ends its span, and HARD_CUT_MARGIN_S after it is cut.
# 3. CUT_KINDS are cut out with HARD_CUT_MARGIN_S around them; spans under MIN_SPAN_S are dropped.
# Shocks are gapped wherever they occur and never decide a cut.

POSITIVE_KIND     = "loss of pulse"
GAP_KIND          = "shock"
CONTIGUOUS_EPS_S  = 1e-3
MIN_SPAN_S        = 30.0
MIN_LOP_S         = 3.0
HARD_CUT_MARGIN_S = 10.0
CUT_KINDS         = ("cuff occlusion", "invalid signal")
TIME_EPS_S        = 1e-9   # tolerance of span-boundary comparisons
BASE_EPS_S        = 1e-6   # tolerance when finding a span's base interval


def _base_intervals(episodes: List[segmentation.Episode], t_end: float, icd_test: bool,
                    coverage: List[segmentation.Episode], pid: str) -> List[Tuple[float, float]]:
    pre  = [e.end_s   for e in episodes if e.kind == "pre-op"]
    post = [e.onset_s for e in episodes if e.kind == "post-op"]
    lo0  = max(pre)  if pre  else 0.0
    hi0  = min(post) if post else t_end
    if icd_test:
        candidates = [(lo0, hi0)]
    else:
        candidates = sorted((max(c.onset_s, lo0), min(c.end_s, hi0)) for c in coverage
                            if min(c.end_s, hi0) > max(c.onset_s, lo0))
    kept = [(a, b) for a, b in candidates if (b - a) >= MIN_SPAN_S]
    if len(kept) < len(candidates):
        environment.log(f"  {pid}: dropped {len(candidates) - len(kept)} base interval(s) "
                        f"< {MIN_SPAN_S}s")
    return kept


def _cuts(episodes: List[segmentation.Episode], icd_test: bool, base_intervals: List[Tuple[float, float]], pid: str
          ) -> Tuple[List[Tuple[float, float]], List[segmentation.Episode]]:
    # The final spans, and the clipped loss-of-pulse and shock episodes used for labelling.
    spans         = []
    lop_and_shock = []
    for base_lo, base_hi in base_intervals:
        ev = []
        for e in episodes:
            if e.kind not in (POSITIVE_KIND, GAP_KIND) or e.end_s <= base_lo or e.onset_s >= base_hi:
                continue
            a, b = max(e.onset_s, base_lo), min(e.end_s, base_hi)
            if not icd_test and (a > e.onset_s + TIME_EPS_S or b < e.end_s - TIME_EPS_S):
                environment.log(f"  {pid}: clipped {e.kind} [{e.onset_s:.2f},{e.end_s:.2f}] "
                                f"-> [{a:.2f},{b:.2f}] to ECG/ABP coverage")
            ev.append(segmentation.Episode(a, b, e.kind))
        ev.sort(key=lambda e: e.onset_s)
        lop_and_shock.extend(ev)

        runs: List[List[segmentation.Episode]] = []
        for e in ev:
            if runs and (e.onset_s - runs[-1][-1].end_s) < CONTIGUOUS_EPS_S:
                runs[-1].append(e)
            else:
                runs.append([e])

        excluded = []
        for run in runs:
            lop = [e for e in run if e.kind == POSITIVE_KIND]
            if not lop:
                continue
            lop_start, lop_end = lop[0].onset_s, lop[-1].end_s
            if (lop_end - lop_start) < MIN_LOP_S:
                excluded.append((max(base_lo, lop_start - HARD_CUT_MARGIN_S),
                                 min(base_hi, lop_end + HARD_CUT_MARGIN_S)))
            elif lop_end < base_hi - TIME_EPS_S:
                excluded.append((lop_end, min(base_hi, lop_end + HARD_CUT_MARGIN_S)))
        for e in episodes:
            if e.kind in CUT_KINDS and e.end_s > base_lo and e.onset_s < base_hi:
                excluded.append((max(base_lo, max(e.onset_s, base_lo) - HARD_CUT_MARGIN_S),
                                 min(base_hi, min(e.end_s, base_hi) + HARD_CUT_MARGIN_S)))

        merged: List[Tuple[float, float]] = []
        for a, b in sorted(excluded):
            if merged and a <= merged[-1][1] + TIME_EPS_S:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        cur, kept = base_lo, []
        for a, b in merged:
            if a > cur:
                kept.append((cur, a))
            cur = max(cur, b)
        if cur < base_hi:
            kept.append((cur, base_hi))
        spans.extend((a, b) for a, b in kept if (b - a) >= MIN_SPAN_S)
    return spans, lop_and_shock


def segment_procedure(n: int, fs: float, episodes: List[segmentation.Episode], icd_test: bool,
                      coverage: List[segmentation.Episode], pid: str) -> List[Tuple[int, int, np.ndarray]]:

    base = _base_intervals(episodes, n / fs, icd_test, coverage, pid)
    spans, lop_and_shock = _cuts(episodes, icd_test, base, pid)
    masked   = [e for e in episodes if e.kind not in (POSITIVE_KIND, GAP_KIND)] + lop_and_shock
    pos_mask = segmentation.mask_from(masked, (POSITIVE_KIND,), n, fs)
    gap_mask = segmentation.mask_from(masked, (GAP_KIND,), n, fs)

    # A cut keeps the episode's last sample (mask_from covers up to floor(end * fs)) in its own span;
    # the edges of a base interval round up.
    def end_bound(t: float) -> int:
        return int(np.clip(np.floor(t * fs) + 1, 0, n))

    def ceil(t: float) -> int:
        return int(np.clip(np.ceil(t * fs), 0, n))

    out = []
    for start_s, end_s in spans:
        lo0, hi0 = next(((a, b) for a, b in base if a - BASE_EPS_S <= start_s and end_s <= b + BASE_EPS_S),
                        (start_s, end_s))
        lo = end_bound(start_s) if start_s > lo0 + TIME_EPS_S else ceil(start_s)
        hi = end_bound(end_s)   if end_s   < hi0 - TIME_EPS_S else ceil(end_s)
        if hi > lo:
            out.append((lo, hi, segmentation.label(pos_mask[lo:hi], gap_mask[lo:hi])))
    return out


# BECA-S6 (event-free daily-life wear, both wrists). Three sync-pulse sequences in the PPG delimit the
# protocol (exp1 scripted motion, exp2 free living, exp3 cuff occlusion); exp23 runs from the second
# sequence to the end of the third.

SYNC_PARTS = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
SYNC_PART_S = 0.5
N_SYNC      = 3


def sync_sequence(fs: float) -> np.ndarray:
    seq    = np.zeros(int(len(SYNC_PARTS) * SYNC_PART_S * fs), dtype=np.float32)
    bounds = np.round(np.linspace(0, len(seq), len(SYNC_PARTS) + 1)).astype(int)
    for lo, hi, v in zip(bounds[:-1], bounds[1:], SYNC_PARTS):
        seq[lo:hi] = v
    return seq


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()


def clean_ppg(ppg: np.ndarray, fs: float) -> np.ndarray:
    w        = int(fs) + (1 - int(fs) % 2)
    detrend  = ppg - _moving_average(ppg, w)
    envelope = _moving_average(np.abs(scipy.signal.hilbert(detrend)), w)
    return detrend / np.maximum(envelope, 1e-3)


def find_sync(ppg: np.ndarray, fs: float, method: str) -> np.ndarray:
    # Onsets of the N_SYNC strongest matches of the inverted sync sequence, at least 10 min apart, sorted.
    x, seq = clean_ppg(ppg.astype(np.float64), fs), sync_sequence(fs)
    if method == "normal_aligner":
        corr = scipy.signal.correlate(x, -seq, mode="valid")
    elif method == "derivative_aligner":
        corr = scipy.signal.correlate(np.diff(x), -np.diff(seq), mode="valid")
    elif method == "second_derivative_aligner":
        corr = scipy.signal.correlate(np.diff(x, 2), -np.diff(seq, 2), mode="valid")
    else:
        raise ValueError(f"unknown aligner {method}")
    peaks, _ = scipy.signal.find_peaks(corr, distance=10 * 60 * fs)
    if len(peaks) < N_SYNC:
        raise RuntimeError(f"found {len(peaks)} sync sequences, need {N_SYNC}")
    return np.sort(peaks[np.argsort(corr[peaks])][-N_SYNC:])


class BecaS6Raw:

    # Under raw_data/BECA-S6/: data/<pid><L|R>/ device exports, annotations/alignment/<pid>.json (aligner and
    # recording corrections), annotations/exp23/<pid><L|R>.json (seconds on the left file's exp23 grid),
    # beca-s6_records.xlsx.

    FS = dataset.FS

    def __init__(self, raw_dir: Path) -> None:
        self.dir = raw_dir

    def participants(self) -> List[str]:
        return sorted(p.stem for p in (self.dir / "annotations" / "alignment").glob("*.json"))

    def records(self) -> pd.DataFrame:
        r = pd.read_excel(self.dir / "beca-s6_records.xlsx", sheet_name="Data")
        r["participant_id"] = r["Participant ID"].map(lambda v: f"{int(v):03d}")
        return r.set_index("participant_id")

    def read_file(self, pid: str, side: str) -> Tuple[np.ndarray, float]:
        d     = self.dir / "data" / f"{pid}{side}"
        ppg   = sorted(d.glob("FT_PPG_*.txt"))
        start = None
        if not ppg:
            raise FileNotFoundError(f"{pid}{side}: no FT_PPG export")
        for line in ppg[0].read_text(errors="ignore").splitlines()[:12]:
            if "date" in line:
                stamp = line.replace("% date : ", "").replace("\ttime : ", " ").strip()
                start = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)
        if start is None:
            raise ValueError(f"{pid}{side}: no date in the FT_PPG header")
        return BecaS2Raw.read_ft(ppg, sorted(d.glob("FT_ACC_*.txt"))), start.timestamp()

    @staticmethod
    def windows(sync_l: np.ndarray, sync_r: np.ndarray, seq_len: int, exps_switched: bool
                ) -> Dict[str, Tuple[int, int]]:
        # exp23 window per file, [second, third sync + sequence), the left file's ends adjusted to an even
        # length; exps_switched: the experiments ran in another order.
        a, b, c = (int(v) for v in sync_l)
        if (b + seq_len - a) % 2:
            a += 1
        if (c + seq_len - b) % 2:
            c -= 1
        ra, rb, rc = (int(v) for v in sync_r)
        if exps_switched:
            return {"L": (a, b + seq_len), "R": (ra, rb + seq_len)}
        return {"L": (b, c + seq_len), "R": (rb, rc + seq_len)}

    @staticmethod
    def to_file(t_s: np.ndarray, window: Tuple[int, int], ref_len: int, fs: float, ref: bool) -> np.ndarray:
        # Annotation seconds (left-file grid of ref_len samples) to sample indices of a file window; another
        # file's window is mapped linearly onto that grid between its own sync pulses.
        lo, hi = window
        scale  = 1.0 if ref else (hi - lo) / (ref_len - 1)
        return lo + np.asarray(t_s, dtype=np.float64) * fs * scale


class BecaS2S6Dataset(dataset.H5Dataset):

    # BECA-S2 participants (segments copied from raw_data/BECA-S2/dataset.h5, built from the raw data first)
    # and BECA-S6 wrists `S6-<pid><L|R>` of person `S6-<pid>`, each on its own sample grid. Kept of BECA-S6:
    # exp23 minus the excluded head (at least SETTLE_S after the sync sequence, while the watches are put
    # on) and tail, minus the cuff inflation on both wrists.

    S2_COHORT = "BECA-S2"
    S6_COHORT = "BECA-S6"
    S6_PREFIX = "S6-"
    FOLD_SEED = 0
    SETTLE_S  = 600.0

    def s2_participants(self, workers: int) -> Iterator[Tuple[str, Records]]:
        path = self.base_dir / "raw_data" / "BECA-S2" / "dataset.h5"
        if not path.exists():
            BecaS2Raw(path.parent).build(path, workers)
        with h5py.File(path, "r") as hf:
            for pid in sorted(hf.keys()):
                yield pid, [(hf[pid][s]["X"][:], hf[pid][s]["ANN"][:], hf[pid][s]["time"][:]) for s in sorted(hf[pid])]

    def s6_wrists(self, raw: BecaS6Raw, pid: str) -> Iterator[Tuple[str, Records]]:
        fs      = BecaS6Raw.FS
        align   = json.loads((raw.dir / "annotations" / "alignment" / f"{pid}.json").read_text())
        ann     = {s: json.loads((raw.dir / "annotations" / "exp23" / f"{pid}{s}.json").read_text()) for s in "LR"}
        files   = {s: raw.read_file(pid, s) for s in "LR"}
        sync    = {s: find_sync(files[s][0][:, 3], fs, align["aligner"]) for s in "LR"}
        seq_len = len(sync_sequence(fs))
        win     = raw.windows(sync["L"], sync["R"], seq_len, align["correction"] == "exps_switched")
        ref_len = win["L"][1] - win["L"][0]
        # The wrist each file was on during exp23.
        file_of = {"L": "R", "R": "L"} if align["correction"] == "watches_switched_after_exp1" else {"L": "L", "R": "R"}
        cuff    = [iv for s in "LR" for iv in ann[s].get("cuff_inflation", [])]
        for side in "LR":
            f       = file_of[side]
            lo, hi  = win[f]
            signals = files[f][0][lo:hi].copy()
            if align["correction"] == "watch_R_upside_down_in_exp23" and side == "R":
                signals[:, 4:6] *= -1.0
            n        = len(signals)

            def to_window(iv: List[float]) -> np.ndarray:
                # Annotation seconds to seconds into this wrist's window.
                return (raw.to_file(np.asarray(iv, float), win[f], ref_len, fs, f == "L") - lo) / fs

            exclude  = [to_window(iv) for iv in ann[side]["exclude"]]
            invalid  = labels.regions((signals[:, 0] <= 0) | (signals[:, 1] <= 0))
            head     = max(float(exclude[0][1]), seq_len / fs + self.SETTLE_S)
            episodes = ([segmentation.Episode(0.0, head, "pre-op"),
                         segmentation.Episode(float(exclude[1][0]), n / fs, "post-op")]
                        + [segmentation.Episode(float(a), float(b), "cuff occlusion")
                           for a, b in (to_window(iv) for iv in cuff)]
                        + [segmentation.Episode(a / fs, (b - 1) / fs, "invalid signal") for a, b in invalid])
            spans    = segment_procedure(n, fs, episodes, True, [], f"{pid}{side}")
            t0       = files[f][1] + lo / fs
            yield (self.S6_PREFIX + pid + side,
                   [(signals[a:b], lab, t0 + (a + np.arange(b - a)) / fs) for a, b, lab in spans])

    def build(self, path: Path, workers: int) -> None:
        environment.log(f"Building {path} ...")
        raw = BecaS6Raw(self.base_dir / "raw_data" / "BECA-S6")

        def participants() -> Iterator[Tuple[str, Records]]:
            yield from self.s2_participants(workers)
            for pid in raw.participants():
                yield from self.s6_wrists(raw, pid)

        dataset.H5Dataset.write(path, participants())

    def cohort(self, pid: str) -> str:
        return self.S6_COHORT if pid.startswith(self.S6_PREFIX) else self.S2_COHORT

    def person(self, pid: str) -> str:
        return pid[:-1] if pid.startswith(self.S6_PREFIX) else pid

    def fold_groups(self) -> List[List[str]]:
        # One group per BECA-S2 participant with an event; the BECA-S6 persons (both wrists together) and
        # the event-free BECA-S2 participants are each shuffled once and dealt over the groups.
        events  = {p for p in self.participant_ids if self.cohort(p) == self.S2_COHORT and self.n_episodes[p] > 0}
        s2      = [p for p in self.participant_ids if self.cohort(p) == self.S2_COHORT and p in events]
        free    = [p for p in self.participant_ids if self.cohort(p) == self.S2_COHORT and p not in events]
        free    = [free[i] for i in np.random.default_rng(self.FOLD_SEED).permutation(len(free))]
        persons = sorted({self.person(p) for p in self.participant_ids if self.cohort(p) == self.S6_COHORT})
        persons = [persons[i] for i in np.random.default_rng(self.FOLD_SEED).permutation(len(persons))]
        wrists  = {q: [p for p in self.participant_ids if self.person(p) == q] for q in persons}
        return [[p] + [w for q in persons[k::len(s2)] for w in wrists[q]] + free[k::len(s2)] for k, p in enumerate(s2)]

    def read_metadata(self) -> pd.DataFrame:
        # BECA-S2 per participant; BECA-S6 per person, wrist circumference averaged over both wrists.
        s2 = BecaS2Raw(self.base_dir / "raw_data" / "BECA-S2").read_metadata(self.participant_ids)
        s2["cohort"] = self.S2_COHORT
        r       = BecaS6Raw(self.base_dir / "raw_data" / "BECA-S6").records()
        r.index = self.S6_PREFIX + r.index
        s6 = pd.DataFrame({
            "sex":                    r["Sex [M/F]"].astype(str),
            "age":                    pd.to_numeric(r["Age [years]"], errors="coerce"),
            "skin_tone":              r["Skin tone [type]"].astype(str),
            "bmi":                    r["Weight [kg]"] / (r["Height [cm]"] / 100.0) ** 2,
            "wrist_circumference_mm": (r["Wrist circumference L [mm]"] + r["Wrist circumference R [mm]"]) / 2.0,
            "cohort":                 self.S6_COHORT,
        }, index=r.index)
        persons = {self.person(p) for p in self.participant_ids}
        df = pd.concat([s2, s6])
        df = df[df.index.isin(persons)].copy()
        df["skin_tone"] = pd.Categorical(df["skin_tone"].astype(str), categories=dataset.SKIN_TONE, ordered=True)
        df.index.name = "participant_id"
        return df


# Example figure: cathlab ECG, PPG and accelerometry around one shock-induced loss of pulse: the longest
# shock-preceded event of the one ICD-test patient whose cathlab recording covers its own events.

EXAMPLE_PARTICIPANT   = "example_participant.txt"   # in the BECA-S2 raw data: a pseudonym stays with the data
EXAMPLE_ECG_CHANNEL   = 1      # lead II
EXAMPLE_ECG_SHIFT_S   = 3.0    # residual cath-to-wrist misalignment of this event, corrected by eye
EXAMPLE_ARTIFACT_S    = 0.5    # the visible shock artifact outlasts the annotated shock
EXAMPLE_X_LIM         = (-20.0, 15.0)
EXAMPLE_PPG_LEAD_IN_S = 120.0  # lets the high-pass filter settle before the window
EXAMPLE_HEIGHT_IN     = 3.445
EXAMPLE_GAP_V_IN      = 0.12
EXAMPLE_LW            = 0.6


def example_event_figure(spec: RunSpec, h5: dataset.H5Dataset) -> None:

    raw     = BecaS2Raw(spec.base_dir / "raw_data" / "BECA-S2")
    pid     = (raw.raw_dir / EXAMPLE_PARTICIPANT).read_text().strip()
    t0      = BecaS2Raw.t0_of(raw.records().loc[pid])
    eps     = sorted(raw.raw_episodes(pid))
    pairs   = [(prev, ep) for prev, ep in zip(eps, eps[1:])
               if ep[2] == POSITIVE_KIND and prev[2] == GAP_KIND]
    shock, lop = max(pairs, key=lambda p: p[1][1] - p[1][0])
    onset      = lop[0] + datetime.timedelta(seconds=EXAMPLE_ARTIFACT_S)
    win_start  = onset + datetime.timedelta(seconds=EXAMPLE_X_LIM[0] - 5.0)
    win_end    = onset + datetime.timedelta(seconds=EXAMPLE_X_LIM[1] + 5.0)

    def rel(t: datetime.datetime) -> float:
        return (t - onset).total_seconds()


    # Cathlab ECG: wrist time to cath time with the offset of the cath segment holding the event.
    segs = pd.read_csv(raw.data_dir / pid / "Cath" / "segment_timestamps_corrected.csv", encoding="utf-8-sig")
    segs["Corrected start time"] = pd.to_datetime(segs["Corrected start time"])
    segs["Corrected end time"]   = pd.to_datetime(segs["Corrected end time"])
    offsets = raw.cath_offsets(pid)
    offset  = next(off for (s, e), off in offsets.items()
                   if any((segs["Sample start index"] == s) & (segs["Sample end index"] == e)
                          & (segs["Corrected start time"] <= onset - datetime.timedelta(seconds=off))
                          & (onset - datetime.timedelta(seconds=off) <= segs["Corrected end time"])))
    cath_fs = raw.cath_sampling_rate(pid)

    def cath_index(t: datetime.datetime) -> int:
        t = t - datetime.timedelta(seconds=offset)
        r = segs[(segs["Corrected start time"] <= t) & (t <= segs["Corrected end time"])].iloc[0]
        frac = ((t - r["Corrected start time"]).total_seconds()
                / (r["Corrected end time"] - r["Corrected start time"]).total_seconds())
        return int(round(r["Sample start index"] + frac * (r["Sample end index"] - r["Sample start index"])))

    shift = datetime.timedelta(seconds=EXAMPLE_ECG_SHIFT_S)
    i0, i1 = cath_index(win_start - shift), cath_index(win_end + shift)
    ecg    = pd.read_csv(raw.data_dir / pid / "Cath" / "data.txt", sep=r"\s+", header=None, skiprows=i0,
                         nrows=i1 - i0, usecols=[EXAMPLE_ECG_CHANNEL]).values[:, 0]
    t_ecg  = (np.arange(i0, i1) - cath_index(onset)) / cath_fs - EXAMPLE_ECG_SHIFT_S

    # Wrist: the features' PPG normalisation and causal high-pass; accelerometry in m/s^2.
    fs      = dataset.FS
    sig     = raw.read_signals(pid, raw.sides()[pid])

    def index(t: datetime.datetime) -> int:
        return int(round((t - t0).total_seconds() * fs))

    j_buf   = index(win_start - datetime.timedelta(seconds=EXAMPLE_PPG_LEAD_IN_S))
    j0, j1  = index(win_start), index(win_end)
    ppg_raw = features.normalized_ppg(sig[j_buf:j1])[:, 0]
    ppg     = scipy.signal.sosfilt(features.FeatureExtractor().ppg_ac_filter, ppg_raw)[j0 - j_buf:]
    acc     = sig[j0:j1, 4:7] / ACC_LSB_PER_G * G
    t_wrist = (np.arange(j0, j1) - index(onset)) / fs

    # The study's figure width and the margins of the combined summary figure.
    width   = spec.figure_width_in
    panel_w = width - summary_figure.MARGIN_LEFT_IN - summary_figure.MARGIN_RIGHT_IN
    panel_h = (EXAMPLE_HEIGHT_IN - summary_figure.MARGIN_BOTTOM_IN - 2 * EXAMPLE_GAP_V_IN
               - summary_figure.MARGIN_TOP_IN) / 3
    height = summary_figure.MARGIN_BOTTOM_IN + 3 * panel_h + 2 * EXAMPLE_GAP_V_IN + summary_figure.MARGIN_TOP_IN
    fig     = plt.figure(figsize=(width, height), dpi=plotting.DPI)
    axes    = []
    for row in range(3):
        y = summary_figure.MARGIN_BOTTOM_IN + (2 - row) * (panel_h + EXAMPLE_GAP_V_IN)
        axes.append(fig.add_axes([summary_figure.MARGIN_LEFT_IN / width, y / height, panel_w / width, panel_h / height],
                                 sharex=axes[0] if axes else None))
    ax_ecg, ax_ppg, ax_acc = axes

    ax_ecg.plot(t_ecg,   ecg, color="tab:blue", lw=EXAMPLE_LW)
    ax_ppg.plot(t_wrist, ppg, color="tab:blue", lw=EXAMPLE_LW)
    for k in range(3):
        ax_acc.plot(t_wrist, acc[:, k], lw=EXAMPLE_LW)
    ax_ecg.set_ylim(-2, 2)
    ax_ppg.set_ylim(-5, 5)
    ax_acc.set_ylim(-13, 13)
    ax_acc.set_xlim(*EXAMPLE_X_LIM)
    ax_ecg.set_ylabel("ECG (mV)",                fontsize=plotting.FONT_AXIS)
    ax_ppg.set_ylabel("PPG (a.u.)",              fontsize=plotting.FONT_AXIS)
    ax_acc.set_ylabel("Acceleration\n(m/s$^2$)", fontsize=plotting.FONT_AXIS)
    ax_acc.set_xlabel("Time relative to cardiac arrest onset (s)", fontsize=plotting.FONT_AXIS)

    shock_span, lop_span = (rel(shock[0]), 0.0), (0.0, rel(lop[1]))
    for ax in axes:
        ax.axvspan(*shock_span, color="tab:green", alpha=0.20, lw=0)
        ax.axvspan(*lop_span,   color="tab:red",   alpha=0.20, lw=0)
        ax.tick_params(labelsize=plotting.FONT_TICK)
        ax.grid(True, alpha=0.3, linewidth=0.4)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    for text, (x0, x1), color in (("Shock", shock_span, "tab:green"),
                                  ("Loss of pulse", (lop_span[0], min(lop_span[1], EXAMPLE_X_LIM[1])), "tab:red")):
        ax_ecg.annotate(text, xy=((x0 + x1) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=plotting.FONT_AXIS,
                        color=color)
    fig.align_ylabels(axes)
    plotting.save(fig, spec.base_dir / "figures" / "example_event_signals")


BECA_S2_S6_SPEC = RunSpec(
    name                = "BECA-S2 + BECA-S6",
    base_dir            = BASE_DIR,
    dataset_cls         = BecaS2S6Dataset,
    # Event-free participants stay in: their monitored time is most of the false-alarm denominator.
    min_episodes        = 0,
    far_min             = 1e-2,
    # Most patients have one to three events, too few for per-participant sensitivity.
    micro_average       = True,
    covariates          = ("sex_m", "age_decades", "bmi", "skin_tone_num", "wrist_circumference_cm", "icd_test"),
    classifier_label    = "Loss-of-pulse classifier",
    panel_label_case    = "lower",
    # A penalized model is still at chance after one epoch, hence the longer rungs.
    search              = SearchSpec(screen_epochs=5, val_size=5),
    gapped              = True,
    time_of_day         = False,
    combined_figure     = True,
    gap_penalties       = (0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000),
    pp_workers          = 4,
    far_covariates      = ("sex_m", "age_decades", "bmi", "skin_tone_num", "wrist_circumference_cm"),
    warm_start          = True,
    steady_start        = True,
    available_history   = True,
    # Armed once the normaliser has 2 min of history and each segment 10 s (the longest feature window).
    arming_s            = 120.0,
    segment_warmup_s    = 10.0,
    crop_step_s         = 10,
    # Debounce lengths that at least half of the validation events last; false alarms from BECA-S6.
    debounce_evaluable  = 0.5,
    far_cohorts         = (BecaS2S6Dataset.S6_COHORT,),
    negative_sampling   = "time",
    config_weight       = 0,
    figure_width_in     = plotting.S2_WIDTH_IN,
)

if __name__ == "__main__":
    run_phases(BECA_S2_S6_SPEC, {"example": example_event_figure})
