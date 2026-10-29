from __future__ import annotations
import math
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import numba
import numpy as np
import scipy.signal
import scipy.stats

from . import dataset, labels
from .dataset import Segment

# Raw signals -> 37 features per sample, then a causal per-participant EMA z-score.

EMA_ALPHA = 1.0 / 1200.0           # per 1 s update: about 20 minutes of memory
EMA_BLOCK = int(dataset.FS)
CLIP      = 50.0


@numba.njit(cache=True)
def _back(i: int, d: int, available_history: bool) -> int:
    # The sample d back, for a change over d samples. Before d samples exist: the first sample with
    # available_history, otherwise none (-1: the change is 0).
    return i - d if i >= d else (0 if available_history else -1)


@numba.njit(cache=True)
def _compute_features_jit(
    ppg_dc:                   np.ndarray,
    ppg_ac:                   np.ndarray,
    acc:                      np.ndarray,
    acc_ac:                   np.ndarray,
    ppg_snr:                  np.ndarray,
    ppg_spectral_entropy:     np.ndarray,
    ppg_spectral_flux:        np.ndarray,
    ppg_moment_window:        int,
    ppg_zcr_window:           int,
    ppg_pp_amp_window:        int,
    ppg_signal_energy_window: int,
    acc_range_sum_window:     int,
    acc_moment_window:        int,
    fs:                       float,
    available_history:        bool,
) -> np.ndarray:

    T     = ppg_dc.shape[0]
    n_acc = acc.shape[1]

    vpg = np.zeros(T, dtype=np.float32)
    apg = np.zeros(T, dtype=np.float32)
    for i in range(1, T):
        vpg[i] = ppg_ac[i, 0] - ppg_ac[i - 1, 0]
    for i in range(1, T):
        apg[i] = vpg[i] - vpg[i - 1]

    ppg_var  = np.zeros(T, dtype=np.float32)
    ppg_skew = np.zeros(T, dtype=np.float32)
    ppg_kurt = np.zeros(T, dtype=np.float32)
    vpg_var  = np.zeros(T, dtype=np.float32)
    apg_var  = np.zeros(T, dtype=np.float32)

    for i in range(ppg_moment_window - 1, T):
        start = i - ppg_moment_window + 1
        n     = ppg_moment_window

        mean_p = 0.0
        for j in range(start, i + 1): mean_p += ppg_ac[j, 0]
        mean_p /= n
        m2p = 0.0
        m3p = 0.0
        m4p = 0.0
        for j in range(start, i + 1):
            d = ppg_ac[j, 0] - mean_p
            d2 = d * d
            m2p += d2
            m3p += d2 * d
            m4p += d2 * d2
        m2p /= n
        m3p /= n
        m4p /= n
        ppg_var[i] = m2p
        if m2p > 1e-24:
            std3         = m2p * math.sqrt(m2p)
            ppg_skew[i]  = m3p / std3
            ppg_kurt[i]  = m4p / (m2p * m2p) - 3.0

        mean_v = 0.0
        for j in range(start, i + 1): mean_v += vpg[j]
        mean_v /= n
        m2v = 0.0
        for j in range(start, i + 1):
            d = vpg[j] - mean_v
            m2v += d * d
        vpg_var[i] = m2v / n

        mean_a = 0.0
        for j in range(start, i + 1): mean_a += apg[j]
        mean_a /= n
        m2a = 0.0
        for j in range(start, i + 1):
            d = apg[j] - mean_a
            m2a += d * d
        apg_var[i] = m2a / n

    for i in range(ppg_moment_window - 1):
        ppg_var[i]  = ppg_var[ppg_moment_window - 1]
        ppg_skew[i] = ppg_skew[ppg_moment_window - 1]
        ppg_kurt[i] = ppg_kurt[ppg_moment_window - 1]
        vpg_var[i]  = vpg_var[ppg_moment_window - 1]
        apg_var[i]  = apg_var[ppg_moment_window - 1]

    ppg_zcr = np.zeros(T, dtype=np.float32)
    for i in range(ppg_zcr_window - 1, T):
        start = i - ppg_zcr_window + 1
        zcr   = 0.0
        for j in range(start, i):
            s1  = 1.0 if ppg_ac[j, 0] > 0.0 else (-1.0 if ppg_ac[j, 0] < 0.0 else 0.0)
            s2  = 1.0 if ppg_ac[j + 1, 0] > 0.0 else (-1.0 if ppg_ac[j + 1, 0] < 0.0 else 0.0)
            zcr += abs(s2 - s1)
        ppg_zcr[i] = zcr / (2.0 * (ppg_zcr_window - 1))
    for i in range(ppg_zcr_window - 1):
        ppg_zcr[i] = ppg_zcr[ppg_zcr_window - 1]

    ppg_pp_amp = np.zeros(T, dtype=np.float32)
    for i in range(ppg_pp_amp_window - 1, T):
        start = i - ppg_pp_amp_window + 1
        mn    = ppg_ac[start, 0]
        mx    = ppg_ac[start, 0]
        for j in range(start + 1, i + 1):
            v = ppg_ac[j, 0]
            if v < mn: mn = v
            if v > mx: mx = v
        ppg_pp_amp[i] = mx - mn
    for i in range(ppg_pp_amp_window - 1):
        ppg_pp_amp[i] = ppg_pp_amp[ppg_pp_amp_window - 1]

    ppg_signal_energy = np.zeros(T, dtype=np.float32)
    for i in range(ppg_signal_energy_window - 1, T):
        start  = i - ppg_signal_energy_window + 1
        energy = 0.0
        for j in range(start, i + 1):
            v = ppg_ac[j, 0]
            energy += v * v
        ppg_signal_energy[i] = math.log(energy + 1.0)
    for i in range(ppg_signal_energy_window - 1):
        ppg_signal_energy[i] = ppg_signal_energy[ppg_signal_energy_window - 1]

    acc_magnitude  = np.zeros(T, dtype=np.float32)
    jerk_magnitude = np.zeros(T, dtype=np.float32)
    for i in range(T):
        mag2 = 0.0
        for j in range(n_acc): mag2 += acc[i, j] ** 2
        acc_magnitude[i] = math.sqrt(mag2)
    for i in range(1, T):
        jerk2 = 0.0
        for j in range(n_acc):
            d = acc[i, j] - acc[i - 1, j]
            jerk2 += d * d
        jerk_magnitude[i] = math.sqrt(jerk2)
    jerk_magnitude[0] = jerk_magnitude[1]

    acc_range_sum = np.zeros(T, dtype=np.float32)
    for i in range(acc_range_sum_window - 1, T):
        start = i - acc_range_sum_window + 1
        rs    = 0.0
        for j in range(n_acc):
            mn = acc_ac[start, j]
            mx = acc_ac[start, j]
            for k in range(start + 1, i + 1):
                v = acc_ac[k, j]
                if v < mn: mn = v
                if v > mx: mx = v
            rs += mx - mn
        acc_range_sum[i] = rs
    for i in range(acc_range_sum_window - 1):
        acc_range_sum[i] = acc_range_sum[acc_range_sum_window - 1]

    acc_log_range_sum = np.zeros(T, dtype=np.float32)
    for i in range(T): acc_log_range_sum[i] = math.log(acc_range_sum[i] + 1.0)

    # Per-axis moments, combined across axes as an L2 norm.
    acc_var  = np.zeros(T, dtype=np.float32)
    acc_skew = np.zeros(T, dtype=np.float32)
    acc_kurt = np.zeros(T, dtype=np.float32)

    for i in range(acc_moment_window - 1, T):
        start     = i - acc_moment_window + 1
        n         = acc_moment_window
        var_sum2  = 0.0
        skew_sum2 = 0.0
        kurt_sum2 = 0.0
        for j in range(n_acc):
            mean_j = 0.0
            for k in range(start, i + 1): mean_j += acc_ac[k, j]
            mean_j /= n
            m2 = 0.0
            m3 = 0.0
            m4 = 0.0
            for k in range(start, i + 1):
                d = acc_ac[k, j] - mean_j
                d2 = d * d
                m2 += d2
                m3 += d2 * d
                m4 += d2 * d2
            var_j    = m2 / n
            var_sum2 += var_j * var_j
            if var_j > 1e-24:
                std3       = var_j * math.sqrt(var_j)
                skew_j     = (m3 / n) / std3
                kurt_j     = (m4 / n) / (var_j * var_j) - 3.0
                skew_sum2 += skew_j * skew_j
                kurt_sum2 += kurt_j * kurt_j
        acc_var[i]  = math.sqrt(var_sum2)
        acc_skew[i] = math.sqrt(skew_sum2)
        acc_kurt[i] = math.sqrt(kurt_sum2)

    for i in range(acc_moment_window - 1):
        acc_var[i]  = acc_var[acc_moment_window - 1]
        acc_skew[i] = acc_skew[acc_moment_window - 1]
        acc_kurt[i] = acc_kurt[acc_moment_window - 1]

    d10  = int(10  * fs)
    d30  = int(30  * fs)
    d60  = int(60  * fs)
    d120 = int(120 * fs)

    out = np.zeros((T, 37), dtype=np.float32)
    for i in range(T):
        b10, b30  = _back(i, d10, available_history), _back(i, d30, available_history)
        b60, b120 = _back(i, d60, available_history), _back(i, d120, available_history)
        mob_ppg    = math.sqrt(vpg_var[i] / (ppg_var[i] + 1e-12))
        out[i,  0] = ppg_dc[i, 0]
        out[i,  1] = ppg_dc[i, 0] - ppg_dc[i - 1, 0] if i > 0 else 0.0
        out[i,  2] = ppg_ac[i, 0]
        out[i,  3] = ppg_var[i]
        out[i,  4] = ppg_skew[i]
        out[i,  5] = ppg_kurt[i]
        out[i,  6] = math.sqrt(ppg_var[i]) / max(abs(ppg_dc[i, 0]), 1e-3)
        out[i,  7] = vpg[i]
        out[i,  8] = apg[i]
        out[i,  9] = mob_ppg
        out[i, 10] = math.sqrt(apg_var[i] * ppg_var[i]) / (vpg_var[i] + 1e-12)
        out[i, 11] = ppg_zcr[i]
        out[i, 12] = ppg_pp_amp[i]
        out[i, 13] = 0.0 if b10  < 0 else math.log(ppg_pp_amp[i] + 1e-6) - math.log(ppg_pp_amp[b10]  + 1e-6)
        out[i, 14] = 0.0 if b30  < 0 else math.log(ppg_pp_amp[i] + 1e-6) - math.log(ppg_pp_amp[b30]  + 1e-6)
        out[i, 15] = 0.0 if b60  < 0 else math.log(ppg_pp_amp[i] + 1e-6) - math.log(ppg_pp_amp[b60]  + 1e-6)
        out[i, 16] = 0.0 if b120 < 0 else math.log(ppg_pp_amp[i] + 1e-6) - math.log(ppg_pp_amp[b120] + 1e-6)
        out[i, 17] = ppg_signal_energy[i]
        out[i, 18] = ppg_snr[i]
        out[i, 19] = ppg_spectral_entropy[i]
        out[i, 20] = ppg_spectral_flux[i]
        out[i, 21] = acc[i, 0]
        out[i, 22] = acc[i, 1]
        out[i, 23] = acc[i, 2]
        out[i, 24] = acc_ac[i, 0]
        out[i, 25] = acc_ac[i, 1]
        out[i, 26] = acc_ac[i, 2]
        out[i, 27] = acc_log_range_sum[i]
        out[i, 28] = 0.0 if b10  < 0 else math.log(acc_range_sum[i] + 1e-6) - math.log(acc_range_sum[b10]  + 1e-6)
        out[i, 29] = 0.0 if b30  < 0 else math.log(acc_range_sum[i] + 1e-6) - math.log(acc_range_sum[b30]  + 1e-6)
        out[i, 30] = 0.0 if b60  < 0 else math.log(acc_range_sum[i] + 1e-6) - math.log(acc_range_sum[b60]  + 1e-6)
        out[i, 31] = 0.0 if b120 < 0 else math.log(acc_range_sum[i] + 1e-6) - math.log(acc_range_sum[b120] + 1e-6)
        out[i, 32] = acc_var[i]
        out[i, 33] = acc_skew[i]
        out[i, 34] = acc_kurt[i]
        out[i, 35] = acc_magnitude[i]
        out[i, 36] = jerk_magnitude[i]

    return out


def normalized_ppg(signals: np.ndarray) -> np.ndarray:
    # PPG divided by the ADC gain and LED power it was recorded with.
    return signals[:, 3:4] / np.clip(signals[:, 0:1] * signals[:, 1:2], 1e-8, None)


class FeatureExtractor:

    FS           = dataset.FS
    CUTOFF_HZ    = 0.5
    FILTER_ORDER = 3
    WINDOW_S     = 2                      # moment, zero-crossing, amplitude, energy and range windows
    FFT_WINDOW_S = 8
    CARDIAC_HZ   = (0.5, 4.0)                # the band of the cardiac SNR
    NOISE_HZ     = (0.5, 5.0)                # noise below and above these
    FFT_CHUNK    = 512                      # FFT windows per batch
    LOG_COLS     = [3, 12, 20, 32, 36]    # ppg_var, ppg_pp_amp, spectral flux, acc_var, jerk

    def __init__(self, steady_start: bool = False, available_history: bool = False) -> None:
        fs = self.FS
        self.steady_start      = steady_start
        self.available_history = available_history
        self.ppg_ac_filter = scipy.signal.butter(self.FILTER_ORDER, self.CUTOFF_HZ, btype="high", output="sos", fs=fs)
        self.ppg_dc_filter = scipy.signal.butter(self.FILTER_ORDER, self.CUTOFF_HZ, btype="low",  output="sos", fs=fs)
        self.acc_filter    = scipy.signal.butter(self.FILTER_ORDER, self.CUTOFF_HZ, btype="high", output="sos", fs=fs)
        self.window        = int(fs * self.WINDOW_S)
        self.fft_window    = int(fs * self.FFT_WINDOW_S)

    def _spectral(self, ppg_ac: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Cardiac-band SNR, spectral entropy and spectral flux over a trailing FFT window.
        fs     = self.FS
        T      = ppg_ac.shape[0]
        N      = self.fft_window
        freqs  = np.arange(N // 2 + 1) * fs / N
        cardiac_mask = (freqs >= self.CARDIAC_HZ[0]) & (freqs <= self.CARDIAC_HZ[1])
        noise_mask   = (freqs <= self.NOISE_HZ[0]) | (freqs >= self.NOISE_HZ[1])
        hann   = np.hanning(N).astype(np.float64)
        s2     = float(np.sum(hann ** 2))
        sig    = ppg_ac[:, 0].astype(np.float64)

        snr     = np.zeros(T, dtype=np.float32)
        entropy = np.zeros(T, dtype=np.float32)
        flux    = np.zeros(T, dtype=np.float32)

        n_wins     = T - N + 1
        prev_psd   = None
        chunk_size = self.FFT_CHUNK
        for chunk_start in range(0, n_wins, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_wins)
            wins  = np.lib.stride_tricks.sliding_window_view(sig, N)[chunk_start:chunk_end]
            wins  = (wins - wins.mean(axis=1, keepdims=True)) * hann[np.newaxis, :]
            psds  = np.abs(np.fft.rfft(wins, axis=1)) ** 2 / (fs * s2)
            psds[:, 1:-1] *= 2.0

            cardiac_sum = psds[:, cardiac_mask].sum(axis=1)
            noise_sum   = psds[:, noise_mask].sum(axis=1)
            snr_chunk   = cardiac_sum / (noise_sum + 1e-12)

            # A window without cardiac-band power (an all-zero span) gives 0/0; report 0, not NaN.
            p_chunk   = psds[:, cardiac_mask] / (cardiac_sum[:, np.newaxis] + 1e-12)
            ent_chunk = np.nan_to_num(scipy.stats.entropy(p_chunk.T), nan=0.0, posinf=0.0, neginf=0.0)

            flux_chunk = np.zeros(chunk_end - chunk_start, dtype=np.float64)
            if prev_psd is not None:
                flux_chunk[0] = np.mean(np.abs(psds[0] - prev_psd))
            if chunk_end - chunk_start > 1:
                flux_chunk[1:] = np.mean(np.abs(np.diff(psds, axis=0)), axis=1)
            prev_psd = psds[-1].copy()

            out_start, out_end = chunk_start + N - 1, chunk_end + N - 1
            snr[out_start:out_end]     = snr_chunk.astype(np.float32)
            entropy[out_start:out_end] = ent_chunk.astype(np.float32)
            flux[out_start:out_end]    = flux_chunk.astype(np.float32)

        snr[:N - 1]     = snr[N - 1]
        entropy[:N - 1] = entropy[N - 1]
        flux[:N - 1]    = flux[N - 1]
        return snr, entropy, flux

    def filter(self, sos: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Causal along axis 0. A filter at rest turns a segment's first sample into a step (ringing at the
        # segment start); steady_start sets the state as if the signal had been constant at that sample.
        if not self.steady_start or len(x) == 0:
            return scipy.signal.sosfilt(sos, x, axis=0).astype(np.float32)
        zi = scipy.signal.sosfilt_zi(sos)[:, :, np.newaxis] * np.asarray(x[0], np.float64)[np.newaxis, np.newaxis, :]
        return scipy.signal.sosfilt(sos, x, axis=0, zi=zi)[0].astype(np.float32)

    def extract(self, signals: np.ndarray) -> np.ndarray:
        ppg_norm = normalized_ppg(signals)
        ppg_ac   = self.filter(self.ppg_ac_filter, ppg_norm)
        ppg_dc   = self.filter(self.ppg_dc_filter, ppg_norm)
        acc      = signals[:, 4:].astype(np.float32)
        acc_ac   = self.filter(self.acc_filter, acc)
        snr, entropy, flux = self._spectral(ppg_ac)
        w = self.window
        feats = _compute_features_jit(ppg_dc, ppg_ac, acc, acc_ac, snr, entropy, flux,
                                      w, w, w, w, w, w, self.FS, self.available_history)
        for col in self.LOG_COLS:
            feats[:, col] = np.log(np.abs(feats[:, col]) + 1e-6).astype(np.float32)
        return feats


def normalize(segments: List[Segment], warm_start: bool = False
              ) -> List[Segment]:
    # Causal per-participant EMA z-score, updated each second from the past only (gapped samples left out).
    # warm_start: the k-th update has weight max(1/k, EMA_ALPHA), a running average until the EMA's memory.
    mu, var = None, None
    k = 1
    out = []
    for pid, si, sig, ann in sorted(segments, key=lambda s: s[1]):
        x      = sig.astype(np.float64)
        normed = np.empty_like(x, dtype=np.float32)
        units  = [slice(0, len(x))] if len(x) <= EMA_BLOCK else \
                 [slice(a, min(a + EMA_BLOCK, len(x))) for a in range(0, len(x), EMA_BLOCK)]
        for sl in units:
            xb   = x[sl]
            keep = np.asarray(ann[sl]) != labels.GAPPED
            ref  = xb[keep] if keep.any() else xb
            b_mu, b_var = ref.mean(axis=0), ref.var(axis=0)
            if mu is None:
                mu, var    = b_mu, b_var
                normed[sl] = ((xb - b_mu) / (np.sqrt(np.maximum(b_var, 0.0)) + 1e-8)).astype(np.float32)
            else:
                normed[sl] = ((xb - mu) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)).astype(np.float32)
                k     += 1
                alpha  = max(1.0 / k, EMA_ALPHA) if warm_start else EMA_ALPHA
                mu  = (1.0 - alpha) * mu  + alpha * b_mu
                var = (1.0 - alpha) * var + alpha * b_var
        out.append((pid, si, normed, ann))
    return out


def preprocess_participant(pid: str, raw: List[Segment], out_path: Path,
                           gapped: bool, warm_start: bool = False, steady_start: bool = False,
                           available_history: bool = False) -> Tuple[str, int]:

    fe     = FeatureExtractor(steady_start, available_history)
    feats  = [(p, si, fe.extract(sig), ann) for p, si, sig, ann in raw]
    # A non-finite feature would poison the participant's EMA state, and every later second with it.
    if not all(np.isfinite(f).all() for _, _, f, _ in feats):
        raise ValueError(f"{pid}: non-finite features; check the raw signals")
    normed = normalize(feats, warm_start)
    if gapped:
        # Gapped rows are 0 (the running mean; a zeroed raw signal would mimic an arrest's flatline) and
        # flagged on an extra 0/1 channel.
        out = []
        for p, si, sig, ann in normed:
            g   = np.asarray(ann) == labels.GAPPED
            sig = np.asarray(sig, dtype=np.float32)
            if g.any():
                sig = sig.copy()
                sig[g] = 0.0
            out.append((p, si, np.concatenate([sig, g.astype(np.float32)[:, None]], axis=1), ann))
        normed = out
    # Near-zero-variance EMA states can give huge z-scores that would overflow float16.
    normed = [(p, si, np.clip(sig, -CLIP, CLIP).astype(np.float16), ann) for p, si, sig, ann in normed]

    tmp = out_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(normed, f)
    os.replace(tmp, out_path)
    return pid, len(normed)
