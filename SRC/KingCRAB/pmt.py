"""Shared deterministic processing for oscilloscope PMT waveforms."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .numerics import trapezoid


@dataclass(frozen=True)
class PMTProcessingConfig:
    termination_ohm: float = 50.0
    pre_signal_s: float = 20e-9
    pulse_start_s: float = 20e-9
    threshold_sigma: float = 5.0
    filter_cutoff_hz: float | None = 1e9
    header_rows: int = 504


def processing_files(filepath, *, header_rows=504, delimiter=","):
    data = np.loadtxt(filepath, delimiter=delimiter, usecols=(0, 1), skiprows=header_rows)
    cuts = np.r_[0, np.flatnonzero(np.diff(data[:, 0]) <= 0) + 1, len(data)]
    return [(data[a:b, 0], data[a:b, 1]) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def load_waveforms(folder, pattern="C1C*.txt", *, header_rows=504):
    waveforms = []
    for path in sorted(Path(folder).glob(pattern)):
        waveforms.extend(processing_files(path, header_rows=header_rows))
    return waveforms


def normalize_waveform_times(waveforms):
    return [(t - t[0], v) for t, v in waveforms]


def subtract_baseline(waveforms, t_pre_signal=20e-9):
    corrected, baselines, sigmas = [], [], []
    for t, v in waveforms:
        mask = t < t_pre_signal
        if np.sum(mask) < 2:
            mask = np.arange(min(10, len(t)))
        baseline = np.mean(v[mask])
        sigma = np.std(v[mask], ddof=1) if np.sum(mask) > 1 else np.std(v[mask])
        corrected.append((t, v - baseline)); baselines.append(baseline); sigmas.append(sigma)
    return corrected, np.asarray(baselines), np.asarray(sigmas)


def lowpass_filter_waveform(t, v, f_cut):
    if len(t) < 2 or f_cut is None:
        return np.asarray(v).copy()
    freqs = np.fft.rfftfreq(len(v), d=np.mean(np.diff(t)))
    spectrum = np.fft.rfft(v); spectrum[freqs > f_cut] = 0.0
    return np.fft.irfft(spectrum, n=len(v))


def filter_waveforms(waveforms, f_cut=1e9):
    return [(t, lowpass_filter_waveform(t, v, f_cut)) for t, v in waveforms]


def extract_observables(waveforms, sigmas, pulse_start_time=20e-9, R=50.0):
    heights, charges = [], []
    for t, v in waveforms:
        heights.append(-np.min(v)); mask = t > pulse_start_time
        charges.append(trapezoid(-v[mask], t[mask]) / R if np.sum(mask) >= 2 else np.nan)
    return np.asarray(heights), np.asarray(charges), np.asarray(sigmas)


def integrate_pulse_region(t, v, R=50.0, pulse_start_time=20e-9):
    if len(t) < 3: return np.nan, np.nan, np.nan
    mask = t >= pulse_start_time
    if np.sum(mask) < 3: mask = np.ones_like(t, dtype=bool)
    candidates = np.where(mask)[0]; k = candidates[np.argmin(v[mask])]
    if v[k] >= 0: return 0.0, t[k], t[k]
    left = right = k
    while left > 0 and v[left] < 0: left -= 1
    while right < len(v)-1 and v[right] < 0: right += 1
    if right <= left: return np.nan, np.nan, np.nan
    return trapezoid(-v[left:right+1], t[left:right+1]) / R, t[left], t[right]


def waveform_duration(waveforms):
    return np.asarray([t[-1] - t[0] for t, _ in waveforms if len(t) > 1])


def average_waveform(waveforms, keep_mask=None):
    keep = np.ones(len(waveforms), bool) if keep_mask is None else np.asarray(keep_mask, bool)
    selected = [w for w, accepted in zip(waveforms, keep) if accepted]
    if not selected: return np.array([]), np.array([]), 0
    t_ref = selected[0][0]
    stack = [v if len(t)==len(t_ref) and np.allclose(t,t_ref) else np.interp(t_ref,t,v) for t,v in selected]
    return t_ref, np.nanmean(np.vstack(stack), axis=0), len(stack)


def estimate_charge_peak(charges, bins=120):
    q = np.asarray(charges); q = q[np.isfinite(q) & (q > 0)]
    if len(q) < 10: return np.nan, np.nan, 0, np.array([]), np.array([])
    lo, hi = np.nanpercentile(q, [1, 95]); q = q[(q >= lo) & (q <= hi)]
    if len(q) < 10 or hi <= lo: return np.nan, np.nan, 0, np.array([]), np.array([])
    counts, edges = np.histogram(q, bins=bins); peak = np.argmax(counts)
    selected = q[(q >= edges[max(peak-1,0)]) & (q <= edges[min(peak+2,len(edges)-1)])]
    mean = np.nanmean(selected); err = np.nanstd(selected, ddof=1)/np.sqrt(len(selected)) if len(selected)>1 else np.nan
    return mean, err, len(selected), counts, edges


def find_voltage_folder(base_folder, voltage):
    base = Path(base_folder)
    candidates = [base/str(voltage), base/f"{voltage}V", base/f"dark_{voltage}", base/f"Dark_{voltage}"]
    return next((p for p in candidates if p.is_dir()), base if base.is_dir() and any(base.glob("C1C*.txt")) else candidates[0])
