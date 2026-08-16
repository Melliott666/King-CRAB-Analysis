"""Notebook-independent helpers extracted from PMT_Terminal_Noise_Comparison.ipynb."""
from __future__ import annotations
from pathlib import Path
import os, re, json, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
 from scipy.integrate import trapezoid, quad
 from scipy.optimize import curve_fit
 from scipy.signal import butter, filtfilt, welch
 from scipy.stats import norm, binom
except ImportError:
 pass


def read_waveforms(filepath, skip_rows=504):
    data = np.loadtxt(filepath, delimiter=",", usecols=(0, 1), skiprows=skip_rows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    split_points = np.where(np.diff(data[:, 0]) < 0)[0] + 1
    return [(part[:, 0], part[:, 1]) for part in np.split(data, split_points) if len(part) > 1]

def baseline_mask(t, mode, guard_s=5e-9):
    mask = np.abs(t) > guard_s if mode == "auto" else t < -guard_s
    if np.count_nonzero(mask) < 10:
        mask = np.arange(len(t)) < max(2, len(t)//5)
    return mask

def waveform_statistics(waveforms, mode):
    rows, corrected = [], []
    for t, v in waveforms:
        mask = baseline_mask(t, mode)
        baseline = np.mean(v[mask])
        sigma = np.std(v[mask], ddof=1)
        vc = v - baseline
        rows.append({
            "baseline_V": baseline,
            "rms_V": sigma,
            "peak_to_peak_V": np.ptp(vc),
            "minimum_corrected_V": np.min(vc),
            "absolute_threshold_crossing": int(np.min(vc) < ABSOLUTE_THRESHOLD_V),
            "normalized_threshold_crossing": int(np.min(vc) < -NORMALIZED_THRESHOLD_SIGMA*sigma),
        })
        corrected.append((t, vc))
    return rows, corrected

def process_empty_dataset(folder, pattern, mode):
    files = sorted(Path(folder).glob(pattern))
    rows, retained = [], []
    for filepath in files:
        file_rows, file_corrected = waveform_statistics(read_waveforms(filepath), mode)
        offset = len(rows)
        for i, row in enumerate(file_rows):
            row.update(waveform=offset+i, source_file=filepath.name)
        rows.extend(file_rows)
        retained.extend(file_corrected[:max(0, MAX_PLOT_WAVEFORMS-len(retained))])
    return files, rows, retained

def common_stack(waveforms):
    if not waveforms:
        return np.array([]), np.empty((0, 0))
    t_ref = waveforms[0][0]
    stack = []
    for t, v in waveforms:
        stack.append(v if len(t)==len(t_ref) and np.allclose(t,t_ref) else np.interp(t_ref,t,v))
    return t_ref, np.asarray(stack)

def median_psd(waveforms):
    t, stack = common_stack(waveforms)
    if len(t)<2 or not len(stack): return np.array([]), np.array([])
    dt=np.median(np.diff(t)); window=np.hanning(len(t))
    spectra=np.fft.rfft(stack*window,axis=1)
    psd=np.abs(spectra)**2*dt/np.sum(window**2)
    return np.fft.rfftfreq(len(t),dt),np.median(psd,axis=0)

def trigger_elapsed_time(filepath):
    stamps=[]
    with open(filepath,errors='replace') as stream:
        for _ in range(SKIP_ROWS-1):
            line=stream.readline().strip()
            m=HEADER_RE.match(line)
            if not m: continue
            frac=float(m.group(3).strip() or 0)
            # The third field is elapsed time from this file's first trigger.
            # The wall-clock field advances in parallel and must not be added.
            stamps.append(frac)
    if len(stamps)<2: return np.nan
    elapsed=stamps[-1]-stamps[0]
    return elapsed if elapsed>0 else np.nan

def trapezoid_integral(y, x):
    """Compatible with NumPy versions before and after trapz removal."""
    integration_function = getattr(np, "trapezoid", None)
    if integration_function is None:
        integration_function = np.trapz
    return integration_function(y, x)

def dark_file_observables(filepath):
    waveforms=read_waveforms(filepath)
    rows=[]; retained=[]
    for t,v in waveforms:
        base_mask=t<DARK_BASELINE_END_S
        baseline=np.mean(v[base_mask]); sigma=np.std(v[base_mask],ddof=1); vc=v-baseline
        search=(t>=DARK_INTEGRATION_WINDOW_S[0])&(t<=DARK_INTEGRATION_WINDOW_S[1])
        height=-np.min(vc[search])
        charge=trapezoid_integral(-vc[search],t[search])/R_TERMINATION_OHM
        rows.append((baseline,sigma,height,charge,t[np.where(search)[0][np.argmin(vc[search])]]))
        if len(retained)<10: retained.append((t,vc))
    return rows,retained

def process_dark_run(terminal,folder,pattern):
    files=sorted(Path(folder).glob(pattern)); all_rows=[]; retained=[]; live=0.0
    for filepath in files:
        rows,traces=dark_file_observables(filepath); all_rows.extend(rows)
        retained.extend(traces[:max(0,MAX_PLOT_WAVEFORMS-len(retained))])
        elapsed=trigger_elapsed_time(filepath)
        if np.isfinite(elapsed): live+=elapsed
    columns=['baseline_V','rms_V','height_V','charge_C','pulse_time_s']
    frame=pd.DataFrame(all_rows,columns=columns)
    if len(frame):
        frame['software_selected']=frame.height_V>DARK_SOFTWARE_THRESHOLD_SIGMA*frame.rms_V
        frame['terminal']=terminal; frame['voltage_V']=DARK_VOLTAGE
    return files,frame,retained,live
