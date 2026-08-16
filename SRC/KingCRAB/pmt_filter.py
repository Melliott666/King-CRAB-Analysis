"""Notebook-independent helpers extracted from PMT_Filter_Comparison.ipynb."""
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


def load_files(indices):
    records=[];t_ref=None
    files=sorted(DATA_FOLDER.glob('C4C*.txt'))
    for index in indices:
        path=files[index]
        x=np.loadtxt(path,delimiter=',',skiprows=504,usecols=(0,1))
        cuts=np.r_[0,np.flatnonzero(np.diff(x[:,0])<=0)+1,len(x)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            t=x[a:b,0]*1e9;v=x[a:b,1]
            if t_ref is None:t_ref=t
            if len(t)==len(t_ref):records.append(v)
    return t_ref,np.asarray(records)

def baseline_subtract(records):return records-records[:,base].mean(axis=1,keepdims=True)

def apply_filter(records,kind,p1,p2):
    if kind=='raw':return records.copy()
    if kind=='fft':
        spectrum=np.fft.rfft(records,axis=1);freq=np.fft.rfftfreq(records.shape[1],dt_s)
        spectrum[:,freq>p1]=0
        return np.fft.irfft(spectrum,n=records.shape[1],axis=1)
    if kind=='combined':
        spectrum=np.fft.rfft(records,axis=1);freq=np.fft.rfftfreq(records.shape[1],dt_s)
        spectrum[:,freq>p1]=0
        fft_filtered=np.fft.irfft(spectrum,n=records.shape[1],axis=1)
        window,order=p2
        return savgol_filter(fft_filtered,window_length=int(window),polyorder=int(order),axis=1,mode='interp')
    return savgol_filter(records,window_length=int(p1),polyorder=int(p2),axis=1,mode='interp')

def config_name(kind,p1,p2):
    if kind=='raw':return 'raw'
    if kind=='fft':return f'FFT {p1/1e6:.0f} MHz'
    if kind=='combined':return f'FFT {p1/1e6:.0f} MHz + SG {int(p2[0])}/{int(p2[1])}'
    return f'SG {int(p1)} samples, order {int(p2)}'

def paired_detection_matrix(name):
    kind,p1,p2=next(c for c in configs if config_name(*c)==name)
    train_f=apply_filter(train,kind,p1,p2);signal_f=apply_filter(injected,kind,p1,p2)
    train_score=-train_f[:,null_region].min(axis=1);signal_score=-signal_f[:,null_region].min(axis=1)
    return np.vstack([signal_score>np.quantile(train_score,1-fpr,method='higher')
                      for fpr in TARGET_FALSE_POSITIVE_RATES])
