"""Notebook-independent helpers extracted from Light_Leak.ipynb."""
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


def header_timestamps(path, header_lines=503):
    stamps=[]
    with path.open(errors='replace') as f:
        for _ in range(header_lines):
            m=HEADER_RE.match(f.readline().strip())
            if m:
                try: stamps.append(float(m.group(1).strip()))
                except ValueError: pass
    return np.asarray(stamps,float)

def process_run(label,folder,save_bank=False):
    rows=[]; spans=[]; interarrival=[]; bank=[]; t_ref=None
    files=sorted(folder.glob('C4C*.txt'))
    for file_index,path in enumerate(files):
        stamps=header_timestamps(path)
        if len(stamps)>1:
            span=stamps[-1]-stamps[0]
            if span>0: spans.append(span)
            interarrival.extend(np.diff(stamps))
        x=np.loadtxt(path,delimiter=',',skiprows=504,usecols=(0,1))
        cuts=np.r_[0,np.flatnonzero(np.diff(x[:,0])<=0)+1,len(x)]
        for segment,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
            t=x[a:b,0]*1e9; v=x[a:b,1]
            pre=t<PRE_END_NS
            baseline=v[pre].mean(); y=v-baseline
            y-=y[pre].mean();sigma=y[pre].std(ddof=1)
            search=(t>=SEARCH_RANGE_NS[0])&(t<=SEARCH_RANGE_NS[1])
            candidates=np.flatnonzero(search); k=candidates[np.argmin(y[search])]
            left=k; right=k
            while left>0 and y[left]<0:left-=1
            while right<len(y)-1 and y[right]<0:right+=1
            q_region=trapezoid(-y[left:right+1],t[left:right+1]*1e-9)/R_OHM
            fixed=(t>=FIXED_CHARGE_RANGE_NS[0])&(t<=FIXED_CHARGE_RANGE_NS[1])
            q_fixed=trapezoid(-y[fixed],t[fixed]*1e-9)/R_OHM
            q_total=trapezoid(-y[search],t[search]*1e-9)/R_OHM
            rows.append((label,file_index,segment,baseline,sigma,-y[k],t[k],q_region,q_fixed,q_total))
            if save_bank: bank.append(y.astype(np.float32))
            if t_ref is None:t_ref=t.astype(np.float32)
    cols=['run','file_index','segment','baseline_V','rms_V','height_V','pulse_time_ns',
          'charge_region_C','charge_fixed_C','charge_total_C']
    frame=pd.DataFrame(rows,columns=cols)
    live=float(np.sum(spans)); n=len(frame)
    timing={'run':label,'files':len(files),'waveforms':n,'summed_live_time_s':live,
            'trigger_rate_Hz':n/live,'median_interarrival_ms':1e3*np.median(interarrival),
            'p01_interarrival_ms':1e3*np.quantile(interarrival,.01),
            'p99_interarrival_ms':1e3*np.quantile(interarrival,.99)}
    return frame,timing,t_ref,np.asarray(bank,dtype=np.float32)

def mean_waveform_from_folder(folder):
    total=None;n=0
    for path in sorted(folder.glob('C4C*.txt')):
        x=np.loadtxt(path,delimiter=',',skiprows=504,usecols=(0,1))
        cuts=np.r_[0,np.flatnonzero(np.diff(x[:,0])<=0)+1,len(x)]
        for a,b in zip(cuts[:-1],cuts[1:]):
            t=x[a:b,0]*1e9;v=x[a:b,1];pre=t<PRE_END_NS;y=v-v[pre].mean()
            y-=y[pre].mean()
            total=y.copy() if total is None else total+y;n+=1
    return total/n
