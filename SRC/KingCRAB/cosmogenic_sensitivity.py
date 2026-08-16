"""Notebook-independent helpers extracted from Cosmogenic_PMT_Waveform_Sensitivity.ipynb."""
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


def gamma_charge(rng,mean_charge_C,relative_sigma):
    if relative_sigma<=0:return float(mean_charge_C)
    shape=1.0/relative_sigma**2;scale=mean_charge_C*relative_sigma**2
    return float(rng.gamma(shape,scale))

def add_impulse(signal,t,peak_time_ns,charge_C):
    launch_ns=peak_time_ns-kernel_peak_offset
    response=np.interp(t-launch_ns,kernel_t,kernel,left=0.0,right=0.0)
    if not np.any(response):return False
    signal-=R_LOAD_OHM*charge_C*response
    return True

def binomial_significance_scan(p0,p1,sample_sizes,n_trials,seed):
    from scipy.stats import binom,norm
    rng=np.random.default_rng(seed);rows=[]
    for n in sample_sizes:
        observed=rng.binomial(n,p1,size=n_trials)
        null_p=np.clip(binom.sf(observed-1,n,p0),np.finfo(float).tiny,1-np.finfo(float).eps)
        z=norm.isf(null_p)
        rows.append({'waveforms':int(n),'median_Z':np.median(z),'Z_16':np.quantile(z,.16),
                     'Z_84':np.quantile(z,.84),'probability_Z_at_least_5':np.mean(z>=5)})
    return pd.DataFrame(rows)

def detected_clusters(rows,rng,pde=None,gap_ns=None):
    if pde is None: pde=PDE
    if gap_ns is None: gap_ns=CLUSTER_GAP_NS
    if len(rows)==0:return []
    d=rows.loc[rng.random(len(rows))<pde,['arrival_time_ns','origin']].sort_values('arrival_time_ns')
    clusters=[]
    for r in d.itertuples(index=False):
        if not clusters or r.arrival_time_ns-clusters[-1][-1][0]>gap_ns:clusters.append([])
        clusters[-1].append((float(r.arrival_time_ns),str(r.origin)))
    return clusters

def reconstruct_cluster(cluster,rng):
    # Background/noise triggers are already included at their measured rate.
    # Test the injected signal alone against the measured global threshold so
    # a random noise maximum is not counted a second time for every known
    # Nexus cluster.
    t=np.arange(0.0,WINDOW_NS,DT_NS); signal=np.zeros_like(t);true_charge=0.0
    t0=cluster[0][0]; placed=0
    for arrival,origin in cluster:
        peak=TRIGGER_PEAK_NS+(arrival-t0)
        if peak>=WINDOW_NS-1:continue
        q=gamma_charge(rng,Q_SPE_C,SPE_RELATIVE_SIGMA)
        if add_impulse(signal,t,peak,q):placed+=1;true_charge+=q
    height=-signal.min()
    return {'height_V':height,'charge_C':true_charge,
            'height_pass_3sigma':height>LL_SIGMA*sigma_global,'placed_pe':placed}

def simulate_gate_population(condition,n_gates,seed):
    rng=np.random.default_rng(seed);out=[]
    sampled=rng.choice(event_ids,size=n_gates,replace=True)
    for gate_id,eid in enumerate(sampled):
        pulse_q=[];pulse_h=[];n_bg=rng.poisson(MEASURED_BACKGROUND_RATE_HZ*GATE_S)
        if n_bg:
            take=background_pool.iloc[rng.integers(len(background_pool),size=n_bg)]
            pulse_q.extend(take.charge_C.to_numpy(float));pulse_h.extend(take.height_V.to_numpy(float))
        rows=grouped.get((condition,int(eid)),photons.iloc[:0])
        clusters=detected_clusters(rows,rng);accepted_cosmic=0;detected_pe=sum(map(len,clusters))
        for cluster in clusters:
            feat=reconstruct_cluster(cluster,rng)
            if feat['height_pass_3sigma']:
                pulse_q.append(feat['charge_C']);pulse_h.append(feat['height_V']);accepted_cosmic+=1
        out.append({'condition':condition,'gate_id':gate_id,'event_id':int(eid),
                    'background_pulses':n_bg,'detected_cosmic_pe':detected_pe,
                    'accepted_cosmic_clusters':accepted_cosmic,'selected_pulses':len(pulse_q),
                    'total_charge_C':sum(pulse_q),'max_charge_C':max(pulse_q,default=0.0),
                    'max_height_V':max(pulse_h,default=0.0)})
    return pd.DataFrame(out)
