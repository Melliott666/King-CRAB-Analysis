"""Notebook-independent helpers extracted from Cosmogenic_Radioactive_PMT_Backgrounds.ipynb."""
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


def load_digitized(path):
    raw=json.loads(path.read_text()); out={}
    for dataset in raw['datasetColl']:
        a=np.asarray([p['value'][:2] for p in dataset['data']],float)
        out[dataset['name']]=a[np.argsort(a[:,0])]
    return out

def log_interp(points,x):
    frame=pd.DataFrame(points,columns=['x','y']).groupby('x',as_index=False).y.mean()
    return float(np.exp(np.interp(x,frame.x,np.log(frame.y))))

def log_extrap(points,x,n=8):
    frame=pd.DataFrame(points,columns=['x','y']).groupby('x',as_index=False).y.mean()
    fit=frame.nsmallest(n,'x'); slope,intercept=np.polyfit(fit.x,np.log(fit.y),1)
    return float(np.exp(intercept+slope*x))

def decode(a):
    return np.char.decode(a,'utf-8',errors='ignore') if a.dtype.kind in 'Sa' else a.astype(str)

def read_source_h5(source,condition,path):
    with h5py.File(path,'r') as h:
        cfg_raw=h['MC/configuration'][:]
        cfg={str(k).strip():unquote(str(v)).strip()
             for k,v in zip(decode(cfg_raw['param_key']),decode(cfg_raw['param_value']))}
        n_events=int(cfg['num_events'])
        d=h['DEBUG/steps'][:]
    particle=decode(d['particle_name']); initial=decode(d['initial_volume'])
    final=decode(d['final_volume']); process=decode(d['proc_name'])
    mask=(particle=='opticalphoton')&((initial=='II_ONE_INCH_SCORE')|(final=='II_ONE_INCH_SCORE'))
    s=d[mask]; p=process[mask]
    keys=np.rec.fromarrays([s['event_id'],s['particle_id']],names='event_id,particle_id')
    _,idx=np.unique(keys,return_index=True); idx.sort(); s=s[idx]; p=p[idx]
    origin=np.where(np.char.find(p,'Electroluminescence')>=0,'S2',
                    np.where(np.char.find(p,'Scintillation')>=0,'S1','other'))
    expected_z={'Ba-133':'56','Eu-152':'63'}[source]
    expected_field='0 kV/m' if condition=='field_off' else '900 kV/m'
    checks={'atomic number':cfg.get('/Generator/IonGenerator/atomic_number')==expected_z,
            'argon':cfg.get('/Geometry/KingCRAB/gastype')=='argon',
            'field':cfg.get('/Geometry/KingCRAB/EL_field_intensity')==expected_field,
            'detector':cfg.get('/Actions/SaveAllSteppingAction/select_volume')=='II_ONE_INCH_SCORE'}
    if not all(checks.values()): raise RuntimeError(f'{path.name}: configuration failure {checks}')
    frame=pd.DataFrame({'source':source,'condition':condition,
                        'event_id':s['event_id'].astype(np.int64),
                        'particle_id':s['particle_id'].astype(np.int64),
                        'origin':origin,'arrival_time_ns':s['time'].astype(float)})
    return frame,n_events,checks

def gamma_cluster_charge(rng,k,size=None):
    shape=np.asarray(k)/SPE_RELATIVE_SIGMA**2
    scale=Q_SPE_C*SPE_RELATIVE_SIGMA**2
    return rng.gamma(shape,scale,size=size)

def simulate_optical_population(source,condition,origin,n_draw=120_000,seed=410000):
    rng=np.random.default_rng(seed)
    g=bins[(bins.source==source)&(bins.condition==condition)&(bins.origin==origin)].copy()
    if len(g)==0:return pd.DataFrame()
    idx=rng.integers(len(g),size=n_draw); sampled=g.iloc[idx].reset_index(drop=True)
    k=rng.binomial(sampled.incident_photons.to_numpy(int),PDE)
    keep=k>0; sampled=sampled.loc[keep].copy(); k=k[keep]
    q=gamma_cluster_charge(rng,k)
    ratio=rng.choice(ratio_pool,size=len(q),replace=True)
    sampled['detected_pe']=k; sampled['charge_C']=q; sampled['height_V']=q*ratio
    sampled['population']=source+' '+sampled.origin+' '+condition.replace('_',' ')
    return sampled

def optical_rate(source,condition):
    return float(rate_summary.query('source==@source and condition==@condition').detected_cluster_rate_Hz.sum())

def rate_weighted_population(condition,sources,n=180_000,seed=419000):
    # Draw triggered pulses from background and the declared optical sources.
    rng=np.random.default_rng(seed)
    frames=[background]; rates=[BACKGROUND_RATE_HZ]
    for source in sources:
        frame=optical[(optical.source==source)&(optical.condition==condition)]
        rate=optical_rate(source,condition)
        if len(frame) and rate>0:
            frames.append(frame); rates.append(rate)
    rates=np.asarray(rates,float); counts=rng.multinomial(n,rates/rates.sum())
    draws=[]
    for frame,count in zip(frames,counts):
        if count:
            draws.append(frame.iloc[rng.integers(len(frame),size=count)][['charge_C','height_V']])
    return pd.concat(draws,ignore_index=True),rates.sum()

def plot_configuration(populations,suptitle):
    fig,ax=plt.subplots(1,2,figsize=(14,5))
    for frame,label,color,rate in populations:
        rate_label=f'{label} ({rate:,.1f} Hz)'
        ax[0].hist(frame.height_V*1e3,bins=140,range=(0,hmax),histtype='step',density=True,lw=1.8,label=rate_label,color=color)
        ax[1].hist(frame.charge_C*1e12,bins=140,range=(0,qmax),histtype='step',density=True,lw=1.8,label=rate_label,color=color)
    ax[0].set(yscale='log',xlabel='negative pulse height [mV]',ylabel='probability density',title='Pulse height')
    ax[1].set(yscale='log',xlabel='integrated charge [pC]',ylabel='probability density',title='Integrated charge')
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle(suptitle); fig.tight_layout(); plt.show()
