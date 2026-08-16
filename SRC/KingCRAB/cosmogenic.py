"""Shared measured-noise and photon-cluster waveform synthesis.

Use ``configure_waveform_model`` once after constructing the empirical impulse
response; this makes calibration state explicit and shared by both cosmogenic notebooks.
"""
import numpy as np
from scipy.integrate import trapezoid

def configure_waveform_model(**calibration):
    globals().update(calibration)

def clean_waveform(v):
    # Primary reconstruction uses the raw digitized samples.  Keep this
    # identity function so measured and simulated paths share one interface.
    return np.asarray(v)

def maximum_in_window(a, width_ns):
    a=np.sort(np.asarray(a,float)); j=0; best=0
    for i,x in enumerate(a):
        while x-a[j] > width_ns: j+=1
        best=max(best,i-j+1)
    return best

def split_segments(x):
    cut=np.r_[0,np.flatnonzero(np.diff(x[:,0])<=0)+1,len(x)]
    return [x[a:b] for a,b in zip(cut[:-1],cut[1:]) if b-a>100]

def bootstrap_noise(rng,n_samples):
    out=[]
    while sum(map(len,out))<n_samples:
        b=noise_bank[rng.integers(len(noise_bank))]
        out.append(b.copy())
    x=np.concatenate(out)[:n_samples]
    return x-x[:int(PRETRIGGER_NS/DT_NS)].mean()

def detect_and_cluster(rows,rng,pde=None):
    if pde is None: pde=PDE
    if len(rows)==0:return []
    accepted=rows.loc[rng.random(len(rows))<pde,['arrival_time_ns','origin']].sort_values('arrival_time_ns')
    clusters=[]
    for r in accepted.itertuples(index=False):
        if not clusters or r.arrival_time_ns-clusters[-1][-1][0]>CLUSTER_GAP_NS: clusters.append([])
        clusters[-1].append((float(r.arrival_time_ns),str(r.origin)))
    return clusters

def make_record(cluster,rng):
    t=np.arange(0,WINDOW_NS,DT_NS); v=bootstrap_noise(rng,len(t)); signal=np.zeros_like(t)
    pe_peaks=[]
    if cluster:
        t0=cluster[0][0]
        for arrival,origin in cluster:
            peak_time=TRIGGER_PEAK_NS+(arrival-t0); launch=peak_time-kernel_peak_offset
            q=rng.gamma(1/SPE_RELATIVE_SIGMA**2,Q_SPE_C*SPE_RELATIVE_SIGMA**2)
            response=np.interp(t-launch,kernel_t,kernel,left=0,right=0)
            signal-=R_LOAD_OHM*q*response; pe_peaks.append((peak_time,origin,q))
    return t,v+signal,signal,pe_peaks

def reconstruct(t,v):
    v=clean_waveform(v)
    pre=t<PRETRIGGER_NS; baseline=v[pre].mean(); y=v-baseline; sigma=y[pre].std()
    k=int(np.argmin(y)); left=k; right=k
    while left>0 and y[left]<0:left-=1
    while right<len(y)-1 and y[right]<0:right+=1
    q=-trapezoid(y[left:right+1],t[left:right+1]*1e-9)/R_LOAD_OHM
    return {'baseline_V':baseline,'noise_rms_V':sigma,'height_V':-y[k],
            'threshold_V':5*sigma,'height_pass':-y[k]>5*sigma,'charge_C':q,
            'reco_pe':q/Q_SPE_C,'left':left,'right':right,'minimum':k}
