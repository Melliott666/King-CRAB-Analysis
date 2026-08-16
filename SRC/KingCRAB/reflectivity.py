"""Mirror-response loading, interpolation, and curve stitching."""
from pathlib import Path
import json
import numpy as np

def load_reflectivity(path):
    raw=json.loads(Path(path).read_text()); a=np.asarray([p['value'] for p in raw['datasetColl'][0]['data']],float)
    a=a[np.argsort(a[:,0])]; wavelength,inv=np.unique(a[:,0],return_inverse=True)
    response=np.zeros(len(wavelength)); counts=np.zeros(len(wavelength)); np.add.at(response,inv,a[:,1]); np.add.at(counts,inv,1)
    return wavelength,response/counts

def make_interpolator(wavelength,response,kind='pchip'):
    if kind=='pchip':
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(wavelength,response,extrapolate=False)
    if kind=='cubic':
        from scipy.interpolate import CubicSpline
        return CubicSpline(wavelength,response,bc_type='natural',extrapolate=False)
    return lambda x:np.interp(x,wavelength,response)

def stitch_curves(w1,r1,w2,r2,join=None,blend_nm=30,n=4000,interp_kind='pchip'):
    f1,f2=make_interpolator(w1,r1,interp_kind),make_interpolator(w2,r2,interp_kind); join=float(w1.max()) if join is None else join
    w=np.linspace(max(w1.min(),w2.min()),w2.max(),n); a,b=f1(w),f2(w); out=np.where(w<=join,a,b)
    if blend_nm>0:
        m=(w>=join-blend_nm)&(w<=join+blend_nm); t=(w[m]-(join-blend_nm))/(2*blend_nm); valid=np.isfinite(a[m])&np.isfinite(b[m]); q=out[m].copy(); q[valid]=(1-t[valid])*a[m][valid]+t[valid]*b[m][valid]; out[m]=q
    return w,out

def geant_wavelength_table(wavelength_nm,response_percent,max_points=80):
    """Return downsampled wavelength and fractional-reflectivity arrays."""
    w=np.asarray(wavelength_nm,float); r=np.clip(np.asarray(response_percent,float)/100,0,1); order=np.argsort(w); w,r=w[order],r[order]
    if max_points and len(w)>max_points:
        take=np.linspace(0,len(w)-1,max_points).astype(int); w,r=w[take],r[take]
    return w,r
