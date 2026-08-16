"""Deterministic sequential geometrical ray tracer for the CRAB periscope.

Coordinates and dimensions are in mm.  The immutable nominal geometry is
transcribed from packages/nexus/source/geometries/KingCRAB.cc (2026-08-15).
This module deliberately has no Geant4 dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
from DATA.crab_nexus_geometry import *
from DATA.crab_gas_properties import *
from DATA.crab_optical_properties import *

def unit(v):
    v=np.asarray(v,float); return v/np.linalg.norm(v)

U0=INITIAL_AXIS; UDIAG=DIAGONAL_AXIS; UOUT=OUTGOING_AXIS
M1=MIRROR_1_CENTER; M2=MIRROR_2_CENTER
# NEXUS fixed endcap bore: 50.8 mm diameter through the 38.1 mm endcap.
ENDCAP_BORE_RADIUS=ENDCAP_BORE_DIAMETER/2
ENDCAP_Z_INNER=ENDCAP_BORE_Z_INNER
ENDCAP_Z_OUTER=ENDCAP_BORE_Z_OUTER

@dataclass(frozen=True)
class Geometry:
    z_l1: float=LENS_1_Z_NOMINAL
    d_m2_l2: float=M2_LENS_2_NOMINAL
    d_m2_ii: float=M2_II_NOMINAL
    lens_diameter: float=LENS_DIAMETER
    lens_thickness: float=LENS_CENTER_THICKNESS
    lens_radius: float=LENS_CURVATURE_RADIUS
    mirror_diameter: float=MIRROR_DIAMETER
    ii_diameter: float=II_DIAMETER
    el_diameter: float=EL_DIAMETER
    el_thickness: float=EL_GAP_THICKNESS
    lens_material: str=LENS_MATERIAL
    gas: str="xenon"
    pressure_bar: float=1.0
    @property
    def l1(self): return np.array([0.,0.,self.z_l1])
    @property
    def l2(self): return M2+self.d_m2_l2*UOUT
    @property
    def ii(self): return M2+self.d_m2_ii*UOUT

BOUNDS=POSITION_BOUNDS_MM
PENALTIES={"uniformity":0.08,"blur":0.05,"unimaged":0.15,"mechanical":10.0}

def n_caf2(wl_nm):
    """Malitson Sellmeier, matching Nexus OpticalMaterialProperties.cc."""
    l=np.asarray(wl_nm)/1000.; l2=l*l
    return np.sqrt(1+0.5675888*l2/(l2-0.050263605**2)+0.4710914*l2/(l2-0.1003909**2)+3.8484723*l2/(l2-34.649040**2))

def n_fused_silica(wl_nm):
    l=np.asarray(wl_nm)/1000.; l2=l*l
    return np.sqrt(1+0.6961663*l2/(l2-0.0684043**2)+0.4079426*l2/(l2-0.1162414**2)+0.8974794*l2/(l2-9.896161**2))

def gas_density_kg_m3(gas, pressure_bar):
    """NEXUS density conventions; Xe table interpolation, Ar interpolation for requested pressures."""
    p=float(pressure_bar)
    if gas.lower().startswith('xe'):
        return float(np.interp(p,XENON_DENSITY_PRESSURE_BAR,XENON_DENSITY_KG_M3))
    return float(np.interp(p,ARGON_DENSITY_PRESSURE_BAR,ARGON_DENSITY_KG_M3))

def gas_refractive_index(gas, pressure_bar, wl_nm):
    """Port of NEXUS GXe/GAr RINDEX functions (energy in eV, density in g/cm3)."""
    wl=np.asarray(wl_nm,float); energy=1239.841984/wl
    if gas.lower().startswith('xe'):
        P=np.array(XENON_REFRACTIVITY_P); E=np.array(XENON_RESONANCE_EV)
        vir=sum(P[i]/(energy**2-E[i]**2) for i in range(3))
        mol_density=(gas_density_kg_m3('xenon',pressure_bar)/1000.)/XENON_MOLAR_MASS_G
        alpha=vir*mol_density
        return np.sqrt(np.maximum(1.,(1-2*alpha)/(1+alpha)))
    # Exact expression in opticalprops::GAr; NEXUS has no pressure argument,
    # so scale refractivity by density relative to its implicit 1-bar model.
    lm=wl/1000.
    n1=1+0.012055*(0.2075*lm**2/(91.012*lm**2-1)+0.0415*lm**2/(87.892*lm**2-1)+4.3330*lm**2/(214.02*lm**2-1))
    return 1+(n1-1)*gas_density_kg_m3('argon',pressure_bar)/1.60279

def emission_wavelength_nm(gas, pressure_bar):
    """NEXUS EL-spectrum centroid: fixed 128 nm Ar; pressure-shifted Xe."""
    return ARGON_EL_CENTROID_NM if gas.lower().startswith('ar') else XENON_EL_INTERCEPT_NM+XENON_EL_PRESSURE_SLOPE_NM_PER_ATM*(pressure_bar/BAR_PER_ATMOSPHERE)

def refract(d,n,n1,n2):
    n=unit(n); d=unit(d)
    if np.dot(d,n)>0:n=-n
    eta=n1/n2; c=-np.dot(n,d); k=1-eta*eta*(1-c*c)
    return None if k<0 else unit(eta*d+(eta*c-math.sqrt(k))*n)

def reflect(d,n): return unit(d-2*np.dot(d,unit(n))*unit(n))

def plane_hit(o,d,p,n):
    q=np.dot(p-o,n)/np.dot(d,n)
    return (o+q*d,q) if q>1e-8 else (None,q)

def disk_hit(o,d,p,n,r):
    x,t=plane_hit(o,d,p,n)
    return (x,t) if x is not None and np.linalg.norm(x-p)<=r else (None,t)

def _equal_weight_disk(n,radius,z=0.,offset=(0.,0.)):
    """Deterministic equal-area golden-angle disk mapping."""
    k=np.arange(n); rr=radius*np.sqrt((k+.5)/n); ph=k*math.pi*(3-math.sqrt(5))
    return np.c_[offset[0]+rr*np.cos(ph),offset[1]+rr*np.sin(ph),np.full(n,z)]

def deterministic_rays(g:Geometry,nfield=7,nap=11,nz=3):
    """Equal-weight EL-volume and entrance-pupil quadrature."""
    layers=[]
    for z in np.linspace(-g.el_thickness/2,g.el_thickness/2,nz): layers.append(_equal_weight_disk(nfield*nfield,g.el_diameter/2,z))
    fields=np.vstack(layers); fw=np.full(len(fields),1/len(fields))
    pupils=_equal_weight_disk(nap*nap,g.lens_diameter/2,g.z_l1); pw=np.full(len(pupils),math.pi*(g.lens_diameter/2)**2/len(pupils))
    return fields,fw,pupils,pw

def fused_silica_abs_length_mm(wl_nm):
    e=1239.841984/np.asarray(wl_nm,float)
    ee=np.array(FUSED_SILICA_ABS_ENERGY_EV); ll=np.array(FUSED_SILICA_ABS_LENGTH_MM)
    return np.interp(e,ee,ll)

def fresnel_T(d,n,n1,n2):
    ci=abs(np.dot(unit(d),unit(n))); st2=(n1/n2)**2*(1-ci*ci)
    if st2>=1:return 0.
    ct=math.sqrt(1-st2); rs=((n1*ci-n2*ct)/(n1*ci+n2*ct))**2; rp=((n1*ct-n2*ci)/(n1*ct+n2*ci))**2
    return 1-.5*(rs+rp)

def lens_trace(o,d,c,axis,g,wl):
    """Trace a plano-convex lens through a spherical first and planar second surface."""
    a=unit(axis); nidx=float(n_fused_silica(wl)); ng=float(gas_refractive_index(g.gas,g.pressure_bar,wl))
    # Spherical vertex at c-t/2, center of curvature downstream by R.
    v=c-a*g.lens_thickness/2; sc=v+a*g.lens_radius
    oc=o-sc; b=np.dot(d,oc); disc=b*b-(np.dot(oc,oc)-g.lens_radius**2)
    if disc<0:return None,None,"lens_aperture",0.
    roots=[-b-math.sqrt(disc),-b+math.sqrt(disc)]; ts=[x for x in roots if x>1e-8]
    if not ts:return None,None,"lens_aperture",0.
    p=o+min(ts)*d
    if np.linalg.norm((p-c)-np.dot(p-c,a)*a)>g.lens_diameter/2:return None,None,"lens_aperture",0.
    tr=fresnel_T(d,p-sc,ng,nidx)
    d2=refract(d,p-sc,ng,nidx)
    if d2 is None:return None,None,"lens_TIR",0.
    p2,t=plane_hit(p+1e-7*d2,d2,c+a*g.lens_thickness/2,a)
    if p2 is None or np.linalg.norm((p2-c)-np.dot(p2-c,a)*a)>g.lens_diameter/2:return None,None,"lens_aperture",0.
    tr*=fresnel_T(d2,a,nidx,ng)*math.exp(-np.linalg.norm(p2-p)/float(fused_silica_abs_length_mm(wl)))
    d3=refract(d2,a,nidx,ng)
    return (p2+1e-7*d3,d3,None,tr) if d3 is not None else (None,None,"lens_TIR",0.)

def mirror_normals():
    # Exact bisectors enforce the documented central path and agree with source rotations.
    return unit(U0-UDIAG),unit(UDIAG-UOUT)

def spectral_weight(wl):
    """Measured CRAB mirror curve; failure is fatal, never a silent fallback."""
    path=MIRROR_REFLECTIVITY_FILE
    raw=json.loads(path.read_text()); data=raw['datasetColl'][0]['data']
    vals=np.array([d['value'] for d in data],float); order=np.argsort(vals[:,0])
    if not(vals[order[0],0]<=wl<=vals[order[-1],0]): raise ValueError(f'{wl} nm outside mirror table range')
    return float(np.interp(wl,vals[order,0],vals[order,1])/100.)

def spectrum_nodes(g,n=5):
    mu=emission_wavelength_nm(g.gas,g.pressure_bar)
    sigma=2.929 if g.gas.startswith('ar') else (14.3 if g.pressure_bar/1.01325<4 else (-.117*g.pressure_bar/1.01325+15.16)/2.35482)
    x=np.linspace(mu-2*sigma,mu+2*sigma,n); w=np.exp(-.5*((x-mu)/sigma)**2); return x,w/w.sum()

def valid_geometry(g):
    reasons=[]
    if not(BOUNDS['z_l1'][0]<=g.z_l1<=BOUNDS['z_l1'][1]):reasons.append('L1 outside vessel clearance')
    if g.z_l1+g.lens_thickness/2>=M1[2]-2:reasons.append('L1 intersects M1 clearance')
    if not(BOUNDS['d_m2_l2'][0]<=g.d_m2_l2<=BOUNDS['d_m2_l2'][1]):reasons.append('L2 outside II tube')
    if not(BOUNDS['d_m2_ii'][0]<=g.d_m2_ii<=BOUNDS['d_m2_ii'][1]):reasons.append('II outside focal region')
    if g.d_m2_l2+g.lens_thickness/2>=g.d_m2_ii-2:reasons.append('L2 intersects II')
    return len(reasons)==0,'; '.join(reasons)

def evaluate(g:Geometry,nfield=5,nap=7,wavelengths=None,details=False,nz=3):
    ok,why=valid_geometry(g)
    if not ok:return {"valid":False,"reason":why,"objective":-PENALTIES['mechanical']}
    if wavelengths is None: wavelengths,specw=spectrum_nodes(g)
    else: wavelengths=np.asarray(wavelengths); specw=np.full(len(wavelengths),1/len(wavelengths))
    fs,fw,ps,pw=deterministic_rays(g,nfield,nap,nz); n1,n2=mirror_normals()
    losses={k:0 for k in ['L1','M1','M2','ENDCAP_BORE','L2','II']}; hits=[]; records=[]; emitted=0.; accepted_solid=0.; weighted=0.
    for wl,sw in zip(wavelengths,specw):
     for fi,f in enumerate(fs):
      for pi,ptarget in enumerate(ps):
       d=unit(ptarget-f); cos=d[2]; solid=pw[pi]/np.linalg.norm(ptarget-f)**2*cos
       emitted+=fw[fi]*sw*solid/(4*math.pi)
       o,d,err,tr1=lens_trace(f,d,g.l1,U0,g,wl)
       if err: losses['L1']+=1; continue
       p,t=disk_hit(o,d,M1,n1,g.mirror_diameter/2)
       if p is None: losses['M1']+=1; continue
       d=reflect(d,n1); o=p+1e-7*d
       p,t=disk_hit(o,d,M2,n2,g.mirror_diameter/2)
       if p is None: losses['M2']+=1; continue
       d=reflect(d,n2); o=p+1e-7*d
       # A finite cylindrical cut through the fixed endcap: passing one face
       # is insufficient for oblique rays, so enforce the aperture at both.
       blocked=False
       for zface in (ENDCAP_Z_INNER,ENDCAP_Z_OUTER):
         hp,ht=disk_hit(o,d,np.array([M2[0],M2[1],zface]),UOUT,ENDCAP_BORE_RADIUS)
         if hp is None: blocked=True; break
       if blocked: losses['ENDCAP_BORE']+=1; continue
       o,d,err,tr2=lens_trace(o,d,g.l2,UOUT,g,wl)
       if err: losses['L2']+=1; continue
       p,t=disk_hit(o,d,g.ii,UOUT,g.ii_diameter/2)
       if p is None: losses['II']+=1; continue
       accepted_solid+=fw[fi]*sw*solid/(4*math.pi)
       w=fw[fi]*sw*solid/(4*math.pi)*spectral_weight(wl)**2*tr1*tr2
       weighted+=w; hits.append(p); records.append((fi,*f[:2],*p[:2],w))
    hits=np.asarray(hits); rec=pd.DataFrame(records,columns=['field_id','field_x','field_y','image_x','image_y','weight'])
    nr=len(fs)*len(ps)*len(wavelengths); conditional_geom=len(rec)/nr; geom=accepted_solid
    field=rec.groupby('field_id').agg(eff=('weight','sum'),ix=('image_x','mean'),iy=('image_y','mean')) if len(rec) else pd.DataFrame()
    vals=np.zeros(len(fs));
    if len(field): vals[field.index.astype(int)]=field.eff.values
    mean=vals.mean(); worst=vals.min(); cv=vals.std()/mean if mean else 99.; coverage=np.mean(vals>0)
    spot=[]
    if len(rec):
      for _,q in rec.groupby('field_id'):
       spot.append(np.sqrt(np.mean((q.image_x-q.image_x.mean())**2+(q.image_y-q.image_y.mean())**2)))
    rms=float(np.mean(spot)) if spot else np.nan
    radii=[]
    if len(rec):
      for _,q in rec.groupby('field_id'):
       if len(q)>=5:
        rr=np.sqrt((q.image_x-q.image_x.mean())**2+(q.image_y-q.image_y.mean())**2); radii.append(float(np.quantile(rr,.8)))
    r80=float(np.mean(radii)) if radii else np.nan
    # linear image mapping gives magnification/distortion
    mag=dist=np.nan
    if len(field)>=3:
      ids=field.index.astype(int); A=np.c_[fs[ids,:2],np.ones(len(ids))]; B=field[['ix','iy']].values
      coef=np.linalg.lstsq(A,B,rcond=None)[0]; pred=A@coef; mag=float(np.sqrt(abs(np.linalg.det(coef[:2,:])))); dist=float(np.sqrt(np.mean(np.sum((B-pred)**2,axis=1))))
    weighted_eff=weighted
    # Independent normalized penalties plus explicit imaging constraints.
    feasible_image=coverage>=.95 and np.isfinite(r80) and r80<=5. and np.isfinite(dist) and dist<=5.
    obj=(weighted_eff/1e-6)-.25*cv-.20*(r80/5 if np.isfinite(r80) else 10)-.20*(dist/5 if np.isfinite(dist) else 10)-.50*(1-coverage)
    if not feasible_image: obj-=5.
    out=dict(valid=True,reason='',objective=obj,weighted_efficiency=weighted_eff,geometrical_acceptance=geom,conditional_geometrical_acceptance=conditional_geom,worst_field_efficiency=worst,field_cv=cv,coverage=coverage,rms_spot_mm=rms,r80_mm=r80,magnification=mag,distortion_mm=dist,n_rays=nr,**{f'loss_{k}':v for k,v in losses.items()})
    if details:out.update(records=rec,fields=fs,field_values=vals,hits=hits)
    return out
