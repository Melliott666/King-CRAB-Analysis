"""Hybrid analytic/deterministic CRAB optics model (corrected v2).

The original module remains available. This module adds analytic checks and
Gauss--Legendre disk/EL-volume quadrature, then uses the existing 3-D finite-
aperture sequential tracer as the final evaluator.
"""
from __future__ import annotations
import math, time
from dataclasses import replace
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from . import raytrace as rt
from .raytrace import *

def onaxis_solid_angle(distance_mm,radius_mm):
    return 2*math.pi*(1-distance_mm/math.sqrt(distance_mm**2+radius_mm**2))

def onaxis_collection_exact(distance_mm,radius_mm): return onaxis_solid_angle(distance_mm,radius_mm)/(4*math.pi)
def onaxis_collection_small(distance_mm,radius_mm): return radius_mm**2/(4*distance_mm**2)

def disk_quadrature(radius,nr,nphi,z=0.,center=(0.,0.)):
    """Tensor-product GL quadrature in u=r^2/a^2 and uniform azimuth."""
    x,wu=leggauss(nr); u=(x+1)/2; wu=wu/2; rows=[]; area=math.pi*radius**2
    for ir,(ui,wi) in enumerate(zip(u,wu)):
      for ip in range(nphi):
        ph=2*math.pi*ip/nphi; w=area*wi/nphi
        rows.append((center[0]+radius*math.sqrt(ui)*math.cos(ph),center[1]+radius*math.sqrt(ui)*math.sin(ph),z,ir,ip,w,w/area))
    return pd.DataFrame(rows,columns=['x','y','z','radial_index','azimuthal_index','area_weight','normalized_weight'])

def deterministic_rays_gl(g,nfield=5,nap=7,nz=3):
    fs=[]
    z,wz=leggauss(nz); z=z*g.el_thickness/2; wz=wz/2
    for iz,(zz,zw) in enumerate(zip(z,wz)):
      q=disk_quadrature(g.el_diameter/2,nfield,4*nfield,z=zz); q['normalized_weight']*=zw; fs.append(q)
    f=pd.concat(fs,ignore_index=True); p=disk_quadrature(g.lens_diameter/2,nap,4*nap,z=g.z_l1)
    return f[['x','y','z']].to_numpy(),f.normalized_weight.to_numpy(),p[['x','y','z']].to_numpy(),p.area_weight.to_numpy()

# The final evaluator resolves this global at runtime, so install corrected GL quadrature.
rt.deterministic_rays=deterministic_rays_gl

def offaxis_aperture_probability(source,g,nr=20,nphi=80):
    q=disk_quadrature(g.lens_diameter/2,nr,nphi,z=g.z_l1); p=q[['x','y','z']].to_numpy(); v=p-np.asarray(source); r=np.linalg.norm(v,axis=1)
    return float(np.sum(q.area_weight.to_numpy()*(v[:,2]/r)/(4*math.pi*r*r)))

def lens1_geometrical_capture(g,nfield=12,nphi_field=48,nap=16,nphi_pupil=64,nz=5):
    """EL-volume averaged probability of geometrically reaching Lens 1.

    This is the common denominator for all ``post_L1_percent`` quantities.
    It contains no refraction, Fresnel loss, absorption, mirror response, or QE.
    """
    z,wz=leggauss(nz); z=z*g.el_thickness/2; wz=wz/2; total=0.
    pupil=disk_quadrature(g.lens_diameter/2,nap,nphi_pupil,z=g.z_l1)
    pp=pupil[['x','y','z']].to_numpy(); pa=pupil.area_weight.to_numpy()
    field=disk_quadrature(g.el_diameter/2,nfield,nphi_field,z=0.)
    for zz,zw in zip(z,wz):
      for (_,s),fw in zip(field.iterrows(),field.normalized_weight):
        src=np.array([s.x,s.y,zz]); v=pp-src; rr=np.linalg.norm(v,axis=1)
        total+=zw*fw*np.sum(pa*(v[:,2]/rr)/(4*math.pi*rr*rr))
    return float(total)

def photon_yield_style_full_el_capture(g,nfield=16,nphi_field=64,nap=20,nphi_pupil=80,nz=7):
    """`Photon_Yield.ipynb::omega_avg_nquad` expressed with deterministic GL.

    The notebook's integral is y0/|delta r|^3 over the complete EL and
    aperture disks, divided by EL area and 4*pi. Here y0 is generalized to
    the actual axial separation for every EL-gap z node. This is mathematically
    the same integral, using current NEXUS dimensions rather than old notebook
    scratch values.
    """
    return lens1_geometrical_capture(g,nfield,nphi_field,nap,nphi_pupil,nz)

def lens1_capture_crosscheck(g=None):
    g=g or Geometry(); onaxis=onaxis_collection_exact(g.z_l1,g.lens_diameter/2)
    full=photon_yield_style_full_el_capture(g)
    d=pd.DataFrame([{'method':'on-axis analytic point (not denominator)','capture_probability':onaxis},
                    {'method':'full EL disk + 7 mm gap to full Lens-1 disk (denominator)','capture_probability':full,
                     'el_radius_mm':g.el_diameter/2,'el_gap_thickness_mm':g.el_thickness,'lens_radius_mm':g.lens_diameter/2,'lens1_z_mm':g.z_l1}])
    return d

def add_post_l1_percent(table,geometry=None):
    """Add paired absolute/percent columns to a result table."""
    out=table.copy(); cache={}
    den=[]
    for _,r in out.iterrows():
      zl=float(r.get('z_l1',geometry.z_l1 if geometry else Geometry().z_l1))
      key=round(zl,9)
      if key not in cache: cache[key]=lens1_geometrical_capture(replace(geometry or Geometry(),z_l1=zl))
      den.append(cache[key])
    out['lens1_geometrical_capture']=den
    for col in ['geometrical_acceptance','weighted_efficiency','worst_field_efficiency']:
      if col in out: out[col+'_post_L1_percent']=100*out[col]/out.lens1_geometrical_capture
    return out

def lens_matrix(g,wl):
    """Thick plano-convex matrix in reduced angle [y,n theta]."""
    ng=float(gas_refractive_index(g.gas,g.pressure_bar,wl)); nl=float(n_fused_silica(wl)); R=g.lens_radius; t=g.lens_thickness
    S=np.array([[1.,0.],[-(nl-ng)/R,1.]])
    T=np.array([[1.,t/nl],[0.,1.]])
    return T@S

def prop(L,n): return np.array([[1.,L/n],[0.,1.]])

def paraxial_system(g,wl=None):
    wl=emission_wavelength_nm(g.gas,g.pressure_bar) if wl is None else wl; ng=float(gas_refractive_index(g.gas,g.pressure_bar,wl)); L=lens_matrix(g,wl)
    object_to_l1=g.z_l1; l1_to_m2=(M1[2]-g.z_l1)+np.linalg.norm(M2-M1); m2_to_l2=g.d_m2_l2
    pre=L@prop(object_to_l1,ng); to_l2=prop(m2_to_l2+l1_to_m2,ng)@pre; after_l2=L@to_l2
    A,B,C,D=after_l2.ravel(); focus_from_l2=-ng*B/D if abs(D)>1e-15 else np.inf
    focus_d_m2=g.d_m2_l2+focus_from_l2; M=prop(focus_from_l2,ng)@after_l2
    power=-L[1,0]; efl=ng/power if power else np.inf
    return {'efl_mm':efl,'bfl_mm':-ng*L[0,0]/L[1,0],'focus_from_l2_mm':focus_from_l2,'predicted_d_m2_ii_mm':focus_d_m2,'magnification':M[0,0],'ABCD':M}

def projected_apertures(g):
    n1,n2=mirror_normals(); return pd.DataFrame([
      ('Lens 1',g.lens_diameter,1.),('Mirror 1',g.mirror_diameter,abs(np.dot(U0,n1))),('Mirror 2',g.mirror_diameter,abs(np.dot(UDIAG,n2))),('Endcap bore',2*ENDCAP_BORE_RADIUS,1.),('Lens 2',g.lens_diameter,1.),('II',g.ii_diameter,1.)],columns=['component','diameter_mm','projection_factor']).assign(projected_diameter_mm=lambda x:x.diameter_mm*x.projection_factor)

def analytic_checks(g):
    z=g.z_l1; a=g.lens_diameter/2; exact=onaxis_collection_exact(z,a); small=onaxis_collection_small(z,a); numeric=offaxis_aperture_probability([0,0,0],g)
    assert abs(numeric-exact)/exact<2e-5
    return {'exact':exact,'small_aperture':small,'disk_quadrature':numeric,'relative_small_error':abs(small-exact)/exact}

def guided_focus_scan(g,nfield=4,nap=5):
    pred=paraxial_system(g)['predicted_d_m2_ii_mm']; lo,hi=BOUNDS['d_m2_ii']; center=np.clip(pred,lo,hi); xs=np.linspace(max(lo,center-12),min(hi,center+12),13); rows=[]
    for x in xs:
      gg=replace(g,d_m2_ii=x); t=time.perf_counter(); m=rt.evaluate(gg,nfield,nap); rows.append({'d_m2_ii':x,'paraxial_prediction':pred,'runtime_s':time.perf_counter()-t,**m})
    return pd.DataFrame(rows)

def throughput_profiles(g=None,nfield=3,nap=4,npoints=9,flat_span_percent=1.0):
    """Return throughput profiles without inventing maxima on flat curves.

    If the full relative throughput span is below ``flat_span_percent``, photon
    count does not constrain that coordinate.  The current hardware position is
    retained for throughput; a separate best-focus position is reported from
    the minimum finite R80 value.
    """
    g=g or Geometry(); rows=[]
    for var,(lo,hi) in BOUNDS.items():
      for x in np.linspace(lo,hi,npoints):
        gg=replace(g,**{var:x}); m=rt.evaluate(gg,nfield,nap)
        rows.append({'variable':var,'position_mm':x,'z_l1':gg.z_l1,'d_m2_l2':gg.d_m2_l2,'d_m2_ii':gg.d_m2_ii,**m})
    d=add_post_l1_percent(pd.DataFrame(rows),g)
    rec=[]
    for var,q in d.groupby('variable',sort=False):
      q=q[np.isfinite(q.weighted_efficiency)].copy(); im=q.weighted_efficiency.idxmax(); numerical_best=q.loc[im]
      mean=float(q.weighted_efficiency.mean()); span=100*float(q.weighted_efficiency.max()-q.weighted_efficiency.min())/mean if mean else np.inf
      sensitive=span>=flat_span_percent; current=float(getattr(g,var))
      chosen_position=float(numerical_best.position_mm) if sensitive else current
      plateau=q[q.weighted_efficiency>=.98*numerical_best.weighted_efficiency]
      focus=q[np.isfinite(q.r80_mm)]; best_focus=float(focus.loc[focus.r80_mm.idxmin()].position_mm) if len(focus) else np.nan
      rec.append({'variable':var,'throughput_sensitive':sensitive,'relative_throughput_span_percent':span,
                  'recommended_mm':chosen_position,'numerical_maximum_mm':float(numerical_best.position_mm),
                  'best_focus_mm':best_focus,'minus_mm':float(chosen_position-plateau.position_mm.min()),
                  'plus_mm':float(plateau.position_mm.max()-chosen_position),'maximum_throughput':float(numerical_best.weighted_efficiency),
                  'recommendation_basis':'throughput maximum' if sensitive else 'throughput insensitive: retain current hardware position',
                  'plateau_definition':'within 2% of profile maximum'})
    return d,pd.DataFrame(rec)

def run_gas_pressure_hybrid(pressures=(1.,3.,5.,7.5,10.),nfield=4,nap=6):
    """Hybrid physical scan retaining gas, pressure, spectrum and Lens-1 normalization."""
    rows=[]
    for gas in ('argon','xenon'):
      for pressure in pressures:
        g=Geometry(gas=gas,pressure_bar=pressure)
        para=paraxial_system(g); focus=float(np.clip(para['predicted_d_m2_ii_mm'],*BOUNDS['d_m2_ii']))
        # Evaluate nominal II and paraxially guided II; geometry remains otherwise nominal.
        for selection,gg in [('nominal',g),('paraxial_guided',replace(g,d_m2_ii=focus))]:
          t=time.perf_counter(); m=rt.evaluate(gg,nfield,nap)
          rows.append({'gas':gas,'pressure_bar':pressure,'selection':selection,
                       'emission_centroid_nm':emission_wavelength_nm(gas,pressure),
                       'gas_refractive_index':float(gas_refractive_index(gas,pressure,emission_wavelength_nm(gas,pressure))),
                       'paraxial_d_m2_ii':para['predicted_d_m2_ii_mm'],'runtime_s':time.perf_counter()-t,
                       'z_l1':gg.z_l1,'d_m2_l2':gg.d_m2_l2,'d_m2_ii':gg.d_m2_ii,**m})
    return add_post_l1_percent(pd.DataFrame(rows),Geometry())

def run_dense_convergence(g=None,orders=((3,4),(4,5),(5,6),(6,7),(7,8),(8,10),(9,12))):
    g=g or Geometry(); rows=[]
    for nf,npup in orders:
      t=time.perf_counter(); m=rt.evaluate(g,nf,npup)
      rows.append({'field_radial_order':nf,'field_azimuth_nodes':4*nf,'pupil_radial_order':npup,
                   'pupil_azimuth_nodes':4*npup,'runtime_s':time.perf_counter()-t,**m})
    out=add_post_l1_percent(pd.DataFrame(rows),g)
    for c in ['geometrical_acceptance','weighted_efficiency','geometrical_acceptance_post_L1_percent','weighted_efficiency_post_L1_percent','r80_mm','distortion_mm']:
      out[c+'_successive_change_percent']=100*out[c].pct_change().abs()
    return out

def run_hybrid_demo():
    g=Geometry(); qf=disk_quadrature(g.el_diameter/2,8,32); qp=disk_quadrature(g.lens_diameter/2,8,32); assert np.isclose(qf.area_weight.sum(),math.pi*(g.el_diameter/2)**2); assert np.isclose(qp.normalized_weight.sum(),1)
    checks=pd.DataFrame([analytic_checks(g)])
    scan=add_post_l1_percent(guided_focus_scan(g),g)
    return checks,scan
