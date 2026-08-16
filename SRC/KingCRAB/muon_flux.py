"""Guan surface-muon spectrum and finite-cylinder rate integration."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from DATA.crab_nexus_geometry import VESSEL_LENGTH, VESSEL_OD, VESSEL_WALL_THICKNESS

GUAN_P = (0.102573, -0.068287, 0.958633, 0.0407253, 0.817285)
ENERGY_MAX_GEV = 1e5

@dataclass(frozen=True)
class CylinderGeometry:
    radius_m: float = (VESSEL_OD/2 - VESSEL_WALL_THICKNESS)/1000
    length_m: float = VESSEL_LENGTH/1000
    @property
    def volume_m3(self): return np.pi*self.radius_m**2*self.length_m

def cos_theta_star(theta):
    c=np.cos(theta); p1,p2,p3,p4,p5=GUAN_P
    return np.sqrt((c*c+p1*p1+p2*c**p3+p4*c**p5)/(1+p1*p1+p2+p4))

def guan_intensity(energy_gev,theta):
    cs=cos_theta_star(theta); low=energy_gev*(1+3.64/(energy_gev*cs**1.29))
    return 0.14*low**-2.7*(1/(1+1.1*energy_gev*cs/115)+.054/(1+1.1*energy_gev*cs/850))*1e4

def projected_area_m2(theta,phi,geometry=CylinderGeometry()):
    q=abs(np.sin(theta)*np.cos(phi))
    return np.pi*geometry.radius_m**2*q+2*geometry.radius_m*geometry.length_m*np.sqrt(max(0,1-q*q))

def energy_integral(theta,energy_min_gev,moment=0):
    return quad(lambda x:(e:=np.exp(x))**moment*guan_intensity(e,theta)*e,np.log(energy_min_gev),np.log(ENERGY_MAX_GEV),epsrel=2e-5,limit=200)[0]

def azimuth_integrated_area(theta,geometry=CylinderGeometry()):
    return quad(lambda phi:projected_area_m2(theta,phi,geometry),0,2*np.pi,epsrel=1e-7)[0]

def calculate_muon_flux(energy_min_gev=.01,geometry=CylinderGeometry()):
    integrand=lambda theta,m=0:energy_integral(theta,energy_min_gev,m)*azimuth_integrated_area(theta,geometry)*np.sin(theta)
    rate=quad(lambda x:integrand(x),0,np.pi/2,epsrel=2e-4)[0]
    erate=quad(lambda x:integrand(x,1),0,np.pi/2,epsrel=2e-4)[0]
    flux=quad(lambda x:energy_integral(x,energy_min_gev)*2*np.pi*np.sin(x)*np.cos(x),0,np.pi/2,epsrel=2e-4)[0]
    directional=quad(lambda x:energy_integral(x,energy_min_gev)*2*np.pi*np.sin(x),0,np.pi/2,epsrel=2e-4)[0]
    return dict(energy_min_gev=energy_min_gev,rate_per_s=rate,rate_per_min=60*rate,mean_energy_gev=erate/rate,horizontal_flux_per_m2_s=flux,horizontal_flux_per_cm2_min=flux*60/1e4,mean_chord_m=geometry.volume_m3*directional/rate)

# Compatibility names used by the two historical muon notebooks.
integrate_over_energy = energy_integral
integrate_area_over_azimuth = azimuth_integrated_area
calculate = calculate_muon_flux

def calculate_results(energy_min_gev=.01, geometry=CylinderGeometry()):
    r=calculate_muon_flux(energy_min_gev,geometry)
    return dict(energy_min_gev=r['energy_min_gev'],rate_per_second=r['rate_per_s'],rate_per_minute=r['rate_per_min'],mean_energy_gev=r['mean_energy_gev'],horizontal_flux_cm2_min=r['horizontal_flux_per_cm2_min'],mean_chord_m=r['mean_chord_m'])
