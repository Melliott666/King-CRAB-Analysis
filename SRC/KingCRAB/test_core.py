"""Fast regression tests for shared non-optical helpers."""
import numpy as np
from .pmt import normalize_waveform_times, subtract_baseline, integrate_pulse_region
from .digitized import pmt_response
from .muon_flux import CylinderGeometry, projected_area_m2

def test_waveform_baseline_and_charge():
    t=np.linspace(0,100e-9,101); v=np.zeros_like(t); v[40:51]=-1e-3
    wave=normalize_waveform_times([(t,v+2e-3)])
    corrected,baseline,sigma=subtract_baseline(wave,20e-9)
    assert np.isclose(baseline[0],2e-3) and sigma[0]<1e-15
    charge,left,right=integrate_pulse_region(*corrected[0],pulse_start_time=20e-9)
    assert charge>0 and left<right

def test_gas_pressure_response():
    ar=pmt_response('argon',1); xe=pmt_response('xenon',10)
    assert ar['wavelength_nm']==128.0 and xe['wavelength_nm']>169.0
    assert ar['gain']>0 and xe['pde']>0

def test_nexus_cylinder_projection():
    g=CylinderGeometry()
    assert g.radius_m>0 and g.length_m>0
    assert np.isclose(projected_area_m2(0,0,g),2*g.radius_m*g.length_m)
