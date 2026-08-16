import math
import numpy as np
from SRC.KingCRAB.analysis import *

def test_solid_angle_small_limit():
    assert abs(onaxis_collection_exact(10000,10)/onaxis_collection_small(10000,10)-1)<2e-6

def test_disk_area_and_normalization():
    q=disk_quadrature(25.4,9,36)
    assert np.isclose(q.area_weight.sum(),math.pi*25.4**2,rtol=1e-13)
    assert np.isclose(q.normalized_weight.sum(),1,rtol=1e-13)

def test_inverse_square_limit():
    g=Geometry(); p1=offaxis_aperture_probability([0,0,g.z_l1-5000],g); p2=offaxis_aperture_probability([0,0,g.z_l1-10000],g)
    assert abs(p1/p2-4)<2e-4

def test_snell_and_reflection():
    assert np.allclose(refract(U0,-U0,1,1.5),U0)
    n1,n2=mirror_normals(); assert np.allclose(reflect(U0,n1),UDIAG); assert np.allclose(reflect(UDIAG,n2),UOUT)

def test_collinearity_and_central_path():
    g=Geometry(); assert np.linalg.norm(np.cross(g.l2-M2,UOUT))==0; assert np.linalg.norm(np.cross(g.ii-M2,UOUT))==0

def test_paraxial_focus_is_finite():
    p=paraxial_system(Geometry()); assert np.isfinite(p['predicted_d_m2_ii_mm']); assert np.isfinite(p['magnification']); assert abs(p['ABCD'][0,1])<1e-10

def test_hybrid_base_interface():
    f,fw,p,pw=rt.deterministic_rays(Geometry(),2,2,2); assert len(f)==len(fw); assert len(p)==len(pw); assert np.isclose(fw.sum(),1); assert np.isclose(pw.sum(),math.pi*(Geometry().lens_diameter/2)**2)

def test_quadrature_matches_exact_solid_angle():
    c=analytic_checks(Geometry()); assert abs(c['disk_quadrature']/c['exact']-1)<2e-5
