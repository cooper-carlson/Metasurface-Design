import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from twofin import Config, stokes, write_gds, merged_fin_polygons
from engine_geometry import coverage, epsilon_map
from engine_validate import fresnel, metrics, at_monitor_planes
from engine_fdtd import split_plane_waves, Settings


def test_coverage_overlap_union():
    from dataclasses import replace
    p=replace(Config(),center1=(0,0),center2=(0,0),relative_deg=0)
    f=coverage(p,80,80,9)
    assert np.isclose(f.mean()*p.period**2,p.fin_length*p.fin_width,rtol=.02)
    assert np.isclose(f.max(),1,atol=1e-6) and f.min() == 0
    assert len(merged_fin_polygons(p)) == 1


def test_coverage_rotated():
    f=coverage(Config(),80,80,7)
    assert (f>=0).all() and (f<=1).all()
    assert 0<f.mean()<.5
    assert np.any((f>0)&(f<1))


def test_eps_bounds():
    cfg=Config()
    eps,spec=epsilon_map(cfg,dx_nm=20,samples=3)
    assert eps.shape == (20,20,spec['nz'])
    assert np.isclose(eps.min(),1)
    assert eps.max() <= cfg.n_fin**2+1e-5


def test_fresnel_unitarity_reciprocity():
    S=fresnel(Config(),-.52,.85)
    m=metrics(S)
    assert m['unitarity_max']<1e-12 and m['reciprocity_max']<1e-12


def test_propagation_shift_unitary():
    S=fresnel(Config(),-.52,.85)
    shifted=at_monitor_planes(S,Config(),-.58,.88)
    assert metrics(shifted)['unitarity_max']<1e-12


def test_y_eigenmodes():
    wave=np.array([1,0],complex)
    from twofin import eigenpolarizations
    s=eigenpolarizations(np.diag([.9+.1j,.7+.2j]))
    assert np.allclose(s[0]['stokes'],[1,1,0,0])


def test_yee_splitting():
    sc=.56
    for n in (1.,1.46):
        omega_dt=.07
        kh=np.arcsin(n*np.sin(omega_dt/2)/sc)
        for forward in (True,False):
            E=np.array([1+0j, .2j])
            h0=np.array([-n*E[1],n*E[0]])
            h = h0*(np.exp(1j*kh) if forward else -np.exp(-1j*kh))
            # undo H time correction in split_plane_waves
            h *= np.exp(-1j*omega_dt/2)
            plus,minus=split_plane_waves(E,h,n,omega_dt,sc)
            assert np.allclose(plus,E if forward else 0,atol=1e-12)
            assert np.allclose(minus,0 if forward else E,atol=1e-12)


def test_sweep_settings_guards():
    cfg=Config()
    Settings().validate(cfg)
    with pytest.raises(ValueError):
        Settings(dx_nm=100).validate(cfg)


def test_gds_export(tmp_path):
    p=Config()
    path=write_gds(p,tmp_path/'twofin.gds')
    assert path.stat().st_size>100


def test_change_reference_planes_matches_fresnel():
    from engine_validate import change_reference_planes
    p=Config(); old=(-.52,.85); new=(-.49,.91)
    a=fresnel(p,*old)
    b=change_reference_planes(a,p,old,new)
    assert np.allclose(b,fresnel(p,*new),atol=1e-12)


def test_yee_fresnel_phase_error_matches_20nm_observation():
    from engine_validate import fresnel_yee, yee_wavenumber
    p = Config()
    dx = .020
    b, t = -.53, .87
    delta = np.max(abs(fresnel_yee(p, b, t, dx)-fresnel(p, b, t)))
    assert np.isclose(delta, .031991587302954455, atol=1e-12)
    assert yee_wavenumber(p, dx, p.n_sub) > 2*np.pi*p.n_sub/p.wavelength


def test_phase_corrected_yee_fresnel_matches_continuum():
    from engine_validate import fresnel_yee, fdtd_to_continuum_planes
    p=Config()
    for b,t in [(-.53,.87),(-.52,.86),(-.4,.96)]:
        corrected=fdtd_to_continuum_planes(
            fresnel_yee(p,b,t,.020),p,(b,t),.020,bare=True)
        assert np.allclose(corrected,fresnel(p,b,t),atol=1e-14)


def test_dispersion_deembedding_reversible_and_unitary():
    from engine_validate import fdtd_to_continuum_planes
    p=Config()
    s=fresnel(p,-.53,.87)
    s2=fdtd_to_continuum_planes(s,p,(-.53,.87),.02,bare=False)
    assert metrics(s2)['unitarity_max']<1e-12
    assert np.max(abs(s2-s))>1e-3
