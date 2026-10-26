"""Local tests that do not assert GPU/TORCWA physics has been validated."""
import sys
from pathlib import Path
import json
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from engine_contour_validation import (polygon_coverage,epsilon_map_contour,
    four_blocks,power_budget,jones_metrics,channel_phase)


def circle(radius=.12,samples=256):
    t=np.linspace(0,2*np.pi,samples,endpoint=False)
    return np.column_stack((radius*np.cos(t),radius*np.sin(t)))


def fake_cfg():
    from types import SimpleNamespace
    return SimpleNamespace(period=.4,wavelength=.633,n_sub=1.46,n_fin=2.584,height=.45)


def test_polygon_area_and_cell_margin():
    c=polygon_coverage(circle(),.4,80,samples=5)
    approximate=c.sum()*(.4/80)**2
    assert abs(approximate-np.pi*.12**2)<2e-4
    assert np.all(c[:5,:]==0) and np.all(c[-5:,:]==0)


def test_polygon_rejects_wrong_shape():
    with pytest.raises(ValueError):polygon_coverage(np.zeros((4,3)),.4,20)


def test_epsilon_map_material_and_interface():
    cfg=fake_cfg()
    eps,spec=epsilon_map_contour(cfg,circle(),dx_nm=20.,samples=3)
    assert eps.shape==(20,20,spec['nz'])
    assert abs(float(eps[10,10,spec['z']<-.03].mean())-cfg.n_sub**2)<.01
    z=spec['z']
    iz=int(np.argmin(abs(z-.15)))
    assert abs(float(eps[10,10,iz])-cfg.n_fin**2)<.05
    outside=int(np.argmin(abs(z-(cfg.height+.2))))
    assert np.isclose(eps[10,10,outside],1.,atol=1e-5)
    bare,_=epsilon_map_contour(cfg,circle(),dx_nm=20.,samples=2,with_contour=False)
    assert bare[10,10,iz]==pytest.approx(1.)


def test_power_budget_identity_s():
    R,T=power_budget(np.eye(4))
    assert np.allclose(R,1) and np.allclose(T,0)
    rng=np.random.default_rng(0)
    q,_=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    R,T=power_budget(q)
    assert np.allclose(R+T,1)
    assert np.allclose(four_blocks(q)['T21'],q[2:,:2])


def test_jordan_vs_scalar_degeneracy():
    ep=np.array([[.3,.22],[0,.3]],complex)
    metrics=jones_metrics(ep)
    assert metrics['gap']<1e-8
    assert metrics['nonnormality']>.01
    assert metrics['jordan_strength']>.01
    assert metrics['overlap']>.999
    scalar=jones_metrics(np.eye(2)*.3)
    assert scalar['nonnormality']<1e-10
    assert scalar['jordan_strength']<1e-10


def test_jones_metrics_circular_basis_invariant():
    rng=np.random.default_rng(2)
    J=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
    U=np.array([[1.,1.],[1j,-1j]])/np.sqrt(2)
    a,b=jones_metrics(J),jones_metrics(U.conj().T@J@U)
    for name in ('nonnormality','anisotropy','discriminant_norm','mean_power'):
        assert a[name]==pytest.approx(b[name],rel=1e-10)


def test_notebook_structure():
    import nbformat
    nb=nbformat.read(HERE/'Fourier_Contour_Engine.ipynb',as_version=4)
    nbformat.validate(nb)
    text='\n'.join(''.join(c.source) for c in nb.cells)
    assert 'CAPTURE_EFIELD' in text and 'RUN_FDTD_CONVERGENCE' in text
    assert 'S_device_faces' in text and 'RUN_FABRICATION_SWEEP' in text
    assert 'RUN_ROTATION_TRANSPORT' in text
    assert "MODEL = 'ellipse'" in text
    import ast
    for cell in nb.cells:
        if cell.cell_type=='code': ast.parse(cell.source)
