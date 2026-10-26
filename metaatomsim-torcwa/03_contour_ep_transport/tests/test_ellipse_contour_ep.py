import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine_ellipse_contour import (ContourRules,ContourDesign,contour_fill,contour_points,
                            fabrication_cost,hard_mask_checks,periodic_margin,harmonic_modes)
from engine_ellipse_inverse import (perturbations, FabricationSpread, save_result, load_result,
    SearchSettings, optimize_start, optical_diagnostics_numpy, optical_diagnostics_torch)
from engine_ellipse_rcwa import solve_contour_torch


def test_ellipse_and_containment():
    rules=ContourRules()
    model=ContourDesign(rules,device='cpu',dtype=torch.float64)
    p=model.values()
    pts=contour_points(p,rules).detach().numpy()
    # Ellipse has unequal x/y radii, unlike the original circular default.
    radii=np.linalg.norm(pts,axis=1)
    assert radii.max()/radii.min()>1.8
    assert (periodic_margin(p,rules).detach().numpy()>rules.cell_margin).all()
    assert pts.shape==(rules.angular_samples,2)
    assert len(model.export())==4+2*len(harmonic_modes(rules))
    assert 2 not in harmonic_modes(rules)


def test_rotated_ellipse_and_asymmetric_harmonics():
    rules=ContourRules()
    m=ContourDesign(rules,dtype=torch.float64)
    with torch.no_grad():
        m.raw[3]=0.6
        m.raw[4]=1.1
        m.raw[5]=-1.2
        m.raw[4+len(harmonic_modes(rules))]=0.8
    p=m.values()
    points=contour_points(p,rules).detach().numpy()
    # Odd Fourier harmonics break inversion symmetry of pure ellipses.
    assert np.linalg.norm(points+np.roll(points,rules.angular_samples//2,axis=0))>.05
    assert np.all(np.max(np.abs(points),axis=0)<rules.period/2-rules.cell_margin)


def test_geometry_gradient_finite_differences():
    rules=ContourRules()
    m=ContourDesign(rules,dtype=torch.float64)
    with torch.no_grad():
        m.raw[3]=0.3
        m.raw[4]=0.5
    objective=lambda:contour_fill(m.values(),rules,96).sum()
    objective().backward()
    for index in (2,3,4):
        analytic=float(m.raw.grad[index])
        with torch.no_grad():
            old=m.raw[index].clone()
            eps=1e-4
            m.raw[index].copy_(old+eps);hi=float(objective())
            m.raw[index].copy_(old-eps);lo=float(objective())
            m.raw[index].copy_(old)
        fd=(hi-lo)/(2*eps)
        assert np.isfinite(analytic)
        assert np.isclose(analytic,fd,atol=.02,rtol=.03), (index,analytic,fd)


def test_geometry_and_fabrication_gradients():
    rules=ContourRules()
    model=ContourDesign(rules,device='cpu',dtype=torch.float64)
    p=model.values()
    fill=contour_fill(p,rules,64)
    fab,parts=fabrication_cost(p,rules)
    (fill.sum()+fab).backward()
    assert model.raw.grad is not None
    assert torch.isfinite(model.raw.grad).all()
    assert float(fill.min().detach())>=0 and float(fill.max().detach())<=1
    assert parts['aspect_proxy']>0


def test_height_penalty_increases_with_height():
    rules=ContourRules()
    shallow=ContourDesign(rules,dtype=torch.float64)
    deep=ContourDesign(rules,dtype=torch.float64)
    with torch.no_grad():
        shallow.raw[1]=-2
        deep.raw[1]=2
    cost_shallow,_=fabrication_cost(shallow.values(),rules)
    cost_deep,_=fabrication_cost(deep.values(),rules)
    assert float(cost_deep.detach())>float(cost_shallow.detach())


def test_perturbations_have_both_etch_signs():
    r=ContourRules()
    a=perturbations(r,count=4,seed=1)
    b=perturbations(r,count=4,seed=1)
    assert [x['bias_um'] for x in a]==[.005,-.005,.005,-.005]
    assert a[0]['harmonic_delta'].shape == (2,len(harmonic_modes(r)))
    for x,y in zip(a,b):
        assert np.array_equal(x['harmonic_delta'],y['harmonic_delta'])


def test_robust_border_and_morphology():
    r=ContourRules()
    m=ContourDesign(r,dtype=torch.float64)
    p=m.values()
    for perturb in perturbations(r,count=4):
        pts=contour_points(p,r,perturb=perturb).detach().numpy()
        assert np.abs(pts).max()<=r.period/2-r.cell_margin+1e-5
    binary=contour_fill(p,r,128,hard=True).detach().numpy()
    report=hard_mask_checks(binary,r)
    assert report['components']==1
    assert report['opening_area_loss']<0.1


def test_checkpoint(tmp_path):
    res=dict(raw=[0.]*(4+2*len(harmonic_modes(ContourRules()))),S=np.eye(4,dtype=np.complex128),loss=.123,history=[dict(iteration=1,loss=.123)])
    p=save_result(tmp_path/'checkpoint.npz',res,metadata={'test':1})
    loaded=load_result(p)
    assert loaded['raw']==res['raw']
    assert np.allclose(loaded['S'],res['S'])
    assert loaded['metadata']['test']==1


def test_mock_torcwa_forward_and_gradient(monkeypatch):
    class FakeRCWA:
        def __init__(self,**kwargs): self.dtype=kwargs['dtype'];self.device=kwargs['device']
        def add_input_layer(self,**kwargs):pass
        def add_output_layer(self,**kwargs):pass
        def set_incident_angle(self,**kwargs):pass
        def add_layer(self,*,eps,thickness): self.eps=eps;self.h=thickness
        def solve_global_smatrix(self):pass
        def S_parameters(self,*,orders,direction,port,polarization,ref_order,power_norm):
            # Synthetic differentiable response, NOT a physical Maxwell solver.
            a=(self.eps.real.mean()-1)*.05 + self.h*.01
            i='xy'.index(polarization[0]);j='xy'.index(polarization[1])
            val=(a*(1+i-j) if port=='reflection' else 0.5+a*(i+j+1)).to(self.dtype)
            return val.reshape(1)
    monkeypatch.setitem(sys.modules,'torcwa',SimpleNamespace(rcwa=FakeRCWA))
    r=ContourRules()
    m=ContourDesign(r,dtype=torch.float64)
    cfg=SimpleNamespace(wavelength=.633,period=.400,n_sub=1.46,n_fin=2.584,n_fin_kappa=0.)
    sett=SimpleNamespace(order=2,raster=32,precision='complex128',device='cpu')
    S=solve_contour_torch(cfg,sett,m.values(),r)
    assert S.shape==(4,4)
    loss=(S.abs()**2).sum()
    loss.backward()
    assert torch.isfinite(m.raw.grad).all()
    assert abs(float(m.raw.grad[0]))>1e-9
    tuning=SearchSettings(seed=3,robust_weight=.2,fab_weight=.5)
    outcome=optimize_start(cfg,sett,r,m.export(),tuning,FabricationSpread(),
            iterations=2,lr=.01,patience=4,n_perturbations=2,verbose=False)
    assert np.isfinite(outcome['loss'])
    assert len(outcome['history'])==2
    assert 'weighted_robustness' in outcome['history'][0]


def test_mock_lossless_constraint(monkeypatch):
    r=ContourRules()
    m=ContourDesign(r,dtype=torch.float64)
    cfg=SimpleNamespace(wavelength=.633,period=.400,n_sub=1.46,n_fin=2.584,n_fin_kappa=.01)
    sett=SimpleNamespace(order=2,raster=32,precision='complex128',device='cpu')
    with pytest.raises(ValueError,match='strictly real'):
        solve_contour_torch(cfg,sett,m.values(),r)


def test_optical_nonnormality_is_not_birefringence():
    # Diagonal birefringence is normal; a Jordan block is nonnormal.
    S=np.zeros((4,4),complex)
    S[2:4,:2]=np.diag([.8,.4])
    d=optical_diagnostics_numpy(S)
    assert d['T21_anisotropy']>.1
    assert d['T21_nonnormality']<1e-12
    S[2:4,:2]=np.array([[.35,.18],[0,.35]])
    dn=optical_diagnostics_numpy(S)
    assert dn['T21_anisotropy']>0
    assert dn['T21_nonnormality']>0.01
    x=optical_diagnostics_torch(torch.as_tensor(S,dtype=torch.complex128))
    assert np.isclose(float(x['T21_nonnormality']),dn['T21_nonnormality'])


def test_disallow_out_of_tolerance_bias():
    r=ContourRules()
    m=ContourDesign(r,dtype=torch.float64)
    with pytest.raises(ValueError,match='Etch bias'):
        contour_points(m.values(),r,perturb={'bias_um':.020})
