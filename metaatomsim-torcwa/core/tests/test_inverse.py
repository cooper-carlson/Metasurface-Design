"""CPU-only tests of the EP mathematics, geometry, autograd and API contract.

These are NOT electromagnetic convergence tests; TORCWA/A6000 checks run in notebook.
"""
import ast
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch
from engine_ep import (ep_diagnostics, discriminant_torch, ep_loss_torch,
                       nonnormality_torch, track_eigenvalues)
from engine_inverse import (BOUNDS, KEYS, BoundedDesign, candidate_to_config,
                            numpy_to_params, enforce_unit_cell, save_checkpoint,
                            load_checkpoint)
from twofin import Config, fins
from engine_geometry import coverage, differentiable_twofin_fill
from engine_rcwa import Settings, solve_torch


def synthetic_ep():
    s=np.zeros((4,4),complex)
    s[:2,:2]=np.eye(2)*.25+.16*np.array([[1j,1],[1,-1j]])
    s[2,2]=.6;s[3,3]=-.5
    return s


def test_passive_reciprocal_full_s_ep_and_normal_counterexample():
    s=synthetic_ep(); d=ep_diagnostics(s)
    assert np.max(np.abs(s-s.T))==0
    assert d['max_power_gain']<1.0
    assert d['best']['gap']<1e-6
    assert d['best']['overlap']>0.999999
    assert d['best']['second_sv']>0.1
    assert d['nonnormality']>0.1
    normal=np.diag([0.2,0.2,0.6,-0.5]).astype(complex)
    di=ep_diagnostics(normal)
    assert di['nonnormality']==0.0
    assert di['best']['second_sv']<1e-9
    assert di['best']['overlap']<1e-8


def test_resultant_discriminant_matches_eigenvalue_gaps():
    rng=np.random.default_rng(12)
    mat=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    A=torch.tensor(mat,dtype=torch.complex128)
    got=float(discriminant_torch(A))
    norm=np.linalg.norm(mat,'fro')/2+1e-8
    roots=np.linalg.eigvals(mat/norm)
    reference=np.prod([abs(roots[i]-roots[j])**2 for i in range(4)
                       for j in range(i+1,4)])
    assert np.isclose(got,reference,rtol=1e-7,atol=1e-7)


def test_discriminant_and_loss_have_finite_gradients():
    A=torch.tensor(synthetic_ep()+0.04*np.eye(4),dtype=torch.complex128)
    x=torch.tensor(0.023,dtype=torch.float64,requires_grad=True)
    shifted=A.clone();shifted[0,0]=shifted[0,0]+x
    loss,di=ep_loss_torch(shifted)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad)


def test_strict_geometry_bounds_and_conversion():
    initial={key:0.5*(lo+hi) for key,(lo,hi) in BOUNDS.items()}
    initial.update(x1=.13,y1=-.13,x2=-.13,y2=.13,
                   theta1=.7,theta2=-.8)
    projected=enforce_unit_cell(initial)
    model=BoundedDesign(projected,device='cpu')
    model.raw.sum().backward()
    p=model.physical()
    cfg=candidate_to_config(Config(),p)
    assert cfg.validate() is cfg
    xy=fins(cfg)
    assert all(np.max(abs(poly))<cfg.period/2-.003 for poly in xy)
    assert cfg.fin2_length is not None
    # geometry conversion must preserve independent world-coordinate centers.
    assert np.allclose([poly.mean(axis=0) for poly in xy],
                       [[p['x1'],p['y1']],[p['x2'],p['y2']]],atol=1e-10)


def test_differentiable_raster_and_binary_geometry_close_when_sharpened():
    values={k:(lo+hi)/2 for k,(lo,hi) in BOUNDS.items()}
    values.update(theta1=.0,theta2=1.0,x1=-.07,y1=-.06,x2=.065,y2=.045)
    params=numpy_to_params(values,device='cpu',dtype=torch.float64)
    params['theta1']=params['theta1'].detach().clone().requires_grad_()
    soft=differentiable_twofin_fill(params,.4,128,edge_um=.004)
    soft.mean().backward()
    assert torch.isfinite(params['theta1'].grad)
    assert abs(float(params['theta1'].grad))>1e-6
    cfg=candidate_to_config(Config(),values)
    binary=coverage(cfg,128,128,samples=5)
    assert np.mean(np.abs(binary-soft.detach().numpy()))<0.04


def test_checkpoint_round_trip(tmp_path):
    v={k:(lo+hi)/2 for k,(lo,hi) in BOUNDS.items()}
    path=tmp_path/'trial.npz'
    record={'S':synthetic_ep(),'params':v,'history':[{'iteration':0,'loss':1.}], 'loss':1.}
    save_checkpoint(path,record)
    loaded=load_checkpoint(path)
    assert np.allclose(loaded['S'],record['S'])
    assert loaded['params']==v
    assert loaded['history']==record['history']


def test_rcwa_torch_api_mocked_only(monkeypatch):
    """Mocked TORCWA: tests autograd plumbing, not electromagnetic correctness."""
    class FakeRCWA:
        def __init__(self,**kw):self.kw=kw
        def add_input_layer(self,eps):pass
        def add_output_layer(self,eps):pass
        def set_incident_angle(self,**kw):pass
        def add_layer(self,thickness,eps):
            self.scalar=eps.mean()*(1+0.25*thickness)
        def solve_global_smatrix(self):pass
        def S_parameters(self,**kw):
            pol=kw['polarization']
            mul={'xx':1.,'xy':.6,'yx':.6,'yy':.9}[pol]
            return (self.scalar * mul * .01).reshape(1)
    monkeypatch.setitem(sys.modules,'torcwa',types.SimpleNamespace(rcwa=FakeRCWA))
    model=BoundedDesign({key:(lo+hi)/2 for key,(lo,hi) in BOUNDS.items()},device='cpu')
    S=solve_torch(Config(),model.values(),Settings(order=1,raster=32,device='cpu'),
                  ports=(-.52,1.05))
    assert tuple(S.shape)==(4,4)
    assert torch.is_complex(S)
    val=abs(S[2,0])**2+S[2,0].real
    val.backward()
    assert torch.isfinite(model.raw.grad).all()
    assert torch.count_nonzero(model.raw.grad)>0
