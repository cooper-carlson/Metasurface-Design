"""CPU-only EP mathematics / optimization tests; TORCWA CUDA runs in notebook."""
import numpy as np
import pytest
import torch
from scipy.linalg import sqrtm
from types import SimpleNamespace

from twofin import Config
import engine_reflection_ep as ep


def sample():
    return dict(L1=.21,W1=.075,theta1=.32,x1=-.045,y1=-.035,
                L2=.18,W2=.065,theta2=1.10,x2=.05,y2=.055,
                height=.32,kappa=0.0)


def unitary_reflection_ep():
    # Halmos unitary dilation of a complex symmetric contraction R.
    R = .2 * np.array([[1j,1],[1,-1j]],complex)
    X = sqrtm(np.eye(2)-R@R.conj().T).astype(complex)
    Y = sqrtm(np.eye(2)-R.conj().T@R).astype(complex)
    return np.block([[R,X],[Y,-R.conj().T]])


def test_reflection_ep_inside_lossless_reciprocal_full_s():
    S=unitary_reflection_ep()
    assert np.allclose(S.conj().T@S,np.eye(4),atol=1e-12)
    assert np.allclose(S,S.T,atol=1e-12)
    d=ep.diagnostics(S)
    passed,reasons=ep.screen_candidate(d)
    assert passed, reasons
    for k in ('R11','R22'):
        assert d[k]['gap']<1e-7
        assert d[k]['overlap']>.999999
        assert d[k]['jordan_strength']>.5
    assert np.allclose(np.linalg.eigvals(S).imag,np.linalg.eigvals(S).imag)  # finite


def test_scalar_guard_and_target_modes():
    R=.2*np.eye(2,dtype=complex)
    d=ep.block_diagnostics(R)
    assert ep.discrim_numpy(R)==0
    assert d['jordan_strength']==0
    S=torch.zeros((4,4),dtype=torch.complex128)
    for target in ep.TARGETS:
        L,terms=ep.loss_torch(S,target=target)
        assert torch.isfinite(L)
        assert terms['R11_jordan']<1e-3
    with pytest.raises(ValueError): ep.loss_torch(S,target='transmission')


def test_circular_change_of_basis_preserves_eigenvalues_overlap():
    S=unitary_reflection_ep()
    A,B=ep.reflection_blocks(S)
    for original,transformed in zip((S[:2,:2],S[2:,2:]),(A,B)):
        assert np.allclose(np.sort_complex(np.linalg.eigvals(original)),
                           np.sort_complex(np.linalg.eigvals(transformed)),atol=1e-8)
    assert not np.allclose(A,B)  # separate incidence sides


def test_reflection_blocks_independent_and_transmission_can_be_nonnormal():
    rng=np.random.default_rng(14)
    X=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    Q,R=np.linalg.qr(X)
    Q=Q@np.diag(np.diag(R)/np.abs(np.diag(R)))
    S=Q.T@Q
    assert np.allclose(S.conj().T@S,np.eye(4))
    assert np.allclose(S,S.T)
    assert ep.block_diagnostics(S[2:,:2])['overlap']>.1
    A,B=ep.reflection_blocks(S)
    assert not np.allclose(A,B)


def test_loss_torch_gradient():
    x=torch.tensor(.15,dtype=torch.float64,requires_grad=True)
    i=torch.tensor(1j,dtype=torch.complex128)
    one=torch.ones((),dtype=torch.complex128)
    a=(x+0.07).to(torch.complex128)
    z=one*0
    R=torch.stack([torch.stack([a*i,one*.18]),
                   torch.stack([one*.18,-a*i])])
    T=one*.5*torch.eye(2,dtype=torch.complex128)
    S=torch.cat([torch.cat([R,T],dim=1),
                 torch.cat([T,R],dim=1)],dim=0)
    value,_=ep.loss_torch(S)
    value.backward()
    assert torch.isfinite(x.grad)
    assert abs(x.grad.item())>1e-6


def test_lossless_bounds_and_material_guard(monkeypatch):
    p=sample()
    model=ep.LosslessDesign(p,device='cpu')
    assert model.physical()['kappa']==0
    assert ep.physical_config(Config(),p).n_fin_kappa == 0
    with pytest.raises(ValueError):ep.physical_config(Config(),{**p,'kappa':.01})
    params=ep.to_tensors(p)
    with pytest.raises(ValueError):ep.simulate_lossless(Config(n_fin_kappa=.01),None,params)
    with pytest.raises(ValueError):ep.simulate_lossless(Config(),None,{**params,'kappa':torch.tensor(.01)})
    calls=[]
    def fake_solver(cfg,pp,settings,**kwargs):
        calls.append(kwargs)
        theta=pp['theta1'].to(torch.complex64)
        R=torch.diag(torch.stack([theta*.1,theta*.2])).to(torch.complex64)
        T=torch.eye(2,dtype=torch.complex64)*.8
        return torch.cat([torch.cat([R,T],dim=1),torch.cat([T,R],dim=1)],dim=0)
    monkeypatch.setattr(ep.engine_rcwa,'solve_torch',fake_solver)
    val,S,_=ep.simulate_lossless(Config(),SimpleNamespace(device='cpu'),params)
    assert torch.isfinite(val)
    assert kwargs_are_fixed_ports(calls[0])
    assert S.shape==(4,4)


def kwargs_are_fixed_ports(kwargs):
    return kwargs['ports']==ep.FIXED_PORTS


def test_adam_checkpoint_path_with_mock(monkeypatch,tmp_path):
    def fake_solver(cfg,p,settings,**kwargs):
        th=p['theta1'].to(torch.complex64)
        R=torch.stack([torch.stack([1j*th, .15+0*th]),
                       torch.stack([.15+0*th,-1j*th])])
        T=torch.eye(2,dtype=torch.complex64)*.9
        return torch.cat([torch.cat([R,T],dim=1),
                          torch.cat([T,R],dim=1)],dim=0)
    monkeypatch.setattr(ep.engine_rcwa,'solve_torch',fake_solver)
    out=ep.adam_refine(Config(),SimpleNamespace(device='cpu',order=2),
                       sample(),target='both',iterations=4,patience=9,verbose=False)
    assert np.isfinite(out['loss'])
    assert out['params']['kappa']==0
    assert len(out['history'])==4
    saved=ep.save_result(tmp_path/'r.npz',out,metadata={'target':'both'})
    loaded=ep.load_result(saved)
    assert loaded['metadata']['target']=='both'
    assert np.allclose(out['S'],loaded['S'])
