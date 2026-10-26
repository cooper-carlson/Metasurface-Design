import numpy as np
import pytest
import torch

from twofin import Config
from engine_transmission_ep import (
    KEYS, circular_unitary, circular_blocks, discrim_numpy,
    block_diagnostics, diagnostics, screen_candidate, loss_torch,
    LosslessDesign, physical_config, simulate_lossless, save_result,load_result,
    track_pair,loop_exchange,
)


def sample():
    p={
      'L1':.21,'W1':.075,'theta1':.32,'x1':-.045,'y1':-.035,
      'L2':.18,'W2':.065,'theta2':1.10,'x2':.05,'y2':.055,
      'height':.32,'kappa':0.0
    }
    return p


def test_ep_not_scalar():
    N=.2*np.array([[1j,1],[1,-1j]])
    Z=.2*np.eye(2)
    assert abs(discrim_numpy(N)) < 1e-15
    assert abs(discrim_numpy(Z)) < 1e-15
    ep=block_diagnostics(N)
    ordinary=block_diagnostics(Z)
    assert ep['jordan_strength']>.5
    assert ep['rank_ratio']<1e-10
    assert ep['overlap']>.999
    assert ordinary['jordan_strength']==0


def test_circular_basis_eigenvalues():
    U=circular_unitary()
    assert np.allclose(U.conj().T@U,np.eye(2))
    J=np.array([[.15+.2j,.1-.05j],[.3+.08j,.25-.10j]])
    S=np.zeros((4,4),complex)
    S[2:,:2]=J;S[:2,2:]=J.T
    A,B=circular_blocks(S)
    assert np.allclose(np.sort_complex(np.linalg.eigvals(J)),np.sort_complex(np.linalg.eigvals(A)))
    assert np.allclose(np.sort_complex(np.linalg.eigvals(J.T)),np.sort_complex(np.linalg.eigvals(B)))
    d=diagnostics(S)
    assert d['transmission_transpose_error']<1e-14


def test_loss_torch_autograd_and_scalar_guard():
    S=torch.eye(4,dtype=torch.complex128)*.2
    v,m=loss_torch(S)
    assert torch.isfinite(v)
    assert m['forward_jordan']<.001
    p=torch.tensor(.12,dtype=torch.float64,requires_grad=True)
    T=torch.stack([torch.stack([p.to(torch.complex128)*1j,torch.ones_like(p).to(torch.complex128)*.1]),
                   torch.stack([torch.ones_like(p).to(torch.complex128)*.1,-p.to(torch.complex128)*1j])])
    S2=torch.cat([torch.cat([torch.zeros_like(T),T.T],dim=1),
                  torch.cat([T,torch.zeros_like(T)],dim=1)],dim=0)
    val,_=loss_torch(S2)
    val.backward()
    assert torch.isfinite(p.grad) and abs(float(p.grad))>1e-6


def test_bounded_model_real_and_in_cell():
    model=LosslessDesign(sample(),device='cpu')
    p=model.values()
    assert set(p)==set(KEYS)|{'kappa'}
    assert float(p['kappa'])==0
    assert p['kappa'].requires_grad is False
    c=physical_config(Config(),model.physical())
    assert c.n_fin_kappa==0
    c.validate()
    with pytest.raises(ValueError):
        physical_config(Config(),{**sample(),'kappa':.01})


def test_simulator_rejects_complex_index(monkeypatch):
    import engine_transmission_ep as ep
    p={k:torch.tensor(v) for k,v in sample().items()}
    with pytest.raises(ValueError):
        simulate_lossless(Config(n_fin_kappa=.001),None,p)
    with pytest.raises(ValueError):
        simulate_lossless(Config(),None,{**p,'kappa':torch.tensor(.001)})
    def fake_solver(cfg,params,settings,**kwargs):
        a=params['theta1'].to(torch.complex64)
        Z=torch.zeros((4,4),dtype=torch.complex64)
        Z[2,0]=a
        Z[0,2]=a
        return Z
    monkeypatch.setattr(ep,'solve_torch',fake_solver)
    val,S,_=simulate_lossless(Config(),None,p)
    assert float(val)>=0
    assert S.shape==(4,4)


def test_checkpoint_and_loop(tmp_path):
    from engine_transmission_ep import load_result,save_result
    S=np.eye(4,dtype=complex)*(.25+.1j)
    result=dict(params=sample(),S=S,history=[{'loss':1.5}],loss=1.5)
    path=save_result(tmp_path/'result.npz',result,{'test':'ok'})
    restored=load_result(path)
    assert restored['metadata']=={'test':'ok'}
    assert np.allclose(restored['S'],S)
    # Square root sheet exchange around a 2x2 defective matrix.
    angles=np.linspace(0,2*np.pi,65)
    matrices=[np.array([[0,1],[.02*np.exp(1j*t),0]]) for t in angles]
    tr=track_pair(matrices)
    assert loop_exchange(tr)


def test_adam_path_mock_solver(monkeypatch):
    import engine_transmission_ep as ep
    from types import SimpleNamespace
    def fake_solver(cfg, params, settings, **kwargs):
        # Differentiable, reciprocal, synthetic S; no TORCWA dependency.
        theta=params['theta1']
        x=(.22+.05*torch.sin(theta)).to(torch.complex64)
        zero=torch.zeros((),device=x.device,dtype=x.dtype)
        J=torch.stack([torch.stack([x,zero]),torch.stack([zero,.2+x*0])])
        Z=torch.zeros_like(J)
        return torch.cat([torch.cat([Z,J.T],1),torch.cat([J,Z],1)],0)
    monkeypatch.setattr(ep,'solve_torch',fake_solver)
    setting=SimpleNamespace(device='cpu',order=2)
    result=ep.adam_refine(Config(),setting,sample(),iterations=3,patience=3,verbose=False)
    assert np.isfinite(result['loss'])
    assert result['params']['kappa']==0
    assert len(result['history'])==3
