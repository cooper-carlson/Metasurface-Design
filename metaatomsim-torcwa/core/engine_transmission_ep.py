"""Lossless two-fin transmission-Jones EP inverse-design helpers.

Requires the sibling modules from TwoFin_EP_Inverse_Design. This file never
modifies the previous absorbing/full-S EP optimization.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.stats import qmc

from engine_inverse import BOUNDS, FIXED_PORTS, enforce_unit_cell, candidate_to_config
from engine_rcwa import solve_torch

# Fixed kappa=0 is NOT an optimizer variable.
KEYS = tuple(k for k in BOUNDS if k != 'kappa')
VERSION = 'lossless-transmission-ep-v1'


def circular_unitary():
    """Jones lab basis [R,L] with normalized Stokes S3(R)=+1."""
    return np.array([[1., 1.], [1j, -1j]]) / np.sqrt(2)


def circular_blocks(S):
    """Change x/y to fixed laboratory circular basis at both ports."""
    U = circular_unitary()
    J21 = np.asarray(S)[2:4, :2]
    J12 = np.asarray(S)[:2, 2:4]
    return U.conj().T @ J21 @ U, U.conj().T @ J12 @ U


def discrim_numpy(J):
    a, b, c, d = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    return (a-d)**2 + 4*b*c


def discrim_torch(J):
    return (J[0,0]-J[1,1])**2 + 4*J[0,1]*J[1,0]



def loss_torch(
    S,
    *,
    min_jordan=0.08,
    min_power=0.12,
    min_eigen_power=0.05,
    discriminant_weight=1.0,
    jordan_weight=15.0,
    power_weight=10.0,
    eigen_power_weight=10.0,
    backward_weight=0.0,
    unitarity_weight=0.0,
    reciprocity_weight=0.0,
):
    """
    Differentiable objective for a transmission Jones-matrix EP2.

    Targets:
      1. Degenerate transmission eigenvalues.
      2. Nonzero traceless part (nontrivial Jordan block).
      3. Non-negligible total transmission.
      4. Nonzero transmission eigenvalue.

    The full 4x4 scattering matrix is used. Physical consistency
    metrics are reported but not penalized by default.

    This searches for scattering EPs, not resonant-mode EPs.
    """
    import torch

    if S.shape != (4, 4) or not torch.is_complex(S):
        raise ValueError("Expected a complex 4x4 scattering matrix")

    if backward_weight < 0:
        raise ValueError("backward_weight must be nonnegative")

    # Transform transmission blocks into the circular basis.
    U = torch.as_tensor(
        circular_unitary(), device=S.device, dtype=S.dtype
    )

    Tf = U.conj().T @ S[2:4, 0:2] @ U
    Tb = U.conj().T @ S[0:2, 2:4] @ U

    def block_objective(J):
        eye = torch.eye(2, device=J.device, dtype=J.dtype)

        # Repeated eigenvalue at degeneracy.
        mu = torch.trace(J) / 2

        # Traceless component: nonzero at a nontrivial EP.
        Q = J - mu * eye

        # Total transmission strength.
        norm2 = torch.abs(J).square().sum()
        scale = norm2 + 1e-10

        # Characteristic discriminant, zero at degeneracy.
        delta = (
            (J[0, 0] - J[1, 1])**2
            + 4 * J[0, 1] * J[1, 0]
        )

        disc2 = torch.abs(delta).square() / scale.square()
        disc = torch.sqrt(disc2 + 1e-30)

        # Distance from a scalar matrix, normalized by ||J||.
        jordan = torch.sqrt(
            torch.abs(Q).square().sum() + 1e-20
        ) / torch.sqrt(scale)

        # Mean transmitted power for two orthogonal inputs.
        power = norm2 / 2

        # Eigenpolarization transmitted power at an exact EP.
        eigen_power = torch.abs(mu).square()

        # Individual optical penalties.
        disc_loss = torch.log1p(20.0 * disc2)

        jordan_loss = torch.relu(
            min_jordan - jordan
        ).square()

        power_loss = torch.relu(
            min_power - power
        ).square()

        eigen_loss = torch.relu(
            min_eigen_power - eigen_power
        ).square()

        total = (
            discriminant_weight * disc_loss
            + jordan_weight * jordan_loss
            + power_weight * power_loss
            + eigen_power_weight * eigen_loss
        )

        metrics = dict(
            disc=disc,
            jordan=jordan,
            power=power,
            eigen_power=eigen_power,
            disc_loss=disc_loss,
            jordan_loss=jordan_loss,
            power_loss=power_loss,
            eigen_loss=eigen_loss,
            total=total,
        )

        return total, metrics

    # Forward and backward transmission diagnostics.
    forward_loss, f = block_objective(Tf)
    backward_loss, b = block_objective(Tb)

    # Optimize forward transmission unless otherwise requested.
    optical_loss = (
        forward_loss + backward_weight * backward_loss
    ) / (1.0 + backward_weight)

    # Physical consistency checks on the full scattering matrix.
    I = torch.eye(4, device=S.device, dtype=S.dtype)

    unitarity = torch.linalg.matrix_norm(
        S.conj().T @ S - I, ord="fro"
    ).square()

    reciprocity = torch.linalg.matrix_norm(
        S - S.T, ord="fro"
    ).square()

    # Include physical penalties only when explicitly enabled.
    objective = (
        optical_loss
        + unitarity_weight * unitarity
        + reciprocity_weight * reciprocity
    )

    components = dict(
        # Existing diagnostic names retained.
        forward_disc=f["disc"],
        backward_disc=b["disc"],
        forward_jordan=f["jordan"],
        backward_jordan=b["jordan"],
        forward_power=f["power"],
        backward_power=b["power"],
        unitarity=unitarity,
        reciprocity=reciprocity,

        # Additional eigenchannel diagnostics.
        forward_eigen_power=f["eigen_power"],
        backward_eigen_power=b["eigen_power"],

        # Individual forward loss components.
        discriminant_loss=f["disc_loss"],
        jordan_loss=f["jordan_loss"],
        power_loss=f["power_loss"],
        eigen_power_loss=f["eigen_loss"],

        # Aggregate losses.
        forward_loss=forward_loss,
        backward_loss=backward_loss,
        optical_loss=optical_loss,
        total_loss=objective,
    )

    return objective, components

class LosslessDesign:
    """Bounded independent geometry; corners stay inside periodic cell."""
    def __init__(self, initial, *, device='cpu', dtype=None):
        import torch
        dtype=dtype or torch.float32
        feasible=enforce_unit_cell(initial)
        initial_raw=[]
        for key in KEYS:
            if key[0] in 'xy' and key[-1] in '12':
                initial_raw.append(0.)
            else:
                lo,hi=BOUNDS[key]
                p=np.clip((float(feasible[key])-lo)/(hi-lo),1e-3,1-1e-3)
                initial_raw.append(float(np.log(p/(1-p))))
        for i in (1,2):
            t=float(feasible[f'theta{i}'])
            ex=(feasible[f'L{i}']*abs(np.cos(t))+feasible[f'W{i}']*abs(np.sin(t)))/2
            ey=(feasible[f'L{i}']*abs(np.sin(t))+feasible[f'W{i}']*abs(np.cos(t)))/2
            for axis, extent in (('x',ex),('y',ey)):
                limit=max(.001,min(.13,.2-.006-extent))
                initial_raw[KEYS.index(f'{axis}{i}')]=float(np.arctanh(
                    np.clip(feasible[f'{axis}{i}']/limit,-.995,.995)))
        self.raw=torch.nn.Parameter(torch.tensor(initial_raw,device=device,dtype=dtype))

    def values(self):
        import torch
        out={}
        for j,key in enumerate(KEYS):
            if key[0] in 'xy' and key[-1] in '12':
                continue
            lo,hi=BOUNDS[key]
            out[key]=lo+(hi-lo)*torch.sigmoid(self.raw[j])
        for i in (1,2):
            t=out[f'theta{i}']
            c=torch.sqrt(torch.cos(t)**2+1e-12)
            s=torch.sqrt(torch.sin(t)**2+1e-12)
            ex=(out[f'L{i}']*c+out[f'W{i}']*s)/2
            ey=(out[f'L{i}']*s+out[f'W{i}']*c)/2
            bx=torch.clamp(torch.minimum(torch.full_like(ex,.13),.2-.006-ex),min=.001)
            by=torch.clamp(torch.minimum(torch.full_like(ey,.13),.2-.006-ey),min=.001)
            out[f'x{i}']=bx*torch.tanh(self.raw[KEYS.index(f'x{i}')])
            out[f'y{i}']=by*torch.tanh(self.raw[KEYS.index(f'y{i}')])
        out['kappa']=torch.zeros_like(out['L1'])  # strictly real material index
        return out

    def physical(self):
        return {k:float(v.detach().cpu()) for k,v in self.values().items()}


def to_tensors(physical, *, device='cpu', dtype=None):
    import torch
    if abs(float(physical.get('kappa', 0.0))) > 0:
        raise ValueError('Lossless design cannot accept nonzero kappa')
    dtype=dtype or torch.float32
    return {k:torch.as_tensor(float(physical[k]) if k!='kappa' else 0.,
                             device=device,dtype=dtype)
            for k in (*KEYS, 'kappa')}


def simulate_lossless(cfg, settings, params, *, edge_um=.006):
    if cfg.n_fin_kappa != 0:
        raise ValueError('Lossless transmission optimization requires cfg.n_fin_kappa=0')
    # A caller cannot silently introduce nonzero absorption.
    if float(params['kappa'].detach().cpu()) != 0:
        raise ValueError('The optimizer requires kappa=0 exactly')
    S=solve_torch(cfg,params,settings,edge_um=edge_um,ports=FIXED_PORTS)
    val,diag=loss_torch(S)
    return val,S,diag


def block_diagnostics(J):
    """2x2 spectral and rank-defect diagnostics in a specified Jones basis."""
    J=np.asarray(J,dtype=np.complex128)
    eig,V=np.linalg.eig(J)
    V/=np.linalg.norm(V,axis=0)
    norm=max(float(np.linalg.norm(J,'fro')),1e-12)
    mu=complex(np.trace(J)/2)
    Q=J-mu*np.eye(2)
    sv=np.linalg.svd(Q,compute_uv=False)
    return dict(eigenvalues=eig, eigenvectors=V, midpoint=mu,
        gap=float(abs(eig[0]-eig[1])/norm),
        overlap=float(abs(np.vdot(V[:,0],V[:,1]))),
        discriminant=complex(discrim_numpy(J)),
        discriminant_norm=float(abs(discrim_numpy(J))/norm**2),
        jordan_strength=float(np.linalg.norm(Q,'fro')/norm),
        rank_ratio=float(sv[-1]/max(sv[0],1e-12)),
        largest_null_sv=float(sv[0]/norm),
        avg_transmission=float(np.sum(np.abs(J)**2)/2))


def diagnostics(S):
    """Full lossless diagnostics and both circular-basis transmission blocks."""
    S=np.asarray(S,np.complex128)
    if S.shape != (4,4):
        raise ValueError('Expected 4x4 complex scattering matrix')
    forward,backward=circular_blocks(S)
    d21,d12=block_diagnostics(forward),block_diagnostics(backward)
    unitary=np.linalg.norm(S.conj().T@S-np.eye(4),ord=2)
    reciprocity=np.linalg.norm(S-S.T,ord='fro')
    return dict(T21_circular=forward,T12_circular=backward,
        T21=d21,T12=d12,unitarity_error=float(unitary),
        reciprocity_error=float(reciprocity),
        transmission_transpose_error=float(np.linalg.norm(S[:2,2:]-S[2:,:2].T,'fro')),
        max_power_gain=float(np.linalg.svd(S,compute_uv=False)[0]**2))


def screen_candidate(d, *, gap_tol=2e-3, overlap_tol=.995,
                     jordan_floor=.07, rank_ratio_tol=1e-3,
                     unitarity_tol=.01, reciprocity_tol=.01):
    """Numerical near-EP screen, not a proof of an exact EP."""
    reasons=[]
    for name in ('T21','T12'):
        b=d[name]
        if b['gap']>=gap_tol: reasons.append(f'{name}: eigenvalue gap too large')
        if b['overlap']<=overlap_tol: reasons.append(f'{name}: eigenvectors not coalesced')
        if b['jordan_strength']<=jordan_floor: reasons.append(f'{name}: nearly scalar')
        if b['rank_ratio']>=rank_ratio_tol: reasons.append(f'{name}: not near rank-one')
    if d['unitarity_error']>=unitarity_tol: reasons.append('full S not unitary within tolerance')
    if d['reciprocity_error']>=reciprocity_tol: reasons.append('reciprocity residual too large')
    return len(reasons)==0,reasons


def report(S, title='Scattering candidate'):
    d=diagnostics(S)
    print(f'\n{title}  [circular basis R=(x+iy)/sqrt(2)]')
    print(f'S†S-I (spectral norm): {d["unitarity_error"]:.3e}; '
          f'S-Sᵀ (Frobenius): {d["reciprocity_error"]:.3e}')
    for name in ('T21','T12'):
        b=d[name]
        print(f'{name}: gap={b["gap"]:.3e}, overlap={b["overlap"]:.6f}, '
              f'disc_norm={b["discriminant_norm"]:.3e}, '
              f'Jordan={b["jordan_strength"]:.3f}, rank ratio={b["rank_ratio"]:.3e}, '
              f'Tavg={b["avg_transmission"]:.4f}')
        print('  λ =',np.round(b['eigenvalues'],7))
    passed,reasons=screen_candidate(d)
    print('NEAR-EP SCREEN:', 'PASS (not exact EP proof)' if passed else 'NOT VALIDATED')
    if reasons: print('  '+'; '.join(reasons))
    return d,passed


def generate_seeds(cfg, settings, *, count=16, keep=3, seed=21,
                   edge_um=.008, verbose=True):
    """Sobol global screen. Each seed uses one full-S RCWA solve."""
    import torch
    if cfg.n_fin_kappa != 0: raise ValueError('n_fin_kappa must be zero')
    sampler=qmc.Sobol(d=len(KEYS),scramble=True,seed=seed)
    samples=sampler.random_base2(int(np.ceil(np.log2(max(2,count)))))[:count]
    found=[]
    for idx,point in enumerate(samples):
        p={k:BOUNDS[k][0]+float(point[j])*(BOUNDS[k][1]-BOUNDS[k][0])
           for j,k in enumerate(KEYS)}
        p=enforce_unit_cell(p)
        p['kappa']=0.
        try:
            with torch.no_grad():
                value,S,_=simulate_lossless(cfg,settings,to_tensors(p,device=settings.device),edge_um=edge_um)
                if not bool(torch.isfinite(value)): continue
                d=diagnostics(S.detach().cpu().numpy())
                if d['unitarity_error']>0.1 or d['max_power_gain']>1.10: continue
                found.append((float(value.detach().cpu()),p,d['T21']['overlap']))
        except (RuntimeError,ValueError) as exc:
            if verbose:print('Skipping seed',idx,str(exc)[:130])
        if verbose and (idx+1)%8==0:print(f'Screened {idx+1}/{count}; viable {len(found)}')
    if not found: raise RuntimeError('No viable Sobol seeds; check RCWA and material conventions')
    found.sort(key=lambda x:x[0])
    return [r[1] for r in found[:keep]],found


def adam_refine(cfg, settings, initial, *, iterations=80, lr=.035,
                edge_um=.006, patience=30, verbose=True):
    import torch
    model=LosslessDesign(initial,device=settings.device)
    opt=torch.optim.Adam([model.raw],lr=lr)
    best=np.inf; best_it=-1; best_values=None; best_S=None; hist=[]
    begin=time.perf_counter()
    for it in range(iterations):
        opt.zero_grad(set_to_none=True)
        objective,S,parts=simulate_lossless(cfg,settings,model.values(),edge_um=edge_um)
        if not bool(torch.isfinite(objective)):break
        objective.backward()
        if model.raw.grad is None or not bool(torch.isfinite(model.raw.grad).all()):break
        torch.nn.utils.clip_grad_norm_([model.raw],1.0)
        v=float(objective.detach().cpu())
        row=dict(iteration=it,loss=v,
            gap_proxy=float(parts['forward_disc'].detach().cpu()),
            strength=float(parts['forward_jordan'].detach().cpu()),
            unitarity=float(parts['unitarity'].detach().cpu()))
        hist.append(row)
        if v<best-1e-6:
            best=v; best_it=it; best_values=model.physical()
            best_S=S.detach().cpu().numpy()
        opt.step()
        if verbose and (it+1)%10==0:
            print(f'  {it+1:3}/{iterations}: loss={v:.4g}, '
                  f'disc={row["gap_proxy"]:.3e}, strength={row["strength"]:.3f}')
        if it-best_it>=patience:break
    if best_values is None:raise RuntimeError('No valid gradient steps')
    return dict(params=best_values, S=best_S, loss=best, history=hist,
                seconds=time.perf_counter()-begin)


def save_result(path,result,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,S=result['S'],params=json.dumps(result['params']),
        history=json.dumps(result['history']),loss=result['loss'],
        metadata=json.dumps(metadata or {}))
    return path


def load_result(path):
    with np.load(path,allow_pickle=False) as f:
        return dict(S=f['S'],params=json.loads(str(f['params'])),
            history=json.loads(str(f['history'])),loss=float(f['loss']),
            metadata=json.loads(str(f['metadata'])))


def physical_config(cfg,params):
    if abs(float(params.get('kappa',0)))>0:
        raise ValueError('Nonzero loss is forbidden in this notebook')
    return candidate_to_config(cfg,{**params,'kappa':0.})


def track_pair(matrices):
    """Track two eigenvalues by nearest continuous branch along a parameter loop."""
    all_values=np.array([np.linalg.eigvals(J) for J in matrices])
    tracked=[all_values[0]]
    for pair in all_values[1:]:
        if np.linalg.norm(pair-tracked[-1]) > np.linalg.norm(pair[::-1]-tracked[-1]):
            pair=pair[::-1]
        tracked.append(pair)
    return np.array(tracked)


def loop_exchange(tracked):
    """True when branch continuation ends closer to a swap than identity.

    Only meaningful when samples form a closed loop with sufficiently small steps
    and eigenvalues are well separated on the path.
    """
    start,end=tracked[0],tracked[-1]
    return bool(np.linalg.norm(end-start[::-1])<np.linalg.norm(end-start))
