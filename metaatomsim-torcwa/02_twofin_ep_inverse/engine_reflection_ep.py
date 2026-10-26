"""Lossless two-fin reflection-Jones EP search, parallel to transmission EP.

Reuses the existing differentiable TORCWA engine and bounded geometry mapping.
R11 and R22 are independent EP objectives; neither follows from the other.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import engine_rcwa
from engine_inverse import BOUNDS, FIXED_PORTS, enforce_unit_cell
from engine_transmission_ep import (
    KEYS, LosslessDesign, to_tensors, physical_config,
    circular_unitary, discrim_numpy, discrim_torch,
    block_diagnostics, track_pair, loop_exchange,
    save_result, load_result,
)

VERSION = 'lossless-reflection-ep-v1'
TARGETS = ('both', 'R11', 'R22')


def reflection_blocks(S):
    """R11 and R22 in fixed lab circular R/L basis (unitary similarity)."""
    S = np.asarray(S)
    U = circular_unitary()
    return U.conj().T @ S[:2, :2] @ U, U.conj().T @ S[2:, 2:] @ U


def _target_names(target):
    if target not in TARGETS:
        raise ValueError(f'target must be one of {TARGETS}')
    return ('R11', 'R22') if target == 'both' else (target,)


def loss_torch(S, *, target='both', min_jordan=0.13, min_power=0.006):
    """Differentiable EP proximity heuristic with scalar/power safeguards.

    Normalized discriminant zero is necessary, not sufficient, for a 2x2 EP.
    For target='both', smooth maximum balances the two independent objectives.
    """
    import torch
    _target_names(target)
    if tuple(S.shape) != (4, 4):
        raise ValueError('S must have shape (4,4)')
    U = torch.as_tensor(circular_unitary(), device=S.device, dtype=S.dtype)
    blocks = {
        'R11': U.conj().T @ S[:2, :2] @ U,
        'R22': U.conj().T @ S[2:, 2:] @ U,
    }
    terms, pieces = {}, {}
    for name, J in blocks.items():
        norm2 = torch.sum(torch.abs(J)**2) + 1e-12
        mu = torch.trace(J) / 2
        Q = J - mu * torch.eye(2, dtype=J.dtype, device=J.device)
        disc = torch.abs(discrim_torch(J)) / norm2
        strength = torch.sqrt(torch.sum(torch.abs(Q)**2) + 1e-20) / torch.sqrt(norm2)
        power = norm2 / 2
        # Smooth barriers avoid winning via a scalar or vanishing reflection.
        terms[name] = (torch.log1p(30 * disc**2) +
                       18 * torch.relu(min_jordan - strength)**2 +
                       12 * torch.relu(min_power - power)**2)
        pieces[f'{name}_disc'] = disc
        pieces[f'{name}_jordan'] = strength
        pieces[f'{name}_power'] = power
    requested = torch.stack([terms[name] for name in _target_names(target)])
    if len(requested) == 2:
        beta = 12.0
        objective = torch.logsumexp(beta * requested, dim=0) / beta - np.log(2) / beta
    else:
        objective = requested[0]
    I = torch.eye(4, dtype=S.dtype, device=S.device)
    unitarity = torch.linalg.matrix_norm(S.conj().T @ S - I, ord='fro')**2
    reciprocity = torch.linalg.matrix_norm(S - S.T, ord='fro')**2
    objective = objective + 0.5 * unitarity + 0.5 * reciprocity
    return objective, {**pieces, 'unitarity': unitarity, 'reciprocity': reciprocity}


def simulate_lossless(cfg, settings, params, *, target='both', edge_um=0.006):
    if float(cfg.n_fin_kappa) != 0 or np.iscomplexobj(cfg.n_fin):
        raise ValueError('The lossless experiment requires strictly real fin index')
    if float(params['kappa'].detach().cpu()) != 0:
        raise ValueError('Nonzero extinction coefficient is forbidden')
    S = engine_rcwa.solve_torch(cfg, params, settings, edge_um=edge_um, ports=FIXED_PORTS)
    value, parts = loss_torch(S, target=target)
    return value, S, parts


def diagnostics(S):
    S = np.asarray(S, dtype=np.complex128)
    if S.shape != (4,4):
        raise ValueError('Expected 4x4 complex S')
    R11, R22 = reflection_blocks(S)
    unitary = np.linalg.norm(S.conj().T @ S - np.eye(4), ord=2)
    reciprocal = np.linalg.norm(S - S.T, ord='fro')
    return dict(R11_circular=R11, R22_circular=R22,
                R11=block_diagnostics(R11), R22=block_diagnostics(R22),
                unitarity_error=float(unitary),
                reciprocity_error=float(reciprocal),
                R11_symmetry_error=float(np.linalg.norm(S[:2,:2]-S[:2,:2].T)),
                R22_symmetry_error=float(np.linalg.norm(S[2:,2:]-S[2:,2:].T)),
                max_power_gain=float(np.linalg.svd(S, compute_uv=False)[0]**2))


def screen_candidate(d, *, target='both', gap_tol=2e-3,
                     overlap_tol=.995, jordan_floor=.13,
                     rank_ratio_tol=1e-3, power_floor=.006,
                     unitarity_tol=.01, reciprocity_tol=.01):
    """Numerical near-EP screen. An exact EP also needs convergence/topology."""
    reasons = []
    for name in _target_names(target):
        b = d[name]
        if b['gap'] >= gap_tol:
            reasons.append(f'{name}: normalized eigenvalue gap too large')
        if b['overlap'] <= overlap_tol:
            reasons.append(f'{name}: eigenvectors not nearly coalesced')
        if b['jordan_strength'] <= jordan_floor:
            reasons.append(f'{name}: nearly scalar')
        if b['rank_ratio'] >= rank_ratio_tol:
            reasons.append(f'{name}: shifted Jones matrix not near rank-one')
        if b['avg_transmission'] <= power_floor:
            reasons.append(f'{name}: reflection power too small')
    if d['unitarity_error'] >= unitarity_tol:
        reasons.append('full S violates expected unitarity')
    if d['reciprocity_error'] >= reciprocity_tol:
        reasons.append('full S violates expected reciprocity')
    return not reasons, reasons


def report(S, title='Reflection candidate', *, target='both'):
    d = diagnostics(S)
    print(f'\n{title}; target={target} (lab circular R=(x+iy)/sqrt(2))')
    print(f'||S†S-I||2={d["unitarity_error"]:.3e}, '
          f'||S-Sᵀ||F={d["reciprocity_error"]:.3e}')
    for name in ('R11', 'R22'):
        b = d[name]
        print(f'{name}: gap={b["gap"]:.3e}, overlap={b["overlap"]:.6f}, '
              f'|Δ|/||R||²={b["discriminant_norm"]:.3e}, '
              f'Jordan={b["jordan_strength"]:.3f}, rank ratio={b["rank_ratio"]:.3e}, '
              f'mean reflectance={b["avg_transmission"]:.4f}')
        print('  Eigenvalues:', np.round(b['eigenvalues'], 7))
    passed, reasons = screen_candidate(d, target=target)
    print('NEAR-EP SCREEN:', 'PASS (not proof)' if passed else 'NOT VALIDATED')
    if reasons:
        print('  ' + '; '.join(reasons))
    return d, passed


def generate_seeds(cfg, settings, *, target='both', count=16, keep=3,
                   seed=23, edge_um=.01, verbose=True):
    """One forward RCWA simulation per Sobol seed; no backprop graph."""
    import torch
    _target_names(target)
    if cfg.n_fin_kappa != 0:
        raise ValueError('Strictly real refractive index required')
    sampler = qmc.Sobol(d=len(KEYS), scramble=True, seed=seed)
    sample = sampler.random_base2(int(np.ceil(np.log2(max(count, 2)))))[:count]
    scored = []
    for i, u in enumerate(sample):
        p = {k: float(BOUNDS[k][0] + u[j]*(BOUNDS[k][1]-BOUNDS[k][0]))
             for j, k in enumerate(KEYS)}
        p = enforce_unit_cell(p)
        p['kappa'] = 0.0
        try:
            with torch.no_grad():
                value, S, _ = simulate_lossless(cfg, settings,
                    to_tensors(p, device=settings.device), target=target, edge_um=edge_um)
                if not bool(torch.isfinite(value)):
                    continue
                d = diagnostics(S.detach().cpu().numpy())
                if d['unitarity_error'] > 0.1 or d['max_power_gain'] > 1.1:
                    continue
                scored.append((float(value.detach().cpu()), p,
                    {key: d[key]['overlap'] for key in ('R11', 'R22')}))
        except (RuntimeError, ValueError) as exc:
            if verbose: print(f'Skipping seed {i}: {str(exc)[:150]}')
        if verbose and (i + 1) % 8 == 0:
            print(f'Screened {i+1}/{count}; valid {len(scored)}')
    scored.sort(key=lambda item: item[0])
    if not scored:
        raise RuntimeError('No viable seeds; inspect RCWA settings and conservation')
    return [x[1] for x in scored[:keep]], scored


def adam_refine(cfg, settings, initial, *, target='both', iterations=60,
                lr=.04, edge_um=.006, patience=25, verbose=True):
    import torch
    _target_names(target)
    design = LosslessDesign(initial, device=settings.device)
    optimizer = torch.optim.Adam([design.raw], lr=lr)
    best_loss = np.inf
    best_iter = -1
    best = None
    history = []
    start = time.perf_counter()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        objective, S, terms = simulate_lossless(cfg, settings, design.values(),
                                                target=target, edge_um=edge_um)
        if not bool(torch.isfinite(objective)):
            break
        objective.backward()
        grad = design.raw.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            break
        torch.nn.utils.clip_grad_norm_([design.raw], max_norm=1.)
        value = float(objective.detach().cpu())
        row = dict(iteration=iteration, loss=value,
                   R11_disc=float(terms['R11_disc'].detach().cpu()),
                   R22_disc=float(terms['R22_disc'].detach().cpu()),
                   R11_jordan=float(terms['R11_jordan'].detach().cpu()),
                   R22_jordan=float(terms['R22_jordan'].detach().cpu()),
                   unitarity=float(terms['unitarity'].detach().cpu()))
        history.append(row)
        if value < best_loss - 1e-7:
            best_loss, best_iter = value, iteration
            best = dict(params=design.physical(), S=S.detach().cpu().numpy(),
                        loss=value, history=history, seconds=0.)
        optimizer.step()
        if verbose and (iteration+1) % 10 == 0:
            print(f'  {iteration+1}/{iterations}: loss={value:.4g}, '
                  f'R11 disc={row["R11_disc"]:.3e}, R22 disc={row["R22_disc"]:.3e}')
        if iteration - best_iter >= patience:
            break
    if best is None:
        raise RuntimeError('No valid gradient step completed')
    best['history'] = history
    best['seconds'] = time.perf_counter() - start
    return best


def phase_scan(cfg, settings, base, *, axis1='theta1', axis2='height',
               target='both', n=9, radii=(np.deg2rad(3), .018)):
    """Optional hard-edge two-dimensional discriminant map for both sides."""
    from engine_rcwa import solve as solve_binary
    _target_names(target)
    xy = np.linspace(-1., 1., n)
    scans = {name: np.full((n, n), np.nan) for name in ('R11', 'R22')}
    for iy, dy in enumerate(xy):
        for ix, dx in enumerate(xy):
            p = dict(base)
            p[axis1] = float(base[axis1] + dx * radii[0])
            p[axis2] = float(base[axis2] + dy * radii[1])
            p = enforce_unit_cell(p)
            p['kappa'] = 0.
            try:
                S = solve_binary(physical_config(cfg, p), settings, ports=FIXED_PORTS)['S']
                d = diagnostics(S)
                for name in scans:
                    scans[name][iy, ix] = d[name]['discriminant_norm']
            except (ValueError, RuntimeError):
                continue
    return xy, scans

__all__ = ['VERSION', 'TARGETS', 'KEYS', 'BOUNDS', 'FIXED_PORTS',
           'LosslessDesign', 'to_tensors', 'physical_config', 'reflection_blocks',
           'discrim_numpy', 'discrim_torch', 'block_diagnostics', 'loss_torch',
           'simulate_lossless', 'diagnostics', 'screen_candidate', 'report',
           'generate_seeds', 'adam_refine', 'phase_scan', 'save_result',
           'load_result', 'track_pair', 'loop_exchange']
