"""Diagnostics and differentiable objectives for full four-channel scattering EPs.

An EP is a defective repeated eigenvalue of the complete scattering matrix,
not merely a zero of a 2x2 Jones-block discriminant.
"""
import itertools
import numpy as np


def _characteristic_coefficients_torch(A):
    """Return monic degree-four polynomial coefficients without eigendecomposition."""
    import torch
    A2 = A @ A
    t1, t2, t3 = torch.trace(A), torch.trace(A2), torch.trace(A2 @ A)
    return torch.stack((torch.ones_like(t1), -t1,
                        (t1*t1-t2)/2,
                        -(t1**3-3*t1*t2+2*t3)/6,
                        torch.linalg.det(A)))


def discriminant_torch(S):
    """Normalized |resultant(p,p')|, zero for any repeated eigenvalue.

    This is NOT by itself an EP test: ordinary/diabolic degeneracies also
    have vanishing discriminant. Differentiable without differentiating eig().
    """
    import torch
    if tuple(S.shape) != (4, 4):
        raise ValueError('Expected a full 4x4 scattering matrix')
    norm = torch.linalg.matrix_norm(S, ord='fro') / 2 + 1e-8
    coeff = _characteristic_coefficients_torch(S/norm)
    deriv = coeff[:4]*torch.arange(4, 0, -1, device=S.device)
    syl = torch.zeros((7, 7), device=S.device, dtype=S.dtype)
    for i in range(3):
        syl[i, i:i+5] = coeff
    for i in range(4):
        syl[i+3, i:i+4] = deriv
    return torch.abs(torch.linalg.det(syl))


def nonnormality_torch(S):
    import torch
    commutator = S.conj().T @ S - S @ S.conj().T
    return torch.linalg.matrix_norm(commutator, ord='fro') / (
        torch.linalg.matrix_norm(S, ord='fro')**2 + 1e-12)


def ep_loss_torch(S, *, min_nonnormality=0.08, min_transmission=0.06,
                  spectral_floor=1e-10):
    """Smooth heuristic candidate-search objective; NOT an EP certificate."""
    import torch
    disc = discriminant_torch(S)
    nn = nonnormality_torch(S)
    transmitted = torch.sum(torch.abs(S[2:,:2])**2)/2
    # Log discriminant avoids nearly-zero gradients far from a degeneracy.
    spectral = torch.log(disc + spectral_floor)
    nonnormal_penalty = 150 * torch.relu(min_nonnormality - nn)**2
    transmission_penalty = 15 * torch.relu(min_transmission - transmitted)**2
    # Reject false candidates created by gross numerical non-passivity.
    largest_power = torch.linalg.eigvalsh(S.conj().T@S)[-1].real
    passivity_penalty = 100*torch.relu(largest_power-1.005)**2
    reciprocity_penalty = 2*torch.linalg.matrix_norm(S-S.T,ord='fro')**2
    value = spectral + nonnormal_penalty + transmission_penalty + passivity_penalty + reciprocity_penalty
    return value, dict(discriminant=disc, nonnormality=nn,
                       forward_power=transmitted, spectral=spectral)


def ep_diagnostics(S):
    """Rank collision pairs using eigenvalue gap and eigenvector coalescence.

    Near a second-order EP, the coalescing eigenvectors become parallel,
    while the second-smallest singular value of S-lambda*I stays nonzero.
    Indicators are convergence-dependent, not an exact numerical proof.
    """
    S = np.asarray(S, dtype=np.complex128)
    if S.shape != (4, 4):
        raise ValueError('Expected full 4x4 S')
    eigenvalues, V = np.linalg.eig(S)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    scale = max(np.linalg.norm(S, ord=2), 1e-12)
    data=[]
    for i,j in itertools.combinations(range(4), 2):
        mid = (eigenvalues[i]+eigenvalues[j])/2
        singular = np.linalg.svd(S-mid*np.eye(4), compute_uv=False)
        overlap = float(abs(np.vdot(V[:,i],V[:,j])))
        gap = float(abs(eigenvalues[i]-eigenvalues[j])/scale)
        separation = min((abs(mid-eigenvalues[k])/scale for k in range(4)
                          if k not in (i,j)), default=float('inf'))
        # Small gap and high vector overlap both matter.
        score = gap + (1-overlap)
        data.append(dict(pair=(i,j),gap=gap,overlap=overlap,
                         midpoint=mid,other_separation=float(separation),
                         smallest_sv=float(singular[-1]/scale),
                         second_sv=float(singular[-2]/scale),score=score))
    pairs = sorted(data,key=lambda p:p['score'])
    comm = S.conj().T@S - S@S.conj().T
    nn = np.linalg.norm(comm,'fro') / max(np.linalg.norm(S,'fro')**2,1e-15)
    sv = np.linalg.svd(S,compute_uv=False)
    return dict(eigenvalues=eigenvalues, eigenvectors=V, pairs=pairs,
                best=pairs[0],nonnormality=float(nn),
                unitary_error=float(np.linalg.norm(S.conj().T@S-np.eye(4),ord=2)),
                reciprocity_error=float(np.linalg.norm(S-S.T,ord='fro')),
                max_power_gain=float(sv[0]**2),
                forward_transmission=float(np.sum(abs(S[2:,:2])**2)/2))


def print_ep_report(S, *, label='Candidate', gap_tol=1e-3,
                    overlap_tol=0.995, sv_separation_tol=1e-2,
                    min_other_separation=1e-2):
    d=ep_diagnostics(S); b=d['best']; i,j=b['pair']
    print(f'\n{label}: full 4x4 scattering EP diagnostics')
    print(f'Nearest/coalescing candidate: eigenvalues {i},{j}')
    print(f'  normalized eigenvalue gap: {b["gap"]:.3e} (target < {gap_tol:.1e})')
    print(f'  eigenvector overlap: {b["overlap"]:.6f} (target > {overlap_tol})')
    print(f'  second-smallest SV at mean eigenvalue: {b["second_sv"]:.3e}')
    print(f'  distance from other eigenvalues: {b["other_separation"]:.3e}')
    print(f'  non-normality: {d["nonnormality"]:.3e}')
    print(f'  reciprocity residual: {d["reciprocity_error"]:.3e}')
    print(f'  largest output/input power gain: {d["max_power_gain"]:.6f}')
    good=(b['gap']<gap_tol and b['overlap']>overlap_tol and
          b['second_sv']>sv_separation_tol and
          b['other_separation']>min_other_separation and
          d['max_power_gain']<=1.005)
    print('  Screening:', 'NEAR-EP CANDIDATE (requires convergence + encircling)' if good
          else 'NOT A VALIDATED FULL-S EP')
    return d, good


def track_eigenvalues(matrices):
    """Continue eigenvalue branches by minimum-distance assignment along a loop."""
    from scipy.optimize import linear_sum_assignment
    arr = np.asarray(matrices)
    eig = np.array([np.linalg.eigvals(S) for S in arr])
    result=[eig[0]]
    for values in eig[1:]:
        cost=abs(result[-1][:,None]-values[None,:])
        r,c=linear_sum_assignment(cost)
        ordered=np.empty(4,complex); ordered[r]=values[c]
        result.append(ordered)
    return np.array(result)
