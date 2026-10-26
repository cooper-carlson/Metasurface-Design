"""Full-wave tests of complex geometric phase: multilayer stacks vs lateral PB supercells.

Everything uses the NV-factorized, area-fraction-mask TORCWA solver of engine_twofin_eta.
Jones conventions: lab linear (x, y); "forward" = from the bottom (substrate) port to the top.

* layer_T: uniform single-layer transmission Jones T(u, theta) (the meta-atom library).
* stack_T: N patterned layers, each rotated by its own theta, separated by homogeneous
  gap/spacer layers, solved as one multilayer RCWA problem (interlayer multiple
  reflections and evanescent coupling included).
* supercell_T: a 1 x N lateral supercell whose cells carry different theta (a PB grating),
  returning the zero-order Jones matrix and selected diffraction orders.
* eigen helpers: tracked right/left eigenvectors, the ordered-transport prediction
  prod_k lambda_n(theta_k) * prod_k <L_n(theta_{k+1})|R_n(theta_k)>, and the biorthogonal
  eigenbranch amplitudes <L_n|M|R_n> used to compare opposite cycle orientations.
"""
import numpy as np
import torch

import engine_twofin_eta as E
from engine_twofin_eta import (Physics, RCWA, vary, _xy_grid, _rect_coverage, rectangles,
                               twofin_normals, fin_eps, _s_matrix)


def _cell_fields(p, physics, raster, theta_deg):
    """Area-fraction fill and NV normals of one cell (square raster)."""
    anchor = p['length1']
    X, Y = _xy_grid(physics, raster, anchor.device, anchor.dtype)
    h = X[1, 0]-X[0, 0]
    f = [_rect_coverage(X, Y, c, a, l, w, h) for c, a, l, w in rectangles(p, theta_deg)]
    fill = 1-(1-f[0])*(1-f[1])
    nx, ny = twofin_normals(p, physics, raster, theta_deg)
    return fill, nx, ny


def _tensors(p, device):
    return {k: torch.as_tensor(float(v), dtype=torch.float64, device=device) for k, v in p.items()}


def _new_sim(physics, opts, L, order):
    NV = E._nv_solver()
    sim = NV(freq=1/physics.wavelength, order=list(order), L=list(L), dtype=torch.complex128, device=opts.device)
    sim.add_input_layer(eps=physics.n_bottom**2)
    sim.add_output_layer(eps=physics.n_top**2)
    sim.set_incident_angle(inc_ang=0., azi_ang=0.)
    return sim


def _add_pattern(sim, physics, fill, nx, ny):
    eps = (physics.n_background**2+(fin_eps(physics)-physics.n_background**2)*fill).to(torch.complex128)
    sim.add_layer_nv(physics.height, eps, nx.to(torch.complex128), ny.to(torch.complex128))


def layer_T(p, physics, opts, u, mod, theta_deg):
    """Uniform single-layer forward Jones T and full 4x4 S at (u, theta)."""
    with torch.no_grad():
        q = _tensors(vary(p, mod, u), opts.device)
        S = E.solve_torch(q, physics, opts, theta_deg=theta_deg).cpu().numpy()
    return S[2:4, :2], S


def stack_T(p, physics, opts, u, mod, thetas, gap_um=0.25, spacer_um=0.50, n_spacer=1.46):
    """N rotated copies of the layer, each on an SiO2 spacer with an air gap above.

    Bottom to top: substrate | layer_0 | air gap | SiO2 spacer | layer_1 | ... | layer_{N-1} | air.
    Every patterned layer therefore keeps its designed environment (SiO2 below, air above).
    Returns the forward (substrate -> air) zero-order Jones matrix.
    """
    with torch.no_grad():
        q = _tensors(vary(p, mod, u), opts.device)
        sim = _new_sim(physics, opts, (physics.period, physics.period), (opts.order, opts.order))
        for k, th in enumerate(thetas):
            fill, nx, ny = _cell_fields(q, physics, opts.raster, float(th))
            _add_pattern(sim, physics, fill, nx, ny)
            if k < len(thetas)-1:
                sim.add_layer(thickness=gap_um, eps=physics.n_background**2)
                sim.add_layer(thickness=spacer_um, eps=n_spacer**2)
        sim.solve_global_smatrix()
        S = _s_matrix(sim, opts.device, torch.complex128).cpu().numpy()
    return S[2:4, :2], S


def supercell_T(p, physics, opts, u, mod, thetas, order_xy, orders=((0, 0),)):
    """1 x N lateral supercell (cells along x) with per-cell rotation thetas.

    Returns {order: 4x4 S block} for the requested diffraction orders (forward transmission
    Jones = S[2:4, :2]); order (m, 0) is deflected by m * lambda / (N * period) along x.
    """
    with torch.no_grad():
        q = _tensors(vary(p, mod, u), opts.device)
        cells = [_cell_fields(q, physics, opts.raster, float(th)) for th in thetas]
        fill = torch.cat([c[0] for c in cells], dim=0)            # x is the first raster index
        nx = torch.cat([c[1] for c in cells], dim=0); ny = torch.cat([c[2] for c in cells], dim=0)
        N = len(thetas)
        sim = _new_sim(physics, opts, (N*physics.period, physics.period), order_xy)
        _add_pattern(sim, physics, fill, nx, ny)
        sim.solve_global_smatrix()
        out = {o: _s_matrix(sim, opts.device, torch.complex128, o).cpu().numpy() for o in orders}
    return out


# ---------------------------------------------------------------- eigen bookkeeping
def eig_rl(T):
    """Right eigenvectors (columns, unit norm) and biorthogonal left rows, sorted by |lambda|."""
    w, V = np.linalg.eig(np.asarray(T, complex))
    idx = np.argsort(-np.abs(w)); w, V = w[idx], V[:, idx]
    V = V/np.linalg.norm(V, axis=0, keepdims=True)
    return w, V, np.linalg.inv(V)


def tracked_eigs(Ts):
    """Eigen-decompositions along an ordered sequence, branches tracked by eigenvector overlap."""
    out = []
    for T in Ts:
        w, V, Lr = eig_rl(T)
        if out:
            Vp = out[-1][1]
            s = np.abs(Vp.conj().T@V)
            if s[0, 0]+s[1, 1] < s[0, 1]+s[1, 0]:
                w, V = w[[1, 0]], V[:, [1, 0]]; Lr = np.linalg.inv(V)
        out.append((w, V, Lr))
    return out


def transport_prediction(Ts):
    """Adiabatic ordered-transport factor for the product T_{N-1}...T_0 around a closed cycle.

    a_n = prod_k lambda_n(k) * prod_k <L_n(k+1)|R_n(k)>, with the cycle closed on step 0. Returns
    the dynamical part, the geometric part and the geometric phase gamma_n = -i log(geometric)
    (transport factor e^{i gamma_n}; the paper's Wilson product W_n = e^{-i gamma_n} is its inverse order).
    """
    E_ = tracked_eigs(list(Ts)+[Ts[0]])
    N = len(Ts)
    dyn = np.prod([E_[k][0] for k in range(N)], axis=0)
    geo = np.prod([np.array([E_[k+1][2][n]@E_[k][1][:, n] for n in range(2)]) for k in range(N)], axis=0)
    swapped = np.abs(E_[N][1][:, 0].conj()@E_[0][1][:, 0]) < np.abs(E_[N][1][:, 1].conj()@E_[0][1][:, 0])
    return dict(dynamical=dyn, geometric=geo, gamma=-1j*np.log(geo), branch_swap=bool(swapped))


def eigenbranch_amplitudes(M, T0):
    """<L_n|M|R_n> in the eigenbasis of the reference (cycle start) matrix T0."""
    w, V, Lr = eig_rl(T0)
    return np.array([Lr[n]@M@V[:, n] for n in range(2)])


def circular(M):
    """Circular basis as in the paper's SM Eq. (36): |R> = (1, -i)/sqrt2, |L> = (1, i)/sqrt2."""
    U = np.array([[1, 1], [-1j, 1j]])/np.sqrt(2)
    return np.linalg.inv(U)@M@U


def conversion_asymmetry(M):
    """Paper Eq. (15): (T_{R<-L} - T_{L<-R}) / (T_{R<-L} + T_{L<-R}) with basis order (R, L)."""
    C = circular(M); a, b = abs(C[0, 1])**2, abs(C[1, 0])**2
    return (a-b)/(a+b)
