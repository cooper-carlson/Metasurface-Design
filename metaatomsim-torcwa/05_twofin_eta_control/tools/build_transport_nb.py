import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code
C=[]
C.append(md(r"""# Complex geometric phase: ordered transport (multilayer) vs lateral arrays

Characterization of the v7b η-control meta-atom (c-Si two-fin on SiO₂, air above, u = junction position) against
**Jana & Genevet, "Complex Geometric Phase from Nonorthogonal Polarization Transport"** and its Supplemental Material (SM).

| Paper claim | Where it is tested |
|---|---|
| Eq. 9 / SM Eq. 22: cycle of $K(\phi,\eta)$ gives $\gamma_\pm=\pi\pm i\pi\eta/\sqrt{1-\eta^2}$ | §1 (analytic + discrete Wilson, SM Eq. 26–28) |
| SM Eq. 29–30: path reversal flips γ, dynamical phase unchanged | §1, §2, §4–§6 |
| Eq. 13–15 / SM Eq. 34–43: geometric propagator, $\mathcal A_{\rm circ}=2\eta/(1+\eta^2)$, SM §XI numbers | §1 |
| SM §XIII: a bilayer reproduces the asymmetry without transport | §1 |
| "A transverse array of unrelated matrices is not by itself a transport cycle" | §6 lateral PB supercells (full RCWA, all orders) |
| Ordered (longitudinal) transport realizes the holonomy | §4 ideal cascade, §5 full-RCWA multilayer stacks |

**Observable (SM §VIII, §XIII).** For a cycle $C$ and its reverse $C^{-1}$ through the *same* parameter values, the dynamical
factor $\prod_k\lambda_n(\theta_k)$ is identical, so the eigenbranch-resolved amplitude ratio
$|\langle L_n|M_C|R_n\rangle| / |\langle L_n|M_{C^{-1}}|R_n\rangle| = e^{-2\,\mathrm{Im}\,\gamma_n}$ isolates the geometric amplitude.
A lateral array's zero order is a sum over cells and cannot depend on their order (ratio 1 in the local-Jones model). A stack's
transmission is an ordered product.

**Physical rotation.** A PB rotation $R(\theta)=e^{-i\theta\sigma_y}$ (lab linear basis) rotates the $(\sigma_z,\sigma_x)$ components
and leaves $\sigma_y$ (circular) invariant. With $T=t_0I+\mathbf t\cdot\boldsymbol\sigma$ and $\hat k=\mathbf t/\sqrt{\mathbf t\cdot\mathbf t}$,
one rotation cycle gives $\gamma_\pm=\pi(1\mp\hat k_y)$ (`tests/test_theta_loop.py`). The paper's model realized by rotation is
$K'=\sigma_z+i\eta\sigma_y$ (non-Hermiticity along the rotation axis). A z-mirror-symmetric layer has $T=T^{\mathsf T}$, so
$\hat k_y=0$ and $\gamma=\pi$ exactly, whatever η."""))
C.append(code(r"""from pathlib import Path
import json, time
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm

from engine_twofin_eta import (Physics, RCWA, Modulation, design_from_raw, numpy_metrics, pauli_khat,
                               theta_loop_phase, transpose_asymmetry, virtual_wilson)
import engine_transport_test as X

OUT = Path('results/complex_pb_transport_test'); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = Path('results/eta_control_v7b_twofin/complex_pb_checkpoint.npz')
PHYS = Physics(n_fin=3.88, k_fin=0.019, n_background=1.0, n_bottom=1.46, n_top=1.0, height=0.495)
MOD = Modulation(parameter='junction_frac', amplitude=0.10)
OPTS = RCWA(order=10, raster=192, device=DEVICE, precision='complex128')   # NV + area-fraction mask
raw = np.load(CKPT, allow_pickle=False)['raw']
GEOM = {k: float(v) for k, v in design_from_raw(torch.tensor(raw, dtype=torch.float64)).items()}
U_TEST = (-1.0, 1.0)                 # u=-1: eta ~ 0.69 ; u=+1: eta ~ 0.03 (control)
THETA_GRID = np.arange(0, 180, 7.5)  # 24-point library; stacks / supercells use sub-grids
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1., -1.]); I2 = np.eye(2)
Rot = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
print('Design:', {k: round(v, 4) for k, v in GEOM.items()}); print(PHYS)"""))
C.append(md(r"""## 1 · The paper's minimal model, verified number by number
$K(\phi,\eta)=\cos\phi\,\sigma_x+\sin\phi\,\sigma_y+i\eta\sigma_z$ (linear basis). Explicit eigenvectors (SM Eq. 12), reciprocal bras
(SM Eq. 16–18), connection (SM Eq. 20–21), discrete Wilson product (SM Eq. 26), geometric propagator (SM Eq. 34, 38), conversion
asymmetry (SM Eq. 42), the numbers of SM §XI and the bilayer of SM §XIII."""))
C.append(code(r"""K = lambda phi, eta: np.cos(phi)*sx + np.sin(phi)*sy + 1j*eta*sz
def paper_V(phi, eta):
    chi = np.sqrt(1-eta**2)
    return np.array([[np.exp(-1j*phi), np.exp(-1j*phi)], [chi-1j*eta, -chi-1j*eta]])/np.sqrt(2)   # SM Eq. 12
checks = {}
for eta in (0.2, 0.5, 0.9):
    chi = np.sqrt(1-eta**2)
    V = paper_V(0.7, eta); Kp = K(0.7, eta)
    assert np.allclose(Kp@V, V@np.diag([chi, -chi]))                                   # SM Eq. 13
    s = np.vdot(V[:, 0], V[:, 1]); assert np.isclose(abs(s), eta) and np.isclose(s, eta*(eta-1j*chi))   # SM Eq. 14-15
    for N in (16, 128, 1024):                                                           # SM Eq. 26-28
        phis = 2*np.pi*np.arange(N+1)/N
        Vs = [paper_V(f, eta) for f in phis]
        W = np.prod([np.diag(np.linalg.inv(Vs[j])@Vs[j+1]) for j in range(N)], axis=0)
        g = 1j*np.log(W)
        checks[(eta, N)] = g
    g = checks[(eta, 1024)]; exact = np.pi*eta/chi
    g_ext = (8*checks[(eta, 1024)] - checks[(eta, 128)])/7          # Richardson: the discrete error is O(1/N)
    print(f"η={eta}: Wilson N=1024 γ± = {np.round(g, 4)} | N→∞ (Richardson) = {np.round(g_ext, 4)} | Eq. 22: π ± i{exact:.4f}")
    assert np.allclose(np.abs(g_ext.imag), exact, rtol=1e-3)
    # path reversal (SM Eq. 29)
    grev = {}
    for N in (128, 1024):
        Vs = [paper_V(f, eta) for f in (2*np.pi*np.arange(N+1)/N)[::-1]]
        grev[N] = 1j*np.log(np.prod([np.diag(np.linalg.inv(Vs[j])@Vs[j+1]) for j in range(N)], axis=0))
    grev_ext = (8*grev[1024] - grev[128])/7
    print(f"      reversed path C⁻¹ (Richardson): γ± = {np.round(grev_ext, 4)}  (SM Eq. 29: -γ)")
    assert np.allclose(grev_ext.imag, -g_ext.imag, rtol=1e-3)
print('SM Eq. 12-29 reproduced (eigenvectors, overlap, Wilson product, path reversal).')

# SM §IX-XI: geometric propagator, circular conversion, representative numbers
U = np.array([[1, 1], [-1j, 1j]])/np.sqrt(2)                                   # SM Eq. 36
for eta, ref in ((0.5, dict(im=1.8138, G=7.06e-4, C=9.54, A=0.800)), (0.9, dict(im=6.487, G=5.39e-12, C=25.58, A=0.9945))):
    chi = np.sqrt(1-eta**2); alpha = np.pi*eta/chi
    J = -expm(-(alpha/chi)*K(0, eta))                                           # SM Eq. 33
    assert np.allclose(J, -np.cosh(alpha)*I2 + np.sinh(alpha)/chi*K(0, eta))    # SM Eq. 34
    Jc = np.linalg.inv(U)@J@U
    assert np.allclose(Jc, [[-np.cosh(alpha), 1j*(1+eta)/chi*np.sinh(alpha)], [-1j*(1-eta)/chi*np.sinh(alpha), -np.cosh(alpha)]])  # SM Eq. 38
    A = X.conversion_asymmetry(J); G = np.exp(-4*np.pi*eta/chi); Cdb = 10*np.log10(abs(Jc[0, 1])**2/abs(Jc[1, 0])**2)
    print(f"η={eta}: Im γ+={alpha:.4f} (SM {ref['im']}), G+/G-={G:.3g} (SM {ref['G']}), C={Cdb:.2f} dB (SM {ref['C']}), A={A:.4f} (SM {ref['A']})")
    assert np.isclose(alpha, ref['im'], rtol=1e-3) and np.isclose(G, ref['G'], rtol=1e-2) and np.isclose(Cdb, ref['C'], atol=0.01) and np.isclose(A, ref['A'], atol=1e-4)
print('SM §IX-XI reproduced.')

# What Eq. 15 does and does not certify: any analytic function a I + b K(0,eta) has the same off-diagonal ratio.
for eta in (0.5, 0.9):
    thin = expm(-1j*0.05*K(0, eta))       # one thin element, no transport, no geometric amplitude
    print(f"η={eta}: single thin K(0,η) element A_circ = {X.conversion_asymmetry(thin):.4f}  vs  2η/(1+η²) = {2*eta/(1+eta**2):.4f}")

# SM §XIII bilayer (partial polarizer + quarter-wave element rotated together)
p_, q_ = 0.9, 0.3
D0 = np.diag([p_, q_]); Q4 = Rot(np.pi/4)@np.diag([np.exp(-1j*np.pi/4), np.exp(1j*np.pi/4)])@Rot(-np.pi/4)
Jtwo = Q4@D0
Jtc = np.linalg.inv(U)@Jtwo@U
print(f"SM §XIII bilayer: E(R<-L)={abs(Jtc[0,1])**2:.3f} (SM 0.405), E(L<-R)={abs(Jtc[1,0])**2:.3f} (SM 0.045), "
      f"A_two={X.conversion_asymmetry(Jtwo):.3f} (SM {(p_**2-q_**2)/(p_**2+q_**2):.3f})")
assert np.isclose(abs(Jtc[0, 1])**2, p_**2/2) and np.isclose(abs(Jtc[1, 0])**2, q_**2/2)
gt = theta_loop_phase(Jtwo)
print(f"   its rigid-rotation cycle (this work's γ = π(1∓k_y)): γ± = {np.round(gt, 4)}, ‖J-Jᵀ‖/‖J‖ = {transpose_asymmetry(Jtwo):.3f}")"""))
C.append(md(r"""## 2 · What physical rotation transports: ideal elements
Ideal thin layers, passive by a common loss κ ≥ η. Stack: ordered product over one rotation cycle (θ: 0→π, i.e. 2θ: 0→2π) for
both orientations. Lateral: the local-Jones zero order is the mean of the same matrices, independent of order.
* $K'(\eta)=\sigma_z+i\eta\sigma_y$: the paper's model realized by rotation (non-Hermiticity along the rotation axis).
* $K(0,\eta)=\sigma_x+i\eta\sigma_z$: the paper's $\phi=0$ generator, rotated physically (non-Hermiticity rotates with it; $T=T^{\mathsf T}$)."""))
C.append(code(r"""def ideal_stack(K0, N, Q, sense, kappa):
    M = I2.astype(complex); L = expm(-1j*K0*Q/N - kappa*Q/N*I2)
    for k in range(N):
        th = sense*np.pi*k/N; M = Rot(th)@L@Rot(-th)@M
    return M
ideal = {}
for name, K0 in (("K'=σz+iησy", sz + 1j*0.5*sy), ("K(0,η)=σx+iησz", sx + 1j*0.5*sz)):
    eta = 0.5; chi = np.sqrt(1-eta**2)
    T0 = expm(-1j*K0*0.01)
    print(f"{name}: one-cycle formula γ± = {np.round(theta_loop_phase(K0), 4)}")
    for N in (16, 64, 256):
        a = {s: X.eigenbranch_amplitudes(ideal_stack(K0, N, 20., s, 0.6), T0) for s in (+1, -1)}
        lat = {s: X.eigenbranch_amplitudes(np.mean([Rot(s*np.pi*k/N)@T0@Rot(-s*np.pi*k/N) for k in range(N)], axis=0), T0) for s in (+1, -1)}
        r = np.abs(a[+1])/np.abs(a[-1]); rl = np.abs(lat[+1])/np.abs(lat[-1])
        print(f"   N={N:4}: stack |a(+)|/|a(-)| = {np.round(r, 3)} | lateral (local) = {np.round(rl, 6)}")
        ideal[(name, N)] = dict(stack=r.tolist(), lateral=rl.tolist())
    print(f"   adiabatic prediction e^(±2 Im γ) = {np.exp(2*np.pi*eta/chi):.2f}, {np.exp(-2*np.pi*eta/chi):.4f}")"""))
C.append(md(r"""## 3 · The meta-atom library T(u, θ) and its rotation-cycle geometry
Uniform lattices of the v7b atom rotated by θ (NV, order 10, raster 192). For each u: η(θ), $\hat k_y(\theta)$ (rotation-invariant
for an ideal local atom), the one-solve formula, and the full lattice Wilson/transport product over θ with path reversal."""))
C.append(code(r"""lib_file = OUT/'library.npz'
if lib_file.is_file():
    z = np.load(lib_file); LIB = {u: z[f'u{u:+.1f}'] for u in U_TEST}
else:
    LIB = {}
    for u in U_TEST:
        t0 = time.time(); LIB[u] = np.array([X.layer_T(GEOM, PHYS, OPTS, u, MOD, float(t))[0] for t in THETA_GRID])
        print(f'library u={u:+.1f}: {len(THETA_GRID)} solves, {time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(lib_file, **{f'u{u:+.1f}': LIB[u] for u in U_TEST}, theta=THETA_GRID)
geo = {}
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for u in U_TEST:
    Ts = LIB[u]; ky = np.array([pauli_khat(T)[1] for T in Ts]); eta = np.array([abs(np.vdot(X.eig_rl(T)[1][:, 0], X.eig_rl(T)[1][:, 1])) for T in Ts])
    axs[0].plot(THETA_GRID, eta, 'o-', label=f'u={u:+.0f}'); axs[1].plot(THETA_GRID, -np.pi*ky.imag, 'o-', label=f'u={u:+.0f}')
    axs[2].plot(THETA_GRID, np.pi*(1-ky.real), 'o-', label=f'u={u:+.0f}')
    rows = {}
    for step in (4, 2, 1):                       # N = 6, 12, 24 layers per cycle
        sub = list(Ts[::step]); fw = X.transport_prediction(sub); bw = X.transport_prediction(sub[:1]+sub[1:][::-1])
        rows[len(sub)] = dict(gamma_fw=fw['gamma'], gamma_bw=bw['gamma'], swap=fw['branch_swap'])
    form = theta_loop_phase(Ts[0])
    geo[u] = dict(rows=rows, formula=form, eta0=float(eta[0]), asym=transpose_asymmetry(Ts[0]))
    print(f"u={u:+.0f}: η(θ=0)={eta[0]:.3f}, ‖T-Tᵀ‖/‖T‖={geo[u]['asym']:.3f}, formula γ± (θ=0) = {np.round(form, 3)}")
    for n, r in rows.items():
        print(f"   N={n:2}: transport γ (forward) = {np.round(r['gamma_fw'], 3)} | reversed = {np.round(r['gamma_bw'], 3)}"
              f"{'  [branch swap]' if r['swap'] else ''}")
    gext = 2*rows[24]['gamma_fw'] - rows[12]['gamma_fw']      # discrete error is O(1/N), as in §1
    geo[u]['gamma_ext'] = gext
    print(f"   N→∞ (Richardson 12/24): Im γ = {np.round(gext.imag, 3)}  vs one-solve formula ±{abs(form[0].imag):.3f}")
for a, t in zip(axs, ('η', 'Im γ₊ (formula, per θ)', 'Re γ₊ (formula, per θ)')): a.set(xlabel='θ (deg)', title=t); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
for u in U_TEST:
    print(f"u={u:+.0f}: single local element (no transport) circular-conversion asymmetry A = {X.conversion_asymmetry(LIB[u][0]):+.3f}")"""))
C.append(md(r"""## 4 · Ideal cascade of meta-atom layers (Jones products, no interlayer reflection)
Exact ordered products of library matrices, $M_\pm=T(\theta_{N-1})\cdots T(\theta_0)$ with $\theta_k=\pm\pi k/N$, compared with the
adiabatic transport prediction $\prod\lambda_n\prod\langle L_n(k{+}1)|R_n(k)\rangle$. Only the dominant branch is reliable when
$|\lambda_2/\lambda_1|^N$ is small, because non-adiabatic leakage from the strong branch then swamps the weak one."""))
C.append(code(r"""casc = {}
for u in U_TEST:
    Ts = LIB[u]
    for step in (4, 2, 1):
        sub = list(Ts[::step]); N = len(sub)
        Mf = np.eye(2, dtype=complex); Mb = np.eye(2, dtype=complex)
        for T in sub: Mf = T@Mf
        for T in sub[:1]+sub[1:][::-1]: Mb = T@Mb
        af, ab = X.eigenbranch_amplitudes(Mf, sub[0]), X.eigenbranch_amplitudes(Mb, sub[0])
        pf, pb = X.transport_prediction(sub), X.transport_prediction(sub[:1]+sub[1:][::-1])
        pred = np.abs(pf['geometric'])/np.abs(pb['geometric'])
        casc[(u, N)] = dict(ratio=(np.abs(af)/np.abs(ab)).tolist(), pred=pred.tolist(),
                            lam_ratio=float(abs(X.eig_rl(sub[0])[0][1]/X.eig_rl(sub[0])[0][0])))
        print(f"u={u:+.0f} N={N:2}: cascade |a(+)|/|a(-)| = {np.round(np.abs(af)/np.abs(ab), 3)} | adiabatic prediction = {np.round(pred, 3)}"
              f" | |λ2/λ1| = {casc[(u, N)]['lam_ratio']:.2f}")"""))
C.append(md(r"""## 5 · Full-RCWA multilayer stacks
N rotated copies of the layer in one RCWA solve: substrate | layer | air gap | SiO₂ spacer | layer | … | air. Every patterned layer keeps
its designed environment, which is what makes $T\neq T^{\mathsf T}$. Interlayer multiple reflections and near-field coupling are
included. Two spacer choices test Fabry-Pérot sensitivity."""))
C.append(code(r"""stack = {}
SPACERS = ((0.25, 0.50), (0.40, 0.30))
for u in U_TEST:
    for N, step in ((6, 4), (12, 2)):
        thf = THETA_GRID[::step]; thb = np.concatenate((thf[:1], -thf[1:][::-1] % 180))
        T0 = LIB[u][0]
        for gap, spc in SPACERS:
            t0 = time.time()
            Mf, _ = X.stack_T(GEOM, PHYS, OPTS, u, MOD, thf, gap, spc)
            Mb, _ = X.stack_T(GEOM, PHYS, OPTS, u, MOD, -thf, gap, spc)   # same angles, reversed rotation sense
            af, ab = X.eigenbranch_amplitudes(Mf, T0), X.eigenbranch_amplitudes(Mb, T0)
            stack[(u, N, gap, spc)] = dict(ratio=(np.abs(af)/np.abs(ab)).tolist(), a_f=af.tolist(), a_b=ab.tolist(),
                                           P=float(np.sum(abs(Mf)**2)/2), A_f=X.conversion_asymmetry(Mf), A_b=X.conversion_asymmetry(Mb))
            print(f"u={u:+.0f} N={N:2} gap {gap} / SiO2 {spc} µm: |a(+)|/|a(-)| = {np.round(np.abs(af)/np.abs(ab), 3)} | "
                  f"cascade {np.round(casc[(u, N)]['ratio'], 3)} | mean T = {stack[(u, N, gap, spc)]['P']:.2e} | "
                  f"A_circ(+,-) = {stack[(u, N, gap, spc)]['A_f']:+.3f}, {stack[(u, N, gap, spc)]['A_b']:+.3f} | {time.time()-t0:.0f}s", flush=True)"""))
C.append(md(r"""## 6 · Lateral PB supercells (full RCWA, all diffraction orders)
A 1×N supercell (cells along x) holds the same N rotations as the stack, side by side: a PB grating whose $\pm1$ orders carry the
$e^{\pm2i\theta}$ conversion. Reversing the rotation sense reverses the order of the cells. Local-Jones prediction: zero order = mean of the
cell matrices (order-independent, ratio 1); order $m$ = discrete Fourier component of the cell matrices. Any deviation of the
zero-order eigenbranch ratio from 1 measures lateral coupling, the only route by which an array could impart ordered transport."""))
C.append(code(r"""lateral = {}
N = 6; thf = THETA_GRID[::4]
for u in U_TEST:
    T0 = LIB[u][0]; cells = LIB[u][::4]
    for oxy in ((6*N, 6), (8*N, 6)):
        t0 = time.time(); res = {}
        for s in (+1, -1):
            th = s*thf % 180
            res[s] = X.supercell_T(GEOM, PHYS, OPTS, u, MOD, th, oxy, orders=((0, 0), (1, 0), (-1, 0)))
        a = {s: X.eigenbranch_amplitudes(res[s][(0, 0)][2:4, :2], T0) for s in (+1, -1)}
        loc0 = np.mean(cells, axis=0); aloc = X.eigenbranch_amplitudes(loc0, T0)
        m1 = {s: {m: X.circular(res[s][(m, 0)][2:4, :2]) for m in (1, -1)} for s in (+1, -1)}
        floc = {m: X.circular(np.mean([c*np.exp(-2j*np.pi*m*j/N) for j, c in enumerate(cells)], axis=0)) for m in (1, -1)}
        lateral[(u, oxy)] = dict(ratio=(np.abs(a[+1])/np.abs(a[-1])).tolist(), a_plus=a[+1].tolist(), a_local=aloc.tolist(),
                                 eff={f'{s:+d}_{m:+d}': (abs(m1[s][m][0, 1])**2, abs(m1[s][m][1, 0])**2) for s in (+1, -1) for m in (1, -1)},
                                 eff_local={f'{m:+d}': (abs(floc[m][0, 1])**2, abs(floc[m][1, 0])**2) for m in (1, -1)})
        print(f"u={u:+.0f} orders {oxy}: zero-order |a(+)|/|a(-)| = {np.round(np.abs(a[+1])/np.abs(a[-1]), 4)} | "
              f"|a(+)| = {np.round(np.abs(a[+1]), 4)} vs local mean {np.round(np.abs(aloc), 4)} | {time.time()-t0:.0f}s", flush=True)
        for s in (+1, -1):
            print('     sense %+d: ' % s + ', '.join(f"order {m:+d} R←L {abs(m1[s][m][0,1])**2:.4f} / L←R {abs(m1[s][m][1,0])**2:.4f}" for m in (1, -1)))
        print('     local Jones: ' + ', '.join(f"order {m:+d} R←L {abs(floc[m][0,1])**2:.4f} / L←R {abs(floc[m][1,0])**2:.4f}" for m in (1, -1)))"""))
C.append(md("## 7 · Summary: does each arrangement impart the complex geometric amplitude?"))
C.append(code(r"""print('Eigenbranch amplitude ratio |a(C)|/|a(C⁻¹)| for the dominant branch (geometric amplitude e^{∓2 Im γ}; 1 = none):')
for u in U_TEST:
    print(f"\nu={u:+.0f}  (η={geo[u]['eta0']:.2f}, ‖T-Tᵀ‖/‖T‖={geo[u]['asym']:.3f}, one-cycle formula Im γ± = ±{abs(geo[u]['formula'][0].imag):.3f})")
    print(f"   adiabatic prediction (N=12): {casc[(u, 12)]['pred'][0]:.3f}")
    print(f"   ideal Jones cascade (N=12):  {casc[(u, 12)]['ratio'][0]:.3f}")
    for gap, spc in SPACERS: print(f"   full-RCWA stack N=12, gap {gap}/SiO2 {spc}: {stack[(u, 12, gap, spc)]['ratio'][0]:.3f}")
    for oxy in ((36, 6), (48, 6)): print(f"   lateral supercell N=6, orders {oxy}: {lateral[(u, oxy)]['ratio'][0]:.4f}")
json.dump(dict(geo={str(u): dict(formula=[str(x) for x in g['formula']], eta0=g['eta0'], asym=g['asym'],
                                 rows={str(n): {k: (str(v) if k != 'swap' else v) for k, v in r.items()} for n, r in g['rows'].items()})
                    for u, g in geo.items()},
               cascade={f'{k}': v for k, v in casc.items()}, stack={f'{k}': {kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in stack.items()},
               lateral={f'{k}': {kk: (str(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in lateral.items()},
               ideal={f'{k}': v for k, v in ideal.items()}), open(OUT/'transport_test_summary.json', 'w'), indent=1, default=str)
print('\nSaved', OUT/'transport_test_summary.json')"""))

C.append(md(r"""## 8 · Separating geometric transport from full-wave artifacts
§5–§6 show two order-dependent effects that are *not* geometric: Fabry-Pérot interference between strongly reflecting layers (the
u=+1 control moves by ±12 %), and near-field coupling between neighbours with very different rotations in a steep lateral gradient.
* **8a.** Stacks with random spacer thicknesses (8 realizations, N=12, order 8). Averaging over Fabry-Pérot phases mimics a thick-spacer or
  broadband measurement. The geometric signal is the shift of $\ln(|a(C)|/|a(C^{-1})|)$ for u=−1 relative to the u=+1 control
  (prediction $2\,\mathrm{Im}\,\gamma\approx0.48$).
* **8b.** Lateral supercells with a gentler gradient (12 cells, 15° per cell). If the lateral zero-order ratio is coupling rather than
  transport, it should move toward 1 as the gradient becomes more adiabatic, for the control as much as for u=−1."""))
C.append(code(r"""rng = np.random.default_rng(2026)
OPTS8 = RCWA(order=8, raster=160, device=DEVICE, precision='complex128')
REAL = [(float(rng.uniform(0.2, 0.6)), float(rng.uniform(0.2, 0.8))) for _ in range(8)]
thf = THETA_GRID[::2]
avg = {u: [] for u in U_TEST}
for u in U_TEST:
    T0 = X.layer_T(GEOM, PHYS, OPTS8, u, MOD, 0.)[0]
    for gap, spc in REAL:
        t0 = time.time()
        Mf, _ = X.stack_T(GEOM, PHYS, OPTS8, u, MOD, thf, gap, spc)
        Mb, _ = X.stack_T(GEOM, PHYS, OPTS8, u, MOD, -thf, gap, spc)
        r = np.abs(X.eigenbranch_amplitudes(Mf, T0))/np.abs(X.eigenbranch_amplitudes(Mb, T0))
        avg[u].append(float(np.log(r[0])))
        print(f"u={u:+.0f} gap {gap:.2f} / SiO2 {spc:.2f}: ln ratio (dominant) = {np.log(r[0]):+.3f} | {time.time()-t0:.0f}s", flush=True)
shift = np.median(avg[-1.0]) - np.median(avg[1.0])
spread = {u: (np.percentile(v, 25), np.percentile(v, 75)) for u, v in avg.items()}
pred = 2*abs(geo[-1.0]['gamma_ext'][0].imag)
print(f"\nmedian ln ratio: u=-1 {np.median(avg[-1.0]):+.3f} (IQR {spread[-1.0][0]:+.3f}..{spread[-1.0][1]:+.3f}), "
      f"u=+1 {np.median(avg[1.0]):+.3f} (IQR {spread[1.0][0]:+.3f}..{spread[1.0][1]:+.3f})")
print(f"geometric signal (u=-1 minus control) = {shift:+.3f}  vs  prediction 2 Im γ = {pred:.3f}")
fig, ax = plt.subplots(figsize=(6, 3.4))
for i, u in enumerate(U_TEST):
    ax.plot([i]*len(avg[u]), avg[u], 'o', alpha=.7); ax.plot([i-.2, i+.2], [np.median(avg[u])]*2, 'k-')
ax.axhline(pred, ls='--', color='gray'); ax.text(1.35, pred, ' 2 Im γ (u=-1)', va='center', fontsize=8)
ax.set(xticks=[0, 1], xticklabels=['u=-1 (η=0.67)', 'u=+1 control (η=0.03)'], ylabel='ln |a(C)|/|a(C⁻¹)|', xlim=(-.5, 1.9),
       title='Full-RCWA stacks, 8 random spacer realizations'); ax.grid(alpha=.3); plt.tight_layout(); plt.show()"""))
C.append(code(r"""lateral12 = {}
N12 = 12; th12 = THETA_GRID[::2]
for u in U_TEST:
    T0 = LIB[u][0]; t0 = time.time(); res = {}
    for s_ in (+1, -1):
        res[s_] = X.supercell_T(GEOM, PHYS, OPTS8, u, MOD, s_*th12 % 180, (6*N12, 6), orders=((0, 0),))
    a = {s_: X.eigenbranch_amplitudes(res[s_][(0, 0)][2:4, :2], T0) for s_ in (+1, -1)}
    aloc = X.eigenbranch_amplitudes(np.mean(LIB[u][::2], axis=0), T0)
    lateral12[u] = (np.abs(a[+1])/np.abs(a[-1])).tolist()
    print(f"u={u:+.0f} 12 cells (15°/cell): zero-order |a(+)|/|a(-)| = {np.round(lateral12[u], 4)} | |a(+)| = {np.round(np.abs(a[+1]), 4)} "
          f"vs local mean {np.round(np.abs(aloc), 4)} | {time.time()-t0:.0f}s", flush=True)
print('\nLateral zero-order ratio (dominant branch), 6 cells (30°/cell) vs 12 cells (15°/cell):')
for u in U_TEST: print(f"   u={u:+.0f}: {lateral[(u, (48, 6))]['ratio'][0]:.3f} -> {lateral12[u][0]:.3f}")
json.dump(dict(stack_ln_ratio={str(u): v for u, v in avg.items()}, realizations=REAL, geometric_shift=float(shift), prediction=float(pred),
               lateral12={str(u): v for u, v in lateral12.items()}), open(OUT/'transport_test_followup.json', 'w'), indent=1)"""))
nb=new_notebook(cells=C,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
nbformat.write(nb,'/home/coopah/Desktop/python-projects/Metasurface-Design/metaatomsim-torcwa/05_twofin_eta_control/TwoFin_ComplexPB_Transport_Characterization.ipynb'); print('written',len(C))
