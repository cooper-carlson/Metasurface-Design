"""Analytic benchmarks, scattering diagnostics and Poincaré visualization."""
import numpy as np
from twofin import stokes, eigenpolarizations


def fresnel(config, bottom_z, top_z, fin_height=None):
    """Power-normalized bare-interface S at the physical monitor planes.

    A continuous-wave e^{-i omega t} convention is used. For bare interface
    the z=0 interface is retained even when the nanofins are absent.
    """
    n1, n2 = config.n_sub, 1.
    r = (n1-n2)/(n1+n2)
    t = 2*np.sqrt(n1*n2)/(n1+n2)
    p1 = np.exp(2j*np.pi*n1*abs(bottom_z)/config.wavelength)
    p2 = np.exp(2j*np.pi*n2*top_z/config.wavelength)
    eye = np.eye(2)
    return np.block([[r*p1*p1*eye, t*p1*p2*eye],
                     [t*p1*p2*eye, -r*p2*p2*eye]])


def yee_wavenumber(config, dx_um, index, courant=0.99 / np.sqrt(3)):
    """1D Yee-grid propagation constant, in radians per micrometer.

    This is valid for normally incident, zero-order plane waves in a
    homogeneous nondispersive medium on a cubic grid.
    """
    omega_dt = 2 * np.pi * courant * dx_um / config.wavelength
    argument = index * np.sin(omega_dt / 2) / courant
    if abs(argument) > 1:
        raise ValueError('The chosen frequency/index exceeds the Yee propagating band')
    return 2 * np.arcsin(argument) / dx_um


def fresnel_yee(config, bottom_z, top_z, dx_um, courant=0.99 / np.sqrt(3)):
    """Bare-interface Fresnel approximation with Yee numerical phase travel.

    Amplitudes use continuum impedance matching. Interface-grid artifacts
    and PML errors remain possible; this is a dispersion-aware benchmark,
    NOT a redefinition of the physical Fresnel result.
    """
    n1, n2 = config.n_sub, 1.
    r = (n1-n2) / (n1+n2)
    t = 2 * np.sqrt(n1*n2) / (n1+n2)
    p1 = np.exp(1j * yee_wavenumber(config, dx_um, n1, courant) * (-bottom_z))
    p2 = np.exp(1j * yee_wavenumber(config, dx_um, n2, courant) * top_z)
    eye = np.eye(2)
    return np.block([[r * p1**2 * eye, t * p1*p2 * eye],
                     [t * p1*p2 * eye, -r * p2**2 * eye]])


def fdtd_to_continuum_planes(S, config, ports, dx_um,
                             courant=0.99 / np.sqrt(3), bare=False):
    """Replace numerical free-space propagation with physical propagation.

    This is a reference-plane de-embedding for *homogeneous* port-to-interface
    regions; it DOES NOT change scattering physics or fix grid errors inside
    the nanostructure. For bare=True the upper interface is z=0; otherwise
    the upper external interface is at the fin top z=config.height.
    """
    bottom_z, top_z = map(float, ports)
    lower_distance = -bottom_z
    upper_distance = top_z - (0 if bare else config.height)
    if min(lower_distance, upper_distance) < 0:
        raise ValueError('Monitor plane is inside the scattering structure')
    k_bottom_physical = 2 * np.pi * config.n_sub / config.wavelength
    k_top_physical = 2 * np.pi / config.wavelength
    dk_bottom = k_bottom_physical - yee_wavenumber(config, dx_um, config.n_sub, courant)
    dk_top = k_top_physical - yee_wavenumber(config, dx_um, 1., courant)
    pb = np.exp(1j * dk_bottom * lower_distance)
    pt = np.exp(1j * dk_top * upper_distance)
    D = np.diag([pb, pb, pt, pt])
    return D @ np.asarray(S, dtype=complex) @ D


def scattering_comparison(S, reference, label='comparison'):
    """Print complex, magnitude, and phase errors without fitting parameters."""
    S, reference = np.asarray(S), np.asarray(reference)
    error = np.abs(S - reference)
    mags = np.abs(S) - np.abs(reference)
    mask = np.abs(reference) > .05
    phase_deg = np.rad2deg(np.angle(S[mask] / reference[mask]))
    data = dict(complex_max=float(error.max()),
                amplitude_max=float(np.max(np.abs(mags))),
                phase_max_deg=float(np.max(np.abs(phase_deg))) if len(phase_deg) else float('nan'))
    print(f'{label}: max |ΔS|={data["complex_max"]:.6f}, '
          f'max ||S|-|Sref||={data["amplitude_max"]:.6f}, '
          f'max |Δphase| (|Sref|>0.05)={data["phase_max_deg"]:.3f}°')
    return data


def at_monitor_planes(S, config, bottom_z, top_z):
    """Move RCWA ports at z=0,h to the FDTD port locations."""
    kb = np.exp(2j*np.pi*config.n_sub*(-bottom_z)/config.wavelength)
    kt = np.exp(2j*np.pi*(top_z-config.height)/config.wavelength)
    D = np.diag([kb,kb,kt,kt])
    return D @ S @ D


def metrics(S):
    S = np.asarray(S, complex)
    Q = S.conj().T @ S
    return dict(reciprocity_max=float(np.max(abs(S-S.T))),
                unitarity_max=float(np.max(abs(Q-np.eye(4)))),
                passivity_lambda_max=float(np.max(np.linalg.eigvalsh(Q))),
                worst_power_error=float(np.max(abs(np.sum(abs(S)**2,axis=0)-1))))


def display_matrix(S, label='S'):
    print(f'{label}: bottom(x,y), top(x,y)\n')
    for row in S:
        print('  '.join(f'{v.real:+.5f}{v.imag:+.5f}j' for v in row))
    print('\nMagnitude:')
    print(np.array2string(abs(S), precision=5, suppress_small=True))
    print('Phase (deg):')
    print(np.array2string(np.rad2deg(np.angle(S)), precision=2, suppress_small=True))


def print_polarization_modes(S):
    labels = [('R₁₁', S[:2,:2]), ('T₂₁', S[2:,:2]),
              ('T₁₂', S[:2,2:]), ('R₂₂', S[2:,2:])]
    for name, block in labels:
        print('\n',name)
        for j, state in enumerate(eigenpolarizations(block),1):
            v = state['eigenvalue']
            print(f'  eigenstate {j}: λ={v.real:+.6f}{v.imag:+.6f}j, |λ|={abs(v):.5f}, phase={np.angle(v,deg=True):+.2f}°, Stokes={np.round(state["stokes"],5)}')


def plot_poincare(S, blocks=('T₂₁','R₁₁')):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    matrices = {'T₂₁':S[2:,:2], 'R₁₁':S[:2,:2], 'T₁₂':S[:2,2:], 'R₂₂':S[2:,2:]}
    fig = plt.figure(figsize=(6*len(blocks),5))
    phi, theta = np.mgrid[0:2*np.pi:48j, 0:np.pi:25j]
    X=np.cos(phi)*np.sin(theta); Y=np.sin(phi)*np.sin(theta); Z=np.cos(theta)
    for i, name in enumerate(blocks):
        ax=fig.add_subplot(1,len(blocks),i+1,projection='3d')
        ax.plot_wireframe(X,Y,Z,color='.75',lw=.35,alpha=.35)
        for j, state in enumerate(eigenpolarizations(matrices[name])):
            s=state['stokes'][1:]
            ax.scatter(*s,s=130,label=f'Eigenstate {j+1}')
            ax.plot([0,s[0]],[0,s[1]],[0,s[2]],lw=1.4)
        ax.set(xlabel='S₁',ylabel='S₂',zlabel='S₃', title=f'{name} eigenpolarizations',
               xlim=(-1,1),ylim=(-1,1),zlim=(-1,1))
        ax.set_box_aspect((1,1,1))
        ax.legend(loc='upper left',fontsize=8)
    fig.tight_layout()
    return fig


def check(S, reference=None, analytic=None, other=None):
    d=metrics(S)
    if reference is not None:
        d['flat_vs_fresnel_max']=float(np.max(abs(reference-analytic)))
        d['flat_unitarity_max']=metrics(reference)['unitarity_max']
    if other is not None:
        d['fdtd_vs_rcwa_max']=float(np.max(abs(S-other)))
    for k,v in d.items():
        print(f'{k:26s} : {v:.6g}')
    return d


def eigenstate_path(scattering_matrices, block=(slice(2,4),slice(0,2))):
    """Track transmission eigenstate by maximal Jones-vector overlap.

    At degeneracies the branch is ill-defined; inspect eigenvalue separation.
    """
    states=[]
    gaps=[]
    previous=None
    for S in scattering_matrices:
        M=S[block]
        eigvals,vecs=np.linalg.eig(M)
        vecs/=np.linalg.norm(vecs,axis=0)
        j=(int(np.argmax(abs(eigvals))) if previous is None
           else int(np.argmax(abs(previous.conj() @ vecs))))
        v=vecs[:,j]
        if previous is not None:
            v*=np.exp(-1j*np.angle(np.vdot(previous,v)))
        states.append(v)
        gaps.append(float(abs(eigvals[0]-eigvals[1])))
        previous=v
    return np.array(states),np.array(gaps)


def pancharatnam_phase(jones_path):
    """Discrete Bargmann phase of geodesically closed pure-state polygon.

    An open path is implicitly closed by a final shortest geodesic; this is
    a kinematic phase, not automatically the measured scattering PB phase.
    """
    path=np.asarray(jones_path,complex)
    path=path/np.linalg.norm(path,axis=1,keepdims=True)
    links=np.sum(path.conj()*np.roll(path,-1,axis=0),axis=1)
    if np.any(abs(links)<1e-9):
        raise ValueError('Path contains nearly orthogonal consecutive states')
    return float(np.angle(np.prod(links)))


def plot_path(jones_path):
    import matplotlib.pyplot as plt
    from twofin import stokes
    path=np.array([stokes(j)[1:] for j in jones_path])
    fig=plt.figure(figsize=(6,5))
    ax=fig.add_subplot(111,projection='3d')
    phi,theta=np.mgrid[0:2*np.pi:40j,0:np.pi:20j]
    ax.plot_wireframe(np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta),
                      color='gray',alpha=.2,lw=.5)
    ax.plot(*path.T,'o-',lw=1.5, ms=4)
    ax.set(xlabel='S₁',ylabel='S₂',zlabel='S₃',title='Eigenpolarization path',
           xlim=(-1,1),ylim=(-1,1),zlim=(-1,1))
    ax.set_box_aspect((1,1,1))
    fig.tight_layout()
    return fig


def quality_gates(S, *, flat=None, analytic=None, other=None, acq=None):
    """Conservative initial screening, not a substitute for convergence."""
    mm=metrics(S)
    rows={'reciprocity < 0.005':(mm['reciprocity_max'],.005),
          'max(S†S eigenvalue) <= 1.005':(mm['passivity_lambda_max'],1.005)}
    if flat is not None and analytic is not None:
        rows['flat Fresnel complex error < 0.01']=(np.max(abs(flat-analytic)),.01)
    if other is not None:
        rows['independent complex S agreement < 0.05']=(np.max(abs(S-other)),.05)
    if acq is not None:
        rows['steady-state drift < 0.03']=(np.max(acq),.03)
    passed=True
    for label,(value,tol) in rows.items():
        ok = np.isfinite(value) and value < tol
        print(('PASS' if ok else 'FAIL'),f' {label}: {value:.5g}')
        passed &= ok
    print('\nPRELIMINARY SCREEN:', 'PASS (convergence still required)' if passed else 'NOT VALIDATED')
    return passed


def change_reference_planes(S, config, old_ports, new_ports):
    """Rephase an S matrix between two homogeneous port-plane choices."""
    a,b=old_ports; c,d=new_ports
    pb=np.exp(2j*np.pi*config.n_sub*(a-c)/config.wavelength)
    pt=np.exp(2j*np.pi*(d-b)/config.wavelength)
    D=np.diag([pb,pb,pt,pt])
    return D@np.asarray(S)@D