"""Checkpoint-matched two-fin polarization, Wilson and spectral diagnostics.

No optimization. Wavelength and frequency use um and inverse um respectively.
A virtual Wilson loop is kinematic, never a measured lateral transport phase.
"""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from engine_twofin_eta import (Physics, RCWA, Modulation, PARAM_KEYS, BOUNDS,
                                design_from_raw, twofin_mask, stokes, fin_eps)


def read_design(folder='results/twofin_eta_transport'):
    root=Path(folder)
    names=('twofin_eta_checkpoint_v2.npz','twofin_eta_final.npz')
    found=next((root/name for name in names if (root/name).is_file()),None)
    if found is None:
        raise FileNotFoundError('Expected the revised optimization checkpoint under '
            f'{root}. Run the optimizer first or change CHECKPOINT_DIR. Tried {names}.')
    with np.load(found, allow_pickle=False) as a:
        raw=np.asarray(a['raw'], dtype=float).reshape(-1)
        if len(raw)!=len(PARAM_KEYS) or not np.isfinite(raw).all():
            raise ValueError('Checkpoint raw geometry has the wrong shape or nonfinite values.')
        if 'metadata' in a:
            meta=json.loads(str(a['metadata']))
            phys=Physics(**meta['physics'])
            mod=Modulation(**meta['modulation'])
            tag=meta.get('objective_tag','unknown')
            saved_bounds=meta.get('bounds',{})
            for k in PARAM_KEYS:
                if k not in saved_bounds or not np.allclose(saved_bounds[k],BOUNDS[k]):
                    raise ValueError(f'Checkpoint geometry bound mismatch: {k}')
        else:
            needed=('physics','modulation','objective')
            if any(key not in a for key in needed):
                raise ValueError('Final checkpoint lacks physical metadata; do not guess settings.')
            phys=Physics(**json.loads(str(a['physics'])))
            mod=Modulation(**json.loads(str(a['modulation'])))
            tag=str(a['objective'])
        vals={k:float(v) for k,v in design_from_raw(torch.tensor(raw,dtype=torch.float64)).items()}
    return {'path':found,'raw':raw,'physics':phys,'modulation':mod,
            'tag':tag,'parameters':vals}


def polarization_metrics(S):
    S=np.asarray(S,complex)
    if S.shape != (4,4): raise ValueError('Expected 4x4 scattering matrix')
    T=S[2:, :2]; R=S[:2,:2]
    eig,V=np.linalg.eig(T)
    V=V/np.linalg.norm(V,axis=0)
    eta=float(abs(np.vdot(V[:,0],V[:,1])))
    n=np.linalg.norm(T,'fro')
    Q=T-np.trace(T)*np.eye(2)/2
    A=np.linalg.norm(Q,'fro')/max(n,1e-30)
    g=abs(eig[0]-eig[1])/max(n,1e-30)
    bad=(g < 0.002 or A < 0.002 or n < 1e-10)
    circ=np.array([[1,1],[1j,-1j]],complex)/np.sqrt(2)
    C=circ.conj().T@T@circ
    return {'eta':eta,'gap':g,'anisotropy':A,
            'nonnormality':float(np.linalg.norm(T.conj().T@T-T@T.conj().T,'fro')/max(n*n,1e-30)),
            'transmitted_x':float(np.sum(abs(T[:,0])**2)),
            'transmitted_y':float(np.sum(abs(T[:,1])**2)),
            'reflected_x':float(np.sum(abs(R[:,0])**2)),
            'reflected_y':float(np.sum(abs(R[:,1])**2)),
            'unitarity_error':float(np.linalg.norm(S.conj().T@S-np.eye(4),'fro')),
            'reciprocity_error':float(np.linalg.norm(S-S.T,'fro')),
            't_R_from_L':C[0,1], 't_L_from_R':C[1,0],
            'eigenvalues':eig, 'eigenvectors':V,
            'condition_number':float(np.linalg.cond(V)), 'defined':not bad}


def tracked_modes(Ts):
    Ts=np.asarray(Ts,complex)
    if Ts.ndim!=3 or Ts.shape[1:]!=(2,2):raise ValueError('Expected [N,2,2] Jones matrices')
    eigenvalues=[]; rights=[]; duals=[]; gaps=[]; etas=[]
    for T in Ts:
        w,V=np.linalg.eig(T)
        V=V/np.linalg.norm(V,axis=0)
        if rights:
            before=rights[-1]
            score=abs(before.conj().T@V)
            if score[0,0]+score[1,1] < score[0,1]+score[1,0]:
                w=w[::-1];V=V[:,::-1]
        if np.linalg.cond(V)>1e12:
            raise ValueError('Eigenbasis is nearly defective; Wilson path unreliable.')
        eigenvalues.append(w);rights.append(V);duals.append(np.linalg.inv(V))
        gaps.append(abs(w[0]-w[1])/max(np.linalg.norm(T),1e-30))
        etas.append(abs(np.vdot(V[:,0],V[:,1])))
    return {'eigenvalues':np.stack(eigenvalues),'R':np.stack(rights),
            'L_bras':np.stack(duals),'gap':np.array(gaps),'eta':np.array(etas)}


def wilson_cycle(Ts, *, orthogonal_control=False):
    """Closed discrete product, using reciprocal-left bras: W=prod L_j R_(j+1).

    R_k right eigvec columns, L_k rows of inv(R_k). If orthogonal_control,
    replace each eigenbasis with its QR orthogonalization; this is a control,
    not a physically inferred local Hermitian generator.
    """
    path=tracked_modes(Ts)
    R=path['R'].copy()
    if orthogonal_control:
        R=np.stack([np.linalg.qr(M)[0] for M in R])
    N=len(R)
    if N<3:raise ValueError('Closed loop requires >=3 samples')
    links=[];diags=[]
    for j in range(N):
        M=np.linalg.solve(R[j],R[(j+1)%N])
        links.append(np.diag(M))
        diags.append(M)
    W=np.prod(np.asarray(links),axis=0)
    # If the best endpoint matching swaps branches, Abelian results are suspect.
    match=np.abs(R[-1].conj().T@R[0])
    branch_exchange=bool(match[0,1]+match[1,0]>match[0,0]+match[1,1])
    return {**path,'W':W,'gamma':1j*np.log(W), 'links':np.array(links),
            'exchange_warning':branch_exchange,'offdiag_links':np.array(diags)}


def rectangular_parameter_loop(n_edge=9,umin=-.7,umax=.7,theta0=0.,theta1=90.):
    """Closed rectangle with no repeated terminal point."""
    if n_edge<3: raise ValueError('At least 3 points per edge')
    bot=[(u,theta0) for u in np.linspace(umin,umax,n_edge)[:-1]]
    right=[(umax,t) for t in np.linspace(theta0,theta1,n_edge)[:-1]]
    top=[(u,theta1) for u in np.linspace(umax,umin,n_edge)[:-1]]
    left=[(umin,t) for t in np.linspace(theta1,theta0,n_edge)[:-1]]
    return np.array(bot+right+top+left,dtype=float)


def real_frequency_s(phys, opts, p, *, hard=True):
    from engine_twofin_eta import solve_torch
    with torch.no_grad():
        S=solve_torch(p,phys,opts,hard=hard)
    return S.detach().cpu().numpy()


def solve_complex_s(p, physics, opts, *, freq, theta_deg=0., hard=True):
    """Complex-freq raw-amplitude zero-order S, not TORCWA power_norm=True.

    Complex frequency is an analytic parameter, not a physical incident wave.
    Constant port indices and normal-incidence lab xy are assumed. The factors
    sqrt(n_out/n_in) are freq-independent and preserve the real-axis
    power-normalized convention for propagating zero orders.
    """
    import torcwa
    if complex(freq)==0: raise ValueError('Zero frequency')
    if opts.precision!='complex128':
        raise ValueError('Complex-frequency diagnostics require complex128.')
    with torch.no_grad():
        mask=twofin_mask(p,physics,opts.raster,theta_deg,opts.edge_um,hard=hard)
        eps=(physics.n_background**2+(fin_eps(physics)-physics.n_background**2)*mask).to(torch.complex128)
        sim=torcwa.rcwa(freq=complex(freq),order=[opts.order,opts.order],
             L=[physics.period,physics.period],device=eps.device,dtype=torch.complex128)
        sim.add_input_layer(eps=physics.n_bottom**2)
        sim.add_output_layer(eps=physics.n_top**2)
        sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
        sim.add_layer(thickness=physics.height,eps=eps)
        sim.solve_global_smatrix()
        arr=np.empty((4,4),np.complex128)
        for output in range(2):
            for opol,ostr in enumerate('xy'):
                for incoming in range(2):
                    direction='forward' if incoming==0 else 'backward'
                    port='reflection' if incoming==output else 'transmission'
                    norm=(np.sqrt((physics.n_top if output else physics.n_bottom)/
                           (physics.n_top if incoming else physics.n_bottom)))
                    for ipol,istr in enumerate('xy'):
                        z=sim.S_parameters(orders=[0,0],ref_order=[0,0],
                             direction=direction,port=port,polarization=ostr+istr,
                             power_norm=False)
                        arr[2*output+opol,2*incoming+ipol]=complex(z.reshape(-1)[0].cpu())*norm
    return arr


def rational_fit(freq, signal, poles=2, *, scale=None, center=None):
    """Linearized rational approximation P_n(z)/(1+sum(q_k z^k)).

    Exploratory fit only. Poles of fit are not certified Maxwell QNMs.
    """
    f=np.asarray(freq,complex).ravel();y=np.asarray(signal,complex).ravel()
    good=np.isfinite(f)&np.isfinite(y)
    f=f[good];y=y[good]
    n=int(poles)
    if n<1 or len(f)<3*n+3:raise ValueError('Too few samples for rational fitting')
    center=complex(np.mean(f) if center is None else center)
    scale=float(max(np.ptp(f.real),np.ptp(f.imag),1e-10) if scale is None else scale)
    z=(f-center)/scale
    # y * (1 + q_1 z + ... + q_n z^n) = p_0 + ... + p_n z^n
    # Use stable scale for large |y|.
    weight=1/np.maximum(1,np.abs(y))
    B=np.column_stack([z**k for k in range(n+1)] +
                      [-y*z**k for k in range(1,n+1)])
    b=y
    x=np.linalg.lstsq(B*weight[:,None],b*weight,rcond=None)[0]
    num=x[:n+1];den=np.concatenate(([1.+0j],x[n+1:]))
    roots_p=np.roots(den[::-1])*scale+center
    roots_z=np.roots(num[::-1])*scale+center
    yhat=np.polynomial.polynomial.polyval(z,num)/np.polynomial.polynomial.polyval(z,den)
    error=float(np.linalg.norm(yhat-y)/max(np.linalg.norm(y),1e-30))
    condition=float(np.linalg.cond(B*weight[:,None]))
    return {'poles':roots_p,'zeros':roots_z,'relative_error':error,
            'condition':condition,'numerator':num,'denominator':den,
            'center':center,'scale':scale,'prediction':yhat}


def unitary_and_diffraction(physics):
    from engine_twofin_eta import diffraction_audit
    return diffraction_audit(physics)


def depth_response(p, physics, opts, *, nz=17, xy_samples=14):
    """Unit-cell average E_x/E_y(z) for x and y illuminations.

    This is TOTAL field, containing both forward and backward components;
    eigenvectors of this response do not establish a local K(z).
    """
    import torcwa
    if opts.precision!='complex128':raise ValueError('Use complex128 for depth sampling')
    with torch.no_grad():
        mask=twofin_mask(p,physics,opts.raster,0.,opts.edge_um,hard=True)
        eps=(physics.n_background**2+
             (fin_eps(physics)-physics.n_background**2)*mask).to(torch.complex128)
        sim=torcwa.rcwa(freq=1/physics.wavelength,order=[opts.order,opts.order],
              L=[physics.period,physics.period],dtype=torch.complex128,device=eps.device)
        sim.add_input_layer(eps=physics.n_bottom**2)
        sim.add_output_layer(eps=physics.n_top**2)
        sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
        sim.add_layer(thickness=physics.height,eps=eps)
        sim.solve_global_smatrix()
        ax=(torch.arange(xy_samples,device=eps.device,dtype=torch.float64)+0.5)*physics.period/xy_samples-physics.period/2
        z=np.linspace(.001*physics.height,.999*physics.height,nz)
        total=np.empty((len(z),2,2),dtype=complex)
        averaged_h=np.empty_like(total)
        for j in range(2):
            sim.source_planewave(amplitude=[1.,0.] if j==0 else [0.,1.],direction='forward')
            for i,zi in enumerate(z):
                (Ex,Ey,_),(Hx,Hy,_)=sim.field_xy(0,ax,ax,z_prop=float(zi))
                total[i,:,j]=[complex(Ex.mean().cpu()),complex(Ey.mean().cpu())]
                averaged_h[i,:,j]=[complex(Hx.mean().cpu()),complex(Hy.mean().cpu())]
    # Recovered forward-amplitude E from E and H in a homogeneous medium
    # would use transverse E+ = (E - zcross(H)/n)/2. Within patterned
    # material no such scalar-n decomposition is justified.
    return {'z':z,'E_xy_response':total,'H_xy_response':averaged_h}
