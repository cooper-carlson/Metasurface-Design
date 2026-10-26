"""Full-stack TORCWA, biorthogonal Wilson analysis, and optional 3-D Meep.

Units: micrometres, frequency 1/um, time convention exp(-i omega t).
Inputs are the exact ContourDesign.values() parameters restored from a checkpoint.
This is characterization only: there are no surrogate optimizer losses here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import numpy as np


@dataclass(frozen=True)
class Pattern:
    """Lateral supercell containing nx * ny translated copies of one contour.

    qx/qy give an integer winding of *orientation* across the supercell.
    Setting qx=qy=0 gives a uniform infinite array (or single period).
    """
    nx: int = 1
    ny: int = 1
    qx: int = 0
    qy: int = 0

    def validate(self):
        if self.nx < 1 or self.ny < 1:
            raise ValueError('Supercell dimensions must be positive integers')
        if int(self.nx) != self.nx or int(self.ny) != self.ny:
            raise ValueError('Supercell dimensions must be integers')
        if int(self.qx) != self.qx or int(self.qy) != self.qy:
            raise ValueError('Spatial winding numbers must be integers')
        return self


@dataclass(frozen=True)
class RCWAOptions:
    order: int = 4
    pixels_per_cell: int = 96
    device: str = 'cpu'
    precision: str = 'complex128'
    diffraction_budget: bool = True

    def validate(self, pattern):
        pattern.validate()
        if self.order < 1 or self.pixels_per_cell < 4*self.order+2:
            raise ValueError('Increase raster or reduce Fourier order')
        if self.precision not in ('complex64','complex128'):
            raise ValueError('Invalid precision')
        return self


def rotation_angles(N, direction=1):
    if N < 2 or int(N) != N:
        raise ValueError('N must be an integer >= 2')
    if direction not in (-1, 0, 1):
        raise ValueError('direction must be -1, 0, or +1')
    return direction*2*np.pi*np.arange(N)/N


def orientations(N, pattern, direction=1):
    """Exact theta[layer, cell-x, cell-y], same closed path per lattice site."""
    pattern.validate()
    a = np.arange(pattern.nx)[:, None] / pattern.nx
    b = np.arange(pattern.ny)[None, :] / pattern.ny
    transverse = 2*np.pi*(pattern.qx*a + pattern.qy*b)
    return rotation_angles(N, direction)[:,None,None] + transverse[None,:,:]


def _float(x):
    return float(x.detach().cpu()) if hasattr(x, 'detach') else float(x)


def _rotated_parameters(p, theta):
    # Both ellipse orientation and Fourier coefficients rotate in the same local frame.
    return dict(p, orientation=p['orientation']+float(theta))


def layer_masks(p, rules, geom, N, pattern, *, direction=1, pixels_per_cell=96):
    """Stack of rasterized XY masks, shape (N,nx*M,ny*M), x-first.

    Uses contour_fill() itself; no independent polygon/raster approximation.
    """
    import torch
    theta = orientations(N, pattern, direction)
    result = []
    for j in range(N):
        row = []
        for x in range(pattern.nx):
            col = []
            for y in range(pattern.ny):
                prm = _rotated_parameters(p, theta[j,x,y])
                col.append(geom.contour_fill(prm, rules, pixels_per_cell, hard=True))
            row.append(torch.cat(col,dim=1))
        result.append(torch.cat(row,dim=0))
    return torch.stack(result,dim=0)


def _sparameter(sim, outgoing, incoming, out_pol, in_pol, orders):
    direction = 'forward' if incoming == 0 else 'backward'
    port = 'reflection' if outgoing == incoming else 'transmission'
    return sim.S_parameters(orders=orders, ref_order=[0,0],
              direction=direction, port=port, polarization=out_pol+in_pol,
              power_norm=True).reshape(-1)


def solve_rcwa(masks, cfg, pattern, opts, *, height_um, wavelength_um=None,
               bottom_index=None, top_index=1., background_index=1.):
    """Full N-slice Maxwell solve, not a product of isolated Jones matrices.

    The 4x4 result collects zero-order ports only. The per-input `total_power`
    includes *all* represented propagating diffraction orders on both sides.
    Ports are the physical faces z=0,H: no arbitrary monitor phase is added.
    """
    import torch
    import torcwa
    opts.validate(pattern)
    N = len(masks)
    if N < 1 or height_um <= 0:
        raise ValueError('Positive thickness and at least one mask required')
    wavelength = float(wavelength_um or cfg.wavelength)
    if wavelength <= 0 or _float(cfg.n_fin_kappa) != 0:
        raise ValueError('Requires positive wavelength and real fin index')
    if masks.shape[1:] != (pattern.nx*opts.pixels_per_cell,
                            pattern.ny*opts.pixels_per_cell):
        raise ValueError('Mask shape does not match selected supercell/raster')
    cdtype = torch.complex128 if opts.precision=='complex128' else torch.complex64
    device = torch.device(opts.device)
    masks = masks.to(device=device, dtype=torch.float64 if cdtype==torch.complex128 else torch.float32)
    period_xy = [float(cfg.period)*pattern.nx, float(cfg.period)*pattern.ny]
    bottom_index = float(cfg.n_sub if bottom_index is None else bottom_index)
    sim = torcwa.rcwa(freq=1./wavelength,order=[opts.order,opts.order],
                      L=period_xy,dtype=cdtype,device=device)
    sim.add_input_layer(eps=bottom_index**2)
    sim.add_output_layer(eps=float(top_index)**2)
    sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
    for fill in masks:
        eps = (background_index**2 + (float(cfg.n_fin)**2-background_index**2)*fill).to(cdtype)
        sim.add_layer(thickness=float(height_um)/N,eps=eps)
    sim.solve_global_smatrix()

    # Specular 4x4 in (bottom x,y; top x,y) order.
    S = np.zeros((4,4),complex)
    for out_side in (0,1):
        for oi, op in enumerate('xy'):
            for in_side in (0,1):
                for ii, ip in enumerate('xy'):
                    k = (out_side*2+oi,in_side*2+ii)
                    S[k] = complex(_sparameter(sim,out_side,in_side,op,ip,[[0,0]])[0].detach().cpu())

    budget = None
    if opts.diffraction_budget:
        all_orders = [[i,j] for i in range(-opts.order,opts.order+1)
                            for j in range(-opts.order,opts.order+1)]
        # Output powers for one unit-amplitude, power-normalized input channel.
        budget = np.zeros((4,2),float)
        for in_side in (0,1):
            for ii, ip in enumerate('xy'):
                for out_side in (0,1):
                    value = 0.
                    for op in 'xy':
                        v = _sparameter(sim,out_side,in_side,op,ip,all_orders)
                        value += float(v.abs().square().sum().detach().cpu())
                    budget[in_side*2+ii,out_side] = value
    return dict(S=S, T=S[2:4,:2], budget=budget, total_power=(budget.sum(1) if budget is not None else None),
                wavelength_um=wavelength, N=N, thickness_um=height_um,
                period_xy_um=period_xy, mode='periodic_rcwa')


def solve_stack(p, rules, geom, cfg, N, *, pattern=Pattern(), opts=RCWAOptions(),
                direction=1, height_um=None, wavelength_um=None,
                bottom_index=None, top_index=1., background_index=1.):
    height_um = _float(p['height']) if height_um is None else float(height_um)
    masks = layer_masks(p,rules,geom,N,pattern,direction=direction,
                        pixels_per_cell=opts.pixels_per_cell)
    output = solve_rcwa(masks,cfg,pattern,opts,height_um=height_um,
                        wavelength_um=wavelength_um,bottom_index=bottom_index,
                        top_index=top_index,background_index=background_index)
    output['direction'] = direction
    return output


def solve_local_slices(p, rules, geom, cfg, N, *, opts=RCWAOptions(), height_um=None,
                       ambient_index=1., wavelength_um=None, direction=1):
    """Isolated ambient-matched *thin slice* transfer matrices, no substrate.

    These are a Jones transport surrogate, not a local Maxwell generator.
    """
    height_um = _float(p['height']) if height_um is None else float(height_um)
    matrices=[]
    masks=layer_masks(p,rules,geom,N,Pattern(),direction=direction,
                      pixels_per_cell=opts.pixels_per_cell)
    for j in range(N):
        out=solve_rcwa(masks[j:j+1],cfg,Pattern(),opts,
                       height_um=height_um/N,wavelength_um=wavelength_um,
                       bottom_index=ambient_index,top_index=ambient_index,
                       background_index=ambient_index)
        matrices.append(out['T'])
    return np.asarray(matrices)


def tracked_biorthogonal(Ts, *, max_condition=1e8):
    """Track right modes; dual rows inv(R) obey L @ R = I.

    Uses eigenvalue-distance plus state similarity in a two-branch assignment,
    with overlap as a tie-breaker. Phase fixing is optional for Wilson closure.
    """
    Ts=np.asarray(Ts,complex)
    if Ts.ndim!=3 or Ts.shape[1:]!=(2,2):
        raise ValueError('Expected Jones matrices (N,2,2)')
    R_list=[];L_list=[];eig_list=[];conds=[];eta=[];gap=[]
    for j,T in enumerate(Ts):
        eig,R = np.linalg.eig(T)
        R=R/np.linalg.norm(R,axis=0)
        if j:
            # Combine polarization and eigenvalue continuity to avoid spurious
            # branch switches for large angular steps.
            prev_gap=max(abs(eig_list[-1][0]-eig_list[-1][1]),1e-12)
            def score(k):
                return sum(abs(np.vdot(R_list[-1][:,i],R[:,k[i]]))**2
                           - 0.75*abs(eig_list[-1][i]-eig[k[i]])**2/prev_gap**2
                           for i in range(2))
            order=max(permutations(range(2)),key=score)
            eig=eig[list(order)];R=R[:,list(order)]
        condition=np.linalg.cond(R)
        if not np.isfinite(condition) or condition>max_condition:
            raise ValueError(f'Nearly singular eigenbasis at slice {j}: condition={condition:.3g}')
        L=np.linalg.inv(R)
        R_list.append(R);L_list.append(L);eig_list.append(eig)
        conds.append(condition)
        eta.append(abs(np.vdot(R[:,0],R[:,1])))
        gap.append(abs(eig[0]-eig[1])/max(np.linalg.norm(T,'fro'),1e-12))
    return dict(R=np.asarray(R_list), L=np.asarray(L_list),
                eigenvalues=np.asarray(eig_list),condition=np.asarray(conds),
                eta=np.asarray(eta),normalized_gap=np.asarray(gap))


def wilson_cycle(track):
    """Paper convention W=prod(L_j R_{j+1}), gamma=+i*log(W).

    Because all links close on the *same stored* first basis, W is gauge
    invariant. Phases modulo 2pi; use continuous tracking across N externally.
    """
    R,L=track['R'],track['L']
    N=len(R)
    links=np.array([[L[j,i,:]@R[(j+1)%N,:,i] for i in range(2)]
                    for j in range(N)],complex)
    if np.any(abs(links)<1e-12):
        raise ValueError('Vanishing biorthogonal link: geometric log undefined')
    W=np.prod(links,axis=0)
    gamma=1j*np.log(W)  # -angle(W) + i log|W|; per paper convention
    # Forward mode projection is L_(j+1) @ R_j, opposite link ordering.
    forward_links=np.array([[L[(j+1)%N,i,:]@R[j,:,i] for i in range(2)]
                            for j in range(N)],complex)
    forward_product=np.prod(forward_links,axis=0)
    edge=L[-1]@R[0]
    branch_warning=(abs(edge[0,1])+abs(edge[1,0]) >
                    1.2*(abs(edge[0,0])+abs(edge[1,1])))
    return dict(links=links, W=W, gamma=gamma,
                forward_projection_links=forward_links,
                forward_projection_product=forward_product,
                forward_projection_gamma=-1j*np.log(forward_product),
                branch_exchange_warning=bool(branch_warning),
                loop_modulus=np.abs(W),
                biorthogonal_identity_error=max(np.linalg.norm(L[j]@R[j]-np.eye(2)) for j in range(N)),
                cyclic_link_magnitudes=np.abs(links[-1]))


def transport_analysis(Ts, *, reference_wavelength_um=None, slice_height_um=None):
    """Wilson holonomy and isolated-slice eigenvalue-product surrogate.

    Product eigenvalues include Fresnel/resonant scattering and are *not* an
    unambiguous dynamical phase of a physical propagation generator.
    """
    t=tracked_biorthogonal(Ts)
    w=wilson_cycle(t)
    eigenprod=np.prod(t['eigenvalues'],axis=0)
    return dict(**t,**w,
        eigenvalue_product=eigenprod,
        eigenvalue_phase=np.angle(eigenprod),
        eigenvalue_log_amplitude=np.log(np.maximum(abs(eigenprod),1e-300)),
        effective_generator_eigenvalues=(1j*np.log(t['eigenvalues'])/slice_height_um
            if slice_height_um else None))


def loop_from_model(N, eta, direction=1):
    """Analytical benchmark from Jana & Genevet, NOT assumed for a real contour."""
    if not 0<=eta<1:raise ValueError('Requires 0 <= eta < 1')
    value=np.pi*eta/np.sqrt(1-eta**2)
    return np.array([np.pi+1j*direction*value, np.pi-1j*direction*value])


def specular_metrics(S):
    S=np.asarray(S,complex)
    Tf=S[2:4,:2]
    return dict(T_forward=np.linalg.norm(Tf,'fro')**2/2,
                R_forward=np.linalg.norm(S[:2,:2],'fro')**2/2,
                reciprocity_error=np.linalg.norm(S-S.T,'fro'),
                specular_unitarity_error=np.linalg.norm(S.conj().T@S-np.eye(4),'fro'),
                # For supercells these are *not* full-array energy diagnostics.
                circular_asymmetry=circular_asymmetry(Tf))


def circular_asymmetry(T):
    uR=np.array([1,-1j])/np.sqrt(2);uL=np.array([1,1j])/np.sqrt(2)
    a=abs(np.vdot(uR,T@uL))**2
    b=abs(np.vdot(uL,T@uR))**2
    return (a-b)/(a+b+1e-15)


def projected_channel(T, u, v):
    u=np.asarray(u,complex);v=np.asarray(v,complex)
    u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
    return np.vdot(v,np.asarray(T)@u)


def one_pole_background(freq_um, complex_amplitudes, *, min_width=1e-4):
    """Real-frequency *phenomenological* Fano fit; not a certified QNM pole.

    Frequency is 1/um; fit t(f)=a0+a1(f-f0)+r/(f-(fr-i*kappa)).
    The background/resonance phasors add, so phase isn't additive.
    """
    from scipy.optimize import least_squares
    freq=np.asarray(freq_um,float)
    y=np.asarray(complex_amplitudes,complex)
    if len(freq)<9 or freq.shape!=y.shape or not np.all(np.isfinite(y)):
        raise ValueError('At least 9 finite complex frequency samples required')
    x=(freq-freq.mean())/(freq.max()-freq.min())
    # Normalize the frequency range to make the least-squares parameters stable.
    if not np.all(np.diff(freq)>0):raise ValueError('Increasing frequencies required')
    def model(z,return_parts=False):
        a0=z[0]+1j*z[1];a1=z[2]+1j*z[3];r=z[4]+1j*z[5]
        pole=z[6]-1j*np.exp(z[7]);bg=a0+a1*x
        res=r/(x-pole)
        return (bg,res) if return_parts else bg+res
    best=None
    for seed in (-.7,0.,.7):
        p0=np.array([y.real.mean(),y.imag.mean(),0,0,.01,.01,seed,np.log(.15)])
        fit=least_squares(lambda z: np.r_[np.real(model(z)-y),np.imag(model(z)-y)],p0,
              bounds=([-np.inf]*6+[-2,np.log(min_width)],
                      [np.inf]*6+[2,np.log(10.)]),max_nfev=3500)
        if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun): best=fit
    bg,res=model(best.x,True)
    pred=bg+res
    span=freq.max()-freq.min()
    f_pole=freq.mean()+span*(best.x[6]-1j*np.exp(best.x[7]))
    return dict(frequency_um=freq,observed=y,background=bg,resonant=res,
                fitted=pred,complex_pole_um=f_pole,
                residual_relative=np.linalg.norm(pred-y)/max(np.linalg.norm(y),1e-15),
                excess_phase=np.angle(pred/(bg+1e-30)),
                note='One-pole phenomenological scattering decomposition; phase of sum is not sum of phases.')

@dataclass(frozen=True)
class FiniteOptions:
    """3D Meep finite-aperture validation, disabled until explicitly requested."""
    nx: int = 3
    ny: int = 3
    resolution: int = 30            # voxels per um; converge before making claims
    pml_um: float = 0.30
    lateral_pad_um: float = 0.35
    z_pad_wavelengths: float = 1.20
    decay_after: float = 120.       # time after source turn-off, Meep units
    pulse_width_freq: float = 0.12  # relative spectral width for pulsed source
    n_profile_points: int = 256

    def validate(self):
        if min(self.nx,self.ny,self.resolution,self.n_profile_points)<=0:
            raise ValueError('Invalid finite-domain discretization')
        if self.pml_um<=0 or self.lateral_pad_um<=0:
            raise ValueError('Finite domain requires lateral PML/padding')
        return self


def finite_prism_specs(p, rules, geom, N, pattern, *, height_um=None, direction=1,
                       angular_samples=256):
    """Geometric input to Meep, testable without Meep itself."""
    import torch
    if pattern.nx < 1 or pattern.ny < 1:
        raise ValueError('Empty array')
    H=_float(p['height']) if height_um is None else float(height_um)
    if H<=0:raise ValueError('Total height must be positive')
    vertices=[]
    sample=torch.arange(angular_samples,device=p['radius'].device,dtype=p['radius'].dtype)
    angles=2*torch.pi*sample/angular_samples
    theta=orientations(N,pattern,direction)
    for j in range(N):
        zmin=-H/2+j*H/N
        for ix in range(pattern.nx):
            for iy in range(pattern.ny):
                q=_rotated_parameters(p,theta[j,ix,iy])
                local=geom.contour_points(q,rules,angles=angles).detach().cpu().numpy()
                cx=(ix-(pattern.nx-1)/2)*float(rules.period)
                cy=(iy-(pattern.ny-1)/2)*float(rules.period)
                vertices.append(dict(layer=j,cell=(ix,iy),
                       xy=local+np.array([cx,cy]), zmin=zmin, zmax=zmin+H/N))
    return vertices


def finite_meep_forward(p, rules, geom, cfg, N, *, opts=FiniteOptions(),
                        height_um=None, wavelength_um=None, direction=1,
                        save_fields=False, pattern=None):
    """Explicit 3D Maxwell simulation with XY/Z PML and a finite source aperture.

    Returns a *forward-only*, aperture-projected 2x2 Jones matrix RELATIVE TO
    THE BARE SUBSTRATE-AIR INTERFACE. This is not a 4x4 port scattering matrix,
    nor a total outgoing-power budget. Spatial field arrays may be returned.

    Six runs: two input polarizations each for uniform substrate, bare
    substrate-air interface, and finite patterned device. The incident-plane
    source is truncated; fields at diffraction/edge angles need monitoring
    separately for a rigorous power budget.
    """
    try:
        import meep as mp
    except ImportError as exc:
        raise ImportError('Finite arrays require a separate Meep 3D installation. '
                          'Install Meep in a supported environment; periodic RCWA '
                          'and analytical sections do not require it.') from exc
    opts.validate()
    pattern=Pattern(nx=opts.nx,ny=opts.ny) if pattern is None else pattern.validate()
    if (pattern.nx,pattern.ny)!=(opts.nx,opts.ny):
        raise ValueError('Finite spatial-winding pattern must match array dimensions')
    H=_float(p['height']) if height_um is None else float(height_um)
    lam=float(cfg.wavelength if wavelength_um is None else wavelength_um)
    slice_pixels = opts.resolution * H / N
    if slice_pixels < 4:
        raise ValueError(f'FDTD cannot resolve {H/N*1000:.2f} nm slices with only '
                         f'{slice_pixels:.2f} cells/layer. Raise resolution to '
                         f'at least {int(np.ceil(4*N/H))} px/um and converge.')
    xy_extent=np.array([opts.nx,opts.ny])*float(cfg.period)
    transverse=xy_extent+2*(opts.lateral_pad_um+opts.pml_um)
    zpad=opts.z_pad_wavelengths*lam
    sz=H+2*(zpad+opts.pml_um)
    cell=mp.Vector3(float(transverse[0]),float(transverse[1]),float(sz))
    pml=[mp.PML(opts.pml_um,direction=mp.X),mp.PML(opts.pml_um,direction=mp.Y),
         mp.PML(opts.pml_um,direction=mp.Z)]
    f=1./lam
    zsource=-H/2-0.72*zpad
    zreflection=-H/2-0.42*zpad
    ztransmission=H/2+0.42*zpad
    # Transverse source does not touch PML; it illuminates the entire array.
    source_xy=xy_extent+opts.lateral_pad_um
    observation_xy=xy_extent
    volume_t=mp.Volume(center=mp.Vector3(0,0,ztransmission),
                       size=mp.Vector3(*map(float,observation_xy),0))
    volume_r=mp.Volume(center=mp.Vector3(0,0,zreflection),
                       size=mp.Vector3(*map(float,observation_xy),0))
    specs=finite_prism_specs(p,rules,geom,N,pattern,height_um=H,
                             direction=direction,angular_samples=opts.n_profile_points)
    base_mat=mp.Medium(index=float(cfg.n_sub))
    fin_mat=mp.Medium(index=float(cfg.n_fin))
    geom_fin=[]
    for shape in specs:
        # vertices at physical XY coordinates; zmin is the prism floor.
        vertices=[mp.Vector3(float(x),float(y),float(shape['zmin']))
                  for x,y in shape['xy']]
        geom_fin.append(mp.Prism(vertices=vertices,height=H/N,
                                axis=mp.Vector3(0,0,1),material=fin_mat))
    # Semi-infinite substrate until the plane of the device entrance.
    # The block extends through lower Z PML to avoid artificial backside reflections.
    substrate=mp.Block(center=mp.Vector3(0,0,(-sz/2-H/2)/2),
                       size=mp.Vector3(mp.inf,mp.inf,(sz-H)/2),material=base_mat)

    def launch(source_pol,geometry,default_material=None):
        src=mp.Source(src=mp.GaussianSource(frequency=f,fwidth=f*opts.pulse_width_freq),
                      component=mp.Ex if source_pol=='x' else mp.Ey,
                      center=mp.Vector3(0,0,zsource),
                      size=mp.Vector3(float(source_xy[0]),float(source_xy[1]),0))
        sim=mp.Simulation(cell_size=cell,dimensions=3,resolution=opts.resolution,
                 boundary_layers=pml,geometry=geometry,sources=[src],
                 default_material=default_material or mp.air)
        tr=sim.add_dft_fields([mp.Ex,mp.Ey],f,0,1,where=volume_t)
        rr=sim.add_dft_fields([mp.Ex,mp.Ey],f,0,1,where=volume_r)
        # Wait for DFT monitors to settle after the pulse; decay condition is
        # measured near output, not a fixed number of timesteps.
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
                   opts.decay_after,mp.Ex if source_pol=='x' else mp.Ey,
                   mp.Vector3(0,0,ztransmission),1e-6))
        transmitted=np.stack([sim.get_dft_array(tr,mp.Ex,0),sim.get_dft_array(tr,mp.Ey,0)])
        reflected=np.stack([sim.get_dft_array(rr,mp.Ex,0),sim.get_dft_array(rr,mp.Ey,0)])
        sim.reset_meep()
        return transmitted,reflected

    records={}
    for pol in 'xy':
        # Common source/port phase convention for each polarization.
        incident=launch(pol,[],base_mat)
        bare=launch(pol,[substrate])
        device=launch(pol,[substrate,*geom_fin])
        records[pol]=(incident,bare,device)

    jones=np.zeros((2,2),complex)
    rexcess=np.zeros((2,2),complex)
    incident_ratio=np.zeros((2,2),complex)
    for j,pol in enumerate('xy'):
        (incT,incR),(bareT,bareR),(devT,devR)=records[pol]
        reference=bareT[j].reshape(-1)
        norm=np.vdot(reference,reference)
        if abs(norm)<1e-15:
            raise RuntimeError('Vanishing reference transmitted field')
        inref=incR[j].reshape(-1)
        inpower=max(np.real(np.vdot(inref,inref)),1e-15)
        for i in range(2):
            jones[i,j]=np.vdot(reference,devT[i].reshape(-1))/norm
            # Total reflected excess over uniform substrate illumination:
            rexcess[i,j]=np.vdot(inref,(devR[i]-incR[i]).reshape(-1))/inpower
            incident_ratio[i,j]=np.vdot(inref,bareR[i].reshape(-1))/inpower
    result=dict(T_relative_bare=jones,R_relative_incident=rexcess,
         R_bare_relative_incident=incident_ratio,
         source='3D Meep finite aperture and XY/Z PML',
         caveat='Forward aperture-projected complex field, NOT all diffraction power or full S',
         N=N,direction=direction,nx=opts.nx,ny=opts.ny,resolution=opts.resolution,pattern=vars(pattern))
    if save_fields:
        result['fields']={pol: dict(transmission=records[pol][2][0],
                            reflection=records[pol][2][1],
                            bare_transmission=records[pol][1][0]) for pol in 'xy'}
    return result

# --- Local-generator diagnostics: a two-way transfer proxy, NOT QNM/Bloch solver. ---

def scattering_to_transfer(S, *, max_condition=1e9):
    """Convert matched-face, two-sided, specular 4x4 S into a 4x4 transfer M.

    [b_forward, b_backward]^T = M [a_forward, a_backward]^T.
    Requires the reverse transmission block to be invertible. In a stack with
    non-negligible evanescent coupling this specular transfer is NOT closed.
    """
    S=np.asarray(S,dtype=complex)
    if S.shape!=(4,4):raise ValueError('S must be 4x4')
    R11,T12,T21,R22=S[:2,:2],S[:2,2:],S[2:,:2],S[2:,2:]
    cond=np.linalg.cond(T12)
    if not np.isfinite(cond) or cond>max_condition:
        raise ValueError(f'Reverse transmission block ill-conditioned: {cond:.3g}')
    X=np.linalg.solve(T12,np.eye(2))
    M=np.block([[T21-R22@X@R11,R22@X],[-X@R11,X]])
    return M


def forward_generator_proxy(S, thickness_um, *, min_flux=1e-5,
                            max_condition=1e9):
    """Forward 2-mode polarization proxy extracted from log of 4x4 transfer.

    Use only for same ambient on both sides of an isolated slice. Select
    transfer eigenvectors with positive incoming directional flux proxy
    ||a+||²-||a-||², where a+ and a- are power-normalized plane-wave ports.
    Branches of log and evanescent modes require independent diagnostics.
    """
    from scipy.linalg import logm
    if thickness_um<=0:raise ValueError('Positive slice thickness required')
    M=scattering_to_transfer(S,max_condition=max_condition)
    K=1j*logm(M)/thickness_um
    vals,V=np.linalg.eig(K)
    flux=np.sum(np.abs(V[:2])**2,axis=0)-np.sum(np.abs(V[2:])**2,axis=0)
    forward=np.flatnonzero(flux>min_flux)
    if len(forward)!=2:
        raise ValueError(f'Need two positive-flux modes, found {len(forward)}; '
                         'do not assign a local geometric phase to this slice')
    R=V[:2,forward]
    norm=np.linalg.norm(R,axis=0)
    if np.any(norm<1e-9):raise ValueError('Forward electric-field projection vanishes')
    R=R/norm
    cond=np.linalg.cond(R)
    if cond>max_condition:raise ValueError('Nearly defective projected local basis')
    local_J=R@np.diag(vals[forward])@np.linalg.inv(R)
    return dict(K4=K,transfer=M,local_J=local_J,
                forward_lambda=vals[forward],right=R,
                eta=float(abs(np.vdot(R[:,0],R[:,1]))),
                polarization_condition=float(cond),
                flux=flux,transfer_condition=float(np.linalg.cond(M)))


def orthogonal_projection_baseline(N, *, winding=1):
    """Only a reference for a uniformly rotating, orthogonal Jones basis.

    Finite-step projection gives gamma.imag = N log|cos(2*pi*winding/N)|
    (unless a link vanishes); it disappears in the continuum limit.
    """
    if N<2:raise ValueError('N >= 2')
    c=abs(np.cos(2*np.pi*winding/N))
    return float('nan') if c<1e-12 else N*np.log(c)


def jones_metrics_numpy(T, *, eps=1e-12):
    T=np.asarray(T,complex)
    values,R=np.linalg.eig(T)
    eta=float(abs(np.vdot(R[:,0],R[:,1])))
    energy=max(float(np.linalg.norm(T,'fro')**2),eps)
    Q=T-0.5*np.trace(T)*np.eye(2)
    return dict(eta=eta,normalized_gap=float(abs(values[0]-values[1])/np.sqrt(energy)),
        anisotropy=float(np.linalg.norm(Q,'fro')/np.sqrt(energy)),
        mean_power=energy/2,
        nonnormality=float(np.linalg.norm(T.conj().T@T-T@T.conj().T,'fro')/energy))
