"""Connected two-fin inverse-design geometry, RCWA and array characterization.

Lengths are micrometres, angles radians unless otherwise noted. Jones ports:
(bottom x,y, top x,y); fixed laboratory polarization axes and power norm.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch

PARAM_KEYS = ('length1','width1','length2','width2','bend_deg','junction_frac')
BOUNDS = {
    'length1':(0.12,0.27), 'width1':(0.045,0.11),
    'length2':(0.12,0.27), 'width2':(0.045,0.11),
    'bend_deg':(35.,155.), 'junction_frac':(-0.23,0.23),
}

@dataclass(frozen=True)
class Physics:
    wavelength: float = 0.633
    period: float = 0.400
    n_fin: float = 2.584
    k_fin: float = 0.0       # extinction coefficient: fin index is n_fin + i k_fin (passive loss)
    n_background: float = 1.46
    n_bottom: float = 1.46
    n_top: float = 1.46
    height: float = 0.35
    cell_margin: float = 0.012

def fin_eps(physics):
    """Complex relative permittivity of the fin material."""
    return complex(physics.n_fin,physics.k_fin)**2


@dataclass(frozen=True)
class RCWA:
    order: int = 3
    raster: int = 128
    precision: str = 'complex64'
    device: str = 'cpu'
    edge_um: float = 0.003   # used only by mask='soft'
    mask: str = 'aa'         # 'aa' area fraction, 'ss' supersampled, 'soft', 'hard'
    factorization: str = 'nv'  # 'nv' normal vector (converges ~order 8) or 'laurent' (plain TORCWA)

MASK_MODES = ('aa', 'ss', 'soft', 'hard')
FACTORIZATIONS = ('laurent', 'nv')

@dataclass(frozen=True)
class Modulation:
    parameter: str = 'bend_deg'
    amplitude: float = 12.0  # +/- physical modulation (degrees for bend_deg)


def device_dtype(opts):
    return torch.float64 if opts.precision == 'complex128' else torch.float32


def design_from_raw(raw):
    """Bounded differentiable physical parameters, safe for Adam."""
    return {key: lo + (hi-lo)*torch.sigmoid(raw[i])
            for i,(key,(lo,hi)) in enumerate(BOUNDS.items())}


def raw_from_physical(values):
    output=[]
    for key,(lo,hi) in BOUNDS.items():
        u=np.clip((values[key]-lo)/(hi-lo),1e-4,1-1e-4)
        output.append(float(np.log(u/(1-u))))
    return np.asarray(output,float)


def vary(p, modulation=Modulation(), u=0.):
    if modulation.parameter not in PARAM_KEYS or modulation.parameter=='junction_frac' and modulation.amplitude>0.5:
        raise ValueError('Unsupported modulation setting')
    d=dict(p)
    d[modulation.parameter]=p[modulation.parameter] + modulation.amplitude*u
    return d


def _torch_scalar_like(x, anchor):
    return torch.as_tensor(x,device=anchor.device,dtype=anchor.dtype)


def rectangles(p, theta_deg=0.):
    """Two true overlapping rectangles; fin 2 crosses fin 1 at an internal junction."""
    L1=p['length1']; W1=p['width1']; L2=p['length2']; W2=p['width2']
    anchor=L1
    t1=_torch_scalar_like(theta_deg,anchor)*math.pi/180
    t2=t1+p['bend_deg']*math.pi/180
    u1=torch.stack((torch.cos(t1),torch.sin(t1)))
    u2=torch.stack((torch.cos(t2),torch.sin(t2)))
    # Fin 1 is centered at origin; fin 2 crosses fin 1 at a variable junction.
    center1=torch.zeros_like(u1)
    center2=p['junction_frac']*L1*u1
    center2=center2+0.12*L2*u2  # junction is inside both rectangles
    return ((center1,t1,L1,W1),(center2,t2,L2,W2))


def _signed_rectangle(x,y,center,angle,length,width):
    dx=x-center[0];dy=y-center[1]
    ca=torch.cos(angle);sa=torch.sin(angle)
    u= ca*dx+sa*dy
    v=-sa*dx+ca*dy
    ax=torch.abs(u)-length/2
    ay=torch.abs(v)-width/2
    outside=torch.sqrt(torch.relu(ax)**2+torch.relu(ay)**2+1e-24)
    inside=torch.minimum(torch.maximum(ax,ay),torch.zeros_like(ax))
    return outside+inside


def _xy_grid(physics, n, device, dtype, cells=1):
    edge=physics.period*cells
    x=(torch.arange(n,device=device,dtype=dtype)+0.5)*edge/n-edge/2
    return torch.meshgrid(x,x,indexing='ij')


def _rect_coverage(x,y,center,angle,length,width,h):
    """Fraction of each square h x h pixel covered by one rotated rectangle.

    A pixel projected onto either rectangle axis is the sum of two uniforms of
    widths h|cos| and h|sin|; its CDF (a ramp with quadratic ends) gives the
    exact coverage of a slab. The rectangle is the product of its two slabs:
    exact along edges and for axis-aligned fins, approximate only at corners.
    Area-preserving and differentiable in the edge pixels.
    """
    dx=x-center[0];dy=y-center[1]
    ca=torch.cos(angle);sa=torch.sin(angle)
    u= ca*dx+sa*dy
    v=-sa*dx+ca*dy
    a=h*torch.maximum(ca.abs(),sa.abs()); b=h*torch.clamp(torch.minimum(ca.abs(),sa.abs()),min=1e-3)
    alpha=(a+b)/2; beta=(a-b)/2
    R=lambda t: torch.relu(t).square()
    F=lambda t: ((R(t+alpha)-R(t+beta)-R(t-beta)+R(t-alpha))/(2*a*b)).clamp(0,1)
    return (F(length/2-u)-F(-length/2-u))*(F(width/2-v)-F(-width/2-v))


def twofin_mask(p,physics=Physics(),raster=128,theta_deg=0.,edge_um=0.003,
                hard=False,grid=None,mode='aa',supersample=8):
    """Fill fraction of the connected two-fin union on the RCWA raster.

    mode='aa' (default) is the differentiable area-fraction mask; 'ss' is a
    non-differentiable supersampled reference of the same coverage; 'soft' is
    the legacy sigmoid edge; hard=True (or mode='hard') is the binary staircase.
    """
    mode='hard' if hard else mode
    if mode not in MASK_MODES:raise ValueError(f'mask mode must be one of {MASK_MODES}')
    anchor=p['length1']
    X,Y=grid if grid is not None else _xy_grid(physics,raster,anchor.device,anchor.dtype)
    h=X[1,0]-X[0,0]
    if mode=='ss':
        k=supersample
        off=((torch.arange(k,device=X.device,dtype=X.dtype)+0.5)/k-0.5)*h
        X=(X[...,None,None]+off[:,None]); Y=(Y[...,None,None]+off[None,:])
    f=[]
    for center,angle,length,width in rectangles(p,theta_deg):
        if mode=='aa':
            f.append(_rect_coverage(X,Y,center,angle,length,width,h));continue
        sdf=_signed_rectangle(X,Y,center,angle,length,width)
        if mode in ('hard','ss'):f.append((sdf<=0).to(anchor.dtype))
        else:f.append(torch.sigmoid(-sdf/edge_um))
    fill=1-(1-f[0])*(1-f[1])
    return fill.mean(dim=(-2,-1)) if mode=='ss' else fill


def rectangle_corners(p,theta_deg=0.):
    corners=[]
    for center,angle,length,width in rectangles(p,theta_deg):
        ca=torch.cos(angle);sa=torch.sin(angle)
        local=torch.stack([
            torch.stack((-length/2,-width/2)),torch.stack((length/2,-width/2)),
            torch.stack((length/2,width/2)),torch.stack((-length/2,width/2))])
        rotation=torch.stack([torch.stack((ca,-sa)),torch.stack((sa,ca))])
        corners.append(local@rotation.T+center)
    return corners


def geometry_penalty(p, physics=Physics(),theta_deg=0.):
    """Discourage cell-edge violations and narrow arms; exact connectedness by construction."""
    corners=torch.cat(rectangle_corners(p,theta_deg),dim=0)
    max_coordinate=torch.abs(corners).max()
    allowed=physics.period/2-physics.cell_margin
    margin=torch.relu(max_coordinate-allowed)
    # Penalize close-to-boundary overhangs and sub-50 nm widths.
    neck=torch.relu(0.050-torch.minimum(p['width1'],p['width2']))
    return 300*margin.square()+200*neck.square(),max_coordinate


def _s_matrix(sim, device, dtype, order_out=(0,0)):
    rows=[]
    for outgoing in (0,1):
        for opol in 'xy':
            vals=[]
            for incoming in (0,1):
                direction='forward' if incoming==0 else 'backward'
                port='reflection' if incoming==outgoing else 'transmission'
                for ipol in 'xy':
                    z=sim.S_parameters(orders=list(order_out),direction=direction,port=port,
                                     polarization=opol+ipol,ref_order=[0,0],power_norm=True)
                    vals.append(z.reshape(-1)[0])
            rows.append(torch.stack(vals))
    return torch.stack(rows)


def twofin_normals(p,physics=Physics(),raster=128,theta_deg=0.,grid=None):
    """Unit normal field of the two-fin union for normal-vector factorization.

    Gradient of the union signed distance (unit norm almost everywhere, exactly
    normal on the fin walls). Its sign is irrelevant: only n_i n_j products enter.
    """
    anchor=p['length1']
    X,Y=grid if grid is not None else _xy_grid(physics,raster,anchor.device,anchor.dtype)
    sdf=torch.minimum(*[_signed_rectangle(X,Y,center,angle,length,width)
                        for center,angle,length,width in rectangles(p,theta_deg)])
    gx=torch.roll(sdf,-1,0)-torch.roll(sdf,1,0)
    gy=torch.roll(sdf,-1,1)-torch.roll(sdf,1,1)
    norm=torch.sqrt(gx.square()+gy.square()+1e-30)
    return gx/norm,gy/norm


_NV_SOLVER=None

def _nv_solver():
    """TORCWA subclass whose patterned layer uses the normal-vector in-plane permittivity.

    eps_t = [[eps]] - sym(Delta [[n n^T]]), Delta = [[eps]] - [[1/eps]]^-1 (Schuster et al.,
    JOSA A 24, 2880 (2007)): the inverse rule for E normal to the fin walls and the
    Laurent rule for tangential E. eps_zz keeps TORCWA's [[eps]]^-1 in P. The
    symmetrized product keeps eps_t Hermitian for lossless media, so the S matrix
    stays unitary and reciprocal at every order (plain Delta N is off by ~1e-3).
    Use complex128: complex64 loses unitarity at the 1e-2..1e-1 level.
    """
    global _NV_SOLVER
    if _NV_SOLVER is not None:return _NV_SOLVER
    import torcwa
    from torcwa.torch_eig import Eig

    class NVRCWA(torcwa.rcwa):
        def add_layer_nv(self,thickness,eps,nx,ny):
            E=self._material_conv(eps)
            D=E-torch.linalg.inv(self._material_conv(1/eps))
            sym=lambda N:(D@N+N@D)/2
            exx=E-sym(self._material_conv(nx*nx))
            exy=-sym(self._material_conv(nx*ny))
            eyy=E-sym(self._material_conv(ny*ny))
            I=torch.eye(self.order_N,dtype=self._dtype,device=self._device);Z=torch.zeros_like(I)
            self.eps_conv.append(E);self.mu_conv.append(I)
            self.layer_N+=1;self.thickness.append(thickness)
            K=torch.vstack((self.Kx_norm,self.Ky_norm))
            self.P.append(torch.hstack((torch.vstack((Z,-I)),torch.vstack((I,Z))))
                          +K@torch.linalg.inv(E)@torch.hstack((self.Ky_norm,-self.Kx_norm)))
            # [[0,-1],[1,0]] @ eps_t; reduces to TORCWA's [[0,-eps],[eps,0]] when isotropic.
            self.Q.append(torch.hstack((torch.vstack((-exy,exx)),torch.vstack((-eyy,exy))))
                          +K@torch.hstack((-self.Ky_norm,self.Kx_norm)))
            PQ=self.P[-1]@self.Q[-1]
            kz,V=Eig.apply(PQ) if self.stable_eig_grad else torch.linalg.eig(PQ)
            kz=torch.sqrt(kz)
            self.kz_norm.append(torch.where(torch.imag(kz)<0,-kz,kz))
            self.E_eigvec.append(V)
            self._solve_layer_smatrix()

    _NV_SOLVER=NVRCWA
    return NVRCWA


def rcwa_from_mask(fill,physics=Physics(),opts=RCWA(),period=None,normals=None):
    import torcwa  # import only inside the requested electromagnetic solve
    if opts.raster < 4*opts.order+2: raise ValueError('Insufficient raster for RCWA order')
    if opts.factorization not in FACTORIZATIONS:
        raise ValueError(f'factorization must be one of {FACTORIZATIONS}')
    nv=opts.factorization=='nv'
    if nv and normals is None:raise ValueError('NV factorization needs the normal field')
    dtype=torch.complex128 if opts.precision=='complex128' else torch.complex64
    period=physics.period if period is None else float(period)
    eps=(physics.n_background**2+
         (fin_eps(physics)-physics.n_background**2)*fill).to(dtype)
    solver=_nv_solver() if nv else torcwa.rcwa
    sim=solver(freq=1/physics.wavelength,order=[opts.order]*2,
               L=[period,period],dtype=dtype,device=fill.device)
    sim.add_input_layer(eps=physics.n_bottom**2)
    sim.add_output_layer(eps=physics.n_top**2)
    sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
    if nv:sim.add_layer_nv(physics.height,eps,*(n.to(dtype) for n in normals))
    else:sim.add_layer(thickness=physics.height,eps=eps)
    sim.solve_global_smatrix()
    return sim,_s_matrix(sim,fill.device,dtype)


def solve_torch(p,physics=Physics(),opts=RCWA(),theta_deg=0.,hard=False):
    """Specular 4x4 S; mask from opts.mask unless hard=True, factorization from opts."""
    fill=twofin_mask(p,physics,opts.raster,theta_deg,opts.edge_um,hard=hard,
                     mode=opts.mask)
    normals=(twofin_normals(p,physics,opts.raster,theta_deg)
             if opts.factorization=='nv' else None)
    return rcwa_from_mask(fill,physics,opts,normals=normals)[1]


def jones_metrics(T, *, min_relative_gap=0.002, min_nonscalar=0.002):
    """Right-eigenvector overlap; conditionally meaningful for nondegenerate T.

    Normalize T *before* evaluating the discriminant to avoid an absolute
    denominator floor that can bias weak polarization responses.
    """
    if T.shape != (2, 2):
        raise ValueError("Expected a 2x2 Jones matrix")
    real = T.real.dtype
    tiny = torch.finfo(real).tiny
    scale2 = T.abs().square().sum()
    U = T / torch.sqrt(scale2.clamp_min(tiny))
    Q = U - torch.trace(U) * torch.eye(2, device=T.device, dtype=T.dtype)/2
    q2 = Q.abs().square().sum()
    delta = (U[0,0]-U[1,1])**2 + 4*U[0,1]*U[1,0]
    disc = torch.abs(delta)
    eta2 = ((q2-disc/2).clamp_min(0) /
            (q2+disc/2).clamp_min(tiny)).clamp(0,1)
    # Small smoothing only at eta=0, not at an absolute optical amplitude.
    eta = torch.sqrt(eta2 + 1e-14)
    gap = torch.sqrt(disc.clamp_min(0))
    anis = torch.sqrt(q2.clamp_min(0))
    comm = U.conj().T @ U-U @ U.conj().T
    nonnormal = torch.linalg.matrix_norm(comm,ord='fro')
    defined = ((scale2 > 100*tiny) &
               (gap >= min_relative_gap) &
               (anis >= min_nonscalar))
    return dict(eta=eta, eta2=eta2, eta_defined=defined,
                gap=gap, anisotropy=anis,
                power=scale2/2, nonnormality=nonnormal,
                delta_abs=disc*scale2)


def diffraction_audit(physics=Physics(), *, theta_deg=0):
    """Normal-incidence exterior Rayleigh thresholds and first-order decay.

    The internal patterned dielectric is NOT a homogeneous port medium.
    """
    g = 2*np.pi/physics.period
    k0 = 2*np.pi/physics.wavelength
    data = {}
    for label, n in (('bottom', physics.n_bottom),
                     ('top', physics.n_top),
                     ('background', physics.n_background),
                     ('fin_reference', physics.n_fin)):
        cutoff = n*physics.period
        kz2 = (n*k0)**2-g**2
        data[label] = dict(index=n, first_order_cutoff_um=cutoff,
             first_order_propagating=bool(kz2 > 0),
             first_order_decay_um=(1/np.sqrt(-kz2) if kz2 < 0 else np.inf))
    data['exterior_only_specular'] = (
        not data['bottom']['first_order_propagating']
        and not data['top']['first_order_propagating'])
    data['interpretation'] = (
        'A 4x4 specular S is complete only for exterior propagating orders. '
        'RCWA still couples higher Fourier orders, including evanescent near fields.')
    return data

def numpy_metrics(S):
    arr=np.asarray(S,complex)
    T=torch.tensor(arr[2:4,0:2],dtype=torch.complex128)
    met=jones_metrics(T)
    return {k:float(v.detach()) for k,v in met.items()}


def loss_diagnostics(Ss, samples=(-1.,0.,1.), mode='span'):
    """Eta-only optical score. Other quantities are diagnostics only."""
    if len(Ss) != len(samples) or len(Ss) < 3:
        raise ValueError('Require at least three S matrices and matching samples.')
    ms=[jones_metrics(S[2:4,:2]) for S in Ss]
    eta=torch.stack([m['eta'] for m in ms])
    uu=torch.as_tensor(samples,device=eta.device,dtype=eta.dtype)
    span=eta.max()-eta.min()
    slope=(eta[-1]-eta[0])/(uu[-1]-uu[0])
    if mode not in ('span','slope'):
        raise ValueError("mode must be 'span' or 'slope'")
    loss=-span if mode=='span' else -torch.abs(slope)
    return loss,dict(slope=slope,eta_span=span,
        eta_minus=eta[0],eta_zero=eta[len(eta)//2],eta_plus=eta[-1],
        min_power=torch.stack([m['power'] for m in ms]).min(),
        min_gap=torch.stack([m['gap'] for m in ms]).min(),
        min_anisotropy=torch.stack([m['anisotropy'] for m in ms]).min(),
        usable=torch.stack([m['eta_defined'] for m in ms]).all())


def sample_three(raw,mod,physics,opts):
    p=design_from_raw(raw)
    S=[]
    for u in (-1.,0.,1.):
        S.append(solve_torch(vary(p,mod,u),physics,opts))
    return S


def circular_conversion(S):
    T=np.asarray(S)[2:4,0:2]
    # Fixed lab convention R=[1,+i]/sqrt(2), L=[1,-i]/sqrt(2)
    U=np.array([[1.,1.],[1j,-1j]])/np.sqrt(2)
    return U.conj().T@T@U


_PAULI=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])

def pauli_khat(T):
    """Complex unit Pauli vector k = t/sqrt(t.t) of T = t0 I + t.sigma (lab linear xy basis).

    k_y is the circular component: the axis of physical rotation R(theta)=exp(-i theta sigma_y).
    Torch tensors stay differentiable; numpy arrays return numpy.
    """
    if isinstance(T,torch.Tensor):
        P=torch.as_tensor(_PAULI,dtype=T.dtype,device=T.device)
        t=torch.einsum('ab,kba->k',T,P)/2
        return t/torch.sqrt((t*t).sum())
    T=np.asarray(T,complex); t=np.einsum('ab,kba->k',T,_PAULI)/2
    return t/np.sqrt(t@t)


def theta_loop_phase(T):
    """Biorthogonal geometric phases gamma_+/- = pi(1 -/+ k_y) of one rigid rotation cycle.

    Covariant limit: the meta-atom is rotated rigidly, theta: 0 -> pi (2 theta: 0 -> 2 pi).
    Re gamma is the path-dependent PB phase; Im gamma redistributes amplitude between the
    two eigenbranches. T = T^T (any z-mirror-symmetric reciprocal layer) gives k_y = 0 and
    gamma = pi exactly, whatever the eigenvector overlap. Checked against virtual_wilson.
    """
    ky=pauli_khat(T)[1]
    return math.pi*(1-ky),math.pi*(1+ky)


def transpose_asymmetry(T):
    """||T - T^T|| / ||T||: zero for z-mirror-symmetric reciprocal layers."""
    T=np.asarray(T,complex);return float(np.linalg.norm(T-T.T)/np.linalg.norm(T))


def eta_from_eigenvectors(T):
    vals,R=np.linalg.eig(np.asarray(T,complex))
    R=R/np.linalg.norm(R,axis=0,keepdims=True)
    return float(np.abs(np.vdot(R[:,0],R[:,1]))),vals,R


def stokes(jones):
    a,b=jones
    d=abs(a)**2+abs(b)**2
    return np.array([2*np.real(np.conj(a)*b)/d,2*np.imag(np.conj(a)*b)/d,
                     (abs(a)**2-abs(b)**2)/d])


def virtual_wilson(Ts):
    """Diagnostic kinematic path ONLY: not measured transport in a lateral array."""
    modes=[]
    for T in Ts:
        w,V=np.linalg.eig(np.asarray(T,complex))
        if modes:
            last=modes[-1][1]
            scores=np.abs(last.conj().T@V)
            if scores[0,0]+scores[1,1] < scores[0,1]+scores[1,0]:
                V=V[:,[1,0]];w=w[[1,0]]
        V=V/np.linalg.norm(V,axis=0,keepdims=True)
        modes.append((w,V))
    links=[]
    for j,(_,V) in enumerate(modes):
        nextV=modes[(j+1)%len(modes)][1]
        dual=np.linalg.inv(V)  # rows are reciprocal left bras
        links.append(np.diag(dual@nextV))
    product=np.prod(np.stack(links),axis=0)
    return dict(W=product,gamma=1j*np.log(product),links=np.asarray(links),
                eta=np.array([eta_from_eigenvectors(T)[0] for T in Ts]))


def local_array(u_grid,theta_grid,lookup):
    """Independent-cell Jones prediction; no ordered longitudinal transport."""
    ugrid=np.asarray(u_grid,float); theta=np.asarray(theta_grid,float)
    if ugrid.shape!=theta.shape:raise ValueError('u and theta maps must match')
    T=np.empty(ugrid.shape+(2,2),complex)
    for ij in np.ndindex(ugrid.shape):
        T[ij]=lookup(float(ugrid[ij]),float(theta[ij]))
    return T


def far_field_local(Tmap,input_jones=(1,0)):
    """Discrete coherent far-field of the local Jones approximation."""
    e=np.einsum('...ab,b->...a',Tmap,np.asarray(input_jones,complex))
    F=np.fft.fftshift(np.fft.fft2(e,axes=(0,1)),axes=(0,1))/np.prod(Tmap.shape[:2])
    return e,F,np.sum(np.abs(F)**2,axis=-1)


def supercell_mask(u_grid,theta_grid,p,mod,physics,opts,hard=False):
    """One *periodic supercell*, not an isolated finite array."""
    cells=int(np.shape(u_grid)[0])
    if np.shape(u_grid)!=(cells,cells) or np.shape(theta_grid)!=(cells,cells):
        raise ValueError('Periodic supercell must be square')
    r=opts.raster
    if r%cells:raise ValueError('Supercell raster must be divisible by cells')
    pp=r//cells
    tile=lambda f:torch.cat([torch.cat([f(vary(p,mod,float(u_grid[i,j])),float(theta_grid[i,j]))
                                        for j in range(cells)],dim=1) for i in range(cells)],dim=0)
    return tile(lambda q,t:twofin_mask(q,physics,pp,t,opts.edge_um,hard=hard,mode=opts.mask))


def supercell_normals(u_grid,theta_grid,p,mod,physics,opts):
    """Per-pixel union normals tiled like supercell_mask (for NV factorization)."""
    cells=int(np.shape(u_grid)[0]);pp=opts.raster//cells
    rows=[]
    for i in range(cells):
        nn=[twofin_normals(vary(p,mod,float(u_grid[i,j])),physics,pp,float(theta_grid[i,j]))
            for j in range(cells)]
        rows.append([torch.cat([n[k] for n in nn],dim=1) for k in (0,1)])
    return tuple(torch.cat([r[k] for r in rows],dim=0) for k in (0,1))


def supercell_solve(u_grid,theta_grid,p,mod,physics,opts):
    grid=supercell_mask(u_grid,theta_grid,p,mod,physics,opts)
    normals=(supercell_normals(u_grid,theta_grid,p,mod,physics,opts)
             if opts.factorization=='nv' else None)
    sim,S0=rcwa_from_mask(grid,physics,opts,period=physics.period*len(u_grid),
                          normals=normals)
    orders={}
    for mx in range(-opts.order,opts.order+1):
        for my in range(-opts.order,opts.order+1):
            val=(mx*mx+my*my)**0.5*physics.wavelength/(physics.period*len(u_grid))
            if val<min(physics.n_top,physics.n_bottom)-1e-8:
                orders[(mx,my)]=_s_matrix(sim,grid.device,S0.dtype,(mx,my)).detach().cpu().numpy()
    return S0.detach().cpu().numpy(),orders


def numpy_rectangles(p,theta_deg=0.):
    pp={k:torch.tensor(float(v),dtype=torch.float64) for k,v in p.items()}
    return [c.detach().numpy() for c in rectangle_corners(pp,theta_deg)]


def rect_area(poly):
    x,y=poly[:,0],poly[:,1]
    return abs(np.sum(x*np.roll(y,-1)-y*np.roll(x,-1)))/2


def check_connected(p):
    from shapely.geometry import Polygon
    a,b=[Polygon(c) for c in numpy_rectangles(p)]
    merged=a.union(b)
    intersection=a.intersection(b).area
    return intersection, bool(merged.is_valid and merged.geom_type == 'Polygon'
                              and intersection > 0)


def export_layout(p, physics, output_base):
    """Write merged union SVG; optional GDS if gdstk installed."""
    from pathlib import Path
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    output_base=Path(output_base)
    output_base.parent.mkdir(parents=True,exist_ok=True)
    merged=unary_union([Polygon(c) for c in numpy_rectangles(p)])
    if merged.geom_type!='Polygon' or len(merged.interiors):
        raise ValueError('Expected one solid connected contour without holes')
    coords=np.asarray(merged.exterior.coords[:-1])
    half=physics.period/2
    pts=' '.join(f'{(x+half)*1000:.3f},{(half-y)*1000:.3f}' for x,y in coords)
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {physics.period*1000} {physics.period*1000}">'
         f'<rect width="100%" height="100%" fill="white" stroke="black"/>'
         f'<polygon points="{pts}" fill="#2563eb"/></svg>')
    svgfile=output_base.with_suffix('.svg');svgfile.write_text(svg)
    try:
        import gdstk
    except ImportError:
        return dict(svg=str(svgfile),gds=None)
    lib=gdstk.Library(unit=1e-6,precision=1e-9)
    cell=lib.new_cell('CONNECTED_TWOFIN')
    cell.add(gdstk.Polygon(coords,layer=1))
    cell.add(gdstk.rectangle((-half,-half),(half,half),layer=10))
    gdsfile=output_base.with_suffix('.gds')
    lib.write_gds(str(gdsfile))
    return dict(svg=str(svgfile),gds=str(gdsfile))
