"""CUDA/CPU FDTD two-port scattering using flaport/fdtd.

Four independent soft-current launches produce the 4x4 S matrix. No hard
field overwrites. Fields are demodulated on-device and split into zero-order
+z/-z plane waves with Yee spatial/time staggering corrections.

IMPORTANT: the actual Maxwell time stepping is delegated to flaport/fdtd.
This is an unvalidated reference implementation pending GPU integration tests.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import numpy as np
from engine_geometry import epsilon_map

C = 299792458.
SNAPSHOT_SCHEMA_VERSION = 2  # Both flat and patterned central x-z E_x fields


@dataclass(frozen=True)
class Settings:
    dx_nm: float = 20.
    samples: int = 5
    pml_um: float = .30
    bottom_um: float = 1.20
    top_um: float = 1.20
    source_bottom: float = -.90
    source_top_offset: float = .90
    port_bottom: float = -.52
    port_top_offset: float = .52
    settle_cycles: int = 14
    measure_cycles: int = 6
    ramp_cycles: int = 4
    backend: str = 'torch.cuda.float32'

    def validate(self, cfg):
        cfg.validate()
        if self.dx_nm <= 0 or self.samples < 1 or self.measure_cycles < 2:
            raise ValueError('Invalid grid, samples or acquisition settings')
        if self.pml_um < 5*self.dx_nm/1000:
            raise ValueError('PML too thin for chosen grid')
        if not (-self.bottom_um+.05 < self.source_bottom < self.port_bottom < -.06):
            raise ValueError('Require lower PML < lower source < lower port < interface')
        if not (cfg.height+.06 < cfg.height+self.port_top_offset <
                cfg.height+self.source_top_offset < cfg.height+self.top_um-.05):
            raise ValueError('Require fin < upper port < upper source < upper PML')
        return self


class SoftPlaneCurrent:
    """Transparent additive E-current plane (no hard field overwrite)."""
    def __init__(self, z_index, pol, wavelength_m, ramp_cycles):
        self.z_index, self.pol = z_index, pol
        self.wavelength_m, self.ramp_cycles = wavelength_m, ramp_cycles

    def register(self, grid):
        self.grid = grid
        grid.sources.append(self)

    def update_E(self):
        t = (self.grid.time_steps_passed+1)*self.grid.time_step
        cycles = t * C / self.wavelength_m
        ramp = math.sin(.5*math.pi*min(1., cycles/self.ramp_cycles))**2
        # Small additive current term. Calibrated by measured incident waves.
        drive = 0.04 * ramp * math.sin(2*math.pi*cycles)
        self.grid.E[:,:,self.z_index,self.pol] += drive

    def update_H(self):
        pass


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def _setup(cfg, settings, has_fins, side, pol):
    import fdtd
    settings.validate(cfg)
    if getattr(cfg, "n_fin_kappa", 0) != 0:
        raise NotImplementedError("This FDTD engine does not model lossy fin media; use RCWA")
    fdtd.set_backend(settings.backend)
    if settings.backend.startswith('torch'):
        import torch
        torch.set_default_dtype(torch.float32 if 'float32' in settings.backend else torch.float64)
    eps, spec = epsilon_map(cfg, dx_nm=settings.dx_nm, samples=settings.samples,
                bottom_um=settings.bottom_um, top_um=settings.top_um, pml_um=settings.pml_um)
    if not has_fins:
        # Remove the fin material while preserving the air/silica interface.
        eps = np.ones_like(eps)
        eps[:,:,spec['z']<0] = cfg.n_sub**2
        near = abs(spec['z']) < spec['dx']/2
        if np.any(near):
            sub_fraction = np.clip((0-(spec['z'][near]-spec['dx']/2))/spec['dx'],0,1)
            eps[:,:,near] = 1+(cfg.n_sub**2-1)*sub_fraction[None,None,:]
    grid = fdtd.Grid(shape=eps.shape, grid_spacing=spec['dx']*1e-6, permittivity=eps)
    grid[0,:,:] = fdtd.PeriodicBoundary(name='periodic_x')
    grid[:,0,:] = fdtd.PeriodicBoundary(name='periodic_y')
    n = spec['npml']
    grid[:,:,:n] = fdtd.PML(name='zlo')
    grid[:,:,-n:] = fdtd.PML(name='zhi')
    iz_b = int(np.argmin(abs(spec['z'] - settings.port_bottom)))
    iz_t = int(np.argmin(abs(spec['z'] - (cfg.height+settings.port_top_offset))))
    source_z = (settings.source_bottom if side==0 else cfg.height+settings.source_top_offset)
    iz_src = int(np.argmin(abs(spec['z']-source_z)))
    if not (n+3 < min(iz_b,iz_src) and max(iz_t,iz_src) < spec['nz']-n-3):
        raise ValueError('Port/source too close to z-PML')
    SoftPlaneCurrent(iz_src, pol, cfg.wavelength*1e-6, settings.ramp_cycles).register(grid)
    spec.update(ports=(iz_b,iz_t), actual_ports=(float(spec['z'][iz_b]),float(spec['z'][iz_t])),
                iz_source=iz_src)
    return grid,spec


def split_plane_waves(E, H, n, omega_dt, courant_number):
    """Separate E± (laboratory x/y) at E-plane using discrete Yee phasing.

    E/H arrays have shape (2,), ordered Ex,Ey / Hx,Hy. H is located
    at z+dx/2 and time t+dt/2. FDTD numerical dispersion is used.
    """
    k_half = np.arcsin(n*np.sin(omega_dt/2)/courant_number)
    u = np.exp(1j*k_half)
    # Correct half-time offset of H relative to E.
    H = H*np.exp(+1j*omega_dt/2)
    h = np.array([H[1],-H[0]]) / n
    plus = (h + E/u) / (u + 1/u)
    return plus, E-plus


def run_case(cfg, settings, side=0, pol=0, has_fins=True, snapshot=False):
    """Return calibrated incident/outgoing modal amplitudes from one run."""
    if side not in (0,1) or pol not in (0,1):
        raise ValueError('Invalid side or polarization')
    grid, spec = _setup(cfg,settings,has_fins,side,pol)
    torch_backend = settings.backend.startswith('torch')
    cuda = settings.backend.startswith('torch.cuda')
    if torch_backend:
        import torch
        backend = torch
        dtype = grid.E.dtype
        cdtype = torch.complex64 if dtype==torch.float32 else torch.complex128
        accum = torch.zeros((2,2,4),device=grid.E.device,dtype=cdtype)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    else:
        accum = np.zeros((2,2,4),dtype=np.complex128)
    T = cfg.wavelength*1e-6/C
    omega_dt = 2*math.pi*grid.time_step/T
    total_steps = int((settings.settle_cycles + settings.measure_cycles)*T/grid.time_step)
    measure_steps = int(settings.measure_cycles*T/grid.time_step)
    if total_steps < measure_steps + 10:
        raise ValueError('Acquisition window too short')
    t0=perf_counter()
    for q in range(total_steps):
        grid.step()
        if q < total_steps-measure_steps:
            continue
        phase = complex(math.cos(omega_dt*(q+1)),math.sin(omega_dt*(q+1)))
        half = 0 if q < total_steps-measure_steps//2 else 1
        for port,iz in enumerate(spec['ports']):
            # Both E and H averaged over the complete periodic cross section.
            if torch_backend:
                ex = grid.E[:,:,iz,:2].mean(dim=(0,1))
                hx = grid.H[:,:,iz,:2].mean(dim=(0,1))
                v = torch.cat((ex,hx)).to(cdtype)
            else:
                ex = grid.E[:,:,iz,:2].mean(axis=(0,1))
                hx = grid.H[:,:,iz,:2].mean(axis=(0,1))
                v = np.concatenate((ex,hx)).astype(complex)
            accum[half,port] += phase*v
    if cuda:
        torch.cuda.synchronize()
    elapsed=perf_counter()-t0
    peak_gpu_mb = float(torch.cuda.max_memory_allocated()/2**20) if cuda else None
    # One-sided Fourier amplitude; compare acquisition halves to detect drift.
    accum = _numpy(accum) * (4 / measure_steps)
    raw = accum.mean(axis=0)
    asym = float(np.max(abs(accum[0]-accum[1])) / max(np.max(abs(raw)),1e-12))
    a=np.zeros(4,complex);b=np.zeros(4,complex)
    for port in (0,1):
        vec=raw[port]
        n=cfg.n_sub if port==0 else 1.
        plus,minus=split_plane_waves(vec[:2],vec[2:],n,omega_dt,grid.courant_number)
        # Amplitudes in sqrt(power) convention.
        a[2*port:2*port+2] = (plus if port==0 else minus)*np.sqrt(n)
        b[2*port:2*port+2] = (minus if port==0 else plus)*np.sqrt(n)
    cut = _numpy(grid.E[:,spec['ny']//2,:,pol]).copy() if snapshot else None
    result=dict(a=a,b=b,raw=raw,steady_state_residual=asym,elapsed_s=elapsed,
         steps=total_steps,steps_per_s=total_steps/elapsed,side=side,pol=pol,
         reference=not has_fins,ports=spec['actual_ports'],
         snapshot=cut,device=str(grid.E.device) if torch_backend else 'cpu',
         cuda_mb=peak_gpu_mb)
    return result


def assemble(cases):
    A=np.column_stack([d['a'] for d in cases])
    B=np.column_stack([d['b'] for d in cases])
    if np.linalg.cond(A)>1e6:
        raise RuntimeError(f'Excitation matrix ill-conditioned: cond(A)={np.linalg.cond(A):g}')
    return np.linalg.solve(A.T,B.T).T, A, B


def _paired_snapshots_valid(data):
    """Accept only two matching, finite, nonempty 2D field maps."""
    if not all(k in data for k in ('snapshot_flat_xz', 'snapshot_twofin_xz')):
        return False
    flat = np.asarray(data['snapshot_flat_xz'])
    fins = np.asarray(data['snapshot_twofin_xz'])
    return (flat.ndim == 2 and flat.size > 0 and flat.shape == fins.shape
            and np.isfinite(flat).all() and np.isfinite(fins).all())


def solve(cfg, settings=Settings(), cache='results', force=False, snapshot=False):
    """Compute power-normalized S; optionally retain paired bottom-x E_x slices.

    Scattering data are cached by geometry and numerical settings. If a cache
    predates field capture, only the two missing field launches are repeated.
    Fresh simulations capture fields during the existing eight launches.
    """
    from engine_validate import (metrics, fresnel, fresnel_yee,
                                 fdtd_to_continuum_planes)
    settings.validate(cfg)
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    signature = json.dumps({'geom': asdict(cfg), 'settings': asdict(settings)},
                           sort_keys=True)
    key = hashlib.sha256(signature.encode()).hexdigest()[:16]
    path = folder / f'fdtd_{key}.npz'

    def complete(result):
        # Only correct homogeneous-region Yee numerical propagation dispersion.
        ports = np.asarray(result['ports'])
        dx = settings.dx_nm / 1000
        result['S_fresnel_yee'] = fresnel_yee(cfg, *ports, dx)
        result['S_phase_aligned'] = fdtd_to_continuum_planes(
            result['S'], cfg, ports, dx, bare=False)
        result['S_flat_phase_aligned'] = fdtd_to_continuum_planes(
            result['S_flat'], cfg, ports, dx, bare=True)
        return result

    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as f:
            cached = {k: f[k] for k in f.files}
        if snapshot and not _paired_snapshots_valid(cached):
            print('FDTD cache contains S but no paired E_x fields; '
                  'capturing two snapshots only.', flush=True)
            flat = run_case(cfg, settings, side=0, pol=0,
                            has_fins=False, snapshot=True)['snapshot']
            fins = run_case(cfg, settings, side=0, pol=0,
                            has_fins=True, snapshot=True)['snapshot']
            cached.update(snapshot_flat_xz=flat, snapshot_twofin_xz=fins,
                          eps_snapshot=fins)  # Legacy alias
            if not _paired_snapshots_valid(cached):
                raise RuntimeError('Paired FDTD snapshots are missing or malformed')
            np.savez_compressed(path, **cached)
        return complete(cached)

    cases = []
    ref = []
    for has_fins, target in ((False, ref), (True, cases)):
        for side in (0, 1):
            for pol in (0, 1):
                print(f'FDTD {"two-fin" if has_fins else "reference"} / '
                      f'{"bottom" if side == 0 else "top"} / {"xy"[pol]}',
                      flush=True)
                target.append(run_case(cfg, settings, side, pol, has_fins,
                                       snapshot=snapshot and side == 0 and pol == 0))
    S, A, B = assemble(cases)
    Sf, Af, Bf = assemble(ref)
    zp = cases[0]['ports']
    analytic = fresnel(cfg, *zp)
    flat = ref[0]['snapshot'] if snapshot else np.array([])
    fins = cases[0]['snapshot'] if snapshot else np.array([])
    result = dict(
        S=S, S_flat=Sf, S_fresnel=analytic, ports=np.asarray(zp),
        input_matrix=A, output_matrix=B,
        acquisition_residuals=np.array([c['steady_state_residual'] for c in cases]),
        durations_s=np.array([c['elapsed_s'] for c in cases + ref]),
        cuda_peak_mb=np.array([c['cuda_mb'] if c['cuda_mb'] is not None
                               else np.nan for c in cases + ref]),
        snapshot_flat_xz=flat, snapshot_twofin_xz=fins,
        eps_snapshot=fins)  # Legacy alias; despite the name, stores E_x
    if snapshot and not _paired_snapshots_valid(result):
        raise RuntimeError('Paired FDTD snapshots are missing or malformed')
    result = complete(result)
    np.savez_compressed(path, **result)
    print('FDTD bare-interface continuum Fresnel error:', np.max(abs(Sf - analytic)))
    print('FDTD bare-interface Yee-phase Fresnel error:',
          np.max(abs(Sf - result['S_fresnel_yee'])))
    print('FDTD structure validation:', metrics(S))
    return result


def profile_gpu_steps(cfg, settings=Settings(), steps=30):
    """Profile unoptimized fdtd kernels; call only on CUDA hardware."""
    import torch
    if not settings.backend.startswith('torch.cuda'):
        raise ValueError('This profiler expects a CUDA FDTD backend')
    grid, _ = _setup(cfg,settings,True,0,0)
    for _ in range(8): grid.step()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,profile_memory=True) as prof:
        for _ in range(steps): grid.step()
    torch.cuda.synchronize()
    return prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20)