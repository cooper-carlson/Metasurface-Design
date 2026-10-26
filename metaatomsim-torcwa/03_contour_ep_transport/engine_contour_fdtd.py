"""Independent flaport/fdtd contour validation, using polygon subpixel fill.

Preserves the previous engine_fdtd plane-wave calibration and Yee-dispersion
reference-plane correction; does not reuse its two-fin geometry.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import numpy as np
from engine_contour_validation import epsilon_map_contour
from engine_fdtd import Settings, SoftPlaneCurrent, split_plane_waves, assemble, _numpy, C


def _setup(cfg,settings,points,patterned,side,pol):
    import fdtd
    settings.validate(cfg)
    fdtd.set_backend(settings.backend)
    if settings.backend.startswith('torch'):
        import torch
        torch.set_default_dtype(torch.float32 if 'float32' in settings.backend else torch.float64)
    eps,spec=epsilon_map_contour(cfg,points,dx_nm=settings.dx_nm,samples=settings.samples,
        bottom_um=settings.bottom_um,top_um=settings.top_um,pml_um=settings.pml_um,
        with_contour=patterned)
    grid=fdtd.Grid(shape=eps.shape,grid_spacing=spec['dx']*1e-6,permittivity=eps)
    grid[0,:,:]=fdtd.PeriodicBoundary(name='periodic_x')
    grid[:,0,:]=fdtd.PeriodicBoundary(name='periodic_y')
    n=spec['npml']
    grid[:,:,:n]=fdtd.PML(name='zlo')
    grid[:,:,-n:]=fdtd.PML(name='zhi')
    iz_b=int(np.argmin(abs(spec['z']-settings.port_bottom)))
    iz_t=int(np.argmin(abs(spec['z']-(cfg.height+settings.port_top_offset))))
    source_z=settings.source_bottom if side==0 else cfg.height+settings.source_top_offset
    iz_src=int(np.argmin(abs(spec['z']-source_z)))
    if not (n+3<min(iz_b,iz_src) and max(iz_t,iz_src)<spec['nz']-n-3):
        raise ValueError('Source or port too close to z-PML')
    SoftPlaneCurrent(iz_src,pol,cfg.wavelength*1e-6,settings.ramp_cycles).register(grid)
    spec.update(ports=(iz_b,iz_t),actual_ports=(float(spec['z'][iz_b]),float(spec['z'][iz_t])))
    return grid,spec


def run_case(cfg,settings,points,side=0,pol=0,patterned=True,snapshot=False):
    """One soft-plane x/y excitation; report power-normalized a,b vectors."""
    if side not in (0,1) or pol not in (0,1):
        raise ValueError('Invalid incident channel')
    grid,spec=_setup(cfg,settings,points,patterned,side,pol)
    torch_backend=settings.backend.startswith('torch')
    cuda=settings.backend.startswith('torch.cuda')
    if torch_backend:
        import torch
        cdtype=torch.complex64 if grid.E.dtype==torch.float32 else torch.complex128
        accum=torch.zeros((2,2,4),device=grid.E.device,dtype=cdtype)
        if cuda:
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    else:
        accum=np.zeros((2,2,4),complex)
    T=cfg.wavelength*1e-6/C
    omega_dt=2*math.pi*grid.time_step/T
    total_steps=int((settings.settle_cycles+settings.measure_cycles)*T/grid.time_step)
    measure_steps=int(settings.measure_cycles*T/grid.time_step)
    if total_steps<measure_steps+10:
        raise ValueError('Acquisition too short')
    t0=perf_counter()
    for q in range(total_steps):
        grid.step()
        if q<total_steps-measure_steps:
            continue
        phase=complex(math.cos(omega_dt*(q+1)),math.sin(omega_dt*(q+1)))
        half=0 if q<total_steps-measure_steps//2 else 1
        for port,iz in enumerate(spec['ports']):
            if torch_backend:
                v=torch.cat((grid.E[:,:,iz,:2].mean(dim=(0,1)),
                             grid.H[:,:,iz,:2].mean(dim=(0,1)))).to(cdtype)
            else:
                v=np.concatenate((grid.E[:,:,iz,:2].mean(axis=(0,1)),
                                  grid.H[:,:,iz,:2].mean(axis=(0,1)))).astype(complex)
            accum[half,port]+=phase*v
    if cuda:torch.cuda.synchronize()
    elapsed=perf_counter()-t0
    peak_mb=float(torch.cuda.max_memory_allocated()/2**20) if cuda else np.nan
    arr=_numpy(accum)*(4/measure_steps)
    raw=arr.mean(axis=0)
    residual=float(np.max(abs(arr[0]-arr[1]))/max(np.max(abs(raw)),1e-12))
    a=np.zeros(4,complex);b=np.zeros(4,complex)
    for port in (0,1):
        n=cfg.n_sub if port==0 else 1.
        plus,minus=split_plane_waves(raw[port,:2],raw[port,2:],n,omega_dt,grid.courant_number)
        a[2*port:2*port+2]=(plus if port==0 else minus)*np.sqrt(n)
        b[2*port:2*port+2]=(minus if port==0 else plus)*np.sqrt(n)
    cut=_numpy(grid.E[:,spec['ny']//2,:,0]).copy() if snapshot else np.array([])
    return dict(a=a,b=b,ports=spec['actual_ports'],residual=residual,
                elapsed_s=elapsed,cuda_mb=peak_mb,snapshot=cut)


def solve(cfg,settings,points,*,cache='results/contour_validation',force=False,snapshot=False):
    """Eight FDTD launches, cache keyed by actual vertices and solver settings."""
    from engine_validate import fresnel,fresnel_yee,fdtd_to_continuum_planes
    settings.validate(cfg)
    points=np.asarray(points,float)
    h=hashlib.sha256()
    h.update(np.ascontiguousarray(points).tobytes())
    h.update(json.dumps(dict(config=asdict(cfg),settings=asdict(settings)),
                        sort_keys=True).encode())
    path=Path(cache)/f'contour_fdtd_{h.hexdigest()[:16]}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    def complete(d):
        zp=tuple(float(x) for x in d['ports']); dx=settings.dx_nm/1000
        d['S_fresnel']=fresnel(cfg,*zp)
        d['S_fresnel_yee']=fresnel_yee(cfg,*zp,dx)
        d['S_flat_phase_aligned']=fdtd_to_continuum_planes(
            d['S_flat'],cfg,zp,dx,bare=True)
        d['S_phase_aligned']=fdtd_to_continuum_planes(
            d['S'],cfg,zp,dx,bare=False)
        return d
    if path.exists() and not force:
        with np.load(path,allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
        if not snapshot or (d.get('snapshot_flat_xz',np.array([])).size and
                            d.get('snapshot_contour_xz',np.array([])).size):
            print('Reused contour FDTD cache:',path)
            return complete(d)
        # Reuse saved S; acquire only missing field snapshots.
        d['snapshot_flat_xz']=run_case(cfg,settings,points,0,0,False,True)['snapshot']
        d['snapshot_contour_xz']=run_case(cfg,settings,points,0,0,True,True)['snapshot']
        np.savez_compressed(path,**d)
        return complete(d)
    refs=[]; cases=[]
    for patterned,target in ((False,refs),(True,cases)):
        for side in (0,1):
            for pol in (0,1):
                print(f'FDTD {"contour" if patterned else "bare"} / '
                      f'{"bottom" if side==0 else "top"} / {"xy"[pol]}',flush=True)
                target.append(run_case(cfg,settings,points,side,pol,patterned,
                          snapshot=snapshot and side==0 and pol==0))
    S,A,B=assemble(cases)
    Sf,Af,Bf=assemble(refs)
    d=dict(S=S,S_flat=Sf,ports=np.asarray(cases[0]['ports']),
           input_matrix=A,output_matrix=B,
           acquisition_residuals=np.array([c['residual'] for c in cases]),
           durations_s=np.array([c['elapsed_s'] for c in cases+refs]),
           cuda_peak_mb=np.array([c['cuda_mb'] for c in cases+refs]),
           snapshot_flat_xz=refs[0]['snapshot'] if snapshot else np.array([]),
           snapshot_contour_xz=cases[0]['snapshot'] if snapshot else np.array([]))
    np.savez_compressed(path,**d)
    return complete(d)
