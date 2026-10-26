"""Independent geometric rasterization and diagnostics for contour validation.

The numpy polygon sampling here does not use the optimizer's soft-mask
implementation. Length units: micrometers. Channel order: bx, by, tx, ty.
"""
from pathlib import Path
import numpy as np


def polygon_coverage(points, period, nx, ny=None, samples=5):
    """Subpixel area fraction of a single closed polygon in one unit cell."""
    from matplotlib.path import Path as MplPath
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('Expected an array of contour vertices (N,2)')
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite polygon vertices')
    if samples < 1:
        raise ValueError('samples must be >= 1')
    ny = nx if ny is None else ny
    x = (np.arange(nx) + .5) * period / nx - period/2
    y = (np.arange(ny) + .5) * period / ny - period/2
    X,Y = np.meshgrid(x,y,indexing='ij')
    path = MplPath(np.vstack((points,points[0])))
    coverage = np.zeros((nx,ny), dtype=np.float32)
    for i in range(samples):
        for j in range(samples):
            subx = X + ((i+.5)/samples-.5) * period/nx
            suby = Y + ((j+.5)/samples-.5) * period/ny
            inside = path.contains_points(np.column_stack((subx.ravel(),suby.ravel())))
            coverage += inside.reshape(nx,ny).astype(np.float32) / (samples*samples)
    return np.clip(coverage,0.,1.)


def epsilon_map_contour(cfg, points, *, dx_nm=20., samples=5,
                        bottom_um=1.20,top_um=1.20,pml_um=.30,
                        with_contour=True):
    """Yee-grid scalar epsilon, subpixel-filled in x/y and z.

    This is a genuine independent FDTD raster, but scalar averaging does not
    implement tensor subpixel corrections at dielectric interfaces.
    """
    dx = dx_nm/1000.
    nx = int(round(cfg.period/dx))
    if abs(nx*dx-cfg.period)>1e-9:
        raise ValueError('Grid spacing must divide the cell period')
    zmin=-bottom_um-pml_um
    zmax=cfg.height+top_um+pml_um
    nz=int(np.ceil((zmax-zmin)/dx))
    z=zmin+(np.arange(nz)+.5)*dx
    cover=(polygon_coverage(points,cfg.period,nx,samples=samples)
           if with_contour else np.zeros((nx,nx),np.float32))
    sub=np.clip((0-(z-dx/2))/dx,0,1)
    fin=np.clip((np.minimum(z+dx/2,cfg.height)-np.maximum(z-dx/2,0))/dx,0,1)
    eps=1+(cfg.n_sub**2-1)*sub[None,None,:]+(cfg.n_fin**2-1)*cover[:,:,None]*fin[None,None,:]
    spec=dict(x=(np.arange(nx)+.5-nx/2)*dx,y=(np.arange(nx)+.5-nx/2)*dx,
              z=z,dx=dx,nx=nx,ny=nx,nz=nz,fill=cover,
              npml=int(round(pml_um/dx)))
    return eps.astype(np.float32),spec


def jones_metrics(J):
    J=np.asarray(J,dtype=complex)
    norm=max(float(np.linalg.norm(J,'fro')**2),1e-15)
    off=J-.5*np.trace(J)*np.eye(2)
    comm=J.conj().T@J-J@J.conj().T
    disc=(J[0,0]-J[1,1])**2+4*J[0,1]*J[1,0]
    values,vecs=np.linalg.eig(J)
    overlap=float(abs(np.vdot(vecs[:,0],vecs[:,1]))/
                  max(np.linalg.norm(vecs[:,0])*np.linalg.norm(vecs[:,1]),1e-15))
    s=np.linalg.svd(off,compute_uv=False)
    return dict(eigenvalues=values,gap=float(abs(values[0]-values[1])/np.sqrt(norm)),
                discriminant=disc,discriminant_norm=float(abs(disc)/norm),
                overlap=overlap,anisotropy=float(np.linalg.norm(off,'fro')/np.sqrt(norm)),
                nonnormality=float(np.linalg.norm(comm,'fro')/norm),
                jordan_strength=float(np.linalg.norm(off,'fro')/np.sqrt(norm)),
                rank_ratio=float(s[-1]/max(s[0],1e-15)),
                mean_power=norm/2)


def four_blocks(S):
    S=np.asarray(S,complex)
    return dict(R11=S[:2,:2],T21=S[2:,:2],T12=S[:2,2:],R22=S[2:,2:])


def power_budget(S):
    S=np.asarray(S,complex)
    reflection=np.array([np.sum(abs(S[:2,j])**2) if j<2 else
                         np.sum(abs(S[2:,j])**2) for j in range(4)])
    transmission=np.array([np.sum(abs(S[2:,j])**2) if j<2 else
                           np.sum(abs(S[:2,j])**2) for j in range(4)])
    return reflection,transmission


def channel_phase(z, amplitude_floor=1e-10):
    return float(np.angle(z,deg=True)) if abs(z)>amplitude_floor else float('nan')


def phase_report(J,label):
    J=np.asarray(J,complex)
    det=np.linalg.det(J); tr=np.trace(J)
    n2=np.linalg.norm(J,'fro')**2
    pdet=.5*channel_phase(det,1e-8*n2)
    ptr=channel_phase(tr,1e-8*np.sqrt(n2))
    print(f'{label}: det half-phase={pdet:+.2f}° (mod 180°); trace phase={ptr:+.2f}°; '
          f'singular values={np.round(np.linalg.svd(J,compute_uv=False),5)}')
    # R/L convention here agrees with engine_transmission_ep circular_unitary.
    vR=np.array([1,1j])/np.sqrt(2)
    for name,v in [('x',[1,0]),('y',[0,1]),('+45°',[1,1]),('R',vR)]:
        v=np.asarray(v,complex);v/=np.linalg.norm(v)
        a=np.vdot(v,J@v)
        print(f'  {name:>5}: |v†Jv|={abs(a):.5f}, phase={channel_phase(a):+.2f}°')


def runtime_checks(S):
    from engine_validate import metrics
    m=metrics(S)
    m['T21']=jones_metrics(S[2:,:2])
    m['T12']=jones_metrics(S[:2,2:])
    return m


def bare_rcwa(cfg, settings, ports):
    """Independent TORCWA bare silica/air interface, with an air spacer.

    The top reference is at z=h before propagating to the requested plane;
    this construction tests the same port phases as the patterned solver.
    """
    import math
    import torch
    import torcwa
    device=torch.device(settings.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    dtype=torch.complex128 if settings.precision=='complex128' else torch.complex64
    sim=torcwa.rcwa(freq=1/cfg.wavelength,order=[settings.order,settings.order],
                    L=[cfg.period,cfg.period],dtype=dtype,device=device)
    sim.add_input_layer(eps=cfg.n_sub**2)
    sim.add_output_layer(eps=1.)
    sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
    sim.add_layer(thickness=cfg.height,
                  eps=torch.ones((settings.raster,settings.raster),device=device,dtype=dtype))
    with torch.no_grad():
        sim.solve_global_smatrix()
        S=np.zeros((4,4),complex)
        for incoming in range(2):
            direction='forward' if incoming==0 else 'backward'
            for outgoing in range(2):
                port='reflection' if incoming==outgoing else 'transmission'
                for i,pi in enumerate('xy'):
                    for j,pj in enumerate('xy'):
                        val=sim.S_parameters(orders=[0,0],direction=direction,port=port,
                             polarization=pi+pj,ref_order=[0,0],power_norm=True)
                        S[2*outgoing+i,2*incoming+j]=complex(val.reshape(-1)[0].detach().cpu())
    from engine_validate import at_monitor_planes
    return at_monitor_planes(S,cfg,*ports)
