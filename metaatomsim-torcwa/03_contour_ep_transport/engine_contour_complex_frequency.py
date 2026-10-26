"""Complex-frequency continuation of periodic single-contour RCWA transmission.

Uses the existing contour geometry, TORCWA physics, and saved candidate rules.
Real-axis power normalization is continued by a constant, not re-evaluated as
real energy flux at non-real frequencies. Time convention exp(-i omega t).
Units: frequency cycles/um (c=1 in TORCWA), length um.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import torch

C_THZ_UM = 299.792458
CIRCULAR_U = np.array([[1, 1], [1j, -1j]], complex)/np.sqrt(2)


def prepare_epsilon(cfg, settings, params, rules, geom, *, hard=True):
    """Rasterize optimized contour once; re-use the same mask for every frequency."""
    if float(cfg.n_fin_kappa) != 0. or not isinstance(cfg.n_fin, (int, float)):
        raise ValueError('Complex sweep requires real, nondispersive material parameters')
    if abs(cfg.period-rules.period)>1e-12:
        raise ValueError('Saved rules period and model period disagree')
    if settings.precision != 'complex128':
        raise ValueError('Complex-plane pole/zero scan requires complex128')
    with torch.no_grad():
        fill=geom.contour_fill(params,rules,settings.raster,hard=hard)
        eps=(1+(float(cfg.n_fin)**2-1)*fill).to(torch.complex128).detach()
    return eps


def complex_transmission(cfg,settings,eps,height,freq,*,ports=None):
    """Calculate analytically normalized T21 at a complex spatial frequency.

    Applies the same normal-incidence real-axis power normalization as the
    optimization engine, extended as a CONSTANT sqrt(n_out/n_in). Using
    power_norm=True off the real axis would introduce nonanalytic real/abs.
    """
    if settings.precision!='complex128' or eps.dtype!=torch.complex128:
        raise ValueError('complex128 required')
    if eps.device.type != torch.device(settings.device).type:
        raise ValueError('Contour epsilon and RCWA device differ')
    if not np.isfinite(freq.real) or not np.isfinite(freq.imag) or freq.real<=0:
        raise ValueError('Complex frequency must be finite with Re(f)>0')
    import torcwa
    with torch.no_grad():
        sim=torcwa.rcwa(freq=complex(freq),order=[settings.order]*2,
                        L=[cfg.period]*2,dtype=torch.complex128,device=eps.device)
        sim.add_input_layer(eps=float(cfg.n_sub)**2)
        sim.add_output_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
        sim.add_layer(thickness=float(height),eps=eps)
        sim.solve_global_smatrix()
        # T21 is forward transmission. Fixed lab-xy basis at both ports.
        J=np.empty((2,2),complex)
        for i, out in enumerate('xy'):
            for j, inc in enumerate('xy'):
                value=sim.S_parameters(orders=[0,0],direction='forward',
                     port='transmission',polarization=out+inc,
                     ref_order=[0,0],power_norm=False)
                J[i,j]=complex(value.reshape(-1)[0].detach().cpu())
    J*=np.sqrt(1.0/float(cfg.n_sub))
    if ports is not None:
        zb,zt=map(float,ports)
        if zb>0 or zt<height:
            raise ValueError('Monitor planes must be outside the device slab')
        # Analytic homogeneous propagation; no conjugation of complex freq.
        D_bot=np.exp(2j*np.pi*complex(freq)*cfg.n_sub*(-zb))
        D_top=np.exp(2j*np.pi*complex(freq)*(zt-height))
        J*=D_bot*D_top
    if not np.isfinite(J).all():
        raise FloatingPointError('Nonfinite TORCWA amplitude near a singularity')
    return J


def frequency_grid(wavelength_um, *, real_span=(.96,1.04),
                   imag_span=(-.06,.015),real_points=101,imag_points=101):
    """Normalized frequency plane, with f0=1/lambda; Im(f)<0 is lower half."""
    if wavelength_um<=0 or real_points<3 or imag_points<3:
        raise ValueError('Invalid grid')
    if not real_span[0]>0 or not real_span[1]>real_span[0]:
        raise ValueError('Invalid real frequency range')
    if not imag_span[1]>imag_span[0]:
        raise ValueError('Invalid imaginary frequency range')
    f0=1.0/wavelength_um
    re=np.linspace(*real_span,real_points)*f0
    im=np.linspace(*imag_span,imag_points)*f0
    return re,im


def cache_identity(cfg,rules,raw,settings,re,im,*,ports=None):
    metadata=dict(model=asdict(cfg),rules=asdict(rules),
            raw_digest=hashlib.sha256(np.asarray(raw,dtype=np.float64).tobytes()).hexdigest(),
            rcwa=asdict(settings),real_freq_um_inv=np.asarray(re).tolist(),
            imag_freq_um_inv=np.asarray(im).tolist(),ports=ports,
            solver_version='complex-contour-v2-analytic-normalization')
    return json.loads(json.dumps(metadata,sort_keys=True)), hashlib.sha256(
         json.dumps(metadata,sort_keys=True).encode()).hexdigest()[:16]


def sweep_grid(cfg,settings,eps,height,re,im,*,cache_path=None,metadata=None,
               solver=complex_transmission,progress_every=1):
    """Direct complex RCWA at every grid point; resume partial rows on restart."""
    re=np.asarray(re,float); im=np.asarray(im,float)
    cache_path=Path(cache_path) if cache_path is not None else None
    T=np.full((len(im),len(re),2,2),np.nan+1j*np.nan,complex)
    done=np.zeros((len(im),len(re)),bool)
    info=json.dumps(metadata or {},sort_keys=True)
    if cache_path and cache_path.is_file():
        with np.load(cache_path,allow_pickle=False) as o:
            if str(o['metadata'])!=info or not np.array_equal(o['re'],re) or not np.array_equal(o['im'],im):
                raise ValueError('Complex-sweep cache parameters changed: use a different path')
            T=o['T'].copy(); done=o['done'].copy()
        print(f'Resuming complex scan: {done.sum()}/{done.size} evaluated.')

    def save():
        if cache_path is None:return
        cache_path.parent.mkdir(parents=True,exist_ok=True)
        temp=cache_path.with_name(cache_path.name+'.partial.npz')
        np.savez_compressed(temp,T=T,done=done,re=re,im=im,metadata=info)
        os.replace(temp,cache_path)

    t0=time.monotonic()
    failed=0
    try:
        for iy,y in enumerate(im):
            for ix,x in enumerate(re):
                if done[iy,ix]:continue
                try:
                    curr=np.asarray(solver(cfg,settings,eps,height,complex(x,y)),complex)
                    if curr.shape!=(2,2) or not np.isfinite(curr).all():
                        raise FloatingPointError('invalid Jones matrix')
                    T[iy,ix]=curr
                except (RuntimeError,ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                    failed+=1
                    if failed<=5:print(f'RCWA failure at ({x:.5g},{y:.5g}): {exc}')
                done[iy,ix]=True
            save()
            if progress_every and (iy+1)%progress_every==0:
                print(f'Imag row {iy+1}/{len(im)}; {done.sum()}/{done.size} evaluated; '
                      f'{np.isfinite(T[:,:,0,0]).sum()} finite; {time.monotonic()-t0:.1f} s')
    except (KeyboardInterrupt,Exception):
        save()
        raise
    save()
    return T


def projected_channel(T,channel='RR'):
    """Fixed circular polarization amplitude, or det(T21); never eigenbranches."""
    T=np.asarray(T,complex)
    if T.shape[-2:]!=(2,2):raise ValueError('Expect (...,2,2) transmission Jones matrices')
    if channel=='det':return np.linalg.det(T)
    if len(channel)!=2 or any(c not in 'RLxy' for c in channel):
        raise ValueError('Select e.g. RR, LR, RL, LL, xx or det')
    pol=dict(R=CIRCULAR_U[:,0],L=CIRCULAR_U[:,1],
             x=np.array([1.,0.]),y=np.array([0.,1.]))
    vout,vin=pol[channel[0]],pol[channel[1]]
    return np.einsum('i,...ij,j->...',vout.conj(),T,vin)


def rational22(re,im,values,reference_freq):
    """Exploratory Padé-type fit a0+a1 z+a2 z² / (1+b1 z+b2 z²).

    Candidate pole/zero positions are not certified quasinormal modes. Check
    convergence in RCWA order, fit window, and direct complex solves.
    """
    f=np.asarray(re)[None,:]+1j*np.asarray(im)[:,None]
    val=np.asarray(values,complex)
    good=np.isfinite(val)&(np.abs(val)<1e9)
    if good.sum()<20:raise ValueError('Not enough finite complex samples to fit')
    scale=np.max(np.abs(f-reference_freq))
    if scale<=0:raise ValueError('No complex span')
    z=((f-reference_freq)/scale)[good]; t=val[good]
    matrix=np.column_stack([np.ones_like(z),z,z*z,-t*z,-t*z*z])
    coef,_,rank,sing=np.linalg.lstsq(matrix,t,rcond=1e-11)
    if rank<5:raise ValueError('Ill-conditioned rational 2/2 fit')
    a=coef[:3];b=np.array([1.,coef[3],coef[4]])
    zp=np.roots(a[::-1]); pp=np.roots(b[::-1])
    reconstructed=(a[0]+a[1]*z+a[2]*z*z)/(b[0]+b[1]*z+b[2]*z*z)
    nrm=np.linalg.norm(t)
    relerr=float(np.linalg.norm(reconstructed-t)/max(nrm,1e-14))
    return dict(zeros=reference_freq+scale*zp,poles=reference_freq+scale*pp,
                fit_error=relerr,coeff_numerator=a,coeff_denominator=b,
                reference_freq=complex(reference_freq),scale=float(scale))


def plot_complex_maps(re,im,T,*,channel='RR',rational=None,amplitude_scale='log10'):
    """Two complex-frequency heatmaps: phase and |transmission coefficient|."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    values=projected_channel(T,channel)
    magnitude=np.abs(values)
    phase=np.angle(values)
    phase=np.ma.masked_where(~np.isfinite(magnitude)|(magnitude<1e-12),phase)
    if amplitude_scale=='log10':
        amp=np.log10(np.maximum(magnitude,1e-12));name=r'$\log_{10}|t|$'
    elif amplitude_scale=='linear':
        amp=magnitude;name=r'$|t|$'
    else:raise ValueError('amplitude_scale must be log10 or linear')
    amp=np.ma.masked_invalid(amp)
    extent=(re[0]*C_THZ_UM,re[-1]*C_THZ_UM,im[0]*C_THZ_UM,im[-1]*C_THZ_UM)
    fig,axs=plt.subplots(1,2,figsize=(13.4,5.2),sharex=True,sharey=True,constrained_layout=True)
    m0=axs[0].imshow(phase,origin='lower',extent=extent,aspect='auto',
        interpolation='nearest',cmap='twilight',vmin=-np.pi,vmax=np.pi)
    m1=axs[1].imshow(amp,origin='lower',extent=extent,aspect='auto',
        interpolation='nearest',cmap='viridis')
    for ax in axs:
        ax.axhline(0,color='white',lw=1.0,ls='--',alpha=.75)
        ax.set_xlabel(r'$\mathrm{Re}(f)$ (THz)')
        ax.set_ylabel(r'$\mathrm{Im}(f)$ (THz)')
    axs[0].set_title(f'Transmission phase: arg(t{channel}) (radians)')
    axs[1].set_title(f'Transmission amplitude: {name}')
    fig.colorbar(m0,ax=axs[0],label='phase (rad)',shrink=.82)
    fig.colorbar(m1,ax=axs[1],label=name,shrink=.82)
    if rational is not None and rational['fit_error']<0.10:
        for ax in axs:
            for label,pts,color,marker in (('2/2 fit zeros',rational['zeros'],'white','o'),
                                            ('2/2 fit poles',rational['poles'],'red','x')):
                coords=np.asarray(pts)*C_THZ_UM
                in_plot=(coords.real>=extent[0])&(coords.real<=extent[1])&(
                    coords.imag>=extent[2])&(coords.imag<=extent[3])
                ax.scatter(coords.real[in_plot],coords.imag[in_plot],s=85,
                         c=color,marker=marker,edgecolors='black' if marker=='o' else None,
                         label=label,zorder=5)
            if len(ax.collections):ax.legend(loc='upper right',fontsize=8)
    return fig,axs
