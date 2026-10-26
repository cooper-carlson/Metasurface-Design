"""Independent GPU RCWA backend with TORCWA, for a shared two-fin polygon.

TORCWA uses interface planes at z=0,h. Optional conversion places S at the
FDTD monitor planes; no alignment by fitting to FDTD outputs is performed.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from engine_geometry import coverage
from engine_validate import at_monitor_planes


@dataclass(frozen=True)
class Settings:
    order: int = 4  # (2*order+1)^2 Fourier harmonics; cost grows steeply.
    raster: int = 160
    samples: int = 3
    device: str = 'cuda'
    precision: str = 'complex64'


def solve(cfg, settings=Settings(), ports=None, bare=False):
    """Return a power-normalized 4x4 zero-order scattering matrix."""
    try:
        import torch
        import torcwa
    except ImportError as exc:
        raise ImportError('Install CUDA PyTorch and torcwa: pip install torcwa') from exc
    if settings.raster < 4*settings.order+2:
        raise ValueError('RCWA raster too small for chosen Fourier order')
    device = torch.device(settings.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use RCWASettings(device="cpu").')
    dtype = torch.complex64 if settings.precision == 'complex64' else torch.complex128
    t0=perf_counter()
    fill = coverage(cfg,settings.raster,settings.raster,samples=settings.samples)
    if bare: fill[:] = 0
    material = 1 + (cfg.n_fin**2-1)*fill
    with torch.no_grad():
        eps = torch.as_tensor(material,device=device,dtype=dtype)
        sim = torcwa.rcwa(freq=1/cfg.wavelength,order=[settings.order,settings.order],
                  L=[cfg.period,cfg.period],dtype=dtype,device=device)
        sim.add_input_layer(eps=cfg.n_sub**2)
        sim.add_output_layer(eps=1.)
        sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
        sim.add_layer(thickness=cfg.height,eps=eps)
        sim.solve_global_smatrix()
        S = np.zeros((4,4),dtype=np.complex128)
        for incoming_side in range(2):
            direction = 'forward' if incoming_side==0 else 'backward'
            for outgoing_side in range(2):
                port = 'reflection' if incoming_side==outgoing_side else 'transmission'
                for i, pol_out in enumerate('xy'):
                    for j, pol_in in enumerate('xy'):
                        z = sim.S_parameters(orders=[0,0],direction=direction,port=port,
                                  polarization=pol_out+pol_in,ref_order=[0,0],power_norm=True)
                        S[2*outgoing_side+i,2*incoming_side+j] = complex(z.reshape(-1)[0].cpu().item())
    if device.type=='cuda': torch.cuda.synchronize()
    elapsed=perf_counter()-t0
    if ports is not None:
        S=at_monitor_planes(S,cfg,*ports)
    return dict(S=S,elapsed_s=elapsed,order=settings.order,raster=settings.raster)
