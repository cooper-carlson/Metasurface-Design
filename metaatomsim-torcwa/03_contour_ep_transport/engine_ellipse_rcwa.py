"""Differentiable TORCWA S matrix for a single real-index Fourier contour.

The 4x4 basis is (bottom x,y, top x,y), with fixed laboratory coordinates
and power normalization. x/y are periodic, z ports are outgoing radiation.
"""
import math
import torch
from engine_ellipse_contour import contour_fill

# Identical fixed monitor planes to the established two-fin inverse engine.
FIXED_PORTS = (-0.52, 1.05)


def _device_matches(actual, requested):
    """Treat unindexed 'cuda' as the device selected by tensor placement."""
    requested = torch.device(requested)
    actual = torch.device(actual)
    return (actual.type == requested.type and
            (requested.index is None or actual.index == requested.index))


def solve_contour_torch(cfg, settings, p, rules, *, edge_um=None,
                        perturb=None, hard=False, ports=FIXED_PORTS,
                        return_fill=False):
    if float(cfg.n_fin_kappa) != 0.0 or not isinstance(cfg.n_fin,(int,float)):
        raise ValueError('Lossless contour solver requires strictly real n_fin')
    if abs(cfg.period-rules.period)>1e-12:
        raise ValueError('Contour rules period must match physical Config period')
    if settings.raster < 4*settings.order+2:
        raise ValueError('RCWA raster undersamples the requested Fourier order')
    import torcwa
    device=torch.device(settings.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device unavailable')
    cdtype=torch.complex128 if settings.precision=='complex128' else torch.complex64
    rdtype=torch.float64 if cdtype==torch.complex128 else torch.float32
    if not _device_matches(p['radius'].device, device) or p['radius'].dtype != rdtype:
        raise ValueError(
            f'Contour geometry: {p["radius"].device}/{p["radius"].dtype}; '
            f'solver expects {device}/{rdtype}')
    # Use the concrete tensor device (e.g., cuda:0) for TORCWA.
    device = p['radius'].device
    if any(t.device != device or t.dtype != rdtype for t in p.values()):
        raise ValueError('All contour tensors must share solver device/precision')
    height=p['height']
    if perturb is not None:
        height=height+float(perturb.get('height_um',0.))
    if float(height.detach().cpu())>=ports[1] or float(height.detach().cpu())<=0:
        raise ValueError('Height outside available monitor planes')
    fill=contour_fill(p,rules,settings.raster,edge_um=edge_um,perturb=perturb,hard=hard)
    # Im(epsilon) = 0 exactly: material absorption is forbidden here.
    eps=(1+(cfg.n_fin**2-1)*fill).to(cdtype)
    sim=torcwa.rcwa(freq=1/cfg.wavelength,order=[settings.order,settings.order],
                    L=[cfg.period,cfg.period],dtype=cdtype,device=device)
    sim.add_input_layer(eps=cfg.n_sub**2)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.,azi_ang=0.)
    sim.add_layer(thickness=height,eps=eps)
    sim.solve_global_smatrix()
    rows=[]
    for outgoing_side in (0,1):
        for out_pol in 'xy':
            row=[]
            for incoming_side in (0,1):
                direction='forward' if incoming_side==0 else 'backward'
                port='reflection' if incoming_side==outgoing_side else 'transmission'
                for in_pol in 'xy':
                    value=sim.S_parameters(orders=[0,0],direction=direction,port=port,
                            polarization=out_pol+in_pol,ref_order=[0,0],power_norm=True)
                    row.append(value.reshape(-1)[0])
            rows.append(torch.stack(row))
    S=torch.stack(rows)
    if ports is not None:
        bottom_z,top_z=ports
        # Reference-plane propagation is retained, including common Jones phase.
        pb=torch.exp(torch.as_tensor(2j*math.pi*cfg.n_sub*(-bottom_z)/cfg.wavelength,
                                   device=device,dtype=cdtype))
        pt=torch.exp(2j*math.pi*(top_z-height)/cfg.wavelength)
        D=torch.diag(torch.stack((pb,pb,pt,pt)))
        S=D@S@D
    return (S,fill) if return_fill else S
