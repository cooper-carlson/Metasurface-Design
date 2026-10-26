"""Differentiable fixed-height rotating contour stacks and Jones invariants.

Full-stack S includes interfaces and multiple reflections. Embedded slices are
separate *diagnostics*, not Maxwell propagation generators. Lengths in um.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TransportSetup:
    layers: int = 6
    background_index: float = 1.0
    bottom_index: float = 1.46
    top_index: float = 1.0
    rotation_sign: int = 1

    def validate(self):
        if self.layers < 1 or int(self.layers) != self.layers:
            raise ValueError('Use a positive integer number of layers')
        if min(self.background_index, self.bottom_index, self.top_index) <= 0:
            raise ValueError('Refractive indices must be positive')
        if self.rotation_sign not in (-1, 0, 1):
            raise ValueError('Rotation sign must be -1, 0 or +1')
        return self


def normalized_jones_metrics(T, *, eps=1e-12):
    """Autograd-compatible overlap, gap, anisotropy, and nonnormality.

    Overlap has no unique meaning at a scalar matrix; the epsilon prevents a
    NaN but the separate anisotropy/gap constraints must exclude that limit.
    """
    import torch
    if tuple(T.shape) != (2, 2) or not torch.is_complex(T):
        raise ValueError('Expected complex 2x2 Jones matrix')
    eye = torch.eye(2, dtype=T.dtype, device=T.device)
    energy = T.abs().square().sum()
    Q = T - 0.5 * torch.trace(T) * eye
    q2 = Q.abs().square().sum()
    delta = (T[0, 0] - T[1, 1])**2 + 4 * T[0, 1] * T[1, 0]
    abs_delta = delta.abs()
    eta2 = ((q2 - 0.5 * abs_delta).clamp_min(0.0)
            / (q2 + 0.5 * abs_delta + eps))
    gap = torch.sqrt(abs_delta + eps) / torch.sqrt(energy + eps)
    anisotropy = torch.sqrt(q2 + eps) / torch.sqrt(energy + eps)
    comm = T.conj().T @ T - T @ T.conj().T
    nonnormality = torch.linalg.matrix_norm(comm, ord='fro') / (energy + eps)
    return dict(eta2=eta2, eta=torch.sqrt(eta2 + eps), gap=gap,
                anisotropy=anisotropy, mean_power=energy / 2,
                nonnormality=nonnormality, delta=delta)


def stack_fabrication_cost(p, rules, *, layers=6, perturb=None):
    """Fabrication heuristics at the *actual* per-slice height H/N.

    Height variation is taken to be a change in TOTAL thickness (not per-slice).
    No artificial extra interface or spacer is introduced.
    """
    from engine_ellipse_contour import fabrication_cost
    setup = TransportSetup(layers=layers).validate()
    pert = {} if perturb is None else dict(perturb)
    height_bias = float(pert.pop('height_um', 0.0))
    layer = dict(p, height=(p['height'] + height_bias) / setup.layers)
    return fabrication_cost(layer, rules, perturb=pert or None)


def solve_rotating_stack_torch(cfg, settings, p, rules, *, setup=TransportSetup(),
                               edge_um=None, perturb=None, hard=False,
                               height_um=None):
    """One differentiable TORCWA solve for all N adjacent patterned slices.

    Output: complex 4x4 zero-order S, in (bottom x,y, top x,y) basis,
    referenced at the physical faces. Lateral BC: periodic with one motif.
    The material embedding can be identical on both sides via setup.
    """
    import torch
    import torcwa
    from engine_ellipse_contour import contour_fill
    setup.validate()
    if float(cfg.n_fin_kappa) != 0 or not isinstance(cfg.n_fin, (int, float)):
        raise ValueError('Stack solver requires a real, lossless refractive index')
    if abs(float(cfg.period) - float(rules.period)) > 1e-12:
        raise ValueError('RCWA period and contour geometry period differ')
    if settings.raster < 4 * settings.order + 2:
        raise ValueError('Too few raster samples for Fourier order')
    dtype = torch.complex128 if settings.precision == 'complex128' else torch.complex64
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    if p['radius'].dtype != real_dtype:
        raise ValueError('Geometry tensor precision does not match RCWA settings')
    device = p['radius'].device
    if device.type != torch.device(settings.device).type:
        raise ValueError('Geometry tensor and RCWA device disagree')
    H = p['height'] if height_um is None else torch.as_tensor(
        height_um, dtype=real_dtype, device=device)
    if perturb:
        H = H + float(perturb.get('height_um', 0.0))
    if float(H.detach().cpu()) <= 0:
        raise ValueError('Physical total stack height must be positive')
    sim = torcwa.rcwa(freq=1 / float(cfg.wavelength),
                      order=[settings.order, settings.order],
                      L=[cfg.period, cfg.period], dtype=dtype, device=device)
    sim.add_input_layer(eps=setup.bottom_index ** 2)
    sim.add_output_layer(eps=setup.top_index ** 2)
    sim.set_incident_angle(inc_ang=0., azi_ang=0.)
    theta = (setup.rotation_sign * 2 * np.pi / setup.layers)
    for j in range(setup.layers):
        angle_err = 0.0
        shift = (0.0, 0.0)
        if perturb:
            if 'layer_orientation_delta_rad' in perturb:
                errors=np.asarray(perturb['layer_orientation_delta_rad'])
                if errors.shape!=(setup.layers,):
                    raise ValueError('Layer orientation variation count must match N')
                angle_err=float(errors[j])
            if 'layer_shift_xy_um' in perturb:
                offsets=np.asarray(perturb['layer_shift_xy_um'])
                if offsets.shape!=(setup.layers,2):
                    raise ValueError('Layer registration offsets must be shape (N,2)')
                shift=tuple(map(float,offsets[j]))
        rotated = dict(p, orientation=p['orientation'] + j*theta + angle_err,
                       cx=p['cx']+shift[0],cy=p['cy']+shift[1])
        fill = contour_fill(rotated, rules, settings.raster, edge_um=edge_um,
                            perturb=perturb, hard=hard)
        eps = (setup.background_index**2 +
               (float(cfg.n_fin)**2 - setup.background_index**2) * fill).to(dtype)
        sim.add_layer(thickness=H / setup.layers, eps=eps)
    sim.solve_global_smatrix()
    rows = []
    for out_side in (0, 1):
        for op in 'xy':
            row = []
            for in_side in (0, 1):
                for ip in 'xy':
                    value = sim.S_parameters(
                        orders=[0, 0], ref_order=[0, 0],
                        direction='forward' if in_side == 0 else 'backward',
                        port='reflection' if in_side == out_side else 'transmission',
                        polarization=op + ip, power_norm=True)
                    row.append(value.reshape(-1)[0])
            rows.append(torch.stack(row))
    return torch.stack(rows)


def solve_embedded_slice_torch(cfg, settings, p, rules, *, N=6,
                               background_index=1., orientation_offset=0.,
                               edge_um=None, perturb=None, hard=False,
                               height_um=None):
    """A matched-medium isolated slice: DIAGNOSTIC, not a local K(z)."""
    from dataclasses import replace
    # N=1 with a thickness H/N, same ambient on both sides.
    base_height = p['height'] if height_um is None else height_um
    q = dict(p, height=base_height / N,
             orientation=p['orientation'] + orientation_offset)
    pert = {} if perturb is None else dict(perturb)
    if 'height_um' in pert:
        pert['height_um'] /= N
    return solve_rotating_stack_torch(
        cfg, settings, q, rules,
        setup=TransportSetup(layers=2, background_index=background_index,
                             bottom_index=background_index,
                             top_index=background_index, rotation_sign=0),
        edge_um=edge_um, perturb=pert or None, hard=hard,
        height_um=base_height / N)
