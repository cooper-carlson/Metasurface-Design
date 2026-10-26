"""Pinwheel meta-atom: four rotated bars around the cell centre, C4 -> C2 by anisotropy.

Design idea (Gorkunov et al., Adv. Opt. Mater. 2024, arXiv:2408.10742): a z-invariant
C4 planar-chiral layer on a substrate gets large circular dichroism when the substrate
couples near-degenerate odd/even modes and the material loss is near critical coupling.
In Pauli form that is a circular, rotation-invariant non-Hermitian term t_y. A small
anisotropy delta adds linear birefringence t_z ~ delta, giving the rotation-realized
model K' = sigma_z + i eta sigma_y with eta ~ |t_y/t_z|: shrinking delta approaches an EP.

Same API names as engine_twofin_eta (design_from_raw, vary, solve_torch, ...) so the
optimizer notebook can switch engines. Lengths in micrometres, angles in degrees.
"""
import math
import numpy as np
import torch
from dataclasses import replace

from engine_twofin_eta import (Physics, RCWA, Modulation, device_dtype, rcwa_from_mask,
                               jones_metrics, numpy_metrics, pauli_khat, theta_loop_phase,
                               transpose_asymmetry, virtual_wilson, diffraction_audit, _xy_grid,
                               _rect_coverage, _signed_rectangle)

PARAM_KEYS = ('bar_length', 'bar_width', 'radius', 'tilt_deg', 'aniso', 'height')
BOUNDS = {
    'bar_length': (0.08, 0.26), 'bar_width': (0.045, 0.12), 'radius': (0.03, 0.14),
    'tilt_deg': (-75., 75.), 'aniso': (-0.45, 0.45), 'height': (0.08, 0.60),
}


def design_from_raw(raw):
    return {key: lo + (hi-lo)*torch.sigmoid(raw[i]) for i, (key, (lo, hi)) in enumerate(BOUNDS.items())}


def raw_from_physical(values):
    out = []
    for key, (lo, hi) in BOUNDS.items():
        u = np.clip((values[key]-lo)/(hi-lo), 1e-4, 1-1e-4); out.append(float(np.log(u/(1-u))))
    return np.asarray(out, float)


def vary(p, modulation, u=0.):
    if modulation.parameter not in PARAM_KEYS: raise ValueError('Unsupported modulation parameter')
    d = dict(p); d[modulation.parameter] = p[modulation.parameter] + modulation.amplitude*u
    return d


def bars(p, theta_deg=0.):
    """Four (centre, angle, length, width) bars; bars 0, 2 lengthened and 1, 3 shortened by aniso."""
    anchor = p['bar_length']
    th = torch.as_tensor(theta_deg, device=anchor.device, dtype=anchor.dtype)*math.pi/180
    out = []
    for k in range(4):
        phi = th + k*math.pi/2
        centre = p['radius']*torch.stack((torch.cos(phi), torch.sin(phi)))
        angle = phi + math.pi/2 + p['tilt_deg']*math.pi/180      # tangential bar, tilted: planar chirality
        length = p['bar_length']*(1 + p['aniso']*(1 if k % 2 == 0 else -1))
        out.append((centre, angle, length, p['bar_width']))
    return out


def motif_mask(p, physics=Physics(), raster=128, theta_deg=0., grid=None):
    """Area-fraction fill of the four-bar union (probabilistic union of exact bar coverages)."""
    anchor = p['bar_length']
    X, Y = grid if grid is not None else _xy_grid(physics, raster, anchor.device, anchor.dtype)
    h = X[1, 0]-X[0, 0]
    empty = torch.ones_like(X)
    for c, a, l, w in bars(p, theta_deg): empty = empty*(1-_rect_coverage(X, Y, c, a, l, w, h))
    return 1-empty


def motif_normals(p, physics=Physics(), raster=128, theta_deg=0.):
    anchor = p['bar_length']
    X, Y = _xy_grid(physics, raster, anchor.device, anchor.dtype)
    sdf = torch.stack([_signed_rectangle(X, Y, c, a, l, w) for c, a, l, w in bars(p, theta_deg)]).min(0).values
    gx = torch.roll(sdf, -1, 0)-torch.roll(sdf, 1, 0); gy = torch.roll(sdf, -1, 1)-torch.roll(sdf, 1, 1)
    n = torch.sqrt(gx.square()+gy.square()+1e-30)
    return gx/n, gy/n


def physics_for(p, physics):
    """The global layer height is a design variable."""
    return replace(physics, height=p['height']) if 'height' in p else physics


def solve_torch(p, physics=Physics(), opts=RCWA(), theta_deg=0., hard=False):
    if hard or opts.mask != 'aa' or opts.factorization != 'nv':
        raise ValueError('engine_motif supports the area-fraction mask with NV factorization only')
    phys = physics_for(p, physics)
    fill = motif_mask(p, phys, opts.raster, theta_deg)
    return rcwa_from_mask(fill, phys, opts, normals=motif_normals(p, phys, opts.raster, theta_deg))[1]


def corners(p, theta_deg=0.):
    out = []
    for c, a, l, w in bars(p, theta_deg):
        ca, sa = torch.cos(a), torch.sin(a)
        local = torch.stack([torch.stack((-l/2, -w/2)), torch.stack((l/2, -w/2)),
                             torch.stack((l/2, w/2)), torch.stack((-l/2, w/2))])
        out.append(local@torch.stack([torch.stack((ca, -sa)), torch.stack((sa, ca))]).T+c)
    return out


def geometry_penalty(p, physics=Physics(), theta_deg=0.):
    """Keep the motif inside the cell (any rotation) and the shorter bars printable."""
    reach = torch.cat(corners(p, theta_deg)).norm(dim=1).max()        # radius of the motif
    margin = torch.relu(reach-(physics.period/2-physics.cell_margin))
    short = p['bar_length']*(1-p['aniso'].abs())
    neck = torch.relu(0.06-short)+torch.relu(0.045-p['bar_width'])
    return 300*margin.square()+200*neck.square(), reach


def check_connected(p):
    """Bars may be separate: report the number of connected pieces instead of a bool."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon(c.detach().cpu().numpy()) for c in corners({k: torch.as_tensor(float(v), dtype=torch.float64) for k, v in p.items()})]
    u = unary_union(polys)
    return u.area, (1 if u.geom_type == 'Polygon' else len(u.geoms))


def export_layout(p, physics, output_base):
    from pathlib import Path
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    output_base = Path(output_base); output_base.parent.mkdir(parents=True, exist_ok=True)
    q = {k: torch.as_tensor(float(v), dtype=torch.float64) for k, v in p.items()}
    merged = unary_union([Polygon(c.numpy()) for c in corners(q)])
    pieces = [merged] if merged.geom_type == 'Polygon' else list(merged.geoms)
    half = physics.period/2
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {physics.period*1000} {physics.period*1000}">',
           '<rect width="100%" height="100%" fill="white" stroke="black"/>']
    for g in pieces:
        pts = ' '.join(f'{(x+half)*1000:.3f},{(half-y)*1000:.3f}' for x, y in np.asarray(g.exterior.coords[:-1]))
        svg.append(f'<polygon points="{pts}" fill="#2563eb"/>')
    svgfile = output_base.with_suffix('.svg'); svgfile.write_text(''.join(svg)+'</svg>')
    try:
        import gdstk
    except ImportError:
        return dict(svg=str(svgfile), gds=None)
    lib = gdstk.Library(unit=1e-6, precision=1e-9); cell = lib.new_cell('PINWHEEL')
    for g in pieces: cell.add(gdstk.Polygon(np.asarray(g.exterior.coords[:-1]), layer=1))
    cell.add(gdstk.rectangle((-half, -half), (half, half), layer=10))
    gdsfile = output_base.with_suffix('.gds'); lib.write_gds(str(gdsfile))
    return dict(svg=str(svgfile), gds=str(gdsfile))
