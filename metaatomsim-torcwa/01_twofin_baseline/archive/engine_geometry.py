"""One polygon geometry shared by FDTD, RCWA, and GDS.

All lengths in micrometers. f_xy is a *numerical* subcell area fraction,
not a claim of exact tensor subpixel smoothing.
"""
import numpy as np
from twofin import fins


def inside_convex(x, y, poly):
    """Vectorized inside test for a convex polygon (x and y broadcast)."""
    poly = np.asarray(poly)
    q = np.roll(poly, -1, axis=0)
    cross = np.stack([(q[i, 0] - poly[i, 0]) * (y - poly[i, 1])
                      - (q[i, 1] - poly[i, 1]) * (x - poly[i, 0])
                      for i in range(len(poly))], axis=0)
    return np.all(cross >= -1e-12, axis=0) | np.all(cross <= 1e-12, axis=0)


def coverage(config, nx, ny, samples=5):
    """Area fraction of the UNION of fins, sampled inside each square pixel."""
    if samples < 1:
        raise ValueError('samples must be positive')
    period = config.period
    x = (np.arange(nx) + .5 - nx / 2) * period / nx
    y = (np.arange(ny) + .5 - ny / 2) * period / ny
    X, Y = np.meshgrid(x, y, indexing='ij')
    fill = np.zeros((nx, ny), dtype=np.float32)
    polys = fins(config)
    for a in range(samples):
        for b in range(samples):
            xp = X + ((a + .5) / samples - .5) * period / nx
            yp = Y + ((b + .5) / samples - .5) * period / ny
            occupied = inside_convex(xp, yp, polys[0]) | inside_convex(xp, yp, polys[1])
            fill += occupied.astype(np.float32) / samples**2
    return np.clip(fill,0.,1.)


def epsilon_map(config, dx_nm=10., samples=5, bottom_um=1.25, top_um=1.25, pml_um=.30):
    """3D scalar epsilon map. Fin/substrate interfaces use volume fractions.

    For true dielectric interface accuracy, a stagger-aware anisotropic tensor
    implementation is still needed; this map is an intermediate improvement.
    """
    dx = dx_nm / 1000
    nx = int(round(config.period / dx))
    if abs(nx * dx - config.period) > 1e-8:
        raise ValueError('dx_nm must evenly divide the unit-cell period')
    zmin = -bottom_um - pml_um
    zmax = config.height + top_um + pml_um
    nz = int(np.ceil((zmax-zmin) / dx))
    z = zmin + (np.arange(nz) + .5) * dx
    xy = coverage(config, nx, nx, samples=samples)
    # Fraction of each z cell below the silica plane.
    f_sub = np.clip((0. - (z - dx/2))/dx, 0, 1)
    # Fraction of each z cell inside the fin slab (0 <= z <= height).
    f_fin_z = np.clip((np.minimum(z + dx/2, config.height) - np.maximum(z - dx/2, 0))/dx, 0, 1)
    eps = 1 + (config.n_sub**2-1)*f_sub[None,None,:] \
            + (config.n_fin**2-1)*xy[:,:,None]*f_fin_z[None,None,:]
    return eps.astype(np.float32), dict(x=(np.arange(nx)+.5-nx/2)*dx,
             y=(np.arange(nx)+.5-nx/2)*dx, z=z, dx=dx, nx=nx, ny=nx, nz=nz,
             fill=xy, npml=int(round(pml_um/dx)))


def plot_geometry(config, samples=7):
    import matplotlib.pyplot as plt
    area = coverage(config, 200, 200, samples=samples)
    fig, ax = plt.subplots(figsize=(5.6, 4.7))
    h = config.period / 2
    im = ax.imshow(area.T, origin='lower', extent=(-h,h,-h,h), cmap='viridis', vmin=0, vmax=1)
    ax.set(xlabel='x (µm)', ylabel='y (µm)', title='Union of two Al₂O₃ fins — fractional coverage', aspect='equal')
    fig.colorbar(im, ax=ax, label='Al₂O₃ fill fraction')
    fig.tight_layout()
    return fig
