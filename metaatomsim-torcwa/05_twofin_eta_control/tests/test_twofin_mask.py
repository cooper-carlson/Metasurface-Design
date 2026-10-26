"""CPU geometry tests for the area-fraction (anti-aliased) two-fin mask."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine_twofin_eta import (Physics, PARAM_KEYS, design_from_raw, twofin_mask,
                               numpy_rectangles)

PHYS = Physics(height=0.633)


def designs(n=6, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        raw = torch.tensor(rng.normal(scale=0.8, size=len(PARAM_KEYS)), dtype=torch.float64)
        yield design_from_raw(raw), float(rng.uniform(0, 180))


def union_area(p, theta):
    q = {k: float(v) for k, v in p.items()}
    return unary_union([Polygon(c) for c in numpy_rectangles(q, theta)]).area


@pytest.mark.parametrize('raster', [96, 192])
def test_area_matches_exact_polygon(raster):
    h2 = (PHYS.period / raster) ** 2
    for p, theta in designs():
        with torch.no_grad():
            area = float(twofin_mask(p, PHYS, raster, theta).sum()) * h2
        exact = union_area(p, theta)
        # Only rectangle corners and the junction crossing are approximate.
        assert abs(area - exact) / exact < 5e-4


def test_area_fraction_matches_supersampled_reference():
    for p, theta in designs(4):
        with torch.no_grad():
            aa = twofin_mask(p, PHYS, 128, theta, mode='aa')
            ss = twofin_mask(p, PHYS, 128, theta, mode='ss', supersample=64)
        edge = (ss > 0) & (ss < 1)
        assert float((aa - ss).abs().max()) < 0.08
        assert float((aa - ss)[edge].abs().mean()) < 0.004


def test_hard_and_soft_modes_still_available():
    p, theta = next(designs(1))
    with torch.no_grad():
        hard = twofin_mask(p, PHYS, 64, theta, hard=True)
        soft = twofin_mask(p, PHYS, 64, theta, mode='soft')
    assert set(torch.unique(hard).tolist()) <= {0.0, 1.0}
    assert ((soft > 0.01) & (soft < 0.99)).any()
    with pytest.raises(ValueError):
        twofin_mask(p, PHYS, 64, theta, mode='blurry')


def test_gradients_finite_for_every_parameter_including_axis_aligned():
    for theta in (0.0, 37.0, 90.0):
        raw = torch.zeros(len(PARAM_KEYS), dtype=torch.float64, requires_grad=True)
        p = design_from_raw(raw)
        p['bend_deg'] = p['bend_deg'] * 0 + 90.0 + raw[4] * 0  # axis-aligned edges
        fill = twofin_mask(p, PHYS, 128, theta)
        (fill * torch.linspace(0, 1, fill.numel(), dtype=fill.dtype).reshape(fill.shape)).sum().backward()
        g = raw.grad.numpy()
        assert np.isfinite(g).all()
        assert (np.abs(g[[0, 1, 2, 3, 5]]) > 0).all()


def test_exact_quarter_turn_covariance():
    for p, theta in designs(3):
        with torch.no_grad():
            m0 = twofin_mask(p, PHYS, 160, theta).numpy()
            m90 = twofin_mask(p, PHYS, 160, theta + 90).numpy()
        assert np.allclose(np.rot90(m0), m90, atol=1e-9)
