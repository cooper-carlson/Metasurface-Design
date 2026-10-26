"""The one-solve theta-loop phase formula against full discrete Wilson loops (CPU)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine_twofin_eta import (pauli_khat, theta_loop_phase, transpose_asymmetry,
                               virtual_wilson, Physics, fin_eps)

R = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def wilson_theta(T0, n=2000):
    g = virtual_wilson([R(t) @ T0 @ R(t).T for t in np.linspace(0, np.pi, n, endpoint=False)])['gamma']
    return np.sort_complex(np.mod(g.real + np.pi, 2 * np.pi) - np.pi + 1j * g.imag)


def wrapped(g):
    g = np.asarray(g)
    return np.sort_complex(np.mod(g.real + np.pi, 2 * np.pi) - np.pi + 1j * g.imag)


def test_formula_matches_wilson_loop_for_general_T():
    rng = np.random.default_rng(1)
    for _ in range(6):
        T0 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        assert np.allclose(wrapped(theta_loop_phase(T0)), wilson_theta(T0), atol=1e-2)


def test_symmetric_T_gives_real_quantized_phase_at_any_overlap():
    rng = np.random.default_rng(2)
    T0 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    T0 = (T0 + T0.T) / 2
    assert transpose_asymmetry(T0) < 1e-14
    assert abs(pauli_khat(T0)[1]) < 1e-14
    assert np.allclose(np.abs(np.real(theta_loop_phase(T0))), np.pi)
    assert np.allclose(np.imag(theta_loop_phase(T0)), 0)


def test_paper_model_realized_by_rotation():
    sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1, -1])
    for eta in (0.2, 0.5, 0.8):
        gp, gm = theta_loop_phase(sz + 1j * eta * sy)
        assert np.isclose(abs(gp.imag), np.pi * eta / np.sqrt(1 - eta ** 2))
        assert np.isclose(gp.imag, -gm.imag)


def test_torch_path_is_differentiable_and_matches_numpy():
    T = torch.tensor([[0.7 + 0.1j, 0.2 - 0.3j], [-0.1 + 0.2j, 0.4 + 0.5j]],
                     dtype=torch.complex128, requires_grad=True)
    ky = pauli_khat(T)[1]
    assert np.isclose(complex(ky.detach()), pauli_khat(T.detach().numpy())[1])
    ky.imag.backward()
    assert torch.isfinite(T.grad).all() and T.grad.abs().sum() > 0


def test_lossy_fin_permittivity():
    assert fin_eps(Physics(n_fin=2.0, k_fin=0.0)) == 4.0
    assert np.isclose(fin_eps(Physics(n_fin=4.0, k_fin=0.5)), (4 + 0.5j) ** 2)
