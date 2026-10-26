# Two-fin η-control — characterization notebook

**Start with `TwoFin_Eta_Characterization.ipynb`.** This project is a companion to the **revised η-only connected-two-fin optimizer** (`twofin_eta_checkpoint_v2.npz`), **not** the older Fourier contour or the earlier slope-only version. The optimizer's output must exist at `results/twofin_eta_transport/twofin_eta_checkpoint_v2.npz` relative to the Jupyter working directory, or edit `CHECKPOINT_DIR` in the notebook. It can also read `twofin_eta_final.npz` with its saved physical metadata.

## Status (2026-10-26): η-control meta-atom and complex-PB transport test

**Solver.** `engine_twofin_eta.RCWA` defaults to the area-fraction mask and the symmetrized normal-vector (NV) factorization
(complex128). Plain-Laurent TORCWA was not order-converged for these fins, so results from v2 and earlier are artifacts.

**Meta-atom (v7b, `results/eta_control_v7b_twofin/`).** A c-Si two-fin (n = 3.88 + 0.019i, approximate), 495 nm tall, on SiO₂ with air above,
Λ = 400 nm. Moving the junction 36 nm along arm 1 tunes η smoothly from 0.03 to 0.70 (eigen-gap ≥ 0.47, P 0.33–0.48, Re γ varies by
0.14 rad). It is converged, period-insensitive and robust to few-nm errors. η still depends on the atom's rotation inside the lattice at high
η (`eta_characterization.txt`, `v7b_eta_control_summary.png`). v8 re-optimization returned essentially the same atom.

**Transport test (`TwoFin_ComplexPB_Transport_Characterization.ipynb`, `results/complex_pb_transport_test/`).** Reproduces the
paper's and SM's numbers (Eq. 9/22, discrete Wilson, path reversal, geometric propagator, §XI, §XIII). Physical PB rotation gives
γ± = π(1 ∓ k̂_y), so a complex part needs non-Hermiticity along the circular axis (T ≠ Tᵀ). The cycle-reversal eigenbranch ratio
reaches the Wilson prediction in an ordered cascade of full-wave layers, about one third of it in full-RCWA stacks after spacer
averaging, and no transport signature appears in lateral supercells (`transport_test_summary.png`).

**Other notebooks.** `SingleLayer_ComplexPB_v4–v6*` (Im γ objectives; the large values came from nonlocal lattice resonances),
`SingleLayer_EtaControl_v7a/v7b/v8*` (η objectives), `engine_motif.py` (pinwheel geometry), `engine_transport_test.py`
(multilayer stacks, 1×N supercells, eigen bookkeeping), `tools/` (notebook generators). `TwoFin_Eta_Characterization.ipynb` still
targets the old v3 TiO₂ checkpoint, and its complex-frequency and depth-field helpers use plain-Laurent TORCWA.

## Contents

- `TwoFin_Eta_Characterization.ipynb`: 33 cells, physics-first characterization and visualization.
- `engine_twofin_characterization.py`: checkpoint reader, polarization mode tracking, closed biorthogonal Wilson products, exploratory complex-frequency solver, rational fits, and optional depth field sampling.
- `engine_twofin_eta.py`: the optimizer's connected geometry/RCWA engine, the source of truth for the two rectangles, discretization and 4×4 power-normalized scattering. **v3 (2026-10-23):** area-fraction mask and normal-vector factorization are the defaults; see `../PROJECTS.md`. Checkpoints from v2 (`twofin_eta_checkpoint_v2.npz`) came from an unconverged solver and should not be interpreted.
- `engine_twofin_finite.py`: unchanged optional finite-array Meep engine.
- `tests/test_twofin_mask.py`: CPU tests of the area-fraction mask (area vs exact polygon, supersampled agreement, gradients, C4). (`test_characterization.py` and `smoke_notebook.py` listed previously are not present.)

## Key workflow

1. **Checkpoint/geometry**: loads exact `raw`, material indices, height and modulation parameters. Rejects bound mismatches instead of substituting defaults. Connected union mask SVG and optional GDS export.
2. **Full complex 4×4 S** and right eigenvector η with an independent check against the analytic overlap formula. Reflection, unitarity, reciprocity and transmission are displayed separately.
3. **Numerical audit**: Fourier-order convergence at fixed raster (default orders 6,8,10,12); independent raster check optional.
4. **Physical parameter sweeps**: dense η(u), nonnormality, gap, anisotropy, transmission, finite-difference sensitivities over all six parameters.
5. **Eigenpolarization evolution**: tracked right and reciprocal-left eigenpolarizations on a Poincaré sphere and tracked transmission eigenvalues.
6. **Kinematic holonomy**: closed `(u, θ)` rectangle, forward/reversed biorthogonal Wilson products, orthogonalized and analytic controls, optional grid-refinement tests. These are **virtual parameter-space paths**, not actual light passing laterally through independent meta-atoms.
7. **PB control**: η versus θ, circular-conversion amplitude and phase, ±2θ references, lattice rotation covariance concerns.
8. **Real-frequency spectrum**: complex full S versus wavelength, transmission/reflection energy, eigenvalues and polarization properties.
9. **Exploratory complex frequency** (opt-in): complex `f` plane, scalar complex Jones entry and determinant magnitudes/phase, rational poles/zeros, train/held-out fit errors, Fourier-order spot-check. This uses **raw** TORCWA S (`power_norm=False`) with a frequency-independent normal-incidence port scaling; it refuses to proceed if its real-axis result disagrees with the ordinary solver.
10. **Optional physical tests**: z-resolved TORCWA E/H fields (total fields, not a local forward generator), spatially reordered periodic supercells with all open diffraction orders, true finite Meep array with lateral PML.
11. **Export**: `results/twofin_eta_characterization/twofin_eta_characterization.npz` plus mask SVG/GDS.

## Scientific interpretation

`η = |⟨R_+|R_-⟩|` is computed from eigenvectors of the *specular far-field transmission Jones matrix*. At a 633 nm wavelength and 400 nm period with silica exterior index 1.46, only the zeroth exterior diffraction order propagates. Evanescent Fourier components are **coupled in the RCWA solve** but are not independent far-field outgoing channels. At a supercell period, new exterior diffraction orders become propagating and must be included in energy bookkeeping.

For a nonscalar 2×2 Jones matrix, overlap depends algebraically on normalized non-scalar strength and eigenvalue gap. Even though the current optimizer's reward uses **only η span**, there is no physical ability to keep both quantities fixed and independently change η. Treat near-scalar and near-defective cases as ill-conditioned.

**Important:** A biorthogonal Wilson product across *different lateral unit cells* is only a mathematical parameter-space construction. For a single-layer array under normal illumination, input light generally interacts with pixels in **parallel**, not in sequence. A nonzero imaginary Wilson phase computed from those Jones eigenvectors does **not** prove actual complex geometric phase. An electromagnetic ordered-transport claim needs a physical path and a justified local generator, alongside resonance/dynamical and discretization controls.

The analytic reference model uses `K(φ,η) = [[iη,e^(-iφ)],[e^(iφ),−iη]]`. Its Wilson-loop result tests the implementation independently of any Maxwell approximation.

### Complex-frequency caution

TORCWA is documented for real-frequency scattering. Nonreal-frequency usage is an **exploratory numerical continuation** and may encounter incorrect outgoing branch choices, Rayleigh branch cuts, convergence problems or material-dispersion errors. The raw-channel port scaling is valid at normal incidence with **frequency-independent homogeneous** port indices and must be checked on the real axis. Rational-fit poles are **not certified QNMs** or EPs. Confirmation requires convergence in Fourier order/raster, stable continuation paths, changes of fit window, and independent mode solvers or appropriate residue tests. Do not confuse frequency poles with eigenvalue degeneracies of a single real-frequency Jones matrix.

## Installation and testing

Put the notebook and all three `.py` modules together in the directory from which you run Jupyter, or copy these files into the existing `metaatomsim-torcwa` working folder. Run the revised η-only optimizer first and **do not overwrite its checkpoint**.

```
python -m pip install numpy scipy torch matplotlib torcwa shapely nbformat pytest
python -m pytest -q tests
jupyter lab
```

`gdstk` and `meep` are optional for GDS and finite-array FDTD. Device is CUDA when available, CPU otherwise. Heavy toggles in cell 1 default off: `RUN_COMPLEX_FREQUENCY`, `RUN_Z_FIELDS`, `RUN_SUPERCELL_ORDER_TEST`, `RUN_FINITE_ARRAY`, and `RUN_WILSON_REFINEMENT`. Even enabled default spectral/Wilson/PB sections require numerous real TORCWA solves.

## Verification performed

- Notebook JSON and every code cell validated/compiled.
- **8/8 CPU tests passed** for overlap identity, Wilson loop, analytic non-Hermitian benchmark, rational fit, checkpoint contract.
- All normally enabled notebook cells passed an end-to-end **synthetic-backend smoke test**. Those synthetic outputs were not saved in the deliverable and **are not physical results**.
- Real TORCWA and Meep simulations could not be executed in this environment. The complex-frequency TORCWA pathway, in particular, **must be validated on your workstation** before interpreting its plots.
