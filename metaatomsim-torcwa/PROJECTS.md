# metaatomsim-torcwa — project index

Each project folder holds its own notebooks, project-only engines, tests and `results/`.
Modules shared by two or more projects live in `core/`. Every notebook's first code cell puts
`../core` on `sys.path`, so **open/run notebooks with their own folder as the working directory**
(the VS Code and JupyterLab default). Results paths (`results/...`) are relative to that folder.

| Folder | Notebooks | Project-only modules | Results |
|---|---|---|---|
| `core/` | — | `twofin`, `engine_geometry`, `engine_rcwa`, `engine_validate`, `engine_fdtd`, `engine_ep`, `engine_inverse`, `engine_transmission_ep`, `engine_contour_checkpoint`, `engine_multilayer_characterization` | — |
| `01_twofin_baseline/` | `TwoFin_Engine` | — | `results/` (FDTD cache, geometry, scattering, GDS). `archive/` = earlier self-contained copy of this project |
| `02_twofin_ep_inverse/` | `TwoFin_EP_Inverse_Design`, `TwoFin_Lossless_Transmission_EP`, `TwoFin_Lossless_Reflection_EP` | `engine_reflection_ep` | `results/ep_design`, `results/lossless_transmission_ep` |
| `03_contour_ep_transport/` | `Elliptical_Contour_Robust_Transmission_EP`, `Fourier_Contour_Engine` | `engine_ellipse_{contour,inverse,rcwa}`, `engine_contour_{validation,fdtd,complex_frequency}`, `engine_transport_design` | `results/elliptical_contour_transmission_ep` (ellipse checkpoints), `results/contour_transmission_ep` (Fourier checkpoints), `results/fourier_contour_validation` |
| `04_multilayer_geometric_phase/` | `Multilayer_Geometric_Phase_Engine` | — | `results/multilayer_geometric_phase` |
| `05_twofin_eta_control/` | `TwoFin_Eta_Control_and_SingleLayer_Array` (v3 optimizer), `TwoFin_Eta_Characterization`, `SingleLayer_ComplexPB_v4…v6*`, `SingleLayer_EtaControl_v7a/v7b/v8*` (optimizers), **`TwoFin_ComplexPB_Transport_Characterization`** (complex-PB transport test) | `engine_twofin_eta`, `engine_motif` (pinwheel), `engine_transport_test` (stacks, lateral supercells), `engine_twofin_finite`, `engine_twofin_characterization` | `results/eta_control_v7b_twofin` (**current η-control meta-atom**), `results/complex_pb_transport_test`, other `results/*_v*` per run. See its `README.md` |

### 05 solver notes (v3, 2026-10-23)

`engine_twofin_eta.RCWA` now defaults to `mask='aa'` (exact area-fraction pixels) and
`factorization='nv'` (symmetrized normal-vector factorization, a TORCWA subclass). Plain-Laurent
TORCWA is **not** order-converged for these high-contrast fins (η drifted 0.71→0.38 over orders
12→20); NV converges by order ~8–10 and stays unitary to 1e-10. Use complex128 with NV. The
complex-frequency and depth-field helpers in `engine_twofin_characterization.py` still build
plain-Laurent TORCWA directly.

## Running notebooks headlessly

`python run_notebook.py <notebook> [--log run.log]` executes in the notebook's folder, saves the
notebook after every cell and streams printed output to the log (`--override` injects settings
for smoke tests).

## Cross-project data dependencies

- `01` loads `02_twofin_ep_inverse/results/ep_design/best_gradient_candidate.npz`.
- `04` reads its input checkpoint from `03_contour_ep_transport/results/elliptical_contour_transmission_ep/`
  (`checkpoint_candidates('ellipse', root='../03_contour_ep_transport/results')`).

## Tests

From this folder: `python -m pytest -q core/tests 02_twofin_ep_inverse/tests 03_contour_ep_transport/tests`.
The top-level `conftest.py` adds `core/` to `sys.path`.

Shared optional dependencies: `requirements-optional.txt`.
