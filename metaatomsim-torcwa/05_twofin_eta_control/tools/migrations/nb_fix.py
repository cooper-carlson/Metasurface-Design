import sys, nbformat
SETUP = ("# Shared engines live in ../core (see ../PROJECTS.md). Run with the notebook folder as cwd.\n"
         "import sys; from pathlib import Path\n"
         "sys.path.insert(0, str((Path.cwd().parent / 'core').resolve()))")
REPLACE = {
  '01_twofin_baseline/TwoFin_Engine.ipynb':
    [("load_checkpoint(OUT/'ep_design'/'best_gradient_candidate.npz')",
      "load_checkpoint(Path('../02_twofin_ep_inverse/results/ep_design/best_gradient_candidate.npz'))")],
  '04_multilayer_geometric_phase/Multilayer_Geometric_Phase_Engine.ipynb':
    [("checkpoint_candidates('ellipse')",
      "checkpoint_candidates('ellipse', root='../03_contour_ep_transport/results')"),
     ("# Path('results/elliptical_contour_transmission_ep/contour_...npz')",
      "# Path('../03_contour_ep_transport/results/elliptical_contour_transmission_ep/contour_...npz')")],
}
for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    first = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code')
    if 'Shared engines live in ../core' not in nb.cells[first].source:
        nb.cells.insert(first, nbformat.v4.new_code_cell(SETUP))
    for old, new in REPLACE.get(path, []):
        n = sum(c.source.count(old) for c in nb.cells)
        assert n >= 1, (path, old)
        for c in nb.cells: c.source = c.source.replace(old, new)
        print(f'{path}: replaced {n}x {old[:50]}')
    nbformat.validate(nb); nbformat.write(nb, path)
    print('ok', path)
