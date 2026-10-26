"""Shared, versioned checkpoint contract for contour optimization and validation.

All units follow the existing contour engines (micrometres). The checkpoint
stores the *full* geometry parameterization, not inferred defaults. This
module does not own an electromagnetic model; both notebooks import the same
engine_ellipse_* (or legacy engine_contour_*) modules.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib
import json
import numpy as np

FORMAT_VERSION = 2
STACK_NAMES = {
    'ellipse': ('engine_ellipse_contour', 'engine_ellipse_rcwa', 'engine_ellipse_inverse'),
    'fourier': ('engine_contour', 'engine_contour_rcwa', 'engine_contour_inverse'),
}


def engine_stack(kind='ellipse'):
    """Return the actual geometry, RCWA, and inverse-design modules."""
    if kind not in STACK_NAMES:
        raise ValueError(f'Unknown contour model {kind!r}; choose {tuple(STACK_NAMES)}')
    return tuple(importlib.import_module(name) for name in STACK_NAMES[kind])


def engine_signatures(kind='ellipse'):
    """Hash all modules that determine geometry, scattering, and EP scores."""
    geom, solver, inverse = engine_stack(kind)
    other = [importlib.import_module(name) for name in
             ('engine_transmission_ep', 'engine_rcwa', 'engine_inverse', 'twofin')]
    modules = (geom, solver, inverse, *other)
    return {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()[:16]
            for m in modules}


def make_manifest(*, cfg, rules, spread, tuning, screen, coarse, refine,
                  verify_orders, verify_raster, kind='ellipse'):
    """Canonical metadata written to the optimizer's candidate_*.npz file."""
    geom, solver, _ = engine_stack(kind)
    if not isinstance(rules, geom.ContourRules):
        raise TypeError('Rules must come from the selected shared geometry engine')
    rules.validate()
    if abs(float(cfg.period) - float(rules.period)) > 1e-12:
        raise ValueError('Material period and contour-rule period disagree')
    if float(cfg.n_fin_kappa) != 0:
        raise ValueError('The lossless contour checkpoint must use zero kappa')
    if float(solver.FIXED_PORTS[1]) <= float(rules.height_bounds[1]) + float(spread.height_um):
        raise ValueError('Top monitor plane collides with a possible contour height')
    manifest = dict(format_version=FORMAT_VERSION,model_kind=kind,
                model=asdict(cfg), rules=asdict(rules),
                spread=asdict(spread), tuning=asdict(tuning),
                screen=asdict(screen), coarse=asdict(coarse), refine=asdict(refine),
                verification=dict(orders=list(verify_orders),
                                  raster=int(verify_raster), precision='complex128'),
                fixed_ports=list(map(float, solver.FIXED_PORTS)),
                code=engine_signatures(kind))
    # JSON-normalize tuples so in-memory and on-disk manifests compare exactly.
    return json.loads(json.dumps(manifest,allow_nan=False))


def manifest_id(metadata):
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]


def checkpoint_candidates(kind='ellipse', root='results'):
    """Only actual optimizer candidates; never read final_results without rules."""
    locations = {
        'ellipse': 'elliptical_contour_transmission_ep',
        'fourier': 'contour_transmission_ep',
    }
    folder = Path(root) / locations[kind]
    return sorted(folder.glob('contour_*.npz'),key=lambda p:p.stat().st_mtime,reverse=True)


def read_candidate(path, *, expected_kind='ellipse', verify_engines=True,
                   allow_engine_mismatch=False):
    """Restore exact rules and validate all raw dimensions and solver identity.

    Returns a dict with cfg/rules/raw/metadata, modules and saved settings.
    Requires checkpoint metadata; never substitutes default fabrication rules.
    """
    from twofin import Config
    from engine_rcwa import Settings as RCWASettings
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as obj:
        if 'raw' not in obj and 'raw_params' not in obj:
            raise ValueError(f'{path}: cannot recover contour from S alone')
        key = 'raw' if 'raw' in obj else 'raw_params'
        raw = np.asarray(obj[key],dtype=np.float64).reshape(-1)
        if 'metadata' not in obj:
            raise ValueError(f'{path}: missing geometry/solver metadata; cannot '
                             'safely reconstruct a contour. Use a contour_*.npz '
                             'checkpoint with rules saved by the optimizer.')
        metadata=json.loads(str(obj['metadata']))
    for required in ('rules','model','refine','fixed_ports'):
        if required not in metadata:
            raise ValueError(f'{path}: missing checkpoint metadata[{required!r}]; '
                             'cannot safely reconstruct optimized geometry')
    kind=metadata.get('model_kind')
    if kind is None:  # Legacy candidates encoded the rules but not model_kind.
        kind = 'ellipse' if 'axis_ratio_bounds' in metadata['rules'] else 'fourier'
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f'{path}: saved {kind!r} model, expected {expected_kind!r}')
    geom,solver,inverse=engine_stack(kind)
    try:
        rules=geom.ContourRules(**metadata['rules'])
        cfg=Config(**metadata['model'])
        settings=RCWASettings(**metadata['refine'])
    except (TypeError,ValueError) as exc:
        raise ValueError(f'{path}: checkpoint schema incompatible with current engines: {exc}') from exc
    rules.validate()
    if abs(cfg.period-rules.period)>1e-12:
        raise ValueError('Saved cfg.period differs from saved contour rules.period')
    modes=geom.harmonic_modes(rules) if hasattr(geom,'harmonic_modes') else tuple(range(1,rules.harmonics+1))
    expected=(4 if kind=='ellipse' else 2)+2*len(modes)
    if len(raw)!=expected:
        raise ValueError(f'{path}: saved {len(raw)} optimizer variables, but saved '
                         f'harmonics={rules.harmonics} and {kind} model require {expected}. '
                         'Checkpoint metadata are stale/inconsistent. Cannot infer the '
                         'original geometric bounds from raw variables. Re-save a '
                         'candidate with the exact optimization rules.')
    saved_ports=np.asarray(metadata['fixed_ports'],dtype=float)
    if saved_ports.shape!=(2,) or not np.allclose(saved_ports,solver.FIXED_PORTS,rtol=0,atol=1e-12):
        raise ValueError('Monitor reference planes differ from those used in optimization')
    if float(cfg.n_fin_kappa)!=0:
        raise ValueError('Expected a strictly real refractive index')
    signatures=metadata.get('code',{})
    current=engine_signatures(kind) if verify_engines else {}
    # Require matching signatures for modules actually recorded in the checkpoint.
    drift={name:(old,current[name]) for name,old in signatures.items()
           if name in current and old!=current[name]}
    if drift and not allow_engine_mismatch:
        formatted='; '.join(f'{name}: {old} != {new}' for name,(old,new) in drift.items())
        raise ValueError('Engine files changed since checkpoint was created: '+formatted+
                         '. Re-run optimization or explicitly set ALLOW_ENGINE_MISMATCH=True '
                         'for a comparison across code revisions.')
    if signatures and (geom.__name__ not in signatures or solver.__name__ not in signatures):
        if not allow_engine_mismatch:
            raise ValueError(f'{path}: optimizer checkpoint does not record the selected '
                             f'{kind} geometry and solver engine signatures. Set '
                             'ALLOW_ENGINE_MISMATCH=True only after verifying the '
                             'checkpoint was produced by those engines.')
    verification=metadata.get('verification',{})
    return dict(path=path,raw=raw,metadata=metadata,kind=kind,cfg=cfg,
                rules=rules,geom=geom,solver=solver,inverse=inverse,
                refine_settings=settings,
                verify_orders=tuple(verification.get('orders',(5,7,9))),
                verify_raster=int(verification.get('raster',256)),
                fixed_ports=tuple(map(float,saved_ports)),
                engine_drift=drift)
