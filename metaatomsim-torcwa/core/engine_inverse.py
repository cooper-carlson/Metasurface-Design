"""GPU-efficient, bounded multistart inverse design using differentiable TORCWA.

No EP is presumed. Candidate ranking is heuristic; only converged full-S
Jordan diagnostics and parameter-loop monodromy support an EP claim.
"""
from dataclasses import replace
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.stats import qmc

from engine_ep import ep_loss_torch, ep_diagnostics
from engine_rcwa import solve_torch

# Physical dimensions are µm; angles are radians; loss is Im(n_fin)>=0.
BOUNDS = {
    'L1': (0.09, 0.34), 'W1': (0.035, 0.15),
    'theta1': (-np.pi/2, np.pi/2), 'x1': (-0.13,0.13), 'y1': (-0.13,0.13),
    'L2': (0.09, 0.34), 'W2': (0.035, 0.15),
    'theta2': (-np.pi/2, np.pi/2), 'x2': (-0.13,0.13), 'y2': (-0.13,0.13),
    'height': (0.15, 0.633), 'kappa': (0.005, 0.22),
}
KEYS = tuple(BOUNDS)
# Fixed experimental reference planes for every candidate (µm).
# EP eigenvalues are NOT invariant to moving these planes.
FIXED_PORTS = (-0.52, 1.05)


def numpy_to_params(values, *, device='cpu', dtype=None):
    import torch
    dtype = dtype or torch.float32
    return {k:torch.as_tensor(float(values[k]),device=device,dtype=dtype)
            for k in KEYS}


def enforce_unit_cell(physical, period=0.400, clearance=0.006):
    """Clamp sampled center coordinates to keep rotated footprints in one cell."""
    values=dict(physical)
    for i in (1,2):
        t=float(values[f'theta{i}'])
        ex=(values[f'L{i}']*abs(np.cos(t))+values[f'W{i}']*abs(np.sin(t)))/2
        ey=(values[f'L{i}']*abs(np.sin(t))+values[f'W{i}']*abs(np.cos(t)))/2
        boundx=max(0.001,min(0.13,period/2-clearance-ex))
        boundy=max(0.001,min(0.13,period/2-clearance-ey))
        values[f'x{i}']=float(np.clip(values[f'x{i}'],-0.995*boundx,0.995*boundx))
        values[f'y{i}']=float(np.clip(values[f'y{i}'],-0.995*boundy,0.995*boundy))
    return values


class BoundedDesign:
    """Unconstrained Adam variables mapped to *strictly feasible* placements."""
    def __init__(self, values, *, device='cuda', dtype=None):
        import torch
        dtype = dtype or torch.float32
        feasible=enforce_unit_cell(values)
        initial=[]
        for key in KEYS:
            lo,hi=BOUNDS[key]
            if key.startswith('x') or key.startswith('y'):
                # Physical centers have a separate geometry-dependent map.
                initial.append(0.0)
            else:
                p=np.clip((float(feasible[key])-lo)/(hi-lo),.001,.999)
                initial.append(math.log(p/(1-p)))
        for i in (1,2):
            t=feasible[f'theta{i}']
            ex=(feasible[f'L{i}']*abs(np.cos(t))+feasible[f'W{i}']*abs(np.sin(t)))/2
            ey=(feasible[f'L{i}']*abs(np.sin(t))+feasible[f'W{i}']*abs(np.cos(t)))/2
            for axis,extent in (('x',ex),('y',ey)):
                limit=max(.001,min(.13,.2-.006-extent))
                center=np.clip(feasible[f'{axis}{i}']/limit,-.995,.995)
                initial[KEYS.index(f'{axis}{i}')]=float(np.arctanh(center))
        self.raw=torch.nn.Parameter(torch.tensor(initial,device=device,dtype=dtype))

    def values(self):
        import torch
        out={}
        for j,key in enumerate(KEYS):
            if key.startswith('x') or key.startswith('y'):
                continue
            lo,hi=BOUNDS[key]
            out[key]=lo+(hi-lo)*torch.sigmoid(self.raw[j])
        for i in (1,2):
            t=out[f'theta{i}']
            ex=(out[f'L{i}']*torch.sqrt(torch.cos(t)**2+1e-12)+
                out[f'W{i}']*torch.sqrt(torch.sin(t)**2+1e-12))/2
            ey=(out[f'L{i}']*torch.sqrt(torch.sin(t)**2+1e-12)+
                out[f'W{i}']*torch.sqrt(torch.cos(t)**2+1e-12))/2
            bx=torch.clamp(torch.minimum(torch.full_like(ex,.13),.2-.006-ex),min=.001)
            by=torch.clamp(torch.minimum(torch.full_like(ey,.13),.2-.006-ey),min=.001)
            out[f'x{i}']=bx*torch.tanh(self.raw[KEYS.index(f'x{i}')])
            out[f'y{i}']=by*torch.tanh(self.raw[KEYS.index(f'y{i}')])
        return out

    def physical(self):
        return {k:float(v.detach().cpu()) for k,v in self.values().items()}


def geometry_penalty(params, period, clearance=0.005):
    """Soft penalty for periodic-edge crossing; rectangles may overlap."""
    import torch
    out=params['L1']*0
    for j in (1,2):
        theta=params[f'theta{j}']
        c=torch.sqrt(torch.cos(theta)**2+1e-8)
        s=torch.sqrt(torch.sin(theta)**2+1e-8)
        ex=params[f'L{j}']*c/2 + params[f'W{j}']*s/2
        ey=params[f'L{j}']*s/2 + params[f'W{j}']*c/2
        lim=period/2-clearance
        bx=torch.sqrt(params[f'x{j}']**2+1e-8)+ex-lim
        by=torch.sqrt(params[f'y{j}']**2+1e-8)+ey-lim
        out=out+torch.relu(bx)**2+torch.relu(by)**2
    return 800*out


def simulation_loss(cfg, settings, params, *, edge_um=0.006):
    S=solve_torch(cfg,params,settings,edge_um=edge_um,ports=FIXED_PORTS)
    ep,components=ep_loss_torch(S)
    geometric=geometry_penalty(params,cfg.period)
    # Prefer lower absorption among comparably close EP candidates.
    regularization=0.05*params['kappa']**2
    return ep+geometric+regularization, S, components


def sobol_screen(cfg, settings, *, candidates=24, keep=4, seed=11,
                 edge_um=0.008, verbose=True):
    """Low-order forward-only global candidate screen (no gradients)."""
    import torch
    sobol=qmc.Sobol(d=len(KEYS),scramble=True,seed=seed)
    points=sobol.random_base2(int(np.ceil(np.log2(max(2,candidates)))))[:candidates]
    candidates_found=[]
    for idx,point in enumerate(points):
        physical={k:BOUNDS[k][0]+float(point[j])*(BOUNDS[k][1]-BOUNDS[k][0])
                  for j,k in enumerate(KEYS)}
        physical=enforce_unit_cell(physical)
        params=numpy_to_params(physical,device=settings.device)
        with torch.no_grad():
            try:
                value,S,_=simulation_loss(cfg,settings,params,edge_um=edge_um)
                if not torch.isfinite(value):
                    continue
                d=ep_diagnostics(S.detach().cpu().numpy())
                if d['max_power_gain'] > 1.05:
                    continue
                rank=float(value.cpu())
                # Include evidence of eigenvector coalescence in start selection.
                rank+=3*max(0,0.8-d['best']['overlap'])
                candidates_found.append((rank,physical,d['best']['gap']))
            except (RuntimeError,ValueError) as e:
                if verbose: print('Skipping unstable seed',idx,':',str(e)[:120])
        if verbose and (idx+1)%8==0:
            print(f'  Screened {idx+1}/{candidates}; viable {len(candidates_found)}')
    if not candidates_found:
        raise RuntimeError('No viable seeds. Try lower RCWA order or different bounds.')
    candidates_found.sort(key=lambda x:x[0])
    return [x[1] for x in candidates_found[:keep]],candidates_found


def adam_refine(cfg,settings,initial,*,iterations=80,lr=0.05,
                edge_um=0.006,patience=30,print_every=10):
    """One differentiable local optimization, returning best recorded iterate."""
    import torch
    model=BoundedDesign(initial,device=settings.device)
    opt=torch.optim.Adam([model.raw],lr=lr)
    best=float('inf'); best_params=None; best_S=None; best_step=0
    hist=[]; start=time.perf_counter()
    for it in range(iterations):
        opt.zero_grad(set_to_none=True)
        params=model.values()
        loss,S,components=simulation_loss(cfg,settings,params,edge_um=edge_um)
        if not torch.isfinite(loss):
            print('Non-finite objective; terminating this start.')
            break
        loss.backward()
        if model.raw.grad is None or not torch.isfinite(model.raw.grad).all():
            print('Non-finite gradient; terminating this start.')
            break
        torch.nn.utils.clip_grad_norm_([model.raw],max_norm=5.0)
        val=float(loss.detach().cpu())
        row={'iteration':it,'loss':val,'disc':float(components['discriminant'].detach().cpu()),
             'nonnormality':float(components['nonnormality'].detach().cpu()),
             'forward_power':float(components['forward_power'].detach().cpu())}
        hist.append(row)
        if val < best-1e-5:
            best,best_step=val,it
            best_params=model.physical()
            best_S=S.detach().cpu().numpy()
        opt.step()
        if (it+1)%print_every==0:
            print(f'    iter {it+1:3d}/{iterations}: L={val:+.4f}, '
                  f'disc={row["disc"]:.2e}, nonnormal={row["nonnormality"]:.3f}')
        if it-best_step>=patience and it>patience:
            print('    Stopped: objective plateau (NOT an EP certificate).')
            break
    if best_params is None:
        raise RuntimeError('No finite gradient step completed')
    return {'loss':best,'params':best_params,'S':best_S,
            'history':hist,'seconds':time.perf_counter()-start}


def multistart(cfg, settings, starts, *, iterations=80, lr=0.05, edge_um=.006):
    attempts=[]
    for n,physical in enumerate(starts,1):
        print(f'\nAdam start {n}/{len(starts)} / order {settings.order}')
        try:
            attempts.append(adam_refine(cfg,settings,physical,
                            iterations=iterations,lr=lr,edge_um=edge_um))
        except (RuntimeError,ValueError) as e:
            print('  Failed:',str(e)[:200])
    if not attempts:
        raise RuntimeError('All local optimizations failed')
    return sorted(attempts,key=lambda d:d['loss'])


def candidate_to_config(cfg, candidate):
    """Map independent fin parameters into the legacy geometry/GDS Config."""
    from twofin import rotation
    params={k:float(candidate[k]) for k in KEYS}
    theta1=np.rad2deg(params['theta1'])
    theta2=np.rad2deg(params['theta2'])
    undo=rotation(-theta1)
    c1=undo@np.array([params['x1'],params['y1']])
    c2=undo@np.array([params['x2'],params['y2']])
    updated=replace(cfg,fin_length=params['L1'],fin_width=params['W1'],
                    fin2_length=params['L2'],fin2_width=params['W2'],
                    global_deg=theta1,relative_deg=theta2-theta1,
                    center1=tuple(c1),center2=tuple(c2),
                    height=params['height'],n_fin_kappa=params['kappa'])
    updated.validate()
    return updated


def save_checkpoint(path, result, extra=None):
    """Export only serializable best-candidate data; preserve complex matrices."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    S=np.asarray(result['S'])
    np.savez_compressed(path,S=S,params=json.dumps(result['params']),
                        history=json.dumps(result['history']),
                        loss=result['loss'],extra=json.dumps(extra or {}))
    return path


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as f:
        return {'S':f['S'],'params':json.loads(str(f['params'])),
                'history':json.loads(str(f['history'])),
                'loss':float(f['loss'])}


def polish_hard_edges(cfg, initial, settings, *, variable_keys=('theta1','theta2','kappa','height'),
                      max_evaluations=50, verbose=True):
    """Optional derivative-free refinement of the *binary* RCWA geometry.

    Powell uses a small parameter subset to contain expensive solver calls.
    The output remains subject to high-order convergence and Jordan checks.
    """
    import torch
    from scipy.optimize import minimize
    from engine_rcwa import solve as rcwa_solve
    from engine_ep import ep_loss_torch
    x0=np.array([initial[k] for k in variable_keys],float)
    bounds=[BOUNDS[k] for k in variable_keys]
    tracker={'best':float('inf'),'candidate':None,'calls':0}

    def objective(x):
        trial=dict(initial)
        trial.update({k:float(v) for k,v in zip(variable_keys,x)})
        trial=enforce_unit_cell(trial)
        try:
            physical=candidate_to_config(cfg,trial)
            S=rcwa_solve(physical,settings,ports=FIXED_PORTS)['S']
            diagnostic=ep_diagnostics(S)
            if diagnostic['max_power_gain']>1.03:
                return 100.
            ep,_=ep_loss_torch(torch.tensor(S,dtype=torch.complex128))
            score=float(ep)+.2*trial['kappa']
        except (RuntimeError,ValueError, FloatingPointError):
            return 100.
        tracker['calls']+=1
        if score<tracker['best']:
            tracker['best'],tracker['candidate']=score,(trial,S)
        if verbose and tracker['calls']%10==0:
            print(f'Binary RCWA evaluation {tracker["calls"]}: objective {score:+.4f}')
        return score

    result=minimize(objective,x0,method='Powell',bounds=bounds,
                    options={'maxfev':max_evaluations,'xtol':1e-3,'ftol':1e-3})
    if tracker['candidate'] is None:
        raise RuntimeError('Hard-edge local refinement found no feasible design')
    trial,S=tracker['candidate']
    return dict(params=trial,S=S,loss=tracker['best'],
                calls=tracker['calls'],optimization_message=str(result.message))
