"""Fabrication-aware differentiable contour search; user-supplied Maxwell solver and loss.

Requires the existing engine_transmission_ep.py for established Jones diagnostics.
Smoothing and morphology are numerical approximations, not fabrication proofs.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
from scipy.stats import qmc

from engine_ellipse_contour import (ContourDesign, fabrication_cost, real_dtype,
                                    harmonic_modes, contour_points)
from engine_ellipse_rcwa import solve_contour_torch
from engine_transmission_ep import loss_torch, diagnostics, screen_candidate


@dataclass(frozen=True)
class SearchSettings:
    n_screen: int = 128
    keep: int = 8
    coarse_iterations: int = 180
    refine_iterations: int = 160
    coarse_lr: float = 0.040
    refine_lr: float = 0.012
    coarse_patience: int = 75
    refine_patience: int = 90
    coarse_perturbations: int = 2
    refine_perturbations: int = 4
    fab_weight: float = 0.45
    robust_weight: float = 0.35
    seed: int = 23
    min_improvement: float = 2e-6
    screening_nonnormality_bonus: float = 0.05
    structured_fraction: float = 0.30  # Letter-inspired contours; remainder Sobol
    diversity_fraction: float = 0.50  # Remaining slots use optical/shape novelty


@dataclass(frozen=True)
class FabricationSpread:
    """Illustrative process-error amplitudes, not measured foundry statistics."""
    etch_bias_um: float = 0.005
    height_um: float = 0.010
    contour_ripple_um: float = 0.003
    # Zero defaults preserve legacy isolated-meta-atom behavior.
    layer_count: int = 0
    rotation_jitter_deg: float = 0.0
    registration_um: float = 0.0


def perturbations(rules, spread=FabricationSpread(), *, count=4, seed=23):
    """Reproducible +/− process variations; each remains non-absorbing.

    The etch bias means isotropic dilation/erosion of the single contour.
    Roughness is low-order, spatially correlated shape distortion.
    """
    rng=np.random.default_rng(seed)
    out=[]
    modes = len(harmonic_modes(rules))
    for j in range(count):
        sign=1 if j%2==0 else -1
        # Identical shape ripple for paired signs; geometries are distinct.
        if j%2==0:
            coeff=rng.normal(size=(2,modes))
            coeff*=spread.contour_ripple_um/0.14/np.maximum(1,np.asarray(harmonic_modes(rules)))
        scenario=dict(bias_um=sign*spread.etch_bias_um,
            height_um=(1 if (j//2)%2==0 else -1)*spread.height_um,
            harmonic_delta=sign*coeff)
        if spread.layer_count>0:
            # Independent per-layer registration/rotation errors are correlated
            # between +/- paired trials, with the sign flipped.
            if j%2==0:
                rotations=rng.normal(size=spread.layer_count)
                shifts=rng.normal(size=(spread.layer_count,2))
            scenario['layer_orientation_delta_rad']=(
                sign*np.deg2rad(spread.rotation_jitter_deg)*rotations)
            scenario['layer_shift_xy_um']=sign*spread.registration_um*shifts
        out.append(scenario)
    return out


def default_optical_loss(S):
    """Replaceable optical objective. Input: complex torch S (4x4, xy ports).

    Return a real scalar torch loss, OR (loss, diagnostics_dict). The
    notebook can pass its own `loss_fn` without editing an engine module.
    All screening, nominal Adam, robust perturbations and smoke tests use it.
    """
    return loss_torch(S)


def evaluate(cfg, settings, p, rules, *, pert=None, hard=False, edge_um=None,
             loss_fn=None, solver_fn=None):
    """Evaluate arbitrary notebook-defined physics and a compatible Maxwell solver."""
    import torch
    S=(solve_contour_torch if solver_fn is None else solver_fn)(
        cfg, settings, p, rules, edge_um=edge_um, perturb=pert, hard=hard)
    outcome=(default_optical_loss if loss_fn is None else loss_fn)(S)
    if isinstance(outcome, tuple):
        loss, parts = outcome
    else:
        loss, parts = outcome, {}
    if not isinstance(loss,torch.Tensor) or loss.ndim!=0 or loss.is_complex():
        raise ValueError('Optical loss must be a real scalar torch Tensor')
    return loss,S,parts


def optical_diagnostics_torch(S):
    """Transmission anisotropy versus nonnormality in the fixed x/y basis.

    Both metrics are unchanged by the same unitary basis change on input
    and output, so they also describe the circular-basis Jones blocks.
    """
    import torch
    out={}
    for name,J in [('T21',S[2:4,:2]),('T12',S[:2,2:4])]:
        energy=(torch.abs(J)**2).sum().clamp_min(1e-12)
        scalar=J-torch.trace(J)*torch.eye(2,device=S.device,dtype=S.dtype)/2
        normpart=torch.sqrt((torch.abs(scalar)**2).sum()/energy)
        comm=J.conj().T@J-J@J.conj().T
        nonnormal=torch.linalg.matrix_norm(comm,ord='fro')/energy
        out[name+'_anisotropy']=normpart
        out[name+'_nonnormality']=nonnormal
        out[name+'_mean_power']=energy/2
    return out


def optical_diagnostics_numpy(S):
    """Detachable scalar metrics for notebook reports and history plots."""
    S=np.asarray(S,dtype=complex)
    out={}
    for name,J in [('T21',S[2:4,:2]),('T12',S[:2,2:4])]:
        energy=max(float(np.sum(abs(J)**2)),1e-12)
        scalar=J-np.trace(J)*np.eye(2)/2
        comm=J.conj().T@J-J@J.conj().T
        out[name+'_anisotropy']=float(np.linalg.norm(scalar,'fro')/np.sqrt(energy))
        out[name+'_nonnormality']=float(np.linalg.norm(comm,'fro')/energy)
        out[name+'_mean_power']=energy/2
    return out


# INITIALIZATION: union-of-bars silhouettes are projected into the SAME
# ellipse + bounded Fourier parameters used by the optical solver. No masks,
# additional shapes, or alternate solvers are introduced in optimization.
SHAPE_FAMILIES = ('H', 'T', 'L', 'F', 'K', 'X', '+', 'Y')


def _letter_profile(family, rng, *, n_angles=256, n_radii=512):
    """Radial projection of a letter silhouette (dimensionless, before fit).

    H/F/K/X/Y are not generally star-shaped; the farthest-hit radial hull
    fills overhangs. Only the fitted radial result enters TORCWA.
    """
    theta=np.arange(n_angles)*2*np.pi/n_angles
    r=np.linspace(0.,1.6,n_radii)
    x=r[None,:]*np.cos(theta)[:,None]
    y=r[None,:]*np.sin(theta)[:,None]
    width=rng.uniform(.21,.35)
    reach=rng.uniform(.72,1.00)
    # Set origin in a connected core, not necessarily at the letter centroid.
    if family=='T':
        y=y+.33
        mask=(((abs(x)<width) & (y<.65) & (y>-.95)) |
              ((abs(x)<reach) & (y>.30) & (y<.65)))
    elif family=='L':
        x=x-.46; y=y-.49
        mask=(((x>-.7)&(x<-.7+2*width)&(y>-.9)&(y<.85)) |
              ((x>-.7)&(x<.90)&(y>-.9)&(y<-.9+2*width)))
    elif family=='H':
        mask=(((abs(x)>.69-width)&(abs(x)<.69+width/2)&(abs(y)<reach)) |
              ((abs(x)<.69+width/2)&(abs(y)<width)))
    elif family=='F':
        x=x-.43; y=y+.30
        mask=(((x>-.72)&(x<-.72+2*width)&(abs(y)<.91)) |
              ((x>-.72)&(x<.93)&(y>.57)&(y<.91)) |
              ((x>-.72)&(x<.57)&(y>-.09)&(y<.22)))
    elif family=='K':
        x=x-.47
        mask=(((x>-.73)&(x<-.73+2*width)&(abs(y)<reach)) |
              ((x>-.66)&(x<.82)&(abs(y-.8*(x+.65))<width)) |
              ((x>-.66)&(x<.82)&(abs(y+.8*(x+.65))<width)))
    elif family=='X':
        mask=((abs(x-y)<width*1.5)|(abs(x+y)<width*1.5)) & (abs(x)<reach) & (abs(y)<reach)
    elif family=='+':
        mask=(((abs(x)<width)&(abs(y)<reach)) |
              ((abs(y)<width)&(abs(x)<reach)))
    elif family=='Y':
        mask=(((abs(x)<width)&(y<.22)&(y>-.93)) |
              ((abs(abs(x)-y*.85)<width*1.5)&(y>-.06)&(y<.91)))
    else:
        raise ValueError(f'Unknown letter profile: {family}')
    # A radial contour cannot represent overhangs: fill along each ray.
    outer=np.where(mask,r[None,:],0.).max(axis=1)
    return theta,np.maximum(outer,.04)


def _fit_profile(rules, theta, target, *, rotation, radius):
    """Fit bounded Fourier coefficients using orthogonal harmonic projection.

    Grid-search ellipse major-axis direction and ratio; no optimization of
    the electromagnetic loss is performed during shape fitting.
    """
    modes=np.asarray(harmonic_modes(rules),float)
    envelope=rules.harmonic_amplitude/modes**1.25
    target_log=np.log(target/np.max(target))
    target_c=target_log-target_log.mean()
    upper=min(rules.axis_ratio_bounds[1]-1e-4,5.5)
    lower=rules.axis_ratio_bounds[0]+1e-4
    ratios=np.linspace(lower,upper,9)
    best=None
    for angle in (0.,np.pi/4,-np.pi/4,np.pi/2-1e-4):
        u=theta-angle
        c,s=np.cos(u),np.sin(u)
        C=np.cos(np.outer(u,modes)); B=np.sin(np.outer(u,modes))
        for ratio in ratios:
            q=np.sqrt(ratio)
            ellipse=1./np.sqrt((c/q)**2+(s*q)**2)
            ell=np.log(ellipse)
            resid=target_c-(ell-ell.mean())
            a=np.clip(2*np.mean(resid[:,None]*C,axis=0),-.985*envelope,.985*envelope)
            b=np.clip(2*np.mean(resid[:,None]*B,axis=0),-.985*envelope,.985*envelope)
            fitted=ell+C@a+B@b
            error=np.mean(((fitted-fitted.mean())-target_c)**2)
            if best is None or error<best[0]:
                best=(error,ratio,angle,a,b)
    error,ratio,angle,a,b=best
    # Rotating all modes with the ellipse rotates the complete silhouette.
    orientation=(angle+rotation+np.pi/2)%np.pi-np.pi/2
    # A pi flip changes odd harmonics; preserve the actual rotated contour.
    wrap=(angle+rotation-orientation)/np.pi
    if round(wrap)%2:
        sign=(-1.)**modes
        a,b=a*sign,b*sign
    logit=lambda t: np.log(t/(1.-t))
    raw=np.zeros(4+2*len(modes))
    raw[0]=logit(np.clip((radius-rules.radius_bounds[0])/
                         (rules.radius_bounds[1]-rules.radius_bounds[0]),.01,.99))
    raw[1]=0.  # No height freedom when height bounds are equal.
    raw[2]=logit(np.clip((ratio-rules.axis_ratio_bounds[0])/
                         (rules.axis_ratio_bounds[1]-rules.axis_ratio_bounds[0]),.001,.999))
    raw[3]=np.arctanh(np.clip(orientation/(np.pi/2),-.995,.995))
    m=len(modes)
    raw[4:4+m]=np.arctanh(np.clip(a/envelope,-.985,.985))
    raw[4+m:4+2*m]=np.arctanh(np.clip(b/envelope,-.985,.985))
    return raw.tolist(),float(np.sqrt(error))


def make_screening_initializations(rules, tuning):
    """Deterministic letter-inspired + Sobol seeds, no RCWA required.

    Returns list of {raw, family, fit_error}, length `tuning.n_screen`.
    Run once for the shape gallery; pass the same list to `screen`.
    """
    rules.validate()
    if not 0 <= tuning.structured_fraction <= 1:
        raise ValueError('structured_fraction must be between 0 and 1')
    if not 0 <= tuning.diversity_fraction <= 1:
        raise ValueError('diversity_fraction must be between 0 and 1')
    if tuning.n_screen < 1:
        raise ValueError('n_screen must be positive')
    rng=np.random.default_rng(tuning.seed)
    n_letter=int(round(tuning.structured_fraction*tuning.n_screen))
    n=4+2*len(harmonic_modes(rules))
    initial=[]
    for k in range(n_letter):
        family=SHAPE_FAMILIES[k%len(SHAPE_FAMILIES)]
        theta,radial=_letter_profile(family,rng)
        rotation=(0.0 if k < len(SHAPE_FAMILIES)
                  else rng.uniform(-.75,.75))
        size=rng.uniform(max(rules.radius_bounds[0]+.015, .115),
                         rules.radius_bounds[1]-.002)
        raw,error=_fit_profile(rules,theta,radial,rotation=rotation,radius=size)
        initial.append(dict(raw=raw,family=family,fit_error=error))
    n_sobol=tuning.n_screen-len(initial)
    if n_sobol:
        sobol=qmc.Sobol(d=n,scramble=True,seed=tuning.seed)
        q=sobol.random_base2(int(np.ceil(np.log2(max(2,n_sobol)))))[:n_sobol]
        for j,v in enumerate(q):
            u=np.clip(v,.03,.97)
            raw=np.empty_like(u)
            # Favor usable footprints over near-point-like random profiles.
            radius_u=.20+.75*u[0]
            ratio_u=.12+.72*u[2]
            raw[:3]=np.log(np.asarray((radius_u,u[1],ratio_u))/
                             (1.-np.asarray((radius_u,u[1],ratio_u))))
            raw[3]=np.arctanh(2*u[3]-1.)
            # Three levels of freeform complexity; full freedom in Adam.
            strength=(.20,.45,.80)[j%3]
            raw[4:]=np.arctanh(2*u[4:]-1.)*strength
            initial.append(dict(raw=raw.tolist(),family='Sobol',fit_error=None))
    return initial


def _shape_signature(raw,rules):
    """Unit-normalized, radially sampled physical contour (no RCWA)."""
    import torch
    with torch.no_grad():
        p=ContourDesign(rules,raw,device='cpu',dtype=torch.float64).values()
        xy=contour_points(p,rules).cpu().numpy()
    radius=np.linalg.norm(xy,axis=1)
    sig=radius[::max(1,len(radius)//64)][:64]
    return sig/(np.max(sig)+1e-12)


def _select_diverse(records, rules, tuning):
    """Retain top optical scores and distinct, optically promising contours."""
    ordered=sorted(records,key=lambda d:d['score'])
    keep=min(tuning.keep,len(ordered))
    if not keep:
        return []
    n_best=max(1,int(round(keep*(1-tuning.diversity_fraction))))
    chosen=ordered[:n_best]
    for entry in chosen:
        entry['selection_reason']='best_score'
    used={id(v) for v in chosen}
    # Preserve one H-, T-, and L-inspired start when screening found one.
    # A scientifically interesting structural family need not rank in the top 4.
    if tuning.structured_fraction>0 and tuning.diversity_fraction>0:
        for family in ('H','T','L'):
            if len(chosen)>=keep: break
            if any(v['family']==family for v in chosen): continue
            match=next((r for r in ordered if r['family']==family and id(r) not in used),None)
            if match is not None:
                match['selection_reason']='letter_coverage'
                chosen.append(match);used.add(id(match))
    # Evaluate novelty against a shortlist, not arbitrary very poor candidates.
    pool=ordered[:max(keep*15, int(len(ordered)*.40))]
    # Ensure each named letter has at least one eligible contender.
    for family in ('H','T','L'):
        members=[r for r in ordered if r['family']==family]
        if members and members[0] not in pool:
            pool.append(members[0])
    sig={id(r):_shape_signature(r['raw'],rules) for r in pool}
    spread=max(np.percentile([r['score'] for r in pool],85)-ordered[0]['score'],1e-8)
    while len(chosen)<keep:
        rest=[r for r in pool if id(r) not in used]
        if not rest:
            rest=[r for r in ordered if id(r) not in used]
            for r in rest:
                sig[id(r)]=_shape_signature(r['raw'],rules)
        if not rest:
            break
        current=[sig[id(r)] for r in chosen]
        def merit(r):
            novelty=min(float(np.linalg.norm(sig[id(r)]-s)/np.sqrt(len(s))) for s in current)
            optical=np.linalg.norm(np.asarray(r['optical_vector'])-
                      np.mean([v['optical_vector'] for v in chosen],axis=0))
            # The optical metrics are bounded for passive scattering and
            # contribute modestly; score still constrains candidate quality.
            reward=novelty+.12*min(optical,1.)
            cost=max(0.,(r['score']-ordered[0]['score'])/spread)
            # Favor missing letter forms, without forcing low-quality seeds.
            letter_bonus=.20 if r['family'] in ('H','T','L') and not any(
                v['family']==r['family'] for v in chosen) else 0.
            return reward+.18*letter_bonus-.16*cost
        next_item=max(rest,key=merit)
        next_item['selection_reason']='shape_optical_diversity'
        chosen.append(next_item);used.add(id(next_item))
    return chosen


def screen(cfg, settings, rules, tuning, *, edge_um=None, verbose=True,
           initializations=None, loss_fn=None, solver_fn=None, fab_fn=None,
           return_records=False, rerank_settings=None, rerank_pool=20):
    import torch
    initializations=(make_screening_initializations(rules,tuning)
                     if initializations is None else initializations)
    starts=[];records=[]
    for k,seed in enumerate(initializations):
        raw=seed['raw']
        design=ContourDesign(rules,raw,device=settings.device,dtype=real_dtype(settings))
        try:
            with torch.no_grad():
                p=design.values()
                L,S,_=evaluate(cfg,settings,p,rules,edge_um=edge_um,
                               loss_fn=loss_fn, solver_fn=solver_fn)
                penalty,_=(fabrication_cost if fab_fn is None else fab_fn)(p,rules)
                optics=optical_diagnostics_torch(S)
                score=float((L+tuning.fab_weight*penalty -
                             tuning.screening_nonnormality_bonus*
                             torch.minimum(optics['T21_nonnormality'],
                               torch.as_tensor(0.5,device=S.device))).detach().cpu())
                d=diagnostics(S.detach().cpu().numpy())
                if np.isfinite(score) and d['max_power_gain']<1.1:
                    obj=dict(index=k,score=score,raw=list(raw),family=seed['family'],
                             fit_error=seed.get('fit_error'),
                             optical_loss=float(L.detach().cpu()),
                             fabrication_loss=float(penalty.detach().cpu()),
                             optical_vector=[float(optics[v].detach().cpu()) for v in
                               ('T21_nonnormality','T21_anisotropy','T21_mean_power')])
                    records.append(obj)
        except (RuntimeError,ValueError) as exc:
            if verbose:print('Screen skip',k,str(exc)[:100])
        if verbose and (k+1)%16==0:
            print(f'Screen {k+1}/{len(initializations)}, feasible {len(records)}')
    if not records:
        raise RuntimeError('No valid screening results; check RCWA configuration')
    # Independent stronger Fourier resolution for promising screening candidates.
    # This stage is scored before diversity selection; it cannot hide an unstable seed.
    if rerank_settings is not None:
        pool=sorted(records,key=lambda d:d['score'])[:min(rerank_pool,len(records))]
        for r in pool:
            design=ContourDesign(rules,r['raw'],device=rerank_settings.device,
                                 dtype=real_dtype(rerank_settings))
            try:
                with torch.no_grad():
                    p=design.values()
                    loss_hi,S_hi,diag_hi=evaluate(cfg,rerank_settings,p,rules,
                        edge_um=edge_um,loss_fn=loss_fn,solver_fn=solver_fn)
                    fab_hi,_=(fabrication_cost if fab_fn is None else fab_fn)(p,rules)
                    score_hi=float((loss_hi+tuning.fab_weight*fab_hi).detach().cpu())
                    lowvec=np.asarray(r['optical_vector'],float)
                    highopt=optical_diagnostics_numpy(S_hi.detach().cpu().numpy())
                    highvec=np.array([highopt[k] for k in
                        ('T21_nonnormality','T21_anisotropy','T21_mean_power')])
                    drift=float(np.linalg.norm(highvec-lowvec))
                    r.update(high_order_score=score_hi,high_order_drift=drift,
                             screening_score=r['score'],score=score_hi+drift)
            except (RuntimeError,ValueError) as exc:
                r.update(high_order_error=str(exc)[:160],score=float('inf'))
        # Candidates excluded from the high-order pool are not eligible for
        # the expensive refinement without a comparable verification solve.
        records=[r for r in pool if np.isfinite(r['score'])]
        if not records: raise RuntimeError('No valid higher-order screening candidates')
        if verbose:
            print('High-order rerank:',[(r['family'],round(r['score'],4),
                  round(r['high_order_drift'],4)) for r in records[:10]])
    ordered=sorted(records,key=lambda d:d['score'])
    selected=_select_diverse(records,rules,tuning)
    for entry in selected:
        entry['selected']=True
    for entry in records:
        entry.setdefault('selected',False)
    if verbose:
        print('Selected seeds (score, family):',
              [(round(r['score'],5),r['family'],r['selection_reason']) for r in selected])
    starts=[r['raw'] for r in selected]
    ranks=[(r['score'],r['raw']) for r in ordered]
    return (starts,ranks,records) if return_records else (starts,ranks)


def optimize_start(cfg,settings,rules,raw,tuning,spread,*,iterations,lr,patience,
                   n_perturbations,edge_um=None,verbose=True,loss_fn=None,
                   solver_fn=None,fab_fn=None):
    import torch
    design=ContourDesign(rules,raw,device=settings.device,dtype=real_dtype(settings))
    optimizer=torch.optim.Adam(design.parameters(),lr=lr)
    trials=perturbations(rules,spread,count=n_perturbations,seed=tuning.seed+197)
    best=float('inf');best_it=0;best_raw=None;best_S=None;rows=[]
    begin=time.perf_counter()
    for it in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        p=design.values()
        try:
            nominal,S,components=evaluate(cfg,settings,p,rules,edge_um=edge_um,
                loss_fn=loss_fn,solver_fn=solver_fn)
            fab,fab_parts=(fabrication_cost if fab_fn is None else fab_fn)(p,rules)
            primary=nominal+tuning.fab_weight*fab
            if not bool(torch.isfinite(primary)):
                break
            loss_now=float(primary.detach().cpu())
            fab_score=float(fab.detach().cpu())
            S_np=S.detach().cpu().numpy()
            optics=optical_diagnostics_numpy(S_np)
            optical_components={k: float(v.detach().cpu()) for k,v in
                components.items() if isinstance(v,torch.Tensor) and v.numel()==1
                and not torch.is_complex(v)}
            primary.backward()
            # Backpropagate one RCWA graph at a time to contain GPU memory.
            del nominal,S,components,fab,fab_parts,primary,p
            robust_sum=0.
            if trials and tuning.robust_weight>0:
                for pert in trials:
                    pp=design.values()
                    lp,_,_=evaluate(cfg,settings,pp,rules,
                                    edge_um=edge_um,pert=pert,loss_fn=loss_fn,solver_fn=solver_fn)
                    fp,_=(fabrication_cost if fab_fn is None else fab_fn)(pp,rules,perturb=pert)
                    # Includes both EP degradation and fabrication-rule violations
                    # in manufactured +/- etch and thickness variants.
                    term=(tuning.robust_weight*lp + .25*tuning.fab_weight*fp)/len(trials)
                    if not bool(torch.isfinite(term)):
                        raise RuntimeError('Non-finite perturbed optical/fab response')
                    robust_sum+=float(term.detach().cpu())
                    term.backward()
                    del pp,lp,fp,term
            full_loss=loss_now+robust_sum
            if design.raw.grad is None or not bool(torch.isfinite(design.raw.grad).all()):
                break
            torch.nn.utils.clip_grad_norm_(design.parameters(),1.0)
            rows.append(dict(iteration=it,loss=full_loss,nominal_and_fab=loss_now,
                             nominal_optical_loss=loss_now-float((tuning.fab_weight*fab_score)),
                             nominal_fabrication_loss=float(fab_score),
                             weighted_robustness=robust_sum, **optics,
                             **{'objective_'+k:v for k,v in optical_components.items()}))
            if full_loss < best-tuning.min_improvement:
                best=full_loss;best_it=it;best_raw=design.export();best_S=S_np
            optimizer.step()
            if verbose and (it+1)%20==0:
                print(f' step {it+1:3}/{iterations}: combined {full_loss:.5g}, robust {robust_sum:.5g}')
            if it-best_it>=patience:break
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            if verbose:print(f' Iteration {it} terminated:',str(exc)[:160])
            break
    if best_raw is None:
        raise RuntimeError('No finite gradient step completed')
    return dict(raw=best_raw,loss=best,S=best_S,history=rows,
                duration_s=time.perf_counter()-begin)


def robust_report(cfg,settings,rules,raw,spread,*,count=10,seed=7001,
                  hard=True,edge_um=None,loss_fn=None,solver_fn=None):
    """Independent (held-out) perturbation evaluation of saved geometry."""
    import torch
    design=ContourDesign(rules,raw,device=settings.device,dtype=real_dtype(settings))
    scenarios=[None]+perturbations(rules,spread,count=count,seed=seed)
    output=[]
    with torch.no_grad():
        p=design.values()
        for j,pert in enumerate(scenarios):
            L,S,_=evaluate(cfg,settings,p,rules,pert=pert,hard=hard,edge_um=edge_um,loss_fn=loss_fn,
                             solver_fn=solver_fn)
            d=diagnostics(S.cpu().numpy())
            passed,reasons=screen_candidate(d)
            optical=optical_diagnostics_numpy(S.cpu().numpy())
            output.append(dict(scenario=j,nominal=j==0, **optical,
                ep_loss=float(L.detach().cpu()),  # Legacy field name
                optical_loss=float(L.detach().cpu()),
                T21_gap=d['T21']['gap'],T12_gap=d['T12']['gap'],
                T21_overlap=d['T21']['overlap'],T12_overlap=d['T12']['overlap'],
                full_S_unitarity=d['unitarity_error'],
                screen_pass=passed,reasons=reasons))
    return output


def save_result(path,result,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,raw=np.asarray(result['raw'],float),S=result['S'],
        loss=result['loss'],history=json.dumps(result['history']),
        metadata=json.dumps(metadata))
    return path


def load_result(path):
    with np.load(path,allow_pickle=False) as npz:
        return dict(raw=npz['raw'].tolist(),S=npz['S'],loss=float(npz['loss']),
            history=json.loads(str(npz['history'])),
            metadata=json.loads(str(npz['metadata'])))
