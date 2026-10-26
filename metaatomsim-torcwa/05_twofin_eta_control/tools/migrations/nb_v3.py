import re, sys, nbformat
path=sys.argv[1]
nb=nbformat.read(path,as_version=4); C=nb.cells
def rep(i,old,new,count=1):
    s=C[i].source; n=s.count(old)
    assert n==count,(i,n,old[:60]); C[i].source=s.replace(old,new)
# --- 0 intro
C[0].source+=("\n\n**v3 solver (Oct 23):** the v2 optimum was a numerical artifact. Its sigmoid edge mask let the "
 "optimizer exploit a graded-index boundary, and plain-Laurent TORCWA was not converged in Fourier order "
 "(η(u=0) drifted 0.71→0.38 over orders 12→20). v3 uses an exact **area-fraction mask** and the symmetrized "
 "**normal-vector (NV) factorization** (converged by order ~8–10, unitary to 1e-10), in complex128. Winners are "
 "re-scored with a higher-order solver before acceptance. An optional penalty keeps the circular-conversion "
 "(PB) phase flat across the η modulation.")
# --- 2 settings
s=C[2].source
a=s.index('SCREEN = RCWA('); b=s.index('SEED = ')
s=s[:a]+("# NV factorization + area-fraction mask are the engine defaults; NV needs complex128.\n"
 "SCREEN = RCWA(order=6, raster=128, device=DEVICE, precision='complex128')\n"
 "COARSE = RCWA(order=8, raster=160, device=DEVICE, precision='complex128')\n"
 "REFINE = RCWA(order=10, raster=192, device=DEVICE, precision='complex128')\n"
 "ACCEPT = RCWA(order=12, raster=256, device=DEVICE, precision='complex128')  # selection only, no gradients\n"
 "CHAR_ORDER = 12\n"
 "VERIFY_ORDERS = (8,10,12,14,16)\n"
 "VERIFY_RASTER = 256\n")+s[b:]
s=s.replace('COARSE_STEPS = 50','COARSE_STEPS = 60').replace('REFINE_STEPS = 65','REFINE_STEPS = 30')
s=s.replace("RUN_RASTER_AUDIT = False  # Optional independent raster check","RUN_RASTER_AUDIT = True  # independent raster check (cheap with the area-fraction mask)")
C[2].source=s
# --- 3 geometry markdown
rep(3,"The union is differentiably rasterized; the final hard mask and connected polygon are checked separately.",
 "The union is rasterized as an exact **area fraction** per pixel (product of two slab coverages per fin, "
 "probabilistic union): differentiable, area-preserving, and converged in raster by ~192. The hard mask is used "
 "only for display and the connected-polygon check.")
# --- 4 seeds: add v2 winner too
s=C[4].source; a=s.index("prior_checkpoint=OUT/"); b=s.index("fig,axes=")
C[4].source=s[:a]+("# Warm-start with earlier optimizer outputs, if available (re-ranked by the v3 solver).\n"
 "for name in ('twofin_eta_checkpoint.npz','twofin_eta_checkpoint_v2.npz'):\n"
 "    prior=OUT/name\n"
 "    if prior.is_file():\n"
 "        old_raw=np.load(prior,allow_pickle=False)['raw']\n"
 "        if old_raw.shape==(len(PARAM_KEYS),) and np.isfinite(old_raw).all():\n"
 "            seeds=np.vstack((old_raw[None,:],seeds))[:N_SCREEN]\n"
 "            print(f'Prior geometry from {name} added to screening seeds.')\n")+s[b:]
# --- 5 objective markdown
rep(5,"**The loss contains no anisotropy, eigenvalue gap, power or nonnormality rewards.**",
 "**The loss contains no anisotropy, eigenvalue gap, power or nonnormality rewards.** The only optional extra term is "
 "`PHASE_WEIGHT` × mean squared deviation of the R←L circular-conversion phase across the modulation samples "
 "(relative to u=0), so that changing η does not also change the PB phase offset. Set `PHASE_WEIGHT=0` for pure η span.")
# --- 6 objective code
rep(6,"OBJECTIVE_TAG = 'twofin_eta_only_span_v2'","OBJECTIVE_TAG = 'twofin_eta_span_flatphase_v3'")
rep(6,"FAB_WEIGHT = 0.40\n","FAB_WEIGHT = 0.40\nPHASE_WEIGHT = 0.25   # per rad^2 of conversion-phase deviation; 0 = eta only\n")
rep(6,"MIN_NUMERICAL_NONSCALE = 0.002\n","MIN_NUMERICAL_NONSCALE = 0.002\nMIN_ACCEPT_GAP = 0.05     # ACCEPT re-score: reject near-defective (near-EP) T\nMIN_CONVERSION = 0.05     # ACCEPT re-score: PB phase is undefined without conversion\nN_RESCORE = 4             # best training iterates per run re-scored with ACCEPT\n_U_CIRC = torch.tensor([[1,1],[1j,-1j]],dtype=torch.complex128)/np.sqrt(2)\n")
rep(6,"""    power=torch.stack([m['power'] for m in ms])""",
"""    U=_U_CIRC.to(device=S_samples[0].device,dtype=S_samples[0].dtype)
    conv=torch.stack([(U.conj().T@S[2:4,:2]@U)[0,1] for S in S_samples])  # R <- L
    dphi=torch.angle(conv/conv[len(conv)//2])
    phase_ms=(dphi**2).mean()
    loss=loss+PHASE_WEIGHT*phase_ms
    power=torch.stack([m['power'] for m in ms])""")
rep(6,"""           power_min=power.min(),valid=valid)""",
"""           power_min=power.min(),valid=valid,
           phase_rms=torch.sqrt(phase_ms),conv_min=conv.abs().min())""")
C[6].source+=('''

def reference_score(raw):
    """Re-score a raw design with the accurate ACCEPT solver (no gradients), for selection only."""
    with torch.no_grad():
        r=torch.as_tensor(np.asarray(raw),device=DEVICE,dtype=device_dtype(ACCEPT))
        loss,d,_=objective(r,ACCEPT)
    out=dict(loss=float(loss.cpu()),**scalars(d))
    out['accepted']=bool(out['valid']) and out['gap_min']>=MIN_ACCEPT_GAP and out['conv_min']>=MIN_CONVERSION
    return out
''')
# --- 7 markdown
C[7].source+=("\n\nAfter each candidate's coarse and refine runs, the `N_RESCORE` best training iterates of each stage "
 "(plus the final one) are **re-scored with `ACCEPT`** (higher order and raster). The winner is the accepted "
 "design with the best ACCEPT loss, never the best training loss. Accepted means valid, minimum gap "
 "≥ `MIN_ACCEPT_GAP` and conversion amplitude ≥ `MIN_CONVERSION`.")
# --- 8 checkpoint name
rep(8,"checkpoint=OUT/'twofin_eta_checkpoint_v2.npz'","checkpoint=OUT/'twofin_eta_checkpoint_v3.npz'")
# --- 10 adam + rescoring
rep(10,"            record=dict(iteration=it,loss=score,**scalars(d))",
       "            record=dict(iteration=it,loss=score,raw=raw.detach().cpu().numpy().tolist(),**scalars(d))")
rep(10,"""                      f"P≥{record['power_min']:.3f}")""",
       """                      f"P≥{record['power_min']:.3f}, Δφ_conv={record['phase_rms']:.3f} rad")""")
s=C[10].source; a=s.index("trajectories=[]"); b=s.index("elif checkpoint.is_file():")
C[10].source=s[:a]+'''def rescore(stage,history,k=N_RESCORE):
    """ACCEPT re-score of the k best valid training iterates of one run, plus its last iterate."""
    pool=sorted([h for h in history if h['valid']],key=lambda h:h['loss'])[:k]
    if history and all(h['iteration']!=history[-1]['iteration'] for h in pool):pool.append(history[-1])
    out=[]
    for h in pool:
        ref=reference_score(h['raw'])
        out.append(dict(stage=stage,iteration=h['iteration'],train_loss=h['loss'],
                        train_span=h['eta_span'],raw=h['raw'],ref=ref))
        print(f"   ACCEPT {stage} it {h['iteration']:3}: train Δη={h['eta_span']:.3f} -> "
              f"Δη={ref['eta_span']:.3f}, gap≥{ref['gap_min']:.3f}, |conv|≥{ref['conv_min']:.3f}, "
              f"Δφ={ref['phase_rms']:.3f} rad, accepted={ref['accepted']}")
    return out

trajectories=[]
rescored=[]
if RUN_OPTIMIZE and ranked:
    for i,entry in enumerate(ranked[:N_KEEP]):
        print(f'\\n>>> Candidate {i+1}/{N_KEEP}')
        c=adam_run(entry['raw'],COARSE,COARSE_STEPS,lr=.02,
                   patience=40,label=f'C{i} coarse')
        r=adam_run(c['raw'],REFINE,REFINE_STEPS,lr=.006,
                   patience=20,label=f'C{i} refine')
        trajectories.append(dict(candidate=i,coarse=c,refine=r))
        for stage,run in (('coarse',c),('refine',r)):
            rescored+=[dict(candidate=i,**row) for row in rescore(stage,run['history'])]
    accepted=[row for row in rescored if row['ref']['accepted']]
    if not accepted:
        print('WARNING: no iterate passed ACCEPT; using the best ACCEPT loss among valid ones.')
        accepted=[row for row in rescored if row['ref']['valid']] or rescored
    winner=min(accepted,key=lambda row:row['ref']['loss'])
    best_raw=np.asarray(winner['raw'],float)
    print(f"Winner: candidate {winner['candidate']} {winner['stage']} it {winner['iteration']} | "
          f"ACCEPT Δη={winner['ref']['eta_span']:.3f}, loss={winner['ref']['loss']:+.4f}")
    np.savez_compressed(checkpoint,raw=best_raw,
        metadata=json.dumps(dict(physics=asdict(PHYS),modulation=asdict(MOD),
          objective_tag=OBJECTIVE_TAG, bounds=BOUNDS,seed=SEED,
          settings=asdict(REFINE),accept_settings=asdict(ACCEPT),
          phase_weight=PHASE_WEIGHT,winner={k:v for k,v in winner.items() if k!='raw'}))),
        history=json.dumps([h for t in trajectories if t['candidate']==winner['candidate']
                            for h in t['refine']['history']]),
        rescored=json.dumps(rescored))
    print('Saved:',checkpoint)
'''+s[b:]
# --- markdown 13 / 17
rep(13,"This is the higher-order, hard-mask **single-layer** response.","This is the `ACCEPT`-order, area-fraction-mask, NV-factorized **single-layer** response.")
rep(17,"## 6 · Hard-mask RCWA convergence","## 6 · RCWA convergence (NV factorization) and solver cross-check")
C[17].source+=("\nThe cross-check after the order sweep compares the area-fraction mask with an 8×8 supersampled mask, "
 "the binary staircase, and plain-Laurent TORCWA at the same order. Laurent disagreeing with NV is expected and is the reason for NV.")
# --- 14 baseline S
rep(14,"solve_torch(pbest,PHYS,REFINE,hard=True)","solve_torch(pbest,PHYS,ACCEPT)")
# --- 18 convergence + cross-check
rep(18,"opts=RCWA(order=order,raster=VERIFY_RASTER,precision='complex128',\n                  device=DEVICE,edge_um=0.003)",
       "opts=RCWA(order=order,raster=VERIFY_RASTER,precision='complex128',device=DEVICE)")
C[18].source+='''
    print('Solver cross-check at order 12, raster 256, η at u=-1,0,+1:')
    base=RCWA(order=12,raster=VERIFY_RASTER,precision='complex128',device=DEVICE)
    for label,o in (('NV area-fraction',base),('NV supersampled',replace(base,mask='ss')),
                    ('NV binary',replace(base,mask='hard')),('Laurent area-fraction',replace(base,factorization='laurent'))):
        with torch.no_grad():
            e=[numpy_metrics(solve_torch(vary(pbest,MOD,u),PHYS,o).cpu().numpy())['eta'] for u in (-1.,0.,1.)]
        print(f'  {label:22} {np.round(e,4)}')
'''
# --- 20 raster list
rep(20,"for raster in (192,256,320):","for raster in (128,192,256,384):")
# --- 22 CHAR order
rep(22,"CHAR=RCWA(order=VERIFY_ORDERS[-1],","CHAR=RCWA(order=CHAR_ORDER,")
# --- 25 robustness order
rep(25,"rob_opts=RCWA(order=5,raster=192,precision='complex128',device=DEVICE)","rob_opts=replace(CHAR,raster=192)")
# --- drop hard=True from every solve (keep the display mask in cell 4)
for i,c in enumerate(C):
    if c.cell_type=='code' and i!=4 and 'hard=True' in c.source:
        c.source=re.sub(r',\s*hard=True','',c.source)
left=[i for i,c in enumerate(C) if 'hard=True' in c.source]
assert left==[4],left
# --- 39 export
rep(39,"objective=OBJECTIVE_TAG, objective_mode=ETA_OBJECTIVE,",
       "objective=OBJECTIVE_TAG, objective_mode=ETA_OBJECTIVE,\n        accept_settings=json.dumps(asdict(ACCEPT)),phase_weight=PHASE_WEIGHT,")
nbformat.validate(nb); nbformat.write(nb,path); print('notebook updated')
