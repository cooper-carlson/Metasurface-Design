"""Single z-invariant layer, passive: which material/cladding families give a complex theta-loop phase?

Figure of merit: Im k_y of the transmission Jones matrix (Im gamma = pi |Im k_y|), with Re k_y
(the path-dependent PB part), transmitted power, overlap eta and ||T-T^T||/||T||.
"""
import sys, time, json, numpy as np, torch
sys.path.insert(0,'/home/coopah/Desktop/python-projects/Metasurface-Design/metaatomsim-torcwa/05_twofin_eta_control')
import engine_twofin_eta as E
from engine_twofin_eta import (Physics, RCWA, design_from_raw, numpy_metrics, pauli_khat, transpose_asymmetry,
                               _xy_grid, _rect_coverage, rectangles, twofin_normals, rcwa_from_mask)
ORDER=int(sys.argv[2]) if len(sys.argv)>2 else 8; RAST=160; dev='cuda'
OPTS=RCWA(order=ORDER,raster=RAST,device=dev,precision='complex128')
X,Y=_xy_grid(Physics(),RAST,dev,torch.float64); h=X[1,0]-X[0,0]
# Approximate 633 nm optical constants (replace with measured data for a real design)
MAT={'TiO2':(2.584,0.0),'cSi':(3.88,0.019),'aSi':(4.2,0.5),'Au':(0.18,3.43)}
SIO2=1.46
def geom(raw):
    p=design_from_raw(torch.tensor(raw,dtype=torch.float64,device=dev))
    f=[_rect_coverage(X,Y,c,a,l,w,h) for c,a,l,w in rectangles(p,0.)]
    return p,f,twofin_normals(p,Physics(),RAST,0.)
def solve_single(phys,fill,normals):
    return rcwa_from_mask(fill,phys,OPTS,normals=normals)[1].detach().cpu().numpy()
def solve_mixed(phys,f1,f2,n2,normals):
    """Arm 1 in phys material, arm 2 in material n2 (z-invariant two-material layer)."""
    import torcwa
    eps=(phys.n_background**2+(E.fin_eps(phys)-phys.n_background**2)*f1
         +(complex(*n2)**2-phys.n_background**2)*f2*(1-f1)).to(torch.complex128)
    NV=E._nv_solver(); sim=NV(freq=1/phys.wavelength,order=[ORDER]*2,L=[phys.period]*2,dtype=torch.complex128,device=dev)
    sim.add_input_layer(eps=phys.n_bottom**2);sim.add_output_layer(eps=phys.n_top**2);sim.set_incident_angle(0.,0.)
    sim.add_layer_nv(phys.height,eps,*(n.to(torch.complex128) for n in normals));sim.solve_global_smatrix()
    return E._s_matrix(sim,dev,torch.complex128).cpu().numpy()
def family(name,rng):
    raw=rng.normal(scale=.9,size=6); p,f,nrm=geom(raw); fill=1-(1-f[0])*(1-f[1])
    if name=='TiO2_embedded (control)':
        ph=Physics(n_fin=2.584,n_background=SIO2,n_bottom=SIO2,n_top=SIO2,height=rng.uniform(.2,.8)); return solve_single(ph,fill,nrm)
    if name=='mixed TiO2|aSi on SiO2':
        ph=Physics(n_fin=2.584,n_background=1.,n_bottom=SIO2,n_top=1.,height=rng.uniform(.1,.6)); return solve_mixed(ph,f[0],f[1],MAT['aSi'],nrm)
    m=name.split()[0]; n,k=MAT[m]
    hr=(.03,.12) if m=='Au' else (.1,.8)
    ph=Physics(n_fin=n,k_fin=k,n_background=1.,n_bottom=SIO2,n_top=1.,height=rng.uniform(*hr))
    return solve_single(ph,fill,nrm)
FAMS=['TiO2_embedded (control)','TiO2 on SiO2/air','cSi on SiO2/air','aSi on SiO2/air','Au on SiO2/air','mixed TiO2|aSi on SiO2']
N=int(sys.argv[1]) if len(sys.argv)>1 else 48
summary={}
for name in FAMS:
    rng=np.random.default_rng(21); rows=[]; t=time.time()
    for k in range(N):
        with torch.no_grad(): S=family(name,rng)
        T=S[2:4,:2]; m=numpy_metrics(S); ky=pauli_khat(T)[1]
        unit=np.linalg.norm(S.conj().T@S) # informational
        rows.append(dict(im_ky=float(abs(ky.imag)),re_ky=float(ky.real),P=m['power'],eta=m['eta'],asym=transpose_asymmetry(T),gap=m['gap']))
    im=np.array([r['im_ky'] for r in rows]); P=np.array([r['P'] for r in rows]); asym=np.array([r['asym'] for r in rows])
    good=im*(P>=0.1)
    best=rows[int(np.argmax(good))]
    print(f"{name:26} |Im k_y| med {np.median(im):.4f} p90 {np.percentile(im,90):.4f} max {im.max():.4f} | "
          f"‖T-Tᵀ‖ med {np.median(asym):.1e} | P med {np.median(P):.2f} | best(P≥0.1): Imγ={np.pi*best['im_ky']:.3f} "
          f"Reky={best['re_ky']:+.3f} P={best['P']:.2f} η={best['eta']:.2f} gap={best['gap']:.2f} | {time.time()-t:.0f}s",flush=True)
    summary[name]=rows
json.dump(summary,open(sys.argv[3] if len(sys.argv)>3 else 'zinv_screen.json','w'))
