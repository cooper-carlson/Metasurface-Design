"""Which structural families can give large eigenpolarization overlap eta? Random-geometry screen."""
import sys, time, numpy as np, torch, torcwa
sys.path.insert(0,'/home/coopah/Desktop/python-projects/Metasurface-Design/metaatomsim-torcwa/05_twofin_eta_control')
import engine_twofin_eta as E
from engine_twofin_eta import Physics, design_from_raw, numpy_metrics, _xy_grid, _rect_coverage, rectangles, twofin_normals, _s_matrix
dev='cuda'; dt=torch.complex128; ORDER=8; RAST=160; lam=0.633; L=0.4
NV=E._nv_solver()
def solve(layers, n_bot, n_top):
    """layers: list of (thickness, eps_grid or scalar, normals or None), bottom to top."""
    sim=NV(freq=1/lam,order=[ORDER]*2,L=[L,L],dtype=dt,device=dev)
    sim.add_input_layer(eps=n_bot**2); sim.add_output_layer(eps=n_top**2); sim.set_incident_angle(0.,0.)
    for h,eps,nrm in layers:
        if nrm is None: sim.add_layer(thickness=h,eps=eps)
        else: sim.add_layer_nv(h,eps.to(dt),nrm[0].to(dt),nrm[1].to(dt))
    sim.solve_global_smatrix(); return _s_matrix(sim,dev,dt).cpu().numpy()
X,Y=_xy_grid(Physics(),RAST,dev,torch.float64); h=X[1,0]-X[0,0]
def twofin(raw,theta):
    p=design_from_raw(torch.tensor(raw,dtype=torch.float64,device=dev))
    f=[_rect_coverage(X,Y,c,a,l,w,h) for c,a,l,w in rectangles(p,theta)]
    return p,f,twofin_normals(p,Physics(),RAST,theta)
def bar(length,width,angle_deg,cx=0.,cy=0.):
    a=torch.tensor(np.deg2rad(angle_deg),dtype=torch.float64,device=dev)
    c=torch.tensor([cx,cy],dtype=torch.float64,device=dev)
    l=torch.tensor(length,dtype=torch.float64,device=dev); w=torch.tensor(width,dtype=torch.float64,device=dev)
    f=_rect_coverage(X,Y,c,a,l,w,h)
    sdf=E._signed_rectangle(X,Y,c,a,l,w); gx=torch.roll(sdf,-1,0)-torch.roll(sdf,1,0); gy=torch.roll(sdf,-1,1)-torch.roll(sdf,1,1)
    n=torch.sqrt(gx**2+gy**2+1e-30); return f,(gx/n,gy/n)
SI=1.46; TIO2=2.584; LOSSY=3.9+0.3j   # LOSSY: generic high-index absorber (roughly a-Si-like at 633 nm)
def fam_embedded(rng):      # current: TiO2 two-fin embedded in silica
    raw=rng.normal(scale=.9,size=6); _,f,n=twofin(raw,0.); fill=1-(1-f[0])*(1-f[1])
    return [(rng.uniform(.2,.7),SI**2+(TIO2**2-SI**2)*fill,n)],SI,SI
def fam_substrate_air(rng):  # break z-mirror: TiO2 two-fin on silica, in air
    raw=rng.normal(scale=.9,size=6); _,f,n=twofin(raw,0.); fill=1-(1-f[0])*(1-f[1])
    return [(rng.uniform(.2,.7),1+(TIO2**2-1)*fill,n)],SI,1.0
def fam_lossy(rng):          # uniform loss: lossy two-fin embedded in silica
    raw=rng.normal(scale=.9,size=6); _,f,n=twofin(raw,0.); fill=1-(1-f[0])*(1-f[1])
    return [(rng.uniform(.08,.35),SI**2+(LOSSY**2-SI**2)*fill,n)],SI,SI
def fam_diffloss(rng):       # differential loss (PT-like): arm1 TiO2, arm2 lossy
    raw=rng.normal(scale=.9,size=6); _,f,n=twofin(raw,0.)
    eps=SI**2+(TIO2**2-SI**2)*f[0]+(LOSSY**2-SI**2)*f[1]*(1-f[0])
    return [(rng.uniform(.15,.5),eps,n)],SI,SI
def fam_bilayer(rng):        # twisted bilayer of TiO2 bars in silica (3D chiral)
    l1,l2=rng.uniform(.15,.34,2); w1,w2=rng.uniform(.05,.12,2); tw=rng.uniform(10,80); hgt=rng.uniform(.15,.45,2); sp=rng.uniform(.02,.3)
    f1,n1=bar(l1,w1,0.); f2,n2=bar(l2,w2,tw)
    return [(hgt[0],SI**2+(TIO2**2-SI**2)*f1,n1),(sp,SI**2,None),(hgt[1],SI**2+(TIO2**2-SI**2)*f2,n2)],SI,SI
def fam_bilayer_diffloss(rng): # orthogonal-ish bars, one lossy, stacked: coupled-resonator PT recipe in polarization
    l1,l2=rng.uniform(.15,.34,2); w1,w2=rng.uniform(.05,.12,2); tw=rng.uniform(60,120); h1=rng.uniform(.15,.45); h2=rng.uniform(.06,.25); sp=rng.uniform(.02,.3)
    f1,n1=bar(l1,w1,0.); f2,n2=bar(l2,w2,tw)
    return [(h1,SI**2+(TIO2**2-SI**2)*f1,n1),(sp,SI**2,None),(h2,SI**2+(LOSSY**2-SI**2)*f2,n2)],SI,SI
FAMS=dict(embedded_twofin=fam_embedded,substrate_air_twofin=fam_substrate_air,lossy_twofin=fam_lossy,
          diffloss_twofin=fam_diffloss,twisted_bilayer=fam_bilayer,diffloss_bilayer=fam_bilayer_diffloss)
N=int(sys.argv[1]) if len(sys.argv)>1 else 40
for name,fn in FAMS.items():
    rng=np.random.default_rng(11); rows=[]; t=time.time()
    for k in range(N):
        layers,nb,nt=fn(rng)
        with torch.no_grad(): S=solve(layers,nb,nt)
        m=numpy_metrics(S); T=S[2:4,:2]
        Tb=S[0:2,2:4]
        rows.append((m['eta'],m['power'],m['gap'],np.linalg.norm(T-T.T)/np.linalg.norm(T),np.linalg.norm(Tb-T.T)/np.linalg.norm(T)))
    r=np.array(rows); e=r[:,0]
    print(f"{name:22} η median {np.median(e):.3f}  p90 {np.percentile(e,90):.3f}  max {e.max():.3f} | "
          f"P median {np.median(r[:,1]):.2f} | ‖T-Tᵀ‖ median {np.median(r[:,3]):.1e} | recip err {r[:,4].max():.1e} | {time.time()-t:.0f}s",flush=True)
