import sys, json, time, numpy as np, torch
sys.path.insert(0,'/home/coopah/Desktop/python-projects/Metasurface-Design/metaatomsim-torcwa/05_twofin_eta_control')
import engine_motif as M
from engine_twofin_eta import Physics, RCWA
P=Physics(n_fin=3.88,k_fin=0.019,n_background=1.,n_bottom=1.46,n_top=1.)
O=RCWA(order=8,raster=160,device='cuda',precision='complex128')
t=lambda d:{k:torch.tensor(float(v),dtype=torch.float64,device='cuda') for k,v in d.items()}
rng=np.random.default_rng(5); N=int(sys.argv[1]); out=[]; t0=time.time()
ANISO=(0.04,0.10,0.18,0.28)
for k in range(N):
    d=dict(bar_length=rng.uniform(.10,.24),bar_width=rng.uniform(.05,.11),radius=rng.uniform(.04,.12),
           tilt_deg=rng.uniform(-70,70),height=rng.uniform(.10,.55))
    if float(M.geometry_penalty(t(dict(d,aniso=0.28)),P)[0])>0: continue
    rows=[]
    for a in ANISO:
        with torch.no_grad(): S=M.solve_torch(t(dict(d,aniso=a)),P,O).cpu().numpy()
        T=S[2:4,:2]; m=M.numpy_metrics(S); ky=M.pauli_khat(T)[1]
        rows.append(dict(aniso=a,ky=[ky.real,ky.imag],P=m['power'],eta=m['eta'],gap=m['gap'],ok=bool(m['gap']>0.03 and m['anisotropy']>0.02)))
    out.append(dict(design=d,rows=rows))
    if (k+1)%16==0: print(f'{k+1}/{N} {time.time()-t0:.0f}s',flush=True)
json.dump(out,open(sys.argv[2],'w'))
# rank: Im gamma span across anisotropy among P>=0.2 & ok points, with Re k_y spread
best=[]
for o in out:
    pts=[r for r in o['rows'] if r['ok'] and r['P']>=0.2]
    if len(pts)<2: continue
    im=[-np.pi*r['ky'][1] for r in pts]; re=[np.pi*(1-r['ky'][0]) for r in pts]
    best.append((max(im)-min(im),max(abs(np.array(im))),max(re)-min(re),o))
best.sort(key=lambda b:-b[0])
for span,mx,respan,o in best[:8]:
    d=o['design']; print(f"Imγ span {span:.2f} max|Imγ| {mx:.2f} Reγ spread {respan:.2f} | L={d['bar_length']*1e3:.0f} W={d['bar_width']*1e3:.0f} r={d['radius']*1e3:.0f} tilt={d['tilt_deg']:.0f} h={d['height']*1e3:.0f} | "
          +' '.join(f"a{r['aniso']}:Imγ{-np.pi*r['ky'][1]:+.2f}/P{r['P']:.2f}/η{r['eta']:.2f}" for r in o['rows']))
