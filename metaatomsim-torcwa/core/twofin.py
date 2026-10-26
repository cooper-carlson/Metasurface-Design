"""Two-fin Al2O3 periodic cell: geometry, GDSII, and Jones/Stokes analysis.

Coordinates and dimensions in micrometers. Time convention E(t)=Re[E exp(-i wt)].
Ports: bottom=1 (silica), top=2 (air), shared laboratory x/y basis.
Stokes S3=+2 Im(Ex* Ey) (Jones [1,+i] is S3=+1).
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import math
import struct
import datetime
import numpy as np


@dataclass(frozen=True)
class Config:
    wavelength: float = 0.633
    period: float = 0.400
    n_sub: float = 1.46       # fused silica, nondispersive approximation
    n_fin: float = 1.76       # alumina real index, nondispersive approximation
    n_fin_kappa: float = 0.0  # passive extinction; 0 means lossless
    height: float = 0.350
    fin_length: float = 0.160
    fin_width: float = 0.064
    fin2_length: float | None = None  # independent second-fin length
    fin2_width: float | None = None   # independent second-fin width
    relative_deg: float = 90.0
    global_deg: float = 0.0
    # Template locations rotate together with the nanofins about origin.
    center1: tuple[float, float] = (0.105, -0.105)
    center2: tuple[float, float] = (-0.105, 0.105)
    resolution: int = 200
    pml: float = 0.40
    sub_depth: float = 1.10   # from interface z=0 to beginning of bottom PML
    air_depth: float = 1.25   # from fin top to beginning of upper PML
    lower_source: float = -0.82
    lower_port: float = -0.48
    upper_port_offset: float = 0.49  # from fin top
    upper_source_offset: float = 0.85
    fwidth_fraction: float = 0.12
    decay_tol: float = 1e-8
    # The local-field trace is NOT an isolated forward-mode propagation path.

    @property
    def upper_port(self):
        return self.height + self.upper_port_offset

    @property
    def upper_source(self):
        return self.height + self.upper_source_offset

    def validate(self):
        assert self.period > 0 and self.height > 0 and self.resolution >= 10
        assert self.fin_length > 0 and self.fin_width > 0 and self.pml > 0
        assert self.fin2_length is None or self.fin2_length > 0
        assert self.fin2_width is None or self.fin2_width > 0
        assert self.n_sub > 0 and self.n_fin > 0 and self.wavelength > 0
        assert self.n_fin_kappa >= 0
        assert -self.sub_depth < self.lower_source < self.lower_port < 0
        assert self.height < self.upper_port < self.upper_source < self.height+self.air_depth
        assert self.period < self.wavelength/max(1, self.n_sub), 'Nonzero substrate diffraction order opens'
        for rect in fins(self):
            if not np.all(np.abs(rect) < self.period/2 - 0.003):
                raise ValueError('Fin crosses unit-cell boundary; modify positions/angles/period')
        return self


def rotation(theta_deg):
    t = np.deg2rad(theta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def fins(p: Config):
    base = np.array([[-p.fin_length/2,-p.fin_width/2],
                     [p.fin_length/2,-p.fin_width/2],
                     [p.fin_length/2,p.fin_width/2],
                     [-p.fin_length/2,p.fin_width/2]])
    L2 = p.fin2_length if p.fin2_length is not None else p.fin_length
    W2 = p.fin2_width if p.fin2_width is not None else p.fin_width
    base2 = np.array([[-L2/2,-W2/2],[L2/2,-W2/2],
                      [L2/2,W2/2],[-L2/2,W2/2]])
    g = rotation(p.global_deg)
    a = (base @ rotation(p.global_deg).T) + np.asarray(p.center1) @ g.T
    b = (base2 @ rotation(p.global_deg+p.relative_deg).T) + np.asarray(p.center2) @ g.T
    return [a, b]


def polygons_overlap(a, b, eps=1e-10):
    # SAT: touching edges are not considered overlap.
    for poly in (a, b):
        for edge in np.roll(poly, -1, axis=0)-poly:
            axis = np.array([-edge[1], edge[0]])
            va, vb = a @ axis, b @ axis
            if va.max() <= vb.min()+eps or vb.max() <= va.min()+eps:
                return False
    return True


def draw_geometry(p, path=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    p.validate()
    fig = plt.figure(figsize=(12, 5.1))
    ax = fig.add_subplot(121)
    ax.add_patch(Rectangle((-p.period/2,-p.period/2), p.period,p.period,
                           facecolor='whitesmoke',edgecolor='black',lw=1.8, label='periodic cell'))
    for i, poly in enumerate(fins(p)):
        ax.add_patch(Polygon(poly, closed=True, facecolor=f'C{i}',edgecolor='black',alpha=.8,
                             label=f'Fin {i+1}: {p.global_deg+(p.relative_deg if i else 0):g}°'))
        cen = np.mean(poly,axis=0)
        ax.annotate(f'{i+1}', cen,ha='center',va='center',fontsize=13,color='white')
    ax.set(aspect='equal',xlim=(-p.period*.6,p.period*.6),ylim=(-p.period*.6,p.period*.6),
           xlabel='x (µm)',ylabel='y (µm)',title='GDS top view (alumina on silica)')
    ax.legend(loc='upper left',fontsize=8)
    ax3 = fig.add_subplot(122,projection='3d')
    for i, poly in enumerate(fins(p)):
        bottom=np.column_stack((poly,np.zeros(4)))
        top=np.column_stack((poly,np.full(4,p.height)))
        for face in [top,bottom]+[np.array([bottom[j],bottom[(j+1)%4],top[(j+1)%4],top[j]]) for j in range(4)]:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            ax3.add_collection3d(Poly3DCollection([face],facecolor=f'C{i}',edgecolor='k',alpha=.7))
    ax3.set(xlim=(-p.period/2,p.period/2),ylim=(-p.period/2,p.period/2),
            zlim=(-.1,p.height+.05),xlabel='x (µm)',ylabel='y (µm)',zlabel='z (µm)',
            title='3D fin geometry; silica below z=0')
    ax3.set_box_aspect((1,1,1.1))
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=170,bbox_inches='tight')
    return fig


# Small dependency-free GDSII writer; all XY coordinates are in 1-nm database units.
def _record(tag, dtype, payload=b''):
    return struct.pack('>HBB',4+len(payload),tag,dtype)+payload


def _real8(x):
    if x == 0: return b'\0'*8
    sign = 0x80 if x<0 else 0
    x=abs(float(x)); exponent=64
    while x>=1: x/=16; exponent+=1
    while x<1/16: x*=16; exponent-=1
    mantissa=int(x*(1<<56)+.5)
    return bytes([sign|exponent])+mantissa.to_bytes(7,'big')


def _asciiz(s):
    b=s.encode('ascii'); return b + (b'\0' if len(b)%2 else b'')


def merged_fin_polygons(p: Config):
    """Union fin footprints, snapped to the GDSII 1-nm database grid.

    Returns one polygon for connected/overlapping fins, or multiple polygons
    for disconnected fins. Both fins share one material and etch height.
    """
    try:
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
    except ImportError as exc:
        raise ImportError("GDS export requires shapely: pip install shapely") from exc

    # Boolean union on integer-nanometer coordinates avoids sub-grid slivers.
    fin_shapes = [Polygon(np.rint(poly * 1000).astype(np.int64)) for poly in fins(p)]
    merged = unary_union(fin_shapes)
    if merged.is_empty or not merged.is_valid:
        raise ValueError("Invalid two-fin geometry after rounding to the GDS grid")
    components = [merged] if isinstance(merged, Polygon) else list(merged.geoms)
    if not all(isinstance(component, Polygon) for component in components):
        raise ValueError("Unexpected non-polygon result from GDS boolean union")
    if any(component.interiors for component in components):
        raise ValueError("GDS writer does not support holes; use a polygon-capable GDS library")
    return [np.asarray(component.exterior.coords[:-1]) / 1000.0 for component in components]


def write_gds(p, path, include_cell_outline=True):
    """Write the *union* of both alumina fins on GDS layer 1.

    Overlap is not double-counted. Disconnected fins remain separate boundaries.
    Layer 10 contains an optional reference outline of the periodic cell.
    """
    p.validate()
    now=datetime.datetime.now()
    stamp=[now.year,now.month,now.day,now.hour,now.minute,now.second]*2
    data=[_record(0,2,struct.pack('>h',600)),
          _record(1,2,struct.pack('>12h',*stamp)),
          _record(2,6,_asciiz('TWOFIN')),
          _record(3,5,_real8(1e-3)+_real8(1e-9)),
          _record(5,2,struct.pack('>12h',*stamp)),
          _record(6,6,_asciiz('ALUMINA_FINS'))]
    shapes=[(1,poly) for poly in merged_fin_polygons(p)]
    if include_cell_outline:
        a=p.period/2
        shapes.append((10,np.array([[-a,-a],[a,-a],[a,a],[-a,a]])))
    for layer,poly in shapes:
        coords=np.round(np.vstack([poly,poly[0]])*1000).astype(np.int32)
        data.extend([_record(8,0), _record(13,2,struct.pack('>h',layer)),
                     _record(14,2,struct.pack('>h',0)),
                     _record(16,3,coords.astype('>i4').tobytes()),_record(17,0)])
    data.extend([_record(7,0),_record(4,0)])
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b''.join(data))
    return path


def stokes(j):
    """Normalized Stokes [S0,S1,S2,S3] from coherent Jones vector."""
    j=np.asarray(j,dtype=complex).reshape(2)
    x,y=j
    s0=np.vdot(j,j).real
    if s0 < 1e-24: raise ValueError('Zero Jones vector has undefined Stokes state')
    return np.array([1., (abs(x)**2-abs(y)**2)/s0,
                     (2*(x.conjugate()*y).real)/s0,
                     (2*(x.conjugate()*y).imag)/s0])


def jones_from_stokes(s):
    s=np.asarray(s,dtype=float).reshape(4)
    if s[0]<=0: raise ValueError('S0 must be positive')
    q=s[1:]/s[0]
    if abs(np.linalg.norm(q)-1)>1e-6:
        raise ValueError('Single coherent FDTD excitation needs a fully polarized Stokes vector')
    # |x|²=(1+s1)/2; construct stable expression at the -S1 pole.
    if 1+q[0] > 1e-9:
        x=np.sqrt((1+q[0])/2)
        y=(q[1]+1j*q[2])/(2*x)
    else:
        x=0j; y=1+0j
    return np.array([x,y],complex)


def eigenpolarizations(M):
    eigenvalues,vecs=np.linalg.eig(np.asarray(M,dtype=complex))
    order=np.argsort(-abs(eigenvalues)); eigenvalues=eigenvalues[order]; vecs=vecs[:,order]
    states=[]
    for j in range(2):
        v=vecs[:,j]/np.linalg.norm(vecs[:,j])
        if abs(v[0])>1e-12: v*=np.exp(-1j*np.angle(v[0]))
        states.append(dict(eigenvalue=eigenvalues[j],jones=v,stokes=stokes(v)))
    return states


def poynting_split(e, h, n):
    """Recover +z/-z zeroth-order E amplitudes from mean tangential E/H."""
    ex,ey=np.asarray(e,complex); hx,hy=np.asarray(h,complex)
    plus=np.array([(ex+hy/n)/2, (ey-hx/n)/2])
    minus=np.array([(ex-hy/n)/2, (ey+hx/n)/2])
    return plus, minus


def assemble_s(experiments, key='modes'):
    """Each experiment has 4-vector of incoming/outgoing, stored by channel."""
    A=np.column_stack([c[key]['incoming'] for c in experiments])
    B=np.column_stack([c[key]['outgoing'] for c in experiments])
    if np.linalg.cond(A)>1e8: raise ValueError('Ill-conditioned incident-state basis')
    return B @ np.linalg.inv(A), A, B


def expected_fresnel(p):
    """Flat-interface power-normalized 4x4 S at z=lower_port and upper_port."""
    nb,nt=p.n_sub,1.0
    r=(nb-nt)/(nb+nt)
    t=2*np.sqrt(nb*nt)/(nb+nt)
    f=1/p.wavelength
    bphase=np.exp(1j*2*np.pi*f*nb*(0-p.lower_port))
    tphase=np.exp(1j*2*np.pi*f*nt*p.upper_port)
    I=np.eye(2)
    return np.block([[r*bphase**2*I, t*bphase*tphase*I],
                     [t*bphase*tphase*I,-r*tphase**2*I]])


def power_summary(S,port,input_jones):
    """Power fractions for one-sided unit-power input, zero-order channels only."""
    j=np.asarray(input_jones,complex).reshape(2); j/=np.linalg.norm(j)
    a=np.zeros(4,complex); a[2*port:2*port+2]=j
    out=np.asarray(S)@a
    return dict(reflectance=float(np.vdot(out[2*port:2*port+2],out[2*port:2*port+2]).real),
                transmittance=float(np.vdot(out[2*(1-port):2*(1-port)+2],out[2*(1-port):2*(1-port)+2]).real),
                loss=float(1-np.vdot(out,out).real),
                outgoing=out)


def geometric_phase(states):
    """Gauge-invariant Pancharatnam phase along a Jones path, geodesically closed.

    Phase = arg(prod <psi_i|psi_{i+1}> * <psi_final|psi_initial>).
    Expects no successive orthogonal states and no orthogonal endpoints.
    """
    v=np.array(states,dtype=complex)
    v/=np.linalg.norm(v,axis=1,keepdims=True)
    links=np.einsum('ij,ij->i',np.conj(v[:-1]),v[1:])
    closure=np.vdot(v[-1],v[0])
    if min(np.min(abs(links)),abs(closure))<1e-8:
        raise ValueError('Geodesic-closure phase is undefined at orthogonal points')
    return float(np.angle(np.prod(links/abs(links))*closure/abs(closure)))


def plot_poincare(states, labels=None, title='Poincaré sphere', path=None):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection='3d')
    u,v=np.meshgrid(np.linspace(0,2*np.pi,52),np.linspace(0,np.pi,28))
    ax.plot_wireframe(np.sin(v)*np.cos(u),np.sin(v)*np.sin(u),np.cos(v),
                      color='lightgray',lw=.45,alpha=.4,rstride=3,cstride=3)
    ax.plot([-1,1],[0,0],[0,0],c='gray',lw=.7)
    ax.plot([0,0],[-1,1],[0,0],c='gray',lw=.7)
    ax.plot([0,0],[0,0],[-1,1],c='gray',lw=.7)
    for i,(st,label) in enumerate(zip(states,labels or [f'State {k+1}' for k in range(len(states))])):
        p=np.asarray(st)
        if p.shape==(4,): p=p[1:]
        ax.scatter(*p,s=80,label=label)
        ax.text(*(1.06*p),label,fontsize=8)
    ax.set(xlabel='S1 / S0',ylabel='S2 / S0',zlabel='S3 / S0',title=title,
           xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1))
    ax.set_box_aspect((1,1,1))
    ax.legend(loc='upper left',fontsize=8)
    if path:
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=170,bbox_inches='tight')
    return fig


def plot_path_on_sphere(jones_path,title='Stokes transport',path=None):
    import matplotlib.pyplot as plt
    vecs=np.array([stokes(v)[1:] for v in jones_path])
    fig=plot_poincare([vecs[0],vecs[-1]],['start','end'],title,path=None)
    ax=fig.axes[0]; ax.plot(*vecs.T,lw=2.3,c='purple')
    if path: fig.savefig(path,dpi=170,bbox_inches='tight')
    return fig


def solid_angle_path(states):
    """Oriented solid angle of Stokes path plus shortest-geodesic closure.

    Fan triangulation with first point as anchor; reliable unless a triangle
    approaches an antipodal degeneracy.
    """
    q=np.array([stokes(j)[1:] for j in states])
    area=0.
    for i in range(1,len(q)-1):
        a,b,c=q[0],q[i],q[i+1]
        area+=2*np.arctan2(np.dot(a,np.cross(b,c)),1+np.dot(a,b)+np.dot(b,c)+np.dot(c,a))
    return float(area)


def rotate_jones(M,theta_deg):
    """Rotated-element model; EXACT only for an in-plane rotation-covariant environment."""
    Q=rotation(theta_deg)
    return Q @ np.asarray(M) @ Q.T


def format_matrix(M,precision=4):
    arr=np.asarray(M,complex)
    return '\n'.join('['+'  '.join(f'{x.real:+.{precision}f}{x.imag:+.{precision}f}j' for x in row)+']'
                     for row in arr)
