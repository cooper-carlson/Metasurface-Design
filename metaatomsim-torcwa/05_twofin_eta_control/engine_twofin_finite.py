"""Optional 3D finite-array Meep field characterization with lateral and vertical PML.

The near-field traces are observations of Maxwell fields, not local propagation
operators or biorthogonal Wilson phases.
"""
from dataclasses import dataclass
import math
import numpy as np
from engine_twofin_eta import numpy_rectangles

@dataclass(frozen=True)
class FiniteSettings:
    resolution: int = 40         # pixels / micron; 3D runs can be expensive
    pml_um: float = 0.35
    side_clearance_um: float = 0.35
    bottom_um: float = 0.80
    top_um: float = 0.80
    run_time: float = 160.0
    sample_z_um: tuple = (-0.12, 0.05, 0.15, 0.30, 0.48)


def prisms_for_array(u_grid,theta_grid,p,mod,physics):
    """Return connected rectangle-prism placements in a finite MxM array."""
    from engine_twofin_eta import vary
    uu=np.asarray(u_grid); tt=np.asarray(theta_grid)
    if uu.shape!=tt.shape or uu.ndim!=2:raise ValueError('Matched 2D grids expected')
    m,n=uu.shape; polys=[]
    for i in range(m):
        for j in range(n):
            pp={k:float(v) for k,v in vary(p,mod,float(uu[i,j])).items()}
            center=np.array([(i-(m-1)/2)*physics.period,
                             (j-(n-1)/2)*physics.period])
            for corner in numpy_rectangles(pp,theta_deg=float(tt[i,j])):
                polys.append(corner+center)
    return polys


def run_finite_meep(u_grid,theta_grid,p,mod,physics,opts=FiniteSettings(),
                    input_pol='x'):
    """Monochromatic DFT E/H fields sampled at fixed z in a single finite layer.

    No periodic boundaries in x/y; all six external directions are PML-terminated.
    Source uses a finite planar aperture; compare to a bare substrate run for
    absolute scattering efficiencies (not provided by this field-only routine).
    """
    try: import meep as mp
    except ImportError as e:raise ImportError('Install Meep in your FDTD environment') from e
    uu=np.asarray(u_grid,float);tt=np.asarray(theta_grid,float)
    m,n=uu.shape
    lx=m*physics.period+2*(opts.side_clearance_um+opts.pml_um)
    ly=n*physics.period+2*(opts.side_clearance_um+opts.pml_um)
    zlo=-opts.bottom_um-opts.pml_um
    zhi=physics.height+opts.top_um+opts.pml_um
    lz=zhi-zlo; zcenter=(zhi+zlo)/2
    # Center all z coordinates relative to the simulation cell origin.
    to_mp_z=lambda z:z-zcenter
    geometry=[]
    substrate_z=(-opts.bottom_um-opts.pml_um)/2
    geometry.append(mp.Block(
        size=mp.Vector3(lx,ly,opts.bottom_um+opts.pml_um),
        center=mp.Vector3(0,0,to_mp_z(substrate_z)),
        material=mp.Medium(index=physics.n_bottom)))
    if physics.n_background != physics.n_top:
        geometry.append(mp.Block(
            size=mp.Vector3(lx,ly,physics.height),
            center=mp.Vector3(0,0,to_mp_z(physics.height/2)),
            material=mp.Medium(index=physics.n_background)))
    if physics.k_fin:raise NotImplementedError('Lossy fins: give Meep a dispersive/conductive medium first')
    medium=mp.Medium(index=physics.n_fin)
    for poly in prisms_for_array(uu,tt,p,mod,physics):
        c=np.mean(poly,axis=0)
        v0=poly[1]-poly[0];v1=poly[3]-poly[0]
        width1=np.linalg.norm(v0);width2=np.linalg.norm(v1)
        e1=v0/width1;e2=v1/width2
        geometry.append(mp.Block(
            size=mp.Vector3(width1,width2,physics.height),
            center=mp.Vector3(float(c[0]),float(c[1]),to_mp_z(physics.height/2)),
            e1=mp.Vector3(float(e1[0]),float(e1[1]),0),
            e2=mp.Vector3(float(e2[0]),float(e2[1]),0),
            material=medium))
    f=1/physics.wavelength
    if input_pol not in ('x','y'):raise ValueError('input_pol must be x or y')
    src=mp.Ex if input_pol=='x' else mp.Ey
    aperture_x=m*physics.period+0.1;aperture_y=n*physics.period+0.1
    source=[mp.Source(mp.GaussianSource(frequency=f,fwidth=f/12),component=src,
        center=mp.Vector3(0,0,to_mp_z(-0.42)),
        size=mp.Vector3(aperture_x,aperture_y,0))]
    sim=mp.Simulation(cell_size=mp.Vector3(lx,ly,lz),geometry=geometry,
        sources=source,boundary_layers=[mp.PML(opts.pml_um)],
        resolution=opts.resolution,default_material=mp.Medium(index=physics.n_top))
    fields=[mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz]
    dfts=[]
    for z in opts.sample_z_um:
        if not(-opts.bottom_um < z < physics.height+opts.top_um):
            raise ValueError(f'Field sample z={z} is outside the physical domain')
        region=mp.Volume(center=mp.Vector3(0,0,to_mp_z(z)),
                         size=mp.Vector3(m*physics.period,n*physics.period,0))
        dfts.append(sim.add_dft_fields(fields,f,0,1,where=region))
    sim.run(until=opts.run_time)
    data={}
    for z,h in zip(opts.sample_z_um,dfts):
        data[float(z)]={field_name:np.squeeze(np.asarray(sim.get_dft_array(h,c,0)))
                        for field_name,c in zip(('Ex','Ey','Ez','Hx','Hy','Hz'),fields)}
    sim.reset_meep()
    return data


def field_diagnostics(result):
    """Flux-weighted output polarization and transverse Poynting fractions."""
    output=[]
    for z,field in sorted(result.items()):
        Ex,Ey,Ez=field['Ex'],field['Ey'],field['Ez']
        Hx,Hy,Hz=field['Hx'],field['Hy'],field['Hz']
        Sx=0.5*np.real(Ey*np.conj(Hz)-Ez*np.conj(Hy))
        Sy=0.5*np.real(Ez*np.conj(Hx)-Ex*np.conj(Hz))
        Sz=0.5*np.real(Ex*np.conj(Hy)-Ey*np.conj(Hx))
        power=np.abs(Ex)**2+np.abs(Ey)**2
        denom=np.sum(power)+1e-30
        s1=np.sum((np.abs(Ex)**2-np.abs(Ey)**2))/denom
        s2=2*np.real(np.sum(np.conj(Ex)*Ey))/denom
        s3=2*np.imag(np.sum(np.conj(Ex)*Ey))/denom
        flux_lateral=np.sum(np.sqrt(Sx**2+Sy**2))
        flux_z=np.sum(np.abs(Sz))+1e-30
        output.append(dict(z_um=z,s1=s1,s2=s2,s3=s3,
                           lateral_to_longitudinal_flux=float(flux_lateral/flux_z)))
    return output
