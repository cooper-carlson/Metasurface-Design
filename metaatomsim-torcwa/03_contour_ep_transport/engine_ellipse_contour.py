"""Single-layer ellipse-base Fourier contour and fabrication-aware penalties.

All geometry is in micrometres. Torch operations retain autograd graphs.
One closed, simply connected island per periodic cell; no holes or layers.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class ContourRules:
    period: float = 0.400
    harmonics: int = 5
    angular_samples: int = 256
    # Cell-edge clearance and radius cap protect against +/- etch bias.
    cell_margin: float = 0.025
    perturbation_pad: float = 0.008
    min_neck_um: float = 0.075
    min_tip_radius_um: float = 0.020
    preferred_aspect: float = 7.0
    maximum_aspect: float = 10.0
    min_periodic_gap_um: float = 0.050
    edge_um: float = 0.004
    # Radius is maximum radial distance, after ellipse/Fourier normalization.
    radius_bounds: tuple = (0.082, 0.165)
    height_bounds: tuple = (0.180, 0.790)
    harmonic_amplitude: float = 0.30
    axis_ratio_bounds: tuple = (1.0, 3.5)  # major/minor radii

    def validate(self):
        if self.harmonics < 3 or self.angular_samples < 8*self.harmonics or self.angular_samples % 2:
            raise ValueError('Insufficient contour angular resolution')
        if self.period <= 0 or self.edge_um <= 0:
            raise ValueError('Invalid dimensions')
        if self.min_periodic_gap_um > 2*self.cell_margin:
            raise ValueError('Periodic gap target exceeds twice the guaranteed margin')
        allowed = self.period/2-self.cell_margin-self.perturbation_pad
        if not (0 < self.radius_bounds[0] < self.radius_bounds[1] <= allowed):
            raise ValueError('Radius bounds must fit periodic margin and perturbation pad')
        if self.height_bounds[1] >= 1.05:
            raise ValueError('Height reaches the fixed upper monitor plane')
        if self.axis_ratio_bounds[0] < 1 or self.axis_ratio_bounds[1] <= self.axis_ratio_bounds[0]:
            raise ValueError('Expected major/minor axis ratio >= 1')
        return self


def real_dtype(settings):
    import torch
    return torch.float64 if settings.precision == 'complex128' else torch.float32


def harmonic_modes(rules):
    """Second harmonic is encoded by the ellipse instead of redundantly."""
    return tuple(n for n in range(1, rules.harmonics+1) if n != 2)


class ContourDesign:
    """Ellipse base and Fourier deformations in the rotated local frame.

    raw = [radius, height, major/minor ratio, orientation, a_modes, b_modes].
    """
    def __init__(self, rules, initial=None, *, device='cpu', dtype=None):
        import torch
        rules.validate()
        self.rules = rules
        dtype = dtype or torch.float32
        n = len(harmonic_modes(rules))
        if initial is None:
            initial = np.zeros(4+2*n)
        self.raw = torch.nn.Parameter(torch.as_tensor(initial,device=device,dtype=dtype).clone())
        if self.raw.numel() != 4+2*n:
            raise ValueError(f'Expected {4+2*n} ellipse/Fourier variables')

    def parameters(self):
        return [self.raw]

    def values(self):
        import torch
        r,x = self.rules,self.raw
        radius = r.radius_bounds[0]+(r.radius_bounds[1]-r.radius_bounds[0])*torch.sigmoid(x[0])
        height = r.height_bounds[0]+(r.height_bounds[1]-r.height_bounds[0])*torch.sigmoid(x[1])
        ratio = r.axis_ratio_bounds[0]+(r.axis_ratio_bounds[1]-r.axis_ratio_bounds[0])*torch.sigmoid(x[2])
        orientation = (torch.pi/2)*torch.tanh(x[3])
        modes=torch.as_tensor(harmonic_modes(r),device=x.device,dtype=x.dtype)
        envelope=r.harmonic_amplitude/(modes**1.25)
        m=len(modes)
        a=envelope*torch.tanh(x[4:4+m])
        b=envelope*torch.tanh(x[4+m:4+2*m])
        # Lateral translation does not affect specular periodic S.
        zero=torch.zeros_like(radius)
        return dict(radius=radius,height=height,axis_ratio=ratio,
                    orientation=orientation,cx=zero,cy=zero,a=a,b=b)

    def export(self):
        return self.raw.detach().cpu().numpy().tolist()


def _angle_grid(rules, anchor):
    import torch
    return 2*torch.pi*torch.arange(rules.angular_samples,device=anchor.device,dtype=anchor.dtype)/rules.angular_samples


def _unscaled_radius(theta,p,rules,*,perturb=None):
    """Ellipse with smooth freeform deformation, in its own rotated frame."""
    import torch
    local = theta-p['orientation']
    q=torch.sqrt(p['axis_ratio'])
    major=p['radius']*q
    minor=p['radius']/q
    ellipse=major*minor/torch.sqrt((minor*torch.cos(local))**2+
                                    (major*torch.sin(local))**2+1e-24)
    modes=torch.as_tensor(harmonic_modes(rules),device=theta.device,dtype=theta.dtype)
    a,b=p['a'],p['b']
    if perturb is not None and 'harmonic_delta' in perturb:
        shift=torch.as_tensor(perturb['harmonic_delta'],device=theta.device,dtype=theta.dtype)
        if tuple(shift.shape)!=(2,len(modes)):
            raise ValueError('Perturbation Fourier shape mismatch')
        a,b=a+shift[0],b+shift[1]
    phases=local[...,None]*modes
    modulation=(a*torch.cos(phases)+b*torch.sin(phases)).sum(-1)
    return ellipse*torch.exp(modulation)


def _radius_at(theta,p,rules,*,perturb=None):
    """Normalize maximum radial size to the trainable radius; no scale saturation."""
    import torch
    raw=_unscaled_radius(theta,p,rules,perturb=perturb)
    highres=_angle_grid(rules,p['radius'])
    maxr=_unscaled_radius(highres,p,rules,perturb=perturb).max()
    # The optimized radius is the actual maximum radial extent, never a null variable.
    scale=p['radius']/(maxr+1e-12)
    radius=raw*scale
    if perturb is not None:
        bias=float(perturb.get('bias_um',0.))
        if abs(bias)>rules.perturbation_pad+1e-12:
            raise ValueError('Etch bias exceeds contour containment pad')
        radius=radius+bias
    return radius


def contour_points(p,rules,*,perturb=None,angles=None):
    import torch
    theta=_angle_grid(rules,p['radius']) if angles is None else angles
    radial=_radius_at(theta,p,rules,perturb=perturb)
    return torch.stack([p['cx']+radial*torch.cos(theta),
                        p['cy']+radial*torch.sin(theta)],-1)


def contour_fill(p,rules,raster,*,edge_um=None,perturb=None,hard=False):
    """2D periodic-cell fill, first dimension x, second y. Differentiable if soft."""
    import torch
    coords=(torch.arange(raster,device=p['radius'].device,dtype=p['radius'].dtype)+.5)*rules.period/raster-rules.period/2
    X,Y=torch.meshgrid(coords,coords,indexing='ij')
    dx,dy=X-p['cx'],Y-p['cy']
    theta=torch.atan2(dy+1e-12,dx+1e-12)
    radial=torch.sqrt(dx*dx+dy*dy+1e-16)
    boundary=_radius_at(theta,p,rules,perturb=perturb)
    if hard:
        return (boundary>=radial).to(radial.dtype)
    width=rules.edge_um if edge_um is None else float(edge_um)
    return torch.sigmoid((boundary-radial)/width)


def fabrication_cost(p,rules,*,perturb=None):
    """Differentiable fabrication *proxies*, not a process-qualified DRC.

    Opposite-ray widths miss some necks. Hard-mask opening/closing and
    calibrated process checks must be performed on the exported layout.
    """
    import torch
    th=_angle_grid(rules,p['radius'])
    radial=_radius_at(th,p,rules,perturb=perturb)
    height=p['height']+(float(perturb.get('height_um',0.)) if perturb is not None else 0.)
    r_opposite=radial+torch.roll(radial,rules.angular_samples//2)
    # Differentiable soft minimum avoids zero gradients from a single sample.
    t=0.003
    neck=-t*torch.logsumexp(-r_opposite/t,dim=0)
    neck=torch.clamp(neck,min=1e-5)
    aspect=height/neck
    # Finite differences w.r.t. theta approximate geometric curvature.
    dth=2*math.pi/rules.angular_samples
    dr=(torch.roll(radial,-1)-torch.roll(radial,1))/(2*dth)
    d2r=(torch.roll(radial,-1)-2*radial+torch.roll(radial,1))/dth**2
    curvature=torch.sqrt((radial**2+2*dr**2-radial*d2r)**2+1e-18)/(
        (radial**2+dr**2+1e-12)**1.5)
    # Penalize too-small curvature radius (too-large curvature).
    curvature_violation=torch.relu(rules.min_tip_radius_um*curvature-1.)
    width_violation=torch.relu((rules.min_neck_um-neck)/rules.min_neck_um)
    aspect_violation=torch.relu((aspect-rules.preferred_aspect)/rules.preferred_aspect)
    curvature_penalty=torch.mean(curvature_violation**2)
    # Tiny regularizer discourages unnecessarily wiggly masks.
    n=torch.as_tensor(harmonic_modes(rules),device=radial.device,dtype=radial.dtype)
    complexity=((n**2*p['a'])**2+(n**2*p['b'])**2).mean()
    components=dict(neck_proxy_um=neck,aspect_proxy=aspect,
       max_curvature=torch.max(curvature),width_penalty=width_violation**2,
       aspect_penalty=aspect_violation**2+.02*(aspect/rules.preferred_aspect)**2,curvature_penalty=curvature_penalty,
       complexity_penalty=complexity)
    hard_aspect=torch.relu(aspect/rules.maximum_aspect-1.)**2
    components['max_aspect_penalty']=hard_aspect
    total=3*components['width_penalty']+2*components['aspect_penalty']+\
          1.5*curvature_penalty+12*hard_aspect+.015*complexity
    return total,components


def periodic_margin(p,rules):
    import torch
    xy=contour_points(p,rules)
    maxcoord=torch.max(torch.abs(xy),dim=0).values
    return rules.period/2-maxcoord


def hard_mask_checks(mask,rules):
    """Heuristic morphological fabrication checks; no absolute width certificate."""
    import scipy.ndimage as ndi
    mask=np.asarray(mask,dtype=bool)
    px=rules.period/mask.shape[0]
    rad=max(1,int(np.ceil(rules.min_neck_um/2/px)))
    yy,xx=np.ogrid[-rad:rad+1,-rad:rad+1]
    footprint=(xx*xx+yy*yy)<=rad**2
    opened=ndi.binary_opening(mask,structure=footprint,border_value=0)
    area=max(int(mask.sum()),1)
    opened_loss=float((mask & ~opened).sum()/area)
    connected_components=ndi.label(mask)[1]
    return dict(opening_area_loss=opened_loss,components=int(connected_components),
      min_border_gap_um=2*min(np.where(mask)[0].min(),mask.shape[0]-1-np.where(mask)[0].max(),
                             np.where(mask)[1].min(),mask.shape[1]-1-np.where(mask)[1].max())*px if mask.any() else 0.,
      mask_area_um2=float(mask.sum()*px**2))


def save_contour_svg(path,coords,period):
    """Simple vector mask preview; SVG is not a calibrated lithography file."""
    from pathlib import Path
    xy=np.asarray(coords,float)
    pixel=900/period
    pts=' '.join(f'{(x+period/2)*pixel:.2f},{(period/2-y)*pixel:.2f}' for x,y in xy)
    data=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 900">'
          f'<rect width="900" height="900" fill="white" stroke="black"/>'
          f'<polygon points="{pts}" fill="black"/></svg>')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(data)
    return path


def save_contour_gds(path,coords,period):
    """Optional actual GDS output; requires gdstk (not needed for optimization)."""
    import gdstk
    from pathlib import Path
    cell=gdstk.Cell('CONTOUR_EP')
    cell.add(gdstk.Polygon(np.asarray(coords,float),layer=1))
    cell.add(gdstk.rectangle((-period/2,-period/2),(period/2,period/2),layer=10))
    lib=gdstk.Library(unit=1e-6,precision=1e-9)
    lib.add(cell)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lib.write_gds(str(path))
    return path
