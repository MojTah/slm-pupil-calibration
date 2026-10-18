"""Area-weighted or hard relay iris; explicit radius and new output directory.

Fractional edge cells use 32x32 midpoint samples (not an analytic area integral).
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, save_npz, new_output

import argparse
from simulation import Configuration
from modal_simulation import ModalModel, build_modal_basis
from orthogonal_carrier import OrthogonalStripModel
from piston_basis import readout_matrices
from reproduce_6d import summarize

def area_weights(f,radius,carrier,sub=32):
    fx=f[None,:]-carrier;fy=f[:,None];h=abs(f[1]-f[0]);r=np.hypot(fx,fy)
    w=(r<=radius).astype(float);edge=np.abs(r-radius)<h;off=(np.arange(sub)+.5)/sub-.5
    ox,oy=np.meshgrid(off*h,off*h)
    for y,x in zip(*np.nonzero(edge)):
        w[y,x]=np.mean((fx[0,x]+ox)**2+(fy[y,0]+oy)**2<=radius**2)
    return w

def soften(model):
    g=model.grid;cache={}
    def iris(radius_lam_over_d,carrier=0.):
        key=(radius_lam_over_d,carrier)
        if key not in cache: cache[key]=area_weights(g.f,radius_lam_over_d/g.config.diameter_pixels,carrier)
        return cache[key]
    g.iris=iris
    model.envelope_support=(iris(model.iris_radius)>0)|np.roll(iris(model.iris_radius,g.config.bright_carrier)>0,-model.carrier_index,axis=1)
    return model

def run(readout,output,domains,angles=128,radius=8.,hard=False,orthogonal=False,canary=False):
    with np.load(readout,allow_pickle=False) as f: w=f['weights']
    if not np.isfinite(radius) or radius<=0: raise ValueError('Radius must be positive and finite')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    for L in domains:
        cfg=Configuration(diameter_pixels=8 if canary else 128,spider_pixels=2 if canary else 8,samples_per_pixel=1,
            domain_diameters=L,bright_carrier=.125,dark_carrier=-.125,pupil_center_diameters=.75)
        make=(lambda m:m) if hard else soften
        mats={}
        for name,model in [('original',ModalModel)]+([('orthogonal',OrthogonalStripModel)] if orthogonal else []):
            mats[name]=readout_matrices(build_modal_basis(make(model(cfg,radius)),'SLM',.73 if canary else 3.,angles,4)['gram'],w)
        opaque=readout_matrices(build_modal_basis(make(ModalModel(cfg,radius)),'B',.73 if canary else 3.,angles,4)['gram'],w)
        identity=f'{"hard" if hard else "area"}_R{radius:.12g}_L{L}_A{angles}'
        save_npz(output/f'mats_{identity}.npz',opaque=opaque,**mats)
        for row in summarize(mats,opaque)['rows']:
            results.append({'iris':'hard' if hard else 'area','radius':radius,'domain':L,'angles':angles,**row})
    save_json(output/'summary.json',{'canary':canary,'rows':results})
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('readout',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--domains',type=int,nargs='+',default=[6,8,10,12,15])
    p.add_argument('--angles',type=int,default=128);p.add_argument('--radius',type=float,default=8.)
    p.add_argument('--hard',action='store_true');p.add_argument('--orthogonal',action='store_true');p.add_argument('--canary',action='store_true')
    a=p.parse_args();run(a.readout,a.output,a.domains,a.angles,a.radius,a.hard,a.orthogonal,a.canary)

if __name__=='__main__': main()
