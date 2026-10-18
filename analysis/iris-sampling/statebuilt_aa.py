"""Hard and area-weighted iris comparison for state-built readouts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, save_npz, new_output

import argparse
from aa_iris import soften
from simulation import Configuration
from modal_simulation import ModalModel, build_modal_basis

def analyze(canary=False,phase_count=720):
    cfg=Configuration(diameter_pixels=8 if canary else 128,spider_pixels=2 if canary else 8,samples_per_pixel=1,
        domain_diameters=6,bright_carrier=.125,dark_carrier=-.125,pupil_center_diameters=.75)
    out={}
    for name,make in (('hard',lambda m:m),('area',soften)):
        args=(1. if canary else 8.,.73 if canary else 3.,8 if canary else 256)
        C=build_modal_basis(make(ModalModel(cfg,args[0])),'SLM',args[1],args[2],4)['gram']
        B=build_modal_basis(make(ModalModel(cfg,args[0])),'B',args[1],args[2],4)['gram']
        jB=yj(*state(B,kind='D'))[1];errors=[]
        for p in phases(phase_count):
            j=yj(*state(C,p))[1];errors.append(100*(np.sum(j*jB)/np.sum(j*j)-1))
        jP=yj(*state(C,kind='pair'))[1]
        out[name]={'coherent':[min(errors),max(errors)],'pair':100*(np.sum(jP*jB)/np.sum(jP*jP)-1)}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phases',type=int,default=720);p.add_argument('--canary',action='store_true')
    a=p.parse_args();new_output(a.output);save_json(a.output,{'canary':a.canary,'phases':a.phases,'results':analyze(a.canary,a.phases)})
