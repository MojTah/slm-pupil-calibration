"""One-parameter geometry sweep; each geometry has its own native D readout."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj,state,phases,load_grams,save_json,new_output,contraction_parser

import argparse
from simulation import Configuration
from modal_simulation import ModalModel,build_modal_basis
from orthogonal_carrier import OrthogonalStripModel

def analyze(D,strip,q,radius,angles=128,phase_count=360,modulation=3.):
    cfg=Configuration(diameter_pixels=D,spider_pixels=strip,samples_per_pixel=1,domain_diameters=6,
                      bright_carrier=q,dark_carrier=-q,pupil_center_diameters=.75)
    B=build_modal_basis(ModalModel(cfg,radius),'B',modulation,angles,4)['gram'];jB=yj(*state(B,kind='D'))[1]
    out={}
    for cmd,M in (('original',ModalModel),('orthogonal',OrthogonalStripModel)):
        Q=build_modal_basis(M(cfg,radius),'SLM',modulation,angles,4)['gram']
        jD=yj(*state(Q,kind='D'))[1];jP=yj(*state(Q,kind='pair'))[1];w=jD/np.sum(jD*jD);gB=np.sum(w*jB)
        gains=[];sb=[]
        for p in phases(phase_count):
            j=yj(*state(Q,p))[1];gains.append(np.sum(w*j));sb.append(100*(np.sum(j*jB)/np.sum(j*j)-1))
        errors=100*(gB/np.array(gains)-1);gg=Q[...,0,0].real;ss=Q[...,2,2].real
        out[cmd]={'pS_pct':100*ss.sum()/(gg+ss).sum(),'BD_pct':100*(gB/np.sum(w*jD)-1),
                  'coh':[errors.min(),errors.max()],'pair':100*(gB/np.sum(w*jP)-1),
                  'statebuilt':[min(sb),max(sb)],'pairbuilt':100*(np.sum(jP*jB)/np.sum(jP*jP)-1)}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('name');p.add_argument('diameter',type=int)
    p.add_argument('strip',type=int);p.add_argument('carrier',type=float);p.add_argument('radius',type=float)
    p.add_argument('--angles',type=int,default=128);p.add_argument('--phases',type=int,default=360)
    p.add_argument('--modulation',type=float,default=3.);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();new_output(a.output)
    save_json(a.output,{'name':a.name,'D':a.diameter,'strip':a.strip,'q':a.carrier,'R':a.radius,
        'angles':a.angles,'phases':a.phases,'modulation':a.modulation,
        'results':analyze(a.diameter,a.strip,a.carrier,a.radius,a.angles,a.phases,a.modulation)})
