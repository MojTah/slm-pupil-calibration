"""Sampled-reference offsets from a fresh reproduce_6d.py output bundle."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser

import argparse

def analyze(bundle, phase_count=7200):
    with np.load(bundle/'readout.npz',allow_pickle=False) as f: w=f['weights']
    with np.load(bundle/'opaque.npz',allow_pickle=False) as f: B=f['gram']
    def moments(Q): return Q.sum(axis=(0,1)),np.einsum('yx,yxij->ij',w,Q)
    FB,MB=moments(B);rB=(MB[0,0]/FB[0,0]).real;out={};ph=phases(phase_count)
    for cmd in ('original','orthogonal'):
        with np.load(bundle/f'{cmd}.npz',allow_pickle=False) as f: F,M=moments(f['gram'])
        gD=(2*M[0,1].real*F[0,0].real-M[0,0].real*2*F[0,1].real)/F[0,0].real**2
        if not np.isfinite(gD) or gD==0: raise ValueError('D gain must be finite and nonzero')
        coherent=((M[0,0]+M[2,2]).real+2*(np.exp(1j*ph)*M[0,2]).real)/((F[0,0]+F[2,2]).real+2*(np.exp(1j*ph)*F[0,2]).real)-rB
        raw={'cmin':coherent.min(),'cmax':coherent.max(),
             'pair':((M[0,0]+M[2,2])/(F[0,0]+F[2,2])).real-rB,'D':(M[0,0]/F[0,0]).real-rB}
        out[cmd]={'gD':gD,'raw':raw,'scaled':{k:v/gD for k,v in raw.items()}}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phases',type=int,default=7200)
    a=p.parse_args();new_output(a.output)
    save_json(a.output,{'phases':a.phases,'results':analyze(a.bundle,a.phases)})
