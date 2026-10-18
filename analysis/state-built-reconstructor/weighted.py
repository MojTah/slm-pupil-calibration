"""Intensity-weighted state-built readout; explicit relative intensity floor."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser


def analyze(grams, phase_count=720, floor=1e-6):
    if not 0 < floor < 1: raise ValueError('Require relative intensity floor in (0,1)')
    jB=yj(*state(grams['opaque'],kind='D'))[1];out={}
    for cmd in ('original','orthogonal'):
        if cmd not in grams: continue
        errors=[]
        for p in phases(phase_count):
            y,j=yj(*state(grams[cmd],p));mask=y>y.max()*floor
            w=np.where(mask,j/np.where(mask,y,1),0)
            errors.append(100*(np.sum(w*jB)/np.sum(w*j)-1))
        out[cmd]=[min(errors),max(errors)]
    return out

if __name__=='__main__':
    p=contraction_parser(__doc__);p.add_argument('--floor',type=float,default=1e-6);a=p.parse_args();new_output(a.output)
    save_json(a.output,{'phases':a.phases,'floor':a.floor,'results':analyze(load_grams(a.grams),a.phases,a.floor)})
