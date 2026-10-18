"""Native and quadrant zero-piston offsets, each scaled to unit D gain."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser


def analyze(grams, phase_count=1800):
    yB=yj(*state(grams['opaque'],kind='D'))[0];n=yB.shape[1]
    X=np.broadcast_to(np.arange(n)+.5-n/2,yB.shape);out={}
    for cmd in ('original','orthogonal'):
        if cmd not in grams: continue
        Q=grams[cmd];yD,jD=yj(*state(Q,kind='D'));yP=yj(*state(Q,kind='pair'))[0]
        for name,w in (('native',jD),('quadrant',np.sign(X))):
            w=w/np.sum(w*jD)
            b=[np.sum(w*(yj(*state(Q,p))[0]-yB)) for p in phases(phase_count)]
            out[f'{cmd}_{name}']={'coherent':[min(b),max(b)],'mid':(min(b)+max(b))/2,
                'half':(max(b)-min(b))/2,'pair':np.sum(w*(yP-yB)),'D':np.sum(w*(yD-yB))}
    return out

if __name__=='__main__':
    p=contraction_parser(__doc__,1800);a=p.parse_args();new_output(a.output)
    save_json(a.output,{'phases':a.phases,'results':analyze(load_grams(a.grams),a.phases)})
