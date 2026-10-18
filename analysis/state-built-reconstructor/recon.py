"""Full-image fixed and state-built readout contractions; sampled phase extrema."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser


def analyze(grams, phase_count=720, fixed_phase_count=1800):
    jB=yj(*state(grams['opaque'],kind='D'))[1]
    n=jB.shape[1]; X=np.broadcast_to(np.arange(n)+.5-n/2,jB.shape)
    out={}
    for cmd in ('original','orthogonal'):
        if cmd not in grams: continue
        Q=grams[cmd]; jD=yj(*state(Q,kind='D'))[1]; jP=yj(*state(Q,kind='pair'))[1]
        def fixed(w):
            gB=np.sum(w*jB)
            gains=np.array([np.sum(w*yj(*state(Q,p))[1]) for p in phases(fixed_phase_count)])
            errors=100*(gB/gains-1)
            return [errors.min(),errors.max(),100*(gB/np.sum(w*jP)-1)]
        es=[]
        for p in phases(phase_count):
            j=yj(*state(Q,p))[1]; es.append(100*(np.sum(j*jB)/np.sum(j*j)-1))
        out[cmd]={'quadrant_fixed':fixed(np.sign(X)), 'Dbuilt_native_fixed':fixed(jD),
                  'pairbuilt_fixed':fixed(jP), 'Cbuilt_selfconsistent_coherent':[min(es),max(es)],
                  'pairbuilt_selfconsistent_pair':100*(np.sum(jP*jB)/np.sum(jP*jP)-1),
                  'Dbuilt_selfconsistent':100*(np.sum(jD*jB)/np.sum(jD*jD)-1)}
    return out

if __name__=='__main__':
    p=contraction_parser(__doc__);p.add_argument('--fixed-phases',type=int,default=1800);a=p.parse_args()
    new_output(a.output)
    save_json(a.output,{'phases':a.phases,'fixed_phases':a.fixed_phases,'results':analyze(load_grams(a.grams),a.phases,a.fixed_phases)})
