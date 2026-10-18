"""One assumed unswitched leakage shape alpha*exp(i*psi)*G; sampled extrema only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser


def analyze(grams, strip_phases=90, leakage_phases=72, amplitudes=(0.,.003,.01,.03)):
    Q=grams['original'];gg=Q[...,0,0].real;gk=Q[...,0,1];ss=Q[...,2,2].real
    gs=Q[...,0,2];ks=Q[...,1,2]
    jB=yj(*state(grams['opaque'],kind='D'))[1];jD=yj(gg,2*gk.real)[1]
    w=jD/np.sum(jD*jD);gB=np.sum(w*jB);out={}
    for a in amplitudes:
        if not np.isfinite(a) or a<0: raise ValueError('Leakage amplitudes must be finite and nonnegative')
        paired=[];coherent=[]
        for psi in phases(leakage_phases):
            z=1+a*np.exp(1j*psi);j0=2*(np.conj(z)*gk).real
            paired.append(100*(gB/np.sum(w*yj(abs(z)**2*gg+ss,j0)[1])-1))
            for phi in phases(strip_phases):
                e=np.exp(1j*phi);I=abs(z)**2*gg+ss+2*(np.conj(z)*e*gs).real;J=j0+2*(e*ks).real
                coherent.append(100*(gB/np.sum(w*yj(I,J)[1])-1))
        out[str(a)]={'pair':[min(paired),max(paired)],'coh':[min(coherent),max(coherent)]}
    return out

if __name__=='__main__':
    p=contraction_parser(__doc__,90);p.add_argument('--leakage-phases',type=int,default=72)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[0.,.003,.01,.03]);a=p.parse_args();new_output(a.output)
    save_json(a.output,{'strip_phases':a.phases,'leakage_phases':a.leakage_phases,
        'amplitudes':a.amplitudes,'results':analyze(load_grams(a.grams),a.phases,a.leakage_phases,a.amplitudes)})
