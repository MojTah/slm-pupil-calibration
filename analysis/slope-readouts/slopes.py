"""Global/local pyramid slope maps and their piston derivatives."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj,state,phases,load_grams,save_json,new_output,contraction_parser


def slopes(I,J,mode,offset=96,pupil_radius=64,mask_radius=64.):
    n=I.shape[0];c=n/2;R=pupil_radius
    if I.shape!=J.shape or I.shape!=(n,n): raise ValueError('Require matching square detector images')
    if R<=0 or offset<R or c-offset-R<0 or c+offset+R>n: raise ValueError('Quadrant windows lie outside detector or overlap')
    if not 0<mask_radius<=R: raise ValueError('Mask radius must lie in (0,pupil radius]')
    sub=np.arange(2*R)+.5-R;YY,XX=np.meshgrid(sub,sub,indexing='ij');mask=YY**2+XX**2<=mask_radius**2
    if not mask.any(): raise ValueError('Mask has no pixels')
    def quad(img,sy,sx): return img[int(c+sy*offset-R):int(c+sy*offset+R),int(c+sx*offset-R):int(c+sx*offset+R)]
    q={(a,b):quad(I,a,b) for a in (-1,1) for b in (-1,1)}
    dq={(a,b):quad(J,a,b) for a in (-1,1) for b in (-1,1)}
    Nx=q[-1,1]+q[1,1]-q[-1,-1]-q[1,-1];Ny=q[1,-1]+q[1,1]-q[-1,-1]-q[-1,1]
    dNx=dq[-1,1]+dq[1,1]-dq[-1,-1]-dq[1,-1];dNy=dq[1,-1]+dq[1,1]-dq[-1,-1]-dq[-1,1]
    S=sum(q.values());dS=sum(dq.values())
    if mode=='global': S=np.full_like(S,S[mask].mean());dS=np.full_like(dS,dS[mask].mean())
    elif mode!='local': raise ValueError('Unknown normalization')
    if np.any(S[mask]<=0): raise ValueError('Slope normalization needs positive intensity in selected mask')
    # Divide only within the mask; unused dark pixels need no artificial floor.
    return (np.concatenate([Nx[mask]/S[mask],Ny[mask]/S[mask]]),
            np.concatenate([(dNx[mask]*S[mask]-Nx[mask]*dS[mask])/S[mask]**2,
                            (dNy[mask]*S[mask]-Ny[mask]*dS[mask])/S[mask]**2]))

def analyze(grams,phase_count=720,offset=96,pupil_radius=64,mask_radius=64.,offsets=False):
    out={}
    for cmd in ('original','orthogonal'):
        if cmd not in grams: continue
        Q=grams[cmd];res={}
        for mode in ('global','local'):
            def project(I,J): return slopes(I,J,mode,offset,pupil_radius,mask_radius)
            sB,jB=project(*state(grams['opaque'],kind='D'));sD,jD=project(*state(Q,kind='D'))
            sP,jP=project(*state(Q,kind='pair'));w=jD/np.dot(jD,jD)
            if offsets:
                bc=[np.dot(w,project(*state(Q,p))[0]-sB) for p in phases(phase_count)]
                res[mode]={'coh':[min(bc),max(bc)],'pair':np.dot(w,sP-sB),'D':np.dot(w,sD-sB)}
            else:
                gB=np.dot(w,jB);gC=[];sb=[]
                for p in phases(phase_count):
                    j=project(*state(Q,p))[1];gC.append(np.dot(w,j));sb.append(100*(np.dot(j,jB)/np.dot(j,j)-1))
                errors=100*(gB/np.array(gC)-1)
                res[mode]={'fixed_coh':[errors.min(),errors.max()],'fixed_pair':100*(gB/np.dot(w,jP)-1),
                    'D_target_pair':100*(np.dot(w,jD)/np.dot(w,jP)-1),'BD':100*(gB/np.dot(w,jD)-1),
                    'statebuilt_coh':[min(sb),max(sb)],'pairbuilt':100*(np.dot(jP,jB)/np.dot(jP,jP)-1)}
        out[cmd]=res
    return out

def main(offsets=False):
    p=contraction_parser(__doc__);p.add_argument('--offset',type=int,default=96)
    p.add_argument('--pupil-radius',type=int,default=64);p.add_argument('--mask-radius',type=float,default=64.)
    a=p.parse_args();new_output(a.output)
    save_json(a.output,{'phases':a.phases,'offset':a.offset,'pupil_radius':a.pupil_radius,'mask_radius':a.mask_radius,
        'results':analyze(load_grams(a.grams),a.phases,a.offset,a.pupil_radius,a.mask_radius,offsets)})

if __name__=='__main__': main()
