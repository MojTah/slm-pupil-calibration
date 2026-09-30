"""Plot phase-pair tolerances and frozen domain gains; no optical propagation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def coherence(imbalance, error):
    return np.sqrt(imbalance**2 + (1-imbalance**2)*np.sin(error/2)**2)


def main():
    # Exact limits and the complex two-exposure definition independently agree.
    d, e = np.meshgrid(np.linspace(0, .3, 301), np.linspace(0, np.pi/3, 301))
    direct = np.abs((1+d)/2 - (1-d)/2*np.exp(1j*e))
    np.testing.assert_allclose(coherence(d, e), direct, atol=1e-14)
    np.testing.assert_allclose(coherence(d, 0), d, atol=1e-14)
    np.testing.assert_allclose(coherence(0, e), np.sin(e/2), atol=1e-14)
    source=ROOT/'local-results/referee-revision-20260928/domain-sensitivity.json'
    records=json.loads((ROOT/'DATA_MANIFEST.json').read_text())
    expected=next(r['sha256'] for r in records if r['path']==source.relative_to(ROOT).as_posix())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    data=json.loads(source.read_text())['rows']
    for row in data:
        assert abs(row['gB']/row['gPair']-1-row['ePair']) < 1e-14
    out=ROOT/'local-results/publication-figures-v0.1.0'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9, 'axes.titlesize':9, 'legend.fontsize':8, 'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(5.7,3.4),layout='constrained')
    k=coherence(d,e)
    im=ax.pcolormesh(d,np.rad2deg(e),k,cmap='cividis',shading='auto',vmin=0,vmax=.55,rasterized=True)
    levels=[.05,.1,.2,.3,.4,.5]
    cs=ax.contour(d,np.rad2deg(e),k,levels=levels,colors='white',linewidths=.9)
    ax.clabel(cs,fmt='%.2f',fontsize=8)
    ax.set(xlabel=r'Raw exposure imbalance $|d|$',ylabel=r'Phase-step error $|\varepsilon|$ (deg)')
    fig.colorbar(im,ax=ax,label=r'Residual coherence $\kappa$')
    for ext in ('pdf','png'): fig.savefig(out/f'pair-tolerances.{ext}',dpi=240)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(6.5,3.0),layout='constrained')
    x=[r['domain'] for r in data]
    styles=[('gB','Opaque target B','#176b87','o','-'),('gPair','Paired SLM','#b05d20','s','--'),
            ('gCmin','SLM minimum gain','#804d9e','^',':'),('gCmax','SLM maximum gain','#555555','v','-.')]
    for key,label,color,marker,ls in styles:
        axes[0].plot(x,[100*(r[key]/data[0][key]-1) for r in data],label=label,color=color,marker=marker,ls=ls,ms=4)
    axes[0].set(title='(a) Common gain change',ylabel='Change from own 6D value (%)')
    axes[0].legend(loc='lower left')
    axes[1].plot(x,[100*r['ePair'] for r in data],color='#176b87',marker='o')
    axes[1].set(title='(b) Opaque-target paired transfer',ylabel='Local scale error (%)',ylim=(.079,.081))
    for ax in axes:
        ax.set(xlabel='Domain side L / D',xticks=x)
        ax.grid(alpha=.2)
    for ext in ('pdf','png'): fig.savefig(out/f'domain-gains.{ext}',dpi=240)
    plt.close(fig)
    (out/'verification.json').write_text(json.dumps({'checks':'PASS: coherence identity, limits, data hash and gain ratio',
        'domain_input_sha256':expected,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(out)


if __name__=='__main__': main()
