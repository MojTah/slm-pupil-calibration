"""Plot the verified phase-pair envelope packet without changing its results."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    start=time.perf_counter();folder=ROOT/'local-results/phase-pair-20260915'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    result=json.loads((folder/'result.json').read_text());assert result['status']=='COMPLETE_ANALYSIS'
    out=(ROOT/args.output_dir).resolve() if args.output_dir else folder/'figures'
    if not out.is_relative_to((ROOT/'local-results').resolve()):raise ValueError('Outside study results')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(6.5,3.4),layout='constrained')
    summaries={}
    for target,color,label in [('D','#176b87','Carrier-only target D'),('B','#b05d20','Opaque target B')]:
        rows=sorted([v for v in result['records'] if v['run']=='modal-domain15-256' and v['target']==target],key=lambda v:v['kappa'])
        x=[v['kappa'] for v in rows];lo=[100*v['scale_error_min'] for v in rows];hi=[100*v['scale_error_max'] for v in rows]
        axes[0].fill_between(x,lo,hi,color=color,alpha=.13)
        style='o-' if target=='D' else 's--'
        axes[0].plot(x,lo,style,markersize=3,color=color,label=label);axes[0].plot(x,hi,style,markersize=3,color=color)
        axes[1].plot(x,[1000*v['maximum_transfer_contribution_rad'] for v in rows],style,markersize=3,color=color,label=label)
        summaries[target]={str(v['kappa']):{k:v[k] for k in ['scale_error_min','scale_error_max','maximum_transfer_contribution_rad','examples']} for v in rows if v['kappa'] in [0.,.1,1.]}
    axes[0].set(ylabel='Local scale error (%)',title='(a) Full shadow-phase interval')
    axes[1].set(ylabel='Maximum transfer term (mrad)',title='(b) Zero reset; |a| ≤ 1 rad')
    for ax in axes:
        ax.set_xlabel('Residual coherence κ');ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Ideal pair: κ = 0   •   Coherent exposure: κ = 1',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(out/f'phase-pair.{ext}',dpi=300)
    plt.close(fig)
    state={'source_result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),'source_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),'elapsed_seconds':time.perf_counter()-start,'summary':summaries,'interpretation':'Curves connect evaluated coherence values; this does not certify all intervening kappa values.15D/256 only.'}
    (out/'summary.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))


if __name__=='__main__':main()
