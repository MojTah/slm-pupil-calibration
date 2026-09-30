"""Plot a completed, hash-verified scalar transfer packet."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    start=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();folder=(ROOT/args.run_dir).resolve()
    if not folder.is_relative_to((ROOT/'local-results').resolve()):raise ValueError('Outside study results')
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='COMPLETE_ANALYSIS'
    assert hashlib.sha256((folder/'curves.csv').read_bytes()).hexdigest()==result['curves_sha256']
    out=(ROOT/args.output_dir).resolve() if args.output_dir else folder/'figures'
    if not out.is_relative_to((ROOT/'local-results').resolve()):raise ValueError('Outside study results')
    out.mkdir(parents=True,exist_ok=True)
    with (folder/'curves.csv').open() as f:rows=list(csv.DictReader(f))
    target=result['records'][0]['target'];largest='modal-domain15-256'
    colors={'nominal':'#176b87','minimum_gain':'#b05d20','maximum_gain':'#804d9e'}
    labels={'nominal':'Phase zero','minimum_gain':'Minimum SLM gain','maximum_gain':'Maximum SLM gain'}
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42})
    styles={'nominal':'-','minimum_gain':'--','maximum_gain':':' }
    fig,axes=plt.subplots(2,2,figsize=(6.5,5.4),layout='constrained')
    ax=axes[0,0]
    for i,r in enumerate(result['records']):
        lo,hi=r['minimum_scale_error']*100,r['maximum_scale_error']*100
        nominal=next(v['scale_error']*100 for v in r['policies'] if v['label']=='nominal')
        ax.plot([lo,hi],[i,i],color='#53565b',linewidth=3)
        ax.plot(nominal,i,'o',color=colors['nominal'])
    ax.set_yticks(range(len(result['records'])),[r['run'].replace('modal-domain','').replace('-','D / ')+' angles' for r in result['records']])
    ax.set_xlabel('Local scale error (%)');ax.set_title('(a) Full shadow-phase interval')
    ax.axvline(0,color='.75',linewidth=.8)
    summaries={}
    for policy in colors:
        rr=[v for v in rows if v['run']==largest and v['policy']==policy]
        x=[float(v['piston_rad']) for v in rr]
        axes[0,1].plot(x,[float(v['transfer_contribution_rad'])*1000 for v in rr],color=colors[policy],linestyle=styles[policy],label=labels[policy])
        axes[1,0].plot(x,[float(v['reset_total_error_rad']) for v in rr],color=colors[policy],linestyle=styles[policy],label=labels[policy])
        axes[1,1].semilogy(x,[float(v['crossover_detected_photons']) if v['crossover_detected_photons'] else float('nan') for v in rr],color=colors[policy],linestyle=styles[policy])
        summaries[policy]={k:max(abs(float(v[k])) for v in rr) for k in ['transfer_contribution_rad','reset_total_error_rad','unreset_total_error_rad']}
        summaries[policy]['at_0.1_rad']=min(rr,key=lambda v:abs(float(v['piston_rad'])-.1))
        summaries[policy]['at_1_rad']=rr[-1]
    axes[1,0].plot(x,[float(v['self_calibration_error_rad']) for v in rr],color='black',linestyle='-.',label='Target self-calibration')
    axes[0,1].set(ylabel='Transfer contribution (mrad)',title='(b) After target zero reset')
    axes[1,0].set(ylabel='Total piston error (rad)',title='(c) Transfer + nonlinearity')
    axes[1,1].set(ylabel='Target detected photons, N*',title='(d) Conditional crossover')
    for ax in [axes[0,1],*axes[1]]:
        ax.set_xlabel('Differential piston (rad)');ax.grid(alpha=.2)
    handles,labels=axes[1,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle(f'Target {target}: 15D / 256 angles in panels (b–d)',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(out/f'transfer.{ext}',dpi=300)
    plt.close(fig)
    state={'target':target,'summaries':summaries,'source_result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),'plot_source_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),'elapsed_seconds':time.perf_counter()-start}
    (out/'summary.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'target':target,'summary':state['summaries'],'elapsed_seconds':state['elapsed_seconds']},indent=2))


if __name__=='__main__':main()
