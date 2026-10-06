"""Display finite-poke bias and local noise diagnostics at equal photon budgets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    start = time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    folder = ROOT/'local-results/finite-calibration-20260915'
    state = json.loads((folder/'result.json').read_text())
    if state['status'] != 'COMPLETE_ANALYSIS': raise ValueError('Incomplete result')
    if hashlib.sha256((folder/'curves.csv').read_bytes()).hexdigest() != state['curves_sha256']:
        raise ValueError('Curves changed')
    with (folder/'curves.csv').open() as stream: rows = list(csv.DictReader(stream))
    selected = [v for v in rows if v['run']=='modal-domain15-256' and v['target']=='B']
    out=(ROOT/args.output_dir).resolve() if args.output_dir else folder/'figures'
    if not out.is_relative_to((ROOT/'local-results').resolve()):raise ValueError('Outside study results')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':10,'legend.fontsize':9,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(3, 1, figsize=(6.8,6.2), sharey=True, sharex=True)
    fig.subplots_adjust(left=.115,right=.975,bottom=.14,top=.84,hspace=.38)
    summaries = []
    for ax, kappa, title in zip(axes, [0.,.1,1.], ['Ideal pair','Imperfect pair','Coherent calibration']):
        current = [v for v in selected if float(v['kappa'])==kappa]
        pokes = sorted({float(v['half_poke_rad']) for v in current})
        bias = [max(abs(float(v['scale_error'])) for v in current if float(v['half_poke_rad'])==h) for h in pokes]
        ax.loglog(pokes, [100*v for v in bias], '--', color='#333333',
                  linewidth=1.5, label='Deterministic |scale error|', zorder=3)
        for photons,color,style in [(10**8,'#b05d20','o-'),(10**10,'#176b87','s-.'),(10**12,'#804d9e','^:')]:
            data=[]
            for h in pokes:
                same = [v for v in current if float(v['half_poke_rad'])==h and int(v['calibration_photons'])==photons]
                valid = all(v['approximation_status']=='LOCAL_EXPANSION_ONLY' for v in same)
                rss = max(float(v['scale_rss_proxy']) for v in same) if valid else float('nan')
                data.append(100*rss)
                summaries.append({'kappa':kappa,'half_poke_rad':h,'photons':photons,
                                  'worst_selected_scale_rss_proxy':rss if valid else None,
                                  'worst_selected_scale_error':bias[pokes.index(h)]})
            ax.loglog(pokes,data,style,color=color,label=f'$N_{{cal}}=10^{{{len(str(photons))-1}}}$',
                      markersize=5,markerfacecolor='white',markeredgewidth=1.2,linewidth=1.7)
            if photons == 10**10:
                best = min((i for i, value in enumerate(data) if value == value),key=data.__getitem__)
                ax.plot(pokes[best],data[best],'o',color=color,markersize=10,fillstyle='none',
                        markeredgewidth=1.1,zorder=4)
                ax.annotate(f'{data[best]:.3f}% at {pokes[best]:g} rad',
                            xy=(pokes[best],data[best]),xytext=(.98,.45 if kappa == 1 else .15),textcoords='axes fraction',
                            ha='right',va='center',fontsize=9,color=color,
                            arrowprops={'arrowstyle':'-','color':color,'linewidth':.8})
        panel = 'abc'[list(axes).index(ax)]
        ax.set_title(f'({panel})  {title}   ($\\kappa={kappa:g}$)',loc='left',pad=7)
        ax.set_xticks([.005,.01,.025,.05,.1,.2],['0.005','0.01','0.025','0.05','0.1','0.2'])
        ax.set_yticks([.1,.3,1.,3.],['0.1','0.3','1','3'])
        ax.set_ylim(.065,7)
        ax.minorticks_off()
        ax.grid(axis='y',color='#d9d9d9',linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3,pad=5)
    fig.supylabel('Local scale-error diagnostic (%)',x=.015,fontsize=11)
    axes[-1].set_xlabel('Calibration half-poke $h$ (rad)',labelpad=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,
               frameon=False,handlelength=3,columnspacing=2)
    fig.text(.54,.025,'Open rings: smallest tested diagnostic at $N_{cal}=10^{10}$',
             ha='center',fontsize=9,color='#333333')
    for ext in ['png','pdf']: fig.savefig(out/f'finite-calibration.{ext}',dpi=300)
    plt.close(fig)
    summary={'result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
             'plot_source_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
             'elapsed_seconds':time.perf_counter()-start, 'evaluated_points':summaries,
             'not_shown_1e6':[(v['half_poke_rad'],v['approximation_status']) for v in selected if int(v['calibration_photons'])==10**6]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'elapsed_seconds':summary['elapsed_seconds'],'files':['finite-calibration.png','finite-calibration.pdf']},indent=2))


if __name__=='__main__': main()
