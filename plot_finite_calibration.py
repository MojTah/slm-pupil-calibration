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
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 1, figsize=(6.5,6.3), sharey=True, sharex=True, layout='constrained')
    summaries = []
    for ax, kappa, title in zip(axes, [0.,.1,1.], ['Ideal pair','Imperfect pair','Coherent calibration']):
        current = [v for v in selected if float(v['kappa'])==kappa]
        pokes = sorted({float(v['half_poke_rad']) for v in current})
        bias = [max(abs(float(v['scale_error'])) for v in current if float(v['half_poke_rad'])==h) for h in pokes]
        ax.loglog(pokes, [100*v for v in bias], 'k--', label='Deterministic scale error')
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
            ax.loglog(pokes,data,style,color=color,label=f'$N_{{cal}}=10^{{{len(str(photons))-1}}}$',markersize=4)
        ax.set(title=f'{title} (κ = {kappa:g})')
        ax.set_xticks([.005,.01,.025,.05,.1,.2],['.005','.01','.025','.05','.1','.2'])
        ax.grid(alpha=.2,which='both')
    fig.supylabel('Local scale-error RSS proxy (%)',fontsize=10)
    axes[-1].set_xlabel('Calibration half-poke h (rad)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9)
    fig.suptitle('Opaque B, 15D / 256 angles; worst selected phase',fontsize=10)
    for ext in ['png','pdf']: fig.savefig(out/f'finite-calibration.{ext}',dpi=300)
    plt.close(fig)
    summary={'result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest(),
             'plot_source_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
             'elapsed_seconds':time.perf_counter()-start, 'evaluated_points':summaries,
             'not_shown_1e6':[(v['half_poke_rad'],v['approximation_status']) for v in selected if int(v['calibration_photons'])==10**6]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'elapsed_seconds':summary['elapsed_seconds'],'files':['finite-calibration.png','finite-calibration.pdf']},indent=2))


if __name__=='__main__': main()
