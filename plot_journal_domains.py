"""Render journal figures from immutable, checksum-verified domain CSVs."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT/'local-results/modal-domain15-256'
    out = ROOT/'local-results/journal-figures-20260915/domains'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':9, 'axes.titlesize':9, 'legend.fontsize':8, 'pdf.fonttype':42})
    sources = {}
    for stem in ['domain6-to15', 'domain12-to15']:
        meta = json.loads((folder/(stem+'.json')).read_text())
        digest = hashlib.sha256((folder/(stem+'.csv')).read_bytes()).hexdigest()
        assert digest == meta['artifacts'][stem+'.csv']
        sources[stem] = digest
        with (folder/(stem+'.csv')).open() as stream:
            rows = list(csv.DictReader(stream))
        lo, hi = meta['refinement_pair']
        fig, axes = plt.subplots(2, 1, figsize=(6.5,5.1), sharex=True, layout='constrained')
        for condition, label, color, marker in [
            ('Cpi/2',r'$\phi=\pi/2$','#176b87','o'),
            ('Cpi',r'$\phi=\pi$','#b05d20','s'),
            ('C3pi/2',r'$\phi=3\pi/2$','#804d9e','^')]:
            contrasts = []
            for domain, style in [(lo,'--'),(hi,'-')]:
                selected = [v for v in rows if v['run']==f'modal-domain{domain}-256']
                base = {float(v['piston_rad']):float(v['raw_readout_rad']) for v in selected if v['condition']=='C0'}
                current = [v for v in selected if v['condition']==condition]
                x = np.array([float(v['piston_rad']) for v in current])
                y = np.array([float(v['raw_readout_rad'])-base[float(v['piston_rad'])] for v in current])
                contrasts.append(y)
                axes[0].plot(x,y,color=color,linestyle=style,marker=marker,markevery=30,
                             markersize=3,markerfacecolor='white',label=f'{label}, {domain}D')
            axes[1].plot(x,contrasts[0]-contrasts[1],color=color,marker=marker,markevery=30,
                         markersize=3,markerfacecolor='white')
        axes[1].axhspan(-.005,.005,color='.5',alpha=.15)
        axes[0].set(ylabel='Contrast relative to phase 0 (rad)',title='(a) Phase contrast with a fixed readout')
        axes[1].set(ylabel='Contrast difference (rad)',xlabel='Differential piston (rad)',
                    title=f'(b) {lo}D minus {hi}D; shading: ±0.005 rad diagnostic target')
        for ax in axes: ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3)
        for ext in ['pdf','png']: fig.savefig(out/(stem+'.'+ext),dpi=300)
        plt.close(fig)
    (out/'summary.json').write_text(json.dumps({'source_csv_sha256':sources,
        'plot_source_lf':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()},indent=2)+'\n')
    print(json.dumps({'status':'RENDERED_FROM_VERIFIED_CSV','figures':2}))


if __name__ == '__main__': main()
