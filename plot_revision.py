"""Render the optical schematic and audit the published domain-sensitivity table.

Uses saved results only; run with the study's existing Python environment.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / 'local-results/phase-pair-20260915/result.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '380682820e4f2d90379fd41e17ea8b2678b70e8086e78c246d4011a9391700c8'
    data = json.loads(source.read_text())
    out = ROOT / 'local-results/referee-revision-20260928'
    out.mkdir(exist_ok=True)
    rows = []
    for domain in (6, 12, 15):
        states = {v['kappa']: v for v in data['records']
                  if v['target'] == 'B' and v['run'] == f'modal-domain{domain}-256'}
        coherent, pair = states[1.], states[0.]
        for v in (coherent, pair):
            assert abs(v['scale_error_min'] - (v['target_gain'] / v['gain_range']['maximum'] - 1)) < 1e-13
            assert abs(v['scale_error_max'] - (v['target_gain'] / v['gain_range']['minimum'] - 1)) < 1e-13
        rows.append(dict(domain=domain, gB=coherent['target_gain'],
                         gCmin=coherent['gain_range']['minimum'], gCmax=coherent['gain_range']['maximum'],
                         gPair=pair['gain_range']['minimum'],
                         eMin=coherent['scale_error_min'], eMax=coherent['scale_error_max'],
                         ePair=pair['scale_error_min'], transferPair=pair['maximum_transfer_contribution_rad']))
    # Check the exact cancellation identity against distinct saved domain gains.
    first, last = rows[0], rows[-1]
    ex = last['gB'] / first['gB'] - 1
    ec = last['gPair'] / first['gPair'] - 1
    q1, q2 = 1 + first['ePair'], 1 + last['ePair']
    assert abs((q2-q1) - q1*(ex-ec)/(1+ec)) < 1e-14
    envelope = {key: [min(r[key] for r in rows), max(r[key] for r in rows)]
                for key in ('eMin', 'eMax', 'ePair', 'transferPair')}
    state = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'rows': rows, 'observed_domain_envelope': envelope,
             'fractional_6_to_15_changes': {'B': ex, 'pair': ec},
             'checks': 'PASS: ratios recomputed and exact cancellation identity checked',
             'interpretation': 'Observed 6D,12D,15D spread; not an infinite-domain error bound.'}
    (out/'domain-sensitivity.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    lines = [r'\begin{table}[ht]', r'\centering',
             r'\caption{Domain sensitivity at 256 modulation positions. Gains use the same fixed readout. Phase extrema are recomputed on each domain. Percentages in the last columns are scale errors, not uncertainties.}',
             r'\label{tab:sensitivity}', r'\begin{tabular}{crrrrrr}', r'\toprule',
             r'$L/D$ & $g_B$ & $g_{C,\min}$ & $g_{C,\max}$ & $e_{B,\min}$ (\%) & $e_{B,\max}$ (\%) & $e_{B,\rm pair}$ (\%)\\', r'\midrule']
    for r in rows:
        lines.append(f"{r['domain']} & {r['gB']:.6f} & {r['gCmin']:.6f} & {r['gCmax']:.6f} & {100*r['eMin']:.4f} & {100*r['eMax']:.4f} & {100*r['ePair']:.5f}" + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (ROOT/'manuscript/domain-sensitivity-table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(6.5, 3.65))
    ax = fig.add_axes([.02, .54, .96, .42]); ax.set(xlim=(0,10), ylim=(0,2)); ax.axis('off')
    labels = [('Pupil / SLM', 'D = 128p; strip = 8p'), ('Relay iris', r'radius $8\lambda/D$'),
              ('Modulation', r'radius $3\lambda/D$'), ('Pyramid', '4 faces'), ('Detector', 'pixel integral; 4D crop')]
    for i, (name, detail) in enumerate(labels):
        x = .05 + 2*i
        ax.add_patch(Rectangle((x,.65),1.8,.7,facecolor='#e8eef2',edgecolor='#263744'))
        ax.text(x+.9,1,name,ha='center',va='center',fontsize=8.5)
        ax.text(x+.9,.38,detail,ha='center',va='center',fontsize=7.5)
        if i < 4: ax.annotate('',xy=(x+2,.99),xytext=(x+1.8,.99),arrowprops=dict(arrowstyle='->'))
    ax.text(5,1.7,'(a) Scalar propagation sequence (schematic)',ha='center')
    for i, (name, shadow, detail) in enumerate([
            ('B: opaque target', '#ffffff', 'bright: no carrier\ncentered iris'),
            ('D: carrier-only control', '#ffffff', r'bright: $+q$'+'\nselected-order iris'),
            ('C: full SLM', '#d28a55', r'bright: $+q$; strip: $-q$'+'\nselected-order iris')]):
        a = fig.add_axes([.025+i*.33,.02,.30,.49]); a.set(xlim=(-1.6,1.6),ylim=(-1.7,1.75),aspect='equal'); a.axis('off')
        pupil = Circle((0,.2),1,facecolor='#bed1dc',edgecolor='#263744'); a.add_patch(pupil)
        strip = Rectangle((-1/16,-.8),1/8,2,facecolor=shadow,edgecolor='none'); strip.set_clip_path(pupil); a.add_patch(strip)
        a.text(-.50,.2,r'$-a/2$',ha='center',va='center',fontsize=9)
        a.text(.50,.2,r'$+a/2$',ha='center',va='center',fontsize=9)
        a.text(0,1.5,name,ha='center',fontsize=9)
        a.text(0,-1.33,detail,ha='center',va='center',fontsize=8)
    fig.savefig(out/'optical-schematic.pdf')
    fig.savefig(out/'optical-schematic.png',dpi=200)
    plt.close(fig)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
