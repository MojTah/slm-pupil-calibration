"""Render fresh manuscript figures/tables; never propagate or load historical data."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from fresh_paper_analysis import ANALYSIS_SOURCES, POKES, EQUAL_POKES
from plot_publication import coherence
from piston_basis import readout_matrices, readout_moments
from analyze_readout_sensitivity import dark
from reproduce_paper import ROOT, digest, verify_bundle
from run_pilot import text_hash, write_json, WallDeadline
from runtime_resources import peak_rss_bytes

FILES = {
    'transfer-summary.csv': {'case', 'readout', 'target', 'g_target', 'g_C_min', 'g_C_max', 'g_pair', 'paired_percent'},
    'phase-curves.csv': {'case', 'target', 'phase_rad', 'scale_error_percent', 'raw_offset_rad'},
    'pairing-summary.csv': {'case', 'target', 'kappa', 'coherent_min_percent', 'coherent_max_percent'},
    'transfer-curves.csv': {'case', 'target', 'kappa', 'piston_rad', 'transfer_min_rad', 'transfer_max_rad', 'target_self_error_rad'},
    'finite-calibration.csv': {'target', 'kappa', 'half_poke_rad', 'calibration_photons', 'scale_error', 'scale_rss_proxy', 'approximation_status'},
    'finite-equal-poke.csv': {'target', 'kappa', 'half_poke_rad', 'calibration_photons', 'equal_scale_rss_proxy', 'equal_approximation_status'},
    'finite-minima.csv': {'target', 'metric', 'calibration_photons', 'kappa', 'half_poke_rad', 'worst_selected_phase_percent'},
    'sensitivity-summary.csv': {'lower', 'higher', 'mean', 'gain', 'relative_variance', 'phase_contrast'},
    'sensitivity-curves.csv': {'lower', 'higher', 'condition', 'piston_rad', 'lower_phase_contrast_rad', 'higher_phase_contrast_rad', 'lower_minus_higher_rad'},
}
RENDER_SOURCES = (*ANALYSIS_SOURCES, 'render_paper.py', 'plot_publication.py')
READOUTS = {'sampled_reference': 'Sampled reference', 'native15_fixed': 'Native 15D',
            'quadrant15_fixed': 'Global quadrant'}
COLORS = ['#176b87', '#b05d20', '#804d9e']
TABLE_ROWS = {'main-table1-readouts':3,'main-table2-command-readouts':3,'main-table3-domains':3,
              'supp-table1-angular':4,'supp-table2-sensitivity':4,'supp-table3-pairing':6,
              'supp-table4-finite':3,'supp-table5-finite':4,'supp-table6-matched-carriers':4,
              'supp-table7-orthogonal-domains':3}
TRANSFER_CASES = {'B':('original-6d-256','original-12d-256','original-15d-128','original-15d-256'),
                  'D':('original-6d-128','original-6d-256','original-12d-256','original-15d-128','original-15d-256')}


def value(text):
    if text == '':
        return None
    if text in ('True', 'False'):
        return text == 'True'
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number):
        raise ValueError('Nonfinite CSV value')
    return number


def load_analysis(folder):
    state = json.loads((folder/'result.json').read_text())
    if state.get('status') != 'COMPLETE_NUMERIC_ANALYSIS' or not isinstance(state.get('canary'), bool):
        raise ValueError('Require completed fresh numeric analysis')
    expected = {name: text_hash(ROOT/name) for name in ANALYSIS_SOURCES}
    if state.get('source_sha256_lf') != expected:
        raise ValueError('Analysis source identity changed')
    if not state.get('analysis_runtime') or not state.get('input_build_runtime'):
        raise ValueError('Analysis runtime provenance missing')
    if not state.get('outputs') or set(state['outputs'])-set(FILES):
        raise ValueError('Unexpected analysis output name')
    tables = {}
    for name, record in state['outputs'].items():
        if digest(folder/name) != record['sha256']:
            raise ValueError('Analysis checksum changed: '+name)
        with (folder/name).open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not FILES[name].issubset(reader.fieldnames or []):
                raise ValueError('Missing CSV columns: '+name)
            rows = [{key: value(val) for key, val in row.items()} for row in reader]
        if len(rows) != record['rows']:
            raise ValueError('CSV row count changed: '+name)
        if any('canary' in row and row['canary'] != state['canary'] for row in rows):
            raise ValueError('Mixed canary identity')
        tables[name] = rows
    return state, tables


def select(rows, **keys):
    return [row for row in rows if all(row[key] == val for key, val in keys.items())]


def one(rows, **keys):
    found = select(rows, **keys)
    if len(found) != 1:
        raise ValueError('Expected one summary row: '+str(keys))
    return found[0]


def curve(rows, column, count):
    rows = sorted(rows, key=lambda row: row[column])
    x = np.array([row[column] for row in rows])
    if len(x) != count or not np.isfinite(x).all() or not np.all(np.diff(x) > 0):
        raise ValueError('Incomplete or duplicate curve samples')
    return x, rows


def finite_curve(rows, kappa, photons, metric='scale_rss_proxy', grid=POKES):
    points = []
    status = 'equal_approximation_status' if metric.startswith('equal_') else 'approximation_status'
    for h in grid:
        current = select(rows, kappa=kappa, calibration_photons=photons, half_poke_rad=h)
        if len(current) != (1 if kappa == 0 else 3):
            raise ValueError('Incomplete finite phase selection')
        valid = all(r[status] == 'LOCAL_EXPANSION_ONLY' and r[metric] is not None for r in current)
        points.append(100*max(r[metric] for r in current) if valid else np.nan)
    return np.array(points)


def target_matrices(build, state):
    manifest = verify_bundle(build)
    if digest(build/'result.json') != state['input_manifest_sha256']:
        raise ValueError('Build does not match analysis input identity')
    cases = {r['case']['id']: r for r in manifest['cases']}
    with np.load(build/cases['reference-6d-128']['artifact']) as data:
        weights = data['weights']
    targets = {}
    for target, case in [('D', 'original-15d-256'), ('B', 'opaque-15d-256')]:
        if case in cases:
            with np.load(build/cases[case]['artifact']) as data:
                matrices = readout_matrices(data['gram'], weights)
            targets[target] = dark(matrices) if target == 'D' else matrices
    return targets


def load_common(folder, analysis, tables, build):
    from reproduce_common_mask import SOURCES, CommonMaskModel, masks
    from simulation import Configuration
    state = json.loads((folder/'result.json').read_text())
    if state.get('schema') != 'fresh-common-mask-v1' or state.get('status') != 'COMPLETE_VALIDATED':
        raise ValueError('Require completed fresh common-mask result')
    if state['input_manifest_sha256'] != analysis['input_manifest_sha256'] or state['canary'] != analysis['canary']:
        raise ValueError('Common-mask input identity or canary flag differs')
    if state['source_sha256_lf'] != {name:text_hash(ROOT/name) for name in SOURCES}:
        raise ValueError('Common-mask source identity changed')
    if state['common']['artifact'] != 'common.npz' or digest(folder/'common.npz') != state['common']['sha256']:
        raise ValueError('Common-mask array checksum changed')
    if digest(folder/'summary.json') != state['summary_sha256']:
        raise ValueError('Common-mask summary checksum changed')
    summary = json.loads((folder/'summary.json').read_text())
    if summary['canary'] != analysis['canary']:
        raise ValueError('Common-mask summary canary flag differs')
    common = one(summary['rows'],command='original',target='common_B')
    direct = one(tables['transfer-summary.csv'],case='original-15d-256',readout='sampled_reference',target='D')
    if not np.allclose([common['g_D'],common['g_pair'],common['g_C_min'],common['g_C_max']],
                       [direct['g_target'],direct['g_pair'],direct['g_C_min'],direct['g_C_max']],rtol=1e-12,atol=1e-14):
        raise ValueError('Common-mask controls differ from fresh analysis')
    if not np.isclose(common['g_target'],state['common']['g_B'],rtol=1e-12,atol=1e-14):
        raise ValueError('Common-mask gain inconsistent')
    ratios=sorted(100*(common['g_target']/common[key]-1) for key in ('g_C_min','g_C_max'))
    if not np.allclose([*ratios,100*(common['g_target']/common['g_pair']-1)],
                       [common['coherent_min_percent'],common['coherent_max_percent'],common['paired_percent']],rtol=1e-12,atol=1e-12):
        raise ValueError('Common-mask ratios inconsistent')
    bundle = verify_bundle(build)
    cases = {r['case']['id']:r['case'] for r in bundle['cases']}
    if (state['case']!=cases['original-15d-256'] or
            state['input_artifacts']!={r['case']['id']:r['sha256'] for r in bundle['cases']}):
        raise ValueError('Common-mask build case or artifact identity differs')
    result = []
    for domain in (6,12,15):
        b = select(tables['transfer-summary.csv'],case=f'original-{domain}d-256',readout='sampled_reference',target='B')
        d = select(tables['transfer-summary.csv'],case=f'original-{domain}d-256',readout='sampled_reference',target='D')
        if not b or not d:
            continue
        b,d = b[0],d[0]
        row = {'domain_D':domain,'centered_ratio_percent':100*(b['g_target']/d['g_target']-1),
               'centered_pair_percent':b['paired_percent']}
        if domain == 15:
            row.update(common_ratio_percent=100*(common['g_target']/d['g_target']-1),common_pair_percent=common['paired_percent'],origin='fresh common-mask propagation')
        else:
            case=cases[f'opaque-{domain}d-256']
            model=CommonMaskModel(Configuration(**case['geometry']),case['iris'])
            equal=masks(model)['equal']; del model
            if not equal:
                raise ValueError('Lower-domain masks differ; a new propagation would be required')
            row.update(common_ratio_percent=row['centered_ratio_percent'],common_pair_percent=row['centered_pair_percent'],origin='exact mask equality; fresh centered control')
        result.append(row)
    return state,result


def schematic():
    fig = plt.figure(figsize=(6.6, 3.9))
    ax = fig.add_axes([.02, .55, .96, .40]); ax.set(xlim=(0, 10), ylim=(0, 2)); ax.axis('off')
    for i, (name, detail) in enumerate([('Pupil / SLM', 'D = 128p; strip = 8p'),
            ('Relay iris', r'radius $8\lambda/D$'), ('Modulation', r'radius $3\lambda/D$'),
            ('Pyramid', '4 faces'), ('Detector', 'pixel integral; 4D crop')]):
        x = .05+2*i
        ax.add_patch(Rectangle((x,.65),1.8,.7,facecolor='#e8eef2',edgecolor='#263744'))
        ax.text(x+.9,1,name,ha='center',va='center',fontsize=8)
        ax.text(x+.9,.35,detail,ha='center',fontsize=7)
        if i < 4:
            ax.annotate('', xy=(x+2,1), xytext=(x+1.8,1), arrowprops={'arrowstyle':'->'})
    ax.text(5,1.65,'(a) Scalar propagation sequence (schematic)',ha='center')
    for i, (name, color, detail) in enumerate([
        ('B: opaque target','white','bright: no carrier\ncentered iris'),
        ('D: carrier-only control','white',r'bright: $+q$'+'\nselected-order iris'),
        ('C: full SLM','#d28a55',r'bright: $+q$; strip: $-q$'+'\nselected-order iris')]):
        ax = fig.add_axes([.025+i*.33,.02,.30,.50]); ax.set(xlim=(-1.6,1.6),ylim=(-1.7,1.75),aspect='equal'); ax.axis('off')
        pupil = Circle((0,.2),1,facecolor='#bed1dc',edgecolor='#263744'); ax.add_patch(pupil)
        strip = Rectangle((-1/16,-.8),1/8,2,facecolor=color); strip.set_clip_path(pupil); ax.add_patch(strip)
        ax.text(-.5,.2,r'$-a/2$',ha='center'); ax.text(.5,.2,r'$+a/2$',ha='center')
        ax.text(0,1.5,name,ha='center',fontsize=9); ax.text(0,-1.3,detail,ha='center',va='center',fontsize=8)
    return fig


def tolerances():
    d, e = np.meshgrid(np.linspace(0,.3,301), np.linspace(0,np.pi/3,301))
    fig, ax = plt.subplots(figsize=(5.7,3.4), layout='constrained')
    k = coherence(d,e)
    image = ax.pcolormesh(d,np.rad2deg(e),k,cmap='cividis',shading='auto',rasterized=True)
    contours = ax.contour(d,np.rad2deg(e),k,levels=[.05,.1,.2,.3,.4,.5],colors='white',linewidths=.9)
    ax.clabel(contours,fmt='%.2f',fontsize=8)
    ax.set(xlabel=r'Raw exposure imbalance $|d|$',ylabel=r'Phase-step error $|\varepsilon|$ (deg)')
    fig.colorbar(image,ax=ax,label=r'Residual coherence $\kappa$')
    return fig


def transfer_records(tables, target):
    current = select(tables['transfer-summary.csv'], command='original', readout='sampled_reference', target=target)
    available = {row['case'] for row in current}
    return ([one(current,case=name) for name in TRANSFER_CASES[target] if name in available],
            [name for name in TRANSFER_CASES[target] if name not in available])


def transfer_coverage(tables, target, has_variance):
    _,missing=transfer_records(tables,target)
    reasons=['missing interval cases: '+', '.join(missing)] if missing else []
    if not has_variance:
        reasons.append('crossover requires matching --build-input')
    return 'PARTIAL: '+'; '.join(reasons) if reasons else 'RENDERED'


def transfer_figure(tables, target, matrices):
    summaries,_ = transfer_records(tables,target)
    record = one(summaries, case='original-15d-256')
    phase0 = one(tables['phase-curves.csv'], case='original-15d-256', target=target, phase_rad=0.)
    gx = record['g_target']
    gains = [gx/(1+phase0['scale_error_percent']/100), record['g_C_min'], record['g_C_max']]
    x, rows = curve(select(tables['transfer-curves.csv'], case='original-15d-256', target=target, kappa=1.), 'piston_rad', 201)
    self_error = np.array([r['target_self_error_rad'] for r in rows])
    response = gx*(x+self_error)
    variance = np.array([readout_moments(matrices, a)['variance_coefficient'] for a in x]) if matrices is not None else None
    if matrices is not None and not np.isclose(readout_moments(matrices,0.)['local_gain'], gx, rtol=1e-12):
        raise ValueError('Target contraction does not match CSV gain')
    fig, axes = plt.subplots(2,2,figsize=(6.6,5.5),layout='constrained')
    for i, r in enumerate(summaries):
        nominal = one(tables['phase-curves.csv'], case=r['case'], target=target, phase_rad=0.)['scale_error_percent']
        axes[0,0].plot([r['coherent_min_percent'],r['coherent_max_percent']], [i,i],color='.4',lw=3)
        axes[0,0].plot(nominal,i,'o',color=COLORS[0])
    axes[0,0].set(yticks=range(len(summaries)),yticklabels=[f"{r['domain_D']:g}D / {r['angles']:g}" for r in summaries],
                  xlabel='Scale error (%)',title='(a) Full phase interval')
    for gain, color, label, style in zip(gains,COLORS,['Phase zero','Minimum gain','Maximum gain'],['-','--',':']):
        delta = response*(1/gain-1/gx)
        axes[0,1].plot(x,1000*delta,color=color,ls=style,label=label)
        axes[1,0].plot(x,self_error+delta,color=color,ls=style)
        if variance is not None:
            crossover = np.full_like(x,np.nan)
            valid = (x != 0) & (delta != 0)
            crossover[valid] = variance[valid]/gain**2/delta[valid]**2
            axes[1,1].semilogy(x,crossover,color=color,ls=style)
    axes[1,0].plot(x,self_error,'k-.',label='Target self-calibration')
    axes[0,1].legend(frameon=False,fontsize=7); axes[1,0].legend(frameon=False,fontsize=7)
    axes[0,1].set(ylabel='Transfer contribution (mrad)',title='(b) After target zero reset')
    axes[1,0].set(ylabel='Total piston error (rad)',title='(c) Transfer + nonlinearity')
    axes[1,1].set(ylabel='Target detected photons, N*',title='(d) Conditional crossover')
    axes[1,1].text(.03,.97,'At a = 0: undefined' if variance is not None else 'Not rendered: --build-input required',
                   transform=axes[1,1].transAxes,va='top',fontsize=8)
    for ax in [axes[0,1],*axes[1]]:
        ax.set_xlabel('Differential piston (rad)'); ax.grid(alpha=.2)
    return fig


def carrier_figure(tables):
    fig, axes = plt.subplots(2,2,figsize=(6.6,5.4),layout='constrained',sharex=True)
    records = [one(tables['transfer-summary.csv'],case=c+'-6d-256',readout='sampled_reference',target='B') for c in ('original','orthogonal')]
    limits = min(r['coherent_min_percent'] for r in records), max(r['coherent_max_percent'] for r in records)
    margin = max((limits[1]-limits[0])*.08,1e-6)
    for i, (command, record, color) in enumerate(zip(('original','orthogonal'),records,COLORS)):
        x, rows = curve(select(tables['phase-curves.csv'],case=command+'-6d-256',target='B'),'phase_rad',721)
        error = np.array([r['scale_error_percent'] for r in rows])
        if error.min() < record['coherent_min_percent']-1e-9 or error.max() > record['coherent_max_percent']+1e-9:
            raise ValueError('Phase curve outside analytic interval')
        axes[0,i].plot(x,error,color=color,label='Coherent')
        axes[0,i].axhline(record['paired_percent'],color='.2',ls='--',label='Ideal raw pair')
        axes[0,i].set(title=f"({'ab'[i]}) {command.capitalize()} carrier",ylabel='Scale error (%)',ylim=(limits[0]-margin,limits[1]+margin))
        axes[0,i].legend(fontsize=7,frameon=False,loc='lower center')
        if command == 'orthogonal':
            inset = axes[0,i].inset_axes([.18,.58,.72,.28]); inset.plot(x,error,color=color)
            inset.axhline(record['paired_percent'],color='.2',ls='--'); inset.tick_params(labelsize=6)
            inset.set_title('Orthogonal zoom (%)',fontsize=7); inset.set_xticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$'])
        axes[1,i].plot(x,[r['raw_offset_rad'] for r in rows],color=color)
        axes[1,i].set(title=f"({'cd'[i]}) Offset span {record['raw_offset_span_rad']:.4g} rad",ylabel='Raw zero offset (rad)',xlabel=r'Strip phase $\phi$')
        for ax in axes[:,i]:
            ax.set_xticks([0,np.pi,2*np.pi],['0',r'$\pi$',r'$2\pi$']); ax.grid(alpha=.2)
    return fig


def pairing_figure(tables):
    fig, axes = plt.subplots(1,2,figsize=(6.6,3.5),layout='constrained')
    for target,color in zip(('D','B'),COLORS):
        rows = sorted(select(tables['pairing-summary.csv'],case='original-15d-256',target=target),key=lambda r:r['kappa'])
        if not rows:
            continue
        x = [r['kappa'] for r in rows]
        lo,hi = [[r[key] for r in rows] for key in ('coherent_min_percent','coherent_max_percent')]
        axes[0].fill_between(x,lo,hi,color=color,alpha=.15)
        axes[0].plot(x,lo,'o-',ms=3,color=color,label=target); axes[0].plot(x,hi,'o-',ms=3,color=color)
        maxima = [1000*max(max(abs(r['transfer_min_rad']),abs(r['transfer_max_rad'])) for r in
                   select(tables['transfer-curves.csv'],case='original-15d-256',target=target,kappa=k)) for k in x]
        axes[1].plot(x,maxima,'o-',ms=3,color=color,label=target)
    axes[0].set(title='(a) Full phase interval',ylabel='Local scale error (%)')
    axes[1].set(title='(b) Zero reset; |a| ≤ 1 rad',ylabel='Maximum transfer (mrad)')
    for ax in axes:
        ax.set_xlabel(r'Residual coherence $\kappa$'); ax.legend(title='Target',frameon=False); ax.grid(alpha=.2)
    return fig


def domain_figure(tables, common=None):
    rows = [one(tables['transfer-summary.csv'],case=f'original-{d}d-256',readout='sampled_reference',target='B') for d in (6,12,15)]
    dark_rows = [one(tables['transfer-summary.csv'],case=f'original-{d}d-256',readout='sampled_reference',target='D') for d in (6,12,15)]
    fig=plt.figure(figsize=(6.6,5.1),layout='constrained'); grid=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[1,1])]
    for key,label,style in [('g_target','D','o-'),('g_C_min','Minimum C','^--'),('g_C_max','Maximum C','s:')]:
        axes[0].plot([6,12,15],[100*(r[key]/dark_rows[0][key]-1) for r in dark_rows],style,label=label,ms=4,mfc='white')
    axes[0].set(title='(a) Absolute-gain drift',ylabel='Change from own 6D value (%)'); axes[0].legend(fontsize=7,ncol=3)
    axes[1].plot([6,12,15],[100*(b['g_target']/d['g_target']-1) for b,d in zip(rows,dark_rows)],'s--',mfc='white',color=COLORS[1],label='Centered mask')
    axes[2].plot([6,12,15],[r['paired_percent'] for r in rows],'s--',mfc='white',color=COLORS[1],label='Centered mask')
    if common and len(common)==3:
        for ax,key in [(axes[1],'common_ratio_percent'),(axes[2],'common_pair_percent')]:
            ax.plot([r['domain_D'] for r in common],[r[key] for r in common],'o-',ms=4,color=COLORS[0],label='Common translated mask')
    else:
        axes[2].text(.03,.04,'Common-mask overlay unavailable',transform=axes[2].transAxes,fontsize=7,color='#9c3333')
    axes[1].set(title='(b) B-to-D gain ratio',ylabel='100 (gB/gD − 1) (%)')
    axes[2].set(title='(c) Original-command pair',ylabel='Local scale error (%)'); axes[2].legend(fontsize=6,frameon=False)
    for ax in axes:
        ax.set(xlabel='Domain side L / D',xticks=[6,12,15]); ax.grid(alpha=.2)
    return fig


def finite_figure(tables):
    rows = select(tables['finite-calibration.csv'],target='B')
    fig, axes = plt.subplots(3,1,figsize=(6.6,6.4),sharex=True,sharey=True,layout='constrained')
    for ax,k in zip(axes,(0.,.1,1.)):
        bias = [100*max(abs(r['scale_error']) for r in select(rows,kappa=k,half_poke_rad=h)) for h in POKES]
        ax.loglog(POKES,bias,'k--',label='Deterministic |scale error|')
        for photons,color,style in zip((10**8,10**10,10**12),COLORS,('o-','s-.','^:')):
            y = finite_curve(rows,k,photons)
            ax.loglog(POKES,y,style,color=color,ms=4,mfc='white',label=f'$N_{{cal}}=10^{{{int(np.log10(photons))}}}$')
            if photons == 10**10 and np.isfinite(y).any():
                best = np.nanargmin(y); ax.plot(POKES[best],y[best],'o',color=color,ms=10,mfc='none')
        ax.set(title=rf'$\kappa={k:g}$',ylabel='Local diagnostic (%)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2,frameon=False); axes[-1].set_xlabel('Calibration half-poke h (rad)')
    return fig


def sensitivity_figure(tables, domain):
    rows = select(tables['sensitivity-curves.csv'],lower=f'original-{domain}d-256',higher='original-15d-256')
    fig, axes = plt.subplots(2,1,figsize=(6.6,5.2),sharex=True,layout='constrained')
    for condition,color in zip(('Cpi/2','Cpi','C3pi/2'),COLORS):
        x, current = curve(select(rows,condition=condition),'piston_rad',201)
        lo,hi,diff = [np.array([r[key] for r in current]) for key in ('lower_phase_contrast_rad','higher_phase_contrast_rad','lower_minus_higher_rad')]
        if not np.allclose(lo-hi,diff,rtol=1e-12,atol=1e-14):
            raise ValueError('Phase contrast difference inconsistent')
        axes[0].plot(x,lo,'--',color=color,label=f'{condition}, {domain}D')
        axes[0].plot(x,hi,'-',color=color,label=f'{condition}, 15D')
        axes[1].plot(x,diff,color=color)
    axes[0].set(ylabel='Phase contrast (rad)',title='(a) Relative to phase zero'); axes[0].legend(ncol=3,fontsize=7)
    axes[1].axhspan(-.005,.005,color='.5',alpha=.15)
    axes[1].set(ylabel='Contrast difference (rad)',xlabel='Differential piston (rad)',title=f'(b) {domain}D minus 15D; shading: ±0.005 rad')
    for ax in axes:
        ax.grid(alpha=.2)
    return fig


def tex_escape(value):
    return str(value).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def tex_table(headers, rows):
    def cell(v):
        return '--' if v is None else f'{v:.8g}' if isinstance(v,(float,int)) and not isinstance(v,bool) else tex_escape(v)
    return '\n'.join([r'\begin{tabular}{'+'l'*len(headers)+'}',r'\hline',
        ' & '.join(map(tex_escape,headers))+r'\\',r'\hline',
        *[' & '.join(map(cell,row))+r'\\' for row in rows],r'\hline',r'\end{tabular}',''])


def numeric_tables(tables):
    transfer = tables['transfer-summary.csv']; sensitivity = tables.get('sensitivity-summary.csv',[])
    output = {}
    def put(name, headers, rows):
        if rows:
            output[name] = {'tex':tex_table(headers,rows),'rows':len(rows)}
    columns = ['coherent_min_percent','coherent_max_percent','paired_percent']
    rows = select(transfer,case='original-15d-256',target='B')
    put('main-table1-readouts',['Readout','Minimum (%)','Maximum (%)','Pair (%)'],[[READOUTS[r['readout']],*[r[k] for k in columns]] for r in rows])
    rows = []
    for label in READOUTS:
        original = select(transfer,case='original-6d-256',target='B',readout=label)
        orthogonal = select(transfer,case='orthogonal-6d-256',target='B',readout=label)
        if original and orthogonal:
            a,b = original[0],orthogonal[0]
            rows.append([READOUTS[label],a[columns[0]],a[columns[1]],a[columns[2]],b[columns[0]],b[columns[1]]])
    put('main-table2-command-readouts',['Readout','Original min (%)','max (%)','pair (%)','Orthogonal min (%)','max (%)'],rows)
    rows = select(transfer,command='original',nominal_angles=256,readout='sampled_reference',target='B')
    put('main-table3-domains',['L/D','gB','gC min','gC max','e min (%)','e max (%)','pair (%)'],[[r['domain_D'],r['g_target'],r['g_C_min'],r['g_C_max'],*[r[k] for k in columns]] for r in rows])
    metrics = [('mean','Mean (rad)',.005),('gain','Gain',.001),('relative_variance','Relative variance',.01),('phase_contrast','Phase contrast (rad)',.005)]
    angular = select(sensitivity,lower='original-6d-128',higher='original-6d-256')
    put('supp-table1-angular',['Quantity','Maximum change','Diagnostic target'],[[label,angular[0][key],target] for key,label,target in metrics] if angular else [])
    put('supp-table2-sensitivity',['Lower','Higher','Mean','Gain','Relative variance','Phase contrast'],[[r['lower'],r['higher'],*[r[key] for key,_,_ in metrics]] for r in sensitivity if r not in angular])
    rows = []
    for r in select(tables['pairing-summary.csv'],case='original-15d-256'):
        if r['kappa'] in (0.,.1,1.):
            current = select(tables['transfer-curves.csv'],case=r['case'],target=r['target'],kappa=r['kappa'])
            peak = 1000*max(max(abs(v['transfer_min_rad']),abs(v['transfer_max_rad'])) for v in current)
            rows.append([r['target'],r['kappa'],r[columns[0]],r[columns[1]],peak])
    put('supp-table3-pairing',['Target','kappa','Minimum (%)','Maximum (%)','Max transfer (mrad)'],rows)
    for number,filename,metric,grid in [(4,'finite-calibration.csv','scale_rss_proxy',POKES),(5,'finite-equal-poke.csv','equal_scale_rss_proxy',EQUAL_POKES)]:
        rows = []; selected = select(tables[filename],target='B')
        if selected:
            for photons in ((10**8,10**10,10**12) if number==4 else (10**6,10**8,10**10,10**12)):
                row = [photons]
                for kappa in (0.,1.):
                    y = finite_curve(selected,kappa,photons,metric,grid)
                    i = np.nanargmin(y) if np.isfinite(y).any() else None
                    row.extend([None,None,None] if i is None else [grid[i],y[i],i in (0,len(grid)-1)])
                rows.append(row)
        put(f'supp-table{number}-finite',['Photons','Pair h','Pair (%)','Boundary','Coherent h','Coherent (%)','Boundary'],rows)
    rows = select(transfer,domain_D=6,nominal_angles=256,readout='sampled_reference')
    put('supp-table6-matched-carriers',['Command','Target','Minimum (%)','Maximum (%)','Pair (%)'],[[r['command'],r['target'],*[r[k] for k in columns]] for r in rows])
    rows = select(transfer,command='orthogonal',nominal_angles=256,readout='sampled_reference',target='B')
    put('supp-table7-orthogonal-domains',['L/D','Minimum (%)','Maximum (%)','Pair (%)'],[[r['domain_D'],*[r[k] for k in columns]] for r in rows])
    return output


def render(input_folder, output, build_input=None, common_mask_input=None):
    start = time.perf_counter(); output = Path(output).resolve()
    folder = Path(input_folder).resolve() if input_folder else None
    state,tables = load_analysis(folder) if folder else ({'canary':False},{})
    build = Path(build_input).resolve() if build_input else None
    if build and not folder:
        raise ValueError('--build-input requires --input')
    targets = target_matrices(build,state) if build else {}
    common_folder = Path(common_mask_input).resolve() if common_mask_input else None
    if common_folder and not build:
        raise ValueError('--common-mask-input requires matching --build-input')
    common_state,common = load_common(common_folder,state,tables,build) if common_folder else ({},[])
    source = {name:text_hash(ROOT/name) for name in RENDER_SOURCES}
    if common_folder:
        source['reproduce_common_mask.py']=text_hash(ROOT/'reproduce_common_mask.py')
    output.mkdir(parents=True,exist_ok=False)
    record = {'status':'RUNNING','canary':state['canary'],'source_sha256_lf':source,'outputs':{},'coverage':{},
        'input_analysis_manifest_sha256':digest(folder/'result.json') if folder else None,
        'input_build_manifest_sha256':digest(build/'result.json') if build else None,
        'input_common_manifest_sha256':digest(common_folder/'result.json') if common_folder else None,
        'common_runtime':common_state.get('runtime'),'common_mask_rows':common,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform()},
        'analysis_runtime':state.get('analysis_runtime'),'runtime_policy':'ALLOW_AND_RECORD; no cross-version byte identity claim',
        'limits':['No full-size validation follows from canary rendering','Common-mask results concern local gains only; finite-piston curves retain centered support',
                  'Extra angular case arrays are available in the expanded graph; not every prose diagnostic is exported here',
                  'Layouts are regenerated; historical figure bytes and typography are not reproduced']}
    write_json(output/'result.json',record)
    def save(fig,name,inputs):
        if state['canary'] and inputs != 'analytic':
            fig.suptitle('CANARY: 8 pupil pixels / 8 angles; not paper results',fontsize=9,color='#9c3333')
        for ext in ('pdf','png'):
            path=output/f'{name}.{ext}'; fig.savefig(path,dpi=160,bbox_inches='tight')
            record['outputs'][path.name]={'sha256':digest(path),'kind':'figure','inputs':inputs}
        plt.close(fig); record['coverage'][name]='RENDERED'; write_json(output/'result.json',record)
    try:
        plt.rcParams.update({'font.size':9,'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
        with WallDeadline(180):
            save(schematic(),'optical-schematic','analytic'); save(tolerances(),'pair-tolerances','analytic')
            if folder:
                for target,name in [('B','transfer-B'),('D','transfer')]:
                    if select(tables['transfer-summary.csv'],case='original-15d-256',target=target):
                        save(transfer_figure(tables,target,targets.get(target)),name,'CSV + fresh build variance' if target in targets else 'CSV only; crossover missing')
                        record['coverage'][name]=transfer_coverage(tables,target,target in targets)
                    else:
                        record['coverage'][name]='MISSING: required target case absent'
                jobs = [('carrier-comparison',lambda:carrier_figure(tables),'orthogonal-6d-256'),
                        ('phase-pair',lambda:pairing_figure(tables),'original-15d-256'),
                        ('domain-gains',lambda:domain_figure(tables,common),'original-12d-256'),
                        ('finite-calibration',lambda:finite_figure(tables),'opaque-target'),
                        ('domain6-to15',lambda:sensitivity_figure(tables,6),'sensitivity'),
                        ('domain12-to15',lambda:sensitivity_figure(tables,12),'sensitivity')]
                for name,plot,required in jobs:
                    available = bool(tables.get('sensitivity-curves.csv')) if required=='sensitivity' else bool(select(tables['finite-calibration.csv'],target='B')) if required=='opaque-target' else bool(select(tables['transfer-summary.csv'],case=required,target='B'))
                    if available:
                        save(plot(),name,'CSV')
                    else:
                        record['coverage'][name]='MISSING: required cases absent'
                for name,table in numeric_tables(tables).items():
                    path=output/(name+'.tex')
                    path.write_text(('% CANARY; not paper values\n' if state['canary'] else '')+table['tex'],encoding='utf-8')
                    record['outputs'][path.name]={'sha256':digest(path),'kind':'table','inputs':'CSV','rows':table['rows']}
                    record['coverage'][name]='RENDERED' if table['rows']==TABLE_ROWS[name] else 'PARTIAL: required rows absent'
                if common:
                    path=output/'supp-common-mask.tex'
                    path.write_text(('% CANARY; not paper values\n' if state['canary'] else '')+tex_table(
                        ['L/D','Centered B/D mismatch (%)','Common B/D mismatch (%)'],
                        [[r['domain_D'],r['centered_ratio_percent'],r['common_ratio_percent']] for r in common]),encoding='utf-8')
                    record['outputs'][path.name]={'sha256':digest(path),'kind':'table','inputs':'CSV + verified common-mask diagnostic'}
                    record['coverage']['supp-common-mask']='RENDERED' if len(common)==3 else 'PARTIAL: required domain controls absent'
                else:
                    record['coverage']['supp-common-mask']='UNAVAILABLE: --common-mask-input required'
                if len(common)!=3 and record['coverage'].get('domain-gains')=='RENDERED':
                    record['coverage']['domain-gains']='PARTIAL: common-mask overlay unavailable'
                for name in TABLE_ROWS:
                    record['coverage'].setdefault(name,'MISSING: required cases absent')
                if digest(folder/'result.json') != record['input_analysis_manifest_sha256']:
                    raise ValueError('Analysis manifest changed during rendering')
                load_analysis(folder)
                if build:
                    verify_bundle(build)
                    if digest(build/'result.json') != record['input_build_manifest_sha256']:
                        raise ValueError('Build manifest changed during rendering')
                if common_folder:
                    if digest(common_folder/'result.json') != record['input_common_manifest_sha256']:
                        raise ValueError('Common-mask manifest changed during rendering')
                    load_common(common_folder,state,tables,build)
            if source != {name:text_hash(ROOT/name) for name in source}:
                raise ValueError('Renderer source changed during execution')
            record['status']='COMPLETE_RENDER' if all(v=='RENDERED' for v in record['coverage'].values()) else 'PARTIAL_COVERAGE'
    except (Exception,KeyboardInterrupt) as error:
        record.update(status='FAILED',error=type(error).__name__+': '+str(error))
        plt.close('all')
    record.update(elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss_bytes())
    write_json(output/'result.json',record)
    print(json.dumps({key:record.get(key) for key in ('status','elapsed_seconds','peak_rss_bytes','error')}),flush=True)
    return 1 if record['status']=='FAILED' else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--input',type=Path); choice.add_argument('--analytic-only',action='store_true')
    parser.add_argument('--build-input',type=Path,help='Matching fresh build, needed for crossover panels')
    parser.add_argument('--common-mask-input',type=Path,help='Matching reproduce_common_mask.py result for updated domain figure/table')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    return render(args.input,args.output,args.build_input,args.common_mask_input)


if __name__=='__main__':
    raise SystemExit(main())
