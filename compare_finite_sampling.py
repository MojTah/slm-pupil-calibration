"""Compare sampling, domain or modulation quadrature with a fixed D readout."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil

from piston_basis import evaluate, gram_diagnostics, readout_matrices, readout_moments
from phase_envelope import phase_range, overlap_coefficients
from simulation import normalize_response

ROOT = Path(__file__).resolve().parent


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axis', choices=('sampling', 'domain', 'quadrature'), default='sampling')
    parser.add_argument('--lower', default='finite-s2-domain6-128')
    parser.add_argument('--higher')
    parser.add_argument('--readout-run')
    parser.add_argument('--output-stem')
    args = parser.parse_args()
    defaults = {'sampling': 'finite-s3-domain6-128', 'domain': 'finite-s2-domain8-128',
                'quadrature': 'finite-s2-domain6-256'}
    names = [args.lower, args.higher or defaults[args.axis]]
    readout_name = args.readout_run or (names[1] if args.axis == 'sampling' else 'finite-s3-domain6-128')
    stem = args.output_stem or args.axis+'-comparison'
    if any(Path(n).name != n or n in ('.', '..') for n in [*names, readout_name, stem]):
        parser.error('Require local run directory names')
    target = ROOT/'local-results'/names[1]/(stem+'.json')
    if target.exists():
        raise RuntimeError('Refusing to overwrite comparison')
    cache = {}
    for name in dict.fromkeys([*names, readout_name]):
        folder = ROOT/'local-results'/name
        manifest = json.loads((folder/'result.json').read_text())
        if manifest['status'] != 'COMPLETE_VALIDATED':
            raise ValueError('Incomplete basis run: '+name)
        for source in ('simulation.py', 'piston_basis.py'):
            if hashlib.sha256((ROOT/source).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != manifest['identity'][source]:
                raise ValueError('Optical/basis source identity mismatch')
        if manifest.get('model_kind') == 'exact_held_pixel_modal_v1':
            if hashlib.sha256((ROOT/'modal_simulation.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest() != manifest['identity']['modal_simulation.py']:
                raise ValueError('Modal source identity mismatch')
        block, = [b for b in manifest['blocks'] if b['family'] == 'SLM']
        artifact = folder/block['artifact']
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != block['sha256']:
            raise ValueError('Artifact hash mismatch')
        with np.load(artifact) as data:
            gram = data['gram']
        gram_diagnostics(gram)
        dark_off = gram.copy()
        dark_off[..., 2, :] = 0
        dark_off[..., :, 2] = 0
        cache[name] = (manifest, gram, dark_off)
    loaded = [cache[name] for name in names]
    kinds = [v[0].get('model_kind', 'sampled_finite_array') for v in loaded]
    if kinds[0] != kinds[1]:
        raise ValueError('Different optical operators require a separate model comparison')
    modal = kinds[0] == 'exact_held_pixel_modal_v1'
    if modal and args.axis == 'sampling':
        raise ValueError('Modal formulation has no subpixel sampling axis')
    lower, higher = [v[0]['config'] for v in loaded]
    varying = {'sampling': 'samples_per_pixel', 'domain': 'domain_diameters', 'quadrature': 'angles'}[args.axis]
    if any(lower[k] != higher[k] for k in ('iris', 'modulation', 'angles', 'detector_diameters') if k != varying) or any(
            v != higher['geometry'][k] for k, v in lower['geometry'].items() if k != varying):
        raise ValueError('Physical settings differ')
    actual_pair = [c['angles'] if args.axis == 'quadrature' else c['geometry'][varying] for c in (lower, higher)]
    admitted_pairs = {'sampling': [[2, 3]], 'domain': [[6, 8], [8, 9], [6, 9]],
                      'quadrature': [[128, 256]]}[args.axis]
    if modal and args.axis == 'domain':
        admitted_pairs = admitted_pairs+[[6, 12], [12, 15], [6, 15]]
    if actual_pair not in admitted_pairs:
        raise ValueError('Require the frozen sampling or domain pair')
    reference_config = cache[readout_name][0]['config']
    if any(reference_config[k] != lower[k] for k in ('iris', 'modulation', 'angles', 'detector_diameters') if not (modal and k == 'angles')) or any(
            v != reference_config['geometry'][k] for k, v in lower['geometry'].items()
            if k not in ('samples_per_pixel', 'domain_diameters')):
        raise ValueError('Readout physical settings differ')
    if modal and reference_config['angles'] != 128:
        raise ValueError('Modal comparisons retain the frozen 128-angle readout')
    if args.axis == 'domain' and (lower['geometry']['samples_per_pixel'] != (1 if modal else 2) or
            reference_config['geometry']['samples_per_pixel'] != 3 or reference_config['geometry']['domain_diameters'] != 6):
        raise ValueError('Domain comparison requires s2 fields and frozen s3/6D readout')
    if args.axis == 'quadrature' and (lower['geometry']['samples_per_pixel'] != (1 if modal else 2) or
            lower['geometry']['domain_diameters'] not in ([6, 15] if modal else [6]) or
            reference_config['geometry']['samples_per_pixel'] != 3 or
            reference_config['geometry']['domain_diameters'] != 6):
        raise ValueError('Require frozen quadrature domain and s3/6D/128 readout')
    y_ref, j_ref = normalize_response(*evaluate(cache[readout_name][2], 0.))
    weights = j_ref/np.sum(j_ref*j_ref)
    origin = float(np.sum(weights*y_ref))
    piston = np.linspace(-1., 1., 201)
    conditions = [('D', 0., True), ('incoherent', 0., False),
                  ('C0', 0., True), ('Cpi/2', np.pi/2, True),
                  ('Cpi', np.pi, True), ('C3pi/2', 3*np.pi/2, True)]
    curves, native, identities, rows, envelopes = [], [], [], [], []
    for name, (manifest, gram, dark_off) in zip(names, loaded):
        y_d, j_d = normalize_response(*evaluate(dark_off, 0.))
        native_w = j_d/np.sum(j_d*j_d)
        native_origin = float(np.sum(native_w*y_d))
        native_c = readout_moments(readout_matrices(gram, native_w), 0.)
        native.append({'run': name, 'C0_zero_offset_rad': native_c['weighted_mean']-native_origin,
                       'C0_zero_gain': native_c['local_gain'],
                       'shadow_only_detected_power': float(gram[..., 2, 2].real.sum()),
                       'shadow_to_D_detected_power': float(gram[..., 2, 2].real.sum()/evaluate(dark_off, 0.)[0].sum())})
        current = {}
        for label, phi, coherent in conditions:
            matrices = readout_matrices(dark_off if label == 'D' else gram, weights)
            current[label] = []
            for a in piston:
                m = readout_moments(matrices, float(a), float(phi), coherent)
                flux = float(evaluate(matrices[0], float(a), float(phi), coherent)[0])
                values = (m['weighted_mean']-origin, m['local_gain'], m['variance_coefficient'], flux)
                if not np.isfinite(values).all() or flux <= 0 or values[2] <= 0:
                    raise ValueError('Nonphysical or nonfinite curve')
                row = {'run': name, 'condition': label, 'piston_rad': float(a),
                       **dict(zip(('raw_readout_rad', 'local_gain', 'variance_coefficient', 'detected_flux'), values))}
                current[label].append(row)
                rows.append(row)
        max_mean, max_flux = 0., 0.
        for i in range(len(piston)):
            inc = current['incoherent'][i]
            for first, opposite in [('C0', 'Cpi'), ('Cpi/2', 'C3pi/2')]:
                a, b = current[first][i], current[opposite][i]
                total = a['detected_flux']+b['detected_flux']
                mean = (a['detected_flux']*a['raw_readout_rad']+b['detected_flux']*b['raw_readout_rad'])/total
                max_mean = max(max_mean, abs(mean-inc['raw_readout_rad']))
                max_flux = max(max_flux, abs(total/2-inc['detected_flux'])/inc['detected_flux'])
        if max_mean > 1e-10 or max_flux > 1e-12:
            raise ValueError('Incoherent identity failed')
        identities.append({'run': name, 'max_raw_mean_identity_error_rad': max_mean,
                           'max_flux_identity_relative_error': max_flux})
        curves.append(current)
        if args.axis == 'quadrature':
            matrices = readout_matrices(gram, weights)
            envelopes.append([phase_range(*overlap_coefficients(matrices, float(a))) for a in piston])
    comparisons, phase_checks = [], []
    for label, _, _ in conditions:
        a, b = curves[0][label], curves[1][label]
        dr = max(abs(x['raw_readout_rad']-y['raw_readout_rad']) for x, y in zip(a, b))
        dg = max(abs(x['local_gain']-y['local_gain']) for x, y in zip(a, b))
        dv = max(abs(x['variance_coefficient']/y['variance_coefficient']-1) for x, y in zip(a, b))
        comparisons.append({'condition': label, 'max_readout_change_rad': dr, 'max_gain_change': dg,
                            'max_relative_variance_coefficient_change': dv,
                            'targets_met': dr < .005 and dg < .001 and dv < .01})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for color, label in zip(('tab:blue', 'tab:orange', 'tab:green'), ('Cpi/2', 'Cpi', 'C3pi/2')):
        contrast = []
        for curve in curves:
            contrast.append(np.array([x['raw_readout_rad']-base['raw_readout_rad']
                                      for x, base in zip(curve[label], curve['C0'])]))
        difference = contrast[0]-contrast[1]
        maximum = float(np.max(abs(difference)))
        phase_checks.append({'condition': label, 'maximum_absolute_intergrid_difference_rad': maximum,
                             'targets_met': maximum < .005,
                             'higher_grid_contrast_range_rad': [float(contrast[1].min()), float(contrast[1].max())]})
        lower_label, higher_label = (('s2', 's3') if args.axis == 'sampling' else
            tuple(str(v)+(' angles' if args.axis == 'quadrature' else 'D') for v in actual_pair))
        axes[0].plot(piston, contrast[1], color=color, label=label+' '+higher_label)
        axes[0].plot(piston, contrast[0], color=color, linestyle='--', linewidth=1, label=label+' '+lower_label)
        axes[1].plot(piston, difference, color=color, label=label)
    axes[1].axhspan(-.005, .005, color='gray', alpha=.15, label='Diagnostic target')
    axes[0].set_ylabel('Phase contrast relative to phase 0 (rad)')
    axes[1].set_ylabel('Phase-contrast difference (rad)')
    axes[1].set_title(lower_label+' minus '+higher_label, fontsize=10)
    for ax in axes:
        ax.set_xlabel('Piston a (rad)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    ref_geometry = reference_config['geometry']
    fig.suptitle(('Modal finite-piston ' if modal else 'Finite-piston ')+args.axis+' check; common s'+str(ref_geometry['samples_per_pixel'])+
                 '/'+str(ref_geometry['domain_diameters'])+'D D readout', fontsize=11)
    folder = target.parent
    fig.savefig(folder/(stem+'.png'), dpi=170)
    fig.savefig(folder/(stem+'.pdf'))
    plt.close(fig)
    with (folder/(stem+'.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    primary = all(c['targets_met'] for c in phase_checks)
    phase = ('finite-quadrature-14' if args.axis == 'quadrature' else
        'finite-sampling-8' if args.axis == 'sampling' else ('finite-domain-9' if actual_pair == [6, 8] else 'finite-domain-10'))
    result = {'phase': phase, 'axis': args.axis, 'refinement_pair': actual_pair, 'model_kind': kinds[0],
              'verdict': 'TARGETS_MET' if primary and all(c['targets_met'] for c in comparisons) else 'REFINEMENT_REQUIRED',
              'primary_phase_contrast_targets_met': primary, 'phase_contrasts': phase_checks,
              'common_readout_comparisons': comparisons, 'grid_native_D_calibrations': native,
              'incoherent_checks': identities, 'rows': len(rows),
              'readout': 'One D weight map and reference origin applied to both grids', 'readout_run': readout_name,
              'limitations': ['Successive differences are not continuum error bounds',
                              'Finite domain sensitivity and higher-sampling C/B transfer remain open'],
              'run_result_hashes': {n: hashlib.sha256((ROOT/'local-results'/n/'result.json').read_bytes()).hexdigest() for n in cache},
              'source_hashes_lf': {n: hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                   for n in ('compare_finite_sampling.py', 'piston_basis.py', 'simulation.py', 'phase_envelope.py')},
              'artifacts': {stem+extension: hashlib.sha256((folder/(stem+extension)).read_bytes()).hexdigest()
                            for extension in ('.png', '.pdf', '.csv')},
              'elapsed_seconds': time.perf_counter()-start, 'peak_rss_bytes': psutil.Process().memory_info().peak_wset}
    if envelopes:
        result['continuous_phase_checks'] = {
            **{'maximum_'+end+'_change_rad': max(abs(a[end]-b[end]) for a,b in zip(*envelopes))
               for end in ('minimum', 'maximum')},
            'maximum_width_change_rad': max(abs((a['maximum']-a['minimum'])-(b['maximum']-b['minimum']))
                                            for a,b in zip(*envelopes)),
            'minimum_relative_flux': [min(r['relative_minimum_flux'] for r in e) for e in envelopes],
            'interpretation': 'Exploratory mean envelope; not gain/noise extrema or continuum bounds'}
    if modal:
        result['source_hashes_lf']['modal_simulation.py'] = hashlib.sha256(
            (ROOT/'modal_simulation.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        result['limitations'].append('Distinct modal boundary prescription; old Bessel angular certificate does not apply')
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'primary_phase_contrast_targets_met', 'phase_contrasts', 'common_readout_comparisons', 'grid_native_D_calibrations', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
