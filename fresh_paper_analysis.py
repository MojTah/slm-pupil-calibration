"""Numeric tables and plot-ready curves from verified fresh-paper bundles.

This is a new calculation record, not a reconstruction of historical provenance.
No figure rendering or publication-readiness verdict is performed here.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from analyze_readout_sensitivity import dark, native
from finite_calibration import finite_poke, inverse_diagnostics
from gain_phase import gain_coefficients, gain_phase_range, gain_value
from phase_envelope import overlap_coefficients, phase_range
from phase_pair import paired_matrices
from piston_basis import readout_matrices, readout_moments
from reproduce_paper import ROOT, SOURCES, digest, verify_bundle
from run_pilot import WallDeadline, text_hash, write_json
from runtime_resources import peak_rss_bytes

KAPPAS = (0., .005, .01, .025, .05, .1, .2, .5, 1.)
POKES = (.005, .01, .025, .05, .1, .2)
EQUAL_POKES = (*POKES, .3, .4, .5)
PHOTONS = (10**6, 10**8, 10**10, 10**12)
PISTONS = np.linspace(-1., 1., 201)
ANALYSIS_SOURCES = (*SOURCES, 'fresh_paper_analysis.py', 'finite_calibration.py',
                    'phase_envelope.py', 'analyze_readout_sensitivity.py', 'analyze_transfer.py')


def gain_summary(matrices, target):
    extrema = gain_phase_range(gain_coefficients(matrices))
    if extrema['inversion_status'] != 'WELL_CONDITIONED':
        raise ValueError('Gain interval is not invertible')
    target_gain = readout_moments(target, 0.)['local_gain']
    pair_gain = readout_moments(paired_matrices(matrices, 0.), 0.)['local_gain']
    interval = sorted(100*(target_gain/extrema[k]-1) for k in ('minimum', 'maximum'))
    return {'g_target': target_gain, 'g_C_min': extrema['minimum'], 'g_C_max': extrema['maximum'],
            'g_pair': pair_gain, 'coherent_min_percent': interval[0], 'coherent_max_percent': interval[1],
            'paired_percent': 100*(target_gain/pair_gain-1)}, extrema


def finite_rows(matrices, target):
    result = []
    gx = readout_moments(target, 0.)['local_gain']
    for kappa in (0., .1, 1.):
        pooled = paired_matrices(matrices, kappa)
        ext = gain_phase_range(gain_coefficients(pooled))
        policies = [('phase_zero', 0.)]
        if kappa:
            policies += [('minimum_analytic_gain', ext['minimum_phase']),
                         ('maximum_analytic_gain', ext['maximum_phase'])]
        for policy, phase in policies:
            for h in EQUAL_POKES:
                slope = finite_poke(pooled, h, phase)
                target_secant = finite_poke(target, h)['finite_gain']
                for photons in PHOTONS:
                    original = inverse_diagnostics(slope, gx, photons)
                    equal = inverse_diagnostics(slope, target_secant, photons)
                    result.append({'kappa': kappa, 'policy': policy, 'phase_rad': phase,
                        'half_poke_rad': h, 'calibration_photons': photons, 'target_gain': gx,
                        'retained_local_poke': h in POKES,
                        'target_equal_poke_gain': target_secant, **slope, **original,
                        **{'equal_'+key: value for key, value in equal.items()}})
    return result


def best_finite(rows):
    best = []
    for metric in ('scale_rss_proxy', 'equal_scale_rss_proxy'):
        for photons in (10**8, 10**10, 10**12):
            for kappa in (0., 1.):
                candidates = []
                grid = POKES if metric == 'scale_rss_proxy' else EQUAL_POKES
                for h in grid:
                    group = [r for r in rows if (r['calibration_photons'], r['kappa'], r['half_poke_rad']) == (photons, kappa, h)]
                    if len(group) != (1 if kappa == 0 else 3):
                        raise ValueError('Finite diagnostic has missing/duplicate phase policies')
                    if all(row[metric] is not None for row in group):
                        candidates.append((max(row[metric] for row in group), h))
                value, h = min(candidates) if candidates else (None, None)
                best.append({'metric': metric, 'calibration_photons': photons, 'kappa': kappa,
                             'half_poke_rad': h, 'worst_selected_phase_percent': None if value is None else 100*value,
                             'grid_boundary': h in (grid[0], grid[-1]) if h is not None else None})
    return best


def sensitivity(left, right):
    curves = []
    changes = {'mean': 0., 'gain': 0., 'relative_variance': 0., 'phase_contrast': 0.}
    for label, phase, coherent in [('D', 0., True), ('incoherent', 0., False),
                                  ('C0', 0., True), ('Cpi/2', np.pi/2, True),
                                  ('Cpi', np.pi, True), ('C3pi/2', 3*np.pi/2, True)]:
        a_mat, b_mat = (dark(left), dark(right)) if label == 'D' else (left, right)
        for piston in PISTONS:
            a = readout_moments(a_mat, float(piston), phase, coherent)
            b = readout_moments(b_mat, float(piston), phase, coherent)
            ca = a['weighted_mean']-readout_moments(a_mat, float(piston), 0.)['weighted_mean']
            cb = b['weighted_mean']-readout_moments(b_mat, float(piston), 0.)['weighted_mean']
            if a['variance_coefficient'] <= 0:
                raise ValueError('Nonpositive baseline variance in sensitivity comparison')
            for key, value in [('mean', abs(b['weighted_mean']-a['weighted_mean'])),
                               ('gain', abs(b['local_gain']-a['local_gain'])),
                               ('relative_variance', abs(a['variance_coefficient']/b['variance_coefficient']-1))]:
                changes[key] = max(changes[key], value)
            if label in ('Cpi/2', 'Cpi', 'C3pi/2'):
                changes['phase_contrast'] = max(changes['phase_contrast'], abs(ca-cb))
            curves.append({'condition': label, 'piston_rad': float(piston),
                           'lower_phase_contrast_rad': ca, 'higher_phase_contrast_rad': cb,
                           'lower_minus_higher_rad': ca-cb})
    return changes, curves


def analyze(folder, output):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    state = verify_bundle(folder)
    identity = {name: text_hash(ROOT/name) for name in ANALYSIS_SOURCES}
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    runtime = {'python': platform.python_version(), 'numpy': np.__version__,
               'scipy': scipy.__version__, 'platform': platform.platform()}
    build_runtime = state['runtime']
    differences = {key: {'build': build_runtime[key], 'analysis': value}
                   for key, value in runtime.items()
                   if key in build_runtime and build_runtime[key] != value}
    result = {'status': 'RUNNING', 'canary': state['canary'], 'source_sha256_lf': identity,
              'input_manifest_sha256': digest(folder/'result.json'), 'outputs': {},
              'analysis_runtime': runtime, 'input_build_runtime': build_runtime,
              'runtime_policy': 'ALLOW_AND_RECORD', 'runtime_differences': differences,
              'runtime_uncompared_keys': [key for key in runtime if key not in build_runtime],
              'runtime_policy_reason': 'Postprocessing validates frozen input identities and records its own runtime. '
                  'Version differences are permitted; cross-version numerical or bitwise identity is not claimed.',
              'scope': 'Numeric tables and plot-ready data; no figures rendered',
              'finite_poke_grid': list(POKES), 'equal_poke_grid': list(EQUAL_POKES),
              'mask_convention': state['mask_convention']}
    write_json(output/'result.json', result)

    def save_csv(name, rows):
        if not rows:
            return
        with (output/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result['outputs'][name] = {'sha256': digest(output/name), 'rows': len(rows)}
        write_json(output/'result.json', result)

    cases = {record['case']['id']: record for record in state['cases']}

    def load(name):
        if digest(folder/cases[name]['artifact']) != cases[name]['sha256']:
            raise ValueError('Artifact changed before analysis: '+name)
        with np.load(folder/cases[name]['artifact']) as saved:
            return saved['gram']

    try:
        with WallDeadline(180):
            with np.load(folder/cases['reference-6d-128']['artifact']) as saved:
                weights = {'sampled_reference': saved['weights']}
            if 'original-15d-256' in cases:
                gram = load('original-15d-256')
                weights['native15_fixed'], tangent = native(gram)
                side = gram.shape[0]
                quadrant = np.broadcast_to(np.sign(np.arange(side)+.5-side/2)[None, :], (side, side)).copy()
                scale = np.sum(quadrant*tangent)
                if abs(scale) <= 1e-12:
                    raise ValueError('Quadrant readout has zero gain')
                weights['quadrant15_fixed'] = quadrant/scale
                del gram, tangent
            contracted = {}
            bright_hashes = {}
            for name, record in cases.items():
                if record['case']['command'] == 'reference':
                    continue
                gram = load(name)
                domain, angles = record['case']['geometry']['domain_diameters'], record['case']['nominal_angles']
                command = record['case']['command']
                if command in ('original', 'orthogonal'):
                    key = domain, angles
                    checksum = hashlib.sha256(gram[..., :2, :2].tobytes()).hexdigest()
                    if key in bright_hashes and bright_hashes[key] != checksum:
                        raise ValueError('Original/orthogonal bright blocks differ')
                    bright_hashes[key] = checksum
                contracted[name] = {label: readout_matrices(gram, w) for label, w in weights.items()}
                del gram
            rows, phase_curves, pairing, transfer_curves, finite, minima = [], [], [], [], [], []
            for name, by_weight in contracted.items():
                command = cases[name]['case']['command']
                if command == 'opaque':
                    continue
                domain = cases[name]['case']['geometry']['domain_diameters']
                angles = cases[name]['case']['nominal_angles']
                opaque = contracted.get(f'opaque-{domain}d-{angles}')
                for label, m in by_weight.items():
                    targets = {'D': dark(m)}
                    if opaque:
                        targets['B'] = opaque[label]
                    for target, tm in targets.items():
                        summary, ext = gain_summary(m, tm)
                        offset = phase_range(*overlap_coefficients(m, 0.))
                        zero = readout_moments(tm, 0.)['weighted_mean']
                        prefix = {'case': name, 'command': command, 'domain_D': domain,
                                  'angles': cases[name]['case']['angles'], 'nominal_angles': angles,
                                  'canary': state['canary'], 'readout': label, 'target': target}
                        a0 = overlap_coefficients(m, 0.)[0]
                        rows.append({**prefix, **summary, 'strip_power_fraction': float(m[0, 2, 2].real/a0),
                            'raw_offset_min_rad': offset['minimum']-zero, 'raw_offset_max_rad': offset['maximum']-zero,
                            'raw_offset_span_rad': offset['maximum']-offset['minimum']})
                        if label != 'sampled_reference':
                            continue
                        coeff = gain_coefficients(m)
                        for phase in np.linspace(0, 2*np.pi, 721):
                            phase_curves.append({**prefix, 'phase_rad': float(phase),
                                'scale_error_percent': 100*(summary['g_target']/gain_value(coeff, phase)-1),
                                'raw_offset_rad': readout_moments(m, 0., float(phase))['weighted_mean']-zero})
                        for kappa in KAPPAS:
                            cm = paired_matrices(m, kappa)
                            psummary, pext = gain_summary(cm, tm)
                            pair_prefix = {**prefix, 'kappa': kappa}
                            pairing.append({**pair_prefix, **psummary})
                            for piston in PISTONS:
                                response = readout_moments(tm, float(piston))['weighted_mean']-zero
                                gx = psummary['g_target']
                                vals = sorted(response*(1/pext[key]-1/gx) for key in ('minimum', 'maximum'))
                                transfer_curves.append({**pair_prefix, 'piston_rad': float(piston),
                                    'transfer_min_rad': vals[0], 'transfer_max_rad': vals[1],
                                    'target_self_error_rad': response/gx-float(piston)})
                        if command == 'original' and domain == 15 and angles == 256:
                            local_rows = finite_rows(m, tm)
                            finite += [{**prefix, **row} for row in local_rows]
                            minima += [{**prefix, **row} for row in best_finite(local_rows)]
            save_csv('transfer-summary.csv', rows)
            save_csv('phase-curves.csv', phase_curves)
            save_csv('pairing-summary.csv', pairing)
            save_csv('transfer-curves.csv', transfer_curves)
            save_csv('finite-calibration.csv', [row for row in finite if row['retained_local_poke']])
            save_csv('finite-equal-poke.csv', finite)
            save_csv('finite-minima.csv', minima)
            sensitivities, sensitivity_curves = [], []
            pairs = [('original-6d-128', 'original-6d-256'), ('original-15d-128', 'original-15d-256'),
                     ('original-6d-256', 'original-12d-256'), ('original-12d-256', 'original-15d-256'),
                     ('original-6d-256', 'original-15d-256')]
            for lo, hi in pairs:
                if lo not in contracted or hi not in contracted:
                    continue
                metrics, curves = sensitivity(contracted[lo]['sampled_reference'], contracted[hi]['sampled_reference'])
                sensitivities.append({'lower': lo, 'higher': hi, **metrics})
                sensitivity_curves += [{'lower': lo, 'higher': hi, **row} for row in curves]
            save_csv('sensitivity-summary.csv', sensitivities)
            save_csv('sensitivity-curves.csv', sensitivity_curves)
            result['finite_status_counts'] = dict(Counter(row['approximation_status'] for row in finite))
            result['uncovered'] = ['All PDF/PNG figure rendering and TeX table layout',
                'Analytic schematic and pair-tolerance figure outputs',
                'Any new mask convention or leakage/fill-factor model',
                'Full-size outputs are not verified by a canary']
            if state['profile'] != 'all':
                result['uncovered'].append('Profile omits opaque/orthogonal and other-domain cases; only representative15D numeric outputs available')
            if identity != {name: text_hash(ROOT/name) for name in ANALYSIS_SOURCES}:
                raise RuntimeError('Analysis source changed during execution')
            if digest(folder/'result.json') != result['input_manifest_sha256']:
                raise RuntimeError('Input manifest changed during analysis')
            verify_bundle(folder)
            if peak_rss_bytes() > 2_000_000_000:
                raise MemoryError('Observed analysis peak exceeds2GB')
        result['status'] = 'COMPLETE_NUMERIC_ANALYSIS'
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='PARTIAL', error=type(error).__name__+': '+str(error))
    result.update(elapsed_seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss_bytes())
    write_json(output/'result.json', result)
    print(json.dumps({key: result.get(key) for key in ('status', 'elapsed_seconds', 'peak_rss_bytes', 'error')}), flush=True)
    return 0 if result['status'] == 'COMPLETE_NUMERIC_ANALYSIS' else 1
