"""Finite calibration-poke and photon-budget comparison from verified arrays."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import psutil
from analyze_transfer import DEFAULT_RUNS, B_TARGET_RUNS, load_gram, file_hash
from finite_calibration import finite_poke, inverse_diagnostics, compare_diagnostics
from phase_pair import paired_matrices
from piston_basis import evaluate, readout_matrices, readout_moments
from run_pilot import WallDeadline
from simulation import normalize_response

ROOT = Path(__file__).resolve().parent
POKES = [.005, .01, .025, .05, .1, .2]
PHOTONS = [10**6, 10**8, 10**10, 10**12]
KAPPAS = [0., .1, 1.]
SOURCES = ['analyze_finite_calibration.py', 'finite_calibration.py',
           'test_finite_calibration.py', 'analyze_transfer.py', 'phase_pair.py',
           'piston_basis.py', 'simulation.py', 'modal_simulation.py', 'run_pilot.py']


def lf_hash(name):
    return hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    out = (ROOT/args.run_dir).resolve()
    if not out.is_relative_to((ROOT/'local-results').resolve()) or out.exists():
        parser.error('Require a fresh study result directory')
    if len(set(args.runs)) != len(args.runs) or any(n not in DEFAULT_RUNS for n in args.runs):
        parser.error('Require a unique subset of the five frozen modal runs')
    estimated = 512*512*9*16*6+512*512*8*5
    if estimated > 2_000_000_000 or psutil.virtual_memory().available < 2*estimated:
        raise MemoryError('Saved-array memory admission failed')
    tests = json.loads((ROOT/'finite-calibration-tests.json').read_text())
    if tests['status'] != 'PASS' or any(lf_hash(n) != h for n,h in tests['sources_lf'].items()):
        raise ValueError('Failed or stale finite-calibration fixtures')
    prior_path = ROOT/'local-results/phase-pair-20260915/result.json'
    prior = json.loads(prior_path.read_text())
    reviewed = json.loads((ROOT/'opaque-pair-summary.json').read_text())
    if prior['status'] != 'COMPLETE_ANALYSIS' or file_hash(prior_path) != reviewed['pair_result_sha256']:
        raise ValueError('Prior reviewed pairing packet changed')
    source_ids = {n:lf_hash(n) for n in SOURCES}
    test_hash = file_hash(ROOT/'finite-calibration-tests.json')
    start = time.perf_counter()
    state = {'status':'RUNNING', 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'sources_lf':source_ids, 'test_sha256':test_hash,
             'prior_pair_result_sha256':file_hash(prior_path), 'runs':args.runs}
    out.mkdir()
    (out/'result.json').write_text(json.dumps(state, indent=2)+'\n')
    try:
        with WallDeadline(30):
            snapshot = out/'source-snapshot'; snapshot.mkdir()
            for n in SOURCES+['finite-calibration-tests.json']:
                (snapshot/n).write_bytes((ROOT/n).read_bytes())
            _, ref, ref_id = load_gram('finite-s3-domain6-128')
            if ref_id != prior['reference']: raise ValueError('Reference changed')
            ref[...,2,:] = 0; ref[..., :,2] = 0
            _, j = normalize_response(*evaluate(ref, 0.))
            w = j/np.sum(j*j); del ref, j
            wh = hashlib.sha256(w.tobytes()).hexdigest()
            if wh != prior['weight_sha256']: raise ValueError('Frozen readout changed')
            rows, inputs, regressions = [], {}, []
            for name in args.runs:
                _, gram, identity = load_gram(name)
                if identity != prior['inputs'][name]: raise ValueError('SLM input changed')
                inputs[name] = identity
                mat = readout_matrices(gram, w); del gram
                d = mat.copy(); d[...,2,:] = 0; d[..., :,2] = 0
                target_gains = {'D':readout_moments(d, 0.)['local_gain']}
                if name in B_TARGET_RUNS:
                    _, bg, bid = load_gram(B_TARGET_RUNS[name], 'B')
                    if bid != prior['inputs'][name+'_B']: raise ValueError('B input changed')
                    if np.any(bg[...,2,:] != 0) or np.any(bg[..., :,2] != 0):
                        raise ValueError('B contains shadow terms')
                    target_gains['B'] = readout_moments(readout_matrices(bg, w), 0.)['local_gain']
                    del bg; inputs[name+'_B'] = bid
                for kappa in KAPPAS:
                    prior_row = next(v for v in prior['records'] if
                                     (v['run'],v['target'],v['kappa']) == (name,'D',kappa))
                    ext = prior_row['gain_range']
                    cm = paired_matrices(mat, kappa)
                    policies = [('phase_zero',0.)]
                    if kappa != 0:
                        policies += [('minimum_analytic_gain',ext['minimum_phase']),
                                     ('maximum_analytic_gain',ext['maximum_phase'])]
                    for label, phase in policies:
                        for h in POKES:
                            slope = finite_poke(cm, h, phase)
                            if label != 'phase_zero':
                                expected = ext['minimum' if label.startswith('minimum') else 'maximum']
                                residual = abs(slope['analytic_gain']-expected)
                                if residual > 1e-12: raise ValueError('Analytic gain regression failed')
                                regressions.append(residual)
                            for target, gx in target_gains.items():
                                previous_target = next(v for v in prior['records'] if
                                                       (v['run'],v['target'],v['kappa']) == (name,target,kappa))
                                if abs(gx-previous_target['target_gain']) > 1e-12:
                                    raise ValueError('Target gain regression failed')
                                for n in PHOTONS:
                                    diagnostics = inverse_diagnostics(slope, gx, n)
                                    rows.append(dict(run=name, target=target, kappa=kappa,
                                                     policy=label, phase_rad=phase, half_poke_rad=h,
                                                     calibration_photons=n, target_gain=gx, **slope, **diagnostics))
            index = {(r['run'],r['target'],r['kappa'],r['policy'],r['half_poke_rad'],r['calibration_photons']):r for r in rows}
            comparisons = []
            for lower,higher in [(DEFAULT_RUNS[1],DEFAULT_RUNS[2]),(DEFAULT_RUNS[2],DEFAULT_RUNS[4]),
                                 (DEFAULT_RUNS[1],DEFAULT_RUNS[4]),(DEFAULT_RUNS[3],DEFAULT_RUNS[4])]:
                if lower not in args.runs or higher not in args.runs: continue
                changes = []
                for key,a in index.items():
                    if key[0] != lower: continue
                    b = index[(higher,)+key[1:]]
                    changes.append({'target':a['target'], 'kappa':a['kappa'], 'policy':a['policy'],
                                    'half_poke_rad':a['half_poke_rad'], 'calibration_photons':a['calibration_photons'],
                                    'phase_change_rad':b['phase_rad']-a['phase_rad'],
                                    **compare_diagnostics(a,b)})
                scale = [v['scale_error_change'] for v in changes if v['scale_error_change'] is not None]
                variance = [v['relative_gain_variance_coefficient_change'] for v in changes if v['relative_gain_variance_coefficient_change'] is not None]
                comparisons.append({'lower':lower, 'higher':higher, 'records':changes,
                                    'scale_comparison_coverage':[len(scale),len(changes)],
                                    'variance_comparison_coverage':[len(variance),len(changes)],
                                    'maximum_scale_error_change':max(scale,default=None),
                                    'maximum_relative_gain_variance_coefficient_change':max(variance,default=None)})
            if any(lf_hash(n) != h for n,h in source_ids.items()): raise ValueError('Source drift')
            if file_hash(ROOT/'finite-calibration-tests.json') != test_hash: raise ValueError('Tests changed')
            if file_hash(prior_path) != state['prior_pair_result_sha256']: raise ValueError('Prior packet changed')
            peak = psutil.Process().memory_info().peak_wset
            if peak > 2_000_000_000: raise MemoryError('Observed peak exceeds2GB')
            with (out/'curves.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            state.update(status='COMPLETE_ANALYSIS', rows=len(rows), inputs=inputs, reference=ref_id,
                         weight_sha256=wh, curves_sha256=file_hash(out/'curves.csv'),
                         comparisons=comparisons, maximum_analytic_gain_regression=max(regressions,default=0.),
                         reporting_status_counts={s:sum(r['approximation_status']==s for r in rows) for s in sorted({r['approximation_status'] for r in rows})},
                         elapsed_seconds=time.perf_counter()-start, peak_rss_bytes=peak,
                         interpretation='Selected phases only. Fixed pooled detected counts per piston. Slope moments exact under conditional multinomial sampling. Reciprocal quantities are local expansions, not an operational estimator or exact MSE; CV cutoff is only a reporting convention.')
            (out/'result.json').write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    except BaseException as exc:
        state.update(status='PARTIAL', error=repr(exc), elapsed_seconds=time.perf_counter()-start)
        (out/'result.json').write_text(json.dumps(state, indent=2)+'\n')
        raise
    print(json.dumps({k:state[k] for k in ['status','rows','elapsed_seconds','peak_rss_bytes','reporting_status_counts']}, indent=2))


if __name__ == '__main__': main()
