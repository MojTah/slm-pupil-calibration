"""Build fresh manuscript Grams, or analyze a verified fresh bundle.

No historical array/configuration/provenance record is an input. Completed cases
can be reused explicitly from an unchanged-source run, including a partial run.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import psutil
import scipy
from modal_simulation import ModalModel, build_modal_basis
from orthogonal_carrier import OrthogonalStripModel
from reproduce_6d import fixed_readout
from run_modal import direct_check
from run_pilot import WallDeadline, text_hash, write_json
from runtime_resources import peak_rss_bytes
from piston_basis import build_basis
from simulation import Configuration, OpticalModel

ROOT = Path(__file__).resolve().parent
SOURCES = ('reproduce_paper.py', 'reproduce_6d.py', 'modal_simulation.py',
           'orthogonal_carrier.py', 'simulation.py', 'piston_basis.py',
           'run_modal.py', 'run_pilot.py', 'runtime_resources.py',
           'gain_phase.py', 'phase_pair.py')
MEMORY_LIMIT = 2_000_000_000


class BrightReference(OpticalModel):
    """Sampled bright-only D reference; the omitted strip block stays zero.

    The B family selects two fields in build_basis. Its relay is D here,
    retaining the pixel-held carrier and off-axis iris of the fixed readout.
    This reference is not the centered-iris physical B target.
    """
    def relay(self, case, piston=0., iris_radius=8.):
        return super().relay('D' if case == 'B' else case, piston, iris_radius)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_plan(profile, canary=False):
    cases = [('reference', 6, 128)]
    if profile == 'representative-15d':
        cases += [('original', 15, 256)]
    elif profile == 'all':
        cases += [('original', domain, angles) for domain, angles in
                  [(6, 128), (6, 256), (12, 256), (15, 128), (15, 256)]]
        cases += [(command, domain, 256) for command in ('opaque', 'orthogonal') for domain in (6, 12, 15)]
        cases += [('opaque', 15, 128), ('orthogonal', 6, 128)]
    else:
        raise ValueError('Unknown fixed profile')
    result = []
    for command, domain, angles in cases:
        geometry = dict(diameter_pixels=8 if canary else 128, spider_pixels=2 if canary else 8,
                        samples_per_pixel=3 if command == 'reference' else 1,
                        domain_diameters=domain, bright_carrier=.125,
                        dark_carrier=-.125, pupil_center_diameters=.75)
        cfg = Configuration(**geometry)
        n = cfg.diameter_pixels*domain*cfg.samples_per_pixel
        side = cfg.diameter_pixels*4
        # Sequential publication retains two cropped output buffers, not three.
        estimate = n*n*16*(20 if command == 'reference' else 32)+side*side*9*16*2
        result.append({'id': f'{command}-{domain}d-{angles}', 'command': command,
                       'geometry': geometry, 'angles': 8 if canary else angles,
                       'nominal_angles': angles, 'iris': 1. if canary else 8.,
                       'modulation': .73 if canary else 3., 'detector_diameters': 4,
                       'estimated_bytes': estimate,
                       'stage_cap_seconds': 60 if canary else 720 if command == 'reference'
                       else 300 if domain == 6 else 660 if domain == 12 else 1200})
    return result


def verify_bundle(folder, require_complete=True):
    folder = Path(folder).resolve()
    state = json.loads((folder/'result.json').read_text())
    if state.get('schema') != 'fresh-paper-v1':
        raise ValueError('Not a fresh-paper bundle')
    if state.get('status') not in ('COMPLETE_VALIDATED', 'PARTIAL'):
        raise ValueError('Bundle is active or has an unknown lifecycle state; reuse is refused')
    if require_complete and state['status'] != 'COMPLETE_VALIDATED':
        raise ValueError('Input bundle is incomplete')
    if state['source_sha256_lf'] != {name: text_hash(ROOT/name) for name in SOURCES}:
        raise ValueError('Execution source identity differs; reuse is refused')
    expected = {case['id']: case for case in case_plan(state['profile'], state['canary'])}
    seen = set()
    for record in state['cases']:
        ident = record['case']['id']
        if ident in seen or record['case'] != expected.get(ident):
            raise ValueError('Case identity is duplicate or does not match its fixed definition')
        seen.add(ident)
        name = record['artifact']
        if name != ident+'.npz' or (folder/name).resolve().parent != folder:
            raise ValueError('Invalid artifact path')
        if digest(folder/name) != record['sha256']:
            raise ValueError('Artifact checksum mismatch: '+name)
    if require_complete and seen != set(expected):
        raise ValueError('Completed bundle lacks required cases')
    return state


def build(output, profile, wall_budget, canary=False, reuse=None):
    if not np.isfinite(wall_budget) or not 0 < wall_budget <= 10800:
        raise ValueError('Require wall budget in (0,10800] seconds')
    plan = case_plan(profile, canary)
    old = verify_bundle(reuse, require_complete=False) if reuse else None
    if old and old['canary'] != canary:
        raise ValueError('Cannot reuse canary artifacts as paper results')
    runtime = {'python': platform.python_version(), 'numpy': np.__version__,
               'scipy': scipy.__version__, 'psutil': psutil.__version__, 'fft_workers': 1}
    if old and old['runtime'] != runtime:
        raise ValueError('Reuse requires unchanged runtime versions')
    old_cases = {r['case']['id']: r for r in old['cases']} if old else {}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    identity = {name: text_hash(ROOT/name) for name in SOURCES}
    state = {'schema': 'fresh-paper-v1', 'status': 'RUNNING', 'profile': profile,
             'canary': canary, 'source_sha256_lf': identity, 'cases': [],
             'started_utc': datetime.now(timezone.utc).isoformat(), 'process_id': os.getpid(),
             'wall_budget_seconds': wall_budget, 'memory_limit_bytes': MEMORY_LIMIT,
             'runtime': runtime,
             'reference_construction': 'Sampled bright-only D; strip Gram row/column omitted as zeros; '
                 'same fixed readout as the bright block of the full SLM reference',
             'mask_convention': 'Published separate centered/translated floating-point masks',
             'scope': 'Fresh production-operator reproduction; no optical convergence or hardware claim'}

    def save():
        state.update(elapsed_seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'result.json', state)

    deadline = None
    try:
        save()
        for case in plan:
            state['active_case'] = case['id']
            save()
            if wall_budget-(time.perf_counter()-start) <= 0:
                raise TimeoutError('Whole-run deadline reached')
            previous = old_cases.get(case['id'])
            if previous:
                if previous['case'] != case:
                    raise ValueError('Reused case definition changed')
                shutil.copyfile(Path(reuse)/previous['artifact'], output/previous['artifact'])
                if digest(output/previous['artifact']) != previous['sha256']:
                    raise ValueError('Copied artifact checksum mismatch')
                state['cases'].append({**previous, 'reused_from_manifest_sha256': digest(Path(reuse)/'result.json')})
                save()
                continue
            if case['estimated_bytes'] > MEMORY_LIMIT or psutil.virtual_memory().available < 2*case['estimated_bytes']:
                raise MemoryError('Memory admission failed before '+case['id'])
            remaining = wall_budget-(time.perf_counter()-start)
            if remaining <= 0:
                raise TimeoutError('Whole-run deadline reached')
            print(json.dumps({'case_started': case['id'], 'estimated_bytes': case['estimated_bytes']}), flush=True)
            t0 = time.perf_counter()
            deadline = WallDeadline(min(remaining, case['stage_cap_seconds']))
            with deadline:
                cfg = Configuration(**case['geometry'])
                if case['command'] == 'reference':
                    model = BrightReference(cfg)
                    basis = build_basis(model, 'B', case['iris'], case['modulation'], case['angles'], 4)
                    arrays = {key: value for key, value in basis.items() if key != 'diagnostics'}
                    arrays['weights'] = fixed_readout(basis['gram'])
                    checks = {'weights_sha256': hashlib.sha256(arrays['weights'].tobytes()).hexdigest(),
                              'reference_relay': 'D', 'reference_strip_terms': 'OMITTED_ZERO'}
                else:
                    model = (OrthogonalStripModel if case['command'] == 'orthogonal' else ModalModel)(cfg, case['iris'])
                    family = 'B' if case['command'] == 'opaque' else 'SLM'
                    basis = build_modal_basis(model, family, case['modulation'], case['angles'], 4)
                    checks = {'direct': direct_check(model, basis, 'B' if family == 'B' else 'C',
                              -.37, 0. if family == 'B' else np.pi/2, case)}
                    arrays = {key: value for key, value in basis.items() if key != 'diagnostics'}
                if peak_rss_bytes() > MEMORY_LIMIT:
                    raise MemoryError('Observed peak exceeds2GB before publication')
                artifact = case['id']+'.npz'
                path = output/artifact
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle, **arrays)
                path.with_suffix('.tmp').replace(path)
                record = {'case': case, 'artifact': artifact, 'sha256': digest(path),
                          'diagnostics': basis['diagnostics'], 'checks': checks,
                          'execution_runtime': runtime,
                          'elapsed_seconds': time.perf_counter()-t0}
            state['cases'].append(record)
            del basis, arrays, model
            gc.collect()
            save()
            if state['peak_rss_bytes'] > MEMORY_LIMIT:
                raise MemoryError('Observed peak exceeds2GB; completed artifacts retained')
        if identity != {name: text_hash(ROOT/name) for name in SOURCES}:
            raise RuntimeError('Source changed during execution')
        if time.perf_counter()-start > wall_budget:
            raise TimeoutError('Whole-run deadline reached before finalization')
        state['status'] = 'COMPLETE_VALIDATED'
        state.pop('active_case', None)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='PARTIAL', error=type(error).__name__+': '+str(error),
                     deadline_expired=deadline.expired.is_set() if deadline else False)
    finally:
        save()
    print(json.dumps({k: state.get(k) for k in ('status', 'elapsed_seconds', 'peak_rss_bytes', 'error')}), flush=True)
    return 0 if state['status'] == 'COMPLETE_VALIDATED' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    builder = actions.add_parser('build')
    builder.add_argument('--profile', choices=('all', 'representative-15d'), required=True)
    builder.add_argument('--output', type=Path, required=True)
    builder.add_argument('--wall-budget', type=float, required=True)
    builder.add_argument('--canary', action='store_true')
    builder.add_argument('--reuse', type=Path)
    analysis = actions.add_parser('analyze')
    analysis.add_argument('--input', type=Path, required=True)
    analysis.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == 'build':
            return build(args.output, args.profile, args.wall_budget, args.canary, args.reuse)
        from fresh_paper_analysis import analyze
        return analyze(args.input, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
