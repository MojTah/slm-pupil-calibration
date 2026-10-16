"""Fresh code-only 6D regeneration of the matched command comparison in Supplement S8.

No archived inputs are read. Outputs are local; an existing directory is refused.
Deadlines interrupt at Python boundaries, and memory checks are not OS limits.
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
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import psutil
import scipy
from scipy.optimize import minimize_scalar

from gain_phase import gain_coefficients, gain_phase_range
from modal_simulation import ModalModel, build_modal_basis
from orthogonal_carrier import OrthogonalStripModel
from phase_pair import paired_matrices
from piston_basis import build_basis, evaluate, readout_matrices, readout_moments
from run_modal import direct_check
from run_pilot import WallDeadline, text_hash, write_json
from runtime_resources import peak_rss_bytes
from simulation import Configuration, OpticalModel, normalize_response

ROOT = Path(__file__).resolve().parent
SOURCES = ('reproduce_6d.py', 'simulation.py', 'piston_basis.py',
           'modal_simulation.py', 'orthogonal_carrier.py', 'gain_phase.py',
           'phase_pair.py', 'run_modal.py', 'run_pilot.py', 'runtime_resources.py')
MEMORY_LIMIT = 2_000_000_000


def fixed_readout(gram):
    """The sampled-reference bright-only tangent, never refit to a modal case."""
    dark = gram.copy()
    dark[..., 2, :] = 0
    dark[..., :, 2] = 0
    _, tangent = normalize_response(*evaluate(dark, 0.))
    norm = np.sum(tangent*tangent)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Reference tangent must have positive finite norm')
    return tangent/norm


def summarize(matrices, opaque):
    """Contract fresh Grams; independently check extrema with quadratic forms."""
    rows, checks = [], {}
    g_b = readout_moments(opaque, 0.)['local_gain']
    for command, m in matrices.items():
        dark = m.copy()
        dark[..., 2, :] = 0
        dark[..., :, 2] = 0
        g_d = readout_moments(dark, 0.)['local_gain']
        extrema = gain_phase_range(gain_coefficients(m))
        if extrema['inversion_status'] != 'WELL_CONDITIONED':
            raise ValueError('Gain inversion is not well conditioned')
        pair = readout_moments(paired_matrices(m, 0.), 0.)['local_gain']

        def direct_gain(phi):
            u = np.array([1., 0., np.exp(1j*phi)])
            du = np.array([0., 1., 0.])
            power = np.einsum('i,kij,j->k', u.conj(), m[:2], u).real
            tangent = 2*np.einsum('i,kij,j->k', u.conj(), m[:2], du).real
            return (tangent[1]*power[0]-power[1]*tangent[0])/power[0]**2

        grid = np.arange(4096)*2*np.pi/4096
        gains = np.array([direct_gain(phi) for phi in grid])
        step = 2*np.pi/len(grid)
        polished = []
        for sign, index in ((1., np.argmin(gains)), (-1., np.argmax(gains))):
            fit = minimize_scalar(lambda phi: sign*direct_gain(phi),
                                  bounds=(grid[index]-step, grid[index]+step),
                                  method='bounded', options={'xatol': 1e-13})
            if not fit.success:
                raise ValueError('Independent phase-extremum check failed')
            polished.append(sign*fit.fun)
        np.testing.assert_allclose(polished, [extrema['minimum'], extrema['maximum']],
                                   rtol=2e-12, atol=2e-13)
        checks[command] = {'gain_extrema': extrema,
                           'independent_extrema_max_error': float(np.max(abs(
                               np.array(polished)-[extrema['minimum'], extrema['maximum']])))}
        for target, gain in (('B', g_b), ('D', g_d)):
            interval = sorted(100*(gain/extrema[key]-1) for key in ('minimum', 'maximum'))
            rows.append({'command': command, 'target': target,
                         'coherent_min_percent': interval[0], 'coherent_max_percent': interval[1],
                         'paired_percent': 100*(gain/pair-1), 'g_target': gain, 'g_pair': pair})
    return {'rows': rows, 'checks': checks}


def run(output, wall_budget=1200., canary=False):
    if not np.isfinite(wall_budget) or not 0 < wall_budget <= 1800:
        raise ValueError('Require wall budget in (0,1800] seconds')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    geometry = dict(diameter_pixels=8 if canary else 128,
                    spider_pixels=2 if canary else 8, domain_diameters=6,
                    bright_carrier=.125, dark_carrier=-.125, pupil_center_diameters=.75)
    iris, modulation = (1., .73) if canary else (8., 3.)
    reference_angles, modal_angles = (8, 8) if canary else (128, 256)
    identity = {name: text_hash(ROOT/name) for name in SOURCES}
    state = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'process_id': os.getpid(), 'canary': canary, 'source_sha256_lf': identity,
             'geometry': geometry, 'iris_radius': iris, 'modulation': modulation,
             'reference_samples_per_pixel': 3, 'reference_angles': reference_angles,
             'modal_angles': modal_angles, 'detector_diameters': 4,
             'wall_budget_seconds': wall_budget, 'memory_limit_bytes': MEMORY_LIMIT,
             'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                         'scipy': scipy.__version__, 'psutil': psutil.__version__, 'fft_workers': 1},
             'stages': [], 'scope': 'Same published optical operators; not an independent simulator'}

    def save():
        state.update(elapsed_seconds=time.perf_counter()-started,
                     peak_rss_bytes=peak_rss_bytes(),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'result.json', state)

    def artifact(name, **arrays):
        path = output/name
        with path.with_suffix('.tmp').open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        path.with_suffix('.tmp').replace(path)
        return {'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    deadline = None
    try:
        save()
        matrices, bright_hash = {}, None
        for name in ('reference', 'original', 'orthogonal', 'opaque'):
            cfg = Configuration(**geometry, samples_per_pixel=3 if name == 'reference' else 1)
            n = cfg.diameter_pixels*cfg.domain_diameters*cfg.samples_per_pixel
            estimated = n*n*16*(20 if name == 'reference' else 32) + (cfg.diameter_pixels*4)**2*9*16*3
            if estimated > MEMORY_LIMIT or psutil.virtual_memory().available < 2*estimated:
                raise MemoryError('Memory admission failed before '+name)
            state['active_stage'] = name
            save()
            print(json.dumps({'stage_started': name, 'estimated_bytes': estimated}), flush=True)
            stage_start = time.perf_counter()
            remaining = wall_budget-(stage_start-started)
            if remaining <= 0:
                raise TimeoutError('Whole-run deadline reached')
            deadline = WallDeadline(min(remaining, 600. if name == 'reference' else 300.))
            with deadline:
                if name == 'reference':
                    model = OpticalModel(cfg)
                    basis = build_basis(model, 'SLM', iris, modulation, reference_angles, 4)
                    weights = fixed_readout(basis['gram'])
                    saved = artifact('readout.npz', weights=weights)
                    checks = {'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest()}
                else:
                    model = (OrthogonalStripModel if name == 'orthogonal' else ModalModel)(cfg, iris)
                    family = 'B' if name == 'opaque' else 'SLM'
                    basis = build_modal_basis(model, family, modulation, modal_angles, 4)
                    checks = {'direct': direct_check(model, basis, 'B' if name == 'opaque' else 'C',
                              -.37, 0. if name == 'opaque' else np.pi/2,
                              {'modulation': modulation, 'angles': modal_angles, 'detector_diameters': 4})}
                    contracted = readout_matrices(basis['gram'], weights)
                    if name == 'opaque':
                        opaque = contracted
                    else:
                        control_hash = hashlib.sha256(basis['gram'][..., :2, :2].tobytes()).hexdigest()
                        if bright_hash is not None and control_hash != bright_hash:
                            raise ValueError('Original/orthogonal D blocks differ')
                        bright_hash = control_hash
                        matrices[name] = contracted
                        checks['bright_block_sha256'] = control_hash
                    saved = artifact(name+'.npz', **{k: v for k, v in basis.items() if k != 'diagnostics'})
                state['stages'].append({'name': name, 'geometry': asdict(cfg),
                    'elapsed_seconds': time.perf_counter()-stage_start, 'artifact': saved,
                    'diagnostics': basis['diagnostics'], 'checks': checks})
            del basis, model
            gc.collect()
            save()
            if state['peak_rss_bytes'] > MEMORY_LIMIT:
                raise MemoryError('Observed peak exceeds 2 GB; completed artifacts retained')
        deadline = WallDeadline(max(0., wall_budget-(time.perf_counter()-started)))
        with deadline:
            summary = summarize(matrices, opaque)
            write_json(output/'summary.json', summary)
        if any(text_hash(ROOT/name) != value for name, value in identity.items()):
            raise RuntimeError('Source changed during execution')
        state.update(status='COMPLETE_VALIDATED', summary_sha256=hashlib.sha256(
            (output/'summary.json').read_bytes()).hexdigest())
        state.pop('active_stage', None)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='PARTIAL', error=type(error).__name__+': '+str(error),
                     deadline_expired=deadline.expired.is_set() if deadline else False)
    finally:
        save()
    print(json.dumps({k: state.get(k) for k in ('status', 'elapsed_seconds', 'peak_rss_bytes', 'error')}), flush=True)
    return 0 if state['status'] == 'COMPLETE_VALIDATED' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New local output directory (never overwritten)')
    parser.add_argument('--wall-budget', type=float, default=1200., help='Whole-run seconds; default1200, maximum1800')
    parser.add_argument('--canary', action='store_true', help='Small 8-pixel/8-angle lifecycle check; not paper results')
    args = parser.parse_args()
    try:
        return run(args.output, args.wall_budget, args.canary)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
