"""Reproduce the local common-mask 15D B diagnostic from a fresh-paper bundle.

Requires a completed reproduce_paper.py bundle, including its sampled reference.
The representative profile supplies original C/D comparisons; the all profile
also supplies fresh centered B and orthogonal C. Missing controls are not invented.
Canary inputs produce canary outputs, never paper numbers. No historical inputs,
fixed gains or private paths are used. Existing outputs are refused.
"""
import argparse
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
from gain_phase import gain_coefficients, gain_phase_range
from modal_simulation import ModalModel
from phase_pair import paired_matrices
from piston_basis import readout_matrices, readout_moments
from reproduce_paper import SOURCES as PAPER_SOURCES, digest, verify_bundle
from run_pilot import WallDeadline, text_hash, write_json
from runtime_resources import peak_rss_bytes
from simulation import Configuration, crop_detector, normalize_response

ROOT = Path(__file__).resolve().parent
SOURCES = ('reproduce_common_mask.py', *PAPER_SOURCES)
MEMORY_LIMIT = 2_000_000_000

# These three numerical definitions preserve the validated common-mask core.
class CommonMaskModel(ModalModel):
    def relay_modes(self, case, piston=0., shadow_phase=0.):
        if case != 'B':
            return super().relay_modes(case, piston, shadow_phase)
        if not np.isfinite([piston, shadow_phase]).all():
            raise ValueError('Finite piston and phase required')
        g = self.grid
        command = g.bright*np.exp(1j*piston*g.piston_shape)
        mask = np.roll(g.iris(self.iris_radius, g.config.bright_carrier),
                       -self.carrier_index, axis=1)
        return tuple(self.held_coefficients(v)*mask for v in
                     (command, 1j*g.piston_shape*command))

def masks(model):
    g = model.grid
    b = g.iris(model.iris_radius)
    d = np.roll(g.iris(model.iris_radius, g.config.bright_carrier),
                -model.carrier_index, axis=1)
    yy, xx = np.where(b != d)
    return {'centered_count': int(b.sum()), 'translated_count': int(d.sum()),
            'centered_sha256': hashlib.sha256(b.tobytes()).hexdigest(),
            'translated_sha256': hashlib.sha256(d.tobytes()).hexdigest(),
            'equal': bool(np.array_equal(b, d)),
            'difference': [[int(x-g.n//2), int(y-g.n//2), bool(b[y,x]), bool(d[y,x])]
                           for y,x in zip(yy,xx)]}

def observable(model, weights, modulation, angles):
    modes, tangent = model.relay_modes('B', 0.)
    intensity, derivative = model.sensor_modes(modes, tangent, modulation, angles)
    n = model.grid.n
    full_flux = float(np.vdot(modes, modes).real*n*n)
    full_derivative = float(2*np.vdot(modes, tangent).real*n*n)
    if abs(intensity.sum()-full_flux) > 1e-11*full_flux+1e-12:
        raise ValueError('Total flux mismatch')
    if abs(derivative.sum()-full_derivative) > 1e-11*full_flux+1e-12:
        raise ValueError('Flux derivative mismatch')
    if intensity.min() < -1e-12:
        raise ValueError('Negative intensity')
    image = crop_detector(intensity, weights.shape[0])
    slope = crop_detector(derivative, weights.shape[0])
    y,j = normalize_response(image, slope)
    if image.sum() > full_flux*(1+1e-12) or not np.isfinite(j).all():
        raise ValueError('Invalid cropped response')
    return {'g_B': float(np.sum(weights*j)), 'raw_mean': float(np.sum(weights*y)),
            'cropped_flux': float(image.sum()), 'total_flux': full_flux,
            'total_flux_error': float(intensity.sum()-full_flux)}, (image.copy(), slope.copy())

def transfer_rows(command, matrices, common_gain, centered_gain=None):
    extrema = gain_phase_range(gain_coefficients(matrices))
    if extrema['inversion_status'] != 'WELL_CONDITIONED':
        raise ValueError('C gain inversion is not well conditioned')
    pair = readout_moments(paired_matrices(matrices, 0.), 0.)['local_gain']
    if not np.isfinite(pair) or abs(pair) < 1e-12:
        raise ValueError('Paired gain is not invertible')
    bright = matrices.copy()
    bright[..., 2, :] = 0
    bright[..., :, 2] = 0
    g_d = readout_moments(bright, 0.)['local_gain']
    rows = []
    targets = [('common_B', common_gain, 'new_common_mask_propagation'),
               ('D', g_d, 'fresh_bundle_bright_block')]
    if centered_gain is not None:
        targets.append(('centered_B', centered_gain, 'fresh_bundle_centered_B'))
    for target, gain, origin in targets:
        interval = sorted(100*(gain/extrema[k]-1) for k in ('minimum','maximum'))
        rows.append({'command': command, 'target': target, 'control_origin': origin,
                     'g_target': gain, 'g_D': g_d, 'g_C_min': extrema['minimum'],
                     'g_C_max': extrema['maximum'], 'g_pair': pair,
                     'coherent_min_percent': interval[0], 'coherent_max_percent': interval[1],
                     'paired_percent': 100*(gain/pair-1)})
    return rows

def run(input_folder, output, wall_budget=900.):
    if not np.isfinite(wall_budget) or not 0 < wall_budget <= 900:
        raise ValueError('Require wall budget in (0,900] seconds')
    start = time.perf_counter()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing output is never overwritten')
    folder = Path(input_folder).resolve()
    bundle = verify_bundle(folder)
    input_hash = digest(folder/'result.json')
    cases = {item['case']['id']: item for item in bundle['cases']}
    reference = cases['reference-6d-128']
    case = cases['original-15d-256']['case']
    cfg = Configuration(**case['geometry'])
    with np.load(folder/reference['artifact'], allow_pickle=False) as data:
        weights = data['weights'].copy()
    weight_hash = hashlib.sha256(weights.tobytes()).hexdigest()
    if (weights.shape != (cfg.diameter_pixels*4,)*2 or not np.isfinite(weights).all()
            or weight_hash != reference['checks']['weights_sha256']):
        raise ValueError('Fresh reference weights are invalid')
    estimate = (cfg.diameter_pixels*cfg.domain_diameters)**2*16*32+weights.size*9*16*2
    if estimate > MEMORY_LIMIT or psutil.virtual_memory().available < 2*estimate:
        raise MemoryError('Memory admission failed')
    identity = {name: text_hash(ROOT/name) for name in SOURCES}
    runtime = {'python': platform.python_version(), 'numpy': np.__version__,
               'scipy': scipy.__version__, 'psutil': psutil.__version__, 'fft_workers': 1}
    output.mkdir(parents=True, exist_ok=False)
    state = {'schema': 'fresh-common-mask-v1', 'status': 'RUNNING',
             'canary': bundle['canary'], 'started_utc': datetime.now(timezone.utc).isoformat(),
             'process_id': os.getpid(), 'source_sha256_lf': identity,
             'input_manifest_sha256': input_hash, 'input_profile': bundle['profile'],
             'input_artifacts': {name: item['sha256'] for name,item in cases.items()},
             'case': case, 'weights_sha256': weight_hash, 'runtime': runtime,
             'input_build_runtime': bundle['runtime'], 'runtime_policy': 'ALLOW_AND_RECORD',
             'runtime_differences': {k:{'build':bundle['runtime'].get(k),'current':v}
                 for k,v in runtime.items() if bundle['runtime'].get(k) != v},
             'estimated_bytes': estimate, 'memory_limit_bytes': MEMORY_LIMIT,
             'wall_budget_seconds': wall_budget,
             'scope': 'Local zero-piston diagnostic; no finite-piston curves or convergence claim'}
    def save():
        state.update(elapsed_seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss_bytes(),
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(output/'result.json', state)
    deadline = None
    try:
        save()
        remaining = wall_budget-(time.perf_counter()-start)
        if remaining <= 0:
            raise TimeoutError('Deadline reached before propagation')
        deadline = WallDeadline(remaining)
        with deadline:
            model = CommonMaskModel(cfg, case['iris'])
            state['masks'] = masks(model)
            value, arrays = observable(model, weights, case['modulation'], case['angles'])
            if peak_rss_bytes() > MEMORY_LIMIT:
                raise MemoryError('Observed peak exceeds2GB')
            with (output/'common.tmp').open('wb') as handle:
                np.savez_compressed(handle, image=arrays[0], tangent=arrays[1])
            (output/'common.tmp').replace(output/'common.npz')
            state['common'] = {**value, 'origin':'new_common_mask_propagation',
                               'artifact':'common.npz', 'sha256':digest(output/'common.npz')}
            del model, arrays
            gc.collect()
            centered_gain = None
            if 'opaque-15d-256' in cases:
                with np.load(folder/cases['opaque-15d-256']['artifact'], allow_pickle=False) as data:
                    centered_gain = readout_moments(readout_matrices(data['gram'],weights),0.)['local_gain']
            rows = []
            bright_hash = None
            for command in ('original','orthogonal'):
                ident = command+'-15d-256'
                if ident not in cases:
                    continue
                with np.load(folder/cases[ident]['artifact'], allow_pickle=False) as data:
                    gram = data['gram']
                    current_hash = hashlib.sha256(gram[...,:2,:2].tobytes()).hexdigest()
                    if bright_hash is not None and current_hash != bright_hash:
                        raise ValueError('Fresh original/orthogonal bright blocks differ')
                    bright_hash = current_hash
                    matrices = readout_matrices(gram,weights)
                rows.extend(transfer_rows(command, matrices, value['g_B'], centered_gain))
            summary = {'canary': bundle['canary'], 'rows': rows,
                       'common_B_gain': value['g_B'], 'centered_B_gain': centered_gain,
                       'centered_B_origin': 'fresh_bundle_centered_B' if centered_gain is not None else 'UNAVAILABLE',
                       'common_minus_centered_gain': value['g_B']-centered_gain if centered_gain is not None else None,
                       'scope': state['scope']}
            write_json(output/'summary.json', summary)
            state['summary_sha256'] = digest(output/'summary.json')
            if input_hash != digest(folder/'result.json'):
                raise ValueError('Input manifest changed during execution')
            verify_bundle(folder)
            if identity != {name:text_hash(ROOT/name) for name in SOURCES}:
                raise ValueError('Source changed during execution')
            if peak_rss_bytes() > MEMORY_LIMIT:
                raise MemoryError('Observed peak exceeds2GB')
            if time.perf_counter()-start > wall_budget:
                raise TimeoutError('Deadline reached before finalization')
            state['status'] = 'COMPLETE_VALIDATED'
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='PARTIAL', error=type(error).__name__+': '+str(error),
                     deadline_expired=deadline.expired.is_set() if deadline else False)
    finally:
        save()
    print(json.dumps({k:state.get(k) for k in ('status','elapsed_seconds','peak_rss_bytes','error')}),flush=True)
    return 0 if state['status']=='COMPLETE_VALIDATED' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='Verified completed fresh-paper bundle')
    p.add_argument('--output',type=Path,required=True,help='New local output directory')
    p.add_argument('--wall-budget',type=float,default=900.)
    a=p.parse_args()
    try:
        return run(a.input,a.output,a.wall_budget)
    except (ValueError,FileExistsError,MemoryError) as error:
        p.error(str(error))

if __name__=='__main__':
    raise SystemExit(main())
