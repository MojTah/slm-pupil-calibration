"""Compare a verified alternative with matched frozen Grams and a fixed readout."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--alternative-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Require a fresh output directory')
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import psutil
    from scipy.linalg import eigvalsh
    from scipy.optimize import minimize_scalar
    import analyze_transfer
    from gain_phase import gain_coefficients, gain_phase_range
    from phase_envelope import overlap_coefficients, phase_range
    from phase_pair import paired_matrices
    from piston_basis import evaluate, readout_matrices, readout_moments
    from simulation import normalize_response
    from run_pilot import WallDeadline
    analyze_transfer.ROOT = args.input_root.resolve()
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    identities = {}
    source_names = ('analyze_transfer.py', 'gain_phase.py', 'phase_envelope.py',
                    'phase_pair.py', 'piston_basis.py', 'simulation.py', 'run_pilot.py')
    source_hashes = {n: digest(args.source_root/n) for n in source_names}

    def load(name, family='SLM'):
        manifest, gram, identity = analyze_transfer.load_gram(name, family)
        identities[name] = identity
        return manifest, gram

    def dark(m):
        d = m.copy()
        d[..., 2, :] = 0
        d[..., :, 2] = 0
        return d

    def direct(matrices, phases):
        # Independent quadratic forms, without gain/overlap coefficient helpers.
        phases = np.asarray(phases)
        u = np.stack([np.ones_like(phases), np.zeros_like(phases), np.exp(1j*phases)], axis=-1)
        du = np.array([0., 1., 0.])
        powers = np.einsum('...i,kij,...j->...k', u.conj(), matrices[:2], u).real
        tangents = 2*np.einsum('...i,kij,j->...k', u.conj(), matrices[:2], du).real
        assert np.min(powers[..., 0]) > 0
        means = powers[..., 1]/powers[..., 0]
        gains = (tangents[..., 1]*powers[..., 0] - powers[..., 1]*tangents[..., 0])/powers[..., 0]**2
        return means, gains

    with WallDeadline(90):
        _, legacy = load('finite-s3-domain6-128')
        _, j = normalize_response(*evaluate(dark(legacy), 0.))
        weights = j/np.sum(j*j)
        del legacy, j
        weight_hash = hashlib.sha256(weights.tobytes()).hexdigest()
        assert weight_hash == '81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972'
        alternate_manifest = json.loads((args.alternative_run/'result.json').read_text())
        alternate_provenance = json.loads((args.alternative_run/'variant-provenance.json').read_text())
        assert alternate_manifest['status'] == alternate_provenance['status'] == 'COMPLETE_VALIDATED'
        assert alternate_provenance['effective_strip_carrier_cycles_per_pitch'] == [0., .125]
        assert alternate_provenance['result_sha256'] == digest(args.alternative_run/'result.json')
        block, = alternate_manifest['blocks']
        assert block['sha256'] == digest(args.alternative_run/block['artifact'])
        identities['orthogonal_y_6D_256'] = {
            'manifest_sha256': digest(args.alternative_run/'result.json'),
            'variant_provenance_sha256': digest(args.alternative_run/'variant-provenance.json'),
            'basis_sha256': block['sha256']}
        rows, curves, checks = [], [], []
        grid = np.arange(65536)*2*np.pi/65536
        step = 2*np.pi/len(grid)
        for command, domain in [('original', 6), ('orthogonal', 6), ('original', 15)]:
            name = f'modal-domain{domain}-256'
            original_manifest, original = load(name)
            if command == 'orthogonal':
                with np.load(args.alternative_run/block['artifact']) as data:
                    gram = data['gram']
                expected = original_manifest['config']
                assert all(alternate_manifest['config'][k] == v for k, v in expected.items())
                np.testing.assert_array_equal(gram[..., :2, :2], original[..., :2, :2])
                del original
            else:
                gram = original
            _, opaque = load(analyze_transfer.B_TARGET_RUNS[name], 'B')
            matrices = readout_matrices(gram, weights)
            b = readout_matrices(opaque, weights)
            del gram, opaque
            d = dark(matrices)
            target = readout_moments(b, 0.)
            g_b, r_b = target['local_gain'], target['weighted_mean']
            g_d = readout_moments(d, 0.)['local_gain']
            coeff = gain_coefficients(matrices)
            extrema = gain_phase_range(coeff)
            assert extrema['inversion_status'] == 'WELL_CONDITIONED'
            pair = readout_moments(paired_matrices(matrices, 0.), 0.)
            overlap = overlap_coefficients(matrices, 0.)
            offset = phase_range(*overlap)
            a0, a1, b0, b1 = overlap
            def ratio_matrix(a, z):
                return np.array([[a+2*z.real, -2*z.imag], [-2*z.imag, a-2*z.real]])
            eigen = eigvalsh(ratio_matrix(a1, b1), ratio_matrix(a0, b0))
            np.testing.assert_allclose(eigen, [offset['minimum'], offset['maximum']], rtol=2e-12, atol=2e-13)
            means, gains = direct(matrices, grid)
            assert np.max(np.abs(gains)) > 0
            dense_gap = max(abs(gains.min()-extrema['minimum']), abs(gains.max()-extrema['maximum']))
            offset_gap = max(abs(means.min()-offset['minimum']), abs(means.max()-offset['maximum']))
            assert dense_gap < 1e-8 and offset_gap < 1e-8
            polished = []
            for sign, index in [(1., int(np.argmin(gains))), (-1., int(np.argmax(gains)))]:
                value = minimize_scalar(lambda p: sign*float(direct(matrices, p)[1]),
                    bounds=(grid[index]-step, grid[index]+step), method='bounded', options={'xatol': 1e-13})
                assert value.success
                polished.append(sign*value.fun)
            np.testing.assert_allclose(polished, [extrema['minimum'], extrema['maximum']], rtol=2e-12, atol=2e-13)
            b_errors = sorted(100*(g_b/extrema[k]-1) for k in ('minimum', 'maximum'))
            d_errors = sorted(100*(g_d/extrema[k]-1) for k in ('minimum', 'maximum'))
            row = {'command': command, 'domain_D': domain, 'modulation_angles': 256,
                'readout': 'archived_fixed', 'g_B': g_b, 'g_D': g_d, 'g_pair': pair['local_gain'],
                'g_C_min': extrema['minimum'], 'g_C_max': extrema['maximum'],
                'g_C_min_phase_rad': extrema['minimum_phase'], 'g_C_max_phase_rad': extrema['maximum_phase'],
                'e_B_min_percent': b_errors[0], 'e_B_max_percent': b_errors[1],
                'e_B_pair_percent': 100*(g_b/pair['local_gain']-1),
                'e_D_min_percent': d_errors[0], 'e_D_max_percent': d_errors[1],
                'e_D_pair_percent': 100*(g_d/pair['local_gain']-1),
                'B_over_D_minus_one_percent': 100*(g_b/g_d-1),
                'strip_power_fraction': float(matrices[0, 2, 2].real/a0),
                'relative_flux_interference_amplitude': float(2*abs(b0)/a0),
                'zero_B_readout_rad': r_b,
                'zero_offset_vs_B_min_rad': offset['minimum']-r_b,
                'zero_offset_vs_B_max_rad': offset['maximum']-r_b,
                'zero_offset_span_rad': offset['maximum']-offset['minimum'],
                'zero_offset_min_phase_rad': offset['minimum_phase'],
                'zero_offset_max_phase_rad': offset['maximum_phase'],
                'zero_offset_pair_vs_B_rad': pair['weighted_mean']-r_b}
            rows.append(row)
            checks.append({'command': command, 'domain_D': domain,
                'dense_phase_count': len(grid), 'dense_gain_extrema_gap': float(dense_gap),
                'dense_offset_extrema_gap_rad': float(offset_gap),
                'polished_gain_extrema_max_error': float(np.max(abs(np.array(polished)-[extrema['minimum'], extrema['maximum']]))),
                'offset_eigenvalue_max_error_rad': float(np.max(abs(eigen-[offset['minimum'], offset['maximum']]))),
                'relative_minimum_flux': extrema['relative_minimum_flux'], 'gain_solver': extrema})
            if domain == 6:
                phases = np.linspace(0, 2*np.pi, 721)
                r, g = direct(matrices, phases)
                for phi, mean, gain in zip(phases, r, g):
                    curves.append({'command': command, 'phase_rad': float(phi),
                        'e_B_percent': float(100*(g_b/gain-1)),
                        'e_D_percent': float(100*(g_d/gain-1)),
                        'zero_offset_vs_B_rad': float(mean-r_b)})
        assert all(digest(args.source_root/n) == h for n, h in source_hashes.items())
        assert psutil.Process().memory_info().peak_wset < 2_000_000_000
        args.output.mkdir(parents=True)
        for filename, records in [('summary.csv', rows), ('phase-curves.csv', curves)]:
            with (args.output/filename).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
        result = {'status': 'PASS', 'started_utc': started,
            'elapsed_seconds': time.perf_counter()-start,
            'peak_rss_bytes': psutil.Process().memory_info().peak_wset,
            'readout_sha256': weight_hash, 'results': rows, 'validation': checks,
            'input_identities': identities, 'source_sha256_raw': source_hashes,
            'analysis_source_sha256': digest(Path(__file__)),
            'definitions': {
                'e_B': '100*(g_B/g_C(phi)-1); target B and C share domain and angular quadrature',
                'e_D': '100*(g_D/g_C(phi)-1)',
                'pair': 'Ideal opposite raw exposures pooled before normalization, coherence=0',
                'strip_power_fraction': 'F_S/(F_D+F_S) in the cropped detector at piston0; not pupil-image contrast',
                'zero_offset': 'r_C(0,phi)-r_B(0), raw normalized archived readout in nominal radian units; no division by g_C',
                'offset_span': 'max_phi r_C(0,phi)-min_phi r_C(0,phi), invariant to fixed reference'},
            'limitations': ['6D comparison only; no new domain-convergence claim',
                'Exact finite-model reduction, not independent optical propagation',
                'No device fringing, phase-response, zero-order or bench validation',
                'Readout and numerical arrays remain private'],
            'output_sha256': {n: digest(args.output/n) for n in ('summary.csv', 'phase-curves.csv')}}
        (args.output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'], 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
