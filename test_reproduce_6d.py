"""Real small propagation and failure-path checks for the public entrypoint."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from piston_basis import evaluate, readout_matrices, readout_moments
from reproduce_6d import fixed_readout
from simulation import normalize_response

ROOT = Path(__file__).resolve().parent


def cli(path, *args):
    return subprocess.run([sys.executable, str(ROOT/'reproduce_6d.py'), '--output', str(path),
                           '--canary', *args], capture_output=True, text=True, timeout=30)


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    folder = tmp_path_factory.mktemp('reproduction')/'fresh'
    result = cli(folder)
    assert result.returncode == 0, result.stdout+result.stderr
    state = json.loads((folder/'result.json').read_text())
    assert state['status'] == 'COMPLETE_VALIDATED'
    assert [s['name'] for s in state['stages']] == ['reference', 'original', 'orthogonal', 'opaque']
    assert state['canary'] is True
    return folder


def test_fresh_optics_and_table_rows(generated):
    weights = np.load(generated/'readout.npz')['weights']
    opaque = np.load(generated/'opaque.npz')['gram']
    summary = json.loads((generated/'summary.json').read_text())
    assert len(summary['rows']) == 4
    for command in ('original', 'orthogonal'):
        gram = np.load(generated/(command+'.npz'))['gram']
        dark = gram.copy()
        dark[..., 2, :] = 0
        dark[..., :, 2] = 0
        # Raw opposite exposures cancel the cross terms before normalization.
        raw = [evaluate(gram, 0., phi) for phi in (.37, .37+np.pi)]
        _, tangent = normalize_response(*[(a+b)/2 for a, b in zip(*raw)])
        pair_gain = np.sum(weights*tangent)
        for target, target_gram in (('B', opaque), ('D', dark)):
            gain = readout_moments(readout_matrices(target_gram, weights), 0.)['local_gain']
            row = next(r for r in summary['rows'] if (r['command'], r['target']) == (command, target))
            assert row['paired_percent'] == pytest.approx(100*(gain/pair_gain-1), abs=1e-10)
            assert row['coherent_min_percent'] <= row['coherent_max_percent']
    first = np.load(generated/'original.npz')['gram'][..., :2, :2]
    second = np.load(generated/'orthogonal.npz')['gram'][..., :2, :2]
    np.testing.assert_array_equal(first, second)


def test_refuses_existing_directory(generated):
    before = (generated/'result.json').read_bytes()
    result = cli(generated)
    assert result.returncode != 0
    assert (generated/'result.json').read_bytes() == before


@pytest.mark.parametrize('budget', ['0.000001', '0.02'])
def test_deadline_preserves_partial_state_and_fresh_recovery(tmp_path, budget):
    result = cli(tmp_path/'timeout', '--wall-budget', budget)
    assert result.returncode == 1
    state = json.loads((tmp_path/'timeout'/'result.json').read_text())
    assert state['status'] == 'PARTIAL'
    assert state['stages'] == []
    assert 'deadline' in state['error'].lower() or state['deadline_expired']
    assert cli(tmp_path/'recovery').returncode == 0


def test_reference_excludes_strip_and_rejects_zero_tangent():
    rng = np.random.default_rng(42)
    fields = rng.normal(size=(4, 4, 3)) + 1j*rng.normal(size=(4, 4, 3))
    gram = fields[..., :, None].conj()*fields[..., None, :]
    expected = gram.copy()
    expected[..., 2, :] = 0
    expected[..., :, 2] = 0
    _, tangent = normalize_response(*evaluate(expected, 0.))
    np.testing.assert_allclose(fixed_readout(gram), tangent/np.sum(tangent*tangent))
    unchanged = gram.copy()
    fixed_readout(gram)
    np.testing.assert_array_equal(gram, unchanged)
    with pytest.raises(ValueError):
        fixed_readout(np.zeros((4, 4, 3, 3), complex))
