"""Actual small CLI, identity, partial recovery and resource-plan contracts."""
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
import numpy as np
import scipy
from reproduce_paper import BrightReference, case_plan, verify_bundle
from reproduce_6d import fixed_readout
from piston_basis import build_basis
from simulation import Configuration, OpticalModel

ROOT = Path(__file__).resolve().parent


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT/'reproduce_paper.py'), *map(str, args)],
                          capture_output=True, text=True, timeout=90)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    folder = tmp_path_factory.mktemp('freshpaper')/'build'
    result = cli('build', '--profile', 'all', '--output', folder, '--canary', '--wall-budget', 60)
    assert result.returncode == 0, result.stdout+result.stderr
    state = verify_bundle(folder)
    assert len(state['cases']) == 14
    assert 'Sampled bright-only D' in state['reference_construction']
    assert state['cases'][0]['checks']['reference_relay'] == 'D'
    assert state['cases'][0]['checks']['reference_strip_terms'] == 'OMITTED_ZERO'
    return folder


def test_case_graph_and_realistic_15d_bounds():
    plan = case_plan('all')
    assert len(plan) == len({c['id'] for c in plan}) == 14
    assert {'opaque-15d-128', 'orthogonal-6d-128'} <= {c['id'] for c in plan}
    representative = case_plan('representative-15d')
    assert [c['id'] for c in representative] == ['reference-6d-128', 'original-15d-256']
    assert representative[1]['estimated_bytes'] == 1962934272
    assert 734 < representative[1]['stage_cap_seconds'] <= 1200
    assert representative[1]['geometry']['diameter_pixels'] == 128
    assert representative[0]['stage_cap_seconds'] == 720


@pytest.mark.parametrize('samples', [1, 3])
def test_bright_reference_preserves_full_slm_readout_bitwise(samples):
    cfg = Configuration(diameter_pixels=8, spider_pixels=2, samples_per_pixel=samples,
                        domain_diameters=6)
    original = build_basis(OpticalModel(cfg), 'SLM', 1., .73, 8, 4)
    bright = build_basis(BrightReference(cfg), 'B', 1., .73, 8, 4)
    for key in ('gram', 'pupil_gram', 'flux_gram'):
        np.testing.assert_array_equal(original[key][..., :2, :2], bright[key][..., :2, :2])
        assert np.all(bright[key][..., 2, :] == 0)
        assert np.all(bright[key][..., :, 2] == 0)
    np.testing.assert_array_equal(fixed_readout(original['gram']), fixed_readout(bright['gram']))


def test_actual_cli_analysis_and_no_overwrite(bundle, tmp_path):
    out = tmp_path/'analysis'
    result = cli('analyze', '--input', bundle, '--output', out)
    assert result.returncode == 0, result.stdout+result.stderr
    state = json.loads((out/'result.json').read_text())
    assert state['status'] == 'COMPLETE_NUMERIC_ANALYSIS'
    assert len(state['outputs']) == 9
    assert state['outputs']['sensitivity-summary.csv']['rows'] == 5
    assert state['canary'] is True and state['uncovered']
    assert not any('Two additional prose angular' in item for item in state['uncovered'])
    assert state['analysis_runtime'] == {'python': platform.python_version(), 'numpy': np.__version__,
                                         'scipy': scipy.__version__, 'platform': platform.platform()}
    assert state['runtime_policy'] == 'ALLOW_AND_RECORD'
    assert state['runtime_differences'] == {}
    before = (bundle/'result.json').read_bytes()
    assert cli('build', '--profile', 'all', '--output', bundle, '--canary', '--wall-budget', 60).returncode != 0
    assert (bundle/'result.json').read_bytes() == before


def test_analysis_records_runtime_drift_without_rewriting_build(bundle, tmp_path):
    source = tmp_path/'other-runtime'
    shutil.copytree(bundle, source)
    manifest = source/'result.json'
    build = json.loads(manifest.read_text())
    build['runtime']['numpy'] = 'test-other-version'
    manifest.write_text(json.dumps(build))
    before = manifest.read_bytes()
    out = tmp_path/'analysis'
    result = cli('analyze', '--input', source, '--output', out)
    assert result.returncode == 0, result.stdout+result.stderr
    analysis = json.loads((out/'result.json').read_text())
    assert analysis['input_build_runtime'] == build['runtime']
    assert analysis['runtime_differences'] == {'numpy': {'build': 'test-other-version', 'analysis': np.__version__}}
    assert analysis['runtime_uncompared_keys'] == ['platform']
    assert analysis['runtime_policy'] == 'ALLOW_AND_RECORD'
    assert 'identity is not claimed' in analysis['runtime_policy_reason']
    from reproduce_paper import digest
    assert analysis['input_manifest_sha256'] == digest(manifest)
    assert manifest.read_bytes() == before


def test_verified_reuse_from_partial(bundle, tmp_path):
    source = tmp_path/'partial'
    shutil.copytree(bundle, source)
    state = json.loads((source/'result.json').read_text())
    state['status'] = 'PARTIAL'
    state['cases'] = state['cases'][:1]
    (source/'result.json').write_text(json.dumps(state))
    out = tmp_path/'recovered'
    result = cli('build', '--profile', 'representative-15d', '--output', out,
                 '--canary', '--wall-budget', 60, '--reuse', source)
    assert result.returncode == 0, result.stdout+result.stderr
    final = verify_bundle(out)
    assert 'reused_from_manifest_sha256' in final['cases'][0]
    assert final['cases'][0]['sha256'] == state['cases'][0]['sha256']
    assert len(final['cases']) == 2


def test_rejects_corruption_and_identity_drift(bundle, tmp_path):
    source = tmp_path/'bad'
    shutil.copytree(bundle, source)
    state = json.loads((source/'result.json').read_text())
    path = source/state['cases'][0]['artifact']
    path.write_bytes(path.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        verify_bundle(source)
    state['source_sha256_lf']['reproduce_paper.py'] = 'wrong'
    (source/'result.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='source identity'):
        verify_bundle(source)


def test_refuses_active_bundle_reuse(bundle, tmp_path):
    state = json.loads((bundle/'result.json').read_text())
    state['status'] = 'RUNNING'
    folder = tmp_path/'active'
    folder.mkdir()
    (folder/'result.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='active'):
        verify_bundle(folder, require_complete=False)


@pytest.mark.parametrize('budget', [.000001, .02])
def test_tiny_deadline_keeps_partial_and_blocks_analysis(tmp_path, budget):
    folder = tmp_path/'timeout'
    result = cli('build', '--profile', 'all', '--output', folder, '--canary', '--wall-budget', budget)
    assert result.returncode == 1
    state = json.loads((folder/'result.json').read_text())
    assert state['status'] == 'PARTIAL' and not state['cases']
    result = cli('analyze', '--input', folder, '--output', tmp_path/'rejected')
    assert result.returncode != 0
    assert not (tmp_path/'rejected').exists()
