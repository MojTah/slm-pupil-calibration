"""Renderer contracts and actual source-only tiny CLI, without historical inputs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from render_paper import (coherence, curve, finite_curve, load_analysis,
                          target_matrices, transfer_figure, numeric_tables,load_common,domain_figure,value,
                          transfer_records,transfer_coverage)
from reproduce_paper import digest

ROOT = Path(__file__).resolve().parent


def cli(script, *args):
    return subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],
                          capture_output=True,text=True,timeout=180)


@pytest.fixture(scope='module')
def fresh(tmp_path_factory):
    folder = tmp_path_factory.mktemp('renderer')
    build, analysis = folder/'build',folder/'analysis'
    result = cli('reproduce_paper.py','build','--profile','all','--canary','--output',build,'--wall-budget',60)
    assert result.returncode == 0, result.stdout+result.stderr
    result = cli('reproduce_paper.py','analyze','--input',build,'--output',analysis)
    assert result.returncode == 0, result.stdout+result.stderr
    common=folder/'common'
    result=cli('reproduce_common_mask.py','--input',build,'--output',common,'--wall-budget',60)
    assert result.returncode==0,result.stdout+result.stderr
    return build,analysis,common


def test_analytic_tolerance_matches_complex_pair():
    d,e = np.meshgrid([0.,.1,.3],[-.5,0.,.5])
    direct = abs((1+d)/2-(1-d)/2*np.exp(1j*e))
    np.testing.assert_allclose(coherence(d,e),direct,atol=1e-15)
    np.testing.assert_allclose(coherence(d,0),d,atol=1e-15)


def test_curve_requires_complete_unique_order():
    x,rows = curve([{'x':1.},{'x':0.}],'x',2)
    np.testing.assert_array_equal(x,[0.,1.])
    with pytest.raises(ValueError,match='curve samples'):
        curve([{'x':0.},{'x':0.}],'x',2)
    with pytest.raises(ValueError,match='curve samples'):
        curve([{'x':0.}],'x',2)


def test_finite_curve_preserves_exclusions_and_phase_maximum():
    rows = [{'kappa':1.,'calibration_photons':10**8,'half_poke_rad':h,
             'scale_rss_proxy':phase+1.,'approximation_status':'LOCAL_EXPANSION_ONLY'}
            for h in (.1,.2) for phase in range(3)]
    np.testing.assert_allclose(finite_curve(rows,1.,10**8,grid=(.1,.2)),[300.,300.])
    rows[0]['approximation_status']='CV_EXCEEDS_REPORTING_CUTOFF'
    assert np.isnan(finite_curve(rows,1.,10**8,grid=(.1,.2))[0])


def test_actual_complete_render_and_no_overwrite(fresh,tmp_path):
    build,analysis,common = fresh
    output = tmp_path/'render'
    result = cli('render_paper.py','--input',analysis,'--build-input',build,'--common-mask-input',common,'--output',output)
    assert result.returncode == 0, result.stdout+result.stderr
    manifest = json.loads((output/'result.json').read_text())
    assert manifest['status']=='COMPLETE_RENDER' and manifest['canary'] is True
    assert len(manifest['coverage'])==21 and len(manifest['outputs'])==31
    assert sum(v['kind']=='figure' for v in manifest['outputs'].values())==20
    assert sum(v['kind']=='table' for v in manifest['outputs'].values())==11
    assert manifest['input_analysis_manifest_sha256']==digest(analysis/'result.json')
    assert manifest['input_build_manifest_sha256']==digest(build/'result.json')
    assert all(digest(output/name)==v['sha256'] for name,v in manifest['outputs'].items())
    assert all((output/name).read_bytes().startswith(b'%PDF') for name in manifest['outputs'] if name.endswith('.pdf'))
    before = (output/'result.json').read_bytes()
    result = cli('render_paper.py','--input',analysis,'--build-input',build,'--output',output)
    assert result.returncode != 0
    assert (output/'result.json').read_bytes()==before


def test_source_schema_checksum_and_name_guards(fresh,tmp_path):
    _,analysis,_ = fresh
    folder = tmp_path/'bad'
    shutil.copytree(analysis,folder)
    manifest = folder/'result.json'
    state = json.loads(manifest.read_text())
    name = 'phase-curves.csv'
    original = (folder/name).read_bytes()
    (folder/name).write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_analysis(folder)
    (folder/name).write_bytes(original.replace(b'phase_rad',b'wrong_key',1))
    state['outputs'][name]['sha256']=digest(folder/name)
    manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='columns'):
        load_analysis(folder)
    state['outputs']['../outside.csv']=state['outputs'].pop(name)
    manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='output name'):
        load_analysis(folder)
    state['source_sha256_lf']['fresh_paper_analysis.py']='stale'
    manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='source identity'):
        load_analysis(folder)


def test_matching_build_and_transfer_numeric_invariants(fresh,tmp_path):
    build,analysis,_ = fresh
    state,tables = load_analysis(analysis)
    records,missing=transfer_records(tables,'B')
    assert len(records)==4 and missing==[]
    assert transfer_coverage(tables,'B',True)=='RENDERED'
    reduced=dict(tables)
    reduced['transfer-summary.csv']=[r for r in tables['transfer-summary.csv'] if not (r['case']=='original-15d-128' and r['target']=='B')]
    assert transfer_coverage(reduced,'B',True)=='PARTIAL: missing interval cases: original-15d-128'
    targets = target_matrices(build,state)
    fig = transfer_figure(tables,'B',targets['B'])
    axes=fig.axes
    assert len(axes[0].lines)==8  # interval and phase-zero dot for all four B cases
    for line in axes[1].lines:
        assert line.get_ydata()[100]==pytest.approx(0.,abs=1e-12)
    self_error=axes[2].lines[-1].get_ydata()
    for transfer,total in zip(axes[1].lines,axes[2].lines):
        np.testing.assert_allclose(total.get_ydata(),self_error+transfer.get_ydata()/1000,atol=1e-14)
    for line in axes[3].lines:
        y=line.get_ydata()
        assert np.isnan(y[100]) and np.all(y[np.isfinite(y)]>0)
    plt.close(fig)
    wrong = dict(state,input_manifest_sha256='wrong')
    with pytest.raises(ValueError,match='input identity'):
        target_matrices(build,wrong)
    tex = numeric_tables(tables)
    assert len(tex)==10
    assert tex['supp-table5-finite']['rows']==4  # includes 1e6


def test_common_identity_and_local_ratio_checks(fresh,tmp_path):
    build,analysis,common=fresh
    state,tables=load_analysis(analysis)
    metadata,rows=load_common(common,state,tables,build)
    assert len(rows)==3
    for row in rows[:2]:
        assert row['centered_ratio_percent']==row['common_ratio_percent']
        assert row['centered_pair_percent']==row['common_pair_percent']
    summary=json.loads((common/'summary.json').read_text())
    reference=next(r for r in summary['rows'] if r['command']=='original' and r['target']=='common_B')
    assert rows[-1]['common_ratio_percent']==pytest.approx(100*(reference['g_target']/reference['g_D']-1))
    with pytest.raises(ValueError,match='input identity'):
        load_common(common,dict(state,input_manifest_sha256='wrong'),tables,build)
    bad=tmp_path/'common'; shutil.copytree(common,bad)
    (bad/'common.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='array checksum'):
        load_common(bad,state,tables,build)
    fig=domain_figure(tables)
    assert any('unavailable' in t.get_text() for t in fig.axes[-1].texts)
    plt.close(fig)


@pytest.mark.parametrize('text',['nan','inf','1e999'])
def test_nonfinite_csv_numbers_rejected(text):
    with pytest.raises(ValueError,match='Nonfinite'):
        value(text)


def test_analytic_cli_needs_no_analysis_or_build(tmp_path):
    output = tmp_path/'analytic'
    result = cli('render_paper.py','--analytic-only','--output',output)
    assert result.returncode==0, result.stdout+result.stderr
    state=json.loads((output/'result.json').read_text())
    assert state['status']=='COMPLETE_RENDER'
    assert set(state['coverage'])=={'optical-schematic','pair-tolerances'}
    assert state['input_analysis_manifest_sha256'] is None
