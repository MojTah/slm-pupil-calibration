"""Small real CLI and independent numerical/provenance checks; no full run."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from reproduce_common_mask import CommonMaskModel, masks, observable
from modal_simulation import ModalModel
from reproduce_paper import verify_bundle
from simulation import Configuration, normalize_response, crop_detector

ROOT=Path(__file__).resolve().parent

def cli(script,*args):
    return subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,args)],
                          capture_output=True,text=True,timeout=90)

@pytest.fixture(scope='module')
def bundles(tmp_path_factory):
    result={}
    for profile in ('representative-15d','all'):
        folder=tmp_path_factory.mktemp('common-mask-'+profile)/'fresh'
        p=cli('reproduce_paper.py','build','--profile',profile,'--canary',
              '--output',folder,'--wall-budget',60)
        assert p.returncode==0,p.stdout+p.stderr
        verify_bundle(folder)
        result[profile]=folder
    return result

@pytest.mark.parametrize('profile',['representative-15d','all'])
def test_cli_formula_controls_provenance_and_no_overwrite(bundles,tmp_path,profile):
    folder=bundles[profile];out=tmp_path/'result'
    before=(folder/'result.json').read_bytes()
    p=cli('reproduce_common_mask.py','--input',folder,'--output',out,'--wall-budget',60)
    assert p.returncode==0,p.stdout+p.stderr
    state=json.loads((out/'result.json').read_text());summary=json.loads((out/'summary.json').read_text())
    assert state['status']=='COMPLETE_VALIDATED' and state['canary'] and summary['canary']
    assert state['input_manifest_sha256']==hashlib.sha256(before).hexdigest()
    assert state['summary_sha256']==hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()
    assert state['common']['sha256']==hashlib.sha256((out/'common.npz').read_bytes()).hexdigest()
    assert state['common']['origin']=='new_common_mask_propagation'
    with np.load(folder/'reference-6d-128.npz') as z:weights=z['weights']
    with np.load(out/'common.npz') as z:image=z['image'];tangent=z['tangent']
    flux=image.sum();weighted=(weights*image).sum()
    gain=(weights*tangent).sum()/flux-weighted*tangent.sum()/flux**2
    assert abs(gain-state['common']['g_B'])<1e-13
    if profile=='all':
        with np.load(folder/'opaque-15d-256.npz') as z:gram=z['gram']
        image=gram[...,0,0].real;tangent=2*gram[...,0,1].real;flux=image.sum()
        expected=(weights*tangent).sum()/flux-(weights*image).sum()*tangent.sum()/flux**2
        assert abs(summary['centered_B_gain']-expected)<1e-13
        assert summary['centered_B_origin']=='fresh_bundle_centered_B'
        assert {r['command'] for r in summary['rows']}=={'original','orthogonal'}
        assert {r['target'] for r in summary['rows']}=={'D','common_B','centered_B'}
    else:
        assert summary['centered_B_gain'] is None and summary['common_minus_centered_gain'] is None
        assert summary['centered_B_origin']=='UNAVAILABLE'
        assert {r['command'] for r in summary['rows']}=={'original'}
        assert {r['target'] for r in summary['rows']}=={'D','common_B'}
    snapshot={f.name:f.read_bytes() for f in out.iterdir()}
    p=cli('reproduce_common_mask.py','--input',folder,'--output',out,'--wall-budget',60)
    assert p.returncode!=0
    assert snapshot=={f.name:f.read_bytes() for f in out.iterdir()}
    assert (folder/'result.json').read_bytes()==before

def test_mask_boundary_and_independent_piston_difference():
    model=CommonMaskModel(Configuration(samples_per_pixel=1,domain_diameters=15),8.)
    assert masks(model)['difference']==[[-72,-96,True,False],[72,-96,True,False],[-72,96,True,False],[72,96,True,False]]
    del model
    cfg=Configuration(diameter_pixels=8,spider_pixels=2,samples_per_pixel=1,domain_diameters=15)
    model=CommonMaskModel(cfg,1.);old=ModalModel(cfg,1.)
    for case in ('C','D'):
        assert all(np.array_equal(a,b) for a,b in zip(model.relay_modes(case,.21,.7),old.relay_modes(case,.21,.7)))
    weights=np.broadcast_to(np.linspace(-1.,1.,32),(32,32)).copy()
    value,_=observable(model,weights,.73,8)
    means=[];step=1e-5
    for piston in (-step,step):
        field,derivative=model.relay_modes('B',piston)
        image,tangent=model.sensor_modes(field,derivative,.73,8)
        image=crop_detector(image,32)
        means.append(float((weights*image).sum()/image.sum()))
    assert abs((means[1]-means[0])/(2*step)-value['g_B'])<1e-9

def test_corrupt_active_and_stale_bundle_rejected(bundles,tmp_path):
    for corruption in ('artifact','active','source'):
        folder=tmp_path/corruption;shutil.copytree(bundles['representative-15d'],folder)
        state=json.loads((folder/'result.json').read_text())
        if corruption=='artifact':
            path=folder/state['cases'][0]['artifact'];path.write_bytes(path.read_bytes()+b'bad')
        elif corruption=='active':state['status']='RUNNING'
        else:state['source_sha256_lf']['reproduce_paper.py']='changed'
        (folder/'result.json').write_text(json.dumps(state))
        out=tmp_path/(corruption+'-out')
        p=cli('reproduce_common_mask.py','--input',folder,'--output',out,'--wall-budget',60)
        assert p.returncode!=0 and not out.exists()

def test_deadline_partial_refusal_and_fresh_recovery(bundles,tmp_path):
    folder=bundles['representative-15d'];out=tmp_path/'partial'
    p=cli('reproduce_common_mask.py','--input',folder,'--output',out,'--wall-budget',.000001)
    assert p.returncode==1,p.stdout+p.stderr
    state=json.loads((out/'result.json').read_text());assert state['status']=='PARTIAL'
    before={f.name:f.read_bytes() for f in out.iterdir()}
    assert cli('reproduce_common_mask.py','--input',folder,'--output',out,'--wall-budget',60).returncode!=0
    assert before=={f.name:f.read_bytes() for f in out.iterdir()}
    p=cli('reproduce_common_mask.py','--input',folder,'--output',tmp_path/'recovery','--wall-budget',60)
    assert p.returncode==0,p.stdout+p.stderr

@pytest.mark.parametrize('budget',['0','-1','901','nan','inf'])
def test_invalid_budget_refused_before_output(tmp_path,budget):
    out=tmp_path/'absent'
    p=cli('reproduce_common_mask.py','--input',tmp_path/'missing','--output',out,'--wall-budget',budget)
    assert p.returncode==2 and not out.exists()
