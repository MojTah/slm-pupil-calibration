"""Small numerical and real CLI checks for the portable added analyses."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parent

def module(relative):
    path=ROOT/'analysis'/relative
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(path.stem,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

common=module('common.py')


def fixture_gram():
    rng=np.random.default_rng(247)
    fields=rng.normal(size=(12,12,3))+1j*rng.normal(size=(12,12,3))
    fields[...,2]*=.03
    return np.einsum('yxi,yxj->yxij',fields.conj(),fields)


def test_normalized_derivative_and_raw_pooling():
    Q=fixture_gram();phi=.37
    I,J=common.state(Q,phi);h=1e-5
    # K is the derivative field; independently evaluate perturbed complex fields via Gram forms.
    def normalized(t):
        u=np.array([1.,t,np.exp(1j*phi)])
        intensity=np.einsum('i,yxij,j->yx',u.conj(),Q,u).real
        return intensity/intensity.sum()
    np.testing.assert_allclose(common.yj(I,J)[1],(normalized(h)-normalized(-h))/(2*h),atol=5e-11)
    I2,J2=common.state(Q,phi+np.pi);Ip,Jp=common.state(Q,kind='pair')
    np.testing.assert_allclose((I+I2)/2,Ip,atol=2e-14)
    np.testing.assert_allclose((J+J2)/2,Jp,atol=2e-14)


@pytest.mark.parametrize('mode',['global','local'])
def test_slope_derivative(mode):
    m=module('slope-readouts/slopes.py');rng=np.random.default_rng(14)
    I=1+rng.random((32,32));J=rng.normal(size=I.shape);h=1e-5
    def slope(x):return m.slopes(x,np.zeros_like(x),mode,6,4,4)[0]
    analytic=m.slopes(I,J,mode,6,4,4)[1]
    np.testing.assert_allclose(analytic,(slope(I+h*J)-slope(I-h*J))/(2*h),atol=1e-9)


def test_area_weights_and_carrier_translation():
    m=module('iris-sampling/aa_iris.py');f=np.arange(-24,24)/16
    w=m.area_weights(f,.3,0);translated=m.area_weights(f,.3,4/16)
    assert np.all((w>=0)&(w<=1))
    np.testing.assert_array_equal(translated,np.roll(w,4,axis=1))


def test_explicit_domain_coverage_and_exclusive_output(tmp_path):
    m=module('state-built-reconstructor/domains.py')
    with pytest.raises(FileNotFoundError,match='Missing requested domains'):m.required_inputs(tmp_path,[6,12,15])
    common.save_json(tmp_path/'result.json',{'v':1})
    with pytest.raises(FileExistsError):common.save_json(tmp_path/'result.json',{'v':2})
    assert json.loads((tmp_path/'result.json').read_text())=={'v':1}


def test_target_only_mixing():
    m=module('iris-sampling/target_only.py')
    assert m.target_only(.985257,.08066,.991992)['target_only_percent']==pytest.approx(.764789,abs=1e-6)
    assert m.target_only(.985257,.08066,.978237)['target_only_percent']==pytest.approx(-.632419,abs=1e-6)


def test_all_entrypoints_are_portable(tmp_path):
    for path in sorted((ROOT/'analysis').rglob('*.py')):
        source=path.read_text()
        assert '/tmp/claude' not in source and 'exec(open(' not in source
        if path.name=='common.py': continue
        result=subprocess.run([sys.executable,str(path),'--help'],cwd=tmp_path,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,(path,result.stderr)


def test_fresh_directory_all_families(tmp_path):
    def run(relative,*args,success=True):
        cmd=[sys.executable,str(ROOT/relative),*map(str,args)]
        result=subprocess.run(cmd,cwd=tmp_path,capture_output=True,text=True,timeout=90)
        assert (result.returncode==0)==success,(cmd,result.stdout,result.stderr)
        return result
    run('reproduce_6d.py','--output','fresh','--canary','--wall-budget',60)
    run('analysis/state-built-reconstructor/grams6d.py','--output','grams6d.npz','--canary')
    for L in (12,15):run('analysis/state-built-reconstructor/grams_dom.py',L,'--output',f'grams{L}d.npz','--canary')
    for name in ('recon','weighted','offsets'):
        extra=['--fixed-phases',8] if name=='recon' else []
        run(f'analysis/state-built-reconstructor/{name}.py','grams6d.npz','--output',f'{name}.json','--phases',8,*extra)
    run('analysis/state-built-reconstructor/domains.py','.', '--output','domains.json','--phases',8)
    assert json.loads((tmp_path/'domains.json').read_text())['domains']==[6,12,15]
    run('analysis/iris-sampling/offsets_sref.py','fresh','--output','offset-sref.json','--phases',8)
    sampled=json.loads((tmp_path/'offset-sref.json').read_text())['results']
    Q=np.load(tmp_path/'fresh/original.npz')['gram'];w=np.load(tmp_path/'fresh/readout.npz')['weights']
    assert sampled['original']['gD']==pytest.approx(np.sum(w*common.yj(*common.state(Q,kind='D'))[1]),abs=2e-13)
    for relative,dirname,radius in [('aa_iris','iris',1.),('radius','radius',1.02)]:
        run(f'analysis/iris-sampling/{relative}.py','fresh/readout.npz','--output',dirname,'--domains',6,'--angles',8,'--radius',radius,'--orthogonal','--canary')
        assert len(json.loads((tmp_path/dirname/'summary.json').read_text())['rows'])==4
        assert len(list((tmp_path/dirname).glob(f'mats_area_R{radius:g}_*.npz')))==1
    run('analysis/iris-sampling/statebuilt_aa.py','--output','statebuilt-aa.json','--phases',8,'--canary')
    for name in ('slopes','offsets_slopes'):
        run(f'analysis/slope-readouts/{name}.py','grams6d.npz','--output',f'{name}.json','--phases',8,'--offset',6,'--pupil-radius',4,'--mask-radius',4)
    run('analysis/leakage/leak.py','grams6d.npz','--output','leak.json','--phases',8,'--leakage-phases',8)
    run('analysis/geometry-sweep/sweep.py','tiny',8,2,.125,1,'--output','geometry.json','--angles',8,'--phases',8,'--modulation',.73)
    run('analysis/iris-sampling/target_only.py','--baseline-target',.985257,'--baseline-pair-percent',.08066,'--shifted-target',.991992,'--output','mix.json')
    run('analysis/state-built-reconstructor/grams6d.py','--output','grams6d.npz','--canary',success=False)
    run('analysis/iris-sampling/aa_iris.py','fresh/readout.npz','--output','iris','--domains',6,'--angles',8,'--radius',1,'--canary',success=False)
