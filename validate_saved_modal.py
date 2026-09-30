"""Validate an immutable saved B modal basis without rebuilding it."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from run_modal import direct_check
import numpy as np
import psutil
import scipy
from modal_simulation import ModalModel
from piston_basis import gram_diagnostics
from run_pilot import ROOT,WallDeadline,text_hash,write_json
from simulation import Configuration


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True,type=Path);p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--wall-budget',required=True,type=float);args=p.parse_args()
    local=(ROOT/'local-results').resolve();parent=(ROOT/args.parent).resolve();out=(ROOT/args.run_dir).resolve()
    if parent.parent!=local or out.parent!=local or out.exists() or not 0<args.wall_budget<=240:p.error('Require immutable local parent, fresh local output and deadline <=240 seconds')
    original_path=parent/'result.json';original_hash=digest(original_path);original=json.loads(original_path.read_text())
    c=original['config'];g=Configuration(**c['geometry'])
    if c['scope']!='opaque_control' or c['model_kind']!='exact_held_pixel_modal_v1':p.error('Require modal B-only parent')
    if original['status'] not in ('PARTIAL','COMPLETE_VALIDATED'):p.error('Unsupported parent state')
    if original['status']=='PARTIAL' and (not original.get('deadline_expired') or original['checks'] or original['unfinished_artifacts']):p.error('Only a clean saved basis interrupted by the deadline may be recovered')
    block,=original['blocks'];artifact=parent/'B.npz'
    if block['family']!='B' or block['artifact']!='B.npz' or digest(artifact)!=block['sha256']:p.error('B basis identity mismatch')
    sources=['modal_simulation.py','test_modal_simulation.py','run_modal.py','simulation.py','piston_basis.py','run_pilot.py']
    identity=dict(original['identity'])
    for n in sources:
        if text_hash(ROOT/n)!=identity[n] or text_hash(parent/'source-snapshot'/n)!=identity[n]:p.error('Producer source identity changed: '+n)
    config_path=parent/'source-snapshot/config.json'
    if text_hash(config_path)!=identity['config'] or json.loads(config_path.read_text())!=c:p.error('Parent configuration mismatch')
    if digest(ROOT/'modal-tests.json')!=identity['prototype_test_record']:p.error('Parent prototype evidence changed')
    identity['validate_saved_modal.py']=text_hash(Path(__file__))
    n=g.diameter_pixels*g.domain_diameters;side=g.diameter_pixels*c['detector_diameters'];estimate=n*n*16*20+side*side*9*16*3
    if estimate>2_000_000_000 or psutil.virtual_memory().available<2*estimate:p.error('Memory admission failed')
    out.mkdir()
    state={'status':'RUNNING','model_kind':c['model_kind'],'execution_kind':'validation_only_recovery','config':c,'identity':identity,'blocks':[],'checks':[],
           'parent':{'run':parent.name,'manifest_sha256':original_hash,'status':original['status'],'basis_sha256':block['sha256'],'elapsed_seconds':original['elapsed_seconds'],'error':original.get('error')},
           'started_utc':datetime.now(timezone.utc).isoformat(),'process_id':psutil.Process().pid,'estimated_bytes':estimate,'producer_runtime':original['runtime'],
           'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'fft_workers':1,'thread_environment':{n:os.environ[n] for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}},
           'computational_deadline_seconds':args.wall_budget,'deadline_note':'Final identity checks and artifact publication follow canceled timer; reported elapsed time includes them'}
    write_json(out/'result.json',state);deadline=None
    try:
        snap=out/'source-snapshot';snap.mkdir()
        for name in sources+['validate_saved_modal.py','modal-tests.json']:shutil.copy2(ROOT/name,snap/name)
        shutil.copy2(config_path,snap/'config.json')
        deadline=WallDeadline(args.wall_budget-(time.perf_counter()-start))
        with deadline:
            with np.load(artifact) as data:basis={k:data[k] for k in ['gram','pupil_gram','flux_gram']}
            expected=[(side,side,3,3),(g.diameter_pixels,g.diameter_pixels,3,3),(3,3)]
            for value,shape in zip(basis.values(),expected):
                if value.shape!=shape or np.any(value[...,2,:]!=0) or np.any(value[..., :,2]!=0):raise ValueError('B shape or shadow mismatch')
                gram_diagnostics(value)
            gram_diagnostics(basis['flux_gram']-basis['gram'].sum(axis=(0,1)),parent_trace=float(np.trace(basis['flux_gram']).real))
            check=direct_check(ModalModel(g,c['iris']),basis,'B',-.37,0.,c)
            if psutil.Process().memory_info().peak_wset>2_000_000_000:raise MemoryError('Observed peak exceeds2GB')
        if digest(original_path)!=original_hash or digest(artifact)!=block['sha256']:raise ValueError('Parent changed')
        if any(text_hash(ROOT/n)!=identity[n] for n in sources+['validate_saved_modal.py']):raise ValueError('Source changed')
        if digest(ROOT/'modal-tests.json')!=identity['prototype_test_record'] or text_hash(config_path)!=identity['config']:raise ValueError('Test or configuration changed')
        state['publishing']='B.npz';write_json(out/'result.json',state)
        shutil.copy2(artifact,out/'B.tmp')
        if digest(out/'B.tmp')!=block['sha256']:raise ValueError('Copied artifact mismatch')
        (out/'B.tmp').replace(out/'B.npz');state.pop('publishing',None)
        state.update(status='COMPLETE_VALIDATED',blocks=[block],checks=[check])
    except (Exception,KeyboardInterrupt) as e:
        state.update(status='PARTIAL',error=type(e).__name__+': '+str(e))
    state.update(elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=psutil.Process().memory_info().peak_wset,deadline_expired=deadline.expired.is_set() if deadline else False)
    state['unfinished_artifacts']=[x.name for x in out.iterdir() if x.suffix in ('.tmp','.npz') and x.name not in [b['artifact'] for b in state['blocks']]]
    write_json(out/'result.json',state)
    print(json.dumps({k:state.get(k) for k in ['status','elapsed_seconds','peak_rss_bytes','deadline_expired','error']}))
    return 0 if state['status']=='COMPLETE_VALIDATED' else 1


if __name__=='__main__':raise SystemExit(main())
