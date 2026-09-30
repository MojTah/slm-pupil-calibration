"""Bounded, immutable runs of the exact held-pixel finite-modal model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import numpy as np
import psutil
import scipy
from modal_simulation import ModalModel, build_modal_basis
from run_pilot import ROOT, WallDeadline, text_hash, write_json
from piston_basis import evaluate
from simulation import Configuration, crop_detector, normalize_response


def direct_check(model, basis, case, piston, phi, config):
    modes,tangent=model.relay_modes(case,piston,phi)
    intensity,derivative=model.sensor_modes(modes,tangent,config['modulation'],config['angles'])
    side=model.grid.config.diameter_pixels*config['detector_diameters']
    predicted=evaluate(basis['gram'],piston,phi)
    direct=(crop_detector(intensity,side),crop_detector(derivative,side))
    errors={}
    for name,a,b in [('intensity',predicted[0],direct[0]),('tangent',predicted[1],direct[1]),
                     *[(k,a,b) for k,a,b in zip(('normalized_intensity','normalized_tangent'),normalize_response(*predicted),normalize_response(*direct))]]:
        error=float(np.linalg.norm(a-b)); scale=float(np.linalg.norm(b))
        if not np.isfinite([error,scale]).all() or error > 1e-11*scale+1e-12:
            raise ValueError('Modal direct validation failed: '+name)
        errors[name]={'absolute':error,'relative':error/scale if scale>1e-12 else None}
    flux,flux_tangent=evaluate(basis['flux_gram'],piston,phi)
    if abs(intensity.sum()-flux)>1e-11*flux+1e-12 or abs(derivative.sum()-flux_tangent)>1e-11*flux+1e-12:
        raise ValueError('Full modal flux identity failed')
    if intensity.min() < -1e-12 or predicted[0].sum()>flux*(1+1e-12):
        raise ValueError('Nonphysical intensity or cropped flux')
    return {'case':case,'piston':piston,'shadow_phase':phi,'errors':errors}


def main():
    start=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--wall-budget',type=float,default=60.)
    args=parser.parse_args()
    if not 0<args.wall_budget<=600: parser.error('Require a deadline in (0,600] seconds')
    folder=(ROOT/args.run_dir).resolve()
    if not folder.is_relative_to((ROOT/'local-results').resolve()) or folder.exists():
        parser.error('Require a new run directory below local-results')
    config_bytes=args.config.read_bytes()
    config=json.loads(config_bytes.decode('utf-8'))
    if config['schema_version']!=1 or config['model_kind']!='exact_held_pixel_modal_v1':
        parser.error('Unsupported configuration identity')
    geometry=Configuration(**config['geometry'])
    if config['scope'] not in ('full','slm_phase_contrast','opaque_control'): parser.error('Unsupported scope')
    if not isinstance(config['angles'],int) or isinstance(config['angles'],bool) or config['angles']<1:
        parser.error('Require positive integer angles')
    if not np.isfinite(config['modulation']) or config['modulation']<0: parser.error('Invalid modulation')
    side=geometry.diameter_pixels*config['detector_diameters']
    n=geometry.diameter_pixels*geometry.domain_diameters
    if not isinstance(side,int) or not 1<=side<=n: parser.error('Invalid detector')
    # Includes the nine full-domain product maps, relay/propagated fields,
    # transform temporaries, geometry and cropped outputs. Check observed RSS.
    estimated=n*n*16*32+side*side*9*16*2
    if estimated>2_000_000_000 or psutil.virtual_memory().available<2*estimated:
        parser.error('Memory admission failed')
    sources=('modal_simulation.py','test_modal_simulation.py','run_modal.py','simulation.py','piston_basis.py','run_pilot.py')
    source_bytes={p:(ROOT/p).read_bytes() for p in sources}
    identities={p:hashlib.sha256(v.replace(b'\r\n',b'\n')).hexdigest() for p,v in source_bytes.items()}
    test_path=ROOT/'modal-tests.json'
    test_bytes=test_path.read_bytes()
    tests=json.loads(test_bytes.decode('utf-8'))
    if tests['status']!='PASS' or any(identities[p]!=v for p,v in tests['sources_lf'].items()):
        parser.error('Modal prototype tests failed or stale')
    identities['config']=hashlib.sha256(config_bytes.replace(b'\r\n',b'\n')).hexdigest()
    identities['prototype_test_record']=hashlib.sha256(test_bytes).hexdigest()
    folder.mkdir()
    state={'status':'RUNNING','model_kind':config['model_kind'],'config':config,'identity':identities,
           'started_utc':datetime.now(timezone.utc).isoformat(),'process_id':os.getpid(),
           'estimated_bytes':estimated,'blocks':[],'checks':[],
           'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'fft_workers':1},
           'interpretation':'Direct reduction validation for a distinct modal operator; angular/domain convergence not established'}
    process=psutil.Process()
    def save():
        state.update(elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=process.memory_info().peak_wset,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/'result.json',state)
    save()
    deadline=None
    try:
        snapshot=folder/'source-snapshot'; snapshot.mkdir()
        for name,data in source_bytes.items(): (snapshot/name).write_bytes(data)
        (snapshot/'config.json').write_bytes(config_bytes)
        (snapshot/'modal-tests.json').write_bytes(test_bytes)
        deadline=WallDeadline(args.wall_budget-(time.perf_counter()-start))
        with deadline:
            model=ModalModel(geometry,config['iris'])
            families={'full':('B','SLM'),'slm_phase_contrast':('SLM',),'opaque_control':('B',)}
            for family in families[config['scope']]:
                state['active']=family; save()
                print(json.dumps({'started':family,'seconds':time.perf_counter()-start}),flush=True)
                basis=build_modal_basis(model,family,config['modulation'],config['angles'],config['detector_diameters'])
                if process.memory_info().peak_wset>2_000_000_000: raise MemoryError('Observed basis peak exceeds2GB')
                path=folder/(family+'.npz')
                state['publishing']=path.name; save()
                with path.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle,**{k:v for k,v in basis.items() if k!='diagnostics'})
                path.with_suffix('.tmp').replace(path)
                state['blocks'].append({'family':family,'artifact':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'diagnostics':basis['diagnostics']})
                state.pop('publishing',None)
                save()
                case='B' if family=='B' else 'C'
                state['checks'].append(direct_check(model,basis,case,-.37,0. if family=='B' else np.pi/2,config))
                if process.memory_info().peak_wset>2_000_000_000: raise MemoryError('Observed peak exceeds2GB')
                save(); del basis
            if any(text_hash(ROOT/p)!=v for p,v in identities.items() if p in sources):
                raise RuntimeError('Source identity changed during run')
            if args.config.read_bytes()!=config_bytes or test_path.read_bytes()!=test_bytes:
                raise RuntimeError('Configuration or test-record identity changed during run')
        state['status']='COMPLETE_VALIDATED'; state.pop('active',None)
    except (Exception,KeyboardInterrupt) as error:
        state.update(status='PARTIAL',error=type(error).__name__+': '+str(error))
    state['deadline_expired']=deadline.expired.is_set() if deadline else False
    manifest_artifacts={b['artifact'] for b in state['blocks']}
    state['unfinished_artifacts']=[p.name for p in folder.iterdir()
        if p.suffix in ('.tmp','.npz') and p.name not in manifest_artifacts and p.name!='result.tmp']
    save()
    print(json.dumps({k:state.get(k) for k in ('status','elapsed_seconds','peak_rss_bytes','error')}))
    return 0 if state['status']=='COMPLETE_VALIDATED' else 1


if __name__=='__main__':
    raise SystemExit(main())
