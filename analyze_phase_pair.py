"""Full-phase calibration envelopes after raw-exposure phase pairing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import psutil
from analyze_transfer import DEFAULT_RUNS,EXPECTED_GEOMETRY,B_TARGET_RUNS,load_gram,file_hash
from gain_phase import gain_coefficients,gain_phase_range
from phase_pair import paired_matrices
from piston_basis import evaluate,readout_matrices,readout_moments
from run_pilot import WallDeadline
from simulation import normalize_response

ROOT=Path(__file__).resolve().parent
KAPPAS=[0.,.005,.01,.025,.05,.1,.2,.5,1.]


def main():
    start=time.perf_counter();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=(ROOT/args.run_dir).resolve()
    if not out.is_relative_to((ROOT/'local-results').resolve()) or out.exists():parser.error('Require fresh study result directory')
    estimated=512*512*9*16*6+512*512*8*5
    if psutil.virtual_memory().available<2*estimated or estimated>2_000_000_000:raise MemoryError('Memory admission failed')
    sources=['analyze_phase_pair.py','phase_pair.py','test_phase_pair.py','analyze_transfer.py','gain_phase.py','piston_basis.py','simulation.py','modal_simulation.py']
    identities={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in sources}
    test_hashes={}
    for test in ['gain-tests.json','phase-pair-tests.json']:
        state=json.loads((ROOT/test).read_text())
        if state['status']!='PASS' or any(hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=v for n,v in state['sources_lf'].items()):raise ValueError('Stale tests: '+test)
        test_hashes[test]=file_hash(ROOT/test)
    with WallDeadline(30):
        _,ref,ref_id=load_gram('finite-s3-domain6-128');ref[...,2,:]=0;ref[..., :,2]=0
        y,j=normalize_response(*evaluate(ref,0.));w=j/np.sum(j*j);del ref
        wh=hashlib.sha256(w.tobytes()).hexdigest()
        assert wh=='81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972'
        rows=[];inputs={};curves={};gains={};pool_regressions=[]
        for name in DEFAULT_RUNS:
            manifest,gram,identity=load_gram(name);c=manifest['config'];g=c['geometry']
            if (g['diameter_pixels'],g['spider_pixels'],g['samples_per_pixel'],g['domain_diameters'],g['bright_carrier'],g['dark_carrier'],g['pupil_center_diameters'],c['iris'],c['modulation'],c['angles'],c['detector_diameters'])!=(128,8,1,EXPECTED_GEOMETRY[name][0],.125,-.125,.75,8.,3.,EXPECTED_GEOMETRY[name][1],4):raise ValueError('Unexpected geometry')
            inputs[name]=identity
            i0,j0=evaluate(gram,0.,0.);ip,jp=evaluate(gram,0.,np.pi)
            pooled=(i0+ip)/2;dp=(j0+jp)/2
            f=pooled.sum();pooled_gain=float(np.sum(w*(dp/f-pooled*dp.sum()/f**2)))
            mat=readout_matrices(gram,w);del gram,i0,j0,ip,jp,pooled,dp
            ideal=readout_moments(paired_matrices(mat,0.),0.)['local_gain']
            error=abs(pooled_gain-ideal)
            if error>1e-12:raise ValueError('Ideal-pair raw pooling regression failed')
            pool_regressions.append({'run':name,'explicit_raw_pooled_gain':pooled_gain,'paired_matrix_gain':ideal,'absolute_error':error})
            d=paired_matrices(mat,0);d[...,2,2]=0;targets={'D':d}
            if name!='modal-domain6-128':
                bm,bg,bid=load_gram(B_TARGET_RUNS[name],'B')
                if {k:v for k,v in bm['config'].items() if k!='scope'}!={k:v for k,v in c.items() if k!='scope'} or bm['config']['scope']!='opaque_control':raise ValueError('B mismatch')
                if np.any(bg[...,2,:]!=0) or np.any(bg[..., :,2]!=0):raise ValueError('B shadow nonzero')
                targets['B']=readout_matrices(bg,w);del bg;inputs[name+'_B']=bid
            for target,tm in targets.items():
                z=readout_moments(tm,0.);gx=z['local_gain']
                values=[readout_moments(tm,float(a)) for a in np.linspace(-1.,1.,201)]
                response=np.array([v['weighted_mean']-z['weighted_mean'] for v in values])
                for kappa in KAPPAS:
                    ext=gain_phase_range(gain_coefficients(paired_matrices(mat,kappa)))
                    if ext['inversion_status']!='WELL_CONDITIONED' or abs(gx)<=ext['inversion_floor']:raise ValueError('Noninvertible paired calibration')
                    gain=[ext[k] for k in ['minimum','maximum']]
                    errors=sorted([gx/v-1 for v in gain])
                    transfer=np.sort(np.array([response*(1/v-1/gx) for v in gain]),axis=0)
                    curves[name,target,kappa]=transfer;gains[name,target,kappa]=errors
                    row={'run':name,'target':target,'kappa':kappa,'gain_range':ext,'target_gain':gx,'scale_error_min':errors[0],'scale_error_max':errors[1],'maximum_transfer_contribution_rad':float(np.max(abs(transfer))),
                         'variance_ratio_to_self_min':min((gx/v)**2 for v in gain),'variance_ratio_to_self_max':max((gx/v)**2 for v in gain),
                         'examples':[{'piston_rad':float(np.linspace(-1.,1.,201)[i]),'transfer_min_rad':float(transfer[:,i].min()),'transfer_max_rad':float(transfer[:,i].max()),'self_error_rad':float(response[i]/gx-np.linspace(-1.,1.,201)[i])} for i in [110,200]]}
                    rows.append(row)
        comparisons=[]
        for lo,hi in [('modal-domain6-256','modal-domain12-256'),('modal-domain12-256','modal-domain15-256'),('modal-domain6-256','modal-domain15-256'),('modal-domain15-128','modal-domain15-256')]:
            for target in ['D','B']:
                for kappa in KAPPAS:
                    x=(lo,target,kappa);y=(hi,target,kappa)
                    ds=float(np.max(abs(np.array(gains[x])-gains[y])))
                    dt=float(np.max(abs(curves[x]-curves[y])))
                    comparisons.append({'lower':lo,'higher':hi,'target':target,'kappa':kappa,'scale_change':ds,'transfer_change_rad':dt,'targets_met':ds<.001 and dt<.005})
        # Coherent endpoints must reproduce previously reviewed transfer outputs.
        regressions=[];previous_packets={}
        for target,path in [('D','calibration-transfer-20260915'),('B','opaque-transfer-20260915')]:
            previous=json.loads((ROOT/'local-results'/path/'result.json').read_text())
            if previous['status']!='COMPLETE_ANALYSIS' or previous['noninvertible_runs']:raise ValueError('Incomplete regression packet')
            if file_hash(ROOT/'local-results'/path/'curves.csv')!=previous['curves_sha256']:raise ValueError('Regression curves changed')
            previous_packets[target]={'path':path,'status':previous['status'],'result_sha256':file_hash(ROOT/'local-results'/path/'result.json'),'curves_sha256':previous['curves_sha256']}
            for record in previous['records']:
                row=next(v for v in rows if (v['run'],v['target'],v['kappa'])==(record['run'],target,1.))
                error=max(abs(row['scale_error_min']-record['minimum_scale_error']),abs(row['scale_error_max']-record['maximum_scale_error']))
                assert error<1e-12
                regressions.append({'run':record['run'],'target':target,'maximum_scale_endpoint_error':error})
        if any(hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h for n,h in identities.items()):raise ValueError('Source drift')
        peak=psutil.Process().memory_info().peak_wset
        if peak>2_000_000_000:raise MemoryError('Observed peak exceeds2GB')
        out.mkdir();snap=out/'source-snapshot';snap.mkdir()
        for n in sources+['gain-tests.json','phase-pair-tests.json']:(snap/n).write_bytes((ROOT/n).read_bytes())
        result={'status':'COMPLETE_ANALYSIS','records':rows,'comparisons':comparisons,'inputs':inputs,'reference':ref_id,'weight_sha256':wh,'coherent_regressions':regressions,'raw_pool_regressions':pool_regressions,'previous_packets':previous_packets,'tests_sha256':test_hashes,'sources_lf':identities,'elapsed_seconds':time.perf_counter()-start,'peak_rss_bytes':peak,
                'interpretation':'Same total detected photons; pooled raw exposures; unchanged shadow power; unknown starting phase; calibration uncertainty omitted; envelope differences do not certify continuum error or hardware tolerance'}
        (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'rows':len(rows),'elapsed_seconds':result['elapsed_seconds'],'failed_comparisons':[c for c in comparisons if not c['targets_met']],
                      'largest_domain':[v for v in rows if v['run']=='modal-domain15-256' and v['kappa'] in [0.,.1,1.]]},indent=2))


if __name__=='__main__':main()
