"""SLM-to-D/B scalar calibration transfer using immutable modal Gram results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import psutil

from gain_phase import gain_coefficients, gain_phase_range, gain_value
from piston_basis import evaluate, readout_matrices, readout_moments
from run_pilot import WallDeadline
from simulation import normalize_response

ROOT=Path(__file__).resolve().parent
DEFAULT_RUNS=['modal-domain6-128','modal-domain6-256','modal-domain12-256',
              'modal-domain15-128','modal-domain15-256']
EXPECTED_GEOMETRY=dict(zip(DEFAULT_RUNS,[(6,128),(6,256),(12,256),(15,128),(15,256)]))
B_TARGET_RUNS={name:name.replace('modal-domain','modal-opaque-domain') for name in DEFAULT_RUNS[1:]}
B_TARGET_RUNS['modal-domain15-256']='modal-opaque-domain15-256-recovered-20260915'


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gram(name, family='SLM'):
    folder=ROOT/'local-results'/name
    p=folder/'result.json';manifest=json.loads(p.read_text())
    if manifest['status']!='COMPLETE_VALIDATED':raise ValueError('Incomplete input: '+name)
    block,=[b for b in manifest['blocks'] if b['family']==family]
    if file_hash(folder/block['artifact'])!=block['sha256']:raise ValueError('Artifact changed')
    for source in ('simulation.py','piston_basis.py')+ (('modal_simulation.py',) if name.startswith('modal-') else ()):
        if hashlib.sha256((ROOT/source).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=manifest['identity'][source]:
            raise ValueError('Model source identity changed')
    with np.load(folder/block['artifact']) as data:gram=data['gram']
    return manifest,gram,{'manifest_sha256':file_hash(p),'basis_sha256':block['sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+')
    parser.add_argument('--target',choices=('D','B'),default='D')
    parser.add_argument('--run-dir',required=True,type=Path)
    args=parser.parse_args();start=time.perf_counter()
    if args.runs is None:
        args.runs=DEFAULT_RUNS if args.target=='D' else DEFAULT_RUNS[1:]
    folder=(ROOT/args.run_dir).resolve()
    if not folder.is_relative_to((ROOT/'local-results').resolve()) or folder.exists():
        parser.error('Require a fresh output below local-results')
    if not args.runs or len(set(args.runs))!=len(args.runs) or any(n not in DEFAULT_RUNS for n in args.runs):
        parser.error('Require a unique subset of the five frozen modal runs')
    if args.target=='B' and any(n not in B_TARGET_RUNS for n in args.runs):
        parser.error('Require a frozen matched opaque target for each selected run')
    estimated_bytes=512*512*9*16*6+512*512*8*5
    if estimated_bytes>2_000_000_000 or psutil.virtual_memory().available<2*estimated_bytes:
        raise MemoryError('Saved-array analysis memory admission failed')
    tests=json.loads((ROOT/'gain-tests.json').read_text())
    if tests['status']!='PASS' or any(hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h for n,h in tests['sources_lf'].items()):
        raise ValueError('Gain tests are failed or stale')
    sources=('analyze_transfer.py','gain_phase.py','test_gain_phase.py','piston_basis.py','simulation.py','modal_simulation.py')
    identity={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in sources}
    with WallDeadline(30):
        reference,ref,ref_id=load_gram('finite-s3-domain6-128')
        ref[...,2,:]=0;ref[..., :,2]=0
        y,j=normalize_response(*evaluate(ref,0.));weights=j/np.sum(j*j);origin=float(np.sum(weights*y));del ref
        weights_hash=hashlib.sha256(weights.tobytes()).hexdigest()
        if weights_hash!='81fc8a8066a6af2613e8b05ca4657368bf9562ea95bb746f178fb7e6ca3f4972':raise ValueError('Wrong readout weights')
        records=[];rows=[];all_curves={}
        for name in args.runs:
            manifest,gram,input_id=load_gram(name);c=manifest['config'];g=c['geometry']
            if c['model_kind']!='exact_held_pixel_modal_v1' or (g['diameter_pixels'],g['spider_pixels'],g['samples_per_pixel'],g['bright_carrier'],g['dark_carrier'],c['iris'],c['modulation'],c['detector_diameters'])!=(128,8,1,.125,-.125,8.,3.,4):
                raise ValueError('Unexpected physical configuration')
            if (g['domain_diameters'],c['angles'])!=EXPECTED_GEOMETRY[name] or g['pupil_center_diameters']!=.75:
                raise ValueError('Unexpected named domain, quadrature or pupil separation')
            matrices=readout_matrices(gram,weights);del gram
            dark=matrices.copy();dark[...,2,:]=0;dark[..., :,2]=0
            d_gain=readout_moments(dark,0.)['local_gain']
            target_id=None
            if args.target=='B':
                b_name=B_TARGET_RUNS[name]
                bm,bg,target_id=load_gram(b_name,'B')
                if {k:v for k,v in bm['config'].items() if k!='scope'}!={k:v for k,v in c.items() if k!='scope'}:
                    raise ValueError('Opaque target configuration mismatch')
                if bm['config']['scope']!='opaque_control' or np.any(bg[...,2,:]!=0) or np.any(bg[..., :,2]!=0):
                    raise ValueError('Opaque target contains a shadow component or wrong scope')
                dark=readout_matrices(bg,weights);del bg
            coeff=gain_coefficients(matrices,0.);ext=gain_phase_range(coeff)
            zero=readout_moments(dark,0.);target_gain=zero['local_gain']
            record={'run':name,'input':input_id,'configuration':c,'target':args.target,'target_input':target_id,'gain_phase_range':ext,'target_zero_gain':target_gain,'D_zero_gain':d_gain,'target_to_D_gain_ratio':target_gain/d_gain if abs(d_gain)>1e-10 else None}
            if ext['inversion_status']!='WELL_CONDITIONED' or abs(target_gain)<=ext['inversion_floor']:
                record['status']='NONINVERTIBLE_CALIBRATION';records.append(record);continue
            policy=[('nominal',0.),('minimum_gain',ext['minimum_phase']),('maximum_gain',ext['maximum_phase'])]
            record.update(status='EVALUATED',policies=[],minimum_scale_error=min(target_gain/ext[k]-1 for k in ('minimum','maximum')),maximum_scale_error=max(target_gain/ext[k]-1 for k in ('minimum','maximum')))
            curves={}
            for label,phi in policy:
                cm=readout_moments(matrices,0.,phi);cg=gain_value(coeff,phi)
                if abs(d_gain)<=ext['inversion_floor']:
                    raise ValueError('D reference gain cannot be inverted for decomposition')
                if abs(target_gain/cg-(target_gain/d_gain)*(d_gain/cg))>1e-12:
                    raise ValueError('Calibration ratio decomposition failed')
                poke={}
                for h in (.01,.005):
                    slope=(readout_moments(matrices,h,phi)['weighted_mean']-readout_moments(matrices,-h,phi)['weighted_mean'])/(2*h)
                    poke[str(h)]={'slope':slope,'absolute_error':abs(slope-cg)}
                if poke['0.005']['absolute_error']>1e-4*max(1.,abs(cg)):
                    raise ValueError('Half-step poke disagrees with analytic calibration slope')
                record['policies'].append({'label':label,'phase':phi,'calibration_gain':cg,'scale_error':target_gain/cg-1,'poke_checks':poke})
                current=[]
                for a in np.linspace(-1.,1.,201):
                    tm=readout_moments(dark,float(a));delta_r=tm['weighted_mean']-zero['weighted_mean'];self_error=delta_r/target_gain-a
                    transfer=delta_r*(1/cg-1/target_gain);variance=tm['variance_coefficient']/cg**2
                    if variance<0 or not np.isfinite([transfer,variance,self_error]).all():raise ValueError('Nonphysical transfer calculation')
                    row={'run':name,'policy':label,'calibration_phase':phi,'piston_rad':float(a),'scale_error':target_gain/cg-1,
                         'self_calibration_error_rad':float(self_error),'transfer_contribution_rad':float(transfer),'reset_total_error_rad':float(self_error+transfer),
                         'unreset_total_error_rad':float((tm['weighted_mean']-cm['weighted_mean'])/cg-a),
                         'transported_variance_coefficient':float(variance),'variance_ratio_to_self':float((target_gain/cg)**2),
                         'crossover_detected_photons':float(variance/transfer**2) if transfer!=0 else None}
                    rows.append(row);current.append(row)
                curves[label]=current
            all_curves[name]=curves;records.append(record)
        comparisons=[]
        pairs=[('modal-domain6-256','modal-domain12-256'),('modal-domain12-256','modal-domain15-256'),('modal-domain6-256','modal-domain15-256'),('modal-domain15-128','modal-domain15-256')]
        for lo,hi in pairs:
            if lo not in all_curves or hi not in all_curves:continue
            changes={}
            for label in ('nominal','minimum_gain','maximum_gain'):
                x,y=all_curves[lo][label],all_curves[hi][label]
                changes[label]={'scale_change':abs(x[0]['scale_error']-y[0]['scale_error']),
                                'max_transfer_contribution_change_rad':max(abs(a['transfer_contribution_rad']-b['transfer_contribution_rad']) for a,b in zip(x,y))}
            comparisons.append({'lower':lo,'higher':hi,'changes':changes,'targets_met':all(v['scale_change']<.001 and v['max_transfer_contribution_change_rad']<.005 for v in changes.values()),
                                'interpretation':'Gain-envelope extremizers can have different phases on each grid; this compares nominal and extremal transfer policies, not a uniform pointwise-in-phase error bound'})
        if any(hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h for n,h in identity.items()):raise ValueError('Analysis source changed')
        if psutil.Process().memory_info().peak_wset>2_000_000_000:raise MemoryError('Observed peak exceeds2GB')
        folder.mkdir();snapshot=folder/'source-snapshot';snapshot.mkdir()
        for n in sources:(snapshot/n).write_bytes((ROOT/n).read_bytes())
        (snapshot/'gain-tests.json').write_bytes((ROOT/'gain-tests.json').read_bytes())
        with (folder/'curves.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['run','policy','piston_rad']);w.writeheader();w.writerows(rows)
        result={'status':'COMPLETE_ANALYSIS','records':records,'comparisons':comparisons,'rows':len(rows),'sources_lf':identity,'reference':ref_id,'weight_sha256':weights_hash,'origin':origin,
                'elapsed_seconds':time.perf_counter()-start,'peak_rss_bytes':psutil.Process().memory_info().peak_wset,'curves_sha256':file_hash(folder/'curves.csv'),'gain_tests_sha256':file_hash(ROOT/'gain-tests.json'),
                'noninvertible_runs':[r['run'] for r in records if r['status']=='NONINVERTIBLE_CALIBRATION'],
                'target':args.target,
                'limitations':(['D target only; matched modal B missing'] if args.target=='D' else ['B target is the matched centered-iris opaque control, not an unfiltered telescope pupil'])+['Unknown quasistatic phase is a bounded nuisance, not a probability distribution','Calibration photon uncertainty excluded','No universal hardware tolerance or convergence of prior failed absolute observables claimed','Unreset errors and crossover counts are policy-specific, not full-phase extrema','An interior phase with g_C=g_X has zero transfer contribution and no finite crossover','Extremal-policy comparisons compare envelopes, not identical-phase errors','Gain extrema use a numerical stationary-point solver, not a certified enclosure']}
        (folder/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'rows':len(rows),'elapsed_seconds':result['elapsed_seconds'],'comparisons':comparisons},indent=2))


if __name__=='__main__':main()
