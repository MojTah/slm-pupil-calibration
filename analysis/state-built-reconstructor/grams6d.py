"""Generate native-readout Grams from public operators, with no saved numerical input."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, save_npz, new_output

import argparse
from simulation import Configuration
from modal_simulation import ModalModel, build_modal_basis
from orthogonal_carrier import OrthogonalStripModel

def generate(output, domain=6, canary=False, include_orthogonal=True):
    new_output(output)
    cfg=Configuration(diameter_pixels=8 if canary else 128,spider_pixels=2 if canary else 8,
        samples_per_pixel=1,domain_diameters=domain,bright_carrier=.125,dark_carrier=-.125,pupil_center_diameters=.75)
    cases=[('original',ModalModel,'SLM'),('opaque',ModalModel,'B')]
    if include_orthogonal: cases.append(('orthogonal',OrthogonalStripModel,'SLM'))
    out={}
    for name,model,family in cases:
        out[name]=build_modal_basis(model(cfg,1. if canary else 8.),family,.73 if canary else 3.,8 if canary else 256,4)['gram']
        print('Completed',name,'domain',domain,flush=True)
    save_npz(output,**out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--canary',action='store_true');a=p.parse_args();generate(a.output,canary=a.canary)
