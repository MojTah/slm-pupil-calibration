"""Original-command readout comparison over every explicitly requested domain."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser

import argparse
from recon import analyze

def required_inputs(directory, domains):
    paths={L:directory/f'grams{L}d.npz' for L in domains}
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError('Missing requested domains: '+', '.join(missing))
    return paths

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input_dir',type=Path)
    p.add_argument('--domains',type=int,nargs='+',default=[6,12,15]);p.add_argument('--phases',type=int,default=1800)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();new_output(a.output)
    paths=required_inputs(a.input_dir,a.domains)
    out={L:analyze(load_grams(path),a.phases,a.phases)['original'] for L,path in paths.items()}
    save_json(a.output,{'status':'COMPLETE','domains':a.domains,'phases':a.phases,'results':out})
