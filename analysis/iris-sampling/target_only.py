"""Mix a shifted target gain with the unchanged baseline calibration gain."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import yj, state, phases, load_grams, save_json, new_output, contraction_parser

import argparse

def target_only(baseline_target, baseline_pair_percent, shifted_target):
    calibration=baseline_target/(1+baseline_pair_percent/100)
    if not np.isfinite(calibration) or calibration==0: raise ValueError('Invalid calibration gain')
    return {'baseline_calibration_gain':calibration,'target_only_percent':100*(shifted_target/calibration-1)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-target',type=float,required=True);p.add_argument('--baseline-pair-percent',type=float,required=True)
    p.add_argument('--shifted-target',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();save_json(a.output,target_only(a.baseline_target,a.baseline_pair_percent,a.shifted_target))
