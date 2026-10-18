"""Generate original and opaque 12D/15D Grams; orthogonal is optional."""
from pathlib import Path
import argparse
from grams6d import generate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('domain',type=int,choices=[6,8,10,12,15])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--canary',action='store_true')
    p.add_argument('--orthogonal',action='store_true');a=p.parse_args()
    generate(a.output,a.domain,a.canary,a.orthogonal)
