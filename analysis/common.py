"""Shared contractions and explicit local outputs for the added analyses."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def yj(intensity, tangent):
    flux = intensity.sum()
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError('Intensity needs positive finite total flux')
    return intensity/flux, tangent/flux-intensity*tangent.sum()/flux**2


def state(gram, phi=None, kind='C'):
    gg, gk = gram[..., 0, 0].real, gram[..., 0, 1]
    if kind == 'D':
        return gg, 2*gk.real
    ss = gram[..., 2, 2].real
    if kind == 'pair':
        return gg+ss, 2*gk.real
    e = np.exp(1j*phi)
    return (gg+ss+2*(e*gram[..., 0, 2]).real,
            2*(gk+np.conj(e)*np.conj(gram[..., 1, 2])).real)


def phases(count):
    if count < 4:
        raise ValueError('At least four phase samples are required')
    return np.linspace(0, 2*np.pi, count, endpoint=False)


def load_grams(path, required=('original', 'opaque')):
    with np.load(path, allow_pickle=False) as data:
        missing = set(required)-set(data.files)
        if missing:
            raise ValueError(f'Missing Gram cases in {path}: {sorted(missing)}')
        out = {key: data[key] for key in required}
        if 'orthogonal' in data.files:
            out['orthogonal'] = data['orthogonal']
    shapes = {v.shape[:2] for v in out.values()}
    if len(shapes) != 1 or any(v.ndim != 4 or v.shape[-2:] != (3, 3)
                               or not np.isfinite(v).all() for v in out.values()):
        raise ValueError('Require finite, equally sized detector Grams ending in (3,3)')
    return out


def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, value):
    path = new_output(path)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x))
        handle.write('\n')


def save_npz(path, **arrays):
    path = new_output(path)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def contraction_parser(description, phase_count=720):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('grams', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phases', type=int, default=phase_count)
    return parser
