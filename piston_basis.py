"""Exact finite-piston reduction for the existing fixed-shadow scalar model.

This does not change the optical model or certify numerical resolution.
"""
import numpy as np

from simulation import crop_detector, ft, ift


def integrated_gram(model, fields, side):
    gram = np.zeros((side, side, 3, 3), dtype=np.complex128)
    for i, left in enumerate(fields):
        for j in range(i, len(fields)):
            product = left.conj()*fields[j]
            value = crop_detector(model.bin_intensity(product), side)
            gram[..., i, j] = value
            gram[..., j, i] = value.conj()
    return gram


def gram_diagnostics(gram, parent_trace=None):
    """Reject material Hermitian/PSD violations; do not repair a matrix."""
    if gram.shape[-2:] != (3, 3) or not np.isfinite(gram).all():
        raise ValueError('Require finite 3x3 Gram matrices')
    flat = gram.reshape(-1, 3, 3)
    scale = np.maximum(np.abs(np.trace(flat, axis1=1, axis2=2).real), np.finfo(float).tiny)
    if parent_trace is not None:
        if not np.isfinite(parent_trace) or parent_trace <= 0:
            raise ValueError('Parent trace must be positive and finite')
        # A subtracted flux remainder may be zero; roundoff scales with its parents.
        scale = np.maximum(scale, parent_trace)
    asymmetry = np.max(abs(flat-flat.swapaxes(1, 2).conj()), axis=(1, 2))/scale
    if np.max(asymmetry) > 1e-12:
        raise ValueError('Gram matrix is not Hermitian')
    relative_minimum = np.linalg.eigvalsh(flat)[:, 0]/scale
    if np.min(relative_minimum) < -1e-12:
        raise ValueError('Gram matrix is not positive semidefinite')
    return {'max_relative_asymmetry': float(np.max(asymmetry)),
            'minimum_eigenvalue_over_trace': float(np.min(relative_minimum))}


def build_basis(model, family, iris_radius=8., modulation=0., angles=128, detector_diameters=4):
    if family not in ('B', 'SLM'):
        raise ValueError('Family must be B or SLM')
    if not np.isfinite(modulation) or modulation < 0 or not isinstance(angles, int) or angles < 1:
        raise ValueError('Require nonnegative modulation and positive angle count')
    side = model.config.diameter_pixels*detector_diameters
    if side > model.coarse_n or side < 1 or not isinstance(side, int):
        raise ValueError('Detector must lie inside the propagation domain')
    bright, tangent = model.relay('B' if family == 'B' else 'D', iris_radius=iris_radius)
    fields = [bright, tangent]
    if family == 'SLM':
        whole, unused = model.relay('C', iris_radius=iris_radius)
        fields.append(whole-bright)
        del whole, unused
    pupil_gram = integrated_gram(model, fields, model.config.diameter_pixels)
    flux_gram = np.zeros((3, 3), dtype=np.complex128)
    for i, left in enumerate(fields):
        for j in range(i, len(fields)):
            flux_gram[i, j] = np.sum(left.conj()*fields[j])/model.config.samples_per_pixel**2
            flux_gram[j, i] = flux_gram[i, j].conj()
    theta = np.arange(angles)*2*np.pi/angles if modulation else [0.]
    gram = np.zeros((side, side, 3, 3), dtype=np.complex128)
    for angle in theta:
        tilt = np.exp(2j*np.pi*modulation/model.config.diameter_pixels*
                      (model.x[None, :]*np.cos(angle)+model.x[:, None]*np.sin(angle)))
        propagated = [ift(ft(field*tilt)*model.pyramid_mask) for field in fields]
        # Each angle is an intensity realization; never average its field coherently.
        gram += integrated_gram(model, propagated, side)/len(theta)
    diagnostics = {name: gram_diagnostics(value) for name, value in
                   [('detector', gram), ('pupil_square', pupil_gram), ('total_flux', flux_gram)]}
    diagnostics['exterior_flux'] = gram_diagnostics(
        flux_gram-gram.sum(axis=(0, 1)), parent_trace=float(np.trace(flux_gram).real))
    return {'gram': gram, 'pupil_gram': pupil_gram, 'flux_gram': flux_gram, 'diagnostics': diagnostics}


def evaluate(gram, piston, shadow_phase=0., coherent=True):
    """Raw quadratic form and its derivative at fixed shadow phase.

    Also accepts signed weighted Gram contractions; positivity is checked on
    physical coefficient maps by gram_diagnostics, without clipping outputs.
    """
    if not np.isfinite(piston) or not np.isfinite(shadow_phase):
        raise ValueError('Piston and shadow phase must be finite')
    c, s = np.cos(piston/2), np.sin(piston/2)
    u = np.array([c, 2*s, np.exp(1j*shadow_phase) if coherent else 0.], dtype=complex)
    du = np.array([-.5*s, c, 0.], dtype=complex)
    image = np.einsum('i,...ij,j->...', u.conj(), gram, u).real
    tangent = 2*np.einsum('i,...ij,j->...', u.conj(), gram, du).real
    if not coherent:
        image = image+gram[..., 2, 2].real
    return image, tangent


def readout_matrices(gram, weights):
    if weights.shape != gram.shape[:-2] or not np.isfinite(weights).all():
        raise ValueError('Weights must match the detector')
    return np.stack([np.sum(gram, axis=(0, 1)),
                     np.einsum('yx,yxij->ij', weights, gram),
                     np.einsum('yx,yxij->ij', weights*weights, gram)])


def readout_moments(matrices, piston, shadow_phase=0., coherent=True):
    powers, tangents = evaluate(matrices, piston, shadow_phase, coherent)
    flux, weighted, squared = powers
    if flux <= 0 or not np.isfinite(powers).all():
        raise ValueError('Require positive finite flux')
    mean = weighted/flux
    return {'weighted_mean': float(mean),
            'local_gain': float(tangents[1]/flux-weighted*tangents[0]/flux**2),
            'variance_coefficient': float(squared/flux-mean**2)}
