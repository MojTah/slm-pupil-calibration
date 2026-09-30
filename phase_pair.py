"""Raw-exposure phase pairing of the three-field Gram representation."""
import numpy as np


def pair_coherence(imbalance=0., phase_error=0.):
    if not np.isfinite([imbalance,phase_error]).all() or abs(imbalance)>1:
        raise ValueError('Require finite phase error and exposure imbalance in [-1,1]')
    return (1+imbalance)/2-(1-imbalance)*np.exp(1j*phase_error)/2


def paired_matrices(matrices, coherence):
    if not np.isfinite(coherence) or abs(coherence)>1+1e-14:
        raise ValueError('Require a physical coherence magnitude no greater than one')
    result=np.array(matrices,dtype=complex,copy=True)
    if result.shape[-2:]!=(3,3) or not np.isfinite(result).all():
        raise ValueError('Require finite three-field Gram matrices')
    result[...,:2,2]*=coherence
    result[...,2,:2]*=np.conj(coherence)
    return result
