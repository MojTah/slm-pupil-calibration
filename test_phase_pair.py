import numpy as np
import pytest
from phase_pair import pair_coherence,paired_matrices
from piston_basis import evaluate,readout_matrices,readout_moments


@pytest.mark.parametrize('imbalance,error',[(0.,0.),(.03,.1),(-.07,-.13),(1.,.6),(-1.,-.2)])
def test_pair_matches_raw_exposure_pooling(imbalance,error):
    rng=np.random.default_rng(507)
    fields=rng.normal(size=(4,4,3))+1j*rng.normal(size=(4,4,3))
    gram=fields[..., :,None].conj()*fields[...,None,:]
    matrices=readout_matrices(gram,rng.normal(size=(4,4)))
    original=matrices.copy();gamma=pair_coherence(imbalance,error)
    assert abs(abs(gamma)**2-(imbalance**2+(1-imbalance**2)*np.sin(error/2)**2))<1e-15
    pair=paired_matrices(matrices,gamma)
    np.testing.assert_array_equal(original,matrices)
    for a,phi in [(-.37,.6),(.1,3.1),(1.,-.2)]:
        left=evaluate(matrices,a,phi);right=evaluate(matrices,a,phi+np.pi+error)
        for actual,x,y in zip(evaluate(pair,a,phi),left,right):
            np.testing.assert_allclose(actual,(1+imbalance)/2*x+(1-imbalance)/2*y,rtol=2e-14,atol=2e-13)
        # Unknown starting phase absorbs the argument of the complex coherence.
        for x,y in zip(evaluate(pair,a,phi),evaluate(paired_matrices(matrices,abs(gamma)),a,phi+np.angle(gamma))):
            np.testing.assert_allclose(x,y,rtol=2e-14,atol=2e-13)


def test_ideal_pair_equals_incoherent_moments():
    rng=np.random.default_rng(508);f=rng.normal(size=(4,4,3))+1j*rng.normal(size=(4,4,3))
    m=readout_matrices(f[..., :,None].conj()*f[...,None,:],rng.normal(size=(4,4)))
    for a in [-1.,0.,.13]:
        x=readout_moments(paired_matrices(m,0),a,1.7)
        y=readout_moments(m,a,0.,False)
        for k in x:np.testing.assert_allclose(x[k],y[k],rtol=2e-14,atol=2e-13)


def test_pool_before_normalizing_unequal_flux_images():
    bright=np.array([[1.,1.]],complex)
    tangent=np.array([[.4+.2j,-.3+.1j]])
    shadow=np.array([[1.,0.]],complex)
    fields=np.stack([bright,tangent,shadow],axis=-1)
    gram=fields[..., :,None].conj()*fields[...,None,:]
    weights=np.array([[1.,0.]])
    intensity=[];derivative=[]
    for sign in [1,-1]:
        field=bright+sign*shadow
        intensity.append(abs(field)**2)
        derivative.append(2*(field.conj()*tangent).real)
    pooled=(intensity[0]+intensity[1])/2
    pooled_derivative=(derivative[0]+derivative[1])/2
    flux=pooled.sum();y=pooled/flux
    j=pooled_derivative/flux-pooled*pooled_derivative.sum()/flux**2
    mean=float(np.sum(weights*y));gain=float(np.sum(weights*j))
    variance=float(np.sum(y*(weights-mean)**2))
    observed=readout_moments(paired_matrices(readout_matrices(gram,weights),0),0.)
    assert mean==pytest.approx(2/3)
    separately_normalized=(intensity[0]/intensity[0].sum()+intensity[1]/intensity[1].sum())/2
    assert float(np.sum(weights*separately_normalized))==pytest.approx(.4)
    np.testing.assert_allclose([observed[k] for k in ['weighted_mean','local_gain','variance_coefficient']],[mean,gain,variance],rtol=2e-14,atol=2e-13)


def test_rejects_nonphysical_pair():
    for args in [(1.01,0.),(0.,np.nan)]:
        with pytest.raises(ValueError):pair_coherence(*args)
    for c in [1.001,np.inf,np.nan]:
        with pytest.raises(ValueError):paired_matrices(np.eye(3),c)
