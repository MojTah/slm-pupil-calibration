"""Independent scalar and quadratic-form checks for gain extrema."""
import math
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from gain_phase import gain_coefficients, gain_phase_range, gain_value
from piston_basis import readout_moments


def independent_gain(v,phi):
    a0,a1,b0,b1,d0,d1,e0,e1 = v
    z = complex(math.cos(phi),math.sin(phi))
    f = a0+2*(z*b0).real
    n = a1+2*(z*b1).real
    return (d1+2*(z*e1).real)/f-n*(d0+2*(z*e0).real)/(f*f)


def bracketed_extrema(v):
    # This is independent of the stationary polynomial and of either phase chart.
    x=np.linspace(0,2*np.pi,4097)[:-1]
    y=np.array([independent_gain(v,t) for t in x]); step=2*np.pi/len(x)
    candidates=[]
    for sign in (-1,1):
        for i in np.flatnonzero((sign*y<=sign*np.roll(y,1))&(sign*y<=sign*np.roll(y,-1))):
            q=minimize_scalar(lambda t:sign*independent_gain(v,t),bounds=(x[i]-step,x[i]+step),method='bounded',options={'xatol':1e-13})
            candidates.append(independent_gain(v,q.x))
    return min(candidates),max(candidates)


@pytest.mark.parametrize('phase',[0.,np.pi-1e-9,np.pi/2,1.731])
def test_analytic_sinusoid(phase):
    v=(1.,0.,0j,0j,0.,1.2,0j,.35*np.exp(-1j*phase))
    r=gain_phase_range(v)
    assert r['minimum']==pytest.approx(.5,abs=2e-13)
    assert r['maximum']==pytest.approx(1.9,abs=2e-13)
    assert not r['slope_crosses_zero']


@pytest.mark.parametrize('slope',[0.,-.8,1.3])
def test_constant_and_zero_gain(slope):
    # N=0, N_a=slope*D gives constant gain with a variable positive denominator.
    b=.2+.1j;v=(1.,0.,b,0j,.4,slope,.03j,slope*b)
    r=gain_phase_range(v)
    assert r['minimum']==pytest.approx(slope,abs=3e-13)
    assert r['maximum']==pytest.approx(slope,abs=3e-13)
    assert r['slope_crosses_zero']==(slope==0)


def test_gain_sign_change():
    r=gain_phase_range((1.,0.,0j,0j,0.,.1,0j,.3j))
    assert r['minimum']<0<r['maximum']
    assert r['slope_crosses_zero']
    assert r['inversion_status']=='CROSSES_ZERO'


@pytest.mark.parametrize('offset',[0.,1e-8])
def test_touching_zero_and_nearby_positive(offset):
    z=np.exp(-1j*.837)
    v=(1.,0.,0j,.5*z,0.,1.+offset,-.5*z,-z)
    r=gain_phase_range(v)
    assert r['minimum']==pytest.approx(offset,abs=2e-13)
    assert r['inversion_status']==('UNRESOLVED_NEAR_ZERO' if offset==0 else 'WELL_CONDITIONED')


@pytest.mark.parametrize('margin',[.14,.001,1.01e-4])
@pytest.mark.parametrize('theta',[0.,np.pi-1e-8,1.13])
def test_reciprocal_flux_at_supported_boundary(margin,theta):
    rho=1-margin;v=(1.,0.,rho/2*np.exp(-1j*theta),0j,0.,1.,0j,0j)
    r=gain_phase_range(v)
    np.testing.assert_allclose([r['minimum'],r['maximum']],[1/(1+rho),1/(1-rho)],rtol=2e-10,atol=1e-11)


def test_input_is_not_mutated():
    v=np.array([2.,.4,.2j,.1j,0.,1.,.1,.2j],dtype=complex);before=v.copy()
    gain_phase_range(v)
    assert np.array_equal(v,before)


def test_normalization_overflow_rejected():
    with pytest.raises(ValueError):gain_phase_range((1e-300,1e300,0j,0j,0.,1.,0j,0j))


def test_variable_denominators_against_independent_search():
    rng=np.random.default_rng(1863)
    for _ in range(24):
        v=(1.,float(rng.normal()),.43*np.exp(1j*rng.uniform(0,2*np.pi)),complex(*rng.normal(size=2)),float(rng.normal()),float(rng.normal()),complex(*rng.normal(size=2)),complex(*rng.normal(size=2)))
        r=gain_phase_range(v);lo,hi=bracketed_extrema(v)
        np.testing.assert_allclose([r['minimum'],r['maximum']],[lo,hi],rtol=2e-10,atol=2e-10)
        assert independent_gain(v,r['minimum_phase'])==pytest.approx(r['minimum'],rel=2e-12,abs=2e-12)
        assert independent_gain(v,r['maximum_phase'])==pytest.approx(r['maximum'],rel=2e-12,abs=2e-12)


def test_coefficients_against_original_quadratic_form():
    rng=np.random.default_rng(963)
    z=rng.normal(size=(31,3))+1j*rng.normal(size=(31,3));q=np.einsum('ni,nj->nij',z.conj(),z)
    w=rng.normal(size=31);m=np.stack([q.sum(axis=0),np.einsum('n,nij->ij',w,q),np.einsum('n,nij->ij',w*w,q)])
    for a in (-.91,0.,.73):
        v=gain_coefficients(m,a)
        for phi in (0.,.36,np.pi,5.9):
            assert gain_value(v,phi)==pytest.approx(readout_moments(m,a,phi)['local_gain'],rel=2e-13,abs=2e-13)
        r=gain_phase_range(v);lo,hi=bracketed_extrema(v)
        np.testing.assert_allclose([r['minimum'],r['maximum']],[lo,hi],rtol=2e-10,atol=2e-10)


@pytest.mark.parametrize('v',[(0.,0.,0j,0j,0.,0.,0j,0j),(1.,0.,.5,0j,0.,0.,0j,0j),(1.,0.,.5-1e-10,0j,0.,0.,0j,0j),(1.,float('nan'),0j,0j,0.,0.,0j,0j),(1.+1j,0.,0j,0j,0.,0.,0j,0j)])
def test_invalid_or_unvalidated_flux(v):
    with pytest.raises(ValueError):gain_phase_range(v)
