"""Independent scalar checks of the continuous-phase envelope."""
import math
import cmath
import random

import pytest
from phase_envelope import phase_range


def value(a0, a1, b0, b1, phi):
    return (a1+2*(cmath.exp(1j*phi)*b1).real)/(a0+2*(cmath.exp(1j*phi)*b0).real)


@pytest.mark.parametrize('ratio', [-3., 0., 2.5])
def test_proportional_coefficients(ratio):
    r = phase_range(2., 2*ratio, .125+.25j, ratio*(.125+.25j))
    assert r['constant']
    assert r['minimum'] == r['maximum'] == ratio


def test_zero_flux_modulation_has_known_sinusoidal_range():
    r = phase_range(4., 8., 0., .6+.8j)
    assert r['minimum'] == 1.5
    assert r['maximum'] == 2.5


@pytest.mark.parametrize('b0', [.5, .5j, .6])
def test_reject_zero_or_negative_flux(b0):
    with pytest.raises(ValueError, match='strictly positive'):
        phase_range(1., 0., b0, .1)


@pytest.mark.parametrize('a0', [0., -1., math.nan, math.inf])
def test_invalid_flux(a0):
    with pytest.raises(ValueError):
        phase_range(a0, 1., 0., .1)


def test_random_complex_cases_against_scalar_phase_evaluation():
    rng = random.Random(260915)
    for _ in range(24):
        a0 = rng.uniform(.5, 3.)
        a1 = rng.uniform(-4., 4.)
        b0 = .5*a0*rng.uniform(0., .8)*cmath.exp(1j*rng.uniform(0., 2*math.pi))
        b1 = rng.uniform(0., 2.)*cmath.exp(1j*rng.uniform(0., 2*math.pi))
        r = phase_range(a0, a1, b0, b1)
        for end in ('minimum', 'maximum'):
            assert value(a0,a1,b0,b1,r[end+'_phase']) == pytest.approx(r[end], rel=1e-12, abs=1e-12)
        for j in range(4096):
            v = value(a0,a1,b0,b1,2*math.pi*j/4096)
            assert r['minimum']-1e-12 <= v <= r['maximum']+1e-12


def test_near_zero_positive_flux_and_small_root():
    r = phase_range(1., 0., .499999, .25)
    assert r['minimum'] == pytest.approx(-.5/(1-.999998), rel=1e-10)
    assert r['maximum'] == pytest.approx(.5/(1+.999998), rel=1e-12)
    assert r['relative_minimum_flux'] > 0
