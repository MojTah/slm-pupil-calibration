import math
import numpy as np
import pytest
from finite_calibration import slope_moments, finite_poke, inverse_diagnostics, compare_diagnostics
from phase_pair import paired_matrices
from piston_basis import readout_matrices, readout_moments


def test_exact_binomial_slope_moments_and_zero_outcome():
    # Each independent poke has three pooled photons and pixel weights -2,+3.
    n, h = 3, .2
    probs = [.7, .2]
    moments = [{'weighted_mean': -2+5*p, 'variance_coefficient': 25*p*(1-p)}
               for p in probs]
    analytic = slope_moments(*moments, h)
    outcomes = []
    for k in range(n+1):
        for j in range(n+1):
            probability = (math.comb(n,k)*probs[0]**k*(1-probs[0])**(n-k)
                           *math.comb(n,j)*probs[1]**j*(1-probs[1])**(n-j))
            outcomes.append((5*(k-j)/(2*h*n), probability))
    mean = sum(x*p for x,p in outcomes)
    variance = sum(p*(x-mean)**2 for x,p in outcomes)
    assert sum(p for x,p in outcomes) == pytest.approx(1.)
    assert mean == pytest.approx(analytic['finite_gain'])
    assert variance == pytest.approx(analytic['gain_variance_coefficient']/(2*n))
    assert sum(p for x,p in outcomes if x == 0) > .1  # A reciprocal can be undefined.


def test_unequal_flux_pair_conditions_on_pooled_counts():
    bright = np.array([[1., 1.]], complex)
    tangent = np.array([[.4+.2j, -.3+.1j]])
    shadow = np.array([[1., 0.]], complex)
    fields = np.stack([bright, tangent, shadow], axis=-1)
    gram = fields[..., :,None].conj()*fields[..., None,:]
    weights = np.array([[-2., 3.]])
    matrices = paired_matrices(readout_matrices(gram, weights), 0.)
    h = .17
    explicit = []
    for a in [h, -h]:
        x = np.cos(a/2)*bright+2*np.sin(a/2)*tangent
        intensity = (abs(x+shadow)**2+abs(x-shadow)**2)/2
        y = intensity/intensity.sum()
        mean = float(np.sum(y*weights))
        explicit.append({'weighted_mean': mean,
                         'variance_coefficient': float(np.sum(y*(weights-mean)**2))})
    observed = finite_poke(matrices, h)
    expected = slope_moments(*explicit, h)
    for key in expected:
        assert observed[key] == pytest.approx(expected[key], rel=1e-13)
    # At zero poke the two exposure fluxes are5 and1; phase allocation is5:1.
    pooled = readout_moments(matrices, 0.)['weighted_mean']
    assert pooled == pytest.approx((-2*4+3*2)/6)
    assert pooled != pytest.approx(((-2*4+3)/5+3)/2)


def test_finite_poke_has_second_order_bias():
    # Lossless two-pixel sine response: probabilities (1+sin(a))/2 and complement.
    fields = np.array([[[1., .5, 0.], [1., -.5, 0.]]], complex)
    gram = fields[..., :,None].conj()*fields[..., None,:]
    matrices = readout_matrices(gram, np.array([[1., -1.]]))
    errors = []
    for h in [.1, .05]:
        m = finite_poke(matrices, h)
        assert m['finite_gain'] == pytest.approx(math.sin(h)/h)
        assert m['analytic_gain'] == pytest.approx(1.)
        errors.append(abs(m['finite_poke_bias']))
    assert errors[0]/errors[1] == pytest.approx(4., rel=.001)


def test_photon_scaling_and_local_expansion_cutoff():
    slope = {'finite_gain': 2., 'gain_variance_coefficient': 4.}
    first = inverse_diagnostics(slope, 2.2, 10000)
    second = inverse_diagnostics(slope, 2.2, 40000)
    assert first['gain_cv'] == pytest.approx(.01)
    assert first['linearized_scale_sd'] == pytest.approx(.011)
    assert first['second_order_scale_mean_shift'] == pytest.approx(.00011)
    assert second['linearized_scale_sd'] == pytest.approx(first['linearized_scale_sd']/2)
    bad = inverse_diagnostics(slope, 2.2, 100)
    assert bad['approximation_status'] == 'CV_EXCEEDS_REPORTING_CUTOFF'
    assert bad['scale_rss_proxy'] is None


@pytest.mark.parametrize('gain', [0., -1., 1e-12])
def test_unsafe_gain_is_not_inverted(gain):
    observed = inverse_diagnostics({'finite_gain': gain, 'gain_variance_coefficient': 1.}, 1., 100)
    assert observed['approximation_status'] == 'NONPOSITIVE_OR_NEAR_ZERO_GAIN'
    assert observed['gain_cv'] is None and observed['scale_error'] is None


def test_rejects_invalid_poke_variance_or_counts():
    good = {'weighted_mean': 0., 'variance_coefficient': 1.}
    for h in [0., -1., math.nan]:
        with pytest.raises(ValueError): slope_moments(good, good, h)
    with pytest.raises(ValueError):
        slope_moments(dict(good, variance_coefficient=-1), good, .1)
    for count in [0, 3, 2.5, math.inf]:
        with pytest.raises(ValueError):
            inverse_diagnostics({'finite_gain': 1., 'gain_variance_coefficient': 1.}, 1., count)


def test_undefined_comparisons_remain_explicit():
    a = {'scale_error':None, 'gain_variance_coefficient':0., 'approximation_status':'NONPOSITIVE_OR_NEAR_ZERO_GAIN'}
    b = {'scale_error':.2, 'gain_variance_coefficient':2., 'approximation_status':'LOCAL_EXPANSION_ONLY'}
    observed = compare_diagnostics(a,b)
    assert observed['scale_error_change'] is None
    assert observed['scale_comparison_reason'] == 'UNDEFINED_SCALE_ERROR'
    assert observed['relative_gain_variance_coefficient_change'] is None
    assert observed['variance_comparison_reason'] == 'ZERO_BASELINE_VARIANCE'
    a.update(scale_error=.1,gain_variance_coefficient=1.,approximation_status='LOCAL_EXPANSION_ONLY')
    valid = compare_diagnostics(a,b)
    assert valid['scale_error_change'] == pytest.approx(.1)
    assert valid['relative_gain_variance_coefficient_change'] == pytest.approx(1.)
    assert valid['both_local_expansion_reported']
