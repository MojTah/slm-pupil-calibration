"""Exact finite-poke slope moments; local reciprocal diagnostics only."""
import math
from piston_basis import readout_moments


def slope_moments(plus, minus, half_poke):
    """Variance coefficient multiplies 1/Ncal, with Ncal/2 pooled counts/poke."""
    values = [half_poke, plus['weighted_mean'], minus['weighted_mean'],
              plus['variance_coefficient'], minus['variance_coefficient']]
    if not all(math.isfinite(x) for x in values) or half_poke <= 0:
        raise ValueError('Require finite moments and a positive half-poke')
    if min(values[-2:]) < 0:
        raise ValueError('Require nonnegative readout variances')
    return {'finite_gain': (values[1]-values[2])/(2*half_poke),
            'gain_variance_coefficient': (values[3]+values[4])/(2*half_poke**2)}


def finite_poke(matrices, half_poke, phase=0.):
    values = slope_moments(readout_moments(matrices, half_poke, phase),
                           readout_moments(matrices, -half_poke, phase), half_poke)
    values['analytic_gain'] = readout_moments(matrices, 0., phase)['local_gain']
    values['finite_poke_bias'] = values['finite_gain']-values['analytic_gain']
    return values


def inverse_diagnostics(slope, target_gain, photons):
    """No operational inverse estimator or exact inverse moments are evaluated."""
    gh, coefficient = slope['finite_gain'], slope['gain_variance_coefficient']
    if not all(math.isfinite(x) for x in [gh, coefficient, target_gain, photons]):
        raise ValueError('Require finite gains, variance and photon count')
    if coefficient < 0 or photons < 2 or photons != int(photons) or int(photons) % 2:
        raise ValueError('Require nonnegative variance and an even total photon count')
    result = {'gain_standard_error': math.sqrt(coefficient/photons),
              'gain_cv': None, 'scale_error': None, 'linearized_scale_sd': None,
              'scale_rss_proxy': None, 'second_order_scale_mean_shift': None}
    if gh <= 1e-10 or target_gain <= 1e-10:
        return dict(result, approximation_status='NONPOSITIVE_OR_NEAR_ZERO_GAIN')
    q = target_gain/gh
    cv = result['gain_standard_error']/gh
    result.update(gain_cv=cv, scale_error=q-1)
    if cv > .05:
        return dict(result, approximation_status='CV_EXCEEDS_REPORTING_CUTOFF')
    return dict(result, approximation_status='LOCAL_EXPANSION_ONLY',
                linearized_scale_sd=abs(q)*cv,
                scale_rss_proxy=math.hypot(q-1, q*cv),
                second_order_scale_mean_shift=q*cv**2)


def compare_diagnostics(lower, higher):
    """Preserve unsupported ratios as undefined instead of dropping the case."""
    scale_defined = lower['scale_error'] is not None and higher['scale_error'] is not None
    variance_defined = lower['gain_variance_coefficient'] > 0
    return {
        'scale_error_change': abs(higher['scale_error']-lower['scale_error']) if scale_defined else None,
        'scale_comparison_reason': None if scale_defined else 'UNDEFINED_SCALE_ERROR',
        'relative_gain_variance_coefficient_change': abs(higher['gain_variance_coefficient']/lower['gain_variance_coefficient']-1) if variance_defined else None,
        'variance_comparison_reason': None if variance_defined else 'ZERO_BASELINE_VARIANCE',
        'both_local_expansion_reported': lower['approximation_status']==higher['approximation_status']=='LOCAL_EXPANSION_ONLY'}
