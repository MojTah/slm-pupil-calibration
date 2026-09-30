"""Exact phase range of a normalized readout with positive flux.

This algebra uses fixed-piston intensity/overlap coefficients; it does not
establish optical convergence or equality with an opaque-spider reference.
"""
import cmath
import math


def phase_range(a0, a1, b0, b1):
    """Range of (a1+2 Re(exp(i phi)b1))/(a0+2 Re(exp(i phi)b0)).

    Return endpoint phases in radians and the phase-incoherent mean. Reject
    zero-flux phases. Near-zero positive flux can make the range ill-conditioned;
    report its relative margin without clipping it into a valid case.
    """
    a0, a1, b0, b1 = float(a0), float(a1), complex(b0), complex(b1)
    if not all(math.isfinite(x) for x in (a0, a1, b0.real, b0.imag, b1.real, b1.imag)):
        raise ValueError('Require finite coefficients')
    if a0 <= 0:
        raise ValueError('Require positive incoherent flux')
    t, z = 2*b0/a0, 2*b1/a0
    rho = abs(t)
    if rho >= 1:
        raise ValueError('Require strictly positive flux at every phase')
    mean = a1/a0
    h = z-mean*t
    margin = 1-rho
    denominator = margin*(1+rho)
    if h == 0:
        return {'minimum': mean, 'maximum': mean, 'minimum_phase': 0.,
                'maximum_phase': 0., 'incoherent_mean': mean,
                'relative_minimum_flux': margin, 'constant': True}
    cross = (h*t.conjugate()).real
    radical = math.hypot(cross, math.sqrt(denominator)*abs(h))
    # Product of roots is -|h|^2/denominator. This form avoids cancellation
    # when one endpoint is small compared with the other.
    q = -cross-math.copysign(radical, cross)
    shifts = sorted((q/denominator, -(abs(h)/q)*abs(h)))
    phases = [cmath.phase((h-delta*t).conjugate())+(math.pi if delta < 0 else 0.)
              for delta in shifts]
    result = {'minimum': mean+shifts[0], 'maximum': mean+shifts[1],
              'minimum_phase': phases[0] % (2*math.pi),
              'maximum_phase': phases[1] % (2*math.pi),
              'incoherent_mean': mean, 'relative_minimum_flux': margin,
              'constant': False}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite phase range')
    return result


def overlap_coefficients(matrices, piston):
    """Flux/readout coefficients from the first two 3x3 contractions."""
    c, k = math.cos(piston/2), 2*math.sin(piston/2)
    a, b = [], []
    for matrix in matrices[:2]:
        a.append(float((c*c*matrix[0, 0]+k*k*matrix[1, 1]+
                        c*k*(matrix[0, 1]+matrix[1, 0])+matrix[2, 2]).real))
        b.append(complex(c*matrix[0, 2]+k*matrix[1, 2]))
    return a[0], a[1], b[0], b[1]
