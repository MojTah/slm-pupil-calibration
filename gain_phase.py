"""Numerical stationary-point extrema of a normalized readout's piston gain.

This uses two tangent-half-angle charts, not the mean-response endpoint phases.
It returns diagnostics, not a rigorous floating-point error enclosure.
"""
import math

import numpy as np
from numpy.polynomial import Polynomial


def gain_value(coefficients, phase):
    """Coefficients: A0,A1,B0,B1,dA0/da,dA1/da,dB0/da,dB1/da."""
    a0,a1,b0,b1,da0,da1,db0,db1 = coefficients
    a0,a1,da0,da1 = [float(v.real) for v in (a0,a1,da0,da1)]
    z = np.exp(1j*phase)
    d,n,dd,dn = [a+2*(z*b).real for a,b in
                 ((a0,b0),(a1,b1),(da0,db0),(da1,db1))]
    if not np.isfinite([d,n,dd,dn]).all() or d <= 0:
        raise ValueError('Require positive flux')
    result = float((dn*d-n*dd)/d**2)
    if not math.isfinite(result):raise ValueError('Nonfinite gain')
    return result


def phase_derivatives(coefficients, phase):
    """First/second phase derivatives from the original trigonometric ratio."""
    a0,a1,b0,b1,da0,da1,db0,db1 = coefficients
    z=np.exp(1j*phase)
    d,n,dd,dn=[float(a.real)+2*(z*b).real for a,b in
               ((a0,b0),(a1,b1),(da0,db0),(da1,db1))]
    dp,np_,ddp,dnp=[-2*(z*b).imag for b in (b0,b1,db0,db1)]
    dpp,npp,ddpp,dnpp=[-2*(z*b).real for b in (b0,b1,db0,db1)]
    p=dn*d-n*dd
    pp=dnp*d+dn*dp-np_*dd-n*ddp
    ppp=dnpp*d+2*dnp*dp+dn*dpp-npp*dd-2*np_*ddp-n*ddpp
    terms=np.array([dnp/d,dn*dp/d**2,-np_*dd/d**2,-n*ddp/d**2,
                    -2*(dn*d-n*dd)*dp/d**3])
    second=ppp/d**2-4*pp*dp/d**3-2*p*dpp/d**3+6*p*dp**2/d**4
    if not np.isfinite(terms).all() or not np.isfinite(second):raise ValueError('Nonfinite gain derivative')
    return float(terms.sum()),float(second),float(1+np.sum(abs(terms)))


def gain_phase_range(coefficients):
    values = np.array(coefficients, dtype=complex, copy=True)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError('Require eight finite gain coefficients')
    if np.any(values[[0,1,4,5]].imag != 0):
        raise ValueError('Intensity and intensity-derivative coefficients must be real')
    a0 = float(values[0].real)
    if a0 <= 0 or 2*abs(values[2]) >= a0:
        raise ValueError('Require strictly positive flux at every phase')
    margin = 1-2*abs(values[2])/a0
    # At smaller margins, coefficient roundoff amplification can exceed the
    # two-chart accuracy check. This conservative domain includes the study.
    if margin < 1e-4:
        raise ValueError('Flux margin outside the validated numerical conditioning range')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        values /= a0
    if not np.isfinite(values).all():raise ValueError('Nonfinite normalized coefficients')
    a0,a1,b0,b1,da0,da1,db0,db1 = values

    def quadratic(a,b,origin):
        b = b*np.exp(1j*origin)
        return Polynomial([a.real+2*b.real,-4*b.imag,a.real-2*b.real])

    charts = []
    for origin in (0.,math.pi/2):
        d,n,dd,dn = [quadratic(a,b,origin) for a,b in
                    ((a0,b0),(a1,b1),(da0,db0),(da1,db1))]
        p = dn*d-n*dd
        first,second = p.deriv()*d,2*p*d.deriv()
        stationary = first-second
        if not all(np.isfinite(v.coef).all() for v in (d,n,dd,dn,p,first,second,stationary)):
            raise ValueError('Nonfinite stationary polynomial')
        scale = max(np.max(abs(first.coef)),np.max(abs(second.coef)),np.finfo(float).tiny)
        # The degree-five coefficient is identically zero, not an adjustable
        # truncation. Check its computed cancellation before solving the quartic.
        if len(stationary.coef)>5 and np.max(abs(stationary.coef[5:]))>64*np.finfo(float).eps*scale:
            raise ValueError('Stationary polynomial cancellation failed')
        c = stationary.coef[:5]
        phases = [(origin+math.pi)%(2*math.pi)]
        residuals = []
        trig_residuals = []
        polish_steps = []
        if np.any(c != 0):
            c = c/np.max(abs(c))
            for root in Polynomial(c).roots():
                if abs(root.imag) <= 1e-8*(1+abs(root.real)):
                    t = float(root.real)
                    # A scale-aware polynomial residual also handles large t.
                    if abs(t)>1:
                        residual = abs(Polynomial(c[::-1])(1/t))/sum(abs(c))
                    else:
                        residual = abs(Polynomial(c)(t))/sum(abs(c))
                    if residual > 1e-8:
                        raise ValueError('Unresolved stationary root')
                    residuals.append(float(residual))
                    phi=(origin+2*math.atan(t))%(2*math.pi)
                    # A small quartic-root error can be magnified by low flux.
                    # Polish locally against the original trigonometric ratio;
                    # never permit a jump to a different stationary branch.
                    for _ in range(3):
                        first_phi,second_phi,derivative_scale=phase_derivatives(values,phi)
                        if abs(first_phi)/derivative_scale<=1e-9 or second_phi==0:break
                        step=first_phi/second_phi
                        if not math.isfinite(step) or abs(step)>1e-3:raise ValueError('Stationary root needs a nonlocal repair')
                        polish_steps.append(abs(step));phi=(phi-step)%(2*math.pi)
                    first_phi,_,derivative_scale=phase_derivatives(values,phi)
                    trig_residual=abs(first_phi)/derivative_scale
                    if trig_residual>1e-7:raise ValueError('Original gain derivative does not vanish')
                    trig_residuals.append(trig_residual)
                    phases.append(phi)
        candidates = [(gain_value(values,phi),phi) for phi in phases]
        low,high = min(candidates),max(candidates)
        charts.append({'minimum':low[0],'maximum':high[0],
                       'minimum_phase':low[1],'maximum_phase':high[1],
                       'stationary_roots':len(residuals),
                       'maximum_trigonometric_stationarity_residual':max(trig_residuals,default=0.),
                       'maximum_phase_polish_step':max(polish_steps,default=0.),
                       'maximum_polynomial_residual':max(residuals,default=0.)})
    disagreement = max(abs(charts[0][k]-charts[1][k]) for k in ('minimum','maximum'))
    if disagreement > 1e-10*(1+max(abs(charts[0][k]) for k in ('minimum','maximum'))):
        raise ValueError('Phase charts disagree on gain extrema')
    result = dict(charts[0])
    floor=1e-10*(1+max(abs(result[k]) for k in ('minimum','maximum')))
    crossing=result['minimum'] < -floor and result['maximum'] > floor
    unresolved=(result['minimum']<=floor and result['maximum']>=-floor)
    result.update(relative_minimum_flux=float(margin),chart_disagreement=float(disagreement),
                  slope_crosses_zero=result['minimum']<=0<=result['maximum'],
                  inversion_floor=float(floor),
                  inversion_status='CROSSES_ZERO' if crossing else 'UNRESOLVED_NEAR_ZERO' if unresolved else 'WELL_CONDITIONED')
    return result


def gain_coefficients(matrices, piston=0.):
    """Flux and weighted-readout coefficients from Hermitian contractions."""
    c,k = math.cos(piston/2),2*math.sin(piston/2)
    dc,dk = -.5*math.sin(piston/2),math.cos(piston/2)
    a,b,da,db = [],[],[],[]
    for m in matrices[:2]:
        cross = (m[0,1]+m[1,0]).real
        a.append(float(c*c*m[0,0].real+k*k*m[1,1].real+c*k*cross+m[2,2].real))
        b.append(complex(c*m[0,2]+k*m[1,2]))
        da.append(float(2*c*dc*m[0,0].real+2*k*dk*m[1,1].real+(dc*k+c*dk)*cross))
        db.append(complex(dc*m[0,2]+dk*m[1,2]))
    return (a[0],a[1],b[0],b[1],da[0],da[1],db[0],db[1])
