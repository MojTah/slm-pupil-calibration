"""Independent physical fixtures for the distinct finite modal model."""
import math
import numpy as np
import pytest

from modal_simulation import ModalModel, build_modal_basis
from piston_basis import evaluate
from simulation import Configuration, crop_detector, ft, ift

RTOL, ATOL = 2e-12, 2e-13


def sinc(x):
    return 1. if x == 0 else math.sin(math.pi*x)/(math.pi*x)


@pytest.fixture(scope='module')
def model():
    return ModalModel(Configuration(diameter_pixels=8, spider_pixels=2,
                      samples_per_pixel=1, domain_diameters=6), iris_radius=1.)


def assert_close(a, b):
    np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL)


def test_held_pixel_coefficients_and_translation(model):
    g = model.grid; n = g.n
    command = np.zeros((n,n)); command[7,13] = 1.
    modes = model.held_coefficients(command)*n*n
    shifted = model.held_coefficients(np.roll(command,1,axis=1))*n*n
    for mx,my in [(0,0),(1,0),(-3,2),(4,-5),(-5,-4)]:
        expected = sinc(mx/n)*sinc(my/n)*np.exp(-2j*np.pi*(mx*g.x[13]+my*g.x[7])/n)
        assert_close(modes[n//2+my,n//2+mx], expected)
        assert_close(shifted[n//2+my,n//2+mx], expected*np.exp(-2j*np.pi*mx/n))


@pytest.mark.parametrize('sideband', [-1,0,1])
def test_carrier_hold_attenuation_before_removal(model, sideband):
    g=model.grid; q=model.carrier_index
    command=np.broadcast_to(np.exp(2j*np.pi*(q+sideband)*g.x/g.n), (g.n,g.n)).copy()
    selected=model.held_coefficients(command)*g.iris(model.iris_radius,g.config.bright_carrier)
    removed=np.roll(selected,-q,axis=1)
    expected=sinc((q+sideband)/g.n)*np.exp(2j*np.pi*sideband*g.x/g.n)
    assert_close(model.samples(removed),np.broadcast_to(expected,(g.n,g.n)))
    assert_close(np.sum(abs(model.samples(removed))**2),g.n*g.n*np.sum(abs(removed)**2))
    rejected=np.conj(command) if sideband == 0 else None
    if rejected is not None:
        output=model.held_coefficients(rejected)*g.iris(model.iris_radius,g.config.bright_carrier)
        assert np.max(abs(output)) < ATOL


@pytest.fixture(scope='module')
def two_modes(model):
    n=model.grid.n
    modes=np.zeros((n,n),complex)
    modes[n//2+3,n//2]=1.
    modes[n//2-3,n//2-4]=(1+1j)/2
    return modes


def test_noncommensurate_tilt_complex_field_and_pixel_integral(model,two_modes):
    g=model.grid; n=g.n; tilt=(.37/n,.23/n)
    yy,xx=np.meshgrid(g.x,g.x,indexing='ij')
    terms=[]
    for mx,my,amplitude in [(0,3,1.),(-4,-3,(1+1j)/2)]:
        shift=g.config.pupil_center_diameters*g.config.diameter_pixels
        multiplier=np.exp(-2j*np.pi*shift*(abs(mx/n+tilt[0])+abs(my/n+tilt[1])))
        terms.append(amplitude*multiplier*np.exp(2j*np.pi*(mx*xx+my*yy)/n))
    envelope=model.envelope(two_modes,tilt)
    assert_close(envelope,terms[0]+terms[1])
    output_tilt=np.exp(2j*np.pi*(tilt[0]*xx+tilt[1]*yy))
    assert_close(envelope*output_tilt,(terms[0]+terms[1])*output_tilt)
    cross=terms[0].conj()*terms[1]
    exact_cross=cross*sinc(-4/n)*sinc(-6/n)
    assert_close(model.integrate_product(cross),exact_cross)
    expected=1.5+2*exact_cross.real
    assert_close(model.integrate_product(abs(envelope)**2),expected)
    assert_close(expected.sum(),1.5*n*n)
    assert np.max(abs(expected-abs(envelope)**2)) > 1e-3


@pytest.mark.parametrize('indices', [(0,0),(1,-2)])
def test_shared_commensurate_operator_boundary(model,two_modes,indices):
    g=model.grid; tilt=np.array(indices)/g.n
    phase=np.exp(2j*np.pi*(tilt[0]*g.x[None,:]+tilt[1]*g.x[:,None]))
    old=ift(ft(model.samples(two_modes)*phase)*g.pyramid_mask)
    assert_close(old,model.envelope(two_modes,tilt)*phase)


@pytest.fixture(scope='module')
def bases(model):
    return {(family,mu):build_modal_basis(model,family,modulation=mu,angles=8)
            for family in ('B','SLM') for mu in (0.,.73)}


@pytest.mark.parametrize('mu', [0.,.73])
@pytest.mark.parametrize('case,a,phi', [('B',-.37,0.),('D',.29,0.),('C',-.37,np.pi/2),('C',.29,.63),('C',0.,0.)])
def test_gram_against_independently_commanded_source(model,bases,mu,case,a,phi):
    g=model.grid
    aperture=g.pupil if case=='C' else g.bright
    phase=a*g.piston_shape
    if case != 'B':
        carrier=np.where(g.shadow,g.config.dark_carrier,g.config.bright_carrier)
        phase=phase+2*np.pi*carrier*g.x[None,:]
    if case == 'C': phase=phase+phi*g.shadow
    command=aperture*np.exp(1j*phase)
    mask=g.iris(1.,0. if case=='B' else g.config.bright_carrier)
    modes=[model.held_coefficients(v)*mask for v in (command,1j*g.piston_shape*command)]
    if case!='B': modes=[np.roll(m,-model.carrier_index,axis=1) for m in modes]
    basis=bases['B' if case=='B' else 'SLM',mu]
    gram=basis['gram'].copy(); total=basis['flux_gram'].copy()
    if case=='D':
        gram[...,2,:]=0; gram[...,:,2]=0; total[2,:]=0; total[:,2]=0
    image,derivative=model.sensor_modes(*modes,modulation=mu,angles=8)
    for predicted,direct in zip(evaluate(gram,a,phi),(image,derivative)):
        assert_close(predicted,crop_detector(direct,32))
    expected_flux,expected_tangent=evaluate(total,a,phi)
    assert_close(expected_flux,image.sum()); assert_close(expected_tangent,derivative.sum())
    assert expected_flux <= np.sum(abs(command)**2)*(1+RTOL)
    assert image.min() > -ATOL
    # This independently commanded relay must also match the convenience path.
    for actual,expected in zip(model.relay_modes(case,a,phi),modes): assert_close(actual,expected)


def test_integration_commutes_with_angle_average(model,two_modes):
    tilts=list(model.tilts(.73,8))
    images=[abs(model.envelope(two_modes,t))**2 for t in tilts]
    assert_close(model.integrate_product(sum(images)/8),sum(model.integrate_product(i) for i in images)/8)


def test_full_flux_incoherent_and_periodicity(bases):
    for family in ('B','SLM'):
        assert_close(bases[family,0.]['flux_gram'],bases[family,.73]['flux_gram'])
    gram=bases['SLM',.73]['gram']; a,phi=.31,.47
    for first,opposite,inc,periodic in zip(evaluate(gram,a,phi),evaluate(gram,a,phi+np.pi),
             evaluate(gram,a,phi,False),evaluate(gram,a+4*np.pi,phi)):
        assert_close((first+opposite)/2,inc); assert_close(first,periodic)


def test_rejects_unsupported_models(model):
    cfg=dict(diameter_pixels=8,spider_pixels=2,samples_per_pixel=1,domain_diameters=6)
    with pytest.raises(ValueError,match='Nyquist'): ModalModel(Configuration(**cfg),iris_radius=2.)
    with pytest.raises(ValueError,match='integer'): ModalModel(Configuration(**cfg,bright_carrier=.13),iris_radius=1.)
    with pytest.raises(ValueError,match='B/D/C'): model.relay_modes('A')


def test_external_modes_cannot_bypass_pixel_integration_bandwidth(model):
    n=model.grid.n
    unsafe=np.zeros((n,n),complex)
    unsafe[n//2,n//2-16]=1.; unsafe[n//2,n//2+16]=1.
    with pytest.raises(ValueError,match='bandwidth'):
        model.sensor_modes(unsafe,np.zeros_like(unsafe))
    with pytest.raises(ValueError,match='bandwidth'):
        model.sensor_modes(np.zeros_like(unsafe),unsafe)


def test_real_integration_retains_exact_real_observable(model,two_modes):
    product=abs(model.envelope(two_modes,(.017,.023)))**2
    assert np.isrealobj(model.integrate_real_product(product))
    assert_close(model.integrate_real_product(product),model.integrate_product(product))
    with pytest.raises(ValueError,match='real'):
        model.integrate_real_product(product.astype(complex)+1j)
