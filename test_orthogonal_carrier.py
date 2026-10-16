"""Independent command fixtures and exact inherited-control checks."""
import sys
sys.dont_write_bytecode = True
import numpy as np
import pytest
from modal_simulation import ModalModel, build_modal_basis
from orthogonal_carrier import OrthogonalStripModel
from piston_basis import evaluate
from simulation import Configuration, crop_detector


@pytest.fixture(scope='module')
def model():
    return OrthogonalStripModel(Configuration(diameter_pixels=8, spider_pixels=2,
        samples_per_pixel=1, domain_diameters=6), iris_radius=1.)


@pytest.mark.parametrize('case', ['B', 'D'])
def test_controls_unchanged(model, case):
    original = ModalModel(model.grid.config, model.iris_radius)
    for a, b in zip(model.relay_modes(case, .29, .63), original.relay_modes(case, .29, .63)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('a,phi', [(0., 0.), (-.37, np.pi/2), (.29, .63)])
def test_independently_commanded_y_strip(model, a, phi):
    g = model.grid
    # Explicit sum of disjoint bright and shadow commands, not the adapter's branch.
    bright = g.bright*np.exp(1j*a*g.piston_shape + 2j*np.pi*.125*g.x[None, :])
    strip = g.shadow*np.exp(1j*phi + 2j*np.pi*.125*g.x[:, None])
    command = bright + strip
    tangent = 1j*g.piston_shape*bright
    mask = g.iris(1., .125)
    independently_commanded = tuple(np.roll(model.held_coefficients(v)*mask,
        -model.carrier_index, axis=1) for v in (command, tangent))
    for actual, expected in zip(model.relay_modes('C', a, phi), independently_commanded):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    # Direct continuous Fourier sums check the command, physical centres and hold.
    for mx, my in [(6, 0), (5, 1), (7, -1), (6, 2)]:
        phase = np.exp(-2j*np.pi*(mx*g.x[None, :] + my*g.x[:, None])/g.n)
        for value in (command, tangent):
            direct = np.sum(value*phase)*np.sinc(mx/g.n)*np.sinc(my/g.n)/g.n**2
            actual = model.held_coefficients(value)[g.n//2+my, g.n//2+mx]
            np.testing.assert_allclose(actual, direct, rtol=2e-12, atol=2e-13)
    basis = build_modal_basis(model, 'SLM', modulation=.73, angles=8)
    image, derivative = model.sensor_modes(*independently_commanded, modulation=.73, angles=8)
    for predicted, actual in zip(evaluate(basis['gram'], a, phi), (image, derivative)):
        np.testing.assert_allclose(predicted, crop_detector(actual, 32), rtol=2e-12, atol=2e-13)


def test_rejects_nonfinite(model):
    with pytest.raises(ValueError, match='finite'):
        model.relay_modes('C', np.nan, 0.)
    with pytest.raises(ValueError, match='finite'):
        model.relay_modes('C', 0., np.inf)
