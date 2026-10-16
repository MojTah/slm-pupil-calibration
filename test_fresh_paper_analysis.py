"""Independent pixel-level/finite-difference checks for fresh contractions."""
import numpy as np
import pytest
from fresh_paper_analysis import gain_summary, finite_rows, best_finite, sensitivity
from piston_basis import readout_matrices, readout_moments


@pytest.fixture
def matrices():
    # A physical intensity basis with a nonzero, positive fixed readout gain.
    fields = np.array([[1., .1, .02j], [1., -.1, .03]], complex).reshape(1, 2, 3)
    gram = fields[..., :, None].conj()*fields[..., None, :]
    return readout_matrices(gram, np.array([[1., -1.]]))


def test_gain_summary_agrees_with_dense_phase(matrices):
    target = matrices.copy()
    target[..., 2, :] = 0
    target[..., :, 2] = 0
    summary, extrema = gain_summary(matrices, target)
    phases = np.linspace(0, 2*np.pi, 10001)
    gains = [readout_moments(matrices, 0., float(p))['local_gain'] for p in phases]
    assert min(gains) == pytest.approx(extrema['minimum'], abs=1e-9)
    assert max(gains) == pytest.approx(extrema['maximum'], abs=1e-9)
    assert summary['coherent_min_percent'] <= summary['coherent_max_percent']


def test_equal_poke_and_reporting_cutoff(matrices):
    target = matrices.copy()
    target[..., 2, :] = 0
    target[..., :, 2] = 0
    rows = finite_rows(matrices, target)
    assert len(rows) == 252
    assert sum(row['retained_local_poke'] for row in rows) == 168
    selected = next(r for r in rows if r['kappa'] == 0 and r['half_poke_rad'] == .2 and r['calibration_photons'] == 10**12)
    h = selected['half_poke_rad']
    fields = np.array([[1., .1, .02j], [1., -.1, .03]], complex)
    def pixel_moments(piston, strip):
        image = abs(np.cos(piston/2)*fields[:, 0]+2*np.sin(piston/2)*fields[:, 1])**2
        if strip:
            image += abs(fields[:, 2])**2
        probability = image/image.sum()
        mean = probability[0]-probability[1]
        return mean, 1-mean*mean
    plus, minus = pixel_moments(h, False), pixel_moments(-h, False)
    direct = (plus[0]-minus[0])/(2*h)
    assert selected['target_equal_poke_gain'] == pytest.approx(direct, abs=1e-15)
    assert selected['equal_scale_error'] == pytest.approx(direct/selected['finite_gain']-1)
    plus, minus = pixel_moments(h, True), pixel_moments(-h, True)
    assert selected['finite_gain'] == pytest.approx((plus[0]-minus[0])/(2*h), abs=1e-15)
    assert selected['gain_variance_coefficient'] == pytest.approx((plus[1]+minus[1])/(2*h*h), abs=1e-12)
    assert len(best_finite(rows)) == 12


def test_identical_sensitivity_is_zero(matrices):
    metrics, rows = sensitivity(matrices, matrices)
    assert len(rows) == 1206
    assert all(value == 0 for value in metrics.values())


def test_sensitivity_uses_published_lower_over_higher_variance(matrices):
    higher = matrices.copy()
    higher[1] *= 1.1
    higher[2] *= 1.21
    metrics, rows = sensitivity(matrices, higher)
    assert metrics['relative_variance'] == pytest.approx(abs(1/1.21-1), abs=1e-12)
    phase_rows = [row for row in rows if row['condition'] in ('Cpi/2', 'Cpi', 'C3pi/2')]
    assert metrics['phase_contrast'] == max(abs(row['lower_minus_higher_rad']) for row in phase_rows)
