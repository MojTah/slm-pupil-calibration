"""Exact held-pixel, finite-modal B/D/C model with quasiperiodic modulation.

This is distinct from periodizing a sampled tilted relay field. The envelope
is band-limited; its intensity products are integrated over physical pixels.
"""
import numpy as np

from simulation import OpticalModel, crop_detector, ft, ift
from piston_basis import gram_diagnostics


class ModalModel:
    def __init__(self, config, iris_radius=8.):
        if config.samples_per_pixel != 1:
            raise ValueError('Modal model requires one sample per physical pixel')
        if not np.isfinite(iris_radius) or not 0 < iris_radius/config.diameter_pixels < .25:
            raise ValueError('Require positive iris with product bandwidth below Nyquist')
        if abs(config.bright_carrier)+iris_radius/config.diameter_pixels >= .5:
            raise ValueError('Selected optical modes must lie below source Nyquist')
        carrier_index = config.bright_carrier*config.diameter_pixels*config.domain_diameters
        if not np.isclose(carrier_index, round(carrier_index), atol=1e-12, rtol=0):
            raise ValueError('Carrier must have an integer domain mode index')
        self.grid = OpticalModel(config)
        self.iris_radius = iris_radius
        self.carrier_index = round(carrier_index)
        f = self.grid.f
        self.origin = np.exp(1j*np.pi*(f[:, None]+f[None, :]))
        self.box = np.sinc(f[:, None])*np.sinc(f[None, :])
        # Preserve the exact masks used for B and carrier-selected D/C; their
        # union also avoids classifying a boundary mode differently by roundoff.
        self.envelope_support = self.grid.iris(iris_radius) | np.roll(
            self.grid.iris(iris_radius, config.bright_carrier), -self.carrier_index, axis=1)

    def held_coefficients(self, command):
        """Continuous Fourier-series coefficients of unit-width held pixels."""
        if command.shape != self.grid.pupil.shape or not np.isfinite(command).all():
            raise ValueError('Require finite command on the physical pixel grid')
        return ft(command)*self.origin.conj()*self.box/self.grid.n

    def samples(self, modes):
        """Evaluate Fourier-series envelope at the physical pixel centers."""
        return ift(modes*self.origin)*self.grid.n

    def relay_modes(self, case, piston=0., shadow_phase=0.):
        if case not in ('B', 'D', 'C'):
            raise ValueError('Only iris-filtered B/D/C controls are supported')
        if not np.isfinite(piston) or not np.isfinite(shadow_phase):
            raise ValueError('Require finite piston and shadow phase')
        g = self.grid
        phase = piston*g.piston_shape
        aperture = g.pupil if case == 'C' else g.bright
        if case != 'B':
            carrier = np.where(g.shadow, g.config.dark_carrier, g.config.bright_carrier)
            phase = phase+2*np.pi*carrier*g.pixel_x[None, :]
        if case == 'C':
            phase = phase+shadow_phase*g.shadow
        command = aperture*np.exp(1j*phase)
        tangent = 1j*g.piston_shape*command
        mask = g.iris(self.iris_radius, 0. if case == 'B' else g.config.bright_carrier)
        modes = [self.held_coefficients(v)*mask for v in (command, tangent)]
        if case != 'B':
            modes = [np.roll(v, -self.carrier_index, axis=1) for v in modes]
        return tuple(modes)

    def envelope(self, modes, tilt=(0., 0.)):
        """De-tilted output: H(f+t); a common output tilt cancels products."""
        if len(tilt) != 2 or not np.isfinite(tilt).all():
            raise ValueError('Require finite x/y tilt frequencies')
        g = self.grid
        shift = g.config.pupil_center_diameters*g.config.diameter_pixels
        mask = np.exp(-2j*np.pi*shift*(abs(g.f[None, :]+tilt[0])+abs(g.f[:, None]+tilt[1])))
        return self.samples(modes*mask)

    def integrate_product(self, product):
        """Exact pixel-box integral of a Nyquist-safe envelope product."""
        return ift(ft(product)*self.box)

    def integrate_real_product(self, product):
        """Real observables stay real; check discarded FFT roundoff globally."""
        if not np.isrealobj(product):
            raise ValueError('Expected a real intensity or tangent product')
        value = self.integrate_product(product)
        scale = float(np.max(abs(product)))
        if np.max(abs(value.imag)) > 1e-12*max(scale, np.finfo(float).tiny):
            raise ValueError('Material imaginary error in a real box integral')
        return value.real

    def sensor_modes(self, modes, derivative, modulation=0., angles=128):
        for value in (modes, derivative):
            if value.shape != self.grid.pupil.shape or not np.isfinite(value).all():
                raise ValueError('Require finite envelope coefficients on the modal grid')
            if np.any(value[~self.envelope_support] != 0):
                raise ValueError('Envelope modes exceed the retained iris bandwidth')
        image = np.zeros_like(self.grid.pupil, dtype=float)
        tangent = np.zeros_like(image)
        for tilt in self.tilts(modulation, angles):
            field = self.envelope(modes, tilt)
            df = self.envelope(derivative, tilt)
            image += abs(field)**2
            tangent += 2*(field.conj()*df).real
        count = angles if modulation else 1
        return (self.integrate_real_product(image/count),
                self.integrate_real_product(tangent/count))

    def tilts(self, modulation, angles):
        if not np.isfinite(modulation) or modulation < 0 or not isinstance(angles, int) or isinstance(angles, bool) or angles < 1:
            raise ValueError('Require nonnegative modulation and positive integer angles')
        for theta in np.arange(angles if modulation else 1)*2*np.pi/angles:
            radius = modulation/self.grid.config.diameter_pixels
            yield (radius*np.cos(theta), radius*np.sin(theta))


def modal_gram(model, fields, side):
    result = np.zeros((side, side, 3, 3), dtype=complex)
    for i, left in enumerate(fields):
        for j in range(i, len(fields)):
            product = (model.integrate_real_product(abs(left)**2) if i == j else
                       model.integrate_product(left.conj()*fields[j]))
            value = crop_detector(product, side)
            result[..., i, j] = value
            result[..., j, i] = value.conj()
    return result


def build_modal_basis(model, family, modulation=0., angles=128, detector_diameters=4):
    if family not in ('B', 'SLM'):
        raise ValueError('Require B or SLM basis family')
    g = model.grid
    side = g.config.diameter_pixels*detector_diameters
    if not isinstance(side, int) or not 1 <= side <= g.n:
        raise ValueError('Detector must lie inside the modal domain')
    bright, tangent = model.relay_modes('B' if family == 'B' else 'D')
    modes = [bright, tangent]
    if family == 'SLM':
        whole, unused = model.relay_modes('C')
        modes.append(whole-bright)
        del whole, unused
    pupil_gram = modal_gram(model, [model.samples(m) for m in modes], g.config.diameter_pixels)
    total = np.zeros((3, 3), dtype=complex)
    for i, left in enumerate(modes):
        for j in range(i, len(modes)):
            total[i, j] = np.vdot(left, modes[j])*g.n**2
            total[j, i] = total[i, j].conj()
    # Integration commutes with incoherent angle averaging. Retain products,
    # never a coherent average of fields from different angles.
    raw = np.zeros((g.n, g.n, 3, 3), dtype=complex)
    count = angles if modulation else 1
    for tilt in model.tilts(modulation, angles):
        fields = [model.envelope(m, tilt) for m in modes]
        for i, left in enumerate(fields):
            for j in range(i, len(fields)):
                raw[..., i, j] += left.conj()*fields[j]/count
    gram = np.zeros((side, side, 3, 3), dtype=complex)
    for i in range(len(modes)):
        for j in range(i, len(modes)):
            product = (model.integrate_real_product(raw[..., i, j].real) if i == j else
                       model.integrate_product(raw[..., i, j]))
            value = crop_detector(product, side)
            gram[..., i, j] = value
            gram[..., j, i] = value.conj()
    diagnostics = {name: gram_diagnostics(value) for name, value in
                   [('detector', gram), ('pupil_square', pupil_gram), ('total_flux', total)]}
    diagnostics['exterior_flux'] = gram_diagnostics(total-gram.sum(axis=(0, 1)), parent_trace=float(np.trace(total).real))
    return {'gram': gram, 'pupil_gram': pupil_gram, 'flux_gram': total, 'diagnostics': diagnostics}
