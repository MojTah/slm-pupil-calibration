"""One same-period strip-command comparison; all other operators are inherited."""
import numpy as np

from modal_simulation import ModalModel


class OrthogonalStripModel(ModalModel):
    def relay_modes(self, case, piston=0., shadow_phase=0.):
        if case != 'C':
            return super().relay_modes(case, piston, shadow_phase)
        if not np.isfinite(piston) or not np.isfinite(shadow_phase):
            raise ValueError('Require finite piston and shadow phase')
        g = self.grid
        # Only the strip grating rotates; bright order, iris and demodulation stay x.
        phase = piston*g.piston_shape + 2*np.pi*g.config.bright_carrier*np.where(
            g.shadow, g.pixel_x[:, None], g.pixel_x[None, :]) + shadow_phase*g.shadow
        command = g.pupil*np.exp(1j*phase)
        mask = g.iris(self.iris_radius, g.config.bright_carrier)
        return tuple(np.roll(self.held_coefficients(v)*mask, -self.carrier_index, axis=1)
                     for v in (command, 1j*g.piston_shape*command))
