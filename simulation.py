"""Scalar SLM/iris/PWFS model; lengths are in physical SLM pixel pitches.

No hardware calibration is implied. All fields, including relay diffraction
outside the nominal pupil, reach the finite periodic detector domain.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift


def ft(field):
    return fftshift(fft2(ifftshift(field), norm="ortho", workers=1))


def ift(field):
    return fftshift(ifft2(ifftshift(field), norm="ortho", workers=1))


@dataclass(frozen=True)
class Configuration:
    diameter_pixels: int = 128
    spider_pixels: int = 8
    samples_per_pixel: int = 2
    domain_diameters: int = 4
    bright_carrier: float = 0.125
    dark_carrier: float = -0.125
    pupil_center_diameters: float = 0.75

    def __post_init__(self):
        for name in ("diameter_pixels", "spider_pixels", "samples_per_pixel", "domain_diameters"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.diameter_pixels % 2 or self.spider_pixels % 2:
            raise ValueError("Use even pupil and spider widths for the centered pixel lattice")
        if self.spider_pixels >= self.diameter_pixels:
            raise ValueError("Spider must be narrower than the pupil")
        if self.domain_diameters < 4:
            raise ValueError("At least four pupil diameters are required")
        for name in ("bright_carrier", "dark_carrier", "pupil_center_diameters"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0.5 < self.pupil_center_diameters < (self.domain_diameters - 1) / 2:
            raise ValueError("Pupil centers must separate pupils and leave a detector margin")


class OpticalModel:
    def __init__(self, config):
        self.config = config
        d, s = config.diameter_pixels, config.samples_per_pixel
        self.coarse_n = d * config.domain_diameters
        self.n = self.coarse_n * s
        self.x = (np.arange(self.n) + 0.5 - self.n / 2) / s
        # The physical aperture and SLM commands stay fixed when s changes.
        self.pixel_x = np.repeat(np.arange(self.coarse_n) + 0.5 - self.coarse_n / 2, s)
        self.pupil = self.pixel_x[:, None] ** 2 + self.pixel_x[None, :] ** 2 <= (d / 2) ** 2
        self.shadow = self.pupil & (np.abs(self.pixel_x[None, :]) < config.spider_pixels / 2)
        self.bright = self.pupil & ~self.shadow
        self.piston_shape = 0.5 * np.sign(self.pixel_x)[None, :] * self.bright
        self.f = fftshift(np.fft.fftfreq(self.n, d=1 / s))
        shift = config.pupil_center_diameters * d
        self.pyramid_mask = np.exp(-2j * np.pi * shift * (np.abs(self.f[:, None]) + np.abs(self.f[None, :])))

    def power(self, field):
        return float(np.sum(np.abs(field) ** 2) / self.config.samples_per_pixel ** 2)

    def iris(self, radius_lam_over_d, carrier=0.0):
        if not np.isfinite(radius_lam_over_d) or radius_lam_over_d <= 0:
            raise ValueError("Iris radius must be positive and finite")
        radius = radius_lam_over_d / self.config.diameter_pixels
        return (self.f[None, :] - carrier) ** 2 + self.f[:, None] ** 2 <= radius ** 2

    def relay(self, case, piston=0.0, iris_radius=8.0):
        """Return the complex field and its exact derivative with respect to piston.

        A: opaque, no iris. B: opaque through centered iris. C: pixel-held
        phase-only SLM through off-axis iris, then continuous-carrier removal.
        D: bright-only pixel-held carrier through the same off-axis relay.
        """
        if not np.isfinite(piston):
            raise ValueError("Piston must be finite")
        if case not in ("A", "B", "C", "D"):
            raise ValueError("Case must be A, B, C or D")
        if case in ("A", "B"):
            field = self.bright * np.exp(1j * piston * self.piston_shape)
            derivative = 1j * self.piston_shape * field
            if case == "B":
                mask = self.iris(iris_radius)
                field, derivative = ift(ft(field) * mask), ift(ft(derivative) * mask)
            return field, derivative
        cfg = self.config
        carrier = np.where(self.shadow, cfg.dark_carrier, cfg.bright_carrier)
        command = piston * self.piston_shape + 2 * np.pi * carrier * self.pixel_x[None, :]
        # Ideal full-2pi response: wrapping exp(i*command) changes nothing.
        aperture = self.bright if case == "D" else self.pupil
        field = aperture * np.exp(1j * command)
        derivative = 1j * self.piston_shape * field
        mask = self.iris(iris_radius, cfg.bright_carrier)
        demodulate = np.exp(-2j * np.pi * cfg.bright_carrier * self.x)[None, :]
        return ift(ft(field) * mask) * demodulate, ift(ft(derivative) * mask) * demodulate

    def bin_intensity(self, values):
        """Integrate onto fixed physical-pitch detector pixels."""
        s = self.config.samples_per_pixel
        return values.reshape(self.coarse_n, s, self.coarse_n, s).sum(axis=(1, 3)) / s ** 2

    def sensor(self, field, derivative=None, modulation=0.0, angles=16):
        """Noiseless detector intensity and optional exact piston derivative."""
        if field.shape != (self.n, self.n) or (derivative is not None and derivative.shape != field.shape):
            raise ValueError("Field shape does not match the optical model")
        if not np.isfinite(modulation) or modulation < 0 or not isinstance(angles, int) or angles < 1:
            raise ValueError("Modulation must be nonnegative and angles a positive integer")
        theta = np.arange(angles) * 2 * np.pi / angles if modulation else [0.0]
        intensity = np.zeros((self.coarse_n, self.coarse_n))
        response = np.zeros_like(intensity) if derivative is not None else None
        for angle in theta:
            tilt = np.exp(2j * np.pi * modulation / self.config.diameter_pixels *
                          (self.x[None, :] * np.cos(angle) + self.x[:, None] * np.sin(angle)))
            propagated = ift(ft(field * tilt) * self.pyramid_mask)
            intensity += self.bin_intensity(np.abs(propagated) ** 2)
            if derivative is not None:
                tangent = ift(ft(derivative * tilt) * self.pyramid_mask)
                response += self.bin_intensity(2 * np.real(propagated.conj() * tangent))
        return intensity / len(theta), None if response is None else response / len(theta)


def crop_detector(values, side_pixels):
    """Centered physical detector window; call on raw intensity/tangent."""
    if (values.ndim != 2 or values.shape[0] != values.shape[1] or
            not isinstance(side_pixels, int) or isinstance(side_pixels, bool) or
            side_pixels < 1 or side_pixels > values.shape[0] or
            (values.shape[0] - side_pixels) % 2):
        raise ValueError("Detector window must be a centered integer pixel crop")
    start = (values.shape[0] - side_pixels) // 2
    return values[start:start + side_pixels, start:start + side_pixels]


def normalize_response(intensity, derivative):
    flux = intensity.sum()
    if not np.isfinite(flux) or flux <= 0:
        raise ValueError("Positive finite detector flux is required")
    return intensity / flux, derivative / flux - intensity * derivative.sum() / flux ** 2


def compare_response(reference, actual):
    denom = float(np.sum(reference * reference))
    absolute = float(np.linalg.norm(actual))
    if denom < 1e-24:
        return {"gain": None, "shape_error": None, "absolute_sensitivity": absolute,
                "reference_sensitivity": float(np.sqrt(denom)), "reference_near_zero": True}
    gain = float(np.sum(reference * actual) / denom)
    return {"gain": gain, "shape_error": float(np.linalg.norm(actual - gain * reference) / np.sqrt(denom)),
            "absolute_sensitivity": absolute, "reference_sensitivity": float(np.sqrt(denom)),
            "reference_near_zero": False}
