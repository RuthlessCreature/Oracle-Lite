"""Unit-checked optical arithmetic; calculations do not certify accuracy."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class OpticalInputs:
    horizontal_fov_mm: float
    horizontal_pixels: int
    pixel_pitch_um: float

    def calculate(self) -> dict:
        for value in (self.horizontal_fov_mm, self.pixel_pitch_um):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('FOV and pixel pitch require positive finite values in mm and um')
        if type(self.horizontal_pixels) is not int or self.horizontal_pixels <= 0:
            raise ValueError('Horizontal pixels must be a positive integer on the FOV axis')
        sensor_width_mm = self.horizontal_pixels * self.pixel_pitch_um / 1000
        return {
            'horizontal_fov_mm': self.horizontal_fov_mm,
            'horizontal_pixels': self.horizontal_pixels,
            'pixel_pitch_um': self.pixel_pitch_um,
            'sampling_um_per_px': self.horizontal_fov_mm * 1000 / self.horizontal_pixels,
            'sensor_width_mm': sensor_width_mm,
            'magnification': sensor_width_mm / self.horizontal_fov_mm,
            'accuracy_verified': False,
        }
