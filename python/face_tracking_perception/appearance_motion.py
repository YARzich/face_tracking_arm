# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Small, smooth floor-level movements of a visible person around its spawn pose."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AppearanceMotion:
    """Return a bounded XY/yaw pose using seconds since the acknowledged appearance."""

    lateral_m: float = 0.14
    depth_m: float = 0.07
    yaw_rad: float = 0.07
    period_sec: float = 6.0

    def __post_init__(self):
        values = (self.lateral_m, self.depth_m, self.yaw_rad, self.period_sec)
        if (not all(math.isfinite(v) for v in values) or min(values[:3]) < 0
                or self.period_sec <= 0):
            raise ValueError('Motion amplitudes must be nonnegative and period positive')

    def pose(self, center, elapsed):
        x, y = center
        bearing = math.atan2(y, x)
        elapsed = max(0.0, elapsed)
        ramp = min(elapsed, 1.0)
        # Integrate a smooth speed ramp: starting motion never overshoots cruise speed.
        motion_time = max(0.0, elapsed - 1.0) + ramp**4 * (2.5 - 3 * ramp + ramp**2)
        phase = 2 * math.pi * motion_time / self.period_sec
        lateral = self.lateral_m * math.sin(phase)
        depth = self.depth_m * math.sin(phase / 1.4)
        x += depth * math.cos(bearing) - lateral * math.sin(bearing)
        y += depth * math.sin(bearing) + lateral * math.cos(bearing)
        yaw = math.atan2(y, x) - math.pi / 2
        yaw += self.yaw_rad * math.sin(phase / 1.6)
        return float(x), float(y), yaw


def companion_center(center, elapsed):
    """Move a second person from farther away to clearly closer than the first."""
    x, y = center
    progress = min(1.0, max(0.0, (elapsed - 3.0) / 5.0))
    progress = progress**3 * (10 - 15 * progress + 6 * progress**2)
    return x + .8 - 1.5 * progress, y + (-.55 if y > 0 else .55)
