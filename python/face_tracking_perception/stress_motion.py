# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Repeatable wide-area movement of the rendered test mannequin, without ROS."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StressSample:
    """Model pose in world metres/radians and the current appearance state."""

    pose: tuple[float, float, float, float]
    visible: bool
    phase: str
    complete: bool


class StressMotion:
    """
    A 134-second route with smooth starts, side arcs and a brief disappearance.

    The standing mesh faces local -Y. Its face centre is approximately 1.75 m
    above the model origin; moving the rigid mesh vertically changes face height
    without claiming an articulated crouch. Every visible segment joins with
    zero velocity and acceleration. The last pose equals the first one.
    """

    duration_sec = 134.0
    face_reference_height_m = 1.75
    initial_pose = (2.1, 0.0, 0.0, -math.pi / 2)

    # Time, model radius, bearing in degrees, approximate face height, phase, visibility.
    _keyframes = (
        (0.0, 2.1, 0.0, 1.75, 'START_HOLD', True),
        (20.0, 2.1, 0.0, 1.75, 'LOWER_APPROACH', True),
        (32.0, 1.4, -25.0, 1.25, 'LEFT_ARC', True),
        (46.0, 2.2, -100.0, 1.85, 'CROSS_FRONT', True),
        (66.0, 2.6, 110.0, 1.50, 'REVERSE', True),
        (82.0, 1.8, 25.0, 1.75, 'SETTLE_HOLD', True),
        (92.0, 1.8, 25.0, 1.75, 'HIDDEN', False),
        (100.0, 2.2, -70.0, 1.55, 'REAPPEAR_HOLD', True),
        (108.0, 2.2, -70.0, 1.55, 'RETURN_FRONT', True),
        (122.0, 2.1, 0.0, 1.75, 'FINAL_HOLD', True),
        (134.0, 2.1, 0.0, 1.75, 'COMPLETE', True),
    )

    def sample(self, elapsed, cycles=0):
        """
        Sample elapsed simulation seconds; zero cycles repeats indefinitely.

        A completed finite run keeps the final visible pose. Negative time is
        clamped to the initial pose, allowing a caller to reset its clock epoch.
        """
        if not math.isfinite(elapsed):
            raise ValueError('Stress scenario elapsed time must be finite')
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
            raise ValueError('Stress scenario cycles must be a nonnegative integer')
        elapsed = max(0.0, elapsed)
        if cycles and elapsed >= cycles * self.duration_sec:
            return StressSample(self.initial_pose, True, 'COMPLETE', True)
        local_time = elapsed % self.duration_sec
        for start, end in zip(self._keyframes, self._keyframes[1:]):
            if local_time < end[0]:
                fraction = (local_time - start[0]) / (end[0] - start[0])
                blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
                radius, bearing, height = (
                    a + (b - a) * blend for a, b in zip(start[1:4], end[1:4]))
                bearing = math.radians(bearing)
                pose = (radius * math.cos(bearing), radius * math.sin(bearing),
                        height - self.face_reference_height_m, bearing - math.pi / 2)
                return StressSample(pose, start[5], start[4], False)
        raise RuntimeError('Stress scenario timeline does not cover the sample time')
