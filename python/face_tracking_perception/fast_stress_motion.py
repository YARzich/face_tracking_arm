# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Fast rendered-person motion and a repeatable nearest-face selection challenge."""

from dataclasses import dataclass
import math


_FACE_REFERENCE_HEIGHT_M = 1.75


def _model_pose(x, y, height):
    """Place the rigid standing mesh facing the base, with approximate face height."""
    return (x, y, height - _FACE_REFERENCE_HEIGHT_M, math.atan2(y, x) - math.pi / 2)


@dataclass(frozen=True)
class FastStressSample:
    """Three visible model poses in world metres/radians and their schedule state."""

    poses: tuple[tuple[float, float, float, float], ...]
    phase: str
    complete: bool


class FastStressMotion:
    """
    Quick direction changes followed by approach, retreat and selection pauses.

    The first person moves; the other two remain visible and stationary. At the
    far pause, the first background person is nearest to the base. Each segment
    starts and ends with zero velocity and acceleration, including cycle joins.
    """

    duration_sec = 96.0
    face_reference_height_m = _FACE_REFERENCE_HEIGHT_M
    initial_poses = (
        _model_pose(2.1, 0.0, 1.75),
        _model_pose(2.8, 1.0, 1.65),
        _model_pose(3.2, -1.1, 1.80),
    )
    _far_x = 3.6 * math.cos(math.radians(32))
    _far_y = 3.6 * math.sin(math.radians(32))

    # Time, model X/Y, approximate face height, phase during the following segment.
    _keyframes = (
        (0.0, 2.1, 0.0, 1.75, 'START_HOLD'),
        (10.0, 2.1, 0.0, 1.75, 'QUICK_LEFT'),
        (11.0, 2.1, .4, 1.75, 'QUICK_RIGHT'),
        (12.2, 2.1, -.4, 1.75, 'QUICK_LEFT'),
        (13.4, 2.1, .4, 1.75, 'QUICK_CENTER'),
        (14.4, 2.1, 0.0, 1.75, 'QUICK_DOWN'),
        (15.4, 2.1, 0.0, 1.60, 'QUICK_UP'),
        (16.6, 2.1, 0.0, 1.90, 'QUICK_CENTER_HEIGHT'),
        (17.6, 2.1, 0.0, 1.75, 'QUICK_SETTLE'),
        (20.0, 2.1, 0.0, 1.75, 'APPROACH'),
        (26.0, 1.4, 0.0, 1.75, 'NEAR_HOLD'),
        (34.0, 1.4, 0.0, 1.75, 'RETREAT'),
        (44.0, _far_x, _far_y, 1.75, 'FAR_HOLD'),
        (56.0, _far_x, _far_y, 1.75, 'RETURN_NEAR'),
        (66.0, 1.4, 0.0, 1.75, 'NEAR_RETURN_HOLD'),
        (76.0, 1.4, 0.0, 1.75, 'RETURN_START'),
        (82.0, 2.1, 0.0, 1.75, 'FINAL_HOLD'),
        (96.0, 2.1, 0.0, 1.75, 'COMPLETE'),
    )

    def sample(self, elapsed, cycles=0):
        """Sample simulation seconds; finite runs keep all three people at the end."""
        if not math.isfinite(elapsed):
            raise ValueError('Fast stress scenario elapsed time must be finite')
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles < 0:
            raise ValueError('Fast stress scenario cycles must be a nonnegative integer')
        elapsed = max(0.0, elapsed)
        if cycles and elapsed >= cycles * self.duration_sec:
            return FastStressSample(self.initial_poses, 'COMPLETE', True)
        local_time = elapsed % self.duration_sec
        for start, end in zip(self._keyframes, self._keyframes[1:]):
            if local_time < end[0]:
                fraction = (local_time - start[0]) / (end[0] - start[0])
                blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
                x, y, height = (a + (b - a) * blend
                                for a, b in zip(start[1:4], end[1:4]))
                poses = (_model_pose(x, y, height), *self.initial_poses[1:])
                return FastStressSample(poses, start[4], False)
        raise RuntimeError('Fast stress scenario timeline does not cover the sample time')
