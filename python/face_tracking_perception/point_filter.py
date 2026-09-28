# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""
Adaptive One Euro filtering for a selected face in a fixed coordinate frame.

Algorithm: Casiez, Roussel and Vogel, https://gery.casiez.net/1euro/.
This implementation uses one cutoff for all three axes, based on the norm of
the filtered raw derivative, so rotating the fixed frame does not change it.
"""

import math


class PointFilter:
    """
    Filter points in metres using acquisition timestamps in seconds.

    Cutoffs are in Hz and beta is in inverse metres. Increasing beta reduces
    smoothing during movement. A changed track, missing frames or a restarted
    clock begins a new sequence without blending unrelated observations.
    """

    def __init__(self, min_cutoff_hz=1.5, beta=32.0, derivative_cutoff_hz=1.0,
                 reset_after_sec=0.25):
        for name, value in (
                ('min_cutoff_hz', min_cutoff_hz), ('beta', beta),
                ('derivative_cutoff_hz', derivative_cutoff_hz),
                ('reset_after_sec', reset_after_sec)):
            if not math.isfinite(value) or value < 0 or (name != 'beta' and value == 0):
                requirement = 'nonnegative' if name == 'beta' else 'positive'
                raise ValueError(f'{name} must be finite and {requirement}')
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.reset_after_sec = float(reset_after_sec)
        self.reset()

    def reset(self):
        """Forget the previous person and all filter history."""
        self._stamp_sec = None
        self._track_id = None
        self._raw_point = None
        self._filtered_point = None
        self._derivative = (0.0, 0.0, 0.0)

    @staticmethod
    def _alpha(cutoff_hz, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff_hz * dt))

    @staticmethod
    def _blend(previous, current, alpha):
        return tuple(old + alpha * (new - old) for old, new in zip(previous, current))

    def update(self, point, stamp_sec, track_id):
        """Return a filtered 3-tuple; invalid measurements leave history intact."""
        try:
            point = tuple(float(value) for value in point)
            stamp_sec = float(stamp_sec)
        except (TypeError, ValueError) as error:
            raise ValueError('point and timestamp must contain finite numbers') from error
        if (len(point) != 3 or not all(math.isfinite(value) for value in point)
                or not math.isfinite(stamp_sec)):
            raise ValueError('point must have three finite coordinates and a finite timestamp')

        dt = None if self._stamp_sec is None else stamp_sec - self._stamp_sec
        if (dt is None or track_id != self._track_id or dt <= 0
                or dt > self.reset_after_sec):
            self._filtered_point = point
            self._derivative = (0.0, 0.0, 0.0)
        else:
            raw_derivative = tuple((new - old) / dt
                                   for old, new in zip(self._raw_point, point))
            self._derivative = self._blend(
                self._derivative, raw_derivative, self._alpha(self.derivative_cutoff_hz, dt))
            cutoff = self.min_cutoff_hz + self.beta * math.hypot(*self._derivative)
            self._filtered_point = self._blend(
                self._filtered_point, point, self._alpha(cutoff, dt))

        self._raw_point = point
        self._stamp_sec = stamp_sec
        self._track_id = track_id
        return self._filtered_point
