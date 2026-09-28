# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Constant-velocity Kalman filtering of a face point in a fixed frame."""

import math

import cv2
import numpy as np


class KalmanPointFilter:
    """
    Estimate position and velocity from timestamped 3D measurements in metres.

    OpenCV performs prediction and correction at the measurement timestamp.
    Measurement noise is isotropic. Independent acceleration in each frame
    supplies process noise; no extrapolation to the current wall clock is used.
    """

    def __init__(self, measurement_std_m=0.02, acceleration_std_mps2=1.5,
                 reset_after_sec=0.25):
        for name, value in (
                ('measurement_std_m', measurement_std_m),
                ('acceleration_std_mps2', acceleration_std_mps2),
                ('reset_after_sec', reset_after_sec)):
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'{name} must be finite and positive') from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            setattr(self, name, value)
        self.reset()

    def reset(self):
        """Forget the previous person and all filter history."""
        self._filter = None
        self._stamp_sec = None
        self._track_id = None

    def _initialize(self, point):
        kalman = cv2.KalmanFilter(6, 3, 0, cv2.CV_64F)
        kalman.measurementMatrix = np.eye(3, 6, dtype=np.float64)
        kalman.measurementNoiseCov = self.measurement_std_m ** 2 * np.eye(3)
        # Initial position comes from one observation. A 1 m/s velocity
        # uncertainty lets the following frames identify motion from rest.
        kalman.errorCovPost = np.diag([self.measurement_std_m ** 2] * 3 + [1.0] * 3)
        kalman.statePost = np.array([*point, 0.0, 0.0, 0.0], dtype=np.float64).reshape(6, 1)
        self._filter = kalman

    def update(self, point, stamp_sec, track_id):
        """Return a filtered 3-tuple; invalid measurements leave history intact."""
        try:
            point = tuple(float(value) for value in point)
            stamp_sec = float(stamp_sec)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('point and timestamp must contain finite numbers') from error
        if (len(point) != 3 or not all(math.isfinite(value) for value in point)
                or not math.isfinite(stamp_sec)):
            raise ValueError('point must have three finite coordinates and a finite timestamp')

        dt = None if self._stamp_sec is None else stamp_sec - self._stamp_sec
        if (dt is None or track_id != self._track_id or dt <= 0
                or dt > self.reset_after_sec):
            self._initialize(point)
            result = point
        else:
            transition = np.eye(6)
            transition[:3, 3:] = dt * np.eye(3)
            acceleration = np.vstack((0.5 * dt ** 2 * np.eye(3), dt * np.eye(3)))
            self._filter.transitionMatrix = transition
            self._filter.processNoiseCov = (
                self.acceleration_std_mps2 ** 2 * acceleration @ acceleration.T)
            self._filter.predict()
            corrected = self._filter.correct(np.array(point, dtype=np.float64).reshape(3, 1))
            result = tuple(float(value) for value in corrected[:3, 0])

        self._stamp_sec = stamp_sec
        self._track_id = track_id
        return result
