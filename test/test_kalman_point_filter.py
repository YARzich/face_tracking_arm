# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Kalman point stability, motion response, timestamps and track isolation."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_perception.kalman_point_filter import KalmanPointFilter  # noqa: E402, I100


def test_constant_point_and_first_sample_are_unchanged():
    point_filter = KalmanPointFilter()
    point = (1.25, -.3, 1.6)
    for index in range(100):
        assert point_filter.update(point, index / 30, 1) == point


def test_reduces_stationary_measurement_noise():
    point_filter = KalmanPointFilter()
    center = np.array([1.5, -.2, 1.7])
    noise = np.random.default_rng(42).normal(0.0, .02, size=(900, 3))
    filtered = np.array([
        point_filter.update(center + error, index / 30, 1)
        for index, error in enumerate(noise)])
    assert np.linalg.norm(filtered[90:] - center) < .55 * np.linalg.norm(noise[90:])
    assert np.linalg.norm(np.mean(filtered[90:] - center, axis=0)) < .002


def test_tracks_constant_velocity_without_persistent_lag():
    point_filter = KalmanPointFilter()
    origin = np.array([.8, -.2, 1.7])
    velocity = np.array([.6, -.2, .1])
    errors = []
    for stamp in np.arange(0, 4, 1 / 30):
        point = origin + velocity * stamp
        filtered = point_filter.update(point, stamp, 1)
        if stamp >= 1:
            errors.append(np.linalg.norm(filtered - point))
    assert max(errors) < .001


@pytest.mark.parametrize('after_velocity', [0.0, -.5])
def test_abrupt_stop_and_reversal_have_bounded_transient(after_velocity):
    point_filter = KalmanPointFilter()
    errors = []
    settled_errors = []
    for index in range(121):
        stamp = index / 30
        x = .5 * min(stamp, 2) + after_velocity * max(0, stamp - 2)
        filtered = point_filter.update((x, 0, 1.7), stamp, 1)
        if stamp >= 2:
            errors.append(abs(filtered[0] - x))
        if stamp >= 2.5:
            settled_errors.append(abs(filtered[0] - x))
    assert max(errors) < .065
    assert max(settled_errors) < .01


def test_uses_actual_elapsed_time_with_irregular_camera_cadence():
    point_filter = KalmanPointFilter()
    shifted = KalmanPointFilter()
    stamps = np.cumsum(np.tile([.012, .048, .025, .035, .09], 40))
    origin, velocity = np.array([1., -.2, 1.7]), np.array([.5, -.3, .2])
    for stamp in stamps:
        point = origin + stamp * velocity
        filtered = point_filter.update(point, stamp, 1)
        shifted_result = shifted.update(point, stamp + 1_000_000, 1)
        np.testing.assert_allclose(filtered, shifted_result, atol=1e-9, rtol=0)
        if stamp > 1:
            assert np.linalg.norm(filtered - point) < .001


def test_rotating_and_translating_the_fixed_frame_preserves_filter_result():
    rotation = np.array([[.36, -.48, .8], [.8, .6, 0], [-.48, .64, .6]])
    translation = np.array([.7, -1.2, .1])
    original, transformed = KalmanPointFilter(), KalmanPointFilter()
    for index in range(100):
        stamp = index / 30
        point = np.array([math.sin(stamp), .2 * math.cos(3 * stamp), 1.7])
        result = original.update(point, stamp, 1)
        result_transformed = transformed.update(rotation @ point + translation, stamp, 1)
        np.testing.assert_allclose(result_transformed, rotation @ result + translation,
                                   atol=1e-12, rtol=0)


@pytest.mark.parametrize('stamp,track_id', [(1.1, 2), (1.251, 1), (.9, 1), (1, 1)])
def test_new_track_gap_and_nonincreasing_time_start_with_current_point(stamp, track_id):
    point_filter = KalmanPointFilter()
    point_filter.update((1, 0, 1), .96, 1)
    point_filter.update((1.1, .1, 1.1), 1, 1)
    new_point = (-2, 1, .5)
    assert point_filter.update(new_point, stamp, track_id) == new_point
    assert point_filter.update(new_point, stamp + .03, track_id) == new_point


def test_explicit_reset_discards_history():
    point_filter = KalmanPointFilter()
    point_filter.update((1, 2, 3), 1, 1)
    point_filter.update((1.1, 2.1, 3.1), 1.03, 1)
    point_filter.reset()
    assert point_filter.update((0, 0, 1), 1.06, 1) == (0, 0, 1)


@pytest.mark.parametrize('parameter', [
    'measurement_std_m', 'acceleration_std_mps2', 'reset_after_sec'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -.1, 0, None, 'invalid'])
def test_rejects_invalid_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        KalmanPointFilter(**{parameter: value})


@pytest.mark.parametrize('point,stamp', [
    ((math.nan, 0, 1), 1.01), ((0, math.inf, 1), 1.01),
    ((0, 0, -math.inf), 1.01), ((0, 1), 1.01), ((0, 1, 2, 3), 1.01),
    ((0, 0, 1), math.nan), ((0, 0, 1), math.inf),
    ((0, 0, 1), -math.inf), (None, 1.01), ((0, 'invalid', 1), 1.01),
])
def test_invalid_measurement_does_not_change_history(point, stamp):
    point_filter, reference = KalmanPointFilter(), KalmanPointFilter()
    for instance in (point_filter, reference):
        instance.update((1, 0, 1), 1, 1)
    with pytest.raises(ValueError):
        point_filter.update(point, stamp, 2)
    assert point_filter.update((1.1, .1, 1.1), 1.03, 1) == reference.update(
        (1.1, .1, 1.1), 1.03, 1)
