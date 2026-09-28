# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Point noise suppression, motion response and independent tracking sequences."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_perception.point_filter import PointFilter  # noqa: E402, I100


def test_constant_point_and_first_sample_are_unchanged():
    point_filter = PointFilter()
    point = (1.25, -.3, 1.6)
    for index in range(100):
        assert point_filter.update(point, index / 30, 1) == point


def test_reduces_stationary_measurement_noise():
    point_filter = PointFilter()
    center = np.array([1.5, -.2, 1.7])
    raw_errors, filtered_errors = [], []
    for index in range(300):
        stamp = index / 30
        noise = .005 * np.array([
            math.sin(2 * math.pi * hz * stamp) for hz in (7, 8, 9)])
        point = center + noise
        filtered = point_filter.update(point, stamp, 1)
        if index >= 30:
            raw_errors.append(noise)
            filtered_errors.append(np.array(filtered) - center)
    assert np.linalg.norm(filtered_errors) < .35 * np.linalg.norm(raw_errors)


def test_adapts_to_fast_motion_with_less_lag_than_fixed_cutoff():
    adaptive, fixed = PointFilter(), PointFilter(beta=0)
    for index in range(61):
        stamp = index / 30
        point = (max(0, stamp - 1), .2, 1.7)
        adaptive_point = adaptive.update(point, stamp, 1)
        fixed_point = fixed.update(point, stamp, 1)
    assert 0 < point[0] - adaptive_point[0] < .03
    assert point[0] - adaptive_point[0] < .3 * (point[0] - fixed_point[0])


@pytest.mark.parametrize('stamp,track_id', [(1.1, 2), (1.251, 1), (.9, 1), (1, 1)])
def test_new_track_gap_and_nonincreasing_time_start_with_current_point(stamp, track_id):
    point_filter = PointFilter()
    point_filter.update((1, 0, 1), .96, 1)
    point_filter.update((1.1, .1, 1.1), 1, 1)
    new_point = (-2, 1, .5)
    assert point_filter.update(new_point, stamp, track_id) == new_point
    assert point_filter.update(new_point, stamp + .03, track_id) == new_point


def test_explicit_reset_discards_history():
    point_filter = PointFilter()
    point_filter.update((1, 2, 3), 1, 1)
    point_filter.update((1.1, 2.1, 3.1), 1.03, 1)
    point_filter.reset()
    assert point_filter.update((0, 0, 1), 1.06, 1) == (0, 0, 1)


@pytest.mark.parametrize('parameter', [
    'min_cutoff_hz', 'beta', 'derivative_cutoff_hz', 'reset_after_sec'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf, -.1])
def test_rejects_invalid_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        PointFilter(**{parameter: value})


@pytest.mark.parametrize('parameter', [
    'min_cutoff_hz', 'derivative_cutoff_hz', 'reset_after_sec'])
def test_cutoffs_and_reset_interval_must_be_positive(parameter):
    with pytest.raises(ValueError, match=parameter):
        PointFilter(**{parameter: 0})


@pytest.mark.parametrize('point,stamp', [
    ((math.nan, 0, 1), 1.01), ((0, math.inf, 1), 1.01),
    ((0, 0, -math.inf), 1.01), ((0, 1), 1.01), ((0, 1, 2, 3), 1.01),
    ((0, 0, 1), math.nan), ((0, 0, 1), math.inf),
    ((0, 0, 1), -math.inf), (None, 1.01), ((0, 'invalid', 1), 1.01),
])
def test_invalid_measurement_does_not_change_history(point, stamp):
    point_filter, reference = PointFilter(), PointFilter()
    for instance in (point_filter, reference):
        instance.update((1, 0, 1), 1, 1)
    with pytest.raises(ValueError):
        point_filter.update(point, stamp, 2)
    assert point_filter.update((1.1, .1, 1.1), 1.03, 1) == reference.update(
        (1.1, .1, 1.1), 1.03, 1)


def test_rotating_and_translating_the_fixed_frame_preserves_filter_result():
    rotation = np.array([[.36, -.48, .8], [.8, .6, 0], [-.48, .64, .6]])
    translation = np.array([.7, -1.2, .1])
    original, transformed = PointFilter(), PointFilter()
    for index in range(100):
        stamp = index / 30
        point = np.array([math.sin(stamp), .2 * math.cos(3 * stamp), 1.7])
        result = original.update(point, stamp, 1)
        result_transformed = transformed.update(rotation @ point + translation, stamp, 1)
        np.testing.assert_allclose(result_transformed, rotation @ result + translation,
                                   atol=1e-12, rtol=0)


def test_uses_elapsed_time_for_irregular_camera_cadence():
    point_filter = PointFilter(beta=0, min_cutoff_hz=1)
    point_filter.update((0, 0, 0), 0, 1)
    early = point_filter.update((1, 0, 0), .01, 1)[0]
    later = point_filter.update((1, 0, 0), .11, 1)[0]
    assert later - early > early
    assert 0 < early < later < 1
