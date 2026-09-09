# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Stable nearest measured face selection without ROS, models or camera hardware."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_perception.face_selection import NearestFaceSelector  # noqa: E402, I100


def acquire(selector, points, start=1):
    assert selector.select(points, start) is None
    assert selector.select(points, start + .034) is None
    return selector.select(points, start + .068)


def test_uses_actual_range_and_confirms_a_new_face():
    selector = NearestFaceSelector()
    assert acquire(selector, [(2, 0, 1), (1, 0, 1), (0, 3, .5)]) == 1
    assert selector.track_id == 1


def test_reordering_and_distance_noise_do_not_switch_people():
    selector = NearestFaceSelector()
    a, b = np.array([2, -.5, 1]), np.array([2.05, .5, 1])
    assert acquire(selector, [a, b]) == 0
    for i in range(1, 15):
        b[0] = 2 + (-1)**i * .08
        assert selector.select([b, a], 1.068 + i * .034) == 1
    assert selector.track_id == 1


def test_switches_only_to_a_persistently_closer_person():
    selector = NearestFaceSelector()
    far, closer = [2, -.5, 1], [1.2, .5, 1]
    assert acquire(selector, [far]) == 0
    assert selector.select([closer, far], 1.1) == 1
    for i in range(1, 12):
        assert selector.select([closer, far], 1.1 + i * .034) == 1
    assert selector.select([closer, far], 1.51) == 0
    assert selector.track_id == 2


def test_occlusion_and_old_stamps_cannot_create_fake_confirmations():
    selector = NearestFaceSelector()
    a, b = [2, -.5, 1], [1.2, .5, 1]
    assert acquire(selector, [a]) == 0
    assert selector.select([], 1.1) is None
    assert selector.select([b], 1.2) is None
    assert selector.select([a], 1.3) == 0
    assert selector.select([a], 1.3) is None
    assert selector.select([b], 2) is None
    assert selector.select([b], 2.034) is None
    assert selector.select([b], 2.068) == 0
    # Clock reset starts a new acquisition instead of retaining a future face.
    assert selector.select([a], .1) is None


def test_invalid_positions_are_rejected():
    with pytest.raises(ValueError):
        NearestFaceSelector().select([[np.nan, 0, 1]], 1)


@pytest.mark.parametrize('fps', [1, 2, 3, 4, 5, 15, 20, 30, 60])
def test_regular_camera_cadence_acquires_and_keeps_one_track(fps):
    selector = NearestFaceSelector()
    points = [[0, 0, 1.5]]
    selected = [selector.select(points, 1 + i / fps) for i in range(20)]
    assert selected[0:2] == [None, None]
    assert all(index == 0 for index in selected[-10:])
    assert selector.track_id == 1


@pytest.mark.parametrize('fps', [1, 2, 4])
def test_low_fps_switch_still_requires_separate_confirmations_and_delay(fps):
    selector = NearestFaceSelector(switch_delay_sec=1.2)
    far, near = [2, -.5, 1], [1, .5, 1]
    for i in range(6):
        selector.select([far], i / fps)
    assert selector.track_id == 1
    results = [selector.select([far, near], (6 + i) / fps) for i in range(12)]
    first_switch = next(i for i, result in enumerate(results) if result == 1)
    assert first_switch >= 2
    assert first_switch / fps >= 1.2
    assert all(result == 1 for result in results[first_switch:])
    assert selector.track_id == 2


def test_one_delayed_frame_does_not_extend_established_fast_confirmation_window():
    selector = NearestFaceSelector()
    point = [[0, 0, 1.5]]
    for i in range(10):
        selector.select([], i * .04)
    assert selector.select(point, .4) is None
    assert selector.select(point, .44) is None
    assert selector.select(point, .74) is None
    assert selector.select(point, .78) is None
    assert selector.select(point, .82) == 0


def test_one_delayed_frame_expires_an_established_fast_track():
    selector = NearestFaceSelector()
    point = [[0, 0, 1.5]]
    for i in range(10):
        selector.select(point, i * .04)
    assert selector.track_id == 1
    assert selector.select(point, 1.16) is None
    assert selector.select(point, 1.20) is None
    assert selector.select(point, 1.24) == 0
    assert selector.track_id == 2


def test_camera_can_slow_down_without_permanently_losing_acquisition():
    selector = NearestFaceSelector()
    point = [[0, 0, 1.5]]
    for i in range(10):
        selector.select(point, i / 30)
    results = [selector.select(point, .8 + i * .5) for i in range(10)]
    assert results[-5:] == [0] * 5


def test_long_pause_restarts_confirmation_without_learning_the_pause_as_cadence():
    selector = NearestFaceSelector()
    point = [[0, 0, 1.5]]
    for i in range(6):
        selector.select(point, i)
    assert selector.track_id == 1
    assert selector.select(point, 10) is None
    assert selector.select(point, 10.04) is None
    assert selector.select(point, 10.08) == 0
    assert selector.track_id == 2
    # A new gap at the recovered fast rate must not inherit the slow camera window.
    selector.select(point, 10.12)
    assert selector.select(point, 10.92) is None


def test_empty_and_repeated_low_rate_frames_cannot_confirm_a_face():
    selector = NearestFaceSelector()
    point = [[0, 0, 1.5]]
    assert selector.select(point, 1) is None
    assert selector.select(point, 2) is None
    for _ in range(5):
        assert selector.select(point, 2) is None
    assert selector.select([], 3) is None
    assert selector.select(point, 4) is None
    assert selector.select(point, 5) is None
    assert selector.select(point, 6) == 0


def test_low_rate_jitter_keeps_the_track_and_clock_rewind_clears_cadence():
    selector = NearestFaceSelector()
    point = [[0, 0, 1.5]]
    stamps = np.cumsum([1, 1.03, .98, 1.01, .99, 1.02, .98])
    results = [selector.select(point, float(stamp)) for stamp in stamps]
    assert results[2:] == [0] * 5
    assert selector.track_id == 1
    assert selector.select(point, .1) is None
    assert selector.select(point, .14) is None
    assert selector.select(point, .18) == 0
    selector.select(point, .22)
    assert selector.select(point, 1.02) is None
