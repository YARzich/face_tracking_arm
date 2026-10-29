# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Behavioral checks for the posture-settling hold criterion."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'tools/posture_settling_metrics.py'
SPEC = importlib.util.spec_from_file_location(
    'posture_settling_metrics', MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def _trace(point_fn, *, target_fn=None, duration=30.0, step=0.05):
    samples = []
    count = round(duration / step)
    for index in range(count + 1):
        stamp = index * step
        p, e, w = point_fn(stamp)
        target = target_fn(stamp) if target_fn else (0.8, 0.0, 0.8)
        samples.append(metrics.SettlingSample(
            stamp_sec=stamp,
            p=p,
            e=e,
            w=w,
            target=target,
            distance_m=math.dist(p, target),
        ))
    return samples


def _static_points(stamp):
    del stamp
    return (0.4, 0.0, 0.8), (0.2, 0.04, 0.7), (0.3, 0.0, 0.8)


def test_quiet_and_small_decaying_motion_pass_full_hold_and_path_limits():
    quiet = metrics.evaluate_settling(_trace(_static_points))

    def decaying(stamp):
        amplitude = 0.008 * math.exp(-stamp / 5.0)
        p, e, w = _static_points(stamp)
        return ((p[0] + amplitude, p[1], p[2]),
                (e[0], e[1] + amplitude, e[2]),
                (w[0], w[1], w[2] + amplitude))

    decay = metrics.evaluate_settling(_trace(decaying))
    assert quiet.passed
    assert decay.passed


@pytest.mark.parametrize('sign', [-1.0, 1.0])
@pytest.mark.parametrize('amplitude_m', [0.002, 0.008])
def test_sustained_oscillation_fails_hold_or_path_for_both_signs(
        sign, amplitude_m):
    def oscillating(stamp):
        p, e, w = _static_points(stamp)
        movement = sign * amplitude_m * math.sin(2.0 * math.pi * stamp / 4.0)
        return ((p[0] + movement, p[1], p[2]), e, w)

    result = metrics.evaluate_settling(_trace(oscillating))
    assert not result.passed
    assert 'point_motion' in result.failures
    if amplitude_m == 0.002:
        assert result.metrics['full_hold_point_range_m']['p'] < 0.01
        assert result.metrics['maximum_sliding_10s_path_m']['p'] > 0.01


def test_slow_drift_fails_full_hold_span_even_inside_distance_band():
    def drifting(stamp):
        p, e, w = _static_points(stamp)
        return ((p[0] + 0.0004 * stamp, p[1], p[2]), e, w)

    samples = _trace(drifting)
    result = metrics.evaluate_settling(samples)
    assert not result.passed
    assert 'point_motion' in result.failures
    assert min(sample.distance_m for sample in samples) >= 0.37
    assert max(sample.distance_m for sample in samples) <= 0.43


def test_small_borderline_jitter_stays_below_hold_and_path_limits():
    def jitter(stamp):
        p, e, w = _static_points(stamp)
        noise = 0.00014 * math.sin(2.0 * math.pi * stamp / 0.7)
        return ((p[0] + noise, p[1], p[2]), e, w)

    assert metrics.evaluate_settling(_trace(jitter)).passed


def test_elbow_motion_is_detected_when_screen_point_is_static():
    def elbow_drift(stamp):
        p, e, w = _static_points(stamp)
        return p, (e[0], e[1] + 0.0004 * stamp, e[2]), w

    result = metrics.evaluate_settling(_trace(elbow_drift))
    assert not result.passed
    assert 'point_motion' in result.failures


def test_target_movement_and_inactivity_are_not_misreported_as_settled():
    moving = metrics.evaluate_settling(_trace(
        _static_points,
        target_fn=lambda stamp: (0.8, 0.01 * math.sin(stamp), 0.8)))
    assert not moving.passed
    assert 'target_not_stationary' in moving.failures

    samples = _trace(_static_points)
    old = samples[250]
    samples[250] = metrics.SettlingSample(
        stamp_sec=old.stamp_sec, p=old.p, e=old.e, w=old.w,
        target=old.target, distance_m=old.distance_m, target_active=False)
    inactive = metrics.evaluate_settling(samples)
    assert not inactive.passed
    assert 'target_inactive' in inactive.failures


@pytest.mark.parametrize('mutation', ['duplicate', 'backwards', 'large_gap'])
def test_invalid_or_under_sampled_timestamps_fail(mutation):
    samples = _trace(_static_points)
    if mutation == 'duplicate':
        samples[50] = samples[49]
    elif mutation == 'backwards':
        old = samples[50]
        samples[50] = metrics.SettlingSample(
            stamp_sec=samples[48].stamp_sec, p=old.p, e=old.e, w=old.w,
            target=old.target, distance_m=old.distance_m)
    else:
        del samples[100:102]
    result = metrics.evaluate_settling(samples)
    assert not result.passed
    assert 'invalid_timestamps' in result.failures


def test_insufficient_hold_cannot_pass_even_if_all_points_are_quiet():
    result = metrics.evaluate_settling(_trace(_static_points, duration=8.0))
    assert not result.passed
    assert 'insufficient_coverage' in result.failures
