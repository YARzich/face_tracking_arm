# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Controls for the fixed-F cycle measurement used by POSTURE-1D."""

import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'posture1d_metrics', ROOT / 'tools/posture1d_metrics.py')
metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def trace(q_fn, e_fn, *, duration=40.0, step=0.05):
    rows = []
    for index in range(round(duration / step) + 1):
        t = index * step
        rows.append(metrics.Sample(
            t, q_fn(t), (0.0, 0.0, e_fn(t))))
    return rows


def test_quiet_control_passes():
    result = metrics.evaluate_fixed_target(trace(lambda _: -2.2, lambda _: 0.4))
    assert result.passed
    assert result.metrics['q3_reversals_over_1deg'] == 0
    assert result.metrics['elbow_span_m'] == 0.0


def test_weak_sustained_oscillation_triggers_cycle_metric():
    rows = trace(lambda t: -2.2 + math.radians(0.8) * math.sin(2 * math.pi * t / 2),
                 lambda _: 0.4)
    result = metrics.evaluate_fixed_target(rows)
    assert not result.passed
    assert 'repeated_q3_reversals' in result.failures


def test_large_q3_cycle_and_elbow_motion_fail():
    rows = trace(lambda t: -2.7 + 0.25 * math.sin(2 * math.pi * t / 2),
                 lambda t: 0.4 + 0.043 * math.sin(2 * math.pi * t / 2))
    result = metrics.evaluate_fixed_target(rows)
    assert not result.passed
    assert 'repeated_q3_reversals' in result.failures
    assert 'elbow_motion' in result.failures
    assert result.metrics['straight_crossings'] > 10


def test_slow_monotonic_elbow_drift_fails_without_cycle():
    rows = trace(lambda _: -2.2, lambda t: 0.4 + 0.0004 * t)
    result = metrics.evaluate_fixed_target(rows)
    assert not result.passed
    assert 'elbow_motion' in result.failures
    assert 'repeated_q3_reversals' not in result.failures


def test_incomplete_or_gapped_capture_cannot_pass():
    rows = trace(lambda _: -2.2, lambda _: 0.4)
    short = metrics.evaluate_fixed_target(rows[:100])
    assert 'insufficient_coverage' in short.failures
    del rows[200:203]
    gapped = metrics.evaluate_fixed_target(rows)
    assert 'invalid_timestamps' in gapped.failures


def test_nearest_by_stamp_selects_closest_available_record():
    records = [{'stamp_sec': 1.0, 'value': 'before'},
               {'stamp_sec': 1.1, 'value': 'after'}]

    assert metrics.nearest_by_stamp(records, 1.08)['value'] == 'after'
    assert metrics.nearest_by_stamp([], 1.0) is None
