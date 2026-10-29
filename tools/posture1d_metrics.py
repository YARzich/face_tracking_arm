# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Offline metrics for the POSTURE-1D fixed-target cycle regression."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Sample:
    """Synchronized measurement at the JointState source stamp."""

    stamp_sec: float
    q3_rad: float
    elbow_xyz_m: tuple[float, float, float]
    valid: bool = True


@dataclass(frozen=True)
class Result:
    """Fixed-F cycle result, including capture-quality failures."""

    passed: bool
    failures: tuple[str, ...]
    metrics: dict


def nearest_by_stamp(records, stamp):
    """Return the nearest timestamped record, or None when history is empty."""
    return min(records, key=lambda item: abs(item['stamp_sec'] - stamp),
               default=None)


def _span(points: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(
        (max(p[i] for p in points) - min(p[i] for p in points)) ** 2
        for i in range(3)))


def evaluate_fixed_target(
        samples: Iterable[Sample], *, straight_q3_rad: float = -2.733184269,
        minimum_coverage_sec: float = 39.95, maximum_gap_sec: float = 0.05,
        reversal_excursion_rad: float = math.radians(1.0),
        maximum_reversals: int = 1, maximum_elbow_span_m: float = 0.01
        ) -> Result:
    """
    Detect repeated >1 degree q3 reversals or >1 cm elbow motion.

    This metric is intended for the fixed-F baseline window, normally elapsed
    20..60 s. It rejects incomplete or unsynchronized data instead of allowing
    a partial capture to pass as quiet.
    """
    data = list(samples)
    metrics = {
        'sample_count': len(data), 'coverage_sec': 0.0,
        'maximum_gap_sec': math.nan, 'q3_range_rad': math.nan,
        'q3_reversals_over_1deg': 0, 'straight_crossings': 0,
        'elbow_xyz_range_m': math.nan, 'elbow_span_m': math.nan,
    }
    failures: list[str] = []
    if len(data) < 2:
        return Result(False, ('insufficient_coverage',), metrics)
    stamps = [s.stamp_sec for s in data]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    if (not all(math.isfinite(t) for t in stamps) or
            any(g <= 0 or g > maximum_gap_sec + 1e-9 for g in gaps)):
        failures.append('invalid_timestamps')
    if not all(s.valid and math.isfinite(s.q3_rad) and
               len(s.elbow_xyz_m) == 3 and
               all(math.isfinite(v) for v in s.elbow_xyz_m) for s in data):
        failures.append('invalid_measurements')
    if all(math.isfinite(t) for t in stamps):
        metrics['coverage_sec'] = stamps[-1] - stamps[0]
        metrics['maximum_gap_sec'] = max(gaps, default=math.nan)
    if metrics['coverage_sec'] < minimum_coverage_sec:
        failures.append('insufficient_coverage')
    if not any(f in failures for f in ('invalid_timestamps', 'invalid_measurements')):
        q = [s.q3_rad for s in data]
        metrics['q3_range_rad'] = max(q) - min(q)
        # Track each running extreme and count a turn only once the movement
        # away from that extreme exceeds the excursion threshold.
        turns = 0
        direction = 0
        peak = q[0]
        for value in q[1:]:
            if direction == 0:
                if abs(value - peak) > 1e-12:
                    direction = 1 if value > peak else -1
                    peak = value
            elif direction > 0:
                peak = max(peak, value)
                if peak - value >= reversal_excursion_rad:
                    turns += 1
                    direction, peak = -1, value
            else:
                peak = min(peak, value)
                if value - peak >= reversal_excursion_rad:
                    turns += 1
                    direction, peak = 1, value
        metrics['q3_reversals_over_1deg'] = turns
        metrics['straight_crossings'] = sum(
            (a - straight_q3_rad) * (b - straight_q3_rad) < 0
            for a, b in zip(q, q[1:]))
        points = [s.elbow_xyz_m for s in data]
        metrics['elbow_xyz_range_m'] = [
            max(p[i] for p in points) - min(p[i] for p in points)
            for i in range(3)]
        metrics['elbow_span_m'] = _span(points)
        if turns > maximum_reversals:
            failures.append('repeated_q3_reversals')
        if metrics['elbow_span_m'] > maximum_elbow_span_m:
            failures.append('elbow_motion')
    return Result(not failures, tuple(dict.fromkeys(failures)), metrics)
