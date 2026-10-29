# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""
Offline acceptance metric for stopping arm motion at a fixed target.

Samples are synchronized actual world-frame points. Distance is reported but
is not a settling proxy: each of P/E/W must stay within a 1 cm full-hold span
and travel no more than 1 cm along its measured path in every sliding 10 s
window. A caller must provide at least 20 Hz measurements to resolve that path.
"""

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class SettlingSample:
    """One synchronized world-frame measurement at ``stamp_sec``."""

    stamp_sec: float
    p: Vector3
    e: Vector3
    w: Vector3
    target: Vector3
    distance_m: float
    target_active: bool = True


@dataclass(frozen=True)
class SettlingResult:
    """Outcome and scalar measurements from an offline settling window."""

    passed: bool
    failures: Tuple[str, ...]
    metrics: dict


def _point_range(points: Sequence[Vector3]) -> float:
    """Return 3-D bounding-box diagonal for a point series."""
    if not points:
        return math.nan
    spans = []
    for axis in range(3):
        values = [point[axis] for point in points]
        spans.append(max(values) - min(values))
    return math.sqrt(sum(span * span for span in spans))


def _valid_vector(point: Sequence[float]) -> bool:
    return len(point) == 3 and all(math.isfinite(value) for value in point)


def _sliding_path_max(stamps: Sequence[float], points: Sequence[Vector3],
                      window_sec: float) -> float:
    """Maximum piecewise-linear path length in any full sliding window."""
    if len(stamps) < 2:
        return math.nan
    lengths = [math.dist(left, right)
               for left, right in zip(points, points[1:])]
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    def path_at(stamp):
        index = bisect_right(stamps, stamp) - 1
        if index < 0:
            return 0.0
        if index >= len(stamps) - 1:
            return cumulative[-1]
        fraction = ((stamp - stamps[index]) /
                    (stamps[index + 1] - stamps[index]))
        return cumulative[index] + fraction * lengths[index]

    start, finish = stamps[0], stamps[-1]
    # For a piecewise-linear trace, the derivative of this window's path is
    # piecewise constant. Its extrema occur when a window edge crosses a sample
    # breakpoint (end == t_i or end-window == t_i).
    candidates = {stamp for stamp in stamps if stamp >= start + window_sec}
    candidates.update(stamp + window_sec for stamp in stamps
                      if start + window_sec <= stamp + window_sec <= finish)
    paths = [path_at(end) - path_at(end - window_sec)
             for end in candidates]
    return max(paths) if paths else math.nan


def evaluate_settling(
        samples: Iterable[SettlingSample], *,
        minimum_coverage_sec: float = 30.0,
        path_window_sec: float = 10.0,
        maximum_sample_gap_sec: float = 0.05,
        maximum_target_range_m: float = 1e-5,
        maximum_hold_span_m: float = 0.01,
        maximum_path_m: float = 0.01) -> SettlingResult:
    """
    Evaluate a fixed-target hold against the project settling criterion.

    The default capture needs at least 30 s, a maximum sample interval of 50 ms
    (20 Hz), and numerical-only target movement (10 micrometers). P, E and W
    are each checked across the complete hold and in every sliding 10 s path.
    Distance min/max are returned for the caller's separate range assessment.
    """
    data = list(samples)
    failures = []
    metrics = {
        'sample_count': len(data),
        'coverage_sec': 0.0,
        'maximum_gap_sec': math.nan,
        'target_range_m': math.nan,
        'full_hold_point_range_m': {},
        'maximum_sliding_10s_path_m': {},
        'distance_min_m': math.nan,
        'distance_max_m': math.nan,
    }
    if not data:
        return SettlingResult(False, ('insufficient_coverage',), metrics)

    stamps = [sample.stamp_sec for sample in data]
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    if any(not math.isfinite(stamp) for stamp in stamps) or any(
            gap <= 0.0 or gap > maximum_sample_gap_sec + 1e-9
            for gap in gaps):
        failures.append('invalid_timestamps')
    if len(data) > 1 and all(math.isfinite(stamp) for stamp in stamps):
        metrics['coverage_sec'] = stamps[-1] - stamps[0]
        metrics['maximum_gap_sec'] = max(gaps)
    if metrics['coverage_sec'] + 1e-9 < minimum_coverage_sec:
        failures.append('insufficient_coverage')

    valid = all(
        _valid_vector(sample.p) and _valid_vector(sample.e)
        and _valid_vector(sample.w) and _valid_vector(sample.target)
        and math.isfinite(sample.distance_m)
        for sample in data)
    if not valid:
        failures.append('invalid_measurements')
    if not all(sample.target_active for sample in data):
        failures.append('target_inactive')

    if valid:
        metrics['target_range_m'] = _point_range(
            [sample.target for sample in data])
        distances = [sample.distance_m for sample in data]
        metrics['distance_min_m'] = min(distances)
        metrics['distance_max_m'] = max(distances)
        if metrics['target_range_m'] > maximum_target_range_m:
            failures.append('target_not_stationary')

        if metrics['coverage_sec'] >= path_window_sec:
            point_ranges = {}
            path_maxima = {}
            for name in ('p', 'e', 'w'):
                points = [getattr(sample, name) for sample in data]
                point_ranges[name] = _point_range(points)
                path_maxima[name] = _sliding_path_max(
                    stamps, points, path_window_sec)
            metrics['full_hold_point_range_m'] = point_ranges
            metrics['maximum_sliding_10s_path_m'] = path_maxima
            if (max(point_ranges.values()) > maximum_hold_span_m or
                    max(path_maxima.values()) > maximum_path_m):
                failures.append('point_motion')
        else:
            failures.append('insufficient_coverage')

    return SettlingResult(not failures, tuple(dict.fromkeys(failures)), metrics)
