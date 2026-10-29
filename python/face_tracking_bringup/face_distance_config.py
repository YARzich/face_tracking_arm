# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Resolve the optional face-distance band and its legacy form."""

import math


def resolve_face_distance_band(tracking, position_deadband_m):
    """
    Return and store finite (minimum, maximum) meter bounds.

    Old tracking YAMLs used ``minimum_face_distance_m`` as one desired range
    center. Preserve that contract by expanding it by the controller's existing
    position deadband when an explicit pair is absent.
    """
    has_minimum = 'face_distance_min_m' in tracking
    has_maximum = 'face_distance_max_m' in tracking
    if has_minimum != has_maximum:
        raise ValueError(
            'tracking.face_distance_min_m and face_distance_max_m must be '
            'set together')
    if has_minimum:
        minimum = _positive(
            tracking['face_distance_min_m'], 'tracking.face_distance_min_m')
        maximum = _positive(
            tracking['face_distance_max_m'], 'tracking.face_distance_max_m')
    else:
        center = _positive(tracking.get('minimum_face_distance_m'),
                           'tracking.minimum_face_distance_m')
        deadband = _nonnegative(
            position_deadband_m, 'controller.position_deadband_m')
        minimum, maximum = center - deadband, center + deadband
    if (not math.isfinite(minimum) or not math.isfinite(maximum) or
            minimum <= 0 or minimum >= maximum):
        raise ValueError(
            'tracking face distance band must satisfy finite '
            '0 < minimum < maximum')
    tracking['face_distance_min_m'] = minimum
    tracking['face_distance_max_m'] = maximum
    return minimum, maximum


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name}: expected a finite positive number')
    return float(value)


def _nonnegative(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0):
        raise ValueError(f'{name}: expected a finite non-negative number')
    return float(value)
