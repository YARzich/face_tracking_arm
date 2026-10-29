# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Deterministic quintic waypoint evaluation for runtime target sequences."""

import json
import math
from pathlib import Path


def load_sequence_definition(path):
    """Load a waypoint definition from JSON for runner preflight and capture."""
    with Path(path).open(encoding='utf-8') as stream:
        definition = json.load(stream)
    if not isinstance(definition, dict) or not isinstance(
            definition.get('waypoints'), list) or not definition['waypoints']:
        raise ValueError('sequence must contain a non-empty waypoints list')
    return definition


def validate_runner_preflight(duration, mode, mode_args):
    """Validate runner inputs without launching ROS or Gazebo."""
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and greater than zero')
    if mode == 'fixed':
        if len(mode_args) != 3:
            raise ValueError('fixed mode requires exactly three coordinates')
        coordinates = [float(value) for value in mode_args]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError('target coordinates must be finite')
    elif mode == 'inband':
        if mode_args:
            raise ValueError('inband mode does not accept extra arguments')
    elif mode == 'sequence':
        if len(mode_args) != 1:
            raise ValueError('sequence mode requires one waypoint JSON path')
        load_sequence_definition(mode_args[0])
    else:
        raise ValueError(f'unsupported mode: {mode}')


def target_at(definition, return_target, elapsed_sec):
    """Resolve a target from absolute waypoint transitions and holds."""
    if not math.isfinite(elapsed_sec):
        raise ValueError('elapsed_sec must be finite')
    points = definition['waypoints']
    if len(points) < 1:
        raise ValueError('sequence needs at least one waypoint')
    resolved = [point['target_world_m'] if point['target_world_m'] else return_target
                for point in points]
    active = 0
    for index in range(1, len(points)):
        transition_start = float(points[index]['transition_start_sec'])
        transition = float(points[index]['transition_sec'])
        if elapsed_sec < transition_start:
            break
        if transition > 0.0 and elapsed_sec < transition_start + transition:
            ratio = (elapsed_sec - transition_start) / transition
            smooth = ratio ** 3 * (10.0 + ratio * (-15.0 + 6.0 * ratio))
            return [a + (b - a) * smooth
                    for a, b in zip(resolved[index - 1], resolved[index])]
        active = index
    return resolved[active]
