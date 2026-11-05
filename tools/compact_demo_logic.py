# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Small pure helpers for the bounded compact Gazebo demo."""

import math


START_PHASES = frozenset({'PATH', 'SETTLING'})
STANDARD_FACE_DISTANCE_M = 0.40
STABLE_HOLD_SEC = 10.0
ARC_DURATION_SEC = 3.0
ARC_PERIOD_SEC = 48.0
POST_RETURN_ARC_SEC = 2.0
POST_RETURN_ARC_RAD = 0.05


def joint1_working_boundary(lower_limit, upper_limit, margin, side):
    """Compute a J1 start at a configured working boundary."""
    lower, upper, margin = float(lower_limit), float(upper_limit), float(margin)
    if side not in {'upper', 'lower'}:
        raise ValueError("side must be 'upper' or 'lower'")
    if margin < 0.0 or 2.0 * margin >= upper - lower:
        raise ValueError('margin must be non-negative and smaller than half the range')
    return upper - margin if side == 'upper' else lower + margin


def target_from_monitor_transform(translation, quaternion_xyzw,
                                  distance_m=STANDARD_FACE_DISTANCE_M):
    """Place the target on monitor_control_frame's local +X normal in world."""
    x, y, z = (float(value) for value in translation)
    qx, qy, qz, qw = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError('monitor transform quaternion must be non-zero')
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    # Rotate the control frame's +X axis by its world-frame quaternion.
    nx = 1.0 - 2.0 * (qy * qy + qz * qz)
    ny = 2.0 * (qx * qy + qw * qz)
    nz = 2.0 * (qx * qz - qw * qy)
    distance = float(distance_m)
    return (x + nx * distance, y + ny * distance, z + nz * distance)


def rotate_target_about_world_z(target, angle_rad, center=(0.0, 0.0, 0.0)):
    """Rotate a world point about a supplied center and the world vertical axis."""
    x, y, z = (float(value) for value in target)
    cx, cy, _ = (float(value) for value in center)
    angle = float(angle_rad)
    dx, dy = x - cx, y - cy
    return (cx + dx * math.cos(angle) - dy * math.sin(angle),
            cy + dx * math.sin(angle) + dy * math.cos(angle), z)


def target_for_scenario(scenario, elapsed_sec, stationary_target,
                        center=(0.0, 0.0, 0.0)):
    """Return a scenario point; all motion stays on the initial target radius."""
    if scenario == 'stationary':
        return tuple(float(value) for value in stationary_target)
    if scenario == 'arc':
        angle = (2.0 * math.pi / ARC_PERIOD_SEC) * min(
            max(float(elapsed_sec), 0.0), ARC_DURATION_SEC)
        return rotate_target_about_world_z(stationary_target, angle, center)
    if scenario == 'after_return':
        return tuple(float(value) for value in stationary_target)
    raise ValueError(f'unknown compact demo scenario: {scenario}')


def post_return_target(elapsed_sec, stationary_target, center=(0.0, 0.0, 0.0)):
    """Smoothly sweep left and right, starting and ending at the held target."""
    elapsed = min(max(float(elapsed_sec), 0.0), POST_RETURN_ARC_SEC)
    progress = elapsed / POST_RETURN_ARC_SEC
    phase = progress * progress * (3.0 - 2.0 * progress)
    angle = POST_RETURN_ARC_RAD * math.sin(2.0 * math.pi * phase)
    return rotate_target_about_world_z(stationary_target, angle, center)


def target_at_sim_elapsed(elapsed_sec, radius_m, height_m, arc_duration_sec, period_sec):
    """Legacy bounded arc helper retained for callers and focused tests."""
    elapsed = min(max(float(elapsed_sec), 0.0), float(arc_duration_sec))
    angle = (2.0 * math.pi / float(period_sec)) * elapsed
    return (float(radius_m * math.cos(angle)),
            float(radius_m * math.sin(angle)), float(height_m))


def freeze_target_on_start(phase, generation, published_target,
                           frozen_target=None, frozen_generation=0):
    """Capture the last published F on the first real compact phase."""
    if frozen_target is not None:
        return frozen_target, frozen_generation
    if phase in START_PHASES and generation > 0:
        return tuple(float(value) for value in published_target), int(generation)
    return None, 0


def generation_completed(phase, generation, started_generation):
    """Accept completion only for the compact generation that actually started."""
    return (started_generation > 0 and generation == started_generation and
            phase == 'COMPLETE')


def target_to_publish(published_target, frozen_target):
    """Keep publishing the captured F after path completion."""
    target = frozen_target if frozen_target is not None else published_target
    return tuple(float(value) for value in target)
