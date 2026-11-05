"""Pure checks for the compact demo's target and phase transitions."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from compact_demo_logic import (  # noqa: E402
    freeze_target_on_start,
    generation_completed,
    joint1_working_boundary,
    post_return_target,
    STANDARD_FACE_DISTANCE_M,
    target_at_sim_elapsed,
    target_for_scenario,
    target_from_monitor_transform,
    target_to_publish,
)


def test_zero_radius_and_arc_boundary_are_floats():
    zero = target_at_sim_elapsed(0, 0, 1.68, 3.0, 48.0)
    boundary = target_at_sim_elapsed(100, 0.85, 1.68, 3.0, 48.0)

    assert zero == (0.0, 0.0, 1.68)
    assert all(type(value) is float for value in zero + boundary)
    assert boundary == target_at_sim_elapsed(3.0, 0.85, 1.68, 3.0, 48.0)


def test_target_motion_uses_sim_elapsed_only_and_stops_at_arc_boundary():
    paused_clock_first = target_at_sim_elapsed(1.5, 0.85, 1.68, 3.0, 48.0)
    paused_clock_again = target_at_sim_elapsed(1.5, 0.85, 1.68, 3.0, 48.0)

    assert paused_clock_first == paused_clock_again
    assert target_at_sim_elapsed(20.0, 0.85, 1.68, 3.0, 48.0) == \
        target_at_sim_elapsed(3.0, 0.85, 1.68, 3.0, 48.0)


def test_preparing_and_upright_start_do_not_freeze_target_but_path_does():
    published = (0.81, 0.25, 1.68)

    frozen, generation = freeze_target_on_start('PREPARING', 4, published)
    assert frozen is None and generation == 0
    frozen, generation = freeze_target_on_start('UPRIGHT_START', 4, published)
    assert frozen is None and generation == 0

    frozen, generation = freeze_target_on_start('PATH', 4, published)
    assert frozen == published and generation == 4
    assert target_to_publish((0.5, 0.5, 1.68), frozen) == published


def test_settling_can_start_if_path_diagnostic_was_missed():
    frozen, generation = freeze_target_on_start(
        'SETTLING', 7, (0.7, 0.2, 1.68))
    assert frozen == (0.7, 0.2, 1.68)
    assert generation == 7


def test_completion_is_scoped_to_started_generation_and_keeps_target():
    frozen_target = (0.8, 0.3, 1.68)

    assert not generation_completed('COMPLETE', 8, 7)
    assert generation_completed('COMPLETE', 7, 7)
    assert target_to_publish((0.1, 0.2, 1.68), frozen_target) == frozen_target


def test_stationary_target_uses_monitor_local_positive_x_normal():
    # 90 degrees about world Z maps monitor +X to world +Y.
    distance = STANDARD_FACE_DISTANCE_M
    target = target_from_monitor_transform(
        (1.0, 2.0, 3.0), (0.0, 0.0, 2 ** -0.5, 2 ** -0.5))

    assert abs(target[0] - 1.0) < 1e-12
    assert abs(target[1] - (2.0 + distance)) < 1e-12
    assert abs(target[2] - 3.0) < 1e-12


def test_post_return_target_has_continuous_zero_angle_endpoints():
    center = (0.4, 0.0, 0.2)
    start = post_return_target(0.0, center)
    left = post_return_target(0.5, center)
    middle = post_return_target(1.0, center)
    right = post_return_target(1.5, center)
    end = post_return_target(2.0, center)

    assert max(abs(value - expected) for point in (start, middle, end)
               for value, expected in zip(point, center)) < 1e-12
    assert left[1] > 0.0 > right[1]
    assert left[2] == right[2] == center[2]
    assert abs(post_return_target(0.001, center)[1] - center[1]) < 1e-5
    assert abs(post_return_target(1.999, center)[1] - center[1]) < 1e-5


def test_j1_start_uses_configured_side_and_margin():
    assert joint1_working_boundary(-6.28, 6.28, 0.20, 'upper') == 6.08
    assert joint1_working_boundary(-6.28, 6.28, 0.20, 'lower') == -6.08


def test_moving_scenarios_rotate_target_around_initial_monitor_origin():
    monitor_origin = (1.0, 2.0, 0.5)
    target = (1.0, 2.4, 0.5)
    moved = target_for_scenario('arc', 3.0, target, monitor_origin)

    assert abs((moved[0] - monitor_origin[0]) ** 2 +
               (moved[1] - monitor_origin[1]) ** 2 - 0.4 ** 2) < 1e-12
