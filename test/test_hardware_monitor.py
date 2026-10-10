# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Test the read-only hardware monitor without importing xarm SDK or ROS runtime."""

from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_bringup.hardware_monitor import (  # noqa: E402
    can_request_controlled_stop, HardwareMonitorFault, HardwareStateMonitor,
    reduced_settings, report_stamp_problem, robot_report, RobotReport)

import pytest  # noqa: E402


def preflight():
    return {
        'serial_number': 'XI13051234B123',
        'reduced_mode': True,
        'reduced_joint_limits_rad': [[-1.0, 1.0]] * 6,
        'reduced_tcp_boundary_mm': [0.0] * 6,
        'reduced_tcp_max_speed_mmps': 100.0,
        'reduced_max_joint_speed_rad_s': 0.3,
        'safety_boundary_enabled': True,
        'collision_rebound_enabled': False,
    }


def sdk_state(**changes):
    values = [1, [0.0] * 6, 100.0, 0.3, [-1.0, 1.0] * 6 + [-2.0, 2.0], 1, 0]
    for key, value in changes.items():
        values[{'mode': 0, 'tcp_offset': 1, 'tcp_max_speed': 2,
                'joint_max_speed': 3, 'joint_ranges': 4,
                'safety_boundary_enabled': 5, 'collision_rebound_enabled': 6}[key]] = value
    return values


def report(**changes):
    values = {'mode': 1, 'state': 2, 'mt_able': 0x3F, 'mt_brake': 0x3F,
              'error': 0, 'warning': 0}
    values.update({'error' if key == 'err' else 'warning' if key == 'warn' else key: value
                   for key, value in changes.items()})
    return RobotReport(**values)


def armed_monitor(clock=lambda: 1.0, **kwargs):
    monitor = HardwareStateMonitor(preflight(), clock=clock, **kwargs)
    monitor.observe_serial('XI13051234B123')
    monitor.observe_report(report(), received_at=clock())
    monitor.observe_sdk(sdk_state(), received_at=clock() + 1e-6)
    assert monitor.ready
    return monitor


def test_monitor_waits_through_owned_startup_transition_then_arms_on_matched_state():
    now = [10.0]
    monitor = HardwareStateMonitor(preflight(), clock=lambda: now[0])
    monitor.observe_serial('XI13051234B123')
    monitor.observe_report(report(mode=0, state=5, mt_able=0), received_at=now[0])
    assert not monitor.ready
    monitor.observe_sdk(sdk_state(), received_at=now[0] + 1e-6)
    assert not monitor.ready
    now[0] += 0.1
    monitor.observe_report(report(state=1), received_at=now[0])
    assert monitor.ready
    monitor.check_health(now[0])


@pytest.mark.parametrize('state', [0, 1, 2])
def test_armed_monitor_accepts_controller_operating_states(state):
    monitor = armed_monitor()
    monitor.observe_report(report(state=state))
    monitor.check_health(1.1)


@pytest.mark.parametrize(('change', 'reason'), [
    ({'mode': 0}, 'mode changed'),
    ({'state': 3}, 'non-motion state 3'),
    ({'state': 4}, 'non-motion state 4'),
    ({'state': 5}, 'non-motion state 5'),
    ({'err': 23}, 'error code 23'),
    ({'mt_able': 0x1F}, 'enable mask'),
    ({'mt_brake': 1}, 'brake mask'),
])
def test_armed_robot_report_faults_fail_closed(change, reason):
    monitor = armed_monitor()
    with pytest.raises(HardwareMonitorFault, match=reason):
        monitor.observe_report(report(**change))
    assert not monitor.ready


def test_robot_warning_is_reported_but_does_not_trip_monitor():
    monitor = armed_monitor()
    monitor.observe_report(report(warn=11))
    assert monitor.ready


def test_controlled_stop_requires_fresh_healthy_report_and_never_clears_existing_stop():
    healthy = report()
    assert can_request_controlled_stop(healthy, 1.0, 1.2)
    assert not can_request_controlled_stop(report(state=5), 1.0, 1.2)
    assert not can_request_controlled_stop(report(err=23), 1.0, 1.2)
    assert not can_request_controlled_stop(report(mt_brake=0), 1.0, 1.2)
    assert not can_request_controlled_stop(report(mt_able=0x1F), 1.0, 1.2)
    assert not can_request_controlled_stop(healthy, 1.0, 1.6)


@pytest.mark.parametrize('change,reason', [
    ({'mode': 0}, 'reduced mode'),
    ({'joint_max_speed': 0.31}, 'joint speed'),
    ({'safety_boundary_enabled': 0}, 'safety-boundary'),
    ({'collision_rebound_enabled': 1}, 'collision-rebound'),
    ({'joint_ranges': [-0.9, 1.0] + [-1.0, 1.0] * 5 + [-2.0, 2.0]}, 'joint limits'),
])
def test_sdk_settings_must_match_the_preflight_snapshot(change, reason):
    monitor = HardwareStateMonitor(preflight())
    with pytest.raises(HardwareMonitorFault, match=reason):
        monitor.observe_sdk(sdk_state(**change))


@pytest.mark.parametrize('change', [
    {'tcp_offset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]},
    {'tcp_max_speed': 99.0},
    {'joint_max_speed': 0.29},
    {'joint_ranges': [-1.0, 0.99] + [-1.0, 1.0] * 5 + [-2.0, 2.0]},
    {'safety_boundary_enabled': 0},
])
def test_any_reduced_sdk_setting_change_after_arm_is_fatal(change):
    monitor = armed_monitor()
    with pytest.raises(HardwareMonitorFault, match='differ|changed'):
        monitor.observe_sdk(sdk_state(**change), received_at=1.1)


def test_joint_limit_fault_reports_axis_and_old_new_ranges():
    monitor = armed_monitor()
    changed = [-1.0, 1.0] * 6 + [-2.0, 2.0]
    changed[8:10] = [-0.8, 0.9]
    with pytest.raises(HardwareMonitorFault, match=r'joint5: old .* new .* rad'):
        monitor.observe_sdk(sdk_state(joint_ranges=changed), received_at=1.1)


def test_sdk_and_report_heartbeats_are_checked_independently():
    now = [1.0]
    monitor = armed_monitor(clock=lambda: now[0])
    now[0] += 0.51
    with pytest.raises(HardwareMonitorFault, match='report heartbeat'):
        monitor.check_health(now[0])

    now = [1.0]
    monitor = armed_monitor(clock=lambda: now[0], report_timeout_sec=1.0)
    now[0] += 0.76
    with pytest.raises(HardwareMonitorFault, match='SDK safety-state poll heartbeat'):
        monitor.check_health(now[0])


def test_connected_robot_serial_must_match_preflight_before_arming():
    monitor = HardwareStateMonitor(preflight())
    monitor.observe_report(report())
    monitor.observe_sdk(sdk_state())
    assert not monitor.ready
    with pytest.raises(HardwareMonitorFault, match='serial differs'):
        monitor.observe_serial('other-controller')


def test_repeated_robot_or_old_sdk_stamps_cannot_refresh_health():
    monitor = armed_monitor()
    with pytest.raises(HardwareMonitorFault, match='timestamp repeated'):
        monitor.observe_report(report(), source_stamp_ns=0)
    monitor = armed_monitor()
    monitor.observe_report(report(), source_stamp_ns=100)
    with pytest.raises(HardwareMonitorFault, match='timestamp repeated'):
        monitor.observe_report(report(), source_stamp_ns=100)
    with pytest.raises(HardwareMonitorFault, match='timestamp repeated'):
        monitor.observe_sdk(sdk_state(), received_at=0.9)


@pytest.mark.parametrize(('stamp', 'now', 'reason'), [
    (0, 1_000_000_000, 'no capture timestamp'),
    (1_000_000_000, 2_000_000_001, 'stale'),
    (2_000_000_000, 1_000_000_000, 'future-dated'),
    (1_000_000_000, 1_100_000_000, None),
])
def test_robot_source_stamp_age_is_bounded(stamp, now, reason):
    problem = report_stamp_problem(stamp, now)
    if reason is None:
        assert problem is None
    else:
        assert reason in problem


def test_startup_timeout_and_invalid_report_do_not_arm():
    now = [0.0]
    monitor = HardwareStateMonitor(preflight(), startup_timeout_sec=0.2,
                                   clock=lambda: now[0])
    monitor.observe_serial('XI13051234B123')
    monitor.observe_report(report(state=5), received_at=now[0])
    monitor.observe_sdk(sdk_state(), received_at=now[0])
    assert not monitor.ready
    now[0] = 0.21
    with pytest.raises(HardwareMonitorFault, match='Timed out'):
        monitor.check_health(now[0])


def test_sdk_decoder_checks_shape_and_finiteness():
    decoded = reduced_settings(sdk_state())
    assert decoded.joint_max_speed == pytest.approx(0.3)
    assert decoded.joint_ranges[-2:] == (-2.0, 2.0)
    with pytest.raises(ValueError, match='Incomplete'):
        reduced_settings([1, 2])
    malformed = sdk_state()
    malformed[1][0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        reduced_settings(malformed)


def test_robot_report_fields_follow_official_message_contract():
    parsed = robot_report(SimpleNamespace(
        mode=1, state=2, mt_able=63, mt_brake=63, err=0, warn=11))
    assert parsed == report(warn=11)
    with pytest.raises(ValueError, match='missing valid'):
        robot_report(SimpleNamespace(mode=1, state=2))
