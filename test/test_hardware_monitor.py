# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Test the read-only hardware monitor without importing xarm SDK or ROS runtime."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_bringup.hardware_fault_recorder import (  # noqa: E402
    BoundedContextHistory, BoundedPublicationPayloadCache, HardwareFaultRecorder,
    missing_publication_payload_companions)
from face_tracking_bringup.hardware_monitor import (  # noqa: E402
    can_request_controlled_stop, face_transform_provenance_details,
    hardware_io_status_details, HardwareMonitorFault, HardwareStateMonitor,
    inhibit_future_result, inhibit_status_confirms_request, reduced_settings,
    report_stamp_problem, robot_report, robot_report_details, RobotReport,
    servo_execution_context_details, servo_trajectory_publication_details,
    stop_future_result, StopEvidence)

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
    message = SimpleNamespace(mode=1, state=2, mt_able=63, mt_brake=63, err=0, warn=11)
    parsed = robot_report(message)
    assert parsed == report(warn=11)
    assert dict(robot_report_details(message))['warn'] == '11'
    with pytest.raises(ValueError, match='missing valid'):
        robot_report(SimpleNamespace(mode=1, state=2))


def test_robot_report_diagnostics_capture_all_declared_fields_with_bounded_values():
    class FullReport(SimpleNamespace):
        def get_fields_and_field_types(self):
            return {'mode': 'uint8', 'state': 'uint8', 'extra': 'string'}

    details = robot_report_details(FullReport(mode=1, state=4, extra='x' * 300))
    assert [name for name, _ in details] == ['mode', 'state', 'extra']
    assert len(dict(details)['extra']) == 256


def test_hardware_io_status_record_preserves_fixed_arrays_and_independent_stamps():
    class Stamp:
        sec = 12
        nanosec = 34

    class Header:
        stamp = Stamp()

    class Status:
        header = Header()
        read_stamp = Stamp()
        write_stamp = Stamp()
        robot_msg_stamp = Stamp()
        fault_stamp = Stamp()
        q = [0.1] * 6
        dq = [0.2] * 6
        command_double = [0.3] * 6
        command_float = [0.4] * 6
        read_epoch = 9

        def get_fields_and_field_types(self):
            return {name: 'field' for name in (
                'header', 'read_stamp', 'write_stamp', 'robot_msg_stamp', 'fault_stamp',
                'read_epoch',
                'q', 'dq', 'command_double', 'command_float')}

    fields = dict(hardware_io_status_details(Status()))
    assert fields['header'] == (12, 34)
    assert fields['read_stamp'] == (12, 34)
    assert fields['fault_stamp'] == (12, 34)
    assert fields['read_epoch'] == 9
    assert fields['q'] == (0.1,) * 6
    assert fields['command_float'] == (0.4,) * 6


def test_fault_report_preserves_simultaneous_error_mode_state_and_masks():
    monitor = armed_monitor()
    conflicted = report(mode=0, state=5, err=23, mt_able=0x1F, mt_brake=0x0F)
    with pytest.raises(HardwareMonitorFault) as error:
        monitor.observe_report(conflicted)
    for reason in ('error code 23', 'mode changed', 'non-motion state 5',
                   'brake mask changed', 'servo-enable mask'):
        assert reason in str(error.value)
    assert monitor.fault_report == conflicted


def test_stop_ack_and_stop_state_report_are_not_measured_standstill():
    evidence = StopEvidence(stable_samples=3, velocity_tolerance=0.01,
                            position_tolerance=1e-4, max_feedback_age_sec=0.2)
    evidence.mark_requested()
    evidence.mark_acknowledged(True)
    assert evidence.acknowledged
    assert not evidence.measured_stopped
    assert evidence.observe_report(report(state=4))
    assert evidence.stop_reported
    assert not evidence.observe_report(report(state=4), valid=False)
    assert not evidence.stop_reported
    assert not evidence.measured_stopped


def test_stop_future_late_and_failed_responses_remain_unconfirmed():
    class Future:
        def __init__(self, response=None, error=None, done=True):
            self.response, self.error, self._done = response, error, done

        def done(self):
            return self._done

        def result(self):
            if self.error:
                raise self.error
            return self.response

    delayed = Future(done=False)
    assert stop_future_result(delayed, 4.0, 5.0) == (False, None, None)
    assert stop_future_result(delayed, 5.1, 5.0) == (
        True, False, 'set_state acknowledgement timed out')
    assert stop_future_result(Future(SimpleNamespace(ret=1, message='rejected')), 1.0, 2.0) == (
        True, False, 'rejected')
    assert stop_future_result(Future(error=RuntimeError('transport failed')), 1.0, 2.0) == (
        True, False, 'transport failed')
    assert stop_future_result(Future(SimpleNamespace(ret=0)), 1.0, 2.0) == (
        True, True, None)


def test_stop_becomes_measured_only_after_fresh_successful_stable_feedback():
    evidence = StopEvidence(stable_samples=3, velocity_tolerance=0.01,
                            position_tolerance=1e-4, max_feedback_age_sec=0.2)
    q = [0.1] * 6
    dq = [0.0] * 6

    def sample(sequence, epoch=4, **changes):
        args = {'read_ok': True, 'read_sequence': sequence, 'epoch': epoch,
                'read_age_sec': 0.01, 'transport_age_sec': 0.01}
        args.update(changes)
        return evidence.observe_feedback(q, dq, **args)

    assert not sample(1)
    assert not sample(1)  # Replaying one successful sample never advances the count.
    assert not sample(2, read_ok=False)
    assert not sample(2)
    assert not sample(3)
    assert sample(4)
    assert not sample(1, epoch=5)
    assert not evidence.measured_stopped


def test_mock_inhibit_stop_ack_report_then_measured_feedback_chain(tmp_path):
    recorder = HardwareFaultRecorder(pre_capacity=4, post_capacity=8)
    evidence = StopEvidence(stable_samples=2, velocity_tolerance=0.01,
                            position_tolerance=1e-4, max_feedback_age_sec=0.2)
    assert recorder.record('pre_fault_sample', 7)
    assert recorder.latch_fault('read failure', 12)
    assert recorder.record('inhibit_latched', 4, 9)
    evidence.mark_requested()
    assert recorder.record('stop_request', 13)
    evidence.mark_acknowledged(True)
    assert recorder.record('stop_ack', True)
    assert evidence.observe_report(report(state=4))
    assert recorder.record('stop_report_transition', 100, 1, 4, 0, 0x3f, 0x3f)
    assert not evidence.measured_stopped
    feedback = {'read_ok': True, 'epoch': 4, 'read_age_sec': 0.01,
                'transport_age_sec': 0.01}
    assert not evidence.observe_feedback([0.0] * 6, [0.0] * 6,
                                         read_sequence=1, **feedback)
    assert not evidence.measured_stopped
    assert evidence.observe_feedback([0.0] * 6, [0.0] * 6,
                                     read_sequence=2, **feedback)
    assert evidence.measured_stopped
    assert recorder.record('stop_measured', 2, 0.0, 0.0)
    assert recorder.close_window('measured')
    _, _, _, transitions, closed = recorder.snapshot()
    assert closed
    assert [event.kind for event in transitions] == [
        'fault', 'inhibit_latched', 'stop_request', 'stop_ack',
        'stop_report_transition', 'stop_measured', 'stop_window_end']
    assert recorder.write_dump(tmp_path, {'runtime_actions_enabled': False}).exists()


@pytest.mark.parametrize('future,now,deadline,reason', [
    (None, 2.0, 1.0, 'no fresh safe report'),
    (SimpleNamespace(done=lambda: False), 2.0, 1.0,
     'set_state acknowledgement timed out'),
    (SimpleNamespace(done=lambda: True,
                     result=lambda: SimpleNamespace(ret=1, message='rejected')),
     0.5, 1.0, 'rejected'),
])
def test_mock_fault_chain_records_unconfirmed_stop_without_inventing_standstill(
        tmp_path, future, now, deadline, reason):
    recorder = HardwareFaultRecorder(pre_capacity=8, post_capacity=8)
    evidence = StopEvidence(stable_samples=2, velocity_tolerance=0.01,
                            position_tolerance=1e-4, max_feedback_age_sec=0.2)
    assert recorder.latch_fault('driver read failure', 7)
    assert not can_request_controlled_stop(None, None, now)
    if future is None:
        assert recorder.record('stop_request_unconfirmed_timeout', reason)
    else:
        evidence.mark_requested()
        assert recorder.record('stop_request', 4)
        done, accepted, detail = stop_future_result(future, now, deadline)
        assert done and accepted is False and detail == reason
        evidence.mark_acknowledged(False)
        assert recorder.record('stop_ack', False, detail)
    assert evidence.observe_report(report(state=4))
    assert recorder.record('stop_report_transition', 1, 1, 4, 0, 0x3f, 0x3f)
    assert not evidence.measured_stopped
    assert not evidence.observe_feedback(
        [0.0] * 6, [0.0] * 6, read_ok=True, read_sequence=1, epoch=3,
        read_age_sec=0.4, transport_age_sec=0.0)
    assert not evidence.measured_stopped
    assert recorder.close_window('timeout')
    _, post, _, transitions, closed = recorder.snapshot()
    assert closed
    assert any(event.kind == 'stop_window_end' for event in post)
    assert 'stop_measured' not in [event.kind for event in transitions]
    dump = json.loads(recorder.write_dump(tmp_path, {'source': 'mock'}).read_text())
    assert dump['pre_fault_window']['pre_fault_window_complete'] is False


@pytest.mark.parametrize(('changes', 'expected'), [
    ({}, True),
    ({'hardware_active': False}, False),
    ({'runtime_actions_enabled': False}, False),
    ({'status_epoch': 8}, False),
    ({'inhibit_epoch': 8}, False),
    ({'inhibit_latched': False}, False),
    ({'inhibit_request_id': 4}, False),
    ({'write_in_flight': 1}, False),
    ({'status_age_sec': 0.51}, False),
    ({'status_age_sec': float('nan')}, False),
])
def test_stop_requires_fresh_matching_inhibit_status_and_drained_write(changes, expected):
    values = {
        'hardware_active': True, 'runtime_actions_enabled': True, 'status_epoch': 9,
        'inhibit_latched': True, 'inhibit_epoch': 9, 'inhibit_request_id': 5,
        'requested_epoch': 9, 'previous_request_id': 4, 'status_age_sec': 0.01,
        'write_in_flight': 0, 'freshness_sec': 0.5}
    values.update(changes)
    assert inhibit_status_confirms_request(**values) is expected


def test_inhibit_service_ack_does_not_replace_matching_status_or_stop_ack():
    future = SimpleNamespace(done=lambda: True,
                             result=lambda: SimpleNamespace(success=True, message='latched'))
    assert inhibit_future_result(future, 1.0, 2.0) == (True, True, None)
    assert not inhibit_status_confirms_request(
        hardware_active=True, runtime_actions_enabled=True, status_epoch=9,
        inhibit_latched=False, inhibit_epoch=0, inhibit_request_id=0,
        requested_epoch=9, previous_request_id=0, status_age_sec=0.01,
        write_in_flight=0)
    assert not inhibit_status_confirms_request(
        hardware_active=True, runtime_actions_enabled=True, status_epoch=9,
        inhibit_latched=True, inhibit_epoch=9, inhibit_request_id=1,
        requested_epoch=9, previous_request_id=0, status_age_sec=0.01,
        write_in_flight=1)
    # A future STOP acknowledgement remains independent of controller state and feedback.
    stop_future = SimpleNamespace(done=lambda: True,
                                  result=lambda: SimpleNamespace(ret=0, message='accepted'))
    assert stop_future_result(stop_future, 1.0, 2.0) == (True, True, None)
    evidence = StopEvidence(stable_samples=2, velocity_tolerance=0.01,
                            position_tolerance=1e-4, max_feedback_age_sec=0.2)
    evidence.mark_acknowledged(True)
    assert evidence.acknowledged
    assert not evidence.stop_reported
    assert not evidence.measured_stopped


def test_new_epoch_or_lost_latch_invalidates_pending_stop_authorization():
    status = {
        'hardware_active': True, 'runtime_actions_enabled': True, 'status_epoch': 9,
        'inhibit_latched': True, 'inhibit_epoch': 9, 'inhibit_request_id': 3,
        'requested_epoch': 9, 'previous_request_id': 2, 'status_age_sec': 0.01,
        'write_in_flight': 0}
    assert inhibit_status_confirms_request(**status)
    status['status_epoch'] = 10
    assert not inhibit_status_confirms_request(**status)
    status.update(status_epoch=9, inhibit_latched=False)
    assert not inhibit_status_confirms_request(**status)
    status.update(inhibit_latched=True, status_age_sec=0.6)
    assert not inhibit_status_confirms_request(**status)


@pytest.mark.parametrize(('response', 'expected'), [
    (SimpleNamespace(success=True, message='latched'), (True, True, None)),
    (SimpleNamespace(success=False, message='inactive'), (True, False, 'inactive')),
    (None, (True, False, 'empty response')),
])
def test_inhibit_service_ack_outcome_is_bounded(response, expected):
    future = SimpleNamespace(done=lambda: True, result=lambda: response)
    assert inhibit_future_result(future, 1.0, 2.0) == expected


def test_timed_out_inhibit_future_is_unconfirmed():
    future = SimpleNamespace(done=lambda: False)
    assert inhibit_future_result(future, 1.5, 1.0) == (
        True, False, 'hardware inhibit acknowledgement timed out')


@pytest.mark.parametrize('q,dq,changes', [
    ([float('nan')] * 6, [0.0] * 6, {}),
    ([0.0] * 5, [0.0] * 6, {}),
    ([0.0] * 6, [0.0] * 6, {'read_age_sec': 0.3}),
    ([0.0] * 6, [0.0] * 6, {'transport_age_sec': 0.3}),
    ([0.0] * 6, [0.0] * 6, {'epoch': -1}),
])
def test_invalid_or_stale_feedback_clears_prior_measured_stop(q, dq, changes):
    evidence = StopEvidence(stable_samples=2, velocity_tolerance=0.01,
                            position_tolerance=1e-4, max_feedback_age_sec=0.2)
    args = {'read_ok': True, 'read_sequence': 1, 'epoch': 1,
            'read_age_sec': 0.01, 'transport_age_sec': 0.01}
    assert not evidence.observe_feedback([0.0] * 6, [0.0] * 6, **args)
    assert evidence.observe_feedback([0.0] * 6, [0.0] * 6,
                                     **{**args, 'read_sequence': 2})
    args.update(changes)
    assert not evidence.observe_feedback(q, dq, **args)
    assert not evidence.measured_stopped


def test_fault_recorder_freezes_pre_fault_and_keeps_first_fault_with_bounded_post_events():
    clock = [0.0]
    recorder = HardwareFaultRecorder(pre_capacity=2, post_capacity=2,
                                     clock=lambda: clock[0])
    for index in range(3):
        clock[0] += 1.0
        assert recorder.record('sample', index)
    assert recorder.latch_fault('first fault', 23)
    assert recorder.latch_fault('replacement fault', 99)
    assert recorder.record('stop_ack', True)
    assert recorder.record('later_report', 4)
    pre, post, primary, transitions, closed = recorder.snapshot()
    assert [item.payload for item in pre] == [(1,), (2,)]
    assert primary.kind == 'fault' and primary.payload == ('first fault', 23)
    assert [item.kind for item in post] == ['stop_ack', 'later_report']
    assert recorder.primary_fault == primary
    assert [item.kind for item in transitions] == [
        'fault', 'stop_ack']
    assert not closed


def test_fault_recorder_retains_ten_seconds_at_status_and_robot_report_rate(tmp_path):
    recorder = HardwareFaultRecorder()
    events = [(tick / 100.0, 'hardware_io_status') for tick in range(1051)]
    for tick in range(2626):
        stamp = tick / 250.0
        events.extend(((stamp, 'robot_report'), (stamp, 'robot_report_full')))
    for stamp, kind in sorted(events):
        assert recorder.record(kind, monotonic_time=stamp)

    assert recorder.latch_fault('read fault', monotonic_time=10.5)
    recorder.close_window('timeout', monotonic_time=10.6)
    dump = json.loads(recorder.write_dump(tmp_path, {}).read_text(encoding='utf-8'))
    window = dump['pre_fault_window']
    assert window['requested_pre_fault_window_sec'] >= 10.0
    assert window['pre_fault_coverage_sec'] >= 10.0
    assert window['pre_fault_window_complete']
    assert window['pre_fault_capacity_drops'] == 0


def test_default_fault_recorder_keeps_more_than_4096_events_over_ten_seconds():
    recorder = HardwareFaultRecorder()
    # Four 100 Hz sources (status, executor tick, TF provenance, and report).
    for tick in range(1051):
        stamp = tick / 100.0
        for kind in ('hardware_io_status', 'servo_execution_context',
                     'face_transform_provenance', 'robot_report'):
            assert recorder.record(kind, monotonic_time=stamp)
    assert recorder.latch_fault('fault', monotonic_time=10.5)
    pre, _, _, _, _ = recorder.snapshot()
    assert len(pre) > 4096
    assert recorder._pre_capacity_drops == 0
    assert recorder._prehistory_metadata(pre, recorder.primary_fault)[
        'pre_fault_window_complete']


def test_bounded_context_history_joins_by_receipt_and_preserves_source_stamp():
    history = BoundedContextHistory(capacity=2)
    assert history.record(100, ('target', 7), receipt_monotonic=1.0)
    assert history.record(200, ('publication', 42), receipt_monotonic=2.0)
    assert history.record(300, ('tail', 9), receipt_monotonic=3.0)
    assert history.overwritten_events == 1

    matched = history.match(2.4, maximum_age_sec=0.5)
    assert matched is not None
    context, age = matched
    assert context.source_stamp_ns == 200
    assert context.payload == ('publication', 42)
    assert age == pytest.approx(0.4)
    assert history.match(4.0, maximum_age_sec=0.2) is None
    assert history.match(float('nan'), maximum_age_sec=1.0) is None


def test_context_join_requires_matching_epoch_and_read_sequence():
    history = BoundedContextHistory(capacity=4)
    payload = (('hardware_epoch', 8), ('hardware_read_sequence', 17),
               ('active_publication_id', 51))
    assert history.record(1234, payload, receipt_monotonic=4.0)
    match = history.match(4.1, 0.2, key=(8, 17))
    assert match is not None
    assert match[0].source_stamp_ns == 1234
    assert dict(match[0].payload)['active_publication_id'] == 51
    assert history.match(4.1, 0.2, key=(8, 18)) is None


def test_servo_context_preserves_target_publication_tail_and_tf_provenance():
    stamp = SimpleNamespace(sec=2, nanosec=3)

    def point(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)
    transform = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id='world'), child_frame_id='screen',
        transform=SimpleNamespace(translation=point(1, 2, 3),
                                  rotation=SimpleNamespace(x=0, y=0, z=0, w=1)))
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp), tick_sequence=9, hardware_epoch=3,
        hardware_read_sequence=12, target_available=True, target_used_for_reference=True,
        target_mode=1, target_id=22, target_source_stamp=stamp, face_stamp=stamp,
        target_frame='camera', target_position=point(0.1, 0.2, 0.3),
        face_position=point(0.4, 0.5, 0.6), tf_provenance_available=False,
        target_transform=transform, active_publication_id=31, latest_publication_id=32,
        braking_tail_publication_id=31, braking_tail_stationary_stamp=stamp,
        dropped_context_records=2, dropped_publication_events=4)
    details = dict(servo_execution_context_details(message))
    assert details['hardware_epoch'] == 3
    assert details['hardware_read_sequence'] == 12
    assert details['active_publication_id'] == 31
    assert details['braking_tail_publication_id'] == 31
    assert details['tf_provenance_available'] is False
    assert details['target_frame'] == 'camera'
    assert details['dropped_context_records'] == 2
    assert details['dropped_publication_events'] == 4


def test_face_transform_provenance_keeps_exact_transforms_and_source_stamp():
    stamp = SimpleNamespace(sec=7, nanosec=11)

    def point(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    def transform(parent, child, x):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=stamp, frame_id=parent), child_frame_id=child,
            transform=SimpleNamespace(translation=point(x, 0, 0),
                                      rotation=SimpleNamespace(x=0, y=0, z=0, w=1)))

    message = SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id='camera'),
        accepted_face_sequence=42, source_face=point(1, 2, 3),
        planning_face=point(4, 5, 6),
        planning_from_face=transform('world', 'camera', 3),
        planning_from_base=transform('world', 'base', 8),
        planning_from_reach_center=transform('world', 'reach_center', 9),
        planning_base=point(8, 0, 0), planning_reach_center=point(9, 0, 0),
        face_transform_lookup_used=True, base_transform_lookup_used=True,
        reach_center_transform_lookup_used=True)
    details = dict(face_transform_provenance_details(message))
    assert details['source_stamp'] == (7, 11)
    assert details['source_frame'] == 'camera'
    assert details['accepted_face_sequence'] == 42
    assert details['planning_face'] == (4.0, 5.0, 6.0)
    assert details['planning_from_face'][1:4] == ('world', 'camera', (3.0, 0.0, 0.0))
    assert details['planning_from_base'][2] == 'base'
    assert details['planning_from_reach_center'][2] == 'reach_center'


def test_servo_trajectory_publication_details_keep_points_and_control_provenance():
    stamp = SimpleNamespace(sec=8, nanosec=15)
    event_stamp = SimpleNamespace(sec=9, nanosec=25)
    duration = SimpleNamespace(sec=0, nanosec=40_000_000)
    point = SimpleNamespace(positions=[1.0, 2.0], velocities=[0.1, 0.2],
                            accelerations=[0.3, 0.4], effort=[], time_from_start=duration)
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=event_stamp, frame_id='publication_frame'),
        publication_kind=2, publication_id=91,
        stationary_time=stamp, tail_start_index=6, hardware_epoch=5,
        hardware_read_sequence=22, control_tick_sequence=49,
        dropped_publication_events=3,
        trajectory=SimpleNamespace(
            header=SimpleNamespace(stamp=stamp, frame_id='base'),
            joint_names=['joint1', 'joint2'], points=[point]))
    details = dict(servo_trajectory_publication_details(message))
    assert details['event_stamp'] == (9, 25)
    assert details['event_frame_id'] == 'publication_frame'
    assert details['publication_kind'] == 2
    assert details['trajectory_header_stamp'] == (8, 15)
    assert details['publication_id'] == 91
    assert details['hardware_epoch'] == 5
    assert details['hardware_read_sequence'] == 22
    assert details['control_tick_sequence'] == 49
    assert details['tail_start_index'] == 6
    assert details['dropped_publication_events'] == 3
    assert details['joint_names'] == ('joint1', 'joint2')
    assert details['points'] == (((1.0, 2.0), (0.1, 0.2), (0.3, 0.4), (),
                                  (0, 40_000_000)),)


def test_fault_recorder_marks_capacity_loss_incomplete_and_dumps_nonfinite_status(tmp_path):
    class Stamp:
        sec = 10
        nanosec = 20

    class Header:
        stamp = Stamp()

    class Status:
        header = Header()
        read_stamp = write_stamp = robot_msg_stamp = fault_stamp = Stamp()
        q = [float('nan')] + [0.0] * 5
        dq = [float('inf')] + [0.0] * 5
        command_double = [float('-inf')] + [0.0] * 5
        command_float = [0.0] * 6

        def get_fields_and_field_types(self):
            return {name: 'field' for name in (
                'header', 'read_stamp', 'write_stamp', 'robot_msg_stamp', 'fault_stamp',
                'q', 'dq', 'command_double', 'command_float')}

    recorder = HardwareFaultRecorder(pre_capacity=2, pre_window_sec=10.0)
    recorder.record('hardware_io_status', 1, monotonic_time=0.0)
    recorder.record('hardware_io_status', 2, monotonic_time=0.1)
    recorder.record('hardware_io_status', 3, monotonic_time=0.2)
    recorder.record('hardware_io_status', hardware_io_status_details(Status()),
                    monotonic_time=10.1)
    recorder.latch_fault('nonfinite read', monotonic_time=10.1)
    recorder.close_window('timeout', monotonic_time=10.2)

    dump = json.loads(recorder.write_dump(tmp_path, {}).read_text(encoding='utf-8'))
    assert not dump['pre_fault_window']['pre_fault_window_complete']
    assert dump['pre_fault_window']['pre_fault_window_capacity_drops'] > 0
    status_fields = dict(dump['pre_fault'][-1]['payload'][0])
    assert status_fields['q'][0] == 'NaN'
    assert status_fields['dq'][0] == 'Infinity'
    assert status_fields['command_double'][0] == '-Infinity'


def test_fault_recorder_closes_stop_window_and_preserves_primary_transitions(tmp_path):
    recorder = HardwareFaultRecorder(pre_capacity=2, post_capacity=1)
    recorder.record('sample', 1)
    recorder.latch_fault('root cause', 23)
    recorder.record('stop_request', 4)
    recorder.record('stop_ack', True)
    assert recorder.close_window('timeout')
    assert not recorder.record('late_report', 5)
    pre, post, primary, transitions, closed = recorder.snapshot()
    assert closed
    assert primary.payload == ('root cause', 23)
    assert [event.kind for event in transitions] == [
        'fault', 'stop_request', 'stop_ack', 'stop_window_end']
    assert post[-1].payload == ('timeout',)
    path = recorder.write_dump(tmp_path, {'config_hash': 'abc'})
    dump = json.loads(path.read_text(encoding='utf-8'))
    assert dump['metadata']['config_hash'] == 'abc'
    assert dump['primary_fault']['payload'] == ['root cause', 23]
    assert dump['transitions'][-1]['kind'] == 'stop_window_end'
    assert dump['post_fault_window_capacity_drops'] == 3
    assert not dump['post_fault_window_complete']


def test_active_publication_payload_survives_long_stationary_hold_once(tmp_path):
    cache = BoundedPublicationPayloadCache(capacity=1)
    active_payload = (('publication_id', 17), ('points', ((0.1, 0.2),)))
    assert cache.record(4, 17, 0.0, active_payload)
    assert cache.set_active(4, (17,))
    assert cache.record(4, 18, 10.0, (('publication_id', 18),))
    assert cache.record(4, 19, 10.5, (('publication_id', 19),))
    assert cache.active_payloads() == (((4, 17), 0.0, active_payload),)

    recorder = HardwareFaultRecorder(pre_window_sec=10.0, clock=lambda: 11.0)
    recorder.record('servo_trajectory_publication', active_payload, monotonic_time=0.0)
    recorder.record('sample', 1, monotonic_time=10.9)
    recorder.latch_fault('fault', monotonic_time=11.0)
    assert recorder.close_window('timeout', monotonic_time=12.0)
    retained = [{'epoch': key[0], 'publication_id': key[1], 'payload': payload}
                for key, receipt, payload in cache.active_payloads() if receipt < 1.0]
    dump = json.loads(recorder.write_dump(
        tmp_path, {'retained_active_publications': retained}).read_text(encoding='utf-8'))
    assert dump['metadata']['retained_active_publications'] == [{
        'epoch': 4, 'publication_id': 17,
        'payload': [['publication_id', 17], ['points', [[0.1, 0.2]]]]}]
    event_payloads = [event['payload'] for event in dump['pre_fault']
                      if event['kind'] == 'servo_trajectory_publication']
    assert len(event_payloads) == 0


def test_active_publication_payload_companion_survives_pre_byte_eviction_once(tmp_path):
    active_payload = (
        ('hardware_epoch', 4), ('publication_id', 17),
        ('points', tuple(() for _ in range(8))))
    cache = BoundedPublicationPayloadCache(capacity=1)
    assert cache.record(4, 17, 10.0, active_payload)
    assert cache.set_active(4, (17,))
    recorder = HardwareFaultRecorder(pre_byte_capacity=1024)
    assert recorder.record(
        'servo_trajectory_publication', 10.0, active_payload, monotonic_time=10.0)
    recorder.record('diagnostic_sample', 1, monotonic_time=10.1)
    assert recorder.latch_fault('fault', monotonic_time=10.2)
    pre_events = recorder.snapshot()[0]
    assert not any(event.kind == 'servo_trajectory_publication' for event in pre_events)

    companions = missing_publication_payload_companions(
        pre_events, cache.active_payloads())
    assert companions == (((4, 17), 10.0, active_payload),)
    companion_metadata = [
        {'epoch': key[0], 'publication_id': key[1], 'receipt_monotonic': receipt,
         'payload': payload}
        for key, receipt, payload in companions]
    assert recorder.close_window('timeout', monotonic_time=10.3)
    dump = json.loads(recorder.write_dump(
        tmp_path, {'retained_active_publications': companion_metadata}).read_text(
            encoding='utf-8'))
    assert dump['metadata']['retained_active_publications'] == [{
        'epoch': 4, 'publication_id': 17, 'receipt_monotonic': 10.0,
        'payload': [['hardware_epoch', 4], ['publication_id', 17],
                    ['points', [[], [], [], [], [], [], [], []]]]}]
    assert not any(event['kind'] == 'servo_trajectory_publication'
                   for event in dump['pre_fault'])

    # If the full event did survive, its payload is not duplicated in metadata.
    recorder_with_payload = HardwareFaultRecorder(pre_byte_capacity=16 * 1024)
    assert recorder_with_payload.record(
        'servo_trajectory_publication', 10.0, active_payload, monotonic_time=10.0)
    recorder_with_payload.latch_fault('fault', monotonic_time=10.2)
    assert missing_publication_payload_companions(
        recorder_with_payload.snapshot()[0], cache.active_payloads()) == ()


def test_publication_cache_enforces_decoded_byte_budget():
    cache = BoundedPublicationPayloadCache(capacity=256, max_bytes=1024 * 1024)
    for publication_id in range(1, 11):
        details = (('publication_id', publication_id), ('points', tuple(() for _ in range(512))))
        cache.record(2, publication_id, float(publication_id), details)
    assert cache.stored_bytes <= 1024 * 1024
    assert cache.overwritten_events > 0


def test_fault_recorder_retries_close_when_nonblocking_lock_is_contended():
    recorder = HardwareFaultRecorder()
    recorder.latch_fault('root cause')
    recorder._lock.acquire()
    try:
        assert not recorder.close_window('timeout')
    finally:
        recorder._lock.release()
    assert recorder.close_window('timeout')
    assert recorder.snapshot()[-1]


def test_fault_recorder_drops_contended_events_without_waiting():
    recorder = HardwareFaultRecorder()
    recorder._lock.acquire()
    try:
        assert not recorder.record('contended')
        assert not recorder.latch_fault('contended')
        assert recorder.dropped_events == 2
    finally:
        recorder._lock.release()


def test_fault_recorder_byte_budget_marks_large_trajectory_history_incomplete(tmp_path):
    recorder = HardwareFaultRecorder(pre_byte_capacity=2048)
    points = tuple(((float(index),) * 6, (), (), (), (0, index))
                   for index in range(2048))
    details = (('publication_id', 1), ('points', points))
    assert recorder.record('servo_trajectory_publication', 1.0, details,
                           monotonic_time=1.0)
    recorder.record('small_context', 2, monotonic_time=10.9)
    recorder.latch_fault('fault', monotonic_time=11.0)
    recorder.close_window('timeout', monotonic_time=12.0)
    dump = json.loads(recorder.write_dump(tmp_path, {}).read_text(encoding='utf-8'))
    coverage = dump['pre_fault_window']
    assert coverage['pre_fault_window_byte_drops'] > 0
    assert not coverage['pre_fault_window_complete']
    assert dump['pre_fault'][-1]['kind'] == 'small_context'


def test_fault_recorder_keeps_normal_100hz_profile_for_full_pre_window(tmp_path):
    recorder = HardwareFaultRecorder()
    for tick in range(1051):
        receipt = tick / 100.0
        points = tuple((tuple(float(tick + point + joint) for joint in range(6)),
                        (), (), (), (0, point)) for point in range(17))
        publication = (('publication_id', tick + 1), ('points', points))
        assert recorder.record('servo_execution_context', tick,
                               monotonic_time=receipt)
        assert recorder.record('hardware_io_status', tick,
                               monotonic_time=receipt + 0.001)
        assert recorder.record('servo_trajectory_publication', receipt, publication,
                               monotonic_time=receipt + 0.002)
    recorder.latch_fault('fault', monotonic_time=10.51)
    recorder.close_window('timeout', monotonic_time=10.6)
    dump = json.loads(recorder.write_dump(tmp_path, {}).read_text(encoding='utf-8'))
    assert dump['pre_fault_window']['pre_fault_coverage_sec'] >= 10.49
    assert dump['pre_fault_window']['pre_fault_window_complete']
    assert recorder.pre_stored_bytes <= 64 * 1024 * 1024
    path = next(tmp_path.glob('hardware_fault_*.json'))
    assert 0 < path.stat().st_size < 64 * 1024 * 1024


def test_fault_transition_chain_survives_fault_repeat_storm(tmp_path):
    recorder = HardwareFaultRecorder(post_capacity=256)
    recorder.latch_fault('first fault')
    for index in range(100):
        recorder.latch_fault('repeat', index)
    recorder.record('stop_request', 4)
    recorder.record('stop_ack', True)
    for index in range(100):
        recorder.latch_fault('repeat after ack', index)
    recorder.close_window('timeout')
    transitions = recorder.snapshot()[3]
    assert [event.kind for event in transitions] == [
        'fault', 'stop_request', 'stop_ack', 'stop_window_end']
    dump = json.loads(recorder.write_dump(tmp_path, {}).read_text(encoding='utf-8'))
    assert dump['transition_overflow_count'] == 0
    assert dump['post_fault_window_complete']
