# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""The runtime analyzer rejects gaps while retaining useful diagnostics."""

import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.spec_from_file_location(
    'analyze_posture1d_capture', ROOT / 'tools/analyze_posture1d_capture.py')
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_gapped_capture_fails_quality_but_keeps_motion_metrics(tmp_path):
    records = [
        {'kind': 'manifest', 'scenario': 'gapped'},
        {'kind': 'preflight', 'status': 'PASS'},
        {'kind': 'capture_end', 'source_complete': True},
    ]
    for stamp, q3, z in ((0.0, -2.9, 1.0), (0.1, -2.4, 1.1), (40.0, -2.9, 1.0)):
        records.append({
            'kind': 'sample', 'elapsed_sec': stamp,
            'joint': {'stamp_sec': stamp, 'q_rad': [0.0, 0.0, q3]},
            'context': {'stamp_sec': stamp, 'target_used': True},
            'target_world_m': [3.0, 0.0, 1.0],
            'transforms_world_at_joint_stamp': {
                'monitor_control_frame': [0.0, 0.0, 0.0],
                'link3': [0.0, 0.0, z], 'link4': [0.0, 0.0, z + 0.1]},
            'transform_errors': {},
        })
    (tmp_path / 'samples.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records))

    result = analysis.analyze(tmp_path, 0.0, 40.0)

    assert not result['capture_quality_complete']
    assert result['behavior_assessment'] == 'NOT_ASSESSED'
    assert result['joint_stamp_gaps_over_50ms'] == 2
    assert 'invalid_timestamps' in result['fixed_target_cycle_metrics']['failures']
    diagnostic = result['fixed_target_cycle_diagnostic_metrics_ignoring_gaps']
    assert diagnostic['metrics']['q3_range_rad'] == 0.5
    assert diagnostic['metrics']['elbow_span_m'] > 0.09


def _fixed_capture(tmp_path, *, wait_indices=(), blank_index=None):
    records = [
        {'kind': 'manifest', 'scenario': 'fixed-control-wait'},
        {'kind': 'preflight', 'status': 'PASS'},
        {'kind': 'capture_end', 'source_complete': True},
    ]
    wait_indices = set(wait_indices)
    for index in range(1501):
        elapsed = index * 0.02
        stamp = 1.0 + elapsed
        wait = index in wait_indices
        active = index != blank_index and not wait
        records.append({
            'kind': 'sample', 'elapsed_sec': elapsed,
            'joint': {'stamp_sec': stamp, 'q_rad': [0, 0, -1, 0, 0, 0]},
            'context': {
                'stamp_sec': stamp, 'control_time_wait': wait,
                'face_tracking_published': active, 'target_used': active,
            },
            'target_world_m': [1.0, 0.0, 0.0],
            'transforms_world_at_joint_stamp': {
                'monitor_control_frame': [0.0, 0.0, 0.0],
                'link3': [0.1, 0.0, 0.0], 'link4': [0.2, 0.0, 0.0]},
            'monitor_rotation_world_at_joint_stamp_xyzw': [0, 0, 0, 1],
            'transform_errors': {},
        })
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'samples.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records))
    return analysis.analyze(tmp_path, 0.0, 30.0)


def test_fixed_hold_ignores_only_explicit_single_control_time_wait(tmp_path):
    result = _fixed_capture(tmp_path, wait_indices={750})

    assert result['capture_quality_complete']
    assert result['behavior_assessment'] == 'PASS'
    assert result['control_time_wait_samples'] == 1
    assert result['evaluated_context_samples'] == 1500
    assert result['all_rows_target_used'] is False
    assert result['all_evaluated_rows_target_used']


def test_fixed_hold_real_blank_target_remains_a_failure(tmp_path):
    result = _fixed_capture(tmp_path, blank_index=750)

    assert result['capture_quality_complete']
    assert result['control_time_wait_samples'] == 0
    assert not result['all_rows_target_used']
    assert result['behavior_assessment'] == 'FAIL'
    assert 'target_inactive' in result['p_e_w_path_metrics']['failures']


def test_fixed_hold_waits_over_50ms_leave_source_quality_unassessed(tmp_path):
    result = _fixed_capture(tmp_path, wait_indices={750, 751, 752})

    assert result['control_time_wait_samples'] == 3
    assert result['maximum_evaluated_context_gap_sec'] > 0.05
    assert not result['capture_quality_complete']
    assert result['behavior_assessment'] == 'NOT_ASSESSED'


def test_sequence_moving_phase_does_not_apply_stationary_target_gate():
    rows = []
    counters = {
        'solver_failure_count': '0', 'emergency_event_count': '0',
        'segment_rejection_count': '0',
        'feedback_motion_limit_failure_count': '0'}
    for index, target_x in enumerate((1.0, 2.0)):
        stamp = 1.0 + index * 0.02
        rows.append({
            'elapsed_sec': index * 0.02,
            'joint': {'stamp_sec': stamp,
                      'q_rad': [0.0, -0.25, -2.5, 0.0, -0.9, 0.0]},
            'context': {'stamp_sec': stamp, 'target_used': True,
                        'face_tracking_published': True,
                        'target_source_stamp_sec': stamp},
            'target_world_m': [target_x, 0.0, 0.0],
            'input_face': {'stamp_sec': stamp},
            'transforms_world_at_joint_stamp': {
                'monitor_control_frame': [0.0, 0.0, 0.0],
                'link3': [0.1, 0.0, 0.0], 'link4': [0.2, 0.0, 0.0]},
            'monitor_rotation_world_at_joint_stamp_xyzw': [0, 0, 0, 1],
            'transform_errors': {},
            'servo_diagnostics': {'statuses': {
                'face_tracking_arm/collision_aware_servo': {
                    'values': dict(counters)}}},
            'servo_status': {'code': 0},
        })
    stage = analysis._sequence_stage(
        rows, {'name': 'moving', 'kind': 'moving'}, 0.0, 0.04)
    assert stage['passed']
    assert stage['settling'] is None
    assert not stage['q3_comfort_gate_applied']
    assert stage['q3_raw_interval_excursion_samples'] == 2
    assert stage['distance_max_m'] > stage['distance_min_m']


def hold_rows(q3):
    counters = {
        'solver_failure_count': '0', 'emergency_event_count': '0',
        'segment_rejection_count': '0',
        'feedback_motion_limit_failure_count': '0', 'braking_event_count': '0'}
    rows = []
    for index in range(3):
        stamp = 1.0 + index * 0.02
        elapsed = index * 0.02
        rows.append({
            'elapsed_sec': elapsed,
            'joint': {'stamp_sec': stamp,
                      'q_rad': [0.0, -0.25, q3, 0.0, -0.9, 0.0]},
            'context': {'stamp_sec': stamp, 'target_used': True,
                        'face_tracking_published': True,
                        'target_source_stamp_sec': stamp},
            'target_world_m': [0.4, 0.0, 0.0],
            'transforms_world_at_joint_stamp': {
                'monitor_control_frame': [0.0, 0.0, 0.0],
                'link3': [0.1, 0.0, 0.0], 'link4': [0.2, 0.0, 0.0]},
            'monitor_rotation_world_at_joint_stamp_xyzw': [0, 0, 0, 1],
            'transform_errors': {},
            'servo_diagnostics': {'statuses': {
                'face_tracking_arm/collision_aware_servo': {
                    'values': dict(counters)}}},
            'servo_status': {'code': 0},
        })
    return rows


def test_sequence_hold_uses_practical_q3_tolerance_and_reports_raw_excursion():
    q3 = -1.955144977 - 0.5 * analysis.math.pi / 180.0

    stage = analysis._sequence_stage(
        hold_rows(q3), {'name': 'hold', 'kind': 'hold'}, 0.0, 0.04)

    assert stage['measurement_quality_complete']
    assert stage['behavior_assessment'] == 'PASS'
    assert stage['q3_comfort_gate_applied']
    assert stage['q3_raw_interval_excursion_samples'] == 3
    assert stage['q3_comfort_violations'] == 0
    assert abs(stage['q3_max_raw_interval_excursion_rad'] -
               0.5 * analysis.math.pi / 180.0) < 1e-9


def test_sequence_hold_reports_one_degree_comfort_violation():
    q3 = -1.955144977 - 1.1 * analysis.math.pi / 180.0

    stage = analysis._sequence_stage(
        hold_rows(q3), {'name': 'hold', 'kind': 'hold'}, 0.0, 0.04)

    assert stage['measurement_quality_complete']
    assert stage['behavior_assessment'] == 'FAIL'
    assert stage['q3_comfort_violations'] == 3


def test_sequence_reports_q3_return_into_comfort_separately():
    rows = hold_rows(-1.955144977 - 1.1 * analysis.math.pi / 180.0)
    rows[1]['joint']['q_rad'][2] = -1.955144977
    rows[2]['joint']['q_rad'][2] = -1.955144977

    stage = analysis._sequence_stage(
        rows, {'name': 'moving', 'kind': 'moving'}, 0.0, 0.04)

    assert stage['q3_raw_interval_recovery_count'] == 1
    assert stage['q3_raw_interval_excursion_samples'] == 1


def test_distance_band_uses_declared_practical_tolerance():
    assert analysis._within_distance_band([0.36, 0.44], (0.37, 0.43), 0.01)
    assert not analysis._within_distance_band([0.359], (0.37, 0.43), 0.01)


def _sequence_capture(
        tmp_path, *, leading_absence=0, gap_at=None,
        never_acquires=False, gaze_error_rad=0.0, kind='acquisition',
        target_missing_prefix=0, gaze_limit_rad=math.radians(1.0),
        explicit_wait_indices=(), context_stamp_source=None):
    """Write a source-complete synthetic sequence with controllable FACE coverage."""
    definition = {
        'analysis_stages': [{'name': 'phase', 'elapsed_sec': [0.0, 30.0],
                             'kind': kind, 'gaze_check_last_sec': 1.0}],
        'acceptance': ({'screen_normal_error_rad': gaze_limit_rad}
                       if gaze_limit_rad is not None else {}),
        'q3_comfort_interval_rad': [-1.955144977, -0.369630908],
        'q3_hold_tolerance_rad': math.radians(1.0),
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    definition_path = tmp_path / 'definition.json'
    definition_path.write_text(json.dumps(definition))
    records = [{'kind': 'sequence_definition', 'return_target_world_m': [0.4, 0.0, 0.0]}]
    counters = {
        'solver_failure_count': '0', 'emergency_event_count': '0',
        'segment_rejection_count': '0', 'feedback_motion_limit_failure_count': '0',
        'lower_rank_partial_count': '0', 'braking_event_count': '0',
    }
    wait_indices = set(explicit_wait_indices)
    for index in range(1501):
        elapsed = index * 0.02
        stamp = 1.0 + elapsed
        face = (not never_acquires and index >= leading_absence and
                index != gap_at and index not in wait_indices)
        context_stamp = 1.0 + (context_stamp_source or {}).get(index, index) * 0.02
        target_mode = 0 if face else 1
        rotation = [0.0, math.sin(gaze_error_rad / 2), 0.0,
                    math.cos(gaze_error_rad / 2)]
        records.append({
            'kind': 'sample', 'elapsed_sec': elapsed,
            'joint': {'stamp_sec': stamp,
                      'q_rad': [0.0, -0.25, -1.0, 0.0, -0.9, 0.0]},
            'context': {
                'stamp_sec': context_stamp, 'target_mode': target_mode,
                'control_time_wait': index in wait_indices,
                'target_used': face, 'face_tracking_published': face,
                'target_source_stamp_sec': context_stamp if face else 0.0,
                'face_stamp_sec': context_stamp if face else 0.0,
            },
            'target_world_m': (None if index < target_missing_prefix
                               else [1.0, 0.0, 0.0]),
            'input_face': ({'stamp_sec': stamp} if face else None),
            'transforms_world_at_joint_stamp': {
                'monitor_control_frame': [0.0, 0.0, 0.0],
                'link3': [0.1, 0.0, 0.0], 'link4': [0.2, 0.0, 0.0]},
            'monitor_rotation_world_at_joint_stamp_xyzw': rotation,
            'transform_errors': {},
            'servo_diagnostics': {'statuses': {
                'face_tracking_arm/collision_aware_servo': {'values': counters}}},
            'servo_status': {'code': 0, 'message': 'No warnings'},
        })
    capture = tmp_path / 'capture'
    capture.mkdir()
    (capture / 'samples.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in records))
    return analysis.analyze_sequence(capture, definition_path)


def test_sequence_acquisition_allows_only_initial_face_absence_prefix(tmp_path):
    result = _sequence_capture(tmp_path, leading_absence=5)

    stage = result['stages'][0]
    assert result['source_quality_complete']
    assert result['capture_passed']
    assert stage['leading_face_absence_samples'] == 5
    assert stage['first_valid_face_elapsed_sec'] == 0.1
    assert stage['continuous_FACE_target_used_samples'] == 1496


def test_sequence_acquisition_prefix_without_target_keeps_row_alignment(tmp_path):
    result = _sequence_capture(
        tmp_path, leading_absence=5, target_missing_prefix=5)

    stage = result['stages'][0]
    assert result['source_quality_complete']
    assert result['capture_passed']
    assert stage['sample_count'] == 1501
    assert stage['metrics_sample_count'] == 1496
    assert stage['continuous_FACE_target_used_samples'] == 1496
    assert stage['safety_counter_values']['solver_failure_count']['first'] == 0
    assert stage['safety_counter_values']['solver_failure_count']['last'] == 0


def test_sequence_acquisition_gap_after_first_face_does_not_get_hidden(tmp_path):
    result = _sequence_capture(tmp_path, leading_absence=5, gap_at=120)

    stage = result['stages'][0]
    assert not result['source_quality_complete']
    assert not result['capture_passed']
    assert stage['leading_face_absence_samples'] == 5
    assert stage['behavior_assessment'] == 'NOT_ASSESSED'


def test_sequence_acquisition_that_never_gets_face_fails(tmp_path):
    result = _sequence_capture(tmp_path, never_acquires=True)

    stage = result['stages'][0]
    assert result['source_quality_complete']
    assert not result['capture_passed']
    assert stage['face_acquired'] is False
    assert stage['behavior_assessment'] == 'FAIL'


def test_sequence_gaze_gate_uses_declared_one_degree_threshold(tmp_path):
    accepted = _sequence_capture(
        tmp_path / 'accepted', gaze_error_rad=math.radians(0.75), kind='hold')
    assert accepted['capture_passed']
    failed = _sequence_capture(
        tmp_path / 'failed', gaze_error_rad=math.radians(1.01), kind='hold')
    assert not failed['capture_passed']
    assert failed['stages'][0]['behavior_assessment'] == 'FAIL'


def test_sequence_gaze_default_remains_point_one_degree_rad(tmp_path):
    accepted = _sequence_capture(
        tmp_path / 'accepted', gaze_error_rad=0.009,
        kind='hold', gaze_limit_rad=None)
    assert accepted['capture_passed']
    assert accepted['stages'][0]['screen_normal_error_gate_limit_rad'] == 0.01
    failed = _sequence_capture(
        tmp_path / 'failed', gaze_error_rad=0.011,
        kind='hold', gaze_limit_rad=None)
    assert not failed['capture_passed']
    assert failed['stages'][0]['screen_normal_error_gate_limit_rad'] == 0.01


def test_sequence_hold_does_not_allow_face_absence_prefix_or_gap(tmp_path):
    result = _sequence_capture(
        tmp_path, leading_absence=5, gap_at=120, kind='hold')

    stage = result['stages'][0]
    assert not result['source_quality_complete']
    assert not result['capture_passed']
    assert stage['leading_face_absence_samples'] == 0
    assert stage['behavior_assessment'] == 'NOT_ASSESSED'


def test_sequence_hold_ignores_only_explicit_single_control_time_wait(tmp_path):
    result = _sequence_capture(
        tmp_path, kind='hold', explicit_wait_indices={120})

    stage = result['stages'][0]
    assert result['source_quality_complete']
    assert result['capture_passed']
    assert stage['control_time_wait_samples'] == 1
    assert stage['evaluated_context_samples'] == 1500
    assert abs(stage['maximum_evaluated_context_gap_sec'] - 0.04) < 1e-9
    assert stage['screen_normal_error_gate_samples'] == 1500


def test_sequence_hold_blank_context_without_wait_marker_is_not_assessed(tmp_path):
    result = _sequence_capture(tmp_path, gap_at=120, kind='hold')

    stage = result['stages'][0]
    assert stage['control_time_wait_samples'] == 0
    assert not result['source_quality_complete']
    assert not result['capture_passed']
    assert stage['behavior_assessment'] == 'NOT_ASSESSED'


def test_sequence_hold_run_of_waits_over_50ms_is_not_assessed(tmp_path):
    result = _sequence_capture(
        tmp_path, kind='hold', explicit_wait_indices={120, 121})

    stage = result['stages'][0]
    assert stage['control_time_wait_samples'] == 2
    assert stage['maximum_evaluated_context_gap_sec'] > 0.05
    assert not result['source_quality_complete']
    assert not result['capture_passed']


def test_sequence_hold_leading_waits_over_50ms_are_not_assessed(tmp_path):
    result = _sequence_capture(
        tmp_path, kind='hold', explicit_wait_indices={0, 1, 2})

    stage = result['stages'][0]
    assert stage['control_time_wait_samples'] == 3
    assert stage['maximum_evaluated_context_edge_gap_sec'] > 0.05
    assert not result['source_quality_complete']
    assert not result['capture_passed']


def test_sequence_hold_trailing_waits_over_50ms_are_not_assessed(tmp_path):
    result = _sequence_capture(
        tmp_path, kind='hold', explicit_wait_indices={1498, 1499, 1500})

    stage = result['stages'][0]
    assert stage['control_time_wait_samples'] == 3
    assert stage['maximum_evaluated_context_edge_gap_sec'] > 0.05
    assert not result['source_quality_complete']
    assert not result['capture_passed']


def test_sequence_hold_repeated_nearest_context_association_is_valid(tmp_path):
    result = _sequence_capture(
        tmp_path, kind='hold', context_stamp_source={120: 119})

    stage = result['stages'][0]
    assert result['source_quality_complete']
    assert result['capture_passed']
    assert stage['evaluated_context_samples'] == 1501
    assert abs(stage['maximum_evaluated_context_gap_sec'] - 0.04) < 1e-9


def test_sequence_hold_source_gap_after_repeated_association_is_not_assessed(
        tmp_path):
    result = _sequence_capture(
        tmp_path, kind='hold', context_stamp_source={
            120: 119, 121: 119})

    stage = result['stages'][0]
    assert stage['maximum_evaluated_context_gap_sec'] > 0.05
    assert not result['source_quality_complete']
    assert not result['capture_passed']


def test_sequence_acquisition_requires_full_final_gaze_window_after_face(tmp_path):
    result = _sequence_capture(tmp_path, leading_absence=1475)

    stage = result['stages'][0]
    assert stage['face_acquired']
    assert stage['measurement_quality_complete']
    assert not stage['screen_normal_error_gate_coverage_complete']
    assert not result['capture_passed']
