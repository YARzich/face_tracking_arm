#!/usr/bin/env python3
# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT
"""Analyze a POSTURE-1D observer capture without editing its raw streams."""

import argparse
import json
import math
from pathlib import Path

from posture1d_metrics import evaluate_fixed_target, Sample
from posture_settling_metrics import evaluate_settling, SettlingSample


def analyze(path: Path, start: float, end: float) -> dict:
    records = [json.loads(line) for line in (path / 'samples.jsonl').read_text().splitlines()]
    manifest = next((r for r in records if r.get('kind') == 'manifest'), {})
    preflight = next((r for r in records if r.get('kind') == 'preflight'), {})
    capture_end = next((r for r in reversed(records) if r.get('kind') == 'capture_end'), {})
    rows = [r for r in records if r.get('kind') == 'sample' and
            start <= r.get('elapsed_sec', -math.inf) <= end]
    rows.sort(key=lambda r: r['joint']['stamp_sec'])
    stamps = [r['joint']['stamp_sec'] for r in rows]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    evaluated_rows = [r for r in rows
                      if not _control_time_wait(r.get('context'))]
    context_skews = [abs(r['joint']['stamp_sec'] - r['context']['stamp_sec'])
                     for r in rows if r.get('context')]
    face_skews = [abs(r['joint']['stamp_sec'] - r['input_face']['stamp_sec'])
                  for r in rows if r.get('input_face')]
    gaze_errors = []
    source_complete = bool(capture_end.get('source_complete'))
    evaluated_context_stamps = [r['context']['stamp_sec'] for r in evaluated_rows
                                if r.get('context')]
    context_stamps_monotonic = all(
        later >= earlier for earlier, later in zip(
            evaluated_context_stamps, evaluated_context_stamps[1:]))
    distinct_context_stamps = [
        stamp for index, stamp in enumerate(evaluated_context_stamps)
        if index == 0 or stamp != evaluated_context_stamps[index - 1]]
    evaluated_context_gaps = [later - earlier for earlier, later
                              in zip(distinct_context_stamps,
                                     distinct_context_stamps[1:])]
    source_time_offset = (stamps[0] - rows[0]['elapsed_sec']) if rows else 0.0
    context_edge_gaps = ([
        max(0.0, evaluated_context_stamps[0] -
            (source_time_offset + start)),
        max(0.0, (source_time_offset + end) - evaluated_context_stamps[-1]),
    ] if evaluated_context_stamps else [math.inf, math.inf])
    strict_sampling = bool(
        len(stamps) > 1 and len(set(stamps)) == len(stamps) and
        (stamps[-1] - stamps[0]) >= end - start - 0.05 and
        max(gaps, default=math.inf) <= 0.05 + 1e-9 and
        all(not r.get('transform_errors') for r in rows) and
        len(context_skews) == len(rows) and
        max(context_skews, default=math.inf) <= 0.05 + 1e-9 and
        len(evaluated_rows) > 1 and
        len(evaluated_context_stamps) == len(evaluated_rows) and
        context_stamps_monotonic and
        max(evaluated_context_gaps, default=math.inf) <= 0.05 + 1e-9 and
        max(context_edge_gaps, default=math.inf) <= 0.05 + 1e-9)
    face_positions = [r['input_face']['world_position'] for r in rows
                      if r.get('input_face') and r['input_face'].get('world_position')]
    target_range = _range(face_positions)
    complete = bool(preflight.get('status') == 'PASS' and source_complete and
                    strict_sampling)
    metric_samples = []
    settling_samples = []
    for row in rows:
        state = row.get('joint') or {}
        world = row.get('transforms_world_at_joint_stamp') or {}
        if 'link3' not in world or 'link4' not in world or 'monitor_control_frame' not in world:
            continue
        target = row.get('target_world_m')
        if target is None and row.get('input_face'):
            target = row['input_face'].get('world_position')
        if target is None and row.get('context'):
            target = row['context'].get('face_position')
        if target is None:
            continue
        rotation = row.get('monitor_rotation_world_at_joint_stamp_xyzw')
        if rotation is not None:
            x, y, z, w = rotation
            normal = (1 - 2 * (y*y + z*z), 2 * (x*y + z*w),
                      2 * (x*z - y*w))
            delta = tuple(target[i] - world['monitor_control_frame'][i]
                          for i in range(3))
            length = math.sqrt(sum(value * value for value in delta))
            if length > 1e-9:
                dot = sum(normal[i] * delta[i] / length for i in range(3))
                gaze_errors.append(math.acos(max(-1.0, min(1.0, dot))))
        metric_samples.append(Sample(state['stamp_sec'], state['q_rad'][2],
                                     tuple(world['link3'])))
        settling_samples.append(SettlingSample(
            state['stamp_sec'], tuple(world['monitor_control_frame']),
            tuple(world['link3']), tuple(world['link4']), tuple(target),
            math.dist(world['monitor_control_frame'], target),
            bool(row.get('context') and (
                row['context'].get('target_used') or
                _control_time_wait(row.get('context'))))))
    cycle = evaluate_fixed_target(metric_samples,
                                  minimum_coverage_sec=end - start - 0.05,
                                  maximum_gap_sec=0.05)
    cycle_diagnostic = evaluate_fixed_target(
        metric_samples, minimum_coverage_sec=end - start - 0.05,
        maximum_gap_sec=max(1.0, end - start + 1.0))
    settling = evaluate_settling(
        settling_samples, minimum_coverage_sec=end - start - 0.05,
        maximum_sample_gap_sec=0.05)
    settling_diagnostic = evaluate_settling(
        settling_samples, minimum_coverage_sec=end - start - 0.05,
        maximum_sample_gap_sec=max(1.0, end - start + 1.0))
    result = {
        'scenario': manifest.get('scenario'), 'preflight': preflight,
        'capture_end': capture_end, 'window_elapsed_sec': [start, end],
        'sample_count': len(rows), 'unique_joint_stamps': len(set(stamps)),
        'window_coverage_sec': stamps[-1] - stamps[0] if stamps else 0.0,
        'maximum_joint_stamp_gap_sec': max(gaps, default=None),
        'joint_stamp_gaps_over_50ms': sum(g > 0.050000001 for g in gaps),
        'control_time_wait_samples': sum(
            _control_time_wait(r.get('context')) for r in rows),
        'evaluated_context_samples': len(evaluated_rows),
        'maximum_evaluated_context_gap_sec': max(
            evaluated_context_gaps, default=None),
        'maximum_evaluated_context_edge_gap_sec': max(
            context_edge_gaps, default=None),
        'maximum_context_joint_skew_sec': max(context_skews, default=None),
        'rows_with_context_skew_over_50ms': sum(
            skew > 0.050000001 for skew in context_skews),
        'maximum_face_joint_skew_sec': max(face_skews, default=None),
        'rows_with_face_skew_over_50ms': sum(
            skew > 0.050000001 for skew in face_skews),
        'rows_with_tf_errors': sum(bool(r.get('transform_errors')) for r in rows),
        'target_world_range_m': target_range,
        'screen_normal_error_rad': {
            'count': len(gaze_errors), 'max': max(gaze_errors, default=None),
            'p95': sorted(gaze_errors)[int(0.95 * (len(gaze_errors) - 1))]
            if gaze_errors else None,
        },
        'all_rows_face_tracking_published': bool(rows) and all(
            r.get('context', {}).get('face_tracking_published', False) for r in rows),
        'all_evaluated_rows_face_tracking_published': bool(evaluated_rows) and all(
            r.get('context', {}).get('face_tracking_published', False)
            for r in evaluated_rows),
        'all_rows_target_used': bool(rows) and all(
            r.get('context', {}).get('target_used', False) for r in rows),
        'all_evaluated_rows_target_used': bool(evaluated_rows) and all(
            r.get('context', {}).get('target_used', False) for r in evaluated_rows),
        'capture_quality_complete': complete,
        'behavior_assessment': (
            'NOT_ASSESSED' if not complete else
            'PASS' if cycle.passed and settling.passed else 'FAIL'),
        'fixed_target_cycle_metrics': {
            'passed': cycle.passed, 'failures': cycle.failures,
            'metrics': cycle.metrics,
        },
        'fixed_target_cycle_diagnostic_metrics_ignoring_gaps': {
            'passed': cycle_diagnostic.passed,
            'failures': cycle_diagnostic.failures,
            'metrics': cycle_diagnostic.metrics,
        },
        'p_e_w_path_metrics': {
            'passed': settling.passed, 'failures': settling.failures,
            'metrics': settling.metrics,
        },
        'p_e_w_diagnostic_metrics_ignoring_gaps': {
            'passed': settling_diagnostic.passed,
            'failures': settling_diagnostic.failures,
            'metrics': settling_diagnostic.metrics,
        },
    }
    cleaned = _clean(result)
    (path / 'analysis.json').write_text(json.dumps(cleaned, indent=2, allow_nan=False) + '\n')
    return cleaned


def analyze_sequence(path: Path, definition_path: Path) -> dict:
    """Evaluate moving-target phases and fixed-target hold phases separately."""
    records = [json.loads(line) for line in
               (path / 'samples.jsonl').read_text().splitlines()]
    definition = json.loads(definition_path.read_text())
    sequence_record = next((row for row in records
                            if row.get('kind') == 'sequence_definition'), {})
    return_target = sequence_record.get('return_target_world_m')
    if return_target is None:
        raise ValueError('capture has no frozen sequence return target')
    rows = [row for row in records if row.get('kind') == 'sample']
    rows.sort(key=lambda row: row['joint']['stamp_sec'])
    stages = []
    for stage in definition['analysis_stages']:
        start, end = stage['elapsed_sec']
        selected = [row for row in rows
                    if start <= row['elapsed_sec'] <= end]
        stage_result = _sequence_stage(selected, stage, start, end, definition)
        stages.append(stage_result)
    start = definition['analysis_stages'][0]['elapsed_sec'][0]
    end = definition['analysis_stages'][-1]['elapsed_sec'][1]
    overall = [row for row in rows
               if start <= row['elapsed_sec'] <= end]
    stamps = [row['joint']['stamp_sec'] for row in overall]
    gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    contexts = [row.get('context') for row in overall]
    context_skews = [abs(row['joint']['stamp_sec'] - context['stamp_sec'])
                     for row, context in zip(overall, contexts) if context]
    evaluated_rows = [(index, row) for index, row in enumerate(overall)
                      if not _control_time_wait(row.get('context'))]
    first_face_index = next((index for index, row in evaluated_rows
                             if _valid_face_command(row.get('context'))), None)
    post_acquisition = [(index, row) for index, row in evaluated_rows
                        if first_face_index is not None and
                        index >= first_face_index]
    target_source_skews = [
        abs(row['joint']['stamp_sec'] -
            row['context']['target_source_stamp_sec'])
        for _, row in post_acquisition if row.get('context') and
        (_valid_face_command(row['context']) or
         _has_source_stamped_face_reference(row.get('context'))) and
        row['context'].get('target_source_stamp_sec', 0.0) > 0.0]
    face_used = bool(overall) and all(
        _valid_face_command(row.get('context')) for row in overall)
    face_used_after_acquisition = bool(post_acquisition) and all(
        _valid_face_command(row.get('context'))
        for _, row in post_acquisition)
    evaluated_context_stamps = [row['context']['stamp_sec']
                                for _, row in evaluated_rows
                                if row.get('context')]
    evaluated_context_monotonic = all(
        later >= earlier for earlier, later in zip(
            evaluated_context_stamps, evaluated_context_stamps[1:]))
    distinct_evaluated_context_stamps = [
        stamp for index, stamp in enumerate(evaluated_context_stamps)
        if index == 0 or stamp != evaluated_context_stamps[index - 1]]
    evaluated_context_gaps = [
        later - earlier for earlier, later in zip(
            distinct_evaluated_context_stamps,
            distinct_evaluated_context_stamps[1:])]
    scenario_stamp_offset = (overall[0]['joint']['stamp_sec'] -
                             overall[0]['elapsed_sec']) if overall else 0.0
    evaluated_context_edge_gaps = ([
        max(0.0, evaluated_context_stamps[0] -
            (scenario_stamp_offset + start)),
        max(0.0, (scenario_stamp_offset + end) -
            evaluated_context_stamps[-1]),
    ] if evaluated_context_stamps else [math.inf, math.inf])
    source_quality_complete = bool(
        len(stamps) > 1 and len(set(stamps)) == len(stamps) and
        len(evaluated_rows) > 1 and
        len(context_skews) == len(overall) and
        len(target_source_skews) == len(post_acquisition) and
        stamps[-1] - stamps[0] >= end - start - 0.05 and
        max(gaps, default=math.inf) <= 0.05 + 1e-9 and
        evaluated_context_monotonic and
        all(gap > 0.0 for gap in evaluated_context_gaps) and
        max(evaluated_context_gaps, default=math.inf) <= 0.05 + 1e-9 and
        max(evaluated_context_edge_gaps, default=math.inf) <= 0.05 + 1e-9 and
        max(context_skews, default=math.inf) <= 0.05 + 1e-9 and
        max(target_source_skews, default=(0.0 if not post_acquisition else math.inf))
        <= 0.05 + 1e-9 and
        not any(row.get('transform_errors') for row in overall))
    behavioral_assessment = (
        'NOT_ASSESSED' if not source_quality_complete else
        'PASS' if all(stage['behavior_assessment'] == 'PASS'
                      for stage in stages) else 'FAIL')
    result = {
        'scenario': 'far_lateral_up_return_near',
        'return_target_world_m_frozen_at_preflight': return_target,
        'window_elapsed_sec': [start, end],
        'sample_count': len(overall),
        'coverage_sec': stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0,
        'maximum_joint_gap_sec': max(gaps, default=None),
        'maximum_context_joint_skew_sec': max((
            abs(row['joint']['stamp_sec'] - row['context']['stamp_sec'])
            for row in overall if row.get('context')), default=None),
        'maximum_target_source_joint_skew_sec': max(target_source_skews,
                                                    default=None),
        'control_time_wait_samples': sum(
            _control_time_wait(row.get('context')) for row in overall),
        'evaluated_context_samples': len(evaluated_rows),
        'maximum_evaluated_context_gap_sec': max(
            evaluated_context_gaps, default=None),
        'maximum_evaluated_context_edge_gap_sec': max(
            evaluated_context_edge_gaps, default=None),
        'source_quality_complete': source_quality_complete,
        'behavior_assessment': behavioral_assessment,
        'rows_with_tf_errors': sum(bool(row.get('transform_errors'))
                                   for row in overall),
        'all_samples_face_and_target_used': face_used,
        'all_evaluated_samples_face_and_target_used': (
            bool(evaluated_rows) and all(
                _valid_face_command(row.get('context'))
                for _, row in evaluated_rows)),
        'face_continuous_from_first_acquisition': face_used_after_acquisition,
        'leading_face_absence_samples': (
            first_face_index if first_face_index is not None else len(overall)),
        'first_valid_face_elapsed_sec': (
            overall[first_face_index]['elapsed_sec']
            if first_face_index is not None else None),
        'stages': stages,
        'capture_passed': bool(
            source_quality_complete and face_used_after_acquisition and
            all(stage['passed'] for stage in stages)),
    }
    cleaned = _clean(result)
    (path / 'sequence_analysis.json').write_text(
        json.dumps(cleaned, indent=2, allow_nan=False) + '\n')
    return cleaned


def _valid_face_command(context):
    """Return whether context proves a source-stamped FACE command was used."""
    if not context or not context.get('face_tracking_published') or not context.get('target_used'):
        return False
    target_mode = context.get('target_mode')
    if target_mode is not None and target_mode != 0:
        return False
    try:
        return float(context.get('target_source_stamp_sec', 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _has_source_stamped_face_reference(context):
    """Whether the live FACE target was available to the control tick."""
    if not context or not context.get('target_available') or not context.get('target_used'):
        return False
    if context.get('target_mode') != 0:
        return False
    try:
        return float(context.get('target_source_stamp_sec', 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _control_time_wait(context):
    """Return whether the producer explicitly marked a non-evaluated wait tick."""
    return bool(context and context.get('control_time_wait', False))


def _sequence_stage(
        rows, stage, start, end, definition=None):
    """Measure per-row gaze/range and apply P/E/W settling only to holds."""
    samples = []
    gaze_errors = []
    gaze_rows = []
    q_values = []
    joint_violations = []
    q3_violations = []
    q3_excursions = []
    q3_outside_flags = []
    cycle_samples = []
    command_rows = 0
    counter_series = {name: [] for name in (
        'solver_failure_count', 'emergency_event_count',
        'segment_rejection_count', 'feedback_motion_limit_failure_count',
        'lower_rank_partial_count', 'braking_event_count',
        'controlled_rearm_count')}
    joint_limits = ((-2 * math.pi, 2 * math.pi), (-2.05949, 2.0944),
                    (-3.92699, 0.191986), (-2 * math.pi, 2 * math.pi),
                    (-1.69297, math.pi), (-2 * math.pi, 2 * math.pi))
    acceptance = (definition or {}).get('acceptance', {})
    q3_comfort = tuple((definition or {}).get(
        'q3_comfort_interval_rad', (-1.955144977, -0.369630908)))
    q3_tolerance = float((definition or {}).get(
        'q3_hold_tolerance_rad', math.radians(1.0)))
    distance_band = tuple(acceptance.get('distance_band_m', (0.37, 0.43)))
    distance_tolerance = float(acceptance.get('distance_band_tolerance_m', 0.01))
    gaze_tolerance = float(acceptance.get('screen_normal_error_rad', 0.01))
    wait_indices = {index for index, row in enumerate(rows)
                    if _control_time_wait(row.get('context'))}
    evaluated_rows = [(index, row) for index, row in enumerate(rows)
                      if index not in wait_indices]
    first_face_index = next((index for index, row in evaluated_rows
                             if _valid_face_command(row.get('context'))), None)
    is_acquisition = stage['kind'] == 'acquisition'
    leading_absence_count = (
        first_face_index if first_face_index is not None else len(rows)
    ) if is_acquisition else 0
    required_source_rows = [
        row for index, row in evaluated_rows
        if (not is_acquisition or
            (first_face_index is not None and index >= first_face_index))]
    behavior_rows = required_source_rows
    servo_status_codes = []
    servo_status_messages = []
    sample_entries = []
    for row_index, row in enumerate(rows):
        world = row.get('transforms_world_at_joint_stamp') or {}
        target = row.get('target_world_m')
        rotation = row.get('monitor_rotation_world_at_joint_stamp_xyzw')
        context = row.get('context') or {}
        q_values.append(row['joint']['q_rad'][2])
        q = row['joint']['q_rad']
        if any(not lower <= value <= upper
               for value, (lower, upper) in zip(q, joint_limits)):
            joint_violations.append(row['elapsed_sec'])
        if q[2] < q3_comfort[0] or q[2] > q3_comfort[1]:
            q3_violations.append(row['elapsed_sec'])
        q3_excursions.append(max(q3_comfort[0] - q[2],
                                 q[2] - q3_comfort[1], 0.0))
        q3_outside_flags.append(q[2] < q3_comfort[0] or q[2] > q3_comfort[1])
        if 'link3' in world:
            cycle_samples.append(Sample(
                row['joint']['stamp_sec'], q[2], tuple(world['link3'])))
        if (_valid_face_command(context) and
                not _control_time_wait(context)):
            command_rows += 1
        statuses = (row.get('servo_diagnostics') or {}).get('statuses') or {}
        values = next(iter(statuses.values()), {}).get('values', {})
        for name in counter_series:
            try:
                counter_series[name].append(int(values[name]))
            except (KeyError, TypeError, ValueError):
                pass
        try:
            servo_status_codes.append(int(row['servo_status']['code']))
            servo_status_messages.append(str(row['servo_status']['message']))
        except (KeyError, TypeError, ValueError):
            pass
        if target is None or 'monitor_control_frame' not in world:
            continue
        distance = math.dist(world['monitor_control_frame'], target)
        sample = SettlingSample(
            row['joint']['stamp_sec'], tuple(world['monitor_control_frame']),
            tuple(world.get('link3', ())), tuple(world.get('link4', ())),
            tuple(target), distance, bool(
                context.get('target_used') or
                _control_time_wait(row.get('context'))))
        samples.append(sample)
        sample_entries.append((row_index, sample))
        if rotation is not None and distance > 1e-9:
            x, y, z, w = rotation
            normal = (1 - 2 * (y*y + z*z), 2 * (x*y + z*w),
                      2 * (x*z - y*w))
            delta = tuple(target[i] - world['monitor_control_frame'][i]
                          for i in range(3))
            dot = sum(normal[i] * delta[i] / distance for i in range(3))
            gaze_errors.append(math.acos(max(-1.0, min(1.0, dot))))
            gaze_rows.append(row)
    max_gaze = max(gaze_errors, default=None)
    distances = [sample.distance_m for sample in samples]
    stamps = [row['joint']['stamp_sec'] for row in rows]
    stage_gaps = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
    context_skews = [abs(row['joint']['stamp_sec'] - row['context']['stamp_sec'])
                     for row in rows if row.get('context')]
    evaluated_context_stamps = [row['context']['stamp_sec']
                                for _, row in evaluated_rows
                                if row.get('context')]
    evaluated_context_monotonic = all(
        later >= earlier for earlier, later in zip(
            evaluated_context_stamps, evaluated_context_stamps[1:]))
    distinct_evaluated_context_stamps = [
        stamp for index, stamp in enumerate(evaluated_context_stamps)
        if index == 0 or stamp != evaluated_context_stamps[index - 1]]
    evaluated_context_gaps = [
        later - earlier for earlier, later in zip(
            distinct_evaluated_context_stamps,
            distinct_evaluated_context_stamps[1:])]
    stage_stamp_offset = (rows[0]['joint']['stamp_sec'] -
                          rows[0]['elapsed_sec']) if rows else 0.0
    evaluated_context_edge_gaps = ([
        max(0.0, evaluated_context_stamps[0] -
            (stage_stamp_offset + start)),
        max(0.0, (stage_stamp_offset + end) -
            evaluated_context_stamps[-1]),
    ] if evaluated_context_stamps else [math.inf, math.inf])
    face_skews = [abs(row['joint']['stamp_sec'] - row['input_face']['stamp_sec'])
                  for row in rows if row.get('input_face')]
    target_source_skews = [
        abs(row['joint']['stamp_sec'] -
            row['context']['target_source_stamp_sec'])
        for row in required_source_rows if row.get('context') and
        (_valid_face_command(row['context']) or
         _has_source_stamped_face_reference(row['context'])) and
        row['context'].get('target_source_stamp_sec', 0.0) > 0.0]
    target_indices = {index for index, _ in sample_entries}
    allowed_missing_target_indices = (
        set(range(leading_absence_count)) | wait_indices)
    target_coverage_complete = all(
        index in target_indices or index in allowed_missing_target_indices
        for index in range(len(rows)))
    geometry_complete = all(
        {'monitor_control_frame', 'link3', 'link4'}.issubset(
            row.get('transforms_world_at_joint_stamp') or {})
        for row in rows)
    strict_source = bool(
        target_coverage_complete and geometry_complete and
        len(context_skews) == len(rows) and
        len(target_source_skews) == len(required_source_rows) and
        len(evaluated_rows) > 1 and
        evaluated_context_monotonic and
        len(set(stamps)) == len(stamps) and len(stamps) > 1 and
        stamps[-1] - stamps[0] >= end - start - 0.05 and
        max(stage_gaps, default=math.inf) <= 0.05 + 1e-9 and
        all(gap > 0.0 for gap in evaluated_context_gaps) and
        max(context_skews, default=math.inf) <= 0.05 + 1e-9 and
        max(evaluated_context_gaps, default=math.inf) <= 0.05 + 1e-9 and
        max(evaluated_context_edge_gaps, default=math.inf) <= 0.05 + 1e-9 and
        max(target_source_skews,
            default=(0.0 if not required_source_rows else math.inf)) <= 0.05 + 1e-9 and
        all(not row.get('transform_errors') for row in rows))
    gaze_gate = stage['kind'] in ('hold', 'acquisition', 'recovery')
    face_gaze = [(row, error) for row, error in zip(gaze_rows, gaze_errors)
                 if _valid_face_command(row.get('context')) and
                 not _control_time_wait(row.get('context'))]
    if stage['kind'] == 'hold':
        gaze_gate_rows = face_gaze
    elif gaze_gate:
        tail_sec = float(stage.get('gaze_check_last_sec', 1.0))
        cutoff = end - tail_sec
        gaze_gate_rows = [(row, error) for row, error in face_gaze
                          if row['elapsed_sec'] >= cutoff]
    else:
        gaze_gate_rows = []
    gaze_gate_errors = [error for _, error in gaze_gate_rows]
    gaze_gate_max = max(gaze_gate_errors, default=None)
    gaze_gate_stamps = [row['joint']['stamp_sec'] for row, _ in gaze_gate_rows]
    gaze_gate_gaps = [later - earlier for earlier, later
                      in zip(gaze_gate_stamps, gaze_gate_stamps[1:])]
    if stage['kind'] == 'hold':
        gaze_gate_coverage_complete = bool(
            len(gaze_gate_stamps) == len(evaluated_rows) and
            len(gaze_gate_stamps) > 1 and
            max(gaze_gate_gaps, default=math.inf) <= 0.05 + 1e-9)
        gaze_gate_coverage_sec = (
            gaze_gate_stamps[-1] - gaze_gate_stamps[0]
            if len(gaze_gate_stamps) > 1 else 0.0)
    elif gaze_gate:
        tail_sec = float(stage.get('gaze_check_last_sec', 1.0))
        gaze_gate_coverage_sec = (
            gaze_gate_stamps[-1] - gaze_gate_stamps[0]
            if len(gaze_gate_stamps) > 1 else 0.0)
        gaze_gate_coverage_complete = bool(
            len(gaze_gate_stamps) > 1 and
            gaze_gate_coverage_sec >= tail_sec - 0.05 - 1e-9 and
            max(gaze_gate_gaps, default=math.inf) <= 0.05 + 1e-9)
    else:
        gaze_gate_coverage_sec = 0.0
        gaze_gate_coverage_complete = True
    behavior_indices = {index for index, _ in evaluated_rows
                        if (not is_acquisition or
                            (first_face_index is not None and
                             index >= first_face_index))}
    behavior_samples = [sample for index, sample in sample_entries
                        if index in behavior_indices]
    q3_gate_applied = stage['kind'] == 'hold'
    q3_practical_violations = sum(
        excursion > q3_tolerance for excursion in q3_excursions
    ) if q3_gate_applied else 0
    safe_statuses = {'Safe primary orientation command only'}
    counter_names = (
        'solver_failure_count', 'emergency_event_count',
        'segment_rejection_count', 'feedback_motion_limit_failure_count')
    status_rows_valid = len(servo_status_codes) == len(rows) and all(
        code == 0 or (code == 4 and message in safe_statuses)
        for code, message in zip(servo_status_codes, servo_status_messages))
    counter_policy_passed = all(
        len(counter_series[name]) == len(rows) and
        max(counter_series[name]) == min(counter_series[name])
        for name in counter_names)
    behavior_passed = bool(
        samples and (not gaze_gate or
                     (gaze_gate_coverage_complete and gaze_gate_max is not None and
                      gaze_gate_max <= gaze_tolerance)) and
        (not is_acquisition or first_face_index is not None) and
        len(behavior_samples) == len(behavior_rows) and
        all(sample.target_active for sample in behavior_samples) and
        len(servo_status_codes) == len(rows) and
        status_rows_valid and
        not joint_violations and not q3_practical_violations and
        command_rows == len(behavior_rows) and
        counter_policy_passed)
    settle = None
    duration = end - start
    q3_cycle = evaluate_fixed_target(
        cycle_samples, minimum_coverage_sec=max(0.0, duration - 0.05),
        maximum_gap_sec=max(1.0, duration + 1.0))
    if stage['kind'] == 'hold':
        settle = evaluate_settling(
            samples, minimum_coverage_sec=duration - 0.05,
            path_window_sec=min(10.0, duration - 0.05),
            maximum_sample_gap_sec=0.05)
        behavior_passed = behavior_passed and settle.passed
        behavior_passed = behavior_passed and not any(
            failure in q3_cycle.failures
            for failure in ('repeated_q3_reversals', 'elbow_motion'))
    distance_gate = None
    if stage.get('require_distance_band'):
        distance_gate = _within_distance_band(
            distances, distance_band, distance_tolerance)
    elif stage['kind'] == 'recovery':
        tail_sec = float(stage.get('distance_check_last_sec', 1.0))
        tail = [sample.distance_m for index, sample in sample_entries
                for row in (rows[index],)
                if row['elapsed_sec'] >= end - tail_sec]
        distance_gate = _within_distance_band(
            tail, distance_band, distance_tolerance)
    if distance_gate is not None:
        behavior_passed = behavior_passed and distance_gate
    passed = strict_source and behavior_passed
    behavior_assessment = (
        'NOT_ASSESSED' if not strict_source else
        'PASS' if behavior_passed else 'FAIL')
    return {
        'name': stage['name'], 'kind': stage['kind'],
        'window_elapsed_sec': [start, end], 'sample_count': len(rows),
        'metrics_sample_count': len(samples),
        'control_time_wait_samples': len(wait_indices),
        'evaluated_context_samples': len(evaluated_rows),
        'maximum_evaluated_context_gap_sec': max(
            evaluated_context_gaps, default=None),
        'maximum_evaluated_context_edge_gap_sec': max(
            evaluated_context_edge_gaps, default=None),
        'distance_min_m': min(distances, default=None),
        'distance_max_m': max(distances, default=None),
        'distance_band_gate_passed': distance_gate,
        'distance_band_tolerance_m': distance_tolerance,
        'screen_normal_error_max_rad': max_gaze,
        'screen_normal_error_gate_max_rad': gaze_gate_max,
        'screen_normal_error_gate_limit_rad': gaze_tolerance,
        'screen_normal_error_gate_samples': len(gaze_gate_rows),
        'screen_normal_error_gate_coverage_sec': gaze_gate_coverage_sec,
        'screen_normal_error_gate_coverage_complete': gaze_gate_coverage_complete,
        'screen_normal_error_gated': gaze_gate,
        'face_acquired': first_face_index is not None,
        'leading_face_absence_samples': leading_absence_count,
        'leading_face_absence_sec': (
            max(0.0, rows[first_face_index]['elapsed_sec'] - start)
            if is_acquisition and first_face_index is not None else
            (end - start if is_acquisition else 0.0)),
        'first_valid_face_elapsed_sec': (
            rows[first_face_index]['elapsed_sec']
            if first_face_index is not None else None),
        'post_acquisition_samples': len(behavior_rows),
        'q3_min_rad': min(q_values, default=None),
        'q3_max_rad': max(q_values, default=None),
        'q3_comfort_rad': list(q3_comfort),
        'q3_hold_tolerance_rad': q3_tolerance,
        'q3_raw_interval_excursion_samples': len(q3_violations),
        'q3_raw_interval_recovery_count': sum(
            previous and not current for previous, current in
            zip(q3_outside_flags, q3_outside_flags[1:])),
        'q3_max_raw_interval_excursion_rad': max(q3_excursions, default=None),
        'q3_reversals_over_1deg': q3_cycle.metrics['q3_reversals_over_1deg'],
        'q3_straight_crossings': q3_cycle.metrics['straight_crossings'],
        'q3_comfort_gate_applied': q3_gate_applied,
        'q3_comfort_violations': q3_practical_violations,
        'physical_joint_limit_violations': len(joint_violations),
        'continuous_FACE_target_used_samples': command_rows,
        'all_face_targets_available_and_used': all(
            _valid_face_command(row.get('context')) or
            _has_source_stamped_face_reference(row.get('context'))
            for _, row in evaluated_rows),
        'servo_status_codes': sorted(set(servo_status_codes)),
        'servo_status_messages': sorted(set(servo_status_messages)),
        'strict_source_tf_alignment': strict_source,
        'maximum_joint_gap_sec': max(stage_gaps, default=None),
        'maximum_context_joint_skew_sec': max(context_skews, default=None),
        'maximum_target_source_joint_skew_sec': max(target_source_skews,
                                                    default=None),
        'maximum_input_callback_joint_skew_sec': max(face_skews, default=None),
        'safety_counter_values': {
            name: {'first': present[0] if present else None,
                   'last': present[-1] if present else None,
                   'delta': present[-1] - present[0] if present else None,
                   'missing_samples': len(values) - len(present)}
            for name, values in counter_series.items()
            if values
            for present in ([value for value in values if value is not None],)
        },
        'settling': ({'passed': settle.passed, 'failures': settle.failures,
                      'metrics': settle.metrics} if settle else None),
        'measurement_quality_complete': strict_source,
        'behavior_assessment': behavior_assessment,
        'passed': passed,
    }


def _range(points):
    if not points:
        return None
    return math.sqrt(sum((max(p[i] for p in points) - min(p[i] for p in points)) ** 2
                         for i in range(3)))


def _within_distance_band(values, band, tolerance):
    return bool(values and min(values) >= band[0] - tolerance and
                max(values) <= band[1] + tolerance)


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--window', nargs=2, type=float, default=(20.0, 60.0))
    parser.add_argument('--sequence', type=Path,
                        help='Analyze a sequence capture with phase-aware hold checks.')
    args = parser.parse_args()
    result = (analyze_sequence(
        args.capture, args.sequence)
              if args.sequence else
              analyze(args.capture, *args.window))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
