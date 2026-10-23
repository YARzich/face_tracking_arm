# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validation helpers for an explicit real-hardware E/T/R profile."""

import hashlib
import json
import math
import re


PROFILE_FIELDS = frozenset((
    'profile_id robot_model model_num firmware_version controller_mode reduced_mode '
    'execution_error_bound_rad reaction_threshold_rad '
    'post_threshold_growth_rad conditions_fingerprint evidence_reference '
    'stop_assumptions_id').split())
FINGERPRINT_FIELDS = frozenset((
    'robot_model model_num firmware_version reduced_mode reduced_joint_limits_rad '
    'reduced_max_joint_speed_rad_s speed_scale max_velocity_rad_s '
    'max_acceleration_rad_s2 max_jerk_rad_s3 payload_kg payload_load_status mount_geometry '
    'controller_mode controller_configuration_sha256 robot_description_sha256 '
    'semantic_description_sha256 '
    'control_period_sec servo_publish_period_sec state_feedback_timeout_sec '
    'hardware_io_sample_timeout_default_sec '
    'hardware_io_transport_timeout_default_sec '
    'robot_msg_timeout_sec maximum_command_latency_sec residual_command_latency_sec '
    'stop_assumptions_id').split())


def execution_conditions_fingerprint(conditions):
    """Hash only the canonical execution conditions, excluding the profile itself."""
    if not isinstance(conditions, dict) or set(conditions) != FINGERPRINT_FIELDS:
        missing = FINGERPRINT_FIELDS - set(conditions or {})
        extra = set(conditions or {}) - FINGERPRINT_FIELDS
        raise ValueError(f'execution conditions fields mismatch: missing={sorted(missing)}, '
                         f'extra={sorted(extra)}')
    payload = json.dumps(conditions, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def execution_conditions_from_launch(config, probe, robot_profile, servo, motion,
                                     controller_mode, stop_assumptions_id,
                                     robot_description, semantic_description, controllers):
    """Build the non-self-referential execution-condition fingerprint input."""
    if not isinstance(probe, dict) or not isinstance(robot_profile, dict):
        raise ValueError('execution conditions require resolved probe and robot profile')
    if not isinstance(config, dict) or not isinstance(servo, dict) or not isinstance(motion, dict):
        raise ValueError('execution conditions require normalized launch configuration')
    return {
        'robot_model': robot_profile['model'],
        'model_num': robot_profile['model_num'],
        'firmware_version': probe['firmware_version'],
        'reduced_mode': bool(probe['reduced_mode']),
        'reduced_joint_limits_rad': probe['effective_position_limits_rad'],
        'reduced_max_joint_speed_rad_s': probe.get('reduced_max_joint_speed_rad_s'),
        'speed_scale': float(config['motion']['speed_scale']),
        'max_velocity_rad_s': list(motion['max_joint_velocity_rad_s']),
        'max_acceleration_rad_s2': list(motion['max_joint_acceleration_rad_s2']),
        'max_jerk_rad_s3': list(motion['max_joint_jerk_rad_s3']),
        # These are configured attached masses, not a live controller load
        # measurement. The profile fingerprint binds the assumption explicitly.
        'payload_kg': float(config['monitor']['mass_kg']) +
        float(config['camera_mount']['mass_kg']),
        'payload_load_status': 'configuration_only_unverified',
        'mount_geometry': {
            'version': config.get('mount_geometry_version', 'legacy'),
            'base_xyz_m': list(config['robot']['base_xyz_m']),
            'base_rpy_deg': list(config['robot']['base_rpy_deg']),
            'monitor_size_m': list(config['monitor']['size_m']),
            'monitor_mount_xyz_m': list(config['monitor']['mount_xyz_m']),
            'monitor_mount_rpy_deg': list(config['monitor']['mount_rpy_deg']),
            'camera_size_m': list(config['camera_mount']['size_m']),
            'camera_mount_xyz_m': list(config['camera_mount']['xyz_m']),
            'camera_mount_rpy_deg': list(config['camera_mount']['rpy_deg']),
        },
        'controller_mode': controller_mode,
        'controller_configuration_sha256': hashlib.sha256(
            json.dumps(controllers, sort_keys=True, separators=(',', ':'),
                       allow_nan=False).encode('utf-8')).hexdigest(),
        'robot_description_sha256': hashlib.sha256(
            robot_description.encode('utf-8')).hexdigest(),
        'semantic_description_sha256': hashlib.sha256(
            semantic_description.encode('utf-8')).hexdigest(),
        'control_period_sec': float(motion['control_period_sec']),
        'servo_publish_period_sec': float(servo['publish_period']),
        'state_feedback_timeout_sec': float(motion['state_feedback_timeout_sec']),
        # The component takes the minimum of these defaults and the state
        # feedback timeout, which is fingerprinted separately above.
        'hardware_io_sample_timeout_default_sec': 0.1,
        'hardware_io_transport_timeout_default_sec': 0.02,
        'robot_msg_timeout_sec': 0.5,
        'maximum_command_latency_sec': float(servo['max_expected_latency']),
        'residual_command_latency_sec': float(motion['residual_command_latency_sec']),
        'stop_assumptions_id': stop_assumptions_id,
    }


def validate_execution_budget_keys(value):
    """Check only the optional block's shape and names, independent of activation."""
    if not isinstance(value, dict):
        raise ValueError('execution_error_budget: expected a mapping')
    unknown = set(value) - PROFILE_FIELDS
    if unknown:
        raise ValueError(f'execution_error_budget: unknown settings {sorted(unknown)}')


def validate_execution_budget_profile(value):
    """Validate a hardware profile's declared fields and E/T/R ordering."""
    validate_execution_budget_keys(value)
    unknown = set(value) - PROFILE_FIELDS
    missing = PROFILE_FIELDS - set(value)
    if unknown or missing:
        raise ValueError('execution_error_budget fields mismatch: '
                         f'missing={sorted(missing)}, unknown={sorted(unknown)}')
    for name in ('profile_id', 'robot_model', 'firmware_version', 'controller_mode',
                 'conditions_fingerprint', 'evidence_reference', 'stop_assumptions_id'):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f'execution_error_budget.{name}: expected nonempty text')
    fingerprint = value['conditions_fingerprint']
    if not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
        raise ValueError('execution_error_budget.conditions_fingerprint: expected SHA-256 hex')
    if type(value['reduced_mode']) is not bool:
        raise ValueError('execution_error_budget.reduced_mode: expected true or false')
    model_num = value['model_num']
    if isinstance(model_num, bool) or not isinstance(model_num, int) or model_num <= 0:
        raise ValueError('execution_error_budget.model_num: expected a positive integer')
    numbers = {}
    for name in ('execution_error_bound_rad', 'reaction_threshold_rad',
                 'post_threshold_growth_rad'):
        number = value[name]
        if (isinstance(number, bool) or not isinstance(number, (int, float)) or
                not math.isfinite(number)):
            raise ValueError(f'execution_error_budget.{name}: expected a finite number')
        numbers[name] = float(number)
    error = numbers['execution_error_bound_rad']
    reaction = numbers['reaction_threshold_rad']
    growth = numbers['post_threshold_growth_rad']
    if error <= 0.0 or reaction < 0.0 or growth < 0.0:
        raise ValueError('execution_error_budget values must be nonnegative and E positive')
    if reaction + growth > error:
        raise ValueError('execution_error_budget requires T + R <= E')
    return {**value, **numbers}


def validate_profile_match(profile, *, model, model_num, firmware, reduced_mode,
                           controller_mode, conditions_fingerprint):
    """Fail closed unless profile matches the connected preflight identity."""
    profile = validate_execution_budget_profile(profile)
    if profile['firmware_version'] != firmware:
        raise ValueError('execution_error_budget profile firmware does not match preflight')
    if profile['reduced_mode'] is not reduced_mode:
        raise ValueError('execution_error_budget reduced mode does not match preflight')
    if profile['controller_mode'] != controller_mode:
        raise ValueError('execution_error_budget controller mode does not match active config')
    if profile['conditions_fingerprint'] != conditions_fingerprint:
        raise ValueError('execution_error_budget conditions do not match active configuration')
    if (not isinstance(model, str) or not model or profile['robot_model'] != model or
            isinstance(model_num, bool) or not isinstance(model_num, int) or
            model_num <= 0 or profile['model_num'] != model_num):
        raise ValueError('execution_error_budget profile model/revision does not match preflight')
    return profile


def validate_active_hardware_profile(config, probe, robot_profile, servo, motion,
                                     controllers, robot_description, semantic_description):
    """Validate a profile against resolved preflight and merged launch parameters."""
    profile = config.get('execution_error_budget') if isinstance(config, dict) else None
    if profile is None:
        raise ValueError(
            'Real robot actions require execution_error_budget; '
            'no evidence-backed E/T/R profile is configured')
    controller_name = robot_profile['controller_name']
    controller = controllers[controller_name]['ros__parameters']
    controller_mode = ':'.join((controller_name,
                                '+'.join(controller['command_interfaces']),
                                controller['interpolation_method']))
    conditions = execution_conditions_from_launch(
        config, probe, robot_profile, servo, motion,
        controller_mode, profile.get('stop_assumptions_id'), robot_description,
        semantic_description, controllers)
    fingerprint = execution_conditions_fingerprint(conditions)
    validated = validate_profile_match(
        profile, model=probe['profile']['model'],
        model_num=probe['profile']['model_num'], firmware=probe['firmware_version'],
        reduced_mode=probe['reduced_mode'], controller_mode=controller_mode,
        conditions_fingerprint=fingerprint)
    return validated, fingerprint, controller_mode
