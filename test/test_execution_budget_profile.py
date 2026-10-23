# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Validation tests for explicit hardware execution-budget profiles."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from face_tracking_bringup.execution_budget_profile import (  # noqa: E402
    execution_conditions_fingerprint,
    execution_conditions_from_launch,
    validate_active_hardware_profile,
    validate_execution_budget_profile,
    validate_profile_match,
)


class ExecutionBudgetProfileTest(unittest.TestCase):
    """Check profile identity, bounds, and launch-condition matching."""

    def setUp(self):
        """Create a complete synthetic profile and its fingerprint."""
        self.conditions = {
            'robot_model': 'xarm6', 'model_num': 1303,
            'firmware_version': '2.7.0', 'reduced_mode': False,
            'reduced_joint_limits_rad': [[-1.0, 1.0]] * 6,
            'reduced_max_joint_speed_rad_s': None, 'speed_scale': 1.0,
            'max_velocity_rad_s': [1.0] * 6,
            'max_acceleration_rad_s2': [2.0] * 6,
            'max_jerk_rad_s3': [20.0] * 6, 'payload_kg': 5.7,
            'mount_geometry': {'version': 2, 'monitor_mass_kg': 0.7},
            'controller_mode': 'position_servoj',
            'controller_configuration_sha256': '0' * 64,
            'robot_description_sha256': '1' * 64,
            'semantic_description_sha256': '2' * 64,
            'control_period_sec': .01, 'servo_publish_period_sec': .01,
            'state_feedback_timeout_sec': .03,
            'hardware_io_sample_timeout_default_sec': .1,
            'hardware_io_transport_timeout_default_sec': .02,
            'robot_msg_timeout_sec': .5,
            'maximum_command_latency_sec': 0.05,
            'residual_command_latency_sec': 0.0,
            'stop_assumptions_id': 'bench-stop-contract-v1',
            'payload_load_status': 'configuration_only_unverified',
        }
        self.fingerprint = execution_conditions_fingerprint(self.conditions)
        self.profile = {
            'profile_id': 'xarm6-1303-lab-v1', 'robot_model': 'xarm6',
            'model_num': 1303, 'firmware_version': '2.7.0',
            'controller_mode': 'position_servoj', 'reduced_mode': False,
            'execution_error_bound_rad': 0.03, 'reaction_threshold_rad': 0.01,
            'post_threshold_growth_rad': 0.02,
            'conditions_fingerprint': self.fingerprint,
            'evidence_reference': 'preflight/stop-test-2026-10-22',
            'stop_assumptions_id': 'bench-stop-contract-v1',
        }

    def test_accepts_exact_profile_and_configuration_match(self):
        """Accept an exact model and launch configuration match."""
        accepted = validate_profile_match(
            self.profile, model='xarm6', model_num=1303, firmware='2.7.0',
            reduced_mode=False, controller_mode='position_servoj',
            conditions_fingerprint=self.fingerprint)
        self.assertEqual(accepted['execution_error_bound_rad'], 0.03)

    def test_rejects_absent_or_unknown_profile_fields(self):
        """Reject missing profiles and unsupported settings."""
        with self.assertRaisesRegex(ValueError, 'expected a mapping'):
            validate_execution_budget_profile(None)
        with self.assertRaisesRegex(ValueError, 'unknown settings'):
            validate_execution_budget_profile(
                {**self.profile, 'runtime_actions_enabled': True})

    def test_rejects_each_identity_or_conditions_mismatch(self):
        """Reject model, firmware, mode, and conditions mismatches."""
        cases = (
            {'robot_model': 'lite6'}, {'model_num': 1305},
            {'firmware_version': '2.7.1'}, {'reduced_mode': True},
            {'conditions_fingerprint': '0' * 64},
        )
        for changed in cases:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_profile_match(
                    {**self.profile, **changed}, model='xarm6', model_num=1303,
                    firmware='2.7.0', reduced_mode=False,
                    controller_mode='position_servoj',
                    conditions_fingerprint=self.fingerprint)

    def test_rejects_controller_mode_mismatch(self):
        """Reject a controller mode different from the configured profile."""
        with self.assertRaisesRegex(ValueError, 'controller mode'):
            validate_profile_match(
                self.profile, model='xarm6', model_num=1303, firmware='2.7.0',
                reduced_mode=False, controller_mode='joint',
                conditions_fingerprint=self.fingerprint)

    def test_requires_t_plus_r_not_greater_than_e(self):
        """Require reaction threshold plus growth to fit within total error."""
        validated = validate_execution_budget_profile(self.profile)
        self.assertEqual(validated['reaction_threshold_rad'], 0.01)
        changed = {
            **self.profile,
            'post_threshold_growth_rad': math.nextafter(0.02, 1.0),
        }
        with self.assertRaisesRegex(ValueError, r'T \+ R <= E'):
            validate_execution_budget_profile(changed)

    def test_rejects_nonfinite_or_nonpositive_error_bound(self):
        """Reject nonfinite, zero, and negative budget values."""
        for field, value in (
                ('execution_error_bound_rad', float('nan')),
                ('execution_error_bound_rad', 0.0),
                ('reaction_threshold_rad', float('inf')),
                ('post_threshold_growth_rad', -0.01)):
            with self.subTest(field=field, value=value), self.assertRaises(
                    ValueError):
                validate_execution_budget_profile(
                    {**self.profile, field: value})

    def test_fingerprint_is_stable_and_covers_motion_and_stop_inputs(self):
        """Include motion limits and stop assumptions in the fingerprint."""
        self.assertEqual(
            execution_conditions_fingerprint(self.conditions),
            self.fingerprint)
        changed = {**self.conditions, 'max_jerk_rad_s3': [21.0] * 6}
        self.assertNotEqual(
            execution_conditions_fingerprint(changed), self.fingerprint)
        changed = {**self.conditions, 'stop_assumptions_id': 'different-stop'}
        self.assertNotEqual(
            execution_conditions_fingerprint(changed), self.fingerprint)

    def test_active_profile_matches_resolved_launch_parameters(self):
        """Match the profile against values resolved from the active launch."""
        config = {
            'robot': {'model': 'xarm6', 'model_num': 1303,
                      'base_xyz_m': [0, 0, 0], 'base_rpy_deg': [0, 0, 0]},
            'motion': {'speed_scale': 1.0},
            'monitor': {'mass_kg': .1, 'size_m': [1, 1, 1],
                        'mount_xyz_m': [0, 0, 0], 'mount_rpy_deg': [0, 0, 0]},
            'camera_mount': {'mass_kg': .2, 'size_m': [1, 1, 1],
                             'xyz_m': [0, 0, 0], 'rpy_deg': [0, 0, 0]},
            'controller': {},
            'mount_geometry_version': 2,
            'execution_error_budget': dict(self.profile),
        }
        probe = {
            'profile': {'model': 'xarm6', 'model_num': 1303},
            'firmware_version': '2.7.0', 'reduced_mode': False,
            'effective_position_limits_rad': [[-1.0, 1.0]] * 6,
            'reduced_joint_limits_rad': [[-1.0, 1.0]] * 6,
            'reduced_max_joint_speed_rad_s': None,
        }
        robot_profile = {
            'model': 'xarm6', 'model_num': 1303, 'payload_kg': 5.4,
            'controller_name': 'arm_controller',
        }
        servo = {'publish_period': .01, 'max_expected_latency': .05}
        motion = {
            'control_period_sec': .01, 'state_feedback_timeout_sec': .03,
            'max_joint_velocity_rad_s': [1.0] * 6,
            'max_joint_acceleration_rad_s2': [2.0] * 6,
            'max_joint_jerk_rad_s3': [20.0] * 6,
            'residual_command_latency_sec': 0.0,
        }
        controllers = {'arm_controller': {'ros__parameters': {
            'command_interfaces': ['position'],
            'interpolation_method': 'splines'}}}
        robot_description = '<robot name="xarm6"/>'
        semantic_description = '<robot name="xarm6_semantic"/>'
        conditions = execution_conditions_from_launch(
            config, probe, robot_profile, servo, motion,
            'arm_controller:position:splines', 'bench-stop-contract-v1',
            robot_description, semantic_description, controllers)
        fingerprint = execution_conditions_fingerprint(conditions)
        config['execution_error_budget'].update(
            controller_mode='arm_controller:position:splines',
            conditions_fingerprint=fingerprint)
        accepted, actual_fingerprint, _ = validate_active_hardware_profile(
            config, probe, robot_profile, servo, motion, controllers,
            robot_description, semantic_description)
        self.assertEqual(accepted['profile_id'], self.profile['profile_id'])
        self.assertEqual(actual_fingerprint, fingerprint)
        self.assertAlmostEqual(conditions['payload_kg'], .3)
        self.assertEqual(
            conditions['payload_load_status'], 'configuration_only_unverified')

        motion['max_joint_jerk_rad_s3'] = [21.0] * 6
        with self.assertRaisesRegex(ValueError, 'conditions'):
            validate_active_hardware_profile(
                config, probe, robot_profile, servo, motion, controllers,
                robot_description, semantic_description)

        motion['max_joint_jerk_rad_s3'] = [20.0] * 6
        motion['state_feedback_timeout_sec'] = .04
        with self.assertRaisesRegex(ValueError, 'conditions'):
            validate_active_hardware_profile(
                config, probe, robot_profile, servo, motion, controllers,
                robot_description, semantic_description)

    def test_active_profile_is_required_only_by_robot_execution_helper(self):
        """Require a budget profile when validating an active robot launch."""
        with self.assertRaisesRegex(
                ValueError, 'require execution_error_budget'):
            validate_active_hardware_profile({}, {}, {}, {}, {}, {}, '', '')


if __name__ == '__main__':
    unittest.main()
