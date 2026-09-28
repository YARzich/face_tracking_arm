# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Names accepted by the editable hardware configuration."""


SECTIONS = {
    'robot': 'model model_num ip report_type base_xyz_m base_rpy_deg',
    'table': 'shape dimensions_m center_m yaw_deg',
    'monitor': 'size_m mass_kg mount_xyz_m mount_rpy_deg',
    'camera_mount': 'xyz_m rpy_deg size_m mass_kg body_frame publish_optical_tf',
    'perception': 'python_executable model_dir detector threads confidence face_width_m '
                  'max_processing_fps preview_fps max_frame_age_sec '
                  'switch_margin_m switch_delay_sec',
    'tracking': 'idle_behavior minimum_face_distance_m safe_reach_radius_m '
                'return_to_rest_delay_sec',
    'motion': 'speed_scale search_speed_rad_s search_sweep_half_range_rad '
              'search_local_half_range_rad search_local_duration_sec '
              'rest_joints_deg search_joints_deg',
    'limits': 'max_velocity_rad_s max_acceleration_rad_s2 max_jerk_rad_s3',
    'controller': '', 'mono_cpu': 'source rectification left',
    'stereo': 'source rectification left right baseline_m sync_slop_sec',
}
CAMERA_FIELDS = frozenset(('device width height fps pixel_format driver_parameters '
                           'control_values camera_name calibration_file frame_id image_topic '
                           'info_topic optical_xyz_m optical_rpy_deg').split())
TRACKING_OPTIONAL = frozenset(('face_target_freshness_timeout_sec rest_position_m').split())
MODE_OPTIONAL = {
    'mono_cpu': frozenset(('sync_slop_sec',)),
    'stereo': frozenset(('num_disparities', 'block_size')),
}
POINT_FILTER_FIELDS = frozenset((
    'enabled method reset_after_sec mono_cpu stereo one_euro').split())
KALMAN_FIELDS = frozenset(('measurement_std_m acceleration_std_mps2').split())
ONE_EURO_FIELDS = frozenset(('min_cutoff_hz beta derivative_cutoff_hz').split())
CONTROLLER_FIELDS = frozenset((
    'position_gain orientation_gain maximum_linear_reference_mps '
    'maximum_angular_reference_radps position_deadband_m pointing_deadband_rad '
    'primary_orientation_tolerance_radps secondary_roll_weight velocity_regularization_weight '
    'hard_clearance_m collision_query_distance_m maximum_collision_constraints '
    'collision_gradient_epsilon monitor_guard_joint_name monitor_guard_min_position_rad '
    'monitor_guard_max_position_rad joint_position_margin_rad residual_command_latency_sec '
    'state_feedback_timeout_sec following_position_tolerance_rad '
    'following_velocity_tolerance_rad_s collision_tracking_error_bound_rad '
    'numerical_distance_reserve_m '
    'default_distance_lipschitz_m_per_rad monitor_near_distance_lipschitz_m_per_rad '
    'solver_max_iterations solver_absolute_tolerance solver_relative_tolerance '
    'solver_time_limit_sec segment_validation_distance_m segment_validation_substeps').split())
