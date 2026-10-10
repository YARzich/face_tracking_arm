// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COLLISION_AWARE_PARAMETERS_HPP_
#define FACE_TRACKING_ARM__COLLISION_AWARE_PARAMETERS_HPP_

#include <string>
#include <vector>

namespace rclcpp
{
class Node;
}  // namespace rclcpp

namespace servo
{
struct Params;
}  // namespace servo

namespace face_tracking_arm
{

struct ControllerParameters
{
  double control_period_sec{0.01};
  double trajectory_controller_period_sec{0.01};
  double incoming_command_timeout_sec{0.10};
  std::string planning_group_name{"xarm6"};
  std::string planning_frame{"world"};
  std::string command_frame{"monitor_control_frame"};
  /// Optional optical camera frame: +Z forward. Empty uses the monitor +X axis.
  std::string gaze_frame;
  /// Derived from the table dimensions; initialization waits for this scene state.
  bool table_collision_enabled{true};

  double position_gain{1.5};
  double orientation_gain{4.0};
  double maximum_linear_reference_mps{0.18};
  double maximum_angular_reference_radps{0.70};
  double position_deadband_m{0.10};
  double pointing_deadband_rad{0.005};
  double primary_orientation_tolerance_radps{0.0002};
  double secondary_roll_weight{0.5};
  double velocity_regularization_weight{1.0e-4};

  double hard_clearance_m{0.015};
  double collision_query_distance_m{0.200};
  int maximum_collision_constraints{24};
  double collision_gradient_epsilon{1.0e-8};

  std::string monitor_guard_joint_name{};
  double monitor_guard_min_position_rad{-1.60};
  double monitor_guard_max_position_rad{1.60};
  double joint_position_margin_rad{0.10};
  std::string timing_mode{"auto"};
  double source_timeout_sec{0.50};
  double face_target_freshness_timeout_sec{0.50};
  double face_prediction_timeout_sec{0.80};
  double residual_command_latency_sec{0.020};
  double state_feedback_timeout_sec{0.040};
  double following_position_tolerance_rad{0.020};
  double following_velocity_tolerance_rad_s{0.25};
  double collision_tracking_error_bound_rad{0.006};
  double numerical_distance_reserve_m{0.0001};
  double default_distance_lipschitz_m_per_rad{2.6193};

  int solver_max_iterations{2000};
  double solver_absolute_tolerance{1.0e-5};
  double solver_relative_tolerance{1.0e-5};
  double solver_time_limit_sec{0.003};
  double segment_validation_distance_m{0.015};
  int segment_validation_substeps{3};

  std::vector<std::string> joint_names;
  std::vector<double> max_joint_velocity_rad_s;
  std::vector<double> max_joint_acceleration_rad_s2;
  std::vector<double> max_joint_jerk_rad_s3;
};

/// Declare active settings and retain deprecated ROS names for old configurations.
/// Explicit overrides of unused parameters emit a warning and cannot affect validation.
[[nodiscard]] ControllerParameters declareControllerParameters(rclcpp::Node & node);

/// Declare the legacy shared moveit_servo.* settings used by the custom backend.
/// Output mode, group, and period are fixed by the collision-aware controller.
[[nodiscard]] servo::Params declareCollisionAwareServoParameters(
  rclcpp::Node & node, const ControllerParameters & controller);

/// Reject configurations outside the calibrated controller safety envelope.
/// Robot-model and Servo-interface compatibility are checked separately at startup.
void validateControllerParameters(const ControllerParameters & parameters);

}  // namespace face_tracking_arm

#endif  // FACE_TRACKING_ARM__COLLISION_AWARE_PARAMETERS_HPP_
