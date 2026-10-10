// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/collision_aware_parameters.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include <moveit_servo/moveit_servo_lib_parameters.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node.hpp>

namespace face_tracking_arm
{
namespace
{

constexpr double kMaximumControlPeriodSec = 0.020;
constexpr double kMaximumTrajectoryControllerPeriodSec = 0.010;
constexpr double kMaximumTargetTimeoutSec = 0.100;
constexpr double kMaximumPositionDeadbandM = 0.250;
constexpr double kMaximumPointingDeadbandRad = 0.500;
constexpr double kMaximumPrimaryPreservationToleranceRadps = 0.050;
constexpr double kMinimumHardClearanceM = 0.015;
constexpr double kMinimumCollisionQueryDistanceM = 0.180;
constexpr std::size_t kMinimumCollisionConstraintRows = 24;
constexpr int kMaximumCollisionConstraintRows = 64;
constexpr double kMaximumCollisionGradientEpsilon = 1.0e-8;
constexpr double kMinimumResidualCommandLatencySec = 0.020;
constexpr double kMinimumNumericalDistanceReserveM = 0.0001;
constexpr double kMaximumSolverTolerance = 1.0e-5;
constexpr double kMaximumSolverTimeSec = 0.003;
constexpr double kMinimumSegmentValidationDistanceM = kMinimumHardClearanceM;
constexpr int kMinimumSegmentValidationSubsteps = 3;
constexpr int kMaximumSegmentValidationSubsteps = 16;

void declareDeprecatedParameters(rclcpp::Node & node)
{
  constexpr std::array<std::pair<const char *, double>, 12> deprecated{{
    {"joint_centering_activation_fraction", 0.35},
    {"joint_centering_max_velocity_rad_s", 0.4},
    {"joint_centering_weight", 3.0},
    {"collision_avoidance_buffer_m", 0.04},
    {"collision_avoidance_max_velocity_mps", 0.12},
    {"collision_avoidance_lookahead_sec", 0.4},
    {"collision_avoidance_weight", 4.0},
    {"lower_singularity_threshold", 40.0},
    {"hard_stop_singularity_threshold", 80.0},
    {"singularity_recovery_gain", 2.0},
    {"singularity_gradient_step_rad", 0.001},
    {"monitor_near_distance_lipschitz_m_per_rad", 0.4543},
  }};
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description =
    "Deprecated compatibility parameter: ignored by the collision-aware controller.";
  const auto & overrides = node.get_node_parameters_interface()->get_parameter_overrides();
  for (const auto & [name, default_value] : deprecated) {
    node.declare_parameter<double>(name, default_value, descriptor);
    if (overrides.find(name) != overrides.end()) {
      RCLCPP_WARN(
        node.get_logger(),
        "Parameter '%s' is deprecated and has no effect on the collision-aware controller; "
        "remove it from this controller's configuration.", name);
    }
  }
}

}  // namespace

ControllerParameters declareControllerParameters(rclcpp::Node & node)
{
  declareDeprecatedParameters(node);
  ControllerParameters result;
  result.control_period_sec = node.declare_parameter<double>(
    "control_period_sec", result.control_period_sec);
  result.trajectory_controller_period_sec = node.declare_parameter<double>(
    "trajectory_controller_period_sec", result.trajectory_controller_period_sec);
  result.incoming_command_timeout_sec = node.declare_parameter<double>(
    "incoming_command_timeout_sec", result.incoming_command_timeout_sec);
  result.planning_group_name = node.declare_parameter<std::string>(
    "planning_group_name", result.planning_group_name);
  result.planning_frame = node.declare_parameter<std::string>(
    "planning_frame", result.planning_frame);
  result.command_frame = node.declare_parameter<std::string>(
    "command_frame", result.command_frame);
  rcl_interfaces::msg::ParameterDescriptor recovery_mode_descriptor;
  recovery_mode_descriptor.description =
    "Recovery mechanism selected at startup: compact or disabled.";
  recovery_mode_descriptor.read_only = true;
  result.recovery_mode = node.declare_parameter<std::string>(
    "recovery.mode", result.recovery_mode, recovery_mode_descriptor);
  result.gaze_frame = node.declare_parameter<std::string>("gaze_frame", result.gaze_frame);
  result.table_collision_enabled = node.declare_parameter<bool>(
    "table_collision_enabled", result.table_collision_enabled);

  result.position_gain = node.declare_parameter<double>("position_gain", result.position_gain);
  result.orientation_gain = node.declare_parameter<double>(
    "orientation_gain", result.orientation_gain);
  result.maximum_linear_reference_mps = node.declare_parameter<double>(
    "maximum_linear_reference_mps", result.maximum_linear_reference_mps);
  result.maximum_angular_reference_radps = node.declare_parameter<double>(
    "maximum_angular_reference_radps", result.maximum_angular_reference_radps);
  result.position_deadband_m = node.declare_parameter<double>(
    "position_deadband_m", result.position_deadband_m);
  result.pointing_deadband_rad = node.declare_parameter<double>(
    "pointing_deadband_rad", result.pointing_deadband_rad);
  result.primary_orientation_tolerance_radps = node.declare_parameter<double>(
    "primary_orientation_tolerance_radps", result.primary_orientation_tolerance_radps);
  result.secondary_roll_weight = node.declare_parameter<double>(
    "secondary_roll_weight", result.secondary_roll_weight);
  result.velocity_regularization_weight = node.declare_parameter<double>(
    "velocity_regularization_weight", result.velocity_regularization_weight);

  result.hard_clearance_m = node.declare_parameter<double>(
    "hard_clearance_m", result.hard_clearance_m);
  result.collision_query_distance_m = node.declare_parameter<double>(
    "collision_query_distance_m", result.collision_query_distance_m);
  result.maximum_collision_constraints = node.declare_parameter<int>(
    "maximum_collision_constraints", result.maximum_collision_constraints);
  result.collision_gradient_epsilon = node.declare_parameter<double>(
    "collision_gradient_epsilon", result.collision_gradient_epsilon);

  result.monitor_guard_joint_name = node.declare_parameter<std::string>(
    "monitor_guard_joint_name", result.monitor_guard_joint_name);
  result.monitor_guard_min_position_rad = node.declare_parameter<double>(
    "monitor_guard_min_position_rad", result.monitor_guard_min_position_rad);
  result.monitor_guard_max_position_rad = node.declare_parameter<double>(
    "monitor_guard_max_position_rad", result.monitor_guard_max_position_rad);
  result.joint_position_margin_rad = node.declare_parameter<double>(
    "joint_position_margin_rad", result.joint_position_margin_rad);
  result.timing_mode = node.declare_parameter<std::string>(
    "timing_mode", result.timing_mode);
  result.source_timeout_sec = node.declare_parameter<double>(
    "source_timeout_sec", result.source_timeout_sec);
  result.face_target_freshness_timeout_sec = node.declare_parameter<double>(
    "face_target_freshness_timeout_sec", result.face_target_freshness_timeout_sec);
  result.face_prediction_timeout_sec = node.declare_parameter<double>(
    "face_prediction_timeout_sec", result.face_prediction_timeout_sec);
  result.residual_command_latency_sec = node.declare_parameter<double>(
    "residual_command_latency_sec", result.residual_command_latency_sec);
  result.state_feedback_timeout_sec = node.declare_parameter<double>(
    "state_feedback_timeout_sec", result.state_feedback_timeout_sec);
  result.following_position_tolerance_rad = node.declare_parameter<double>(
    "following_position_tolerance_rad", result.following_position_tolerance_rad);
  result.following_velocity_tolerance_rad_s = node.declare_parameter<double>(
    "following_velocity_tolerance_rad_s", result.following_velocity_tolerance_rad_s);
  result.collision_tracking_error_bound_rad = node.declare_parameter<double>(
    "collision_tracking_error_bound_rad", result.collision_tracking_error_bound_rad);
  result.numerical_distance_reserve_m = node.declare_parameter<double>(
    "numerical_distance_reserve_m", result.numerical_distance_reserve_m);
  result.default_distance_lipschitz_m_per_rad = node.declare_parameter<double>(
    "default_distance_lipschitz_m_per_rad",
    result.default_distance_lipschitz_m_per_rad);
  result.solver_max_iterations = node.declare_parameter<int>(
    "solver_max_iterations", result.solver_max_iterations);
  result.solver_absolute_tolerance = node.declare_parameter<double>(
    "solver_absolute_tolerance", result.solver_absolute_tolerance);
  result.solver_relative_tolerance = node.declare_parameter<double>(
    "solver_relative_tolerance", result.solver_relative_tolerance);
  result.solver_time_limit_sec = node.declare_parameter<double>(
    "solver_time_limit_sec", result.solver_time_limit_sec);
  result.segment_validation_distance_m = node.declare_parameter<double>(
    "segment_validation_distance_m", result.segment_validation_distance_m);
  result.segment_validation_substeps = node.declare_parameter<int>(
    "segment_validation_substeps", result.segment_validation_substeps);

  result.joint_names = node.declare_parameter<std::vector<std::string>>(
    "joint_names", std::vector<std::string>{});
  result.max_joint_velocity_rad_s = node.declare_parameter<std::vector<double>>(
    "max_joint_velocity_rad_s", std::vector<double>{});
  result.max_joint_acceleration_rad_s2 = node.declare_parameter<std::vector<double>>(
    "max_joint_acceleration_rad_s2", std::vector<double>{});
  result.max_joint_jerk_rad_s3 = node.declare_parameter<std::vector<double>>(
    "max_joint_jerk_rad_s3", std::vector<double>{});
  return result;
}

servo::Params declareCollisionAwareServoParameters(
  rclcpp::Node & node, const ControllerParameters & controller)
{
  servo::Params result;
  result.thread_priority = node.declare_parameter<std::int64_t>(
    "moveit_servo.thread_priority", result.thread_priority);
  result.max_expected_latency = node.declare_parameter<double>(
    "moveit_servo.max_expected_latency", 0.05);
  result.command_out_topic = node.declare_parameter<std::string>(
    "moveit_servo.command_out_topic", "/lite6_arm_controller/joint_trajectory");
  result.joint_topic = node.declare_parameter<std::string>(
    "moveit_servo.joint_topic", "/joint_states");
  result.monitored_planning_scene_topic = node.declare_parameter<std::string>(
    "moveit_servo.monitored_planning_scene_topic", "/planning_scene");
  result.is_primary_planning_scene_monitor = node.declare_parameter<bool>(
    "moveit_servo.is_primary_planning_scene_monitor", true);

  // composeTrajectoryMessage consumes these fields, but they are properties of
  // this backend's fixed output contract, not user-selectable Servo settings.
  result.publish_period = controller.control_period_sec;
  result.move_group_name = controller.planning_group_name;
  result.command_out_type = "trajectory_msgs/JointTrajectory";
  result.command_in_type = "speed_units";
  result.publish_joint_positions = true;
  result.publish_joint_velocities = false;
  result.publish_joint_accelerations = false;
  result.check_collisions = false;
  result.use_smoothing = false;
  if (result.thread_priority < 0 ||
    result.thread_priority > std::numeric_limits<int>::max() ||
    result.command_out_topic.empty() || result.joint_topic.empty() ||
    result.monitored_planning_scene_topic.empty())
  {
    throw std::invalid_argument("invalid shared MoveIt Servo runtime settings");
  }
  return result;
}

void validateControllerParameters(const ControllerParameters & parameters)
{
  const auto positive = [](const double value) {
      return std::isfinite(value) && value > 0.0;
    };
  if (parameters.recovery_mode != "compact" && parameters.recovery_mode != "disabled") {
    throw std::invalid_argument(
            "recovery.mode must be either 'compact' or 'disabled'");
  }
  if (!positive(parameters.control_period_sec) ||
    parameters.control_period_sec > kMaximumControlPeriodSec ||
    !positive(parameters.trajectory_controller_period_sec) ||
    parameters.trajectory_controller_period_sec >
    kMaximumTrajectoryControllerPeriodSec ||
    parameters.trajectory_controller_period_sec > parameters.control_period_sec ||
    !positive(parameters.incoming_command_timeout_sec) ||
    parameters.incoming_command_timeout_sec < parameters.control_period_sec ||
    parameters.incoming_command_timeout_sec > kMaximumTargetTimeoutSec ||
    parameters.planning_group_name.empty() ||
    parameters.planning_frame != "world" ||
    parameters.command_frame != "monitor_control_frame" ||
    !positive(parameters.position_gain) ||
    !positive(parameters.orientation_gain) ||
    !positive(parameters.maximum_linear_reference_mps) ||
    !positive(parameters.maximum_angular_reference_radps) ||
    !std::isfinite(parameters.position_deadband_m) ||
    parameters.position_deadband_m < 0.0 ||
    parameters.position_deadband_m > kMaximumPositionDeadbandM ||
    !std::isfinite(parameters.pointing_deadband_rad) ||
    parameters.pointing_deadband_rad < 0.0 ||
    parameters.pointing_deadband_rad > kMaximumPointingDeadbandRad ||
    !positive(parameters.primary_orientation_tolerance_radps) ||
    parameters.primary_orientation_tolerance_radps >
    kMaximumPrimaryPreservationToleranceRadps ||
    !positive(parameters.secondary_roll_weight) ||
    !positive(parameters.velocity_regularization_weight) ||
    !positive(parameters.hard_clearance_m) ||
    parameters.hard_clearance_m < kMinimumHardClearanceM ||
    !positive(parameters.collision_query_distance_m) ||
    parameters.collision_query_distance_m < kMinimumCollisionQueryDistanceM ||
    parameters.collision_query_distance_m <= parameters.hard_clearance_m ||
    parameters.maximum_collision_constraints <
    static_cast<int>(kMinimumCollisionConstraintRows) ||
    parameters.maximum_collision_constraints > kMaximumCollisionConstraintRows ||
    !positive(parameters.collision_gradient_epsilon) ||
    parameters.collision_gradient_epsilon > kMaximumCollisionGradientEpsilon ||
    (!parameters.monitor_guard_joint_name.empty() &&
    (!std::isfinite(parameters.monitor_guard_min_position_rad) ||
    !std::isfinite(parameters.monitor_guard_max_position_rad) ||
    parameters.monitor_guard_min_position_rad >= parameters.monitor_guard_max_position_rad)) ||
    !std::isfinite(parameters.joint_position_margin_rad) ||
    parameters.joint_position_margin_rad < 0.0 ||
    (parameters.timing_mode != "auto" && parameters.timing_mode != "manual") ||
    !positive(parameters.source_timeout_sec) || parameters.source_timeout_sec > 1.0 ||
    !positive(parameters.face_target_freshness_timeout_sec) ||
    parameters.face_target_freshness_timeout_sec > 1.0 ||
    !std::isfinite(parameters.face_prediction_timeout_sec) ||
    parameters.face_prediction_timeout_sec < 0.0 ||
    parameters.face_prediction_timeout_sec > 2.0 ||
    (parameters.face_prediction_timeout_sec > 0.0 &&
    parameters.face_prediction_timeout_sec <= parameters.face_target_freshness_timeout_sec) ||
    !std::isfinite(parameters.residual_command_latency_sec) ||
    parameters.residual_command_latency_sec < kMinimumResidualCommandLatencySec ||
    !positive(parameters.state_feedback_timeout_sec) ||
    parameters.state_feedback_timeout_sec > parameters.incoming_command_timeout_sec ||
    !positive(parameters.following_position_tolerance_rad) ||
    !positive(parameters.following_velocity_tolerance_rad_s) ||
    !positive(parameters.collision_tracking_error_bound_rad) ||
    parameters.collision_tracking_error_bound_rad >
    parameters.following_position_tolerance_rad ||
    !std::isfinite(parameters.numerical_distance_reserve_m) ||
    parameters.numerical_distance_reserve_m < kMinimumNumericalDistanceReserveM ||
    !std::isfinite(parameters.default_distance_lipschitz_m_per_rad) ||
    parameters.default_distance_lipschitz_m_per_rad < 0.0 ||
    parameters.solver_max_iterations <= 0 ||
    !positive(parameters.solver_absolute_tolerance) ||
    parameters.solver_absolute_tolerance > kMaximumSolverTolerance ||
    !positive(parameters.solver_relative_tolerance) ||
    parameters.solver_relative_tolerance > kMaximumSolverTolerance ||
    !positive(parameters.solver_time_limit_sec) ||
    parameters.solver_time_limit_sec > kMaximumSolverTimeSec ||
    parameters.solver_time_limit_sec >= parameters.control_period_sec ||
    !positive(parameters.segment_validation_distance_m) ||
    parameters.segment_validation_distance_m < kMinimumSegmentValidationDistanceM ||
    parameters.segment_validation_distance_m < parameters.hard_clearance_m ||
    parameters.segment_validation_distance_m > parameters.collision_query_distance_m ||
    parameters.segment_validation_substeps < kMinimumSegmentValidationSubsteps ||
    parameters.segment_validation_substeps > kMaximumSegmentValidationSubsteps)
  {
    throw std::invalid_argument("invalid collision-aware Servo configuration");
  }
}

}  // namespace face_tracking_arm
