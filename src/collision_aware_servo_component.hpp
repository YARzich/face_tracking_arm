// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COLLISION_AWARE_SERVO_COMPONENT_HPP_
#define COLLISION_AWARE_SERVO_COMPONENT_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <controller_manager_msgs/msg/controller_manager_activity.hpp>
#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/planning_scene_monitor/planning_scene_monitor.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit_msgs/msg/servo_status.hpp>
#include <moveit_msgs/srv/servo_command_type.hpp>
#include <moveit_servo/moveit_servo_lib_parameters.hpp>
#include <moveit_servo/utils/common.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_helpers.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "command_session.hpp"
#include "servo_telemetry.hpp"
#include "servo_types.hpp"
#include "face_tracking_arm/collision_aware_parameters.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/msg/tracking_target.hpp"
#include "face_tracking_arm/control_time_grid.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/following_error_policy.hpp"
#include "face_tracking_arm/hierarchical_velocity_qp.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"
#include "face_tracking_arm/rolling_statistics.hpp"
#include "face_tracking_arm/motion_reference.hpp"

namespace face_tracking_arm
{

enum class CommandPublicationStatus
{
  kSuccess,
  kRecoverableCommandDeadline,
  kRecoverableTimingFailure,
  kInvariantFailure,
};

enum class BootstrapFailureKind
{
  kRetryableState,
  kSafetyOrInvariant,
};

enum class RearmReadiness
{
  kWaiting,
  kReady,
  kUnsafe,
};

enum class ActualFeedbackSafety
{
  kWaiting,
  kSafe,
  kUnsafe,
};

struct ValidatedCommand
{
  moveit_servo::KinematicState next_state;
  control::EmergencyBrakeTail braking_tail;
};

struct BootstrapCommand
{
  moveit_servo::KinematicState seed_state;
  ValidatedCommand first_command;
};

// Private component declaration. Its state has one owner; implementation files
// group initialization, control, feedback, trajectories, recovery and diagnostics.
class CollisionAwareServoComponent final : public rclcpp::Node
{
public:
  explicit CollisionAwareServoComponent(const rclcpp::NodeOptions & options);

  ~CollisionAwareServoComponent() override;

private:
  void initialize();

  void shutdown_resources();

  void complete_initialization_when_ready();

  void validate_servo_parameters() const;

  void initialize_motion_limits(const moveit::core::RobotModel & robot_model);

  void receive_pose(geometry_msgs::msg::PoseStamped::ConstSharedPtr message);

  void receive_target(const msg::TrackingTarget & target);

  void receive_controller_activity(
    const controller_manager_msgs::msg::ControllerManagerActivity & message) noexcept;

  std::optional<control_msgs::msg::JointTrajectoryControllerState> current_controller_state() const;

  void switch_command_type(
    const moveit_msgs::srv::ServoCommandType::Request & request,
    moveit_msgs::srv::ServoCommandType::Response & response);

  void set_paused(
    const std_srvs::srv::SetBool::Request & request,
    std_srvs::srv::SetBool::Response & response);

  ActualFeedbackSafety validate_controller_feedback_collision_state(
    const control_msgs::msg::JointTrajectoryControllerState & controller_state,
    const std::string & context, std::string & reason);

  RearmReadiness safe_rearm_ready(
    const rclcpp::Time & current_time, std::string & reason);

  void control_loop();

  void publish_runtime_diagnostics();

  void control_tick();

  bool prepare_control_cycle(
    const rclcpp::Time & current_time,
    const std::chrono::steady_clock::time_point & current_wall_time,
    const control::ControlTimeObservation & time_observation);

  void process_startup_cycle(
    const moveit_servo::KinematicState & state,
    const moveit::core::RobotState & robot_state,
    const planning_scene::PlanningScene & planning_scene);

  std::optional<moveit_servo::KinematicState> predicted_or_measured_state(
    const rclcpp::Time & current_time, std::string & failure_reason,
    bool & requires_latched_halt,
    std::optional<control::JointMotionState> * actual_feedback = nullptr,
    std::optional<rclcpp::Time> * actual_feedback_stamp = nullptr);

  void observe_controller_feedback(
    const control_msgs::msg::JointTrajectoryControllerState & state,
    const rcl_clock_type_t clock_type);

  bool feedback_motion_within_limits(
    const control_msgs::msg::JointTrajectoryControllerState & state,
    std::string & failure_reason) const;

  std::optional<msg::TrackingTarget> current_target(
    const rclcpp::Time & tick_time) const;

  control::QpResult solve_with_telemetry(
    const control::HierarchicalVelocityTask & task,
    const control::JointMotionState & state,
    const std::vector<control::LinearVelocityConstraint> & constraints);

  control::QpResult brake_with_telemetry(
    const control::JointMotionState & state,
    const std::vector<control::LinearVelocityConstraint> & constraints);

  void log_qp_failure(
    const CommandAttempt attempt,
    const control::QpResult & result,
    const control::JointMotionState & motion_state,
    const std::vector<control::LinearVelocityConstraint> & constraints,
    const control::CollisionConstraintDiagnostics & collision_diagnostics) const;

  void record_segment_rejection(
    const CommandAttempt attempt,
    const control::CollisionSegmentValidationResult & result);

  std::optional<BootstrapCommand> make_stationary_bootstrap(
    const moveit_servo::KinematicState & measured_state,
    const moveit::core::RobotState & robot_state,
    const planning_scene::PlanningScene & planning_scene,
    std::string & failure_reason, BootstrapFailureKind & failure_kind);

  std::optional<ValidatedCommand> validated_candidate(
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const control::JointMotionState & current_motion_state,
    const Eigen::VectorXd & joint_velocity,
    const planning_scene::PlanningScene & planning_scene,
    const bool segment_validation_required,
    const CommandAttempt attempt,
    std::string & failure_reason,
    const control::CollisionConstraintDiagnostics & collision_diagnostics,
    CandidateRejectionKind & rejection_kind);

  std::optional<moveit_servo::KinematicState> make_next_state(
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const Eigen::VectorXd & joint_velocity,
    std::string & failure_reason);

  CommandPublicationStatus enqueue_and_publish(
    const moveit_servo::KinematicState & current_state,
    ValidatedCommand command,
    const rclcpp::Time & current_time,
    std::string & failure_reason);

  void publish_status(const std::int8_t code, std::string message);

  void enter_safety_wait(std::string message);

  void reset_feedback_derivative_history();

  void enter_startup_retry(std::string message);

  void handle_recoverable_control_gap(std::string message);

  void enter_controlled_rearm(std::string message);

  ControllerParameters parameters_;
  CommandSession commands_;
  ServoTelemetry telemetry_;
  std::mutex initialization_mutex_;
  std::optional<rclcpp::PreShutdownCallbackHandle> shutdown_callback_;
  control::JointMotionLimits motion_limits_;
  Eigen::VectorXd physical_joint_velocity_limits_;
  servo::Params servo_parameters_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
  const moveit::core::JointModelGroup * joint_model_group_{nullptr};
  const moveit::core::LinkModel * command_link_{nullptr};
  std::unique_ptr<control::MotionReference> motion_reference_;
  std::unique_ptr<control::HierarchicalVelocityQp> velocity_qp_;
  std::unique_ptr<control::EmergencyBrakeTailGenerator> emergency_brake_tail_generator_;
  std::unique_ptr<control::CollisionConstraintBuilder> collision_constraint_builder_;

  bool use_tracking_target_{false};
  rclcpp::Subscription<msg::TrackingTarget>::SharedPtr tracking_target_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
  rclcpp::Subscription<controller_manager_msgs::msg::ControllerManagerActivity>::SharedPtr
    controller_activity_subscription_;
  rclcpp::Subscription<control_msgs::msg::JointTrajectoryControllerState>::SharedPtr
    controller_state_subscription_;
  rclcpp::Publisher<moveit_msgs::msg::ServoStatus>::SharedPtr status_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_publisher_;
  rclcpp::Service<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_command_type_service_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr pause_service_;
  rclcpp::TimerBase::SharedPtr initialization_timer_;

  mutable std::mutex target_mutex_;
  std::optional<msg::TrackingTarget> latest_pose_;
  std::atomic<std::uint64_t> pose_message_count_{0};
  std::atomic<std::uint64_t> pose_accepted_count_{0};
  std::atomic<std::uint64_t> pose_invalid_count_{0};
  std::atomic<std::uint64_t> pose_future_count_{0};
  std::atomic<std::uint64_t> pose_not_newer_count_{0};
  mutable std::mutex controller_state_mutex_;
  std::optional<control_msgs::msg::JointTrajectoryControllerState> latest_controller_state_;
  bool initial_state_received_{false};
  ControllerMode controller_mode_{ControllerMode::kBraking};
  std::string latched_halt_reason_;
  std::uint64_t last_validated_scene_revision_{0};
  std::optional<rclcpp::Time> last_feedback_sample_time_;
  std::optional<Eigen::VectorXd> last_feedback_velocity_;
  std::optional<Eigen::VectorXd> last_feedback_acceleration_;

  std::thread control_thread_;
  std::atomic<bool> initialization_started_{false};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> stop_control_{false};
  std::atomic<bool> accept_pose_commands_{false};
  std::atomic<bool> controller_activity_received_{false};
  std::atomic<bool> trajectory_controller_active_{false};
  std::atomic<std::uint64_t> planning_scene_collision_revision_{0};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> paused_{false};
  std::atomic<bool> halt_latched_{false};
  std::atomic<bool> pause_terminal_pending_{false};
  std::atomic<bool> reset_queue_requested_{false};
};


}  // namespace face_tracking_arm

#endif  // COLLISION_AWARE_SERVO_COMPONENT_HPP_
