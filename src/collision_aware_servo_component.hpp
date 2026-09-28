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

enum class ControllerMode
{
  kTracking,
  kBraking,
  kWaitingForSafeState,
};

enum class CommandAttempt
{
  kTracking,
  kBraking,
};

enum class CommandPublicationStatus
{
  kSuccess,
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

enum class TimelineRecoveryStatus
{
  kSuccess,
  kExhaustedPublishedSuffix,
  kInvariantFailure,
};

struct ExpectedCommandState
{
  Eigen::VectorXd positions;
  Eigen::VectorXd velocities;
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

inline constexpr double kFiniteEpsilon = 1.0e-12;

inline constexpr double kStoppedVelocityRadps = 1.0e-2;

inline constexpr double kSafeRearmSettleBudgetSec = 2.0;

// The JTC reconstructs the same spline with finite timestamp/float precision.
// Tens of microradians remain far inside the collision tracking reserve.

inline constexpr double kControllerCommandAlignmentToleranceRad = 1.0e-4;

inline constexpr double kControllerSpeedScalingTolerance = 1.0e-9;

inline constexpr std::size_t kTelemetryWindowSamples = 1000;

inline constexpr std::size_t kPublishedTrajectoryHistoryCapacity = 16;

inline const char * command_attempt_name(const CommandAttempt attempt)
{
  return attempt == CommandAttempt::kTracking ? "tracking" : "braking";
}

inline std::int64_t seconds_to_nanoseconds(const double seconds)
{
  if (!std::isfinite(seconds) || seconds < 0.0) {
    throw std::invalid_argument("duration must be finite and non-negative");
  }
  constexpr double nanoseconds_per_second = 1.0e9;
  const double nanoseconds = seconds * nanoseconds_per_second;
  if (nanoseconds > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("duration cannot be represented in nanoseconds");
  }
  return static_cast<std::int64_t>(std::llround(nanoseconds));
}

inline bool finite_state(const moveit_servo::KinematicState & state, const std::size_t size)
{
  return state.joint_names.size() == size &&
         static_cast<std::size_t>(state.positions.size()) == size &&
         static_cast<std::size_t>(state.velocities.size()) == size &&
         static_cast<std::size_t>(state.accelerations.size()) == size &&
         state.positions.allFinite() && state.velocities.allFinite() &&
         state.accelerations.allFinite();
}

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

  void clear_bootstrap_ack_state() noexcept;

  void start_rearm_deadline(const rclcpp::Time & current_time);

  void ensure_rearm_deadline(const rclcpp::Time & current_time);

  void clear_rearm_deadlines() noexcept;

  [[nodiscard]] bool rearm_deadline_expired(
    const rclcpp::Time & current_time,
    const std::chrono::steady_clock::time_point & wall_time) const noexcept;

  void record_healthy_publication() noexcept;

  ActualFeedbackSafety validate_controller_feedback_collision_state(
    const control_msgs::msg::JointTrajectoryControllerState & controller_state,
    const std::string & context, std::string & reason);

  RearmReadiness safe_rearm_ready(
    const rclcpp::Time & current_time, std::string & reason);

  void control_loop();

  void observe_controller_mode() noexcept;

  void publish_runtime_diagnostics();

  void control_tick();

  void process_startup_cycle(
    const moveit_servo::KinematicState & state,
    const moveit::core::RobotState & robot_state,
    const planning_scene::PlanningScene & planning_scene);

  const control::PublishedTrajectoryRecord * active_published_record() const noexcept;

  std::optional<ExpectedCommandState> expected_command_state_at(
    const rclcpp::Time & sample_time) const;

  std::optional<moveit_servo::KinematicState> predicted_or_measured_state(
    const rclcpp::Time & current_time, std::string & failure_reason,
    bool & requires_latched_halt);

  TimelineRecoveryStatus recover_published_timeline(
    const std::uint64_t skipped_periods, const rclcpp::Time & current_time,
    std::string & failure_reason);

  void observe_controller_feedback(
    const control_msgs::msg::JointTrajectoryControllerState & state,
    const rcl_clock_type_t clock_type);

  bool feedback_motion_within_limits(
    const control_msgs::msg::JointTrajectoryControllerState & state,
    std::string & failure_reason) const;

  std::optional<msg::TrackingTarget> current_target(
    const rclcpp::Time & current_time) const;

  void reset_tick_telemetry();

  static void observe_minimum(
    std::optional<double> & observed_minimum,
    const double value) noexcept;

  void observe_clearances(const double self_distance_m, const double world_distance_m) noexcept;

  void observe_motion_state(const control::JointMotionState & state) noexcept;

  void observe_command_transition(
    const control::JointMotionState & current_state,
    const moveit_servo::KinematicState & next_state) noexcept;

  void record_qp_result(
    const std::chrono::steady_clock::time_point & start,
    const control::QpResult & result) noexcept;

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
    const control::CollisionConstraintDiagnostics & collision_diagnostics,
    const double condition_number) const;

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
    const control::CollisionConstraintDiagnostics * path_clearance = nullptr);

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
  std::mutex initialization_mutex_;
  std::optional<rclcpp::PreShutdownCallbackHandle> shutdown_callback_;
  control::JointMotionLimits motion_limits_;
  Eigen::VectorXd physical_joint_velocity_limits_;
  std::unique_ptr<control::ControlTimeGrid> control_time_grid_;
  std::shared_ptr<servo::ParamListener> servo_parameter_listener_;
  servo::Params servo_parameters_;
  planning_scene_monitor::PlanningSceneMonitorPtr planning_scene_monitor_;
  const moveit::core::JointModelGroup * joint_model_group_{nullptr};
  const moveit::core::LinkModel * command_link_{nullptr};
  std::unique_ptr<control::MotionReference> motion_reference_;
  std::unique_ptr<control::HierarchicalVelocityQp> velocity_qp_;
  std::unique_ptr<control::EmergencyBrakeTailGenerator> emergency_brake_tail_generator_;
  std::unique_ptr<control::CollisionConstraintBuilder> collision_constraint_builder_;
  control::PublishedTrajectoryHistory published_trajectory_history_{
    kPublishedTrajectoryHistoryCapacity};
  telemetry::RollingStatistics control_tick_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics control_cycle_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics qp_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics collision_build_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics collision_self_query_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics collision_world_query_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics segment_validation_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics path_validation_statistics_{kTelemetryWindowSamples};
  telemetry::RollingStatistics pointing_error_statistics_{kTelemetryWindowSamples};

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
  // The branch queue is the prefix that may be extended on the next tick. The
  // execution queue is an immutable copy of the latest message, including its
  // already-published braking suffix and the one point withheld by Servo's
  // trajectory composer.
  std::deque<moveit_servo::KinematicState> command_queue_;
  std::optional<control::PublicationId> active_publication_id_;
  std::size_t max_command_queue_points_{0};
  bool command_epoch_active_{false};
  bool command_epoch_has_published_{false};
  bool bootstrap_ack_pending_{false};
  std::optional<control::PublicationId> bootstrap_publication_id_;
  std::optional<rclcpp::Time> bootstrap_trajectory_origin_;
  std::optional<Eigen::VectorXd> bootstrap_seed_positions_;
  bool rearm_pending_{false};
  bool published_tail_rearm_pending_{false};
  std::size_t stable_rearm_samples_{0};
  std::optional<rclcpp::Time> last_rearm_sample_time_;
  std::optional<rclcpp::Time> rearm_deadline_;
  std::optional<std::chrono::steady_clock::time_point> rearm_wall_deadline_;
  std::size_t consecutive_automatic_rearms_{0};
  std::size_t healthy_publications_since_rearm_{0};
  bool initial_state_received_{false};
  ControllerMode controller_mode_{ControllerMode::kBraking};
  std::string latched_halt_reason_;
  std::optional<ControllerMode> last_observed_controller_mode_;
  std::int8_t last_servo_status_code_{moveit_msgs::msg::ServoStatus::INVALID};
  std::string last_servo_status_message_{"Not started"};
  std::optional<double> last_condition_number_;
  double maximum_condition_number_{0.0};
  bool last_singularity_guidance_requested_{false};
  bool last_singularity_guidance_available_{false};
  double last_singularity_guidance_activation_{0.0};
  std::optional<double> last_singularity_gradient_norm_;
  std::optional<double> last_singularity_guidance_reference_rad_s_;
  std::optional<double> last_singularity_guidance_achieved_rad_s_;
  std::optional<double> last_minimum_self_distance_m_;
  std::optional<double> last_minimum_world_distance_m_;
  std::optional<double> last_minimum_robust_headroom_m_;
  std::string last_minimum_robust_headroom_pair_{"unavailable"};
  std::optional<double> minimum_observed_self_distance_m_;
  std::optional<double> minimum_observed_world_distance_m_;
  std::optional<double> minimum_observed_robust_headroom_m_;
  std::string minimum_observed_robust_headroom_pair_{"unavailable"};
  std::optional<double> last_actual_minimum_self_distance_m_;
  std::optional<double> last_actual_minimum_world_distance_m_;
  std::string last_actual_closest_pair_{"unavailable"};
  std::uint64_t actual_state_validation_count_{0};
  std::uint64_t actual_state_validation_failure_count_{0};
  std::uint64_t last_validated_scene_revision_{0};
  std::uint64_t planning_scene_invalidation_count_{0};
  std::string last_closest_pair_{"unavailable"};
  std::size_t last_active_collision_rows_{0};
  std::size_t last_outside_collision_viability_contacts_{0};
  std::optional<double> last_position_error_m_;
  std::optional<double> last_roll_error_rad_;
  std::optional<double> last_screen_normal_pointing_error_rad_;
  double max_observed_joint_velocity_rad_s_{0.0};
  double max_observed_joint_acceleration_rad_s2_{0.0};
  double max_observed_joint_jerk_rad_s3_{0.0};
  std::optional<rclcpp::Time> last_feedback_sample_time_;
  std::optional<Eigen::VectorXd> last_feedback_velocity_;
  std::optional<Eigen::VectorXd> last_feedback_acceleration_;
  std::optional<Eigen::VectorXd> last_feedback_jerk_;
  double max_feedback_joint_velocity_rad_s_{0.0};
  double max_feedback_joint_acceleration_rad_s2_{0.0};
  double max_feedback_joint_jerk_rad_s3_{0.0};
  std::string max_feedback_joint_velocity_name_{"unavailable"};
  std::string max_feedback_joint_acceleration_name_{"unavailable"};
  std::string max_feedback_joint_jerk_name_{"unavailable"};
  std::uint64_t feedback_motion_limit_failure_count_{0};
  double lifetime_maximum_control_tick_ms_{0.0};
  std::uint64_t missed_control_deadlines_{0};
  std::uint64_t skipped_control_periods_{0};
  std::uint64_t timeline_recovery_events_{0};
  std::uint64_t timeline_recovered_periods_{0};
  std::uint64_t timeline_recovery_failures_{0};
  std::uint64_t controlled_rearm_count_{0};
  std::string last_controlled_rearm_reason_{"none"};
  std::uint64_t rearm_timeout_count_{0};
  std::uint64_t rearm_budget_exhaustion_count_{0};
  std::uint64_t startup_retry_count_{0};
  std::string last_startup_retry_reason_{"none"};
  std::uint64_t clock_not_advancing_observations_{0};
  std::uint64_t clock_rewind_count_{0};
  std::uint64_t published_command_count_{0};
  std::optional<double> last_published_horizon_sec_;
  std::optional<double> minimum_published_horizon_sec_;
  std::optional<double> last_replacement_lead_sec_;
  std::optional<double> minimum_replacement_lead_sec_;
  std::size_t startup_warmup_cycles_{0};
  bool startup_warmup_complete_{false};
  std::uint64_t bootstrap_publication_count_{0};
  std::uint64_t bootstrap_acknowledgement_count_{0};
  std::uint64_t bootstrap_segment_validation_count_{0};
  std::optional<double> last_controller_speed_scaling_factor_;
  std::uint64_t controller_speed_scaling_failure_count_{0};
  std::optional<double> last_controller_reference_alignment_error_rad_;
  std::optional<double> last_controller_output_alignment_error_rad_;
  double max_controller_reference_alignment_error_rad_{0.0};
  double max_controller_output_alignment_error_rad_{0.0};
  std::uint64_t command_alignment_check_count_{0};
  std::uint64_t command_alignment_failure_count_{0};
  std::uint64_t older_publication_recovery_count_{0};
  std::optional<double> last_following_position_error_rad_;
  std::optional<double> last_following_velocity_error_rad_s_;
  std::string last_following_position_error_joint_{"unavailable"};
  std::string last_following_velocity_error_joint_{"unavailable"};
  double max_following_position_error_rad_{0.0};
  double max_following_velocity_error_rad_s_{0.0};
  std::string max_following_position_error_joint_{"unavailable"};
  std::string max_following_velocity_error_joint_{"unavailable"};
  std::uint64_t following_check_count_{0};
  std::uint64_t following_error_count_{0};
  std::uint64_t solver_failure_count_{0};
  std::uint64_t primary_only_count_{0};
  std::uint64_t braking_event_count_{0};
  std::uint64_t braking_tick_count_{0};
  std::uint64_t emergency_event_count_{0};
  std::uint64_t emergency_tick_count_{0};
  std::size_t last_braking_tail_points_{0};
  std::size_t maximum_braking_tail_points_{0};
  std::uint64_t braking_tail_generation_failure_count_{0};
  std::uint64_t braking_tail_segment_validation_count_{0};
  std::uint64_t braking_tail_collision_rejection_count_{0};
  std::uint64_t task_failure_count_{0};
  std::uint64_t moveit_validation_failure_count_{0};
  std::uint64_t segment_validation_count_{0};
  std::uint64_t segment_rejection_count_{0};
  std::uint64_t tracking_segment_rejection_count_{0};
  std::uint64_t braking_segment_rejection_count_{0};
  std::string last_segment_rejection_attempt_{"none"};
  std::string last_segment_rejection_reason_{"none"};
  bool last_segment_rejection_input_valid_{false};
  bool last_segment_rejection_unsafe_{false};
  std::size_t last_segment_rejection_evaluated_samples_{0};
  std::size_t last_segment_rejection_first_unsafe_sample_{
    std::numeric_limits<std::size_t>::max()};
  std::string last_segment_rejection_closest_pair_{"none"};
  std::optional<double> last_segment_rejection_first_unsafe_fraction_;
  std::optional<double> last_segment_rejection_minimum_self_distance_m_;
  std::optional<double> last_segment_rejection_minimum_world_distance_m_;
  std::optional<double> last_segment_rejection_minimum_distance_m_;

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
