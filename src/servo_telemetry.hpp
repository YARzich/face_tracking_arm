// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef SERVO_TELEMETRY_HPP_
#define SERVO_TELEMETRY_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <moveit_msgs/msg/servo_status.hpp>
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/rolling_statistics.hpp"
#include "servo_types.hpp"

namespace face_tracking_arm
{

// Observations only: no publishers, locks, or decisions about robot motion.
// Updated and read by the control thread; input callbacks have separate atomics.
struct ServoTelemetry
{
  static constexpr std::size_t kTelemetryWindowSamples = 1000;

  void observeMode(ControllerMode mode) noexcept;
  void resetTick();
  static void observeMinimum(std::optional<double> & observed_minimum, double value) noexcept;
  void observeClearances(double self_distance_m, double world_distance_m) noexcept;
  void observeMotionState(const control::JointMotionState & state) noexcept;
  void observeCommandTransition(
    const control::JointMotionState & current_state,
    const moveit_servo::KinematicState & next_state, double period_sec) noexcept;
  void recordQpResult(
    const std::chrono::steady_clock::time_point & start, const control::QpResult & result) noexcept;
  void recordSegmentRejection(
    CommandAttempt attempt, const control::CollisionSegmentValidationResult & result);

  telemetry::RollingStatistics control_tick_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics control_cycle_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics qp_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics collision_build_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics collision_self_query_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics collision_world_query_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics segment_validation_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics path_validation_statistics{kTelemetryWindowSamples};
  telemetry::RollingStatistics pointing_error_statistics{kTelemetryWindowSamples};

  std::optional<ControllerMode> last_observed_controller_mode;
  std::int8_t last_servo_status_code{moveit_msgs::msg::ServoStatus::INVALID};
  std::string last_servo_status_message{"Not started"};
  std::optional<double> last_minimum_self_distance_m;
  std::optional<double> last_minimum_world_distance_m;
  std::optional<double> last_minimum_robust_headroom_m;
  std::string last_minimum_robust_headroom_pair{"unavailable"};
  std::optional<double> minimum_observed_self_distance_m;
  std::optional<double> minimum_observed_world_distance_m;
  std::optional<double> minimum_observed_robust_headroom_m;
  std::string minimum_observed_robust_headroom_pair{"unavailable"};
  std::optional<double> last_actual_minimum_self_distance_m;
  std::optional<double> last_actual_minimum_world_distance_m;
  std::string last_actual_closest_pair{"unavailable"};
  std::uint64_t actual_state_validation_count{0};
  std::uint64_t actual_state_validation_failure_count{0};
  std::uint64_t planning_scene_invalidation_count{0};
  std::string last_closest_pair{"unavailable"};
  std::size_t last_active_collision_rows{0};
  std::size_t last_outside_collision_viability_contacts{0};
  std::optional<double> last_position_error_m;
  std::optional<double> last_roll_error_rad;
  std::optional<double> last_screen_normal_pointing_error_rad;
  double max_observed_joint_velocity_rad_s{0.0};
  double max_observed_joint_acceleration_rad_s2{0.0};
  double max_observed_joint_jerk_rad_s3{0.0};
  double max_feedback_joint_velocity_rad_s{0.0};
  double max_feedback_joint_acceleration_rad_s2{0.0};
  double max_feedback_joint_jerk_rad_s3{0.0};
  std::string max_feedback_joint_velocity_name{"unavailable"};
  std::string max_feedback_joint_acceleration_name{"unavailable"};
  std::string max_feedback_joint_jerk_name{"unavailable"};
  std::uint64_t feedback_motion_limit_failure_count{0};
  double lifetime_maximum_control_tick_ms{0.0};
  std::uint64_t missed_control_deadlines{0};
  std::uint64_t skipped_control_periods{0};
  std::uint64_t timeline_recovery_failures{0};
  std::uint64_t controlled_rearm_count{0};
  std::string last_controlled_rearm_reason{"none"};
  std::uint64_t rearm_timeout_count{0};
  std::uint64_t startup_retry_count{0};
  std::string last_startup_retry_reason{"none"};
  std::uint64_t clock_not_advancing_observations{0};
  std::uint64_t clock_rewind_count{0};
  std::uint64_t published_command_count{0};
  std::optional<double> last_published_horizon_sec;
  std::optional<double> minimum_published_horizon_sec;
  std::optional<double> last_replacement_lead_sec;
  std::optional<double> minimum_replacement_lead_sec;
  std::uint64_t bootstrap_publication_count{0};
  std::uint64_t bootstrap_acknowledgement_count{0};
  std::uint64_t bootstrap_segment_validation_count{0};
  std::optional<double> last_controller_speed_scaling_factor;
  std::uint64_t controller_speed_scaling_failure_count{0};
  std::optional<double> last_controller_reference_alignment_error_rad;
  std::optional<double> last_controller_output_alignment_error_rad;
  double max_controller_reference_alignment_error_rad{0.0};
  double max_controller_output_alignment_error_rad{0.0};
  std::uint64_t command_alignment_check_count{0};
  std::uint64_t command_alignment_failure_count{0};
  std::uint64_t older_publication_recovery_count{0};
  std::optional<double> last_following_position_error_rad;
  std::optional<double> last_following_velocity_error_rad_s;
  std::string last_following_position_error_joint{"unavailable"};
  std::string last_following_velocity_error_joint{"unavailable"};
  double max_following_position_error_rad{0.0};
  double max_following_velocity_error_rad_s{0.0};
  std::string max_following_position_error_joint{"unavailable"};
  std::string max_following_velocity_error_joint{"unavailable"};
  std::uint64_t following_check_count{0};
  std::uint64_t following_error_count{0};
  std::uint64_t solver_failure_count{0};
  std::uint64_t primary_only_count{0};
  std::uint64_t braking_event_count{0};
  std::uint64_t braking_tick_count{0};
  std::uint64_t emergency_event_count{0};
  std::uint64_t emergency_tick_count{0};
  std::size_t last_braking_tail_points{0};
  std::size_t maximum_braking_tail_points{0};
  std::uint64_t braking_tail_generation_failure_count{0};
  std::uint64_t braking_tail_segment_validation_count{0};
  std::uint64_t braking_tail_collision_rejection_count{0};
  std::uint64_t braking_tail_distance_query_count{0};
  std::uint64_t braking_tail_certified_path_block_count{0};
  std::uint64_t braking_tail_inconclusive_path_block_count{0};
  std::size_t last_braking_tail_certified_segment_count{0};
  std::size_t last_braking_tail_uncertified_segment_count{0};
  std::uint64_t moveit_validation_failure_count{0};
  std::uint64_t segment_validation_count{0};
  std::uint64_t segment_rejection_count{0};
  std::uint64_t tracking_segment_rejection_count{0};
  std::uint64_t braking_segment_rejection_count{0};
  std::string last_segment_rejection_attempt{"none"};
  std::string last_segment_rejection_reason{"none"};
  bool last_segment_rejection_input_valid{false};
  bool last_segment_rejection_unsafe{false};
  std::size_t last_segment_rejection_evaluated_samples{0};
  std::size_t last_segment_rejection_first_unsafe_sample{
    std::numeric_limits<std::size_t>::max()};
  std::string last_segment_rejection_closest_pair{"none"};
  std::optional<double> last_segment_rejection_first_unsafe_fraction;
  std::optional<double> last_segment_rejection_minimum_self_distance_m;
  std::optional<double> last_segment_rejection_minimum_world_distance_m;
  std::optional<double> last_segment_rejection_minimum_distance_m;
};

}  // namespace face_tracking_arm

#endif  // SERVO_TELEMETRY_HPP_
