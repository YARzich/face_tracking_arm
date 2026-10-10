// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "servo_telemetry.hpp"

#include <algorithm>
#include <cmath>

namespace face_tracking_arm
{

void ServoTelemetry::observeMode(const ControllerMode mode) noexcept
{
  if (mode == ControllerMode::kBraking) {
    ++braking_tick_count;
  } else if (mode == ControllerMode::kWaitingForSafeState) {
    ++emergency_tick_count;
  }

  if (!last_observed_controller_mode.has_value() ||
    *last_observed_controller_mode != mode)
  {
    if (mode == ControllerMode::kBraking) {
      ++braking_event_count;
    } else if (mode == ControllerMode::kWaitingForSafeState) {
      ++emergency_event_count;
    }
    last_observed_controller_mode = mode;
  }
}

void ServoTelemetry::resetTick()
{
  last_minimum_self_distance_m.reset();
  last_minimum_world_distance_m.reset();
  last_minimum_robust_headroom_m.reset();
  last_minimum_robust_headroom_pair = "unavailable";
  last_closest_pair = "unavailable";
  last_active_collision_rows = 0;
  last_outside_collision_viability_contacts = 0;
  last_position_error_m.reset();
  last_roll_error_rad.reset();
  last_screen_normal_pointing_error_rad.reset();
}

void ServoTelemetry::observeMinimum(
  std::optional<double> & observed_minimum,
  const double value) noexcept
{
  if (!std::isfinite(value)) {
    return;
  }
  if (!observed_minimum.has_value() || value < *observed_minimum) {
    observed_minimum = value;
  }
}

void ServoTelemetry::observeClearances(
  const double self_distance_m,
  const double world_distance_m) noexcept
{
  observeMinimum(minimum_observed_self_distance_m, self_distance_m);
  observeMinimum(minimum_observed_world_distance_m, world_distance_m);
}

void ServoTelemetry::observeMotionState(
  const control::JointMotionState & state) noexcept
{
  if (state.velocity.size() > 0 && state.velocity.allFinite()) {
    max_observed_joint_velocity_rad_s = std::max(
      max_observed_joint_velocity_rad_s, state.velocity.cwiseAbs().maxCoeff());
  }
  if (state.acceleration.size() > 0 && state.acceleration.allFinite()) {
    max_observed_joint_acceleration_rad_s2 = std::max(
      max_observed_joint_acceleration_rad_s2,
      state.acceleration.cwiseAbs().maxCoeff());
  }
}

void ServoTelemetry::observeCommandTransition(
  const control::JointMotionState & current_state,
  const moveit_servo::KinematicState & next_state, const double period_sec) noexcept
{
  if (next_state.velocities.size() > 0 && next_state.velocities.allFinite()) {
    max_observed_joint_velocity_rad_s = std::max(
      max_observed_joint_velocity_rad_s, next_state.velocities.cwiseAbs().maxCoeff());
  }
  if (next_state.accelerations.size() > 0 && next_state.accelerations.allFinite()) {
    max_observed_joint_acceleration_rad_s2 = std::max(
      max_observed_joint_acceleration_rad_s2,
      next_state.accelerations.cwiseAbs().maxCoeff());
  }
  if (current_state.acceleration.size() == next_state.accelerations.size() &&
    next_state.accelerations.size() > 0)
  {
    double maximum_jerk = 0.0;
    for (Eigen::Index index = 0; index < next_state.accelerations.size(); ++index) {
      const double jerk = std::abs(
        (next_state.accelerations[index] - current_state.acceleration[index]) /
        period_sec);
      if (!std::isfinite(jerk)) {
        return;
      }
      maximum_jerk = std::max(maximum_jerk, jerk);
    }
    max_observed_joint_jerk_rad_s3 = std::max(
      max_observed_joint_jerk_rad_s3, maximum_jerk);
  }
}

void ServoTelemetry::recordQpResult(
  const std::chrono::steady_clock::time_point & start,
  const control::QpResult & result) noexcept
{
  const double duration_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  (void)qp_statistics.add(duration_ms);
  if (!result.command_available()) {
    ++solver_failure_count;
  }
  if (result.status == control::QpStatus::kPrimaryOnly) {
    ++primary_only_count;
  }
}

void ServoTelemetry::recordSegmentRejection(
  const CommandAttempt attempt,
  const control::CollisionSegmentValidationResult & result)
{
  ++segment_rejection_count;
  if (attempt == CommandAttempt::kTracking) {
    ++tracking_segment_rejection_count;
  } else {
    ++braking_segment_rejection_count;
  }
  last_segment_rejection_attempt = command_attempt_name(attempt);
  last_segment_rejection_reason = result.failure_reason.empty() ?
    "hard clearance or protected-joint corridor violated" : result.failure_reason;
  last_segment_rejection_input_valid = result.input_valid;
  last_segment_rejection_unsafe = result.unsafe;
  last_segment_rejection_evaluated_samples = result.evaluated_samples;
  last_segment_rejection_first_unsafe_sample = result.first_unsafe_sample;
  last_segment_rejection_closest_pair = result.closest_pair.empty() ?
    "unavailable" : result.closest_pair;
  if (std::isfinite(result.first_unsafe_fraction)) {
    last_segment_rejection_first_unsafe_fraction = result.first_unsafe_fraction;
  } else {
    last_segment_rejection_first_unsafe_fraction.reset();
  }
  if (std::isfinite(result.minimum_self_distance_m)) {
    last_segment_rejection_minimum_self_distance_m = result.minimum_self_distance_m;
  } else {
    last_segment_rejection_minimum_self_distance_m.reset();
  }
  if (std::isfinite(result.minimum_world_distance_m)) {
    last_segment_rejection_minimum_world_distance_m = result.minimum_world_distance_m;
  } else {
    last_segment_rejection_minimum_world_distance_m.reset();
  }
  if (std::isfinite(result.minimum_distance_m)) {
    last_segment_rejection_minimum_distance_m = result.minimum_distance_m;
  } else {
    last_segment_rejection_minimum_distance_m.reset();
  }
}

}  // namespace face_tracking_arm
