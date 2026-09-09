// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/emergency_brake_tail.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "face_tracking_arm/detail/scalar_braking_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

[[nodiscard]] bool finite(const Eigen::VectorXd & value)
{
  return value.array().isFinite().all();
}

[[nodiscard]] bool validConfig(const EmergencyBrakeTailConfig & config)
{
  return std::isfinite(config.period_sec) && config.period_sec > 0.0 &&
         config.maximum_braking_steps > 0U && config.terminal_hold_points >= 2U &&
         std::isfinite(config.comparison_tolerance) &&
         config.comparison_tolerance > 0.0;
}

[[nodiscard]] bool validInput(
  const JointMotionState & state, const JointMotionLimits & limits,
  const double tolerance, std::string & failure_reason)
{
  const Eigen::Index joint_count = state.position.size();
  if (joint_count <= 0 || state.velocity.size() != joint_count ||
    state.acceleration.size() != joint_count ||
    limits.lower_position.size() != joint_count ||
    limits.upper_position.size() != joint_count ||
    limits.position_margin.size() != joint_count ||
    limits.max_velocity.size() != joint_count ||
    limits.max_acceleration.size() != joint_count ||
    limits.max_jerk.size() != joint_count)
  {
    failure_reason = "motion state and limit dimensions do not match";
    return false;
  }
  if (!finite(state.position) || !finite(state.velocity) ||
    !finite(state.acceleration) || !finite(limits.lower_position) ||
    !finite(limits.upper_position) || !finite(limits.position_margin) ||
    !finite(limits.max_velocity) || !finite(limits.max_acceleration) ||
    !finite(limits.max_jerk))
  {
    failure_reason = "motion state and limits must be finite";
    return false;
  }

  for (Eigen::Index index = 0; index < joint_count; ++index) {
    const double safe_lower =
      limits.lower_position[index] + limits.position_margin[index];
    const double safe_upper =
      limits.upper_position[index] - limits.position_margin[index];
    if (limits.lower_position[index] >= limits.upper_position[index] ||
      limits.position_margin[index] < 0.0 || safe_lower > safe_upper ||
      limits.max_velocity[index] <= 0.0 ||
      limits.max_acceleration[index] <= 0.0 || limits.max_jerk[index] <= 0.0)
    {
      failure_reason = "joint motion limits are invalid";
      return false;
    }
    if (state.position[index] < safe_lower - tolerance ||
      state.position[index] > safe_upper + tolerance)
    {
      failure_reason = "branch position is outside the safe joint interval";
      return false;
    }
    if (std::abs(state.velocity[index]) > limits.max_velocity[index] + tolerance ||
      std::abs(state.acceleration[index]) >
      limits.max_acceleration[index] + tolerance)
    {
      failure_reason = "branch velocity or acceleration exceeds its joint limit";
      return false;
    }
  }
  return true;
}

}  // namespace

bool EmergencyBrakeTail::command_available() const noexcept
{
  return status == EmergencyBrakeTailStatus::kSuccess && !points.empty();
}

EmergencyBrakeTailGenerator::EmergencyBrakeTailGenerator(
  EmergencyBrakeTailConfig config)
: config_(std::move(config))
{
  if (!validConfig(config_)) {
    throw std::invalid_argument("emergency braking-tail configuration is invalid");
  }
}

EmergencyBrakeTail EmergencyBrakeTailGenerator::generate(
  const JointMotionState & branch_state, const JointMotionLimits & limits) const
{
  EmergencyBrakeTail result;
  if (!validInput(
      branch_state, limits, config_.comparison_tolerance, result.failure_reason))
  {
    return result;
  }

  std::vector<detail::ScalarBrakingRollout> joint_rollouts;
  joint_rollouts.reserve(static_cast<std::size_t>(branch_state.position.size()));
  std::size_t maximum_braking_steps = 0U;
  for (Eigen::Index index = 0; index < branch_state.position.size(); ++index) {
    detail::ScalarBrakingOptions options;
    options.period_sec = config_.period_sec;
    options.comparison_tolerance = config_.comparison_tolerance;
    options.maximum_braking_steps = config_.maximum_braking_steps;
    const detail::ScalarBrakingRollout rollout = detail::rolloutScalarBraking(
      detail::ScalarBrakingState{
        branch_state.position[index], branch_state.velocity[index],
        branch_state.acceleration[index]},
      detail::ScalarBrakingLimits{
        limits.max_velocity[index], limits.max_acceleration[index],
        limits.max_jerk[index]},
      options, true);
    if (rollout.status == detail::ScalarBrakingStatus::kMaximumStepsExceeded) {
      result.status = EmergencyBrakeTailStatus::kMaximumStepsExceeded;
      result.failure_reason = "braking tail did not reach an exact stationary state";
      return result;
    }
    if (!rollout.commandAvailable()) {
      result.status = EmergencyBrakeTailStatus::kInfeasible;
      result.failure_reason =
        "no dynamically valid braking step exists for joint index " +
        std::to_string(index);
      return result;
    }

    const double safe_lower =
      limits.lower_position[index] + limits.position_margin[index];
    const double safe_upper =
      limits.upper_position[index] - limits.position_margin[index];
    if (rollout.minimum_position < safe_lower - config_.comparison_tolerance ||
      rollout.maximum_position > safe_upper + config_.comparison_tolerance)
    {
      result.status = EmergencyBrakeTailStatus::kInfeasible;
      result.failure_reason =
        "braking tail would violate a joint bound at index " + std::to_string(index);
      return result;
    }
    maximum_braking_steps = std::max(
      maximum_braking_steps, rollout.points.size() - 1U);
    joint_rollouts.push_back(std::move(rollout));
  }

  result.points.reserve(maximum_braking_steps + config_.terminal_hold_points + 1U);
  for (std::size_t step = 0; step <= maximum_braking_steps; ++step) {
    JointMotionState state;
    state.position.resize(branch_state.position.size());
    state.velocity.resize(branch_state.position.size());
    state.acceleration.resize(branch_state.position.size());
    for (Eigen::Index index = 0; index < branch_state.position.size(); ++index) {
      const auto & rollout = joint_rollouts[static_cast<std::size_t>(index)];
      const auto & scalar_state = rollout.points[std::min(step, rollout.points.size() - 1U)];
      state.position[index] = scalar_state.position;
      state.velocity[index] = scalar_state.velocity;
      state.acceleration[index] = scalar_state.acceleration;
    }
    result.points.push_back(std::move(state));
  }
  result.first_stationary_point = maximum_braking_steps;

  for (std::size_t point = 0; point < config_.terminal_hold_points; ++point) {
    result.points.push_back(result.points[result.first_stationary_point]);
  }
  result.status = EmergencyBrakeTailStatus::kSuccess;
  return result;
}

const EmergencyBrakeTailConfig & EmergencyBrakeTailGenerator::config() const noexcept
{
  return config_;
}

}  // namespace face_tracking_arm::control
