// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/collision_constraints.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "collision_stopping.hpp"
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"

namespace face_tracking_arm::control::collision_detail
{

namespace
{

constexpr std::size_t kMaximumSimulationTicks = 10000U;

[[nodiscard]] detail::ScalarBrakingRollout rolloutJointTowardCollision(
  const double position, const double velocity, const double acceleration,
  const double direction, const double maximum_velocity,
  const double maximum_acceleration, const double maximum_jerk,
  const JointMotionLimits & motion_limits, const double period_sec,
  const std::size_t adverse_ticks, const bool record_points)
{
  if (!motion_limits.numerical_policy.has_value() ||
    !motion_limits.numerical_policy->valid())
  {
    throw std::invalid_argument("joint numerical policy is missing or invalid");
  }

  detail::ScalarBrakingOptions options;
  options.period_sec = period_sec;
  options.numerical_policy = *motion_limits.numerical_policy;
  options.maximum_braking_steps = 1000U;
  options.adverse_ticks = adverse_ticks;
  return detail::rolloutScalarBraking(
    detail::ScalarBrakingState{
        direction * position, direction * velocity, direction * acceleration},
    detail::ScalarBrakingLimits{
        maximum_velocity, maximum_acceleration, maximum_jerk}, options, record_points);
}

[[nodiscard]] double collisionTailReserveForReachableBox(
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const ReachableVelocityInterval & velocity_interval,
  const double period_sec, const double residual_latency_sec)
{
  const double residual_ticks_value = std::ceil(residual_latency_sec / period_sec);
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks))
  {
    throw std::invalid_argument("vector collision stopping horizon is invalid");
  }
  const std::size_t adverse_ticks = static_cast<std::size_t>(residual_ticks_value);

  double total_reserve = 0.0;
  for (Eigen::Index index = 0; index < distance_gradient.size(); ++index) {
    const double gradient = distance_gradient[index];
    if (gradient == 0.0) {
      continue;
    }

    const double toward_direction = gradient > 0.0 ? -1.0 : 1.0;
    const double endpoint_velocity = toward_direction > 0.0 ?
      velocity_interval.upper[index] : velocity_interval.lower[index];
    const double endpoint_acceleration =
      (endpoint_velocity - motion_state.velocity[index]) / period_sec;
    const double endpoint_position =
      motion_state.position[index] + endpoint_velocity * period_sec;
    const detail::ScalarBrakingRollout rollout = rolloutJointTowardCollision(
      endpoint_position, endpoint_velocity, endpoint_acceleration,
      toward_direction, motion_limits.max_velocity[index],
      motion_limits.max_acceleration[index], motion_limits.max_jerk[index],
      motion_limits, period_sec, adverse_ticks, false);
    if (!rollout.commandAvailable()) {
      // An infeasible extreme cannot contribute a finite reserve. The caller
      // then falls back to the most-receding reachable cap and the final tail
      // certificate remains the acceptance gate for every concrete command.
      return std::numeric_limits<double>::infinity();
    }
    const double minimum_joint_position = toward_direction > 0.0 ?
      rollout.minimum_position : -rollout.maximum_position;
    const double maximum_joint_position = toward_direction > 0.0 ?
      rollout.maximum_position : -rollout.minimum_position;
    if (minimum_joint_position < motion_limits.lower_position[index] -
      kComparisonTolerance || maximum_joint_position >
      motion_limits.upper_position[index] + kComparisonTolerance)
    {
      return std::numeric_limits<double>::infinity();
    }
    total_reserve += std::abs(gradient) * std::max(
      0.0, rollout.maximum_position - toward_direction * endpoint_position);
  }
  return total_reserve;
}

}  // namespace

[[nodiscard]] bool computeReachableVelocityInterval(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const double period_sec,
  ReachableVelocityInterval & interval,
  std::string & error)
{
  const Eigen::Index joint_count = state.position.size();
  interval.lower.resize(joint_count);
  interval.upper.resize(joint_count);
  for (Eigen::Index index = 0; index < joint_count; ++index) {
    const double minimum_acceleration = std::max(
      -limits.max_acceleration[index],
      state.acceleration[index] - limits.max_jerk[index] * period_sec);
    const double maximum_acceleration = std::min(
      limits.max_acceleration[index],
      state.acceleration[index] + limits.max_jerk[index] * period_sec);

    const double safe_lower_position =
      limits.lower_position[index];
    const double safe_upper_position =
      limits.upper_position[index];
    interval.lower[index] = std::max({
        -limits.max_velocity[index],
        state.velocity[index] + minimum_acceleration * period_sec,
        (safe_lower_position - state.position[index]) / period_sec});
    interval.upper[index] = std::min({
        limits.max_velocity[index],
        state.velocity[index] + maximum_acceleration * period_sec,
        (safe_upper_position - state.position[index]) / period_sec});
    if (!std::isfinite(interval.lower[index]) || !std::isfinite(interval.upper[index])) {
      error = "reachable joint-velocity interval is non-finite";
      return false;
    }
    if (interval.lower[index] > interval.upper[index]) {
      error = "reachable joint-velocity interval is empty";
      return false;
    }
  }
  return true;
}

[[nodiscard]] ReachableApproachInterval projectApproachInterval(
  const Eigen::VectorXd & distance_gradient,
  const ReachableVelocityInterval & velocity_interval)
{
  ReachableApproachInterval result;
  result.lower_contribution.resize(distance_gradient.size());
  result.upper_contribution.resize(distance_gradient.size());
  for (Eigen::Index index = 0; index < distance_gradient.size(); ++index) {
    if (distance_gradient[index] >= 0.0) {
      result.lower_contribution[index] =
        -distance_gradient[index] * velocity_interval.upper[index];
      result.upper_contribution[index] =
        -distance_gradient[index] * velocity_interval.lower[index];
    } else {
      result.lower_contribution[index] =
        -distance_gradient[index] * velocity_interval.lower[index];
      result.upper_contribution[index] =
        -distance_gradient[index] * velocity_interval.upper[index];
    }
    result.minimum_mps += result.lower_contribution[index];
    result.maximum_mps += result.upper_contribution[index];
  }
  return result;
}

[[nodiscard]] double requiredCollisionDistanceForApproachCap(
  const double cap,
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const ReachableVelocityInterval & velocity_interval,
  const ReachableApproachInterval & approach_interval,
  const double period_sec,
  const double residual_latency_sec)
{
  if (!std::isfinite(cap) || cap < approach_interval.minimum_mps - kComparisonTolerance ||
    cap > approach_interval.maximum_mps + kComparisonTolerance)
  {
    throw std::invalid_argument("approach cap lies outside the reachable interval");
  }
  const double tail_reserve = collisionTailReserveForReachableBox(
    distance_gradient, motion_state, motion_limits, velocity_interval,
    period_sec, residual_latency_sec);
  return std::max(0.0, cap * period_sec + tail_reserve);
}

[[nodiscard]] bool validateMotionInput(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index expected_size,
  std::string & error)
{
  const auto correct_size = [expected_size](const Eigen::VectorXd & value) {
      return value.size() == expected_size;
    };
  if (!correct_size(state.position) || !correct_size(state.velocity) ||
    !correct_size(state.acceleration) || !correct_size(limits.lower_position) ||
    !correct_size(limits.upper_position) || !correct_size(limits.position_margin) ||
    !correct_size(limits.max_velocity) ||
    !correct_size(limits.max_acceleration) || !correct_size(limits.max_jerk))
  {
    error = "motion state/limit dimensions do not match the JointModelGroup";
    return false;
  }
  if (!isFinite(state.position) || !isFinite(state.velocity) ||
    !isFinite(state.acceleration) || !isFinite(limits.lower_position) ||
    !isFinite(limits.upper_position) || !isFinite(limits.position_margin) ||
    !isFinite(limits.max_velocity) ||
    !isFinite(limits.max_acceleration) || !isFinite(limits.max_jerk))
  {
    error = "motion state/limits contain non-finite values";
    return false;
  }
  if ((limits.lower_position.array() >= limits.upper_position.array()).any() ||
    (limits.position_margin.array() < 0.0).any() ||
    ((limits.lower_position + limits.position_margin).array() >
    (limits.upper_position - limits.position_margin).array()).any() ||
    (limits.max_velocity.array() <= 0.0).any() ||
    (limits.max_acceleration.array() <= 0.0).any() ||
    (limits.max_jerk.array() <= 0.0).any())
  {
    error = "motion magnitude limits must be positive";
    return false;
  }
  return true;
}

[[nodiscard]] bool validateSegmentMotionState(
  const JointMotionState & state,
  const Eigen::Index expected_size,
  std::string & error)
{
  if (state.position.size() != expected_size || state.velocity.size() != expected_size ||
    state.acceleration.size() != expected_size)
  {
    error = "segment motion-state dimensions do not match the JointModelGroup";
    return false;
  }
  if (!isFinite(state.position) || !isFinite(state.velocity) ||
    !isFinite(state.acceleration))
  {
    error = "segment motion state contains non-finite values";
    return false;
  }
  return true;
}

}  // namespace face_tracking_arm::control::collision_detail

namespace face_tracking_arm::control
{

using collision_detail::ReachableVelocityInterval;
using collision_detail::ReachableApproachInterval;
using collision_detail::computeReachableVelocityInterval;
using collision_detail::projectApproachInterval;
using collision_detail::requiredCollisionDistanceForApproachCap;
using collision_detail::validateMotionInput;
using collision_detail::isFinite;
using collision_detail::kComparisonTolerance;
using collision_detail::kMaximumSimulationTicks;
using collision_detail::rolloutJointTowardCollision;
using collision_detail::collisionTailReserveForReachableBox;

double jerkLimitedStoppingDistance(
  const double toward_velocity,
  const double toward_acceleration,
  const double max_acceleration,
  const double max_jerk,
  const double period_sec,
  const double residual_latency_sec)
{
  constexpr std::size_t kMaximumSimulationTicks = 10000;
  if (!std::isfinite(toward_velocity) || !std::isfinite(toward_acceleration) ||
    !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
    !std::isfinite(max_jerk) || max_jerk <= 0.0 ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0 ||
    std::abs(toward_acceleration) > max_acceleration + kComparisonTolerance)
  {
    throw std::invalid_argument("invalid jerk-limited stopping-distance input");
  }

  const double residual_ticks_value = std::ceil(residual_latency_sec / period_sec);
  const double acceleration_step = max_jerk * period_sec;
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks) ||
    !std::isfinite(acceleration_step) || acceleration_step <= 0.0)
  {
    throw std::invalid_argument("discrete stopping horizon is invalid");
  }
  const auto residual_ticks = static_cast<std::size_t>(residual_ticks_value);

  double position = 0.0;
  double velocity = toward_velocity;
  double acceleration = toward_acceleration;
  double maximum_position = 0.0;
  std::size_t simulated_ticks = 0;
  const auto integrate_tick = [&]() {
      velocity += acceleration * period_sec;
      position += velocity * period_sec;
      if (!std::isfinite(position) || !std::isfinite(velocity) ||
        !std::isfinite(acceleration))
      {
        throw std::invalid_argument("discrete stopping trajectory is non-finite");
      }
      maximum_position = std::max(maximum_position, position);
      ++simulated_ticks;
    };

  for (std::size_t tick = 0; tick < residual_ticks; ++tick) {
    acceleration = std::min(max_acceleration, acceleration + acceleration_step);
    integrate_tick();
  }
  while ((velocity > 0.0 || acceleration > 0.0) &&
    simulated_ticks < kMaximumSimulationTicks)
  {
    acceleration = std::max(-max_acceleration, acceleration - acceleration_step);
    integrate_tick();
  }
  if (velocity > 0.0 || acceleration > 0.0) {
    throw std::invalid_argument("discrete stopping trajectory exceeds bounded horizon");
  }
  return std::max(0.0, maximum_position);
}

double maximumSafeVelocityTowardBoundary(
  const double remaining_distance,
  const double current_velocity_toward_boundary,
  const double minimum_reachable_velocity,
  const double maximum_reachable_velocity,
  const double max_acceleration,
  const double max_jerk,
  const double period_sec,
  const double residual_latency_sec)
{
  if (!std::isfinite(remaining_distance) ||
    !std::isfinite(current_velocity_toward_boundary) ||
    !std::isfinite(minimum_reachable_velocity) ||
    !std::isfinite(maximum_reachable_velocity) ||
    minimum_reachable_velocity > maximum_reachable_velocity ||
    !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
    !std::isfinite(max_jerk) || max_jerk <= 0.0 ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0)
  {
    throw std::invalid_argument("invalid safe-boundary-velocity input");
  }

  const double minimum_acceleration =
    (minimum_reachable_velocity - current_velocity_toward_boundary) / period_sec;
  const double maximum_acceleration_candidate =
    (maximum_reachable_velocity - current_velocity_toward_boundary) / period_sec;
  if (std::abs(minimum_acceleration) > max_acceleration + kComparisonTolerance ||
    std::abs(maximum_acceleration_candidate) > max_acceleration + kComparisonTolerance)
  {
    throw std::invalid_argument(
            "reachable boundary-velocity interval violates the acceleration limit");
  }

  const auto required_distance = [&](const double candidate_velocity) {
      const double candidate_acceleration =
        (candidate_velocity - current_velocity_toward_boundary) / period_sec;
      return candidate_velocity * period_sec + jerkLimitedStoppingDistance(
        candidate_velocity, candidate_acceleration, max_acceleration, max_jerk,
        period_sec, residual_latency_sec);
    };

  if (required_distance(minimum_reachable_velocity) > remaining_distance) {
    return minimum_reachable_velocity;
  }
  if (required_distance(maximum_reachable_velocity) <= remaining_distance) {
    return maximum_reachable_velocity;
  }

  double lower = minimum_reachable_velocity;
  double upper = maximum_reachable_velocity;
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    if (required_distance(middle) <= remaining_distance) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

double jerkLimitedCollisionStoppingDistance(
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const Eigen::VectorXd & candidate_joint_velocity,
  const JointMotionLimits & motion_limits,
  const double period_sec,
  const double residual_latency_sec)
{
  const Eigen::Index joint_count = distance_gradient.size();
  std::string input_error;
  if (joint_count <= 0 || candidate_joint_velocity.size() != joint_count ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0 ||
    !isFinite(distance_gradient) || !isFinite(candidate_joint_velocity) ||
    !validateMotionInput(motion_state, motion_limits, joint_count, input_error))
  {
    throw std::invalid_argument(
            input_error.empty() ?
            "invalid vector collision stopping-distance input" : input_error);
  }

  const Eigen::VectorXd candidate_acceleration =
    (candidate_joint_velocity - motion_state.velocity) / period_sec;
  const Eigen::VectorXd candidate_position =
    motion_state.position + candidate_joint_velocity * period_sec;
  const Eigen::VectorXd minimum_position =
    motion_limits.lower_position;
  const Eigen::VectorXd maximum_joint_position =
    motion_limits.upper_position;
  if (!isFinite(candidate_acceleration) || !isFinite(candidate_position) ||
    (candidate_joint_velocity.cwiseAbs().array() >
    motion_limits.max_velocity.array() + kComparisonTolerance).any() ||
    (candidate_acceleration.cwiseAbs().array() >
    motion_limits.max_acceleration.array() + kComparisonTolerance).any() ||
    ((candidate_acceleration - motion_state.acceleration).cwiseAbs().array() >
    motion_limits.max_jerk.array() * period_sec + kComparisonTolerance).any() ||
    (candidate_position.array() < minimum_position.array() - kComparisonTolerance).any() ||
    (candidate_position.array() > maximum_joint_position.array() + kComparisonTolerance).any())
  {
    throw std::invalid_argument("candidate joint velocity is not reachable in one period");
  }

  if (distance_gradient.cwiseAbs().maxCoeff() <= 0.0) {
    throw std::invalid_argument("distance gradient has no controllable projected limits");
  }

  const double residual_ticks_value = std::ceil(residual_latency_sec / period_sec);
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks))
  {
    throw std::invalid_argument("vector collision stopping horizon is invalid");
  }
  const std::size_t adverse_ticks = static_cast<std::size_t>(residual_ticks_value);

  std::vector<detail::ScalarBrakingRollout> joint_rollouts;
  joint_rollouts.reserve(static_cast<std::size_t>(joint_count));
  std::size_t maximum_points = 1U;
  for (Eigen::Index index = 0; index < joint_count; ++index) {
    const double gradient = distance_gradient[index];
    if (gradient == 0.0) {
      detail::ScalarBrakingRollout stationary;
      stationary.status = detail::ScalarBrakingStatus::kSuccess;
      stationary.points.push_back(detail::ScalarBrakingState{
          candidate_position[index], candidate_joint_velocity[index],
          candidate_acceleration[index]});
      joint_rollouts.push_back(std::move(stationary));
      continue;
    }
    const double toward_direction = gradient > 0.0 ? -1.0 : 1.0;
    detail::ScalarBrakingRollout rollout = rolloutJointTowardCollision(
      candidate_position[index], candidate_joint_velocity[index],
      candidate_acceleration[index], toward_direction,
      motion_limits.max_velocity[index], motion_limits.max_acceleration[index],
      motion_limits.max_jerk[index], motion_limits, period_sec, adverse_ticks, true);
    if (!rollout.commandAvailable()) {
      throw std::invalid_argument(
              "shared scalar collision braking rollout is infeasible at joint " +
              std::to_string(index));
    }
    const double minimum_joint_position = toward_direction > 0.0 ?
      rollout.minimum_position : -rollout.maximum_position;
    const double maximum_joint_position = toward_direction > 0.0 ?
      rollout.maximum_position : -rollout.minimum_position;
    if (minimum_joint_position < motion_limits.lower_position[index] -
      kComparisonTolerance || maximum_joint_position >
      motion_limits.upper_position[index] + kComparisonTolerance)
    {
      throw std::invalid_argument(
              "shared scalar collision braking rollout violates a joint position bound");
    }
    maximum_points = std::max(maximum_points, rollout.points.size());
    joint_rollouts.push_back(std::move(rollout));
  }

  double maximum_projected_approach = 0.0;
  for (std::size_t tick = 0; tick < maximum_points; ++tick) {
    double projected_approach = 0.0;
    for (Eigen::Index index = 0; index < joint_count; ++index) {
      const double gradient = distance_gradient[index];
      if (gradient == 0.0) {
        continue;
      }
      const auto & points = joint_rollouts[static_cast<std::size_t>(index)].points;
      const detail::ScalarBrakingState & point = points[std::min(tick, points.size() - 1U)];
      const detail::ScalarBrakingState & start = points.front();
      projected_approach += std::abs(gradient) *
        (point.position - start.position);
    }
    if (!std::isfinite(projected_approach)) {
      throw std::invalid_argument("shared scalar collision rollout is non-finite");
    }
    maximum_projected_approach = std::max(maximum_projected_approach, projected_approach);
  }
  return maximum_projected_approach;
}

double maximumSafeCollisionApproachSpeed(
  const double remaining_distance,
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const double period_sec,
  const double residual_latency_sec)
{
  const Eigen::Index joint_count = distance_gradient.size();
  std::string input_error;
  if (joint_count <= 0 || !std::isfinite(remaining_distance) || !isFinite(distance_gradient) ||
    !validateMotionInput(motion_state, motion_limits, joint_count, input_error))
  {
    throw std::invalid_argument("invalid collision approach-envelope input");
  }

  ReachableVelocityInterval reachable_velocity;
  std::string interval_error;
  if (!computeReachableVelocityInterval(
      motion_state, motion_limits, period_sec, reachable_velocity, interval_error))
  {
    throw std::invalid_argument(interval_error);
  }
  const ReachableApproachInterval reachable_approach =
    projectApproachInterval(distance_gradient, reachable_velocity);
  const double tail_reserve = collisionTailReserveForReachableBox(
    distance_gradient, motion_state, motion_limits, reachable_velocity,
    period_sec, residual_latency_sec);
  double cap = reachable_approach.minimum_mps;
  if (std::isfinite(tail_reserve)) {
    cap = std::clamp(
      (remaining_distance - tail_reserve) / period_sec,
      reachable_approach.minimum_mps, reachable_approach.maximum_mps);
  }

  // A dynamically valid first command from the shared emergency-braking
  // policy remains available even when the box reserve is larger than the
  // clearance. This is only a QP row floor; segment and full-tail certificates
  // still validate every concrete command before publication.
  EmergencyBrakeTailConfig tail_config;
  tail_config.period_sec = period_sec;
  tail_config.terminal_hold_points = 2U;
  const EmergencyBrakeTail brake_anchor = EmergencyBrakeTailGenerator(
    tail_config).generate(motion_state, motion_limits);
  if (brake_anchor.command_available() && brake_anchor.points.size() >= 2U) {
    const Eigen::VectorXd anchor_velocity = brake_anchor.points[1].velocity;
    bool anchor_reachable = anchor_velocity.size() == joint_count;
    for (Eigen::Index index = 0; index < joint_count && anchor_reachable; ++index) {
      anchor_reachable = anchor_velocity[index] >=
        reachable_velocity.lower[index] - kComparisonTolerance &&
        anchor_velocity[index] <=
        reachable_velocity.upper[index] + kComparisonTolerance;
    }
    if (anchor_reachable) {
      cap = std::max(cap, -distance_gradient.dot(anchor_velocity));
    }
  }
  return std::clamp(
    cap, reachable_approach.minimum_mps, reachable_approach.maximum_mps);
}

}  // namespace face_tracking_arm::control
