// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/collision_constraints.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "collision_stopping.hpp"

namespace face_tracking_arm::control::collision_detail
{

namespace
{

/// Bound every joint's approach contribution under a common approach cap.
/// Shared slack is intentionally reused per joint for a conservative stopping bound.
[[nodiscard]] Eigen::VectorXd conservativeCandidateVelocityForApproachCap(
  const double cap,
  const Eigen::VectorXd & distance_gradient,
  const ReachableVelocityInterval & velocity_interval,
  const ReachableApproachInterval & approach_interval)
{
  if (!std::isfinite(cap) ||
    cap < approach_interval.minimum_mps - kComparisonTolerance ||
    cap > approach_interval.maximum_mps + kComparisonTolerance)
  {
    throw std::invalid_argument("approach cap lies outside the reachable interval");
  }

  Eigen::VectorXd candidate_velocity(distance_gradient.size());
  for (Eigen::Index index = 0; index < distance_gradient.size(); ++index) {
    if (distance_gradient[index] == 0.0) {
      candidate_velocity[index] = velocity_interval.lower[index];
      continue;
    }
    const double contribution_cap = std::clamp(
      cap - (approach_interval.minimum_mps -
      approach_interval.lower_contribution[index]),
      approach_interval.lower_contribution[index],
      approach_interval.upper_contribution[index]);
    candidate_velocity[index] = std::clamp(
      -contribution_cap / distance_gradient[index],
      velocity_interval.lower[index], velocity_interval.upper[index]);
  }
  return candidate_velocity;
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
  const Eigen::VectorXd candidate_velocity =
    conservativeCandidateVelocityForApproachCap(
    cap, distance_gradient, velocity_interval, approach_interval);
  const double stopping_distance = jerkLimitedCollisionStoppingDistance(
    distance_gradient, motion_state, candidate_velocity, motion_limits,
    period_sec, residual_latency_sec);
  return std::max(0.0, cap * period_sec + stopping_distance);
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
  constexpr std::size_t kMaximumSimulationTicks = 10000;
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

  const double residual_ticks_value = std::ceil(residual_latency_sec / period_sec);
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks))
  {
    throw std::invalid_argument("vector collision stopping horizon is invalid");
  }
  const auto residual_ticks = static_cast<std::size_t>(residual_ticks_value);

  // Work in per-joint approach coordinates. This preserves every individual
  // acceleration/jerk saturation while making both gradient signs symmetric.
  Eigen::VectorXd velocity_contribution =
    -distance_gradient.array() * candidate_joint_velocity.array();
  Eigen::VectorXd acceleration_contribution =
    -distance_gradient.array() * candidate_acceleration.array();
  const Eigen::VectorXd maximum_acceleration_contribution =
    distance_gradient.cwiseAbs().array() * motion_limits.max_acceleration.array();
  const Eigen::VectorXd jerk_step =
    distance_gradient.cwiseAbs().array() * motion_limits.max_jerk.array() * period_sec;
  if (!isFinite(velocity_contribution) || !isFinite(acceleration_contribution) ||
    !isFinite(maximum_acceleration_contribution) || !isFinite(jerk_step) ||
    maximum_acceleration_contribution.maxCoeff() <= 0.0)
  {
    throw std::invalid_argument("distance gradient has no controllable projected limits");
  }

  double position = 0.0;
  double maximum_position = 0.0;
  std::size_t simulated_ticks = 0;
  const auto integrate_tick = [&]() {
      velocity_contribution += acceleration_contribution * period_sec;
      const double approach_velocity = velocity_contribution.sum();
      position += approach_velocity * period_sec;
      if (!isFinite(velocity_contribution) || !isFinite(acceleration_contribution) ||
        !std::isfinite(approach_velocity) || !std::isfinite(position))
      {
        throw std::invalid_argument("vector collision stopping trajectory is non-finite");
      }
      maximum_position = std::max(maximum_position, position);
      ++simulated_ticks;
    };

  for (std::size_t tick = 0; tick < residual_ticks; ++tick) {
    acceleration_contribution =
      (acceleration_contribution + jerk_step).cwiseMin(
      maximum_acceleration_contribution);
    integrate_tick();
  }

  auto approach_velocity = velocity_contribution.sum();
  auto approach_acceleration = acceleration_contribution.sum();
  while ((approach_velocity > 0.0 || approach_acceleration > 0.0) &&
    simulated_ticks < kMaximumSimulationTicks)
  {
    acceleration_contribution =
      (acceleration_contribution - jerk_step).cwiseMax(
      -maximum_acceleration_contribution);
    integrate_tick();
    approach_velocity = velocity_contribution.sum();
    approach_acceleration = acceleration_contribution.sum();
  }
  if (approach_velocity > 0.0 || approach_acceleration > 0.0) {
    throw std::invalid_argument("vector collision stopping trajectory exceeds bounded horizon");
  }
  return std::max(0.0, maximum_position);
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
  const auto required_distance = [&](const double candidate_approach_speed) {
      return requiredCollisionDistanceForApproachCap(
        candidate_approach_speed, distance_gradient, motion_state, motion_limits,
        reachable_velocity, reachable_approach, period_sec, residual_latency_sec);
    };

  if (required_distance(reachable_approach.minimum_mps) > remaining_distance) {
    return reachable_approach.minimum_mps;
  }
  if (required_distance(reachable_approach.maximum_mps) <= remaining_distance) {
    return reachable_approach.maximum_mps;
  }

  double lower = reachable_approach.minimum_mps;
  double upper = reachable_approach.maximum_mps;
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

}  // namespace face_tracking_arm::control
