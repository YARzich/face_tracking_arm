// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__DETAIL__SCALAR_BRAKING_POLICY_HPP_
#define FACE_TRACKING_ARM__DETAIL__SCALAR_BRAKING_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace face_tracking_arm::control::detail
{

struct ScalarBrakingState
{
  double position{0.0};
  double velocity{0.0};
  double acceleration{0.0};
};

struct ScalarBrakingLimits
{
  double maximum_velocity{0.0};
  double maximum_acceleration{0.0};
  double maximum_jerk{0.0};
};

struct ScalarBrakingOptions
{
  double period_sec{0.01};
  double comparison_tolerance{1.0e-9};
  std::size_t maximum_braking_steps{1000U};
  std::size_t adverse_ticks{0U};
};

enum class ScalarBrakingStatus
{
  kSuccess,
  kInvalidInput,
  kInfeasible,
  kMaximumStepsExceeded,
};

struct ScalarBrakingRollout
{
  ScalarBrakingStatus status{ScalarBrakingStatus::kInvalidInput};
  std::vector<ScalarBrakingState> points;
  double minimum_position{0.0};
  double maximum_position{0.0};
  double minimum_velocity{0.0};
  double maximum_velocity{0.0};

  [[nodiscard]] bool commandAvailable() const noexcept
  {
    return status == ScalarBrakingStatus::kSuccess;
  }
};

namespace scalar_braking_policy
{

constexpr std::size_t kMaximumRecoverySteps = 10000U;

[[nodiscard]] inline bool finiteState(const ScalarBrakingState & state)
{
  return std::isfinite(state.position) && std::isfinite(state.velocity) &&
         std::isfinite(state.acceleration);
}

[[nodiscard]] inline bool validInputs(
  const ScalarBrakingState & state, const ScalarBrakingLimits & limits,
  const ScalarBrakingOptions & options)
{
  return finiteState(state) && std::isfinite(limits.maximum_velocity) &&
         limits.maximum_velocity > 0.0 &&
         std::isfinite(limits.maximum_acceleration) &&
         limits.maximum_acceleration > 0.0 &&
         std::isfinite(limits.maximum_jerk) && limits.maximum_jerk > 0.0 &&
         std::isfinite(options.period_sec) && options.period_sec > 0.0 &&
         std::isfinite(options.comparison_tolerance) &&
         options.comparison_tolerance >= 0.0 &&
         options.maximum_braking_steps > 0U &&
         std::abs(state.velocity) <=
         limits.maximum_velocity + options.comparison_tolerance &&
         std::abs(state.acceleration) <=
         limits.maximum_acceleration + options.comparison_tolerance;
}

[[nodiscard]] inline double moveTowardZero(
  const double value, const double maximum_delta)
{
  if (value > maximum_delta) {
    return value - maximum_delta;
  }
  if (value < -maximum_delta) {
    return value + maximum_delta;
  }
  return 0.0;
}

[[nodiscard]] inline double recoveryVelocityLoss(
  double signed_acceleration, const double acceleration_step,
  const double period_sec)
{
  double loss = 0.0;
  for (std::size_t step = 0; step < kMaximumRecoverySteps; ++step) {
    signed_acceleration = std::min(0.0, signed_acceleration + acceleration_step);
    if (signed_acceleration >= 0.0) {
      return loss;
    }
    loss -= signed_acceleration * period_sec;
    if (!std::isfinite(loss)) {
      return std::numeric_limits<double>::infinity();
    }
  }
  return std::numeric_limits<double>::infinity();
}

struct ScalarBrakingStep
{
  bool valid{false};
  double velocity{0.0};
  double acceleration{0.0};
};

[[nodiscard]] inline ScalarBrakingStep nextStep(
  const double velocity, const double acceleration,
  const ScalarBrakingLimits & limits, const ScalarBrakingOptions & options)
{
  // The terminal root must be stable enough that the reconstructed trajectory
  // remains inside the hard jerk bound after finite-difference validation.
  constexpr int kBisectionIterations = 64;
  const double period = options.period_sec;
  const double acceleration_step = limits.maximum_jerk * period;
  const double tolerance = options.comparison_tolerance;

  if (velocity == 0.0 && acceleration == 0.0) {
    return {true, 0.0, 0.0};
  }

  if (velocity != 0.0) {
    const double terminal_acceleration = -velocity / period;
    if (std::abs(terminal_acceleration) <= limits.maximum_acceleration + tolerance &&
      std::abs(terminal_acceleration - acceleration) <= acceleration_step + tolerance &&
      std::abs(terminal_acceleration) <= acceleration_step + tolerance)
    {
      return {true, 0.0, terminal_acceleration};
    }
  } else if (std::abs(acceleration) <= acceleration_step + tolerance) {
    return {true, 0.0, 0.0};
  }

  if (velocity == 0.0) {
    const double next_acceleration = moveTowardZero(acceleration, acceleration_step);
    return {true, next_acceleration * period, next_acceleration};
  }

  const double direction = std::copysign(1.0, velocity);
  const double speed = direction * velocity;
  const double signed_acceleration = direction * acceleration;
  const double minimum_acceleration = std::max(
    -limits.maximum_acceleration, signed_acceleration - acceleration_step);
  const double maximum_acceleration = std::min(
    limits.maximum_acceleration, signed_acceleration + acceleration_step);

  const auto viability_slack = [&](const double candidate_acceleration) {
      const double next_speed = speed + candidate_acceleration * period;
      return next_speed - recoveryVelocityLoss(
        candidate_acceleration, acceleration_step, period);
    };

  const double maximum_slack = viability_slack(maximum_acceleration);
  if (!std::isfinite(maximum_slack)) {
    return {};
  }
  if (maximum_slack < -tolerance) {
    const double next_acceleration = moveTowardZero(acceleration, acceleration_step);
    const double next_velocity = velocity + next_acceleration * period;
    if (std::abs(next_velocity) > limits.maximum_velocity + tolerance) {
      return {};
    }
    return {true, next_velocity, next_acceleration};
  }

  double selected_acceleration = minimum_acceleration;
  const double minimum_slack = viability_slack(minimum_acceleration);
  if (!std::isfinite(minimum_slack)) {
    return {};
  }
  if (minimum_slack < 0.0) {
    double unsafe = minimum_acceleration;
    double safe = maximum_acceleration;
    for (int iteration = 0; iteration < kBisectionIterations; ++iteration) {
      const double candidate = 0.5 * (unsafe + safe);
      if (viability_slack(candidate) >= 0.0) {
        safe = candidate;
      } else {
        unsafe = candidate;
      }
    }
    selected_acceleration = safe;
  }

  const double next_acceleration = direction * selected_acceleration;
  const double next_velocity = velocity + next_acceleration * period;
  if (!std::isfinite(next_velocity) || !std::isfinite(next_acceleration) ||
    std::abs(next_velocity) > limits.maximum_velocity + tolerance)
  {
    return {};
  }
  return {true, next_velocity, next_acceleration};
}

}  // namespace scalar_braking_policy

/// Roll out the exact scalar braking policy used to construct the published tail.
///
/// QP safety bounds call this without recording points. The tail generator calls
/// the same function with recording enabled and therefore cannot follow a
/// different stopping trajectory. Optional adverse ticks model already-committed
/// motion toward the positive direction before braking starts.
[[nodiscard]] inline ScalarBrakingRollout rolloutScalarBraking(
  const ScalarBrakingState & initial_state, const ScalarBrakingLimits & limits,
  const ScalarBrakingOptions & options, const bool record_points)
{
  ScalarBrakingRollout result;
  if (!scalar_braking_policy::validInputs(initial_state, limits, options)) {
    return result;
  }

  ScalarBrakingState state = initial_state;
  result.minimum_position = state.position;
  result.maximum_position = state.position;
  result.minimum_velocity = state.velocity;
  result.maximum_velocity = state.velocity;
  if (record_points) {
    result.points.push_back(state);
  }

  const auto append_state = [&](const ScalarBrakingState & next_state) {
      state = next_state;
      result.minimum_position = std::min(result.minimum_position, state.position);
      result.maximum_position = std::max(result.maximum_position, state.position);
      result.minimum_velocity = std::min(result.minimum_velocity, state.velocity);
      result.maximum_velocity = std::max(result.maximum_velocity, state.velocity);
      if (record_points) {
        result.points.push_back(state);
      }
    };
  const auto dynamically_valid = [&](const ScalarBrakingState & previous,
    const ScalarBrakingState & next) {
      const double tolerance = options.comparison_tolerance;
      const double jerk_tolerance = tolerance / options.period_sec;
      const double jerk = (next.acceleration - previous.acceleration) / options.period_sec;
      return scalar_braking_policy::finiteState(next) &&
             std::abs(next.velocity) <= limits.maximum_velocity + tolerance &&
             std::abs(next.acceleration) <= limits.maximum_acceleration + tolerance &&
             std::abs(jerk) <= limits.maximum_jerk + jerk_tolerance;
    };

  const double acceleration_step = limits.maximum_jerk * options.period_sec;
  for (std::size_t tick = 0; tick < options.adverse_ticks; ++tick) {
    ScalarBrakingState next = state;
    next.acceleration = std::min(
      limits.maximum_acceleration, state.acceleration + acceleration_step);
    next.velocity = state.velocity + next.acceleration * options.period_sec;
    next.position = state.position + next.velocity * options.period_sec;
    if (!dynamically_valid(state, next)) {
      result.status = ScalarBrakingStatus::kInfeasible;
      return result;
    }
    append_state(next);
  }

  if (state.velocity == 0.0 && state.acceleration == 0.0) {
    result.status = ScalarBrakingStatus::kSuccess;
    return result;
  }

  for (std::size_t step = 0; step < options.maximum_braking_steps; ++step) {
    const scalar_braking_policy::ScalarBrakingStep joint_step =
      scalar_braking_policy::nextStep(
      state.velocity, state.acceleration, limits, options);
    if (!joint_step.valid) {
      result.status = ScalarBrakingStatus::kInfeasible;
      return result;
    }

    ScalarBrakingState next;
    next.velocity = joint_step.velocity;
    next.acceleration = joint_step.acceleration;
    next.position = state.position + next.velocity * options.period_sec;
    if (!dynamically_valid(state, next)) {
      result.status = ScalarBrakingStatus::kInfeasible;
      return result;
    }
    append_state(next);
    if (state.velocity == 0.0 && state.acceleration == 0.0) {
      result.status = ScalarBrakingStatus::kSuccess;
      return result;
    }
  }

  result.status = ScalarBrakingStatus::kMaximumStepsExceeded;
  return result;
}

}  // namespace face_tracking_arm::control::detail

#endif  // FACE_TRACKING_ARM__DETAIL__SCALAR_BRAKING_POLICY_HPP_
