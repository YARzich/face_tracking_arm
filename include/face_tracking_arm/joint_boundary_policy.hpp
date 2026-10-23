// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__JOINT_BOUNDARY_POLICY_HPP_
#define FACE_TRACKING_ARM__JOINT_BOUNDARY_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace face_tracking_arm::control
{

// Distinct wrappers keep tolerances with different units and purposes apart.
struct Radians
{
  double value;
};

inline constexpr Radians kControllerPositionFeedbackTolerance{1.0e-6};

struct RadiansPerSecond
{
  double value;
};
struct RadiansPerSecondSquared
{
  double value;
};
struct RadiansPerSecondCubed
{
  double value;
};
struct SolverResidual
{
  double value;
};

struct PositionInterval;

/// IEEE-754 forward-error bound for an accumulated sum of per-operation operand
/// magnitudes. `absolute_operand_sum` is the sum across the path, not the scale
/// of only its final result. For subtraction/division, callers propagate the
/// absolute operand sum through the divisor; this covers cancellation. The
/// operation count bounds propagation of earlier rounding error.
[[nodiscard]] inline std::optional<double> machineRoundoffBound(
  const double absolute_operand_sum, const std::size_t rounded_operations) noexcept
{
  if (!std::isfinite(absolute_operand_sum) || absolute_operand_sum < 0.0 ||
    rounded_operations == 0U)
  {
    return std::nullopt;
  }
  const double product = static_cast<double>(rounded_operations) *
    std::numeric_limits<double>::epsilon();
  if (!std::isfinite(product) || product >= 1.0) {
    return std::nullopt;
  }
  return (product / (1.0 - product)) *
         absolute_operand_sum;
}

/// Roundoff bound for a finite difference, retaining cancellation-sensitive inputs.
[[nodiscard]] inline std::optional<double> machineDifferenceQuotientBound(
  const double upper_operand, const double lower_operand, const double divisor,
  const std::size_t rounded_operations) noexcept
{
  if (!std::isfinite(upper_operand) || !std::isfinite(lower_operand) ||
    !std::isfinite(divisor) || divisor == 0.0)
  {
    return std::nullopt;
  }
  const double operand_sum =
    (std::abs(upper_operand) + std::abs(lower_operand)) / std::abs(divisor);
  return machineRoundoffBound(operand_sum, rounded_operations);
}

struct NumericalPolicy
{
  Radians position_error;
  RadiansPerSecond velocity_error;
  RadiansPerSecondSquared acceleration_error;
  RadiansPerSecondCubed jerk_error;
  SolverResidual solver_residual;

  /// Builds a shared policy for the discrete backward-Euler motion path.
  /// q/v/a/j allowances cover the local stored-state expressions only; they do
  /// not bound divergence from an ideal trajectory over a tracking epoch.
  /// Jerk uses the finite-difference operand magnitudes and the local rounding
  /// of the acceleration values, all divided by dt. These are arithmetic
  /// bounds only; they do not model physical execution error or an ideal
  /// trajectory's divergence from the stored discrete states.
  [[nodiscard]] static std::optional<NumericalPolicy> fromDiscreteMotionLimits(
    const PositionInterval & position_interval, const double maximum_velocity_radps,
    const double maximum_acceleration_radps2, const double maximum_jerk_radps3,
    const double period_sec, const SolverResidual residual) noexcept;

  /// Derives distinct q/v/a/jerk allowances from scales and operation counts.
  /// The residual is supplied separately and never contributes to these bounds.
  [[nodiscard]] static std::optional<NumericalPolicy> fromOperationBudgets(
    const double position_operand_sum_rad, const std::size_t position_operations,
    const double velocity_operand_sum_radps, const std::size_t velocity_operations,
    const double acceleration_operand_sum_radps2,
    const std::size_t acceleration_operations,
    const double jerk_operand_sum_radps3, const std::size_t jerk_operations,
    const SolverResidual residual) noexcept
  {
    const auto q = machineRoundoffBound(position_operand_sum_rad, position_operations);
    const auto v = machineRoundoffBound(velocity_operand_sum_radps, velocity_operations);
    const auto a = machineRoundoffBound(
      acceleration_operand_sum_radps2, acceleration_operations);
    const auto j = machineRoundoffBound(jerk_operand_sum_radps3, jerk_operations);
    if (!q || !v || !a || !j) {
      return std::nullopt;
    }
    NumericalPolicy policy{
      Radians{*q}, RadiansPerSecond{*v}, RadiansPerSecondSquared{*a},
      RadiansPerSecondCubed{*j}, residual};
    return policy.valid() ? std::optional<NumericalPolicy>{policy} : std::nullopt;
  }

  [[nodiscard]] bool valid() const noexcept
  {
    return finiteNonnegative(position_error.value) &&
           finiteNonnegative(velocity_error.value) &&
           finiteNonnegative(acceleration_error.value) &&
           finiteNonnegative(jerk_error.value) &&
           finiteNonnegative(solver_residual.value);
  }

private:
  [[nodiscard]] static bool finiteNonnegative(const double value) noexcept
  {
    return std::isfinite(value) && value >= 0.0;
  }
};

struct PositionInterval
{
  double lower_rad;
  double upper_rad;

  [[nodiscard]] bool valid() const noexcept
  {
    return std::isfinite(lower_rad) && std::isfinite(upper_rad) &&
           lower_rad < upper_rad;
  }

  [[nodiscard]] bool contains(const double position_rad) const noexcept
  {
    return std::isfinite(position_rad) &&
           position_rad >= lower_rad && position_rad <= upper_rad;
  }
};

inline std::optional<NumericalPolicy> NumericalPolicy::fromDiscreteMotionLimits(
  const PositionInterval & position_interval, const double maximum_velocity_radps,
  const double maximum_acceleration_radps2, const double maximum_jerk_radps3,
  const double period_sec, const SolverResidual residual) noexcept
{
  if (!position_interval.valid() || !std::isfinite(maximum_velocity_radps) ||
    maximum_velocity_radps <= 0.0 || !std::isfinite(maximum_acceleration_radps2) ||
    maximum_acceleration_radps2 <= 0.0 || !std::isfinite(maximum_jerk_radps3) ||
    maximum_jerk_radps3 <= 0.0 || !std::isfinite(period_sec) || period_sec <= 0.0)
  {
    return std::nullopt;
  }
  const double position_scale = std::max(
    std::abs(position_interval.lower_rad), std::abs(position_interval.upper_rad));
  const double velocity_step = maximum_velocity_radps * period_sec;
  // Each q candidate is compared using its actual stored q and one computed
  // increment. Do not bound the difference from an unrounded trajectory over
  // a tracking epoch: the next stored q becomes the next operation's input.
  // Acceleration may be formed either by v+a*dt or by (v1-v0)/dt; take the
  // larger local operand scale. Jerk's quotient must account both for its own
  // cancellation-sensitive operands and for that local acceleration rounding.
  const double position_operands = position_scale + velocity_step;
  const double velocity_operands = maximum_velocity_radps +
    maximum_acceleration_radps2 * period_sec;
  const double acceleration_recurrence_operands = maximum_acceleration_radps2 +
    maximum_jerk_radps3 * period_sec;
  const double acceleration_difference_operands =
    2.0 * maximum_velocity_radps / period_sec;
  const double acceleration_operands = std::max(
    acceleration_recurrence_operands, acceleration_difference_operands);
  const double jerk_operands = 2.0 * maximum_acceleration_radps2 / period_sec;
  if (!std::isfinite(velocity_step) || !std::isfinite(position_operands) ||
    !std::isfinite(velocity_operands) || !std::isfinite(acceleration_operands) ||
    !std::isfinite(jerk_operands))
  {
    return std::nullopt;
  }
  auto policy = fromOperationBudgets(
    position_operands, 2U,
    velocity_operands, 2U,
    acceleration_operands, 2U,
    jerk_operands, 2U, residual);
  if (!policy) {
    return std::nullopt;
  }
  const double propagated_acceleration_error =
    2.0 * policy->acceleration_error.value / period_sec;
  if (!std::isfinite(propagated_acceleration_error)) {
    return std::nullopt;
  }
  policy->jerk_error.value += propagated_acceleration_error;
  return policy->valid() ? policy : std::nullopt;
}

enum class BoundaryClass
{
  kInterior,
  kBoundary,
  kOutside,
  kInvalid,
};

/// Classifies q against the same interval semantics for feedback, corridor and rearm.
/// The caller provides q's explicit numerical error budget; solver residual is never used.
[[nodiscard]] inline BoundaryClass classifyPosition(
  const double position_rad, const PositionInterval & interval,
  const Radians numerical_error) noexcept
{
  if (!interval.valid() || !std::isfinite(position_rad) ||
    !std::isfinite(numerical_error.value) || numerical_error.value < 0.0)
  {
    return BoundaryClass::kInvalid;
  }
  if (position_rad < interval.lower_rad) {
    return interval.lower_rad - position_rad <= numerical_error.value ?
           BoundaryClass::kBoundary : BoundaryClass::kOutside;
  }
  if (position_rad > interval.upper_rad) {
    return position_rad - interval.upper_rad <= numerical_error.value ?
           BoundaryClass::kBoundary : BoundaryClass::kOutside;
  }
  return std::min(position_rad - interval.lower_rad,
           interval.upper_rad - position_rad) <= numerical_error.value ?
         BoundaryClass::kBoundary : BoundaryClass::kInterior;
}

/// Half-ULP uncertainty for a feedback angle read as float then promoted to double.
/// This describes sample representation only; it is never an execution reserve.
[[nodiscard]] inline std::optional<Radians> floatMeasurementHalfUlp(
  const double promoted_float_sample_rad) noexcept
{
  if (!std::isfinite(promoted_float_sample_rad)) {
    return std::nullopt;
  }
  const float sample = static_cast<float>(promoted_float_sample_rad);
  if (!std::isfinite(sample) || static_cast<double>(sample) != promoted_float_sample_rad) {
    return std::nullopt;
  }
  const float lower = std::nextafter(sample, -std::numeric_limits<float>::infinity());
  const float upper = std::nextafter(sample, std::numeric_limits<float>::infinity());
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    return std::nullopt;
  }
  const double half_ulp = 0.5 * std::max(
    static_cast<double>(sample) - static_cast<double>(lower),
    static_cast<double>(upper) - static_cast<double>(sample));
  return Radians{half_ulp};
}

/// Classifies a promoted float measurement using its representational uncertainty.
/// A boundary result is uncertain and must not by itself authorize motion.
[[nodiscard]] inline BoundaryClass classifyMeasuredPosition(
  const double promoted_float_sample_rad, const PositionInterval & interval) noexcept
{
  const auto measurement_error = floatMeasurementHalfUlp(promoted_float_sample_rad);
  return measurement_error ? classifyPosition(
    promoted_float_sample_rad, interval, *measurement_error) : BoundaryClass::kInvalid;
}

/// Exact double-domain check for a candidate before converting it to wire format.
[[nodiscard]] inline bool isNominalPosition(
  const double position_rad, const PositionInterval & nominal) noexcept
{
  return nominal.contains(position_rad);
}

template<typename Quantity>
[[nodiscard]] inline BoundaryClass classifyMagnitude(
  const Quantity value, const Quantity hard_limit, const Quantity numerical_error) noexcept
{
  if (!std::isfinite(value.value) || !std::isfinite(hard_limit.value) ||
    !std::isfinite(numerical_error.value) || hard_limit.value <= 0.0 ||
    numerical_error.value < 0.0)
  {
    return BoundaryClass::kInvalid;
  }
  const double magnitude = std::abs(value.value);
  if (magnitude > hard_limit.value) {
    return magnitude - hard_limit.value <= numerical_error.value ?
           BoundaryClass::kBoundary : BoundaryClass::kOutside;
  }
  return hard_limit.value - magnitude <= numerical_error.value ?
         BoundaryClass::kBoundary : BoundaryClass::kInterior;
}

[[nodiscard]] inline BoundaryClass classifyVelocity(
  const RadiansPerSecond value, const RadiansPerSecond hard_limit,
  const RadiansPerSecond numerical_error) noexcept
{
  return classifyMagnitude(value, hard_limit, numerical_error);
}

[[nodiscard]] inline BoundaryClass classifyAcceleration(
  const RadiansPerSecondSquared value, const RadiansPerSecondSquared hard_limit,
  const RadiansPerSecondSquared numerical_error) noexcept
{
  return classifyMagnitude(value, hard_limit, numerical_error);
}

[[nodiscard]] inline BoundaryClass classifyJerk(
  const RadiansPerSecondCubed value, const RadiansPerSecondCubed hard_limit,
  const RadiansPerSecondCubed numerical_error) noexcept
{
  return classifyMagnitude(value, hard_limit, numerical_error);
}

struct PositionBoundSet
{
  PositionInterval factory;
  PositionInterval hardware;
  PositionInterval working;
  PositionInterval wire;
};

/// Builds immutable per-joint bounds once during setup.
/// `execution_error` is an explicitly modelled command/tracking budget. The wire
/// interval first reserves that budget inside working limits, then rounds both
/// endpoints inward to values representable by the driver's float interface.
[[nodiscard]] inline std::optional<PositionBoundSet> makePositionBoundSet(
  const PositionInterval & factory, const PositionInterval & hardware,
  const PositionInterval & working, const Radians execution_error) noexcept
{
  if (!factory.valid() || !hardware.valid() || !working.valid() ||
    !std::isfinite(execution_error.value) || execution_error.value < 0.0 ||
    hardware.lower_rad < factory.lower_rad || hardware.upper_rad > factory.upper_rad ||
    working.lower_rad < hardware.lower_rad || working.upper_rad > hardware.upper_rad)
  {
    return std::nullopt;
  }

  const double reserved_lower = working.lower_rad + execution_error.value;
  const double reserved_upper = working.upper_rad - execution_error.value;
  if (!std::isfinite(reserved_lower) || !std::isfinite(reserved_upper) ||
    reserved_lower >= reserved_upper)
  {
    return std::nullopt;
  }

  float wire_lower = static_cast<float>(reserved_lower);
  if (!std::isfinite(wire_lower)) {
    return std::nullopt;
  }
  if (static_cast<double>(wire_lower) < reserved_lower) {
    wire_lower = std::nextafter(wire_lower, INFINITY);
  }
  float wire_upper = static_cast<float>(reserved_upper);
  if (!std::isfinite(wire_upper)) {
    return std::nullopt;
  }
  if (static_cast<double>(wire_upper) > reserved_upper) {
    wire_upper = std::nextafter(wire_upper, -INFINITY);
  }
  const PositionInterval wire{
    static_cast<double>(wire_lower), static_cast<double>(wire_upper)};
  if (!wire.valid() || wire.lower_rad < reserved_lower || wire.upper_rad > reserved_upper) {
    return std::nullopt;
  }
  return PositionBoundSet{factory, hardware, working, wire};
}

/// Checks the final float that would be sent; it deliberately does not clamp it.
[[nodiscard]] inline bool isWireCommand(
  const double command_rad,
  const PositionInterval & wire) noexcept
{
  if (!isNominalPosition(command_rad, wire)) {
    return false;
  }
  const float transmitted = static_cast<float>(command_rad);
  return std::isfinite(transmitted) &&
         static_cast<double>(transmitted) >= wire.lower_rad &&
         static_cast<double>(transmitted) <= wire.upper_rad;
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__JOINT_BOUNDARY_POLICY_HPP_
