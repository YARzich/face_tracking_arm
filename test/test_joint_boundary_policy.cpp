// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

TEST(JointBoundaryPolicy, KeepsNumericalBudgetsTypedAndIndependent)
{
  const NumericalPolicy policy{
    Radians{1.0e-12}, RadiansPerSecond{2.0e-12},
    RadiansPerSecondSquared{3.0e-12}, RadiansPerSecondCubed{4.0e-12},
    SolverResidual{1.0e-2}};
  ASSERT_TRUE(policy.valid());

  const PositionInterval interval{-1.0, 1.0};
  EXPECT_EQ(classifyPosition(1.0 + 2.0e-12, interval, policy.position_error),
    BoundaryClass::kOutside);
  EXPECT_EQ(classifyPosition(1.0 + 0.5e-12, interval, policy.position_error),
    BoundaryClass::kBoundary);
}

TEST(JointBoundaryPolicy, DerivesRoundoffFromOperationCountsApartFromSolverResidual)
{
  const auto strict_residual = NumericalPolicy::fromOperationBudgets(
    2.0, 18U, 0.5, 12U, 1.0, 12U, 20.0, 4U, SolverResidual{1.0e-8});
  const auto loose_residual = NumericalPolicy::fromOperationBudgets(
    2.0, 18U, 0.5, 12U, 1.0, 12U, 20.0, 4U, SolverResidual{1.0e-2});
  ASSERT_TRUE(strict_residual.has_value());
  ASSERT_TRUE(loose_residual.has_value());
  EXPECT_EQ(strict_residual->position_error.value, loose_residual->position_error.value);
  EXPECT_EQ(strict_residual->velocity_error.value, loose_residual->velocity_error.value);
  EXPECT_EQ(strict_residual->acceleration_error.value,
    loose_residual->acceleration_error.value);
  EXPECT_EQ(strict_residual->jerk_error.value, loose_residual->jerk_error.value);
  EXPECT_NE(strict_residual->solver_residual.value, loose_residual->solver_residual.value);
  EXPECT_GT(strict_residual->jerk_error.value, strict_residual->position_error.value);
  EXPECT_FALSE(machineRoundoffBound(1.0, 0U).has_value());
  const auto cancellation_bound = machineDifferenceQuotientBound(1.0, 1.0, 0.01, 2U);
  ASSERT_TRUE(cancellation_bound.has_value());
  const double expected_cancellation_bound =
    (2.0 * std::numeric_limits<double>::epsilon()) /
    (1.0 - 2.0 * std::numeric_limits<double>::epsilon()) * 200.0;
  EXPECT_DOUBLE_EQ(*cancellation_bound, expected_cancellation_bound);
  EXPECT_GT(*cancellation_bound, 1.0e-14);  // Inputs cancel to zero, but their scale remains.
  EXPECT_FALSE(machineDifferenceQuotientBound(1.0, 1.0, 0.0, 2U).has_value());
}

TEST(JointBoundaryPolicy, DiscreteFactoryBoundsEachStoredPositionUpdate)
{
  constexpr double velocity_step = 1.0 * 0.01;
  const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-1.0, 1.0}, 1.0, 2.0, 4.0, 0.01, SolverResidual{1.0e-6});
  ASSERT_TRUE(policy.has_value());

  const double operand_sum = 1.0 + velocity_step;
  const double gamma = (2.0 * std::numeric_limits<double>::epsilon()) /
    (1.0 - 2.0 * std::numeric_limits<double>::epsilon());
  EXPECT_DOUBLE_EQ(policy->position_error.value, gamma * operand_sum);
  EXPECT_LT(policy->position_error.value, 1.0e-14);
  EXPECT_EQ(policy->solver_residual.value, 1.0e-6);

  double accumulated = 0.3;
  for (std::size_t tick = 0; tick < 10000U; ++tick) {
    const double stored_q = accumulated;
    const double increment = tick % 2U == 0U ? velocity_step : -velocity_step;
    const long double exact_local = static_cast<long double>(stored_q) +
      static_cast<long double>(increment);
    accumulated += increment;
    EXPECT_LE(std::abs(static_cast<long double>(accumulated) - exact_local),
      policy->position_error.value);
  }
}

TEST(JointBoundaryPolicy, JerkBudgetIncludesLocalAccelerationRounding)
{
  constexpr double dt = 0.01;
  const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-10.0, 10.0}, 1.0, 2.0, 20.0, dt, SolverResidual{1.0e-6});
  ASSERT_TRUE(policy.has_value());

  // Counterexample from the terminal braking transition: both stored
  // accelerations are locally rounded, so their finite difference can exceed
  // J by a few picoradians/s^3 even though each acceleration passes its check.
  constexpr double previous_acceleration = 1.0489656018797255e-14;
  constexpr double next_acceleration = -0.20000000000002108;
  const double jerk = (next_acceleration - previous_acceleration) / dt;
  ASSERT_GT(std::abs(jerk), 20.0);
  EXPECT_EQ(classifyJerk(
      RadiansPerSecondCubed{jerk}, RadiansPerSecondCubed{20.0},
      policy->jerk_error), BoundaryClass::kBoundary);
  EXPECT_GT(policy->jerk_error.value, std::abs(jerk) - 20.0);
}

TEST(JointBoundaryPolicy, UsesOneBoundaryClassificationForEitherEdge)
{
  const PositionInterval interval{-1.0, 1.0};
  const Radians error{0.01};
  EXPECT_EQ(classifyPosition(-1.0, interval, error), BoundaryClass::kBoundary);
  EXPECT_EQ(classifyPosition(1.0, interval, error), BoundaryClass::kBoundary);
  EXPECT_EQ(classifyPosition(-1.005, interval, error), BoundaryClass::kBoundary);
  EXPECT_EQ(classifyPosition(1.02, interval, error), BoundaryClass::kOutside);
  EXPECT_EQ(classifyPosition(0.5, interval, error), BoundaryClass::kInterior);
}

TEST(JointBoundaryPolicy, FloatFeedbackRepresentationIsUncertaintyNotPermission)
{
  const float below_one =
    std::nextafter(1.0F, -std::numeric_limits<float>::infinity());
  const double float_midpoint =
    (static_cast<double>(below_one) + static_cast<double>(1.0F)) / 2.0;
  const PositionInterval interval{-1.0, float_midpoint};
  EXPECT_EQ(classifyMeasuredPosition(static_cast<double>(below_one), interval),
    BoundaryClass::kBoundary);
  EXPECT_EQ(classifyMeasuredPosition(1.0, interval),
    BoundaryClass::kBoundary);
  EXPECT_FALSE(isNominalPosition(1.0, interval));
  EXPECT_EQ(classifyMeasuredPosition(std::numeric_limits<double>::quiet_NaN(), interval),
    BoundaryClass::kInvalid);
}

TEST(JointBoundaryPolicy, ClassifiesVelocityAccelerationAndJerkWithTheirOwnUnits)
{
  EXPECT_EQ(classifyVelocity(
      RadiansPerSecond{1.0 + 1.0e-12}, RadiansPerSecond{1.0},
      RadiansPerSecond{2.0e-12}), BoundaryClass::kBoundary);
  EXPECT_EQ(classifyVelocity(
      RadiansPerSecond{-1.0 - 4.0e-12}, RadiansPerSecond{1.0},
      RadiansPerSecond{2.0e-12}), BoundaryClass::kOutside);
  EXPECT_EQ(classifyAcceleration(
      RadiansPerSecondSquared{2.0}, RadiansPerSecondSquared{2.0},
      RadiansPerSecondSquared{0.0}), BoundaryClass::kBoundary);
  EXPECT_EQ(classifyJerk(
      RadiansPerSecondCubed{19.0}, RadiansPerSecondCubed{20.0},
      RadiansPerSecondCubed{0.5}), BoundaryClass::kInterior);
  EXPECT_EQ(classifyJerk(
      RadiansPerSecondCubed{20.25}, RadiansPerSecondCubed{20.0},
      RadiansPerSecondCubed{0.5}), BoundaryClass::kBoundary);
}

TEST(JointBoundaryPolicy, ReservesExplicitBudgetThenRoundsWireEndpointsInward)
{
  const auto bounds = makePositionBoundSet(
    PositionInterval{-2.0, 2.0}, PositionInterval{-1.5, 1.5},
    PositionInterval{-1.0, 1.0}, Radians{0.1});
  ASSERT_TRUE(bounds.has_value());
  EXPECT_EQ(bounds->factory.lower_rad, -2.0);
  EXPECT_EQ(bounds->hardware.upper_rad, 1.5);
  EXPECT_GE(bounds->wire.lower_rad, -0.9);
  EXPECT_LE(bounds->wire.upper_rad, 0.9);
  EXPECT_TRUE(isWireCommand(bounds->wire.lower_rad, bounds->wire));
  EXPECT_TRUE(isWireCommand(bounds->wire.upper_rad, bounds->wire));
  EXPECT_FALSE(isWireCommand(bounds->wire.lower_rad - 1.0e-6, bounds->wire));
  EXPECT_FALSE(isWireCommand(bounds->wire.upper_rad + 1.0e-6, bounds->wire));
}

TEST(JointBoundaryPolicy, RequiresNominalContainmentBeforeFloatConversion)
{
  const auto bounds = makePositionBoundSet(
    PositionInterval{-2.0, 2.0}, PositionInterval{-1.5, 1.5},
    PositionInterval{-1.0, 1.0}, Radians{0.1});
  ASSERT_TRUE(bounds.has_value());
  const double just_above_nominal = std::nextafter(
    bounds->wire.upper_rad, std::numeric_limits<double>::infinity());
  ASSERT_EQ(static_cast<float>(just_above_nominal),
    static_cast<float>(bounds->wire.upper_rad));
  EXPECT_FALSE(isNominalPosition(just_above_nominal, bounds->wire));
  EXPECT_FALSE(isWireCommand(just_above_nominal, bounds->wire));
}

TEST(JointBoundaryPolicy, CoversBothEdgesForSixSyntheticJointsAndAdjacentValues)
{
  const std::array<PositionInterval, 6U> intervals{{
    {-1.0, 1.0}, {-1.1, 1.1}, {-1.2, 1.2}, {-1.3, 1.3}, {-1.4, 1.4}, {-1.5, 1.5}}};
  for (const auto & interval : intervals) {
    const double below_lower = std::nextafter(
      interval.lower_rad, -std::numeric_limits<double>::infinity());
    const double above_upper = std::nextafter(
      interval.upper_rad, std::numeric_limits<double>::infinity());
    const float lower_float = static_cast<float>(interval.lower_rad);
    const float upper_float = static_cast<float>(interval.upper_rad);
    const float below_lower_float = std::nextafter(
      lower_float, -std::numeric_limits<float>::infinity());
    const float above_upper_float = std::nextafter(
      upper_float, std::numeric_limits<float>::infinity());
    EXPECT_TRUE(isNominalPosition(interval.lower_rad, interval));
    EXPECT_TRUE(isNominalPosition(interval.upper_rad, interval));
    EXPECT_FALSE(isNominalPosition(below_lower, interval));
    EXPECT_FALSE(isNominalPosition(above_upper, interval));
    EXPECT_FALSE(isNominalPosition(static_cast<double>(below_lower_float), interval));
    EXPECT_FALSE(isNominalPosition(static_cast<double>(above_upper_float), interval));

    const auto bounds = makePositionBoundSet(interval, interval, interval, Radians{0.0});
    ASSERT_TRUE(bounds.has_value());
    EXPECT_TRUE(isWireCommand(bounds->wire.lower_rad, bounds->wire));
    EXPECT_TRUE(isWireCommand(bounds->wire.upper_rad, bounds->wire));
    EXPECT_FALSE(isWireCommand(std::nextafter(
        bounds->wire.lower_rad, -std::numeric_limits<double>::infinity()), bounds->wire));
    EXPECT_FALSE(isWireCommand(std::nextafter(
        bounds->wire.upper_rad, std::numeric_limits<double>::infinity()), bounds->wire));
    const float below_wire_lower = std::nextafter(
      static_cast<float>(bounds->wire.lower_rad), -std::numeric_limits<float>::infinity());
    const float above_wire_upper = std::nextafter(
      static_cast<float>(bounds->wire.upper_rad), std::numeric_limits<float>::infinity());
    EXPECT_FALSE(isWireCommand(static_cast<double>(below_wire_lower), bounds->wire));
    EXPECT_FALSE(isWireCommand(static_cast<double>(above_wire_upper), bounds->wire));
  }
}

TEST(JointBoundaryPolicy, RejectsInconsistentBoundsAndUnrepresentableIntervals)
{
  EXPECT_FALSE(makePositionBoundSet(
      PositionInterval{-1.0, 1.0}, PositionInterval{-1.1, 0.9},
      PositionInterval{-0.8, 0.8}, Radians{0.0}).has_value());
  EXPECT_FALSE(makePositionBoundSet(
      PositionInterval{-1.0, 1.0}, PositionInterval{-1.0, 1.0},
      PositionInterval{-0.1, 0.1}, Radians{0.1}).has_value());
  EXPECT_FALSE(makePositionBoundSet(
      PositionInterval{-1.0, 1.0}, PositionInterval{-1.0, 1.0},
      PositionInterval{-1.0, 1.0}, Radians{std::numeric_limits<double>::quiet_NaN()})
    .has_value());
}

TEST(JointBoundaryPolicy, RejectsInvalidNumericsAndNeverClampsCommands)
{
  NumericalPolicy invalid{
    Radians{-1.0}, RadiansPerSecond{0.0}, RadiansPerSecondSquared{0.0},
    RadiansPerSecondCubed{0.0}, SolverResidual{0.0}};
  EXPECT_FALSE(invalid.valid());
  EXPECT_EQ(classifyPosition(std::numeric_limits<double>::infinity(),
      PositionInterval{-1.0, 1.0}, Radians{0.0}), BoundaryClass::kInvalid);
  EXPECT_FALSE(isWireCommand(std::numeric_limits<double>::quiet_NaN(),
      PositionInterval{-1.0, 1.0}));
  EXPECT_EQ(classifyPosition(std::numeric_limits<double>::quiet_NaN(),
      PositionInterval{-1.0, 1.0}, Radians{0.0}), BoundaryClass::kInvalid);
  EXPECT_EQ(classifyPosition(0.0, PositionInterval{-1.0, 1.0},
      Radians{std::numeric_limits<double>::infinity()}), BoundaryClass::kInvalid);
  EXPECT_EQ(classifyVelocity(RadiansPerSecond{std::numeric_limits<double>::quiet_NaN()},
      RadiansPerSecond{1.0}, RadiansPerSecond{0.0}), BoundaryClass::kInvalid);
  EXPECT_EQ(classifyAcceleration(
      RadiansPerSecondSquared{std::numeric_limits<double>::infinity()},
      RadiansPerSecondSquared{1.0}, RadiansPerSecondSquared{0.0}),
    BoundaryClass::kInvalid);
  EXPECT_EQ(classifyJerk(RadiansPerSecondCubed{1.0},
      RadiansPerSecondCubed{std::numeric_limits<double>::infinity()},
      RadiansPerSecondCubed{0.0}), BoundaryClass::kInvalid);
}

}  // namespace
}  // namespace face_tracking_arm::control
