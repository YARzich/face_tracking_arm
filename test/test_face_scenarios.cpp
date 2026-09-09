// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "test_face_scenarios.hpp"

namespace face_tracking_arm::test_targets
{
namespace
{
constexpr double kTolerance = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

TEST(FaceScenarios, ParsesEveryLaunchChoiceAndRejectsUnknownNames)
{
  EXPECT_EQ(parse_scenario("disabled"), Scenario::kDisabled);
  EXPECT_EQ(parse_scenario("stationary"), Scenario::kStationary);
  EXPECT_EQ(parse_scenario("circle"), Scenario::kCircle);
  EXPECT_EQ(parse_scenario("people"), Scenario::kPeople);
  EXPECT_EQ(parse_scenario("walk_around"), Scenario::kWalkAround);
  EXPECT_THROW(parse_scenario(""), std::invalid_argument);
  EXPECT_THROW(parse_scenario("unknown"), std::invalid_argument);
  EXPECT_FALSE(sample(Scenario::kDisabled, 100.0).target.has_value());
}

TEST(FaceScenarios, RejectsInvalidElapsedTime)
{
  for (double elapsed : {-0.01, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()})
  {
    EXPECT_THROW(sample(Scenario::kPeople, elapsed), std::invalid_argument);
  }
}

TEST(FaceScenarios, RetainsOriginalPeopleAndTargetLossWindows)
{
  for (double elapsed : {0.0, 2.399, 7.4, 9.999, 20.2, 23.999, 120.0, 124.0, 131.999}) {
    EXPECT_FALSE(sample(Scenario::kPeople, elapsed).target.has_value()) << elapsed;
  }
  const auto a = sample(Scenario::kPeople, 2.4).target.value();
  EXPECT_NEAR(a.x, 0.76, kTolerance);
  EXPECT_NEAR(a.y, 0.41, kTolerance);
  EXPECT_NEAR(a.z, 1.48, kTolerance);
  const auto b = sample(Scenario::kPeople, 10.0).target.value();
  EXPECT_NEAR(b.x, 0.78, kTolerance);
  EXPECT_NEAR(b.y, -0.43, kTolerance);
  EXPECT_NEAR(b.z, 1.88, kTolerance);
  EXPECT_TRUE(sample(Scenario::kPeople, 13.799).target.has_value());
  const auto c = sample(Scenario::kPeople, 13.8).target.value();
  EXPECT_NEAR(c.x, 0.73, kTolerance);
  EXPECT_NEAR(c.y, 0.31, kTolerance);
  EXPECT_NEAR(c.z, 1.68, kTolerance);
  const auto before_shift = sample(Scenario::kPeople, 16.8).target.value();
  const auto after_shift = sample(Scenario::kPeople, 17.2).target.value();
  EXPECT_LT(after_shift.y - before_shift.y, -0.08);
}

TEST(FaceScenarios, CoversEightSidesAtDifferentAdultFaceHeights)
{
  constexpr std::array<double, 8> heights{1.60, 1.75, 1.90, 1.65, 1.80, 2.00, 1.70, 1.85};
  for (std::size_t side = 0; side < heights.size(); ++side) {
    const double start = 24.0 + 6.0 * side;
    const auto first = sample(Scenario::kPeople, start);
    const auto last = sample(Scenario::kPeople, start + 5.999);
    ASSERT_TRUE(first.target.has_value());
    ASSERT_TRUE(last.target.has_value());
    const auto point = *first.target;
    const double angle = static_cast<double>(side) * kPi / 4.0;
    EXPECT_NEAR(point.x, 0.85 * std::cos(angle), kTolerance);
    EXPECT_NEAR(point.y, 0.85 * std::sin(angle), kTolerance);
    EXPECT_NEAR(point.z, heights[side] - 0.12, kTolerance);
    EXPECT_GT(std::hypot(point.x, point.y), 0.50 + 0.25);
    EXPECT_DOUBLE_EQ(point.x, last.target->x);
    EXPECT_DOUBLE_EQ(point.y, last.target->y);
    EXPECT_DOUBLE_EQ(point.z, last.target->z);
    EXPECT_EQ(first.phase, last.phase);
    EXPECT_NE(first.phase, sample(Scenario::kPeople, start + 6.0).phase);
  }
}

TEST(FaceScenarios, WalkMakesFullTurnOutsideTableAtConstantHumanHeight)
{
  for (int index = 0; index <= 2400; ++index) {
    const double elapsed = 0.02 * index;
    const auto walk = sample(Scenario::kWalkAround, elapsed).target.value();
    EXPECT_NEAR(std::hypot(walk.x, walk.y), 0.85, kTolerance);
    EXPECT_GE(walk.z, 1.67 - kTolerance);
    EXPECT_LE(walk.z, 1.69 + kTolerance);
    if (elapsed < 48.0) {
      const auto embedded = sample(Scenario::kPeople, 72.0 + elapsed).target.value();
      EXPECT_NEAR(walk.x, embedded.x, kTolerance);
      EXPECT_NEAR(walk.y, embedded.y, kTolerance);
      EXPECT_NEAR(walk.z, embedded.z, kTolerance);
    }
  }
  EXPECT_NEAR(sample(Scenario::kWalkAround, 0.0).target->x, 0.85, kTolerance);
  EXPECT_NEAR(sample(Scenario::kWalkAround, 12.0).target->y, 0.85, kTolerance);
  EXPECT_NEAR(sample(Scenario::kWalkAround, 24.0).target->x, -0.85, kTolerance);
  EXPECT_NEAR(sample(Scenario::kWalkAround, 36.0).target->y, -0.85, kTolerance);
}

TEST(FaceScenarios, WalkHasNoPositionOrVelocityJumpAtLoopBoundary)
{
  constexpr double dt = 0.001;
  const auto before = sample(Scenario::kWalkAround, 48.0 - dt).target.value();
  const auto at = sample(Scenario::kWalkAround, 48.0).target.value();
  const auto after = sample(Scenario::kWalkAround, 48.0 + dt).target.value();
  const auto start = sample(Scenario::kWalkAround, 0.0).target.value();
  EXPECT_NEAR(at.x, start.x, kTolerance);
  EXPECT_NEAR(at.y, start.y, kTolerance);
  EXPECT_NEAR(at.z, start.z, kTolerance);
  EXPECT_NEAR((at.x - before.x) / dt, (after.x - at.x) / dt, 0.0001);
  EXPECT_NEAR((at.y - before.y) / dt, (after.y - at.y) / dt, 0.0001);
  EXPECT_NEAR((at.z - before.z) / dt, (after.z - at.z) / dt, 0.0001);
}

TEST(FaceScenarios, PeopleCycleRepeatsTargetsAndLossesExactly)
{
  for (double elapsed = 0.0; elapsed < kPeopleCycleDurationSec; elapsed += 0.125) {
    const auto first = sample(Scenario::kPeople, elapsed);
    const auto repeated = sample(Scenario::kPeople, elapsed + kPeopleCycleDurationSec);
    EXPECT_EQ(first.phase, repeated.phase);
    ASSERT_EQ(first.target.has_value(), repeated.target.has_value());
    if (first.target) {
      EXPECT_NEAR(first.target->x, repeated.target->x, kTolerance);
      EXPECT_NEAR(first.target->y, repeated.target->y, kTolerance);
      EXPECT_NEAR(first.target->z, repeated.target->z, kTolerance);
    }
  }
}

TEST(FaceScenarios, AllPublishedTargetsUseFaceHeightAboveFloor)
{
  for (Scenario scenario : {Scenario::kStationary, Scenario::kCircle, Scenario::kPeople,
      Scenario::kWalkAround})
  {
    for (double t = 0.0; t < kPeopleCycleDurationSec; t += 0.1) {
      const auto target = sample(scenario, t).target;
      if (target) {
        EXPECT_TRUE(std::isfinite(target->x));
        EXPECT_TRUE(std::isfinite(target->y));
        EXPECT_GE(target->z, 1.45);
        EXPECT_LE(target->z, 1.91);
      }
    }
  }
  EXPECT_NEAR(sample(Scenario::kStationary, 0.0).target->z, 1.68, kTolerance);
}

}  // namespace
}  // namespace face_tracking_arm::test_targets
