// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>

#include "face_tracking_arm/following_error_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr FollowingErrorPolicy kPolicy{
  0.020,
  0.250,
};

TEST(FollowingErrorPolicy, ContinuesInsideNormalTrackingTolerance)
{
  EXPECT_EQ(
    classifyFollowingError(0.019, 0.249, kPolicy),
    FollowingErrorAction::kContinue);
}

TEST(FollowingErrorPolicy, RecoversBoundedVelocityTransientInsideCollisionTube)
{
  EXPECT_EQ(
    classifyFollowingError(0.003468, 0.292092, kPolicy),
    FollowingErrorAction::kControlledRearm);
}

TEST(FollowingErrorPolicy, RecoversFinitePositionLagAfterIndependentSafetyValidation)
{
  EXPECT_EQ(
    classifyFollowingError(0.020001, 0.100, kPolicy),
    FollowingErrorAction::kControlledRearm);
  EXPECT_EQ(
    classifyFollowingError(0.006001, 0.251, kPolicy),
    FollowingErrorAction::kControlledRearm);
}

TEST(FollowingErrorPolicy, RecoversFiniteVelocityLagButRejectsInvalidInput)
{
  EXPECT_EQ(
    classifyFollowingError(0.003, 0.300001, kPolicy),
    FollowingErrorAction::kControlledRearm);
  EXPECT_EQ(
    classifyFollowingError(
      std::numeric_limits<double>::quiet_NaN(), 0.1, kPolicy),
    FollowingErrorAction::kLatchedHalt);
}


TEST(FollowingErrorPolicy, RepeatedRecordedLagHasNoAttemptBudget)
{
  for (int attempt = 0; attempt < 1000; ++attempt) {
    EXPECT_EQ(
      classifyFollowingError(0.020285, 0.057148, kPolicy),
      FollowingErrorAction::kControlledRearm);
    EXPECT_EQ(classifyFollowingError(0.0, 0.0, kPolicy), FollowingErrorAction::kContinue);
  }
}

}  // namespace
}  // namespace face_tracking_arm::control
