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
  0.006,
  0.300,
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

TEST(FollowingErrorPolicy, LatchesOutsideCollisionTrackingTube)
{
  EXPECT_EQ(
    classifyFollowingError(0.020001, 0.100, kPolicy),
    FollowingErrorAction::kLatchedHalt);
  EXPECT_EQ(
    classifyFollowingError(0.006001, 0.251, kPolicy),
    FollowingErrorAction::kLatchedHalt);
}

TEST(FollowingErrorPolicy, LatchesExcessiveVelocityAndInvalidInput)
{
  EXPECT_EQ(
    classifyFollowingError(0.003, 0.300001, kPolicy),
    FollowingErrorAction::kLatchedHalt);
  EXPECT_EQ(
    classifyFollowingError(
      std::numeric_limits<double>::quiet_NaN(), 0.1, kPolicy),
    FollowingErrorAction::kLatchedHalt);
}

}  // namespace
}  // namespace face_tracking_arm::control
