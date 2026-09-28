// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "face_tracking_arm/posture_recovery_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{
bool needs(double position, double velocity, double face_speed = 0.11)
{
  return needsPostureRecovery(position, velocity, -2.0 * M_PI, 2.0 * M_PI, 0.1, face_speed);
}

TEST(PostureRecoveryPolicy, OldThresholdAndStationaryPersonDoNotTrigger)
{
  for (const double sign : {-1.0, 1.0}) {
    EXPECT_FALSE(needs(sign * 3.9, sign * 0.13));
    EXPECT_FALSE(needs(sign * 5.3, 0.0, 0.0));
    EXPECT_FALSE(needs(sign * 6.1, sign * 0.05, 0.0));
  }
}

TEST(PostureRecoveryPolicy, TriggersOnlyForApproachOrCloseBoundaryStall)
{
  for (const double sign : {-1.0, 1.0}) {
    EXPECT_TRUE(needs(sign * 5.0, sign * 0.13));
    EXPECT_FALSE(needs(sign * 5.0, -sign * 0.13));
    EXPECT_FALSE(needs(sign * 5.0, 0.0));
    EXPECT_TRUE(needs(sign * 6.0, 0.0));
    EXPECT_TRUE(needs(sign * 6.0, 0.0, 0.001));
    EXPECT_FALSE(needs(sign * 6.0, -sign * 0.001));
    EXPECT_FALSE(needs(sign * 6.0, -sign * 0.13));
  }
}

TEST(PostureRecoveryPolicy, FasterApproachStartsEarlierAndDoesNotWrapJoints)
{
  EXPECT_FALSE(needs(4.8, 0.13));
  EXPECT_TRUE(needs(4.8, 0.5));
  EXPECT_FALSE(needs(0.1, 0.13));
  EXPECT_TRUE(needs(6.1, 0.13));
}

TEST(PostureRecoveryPolicy, IgnoresNarrowRangesAndInvalidInput)
{
  EXPECT_FALSE(needsPostureRecovery(2.4, 0.2, -2.6, 2.6, 0.1, 0.1));
  EXPECT_FALSE(needs(std::numeric_limits<double>::quiet_NaN(), 0.13));
  EXPECT_FALSE(needs(5.0, std::numeric_limits<double>::infinity()));
}
TEST(PointingProgressMonitor, StaticMisalignmentRetriesWithoutAFiniteAttemptBudget)
{
  PointingProgressMonitor monitor;
  EXPECT_FALSE(monitor.update(0.4, 0.0));
  EXPECT_FALSE(monitor.update(0.4, 1.4));
  EXPECT_TRUE(monitor.update(0.4, 1.5));
  EXPECT_TRUE(monitor.update(0.4, 300.0));
  EXPECT_EQ(recoveryRelaxation(0), 0U);
  EXPECT_EQ(recoveryRelaxation(2), 1U);
  EXPECT_EQ(recoveryRelaxation(4), 2U);
  EXPECT_EQ(recoveryRelaxation(100000), 2U);
}

TEST(PointingProgressMonitor, ProgressAlignmentAndClockResetRestartTheWindow)
{
  PointingProgressMonitor monitor;
  EXPECT_FALSE(monitor.update(0.4, 0.0));
  EXPECT_FALSE(monitor.update(0.35, 1.0));
  EXPECT_FALSE(monitor.update(0.35, 2.0));
  EXPECT_TRUE(monitor.update(0.35, 2.5));
  EXPECT_FALSE(monitor.update(0.01, 2.6));
  EXPECT_FALSE(monitor.update(0.35, 3.0));
  EXPECT_FALSE(monitor.update(0.35, 0.0));
  EXPECT_FALSE(monitor.update(std::numeric_limits<double>::quiet_NaN(), 5.0));
  EXPECT_FALSE(monitor.update(0.35, 6.0));
}

}  // namespace
}  // namespace face_tracking_arm::control
