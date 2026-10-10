// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <stdexcept>

#include "face_tracking_arm/idle_search.hpp"

namespace face_tracking_arm::control
{

TEST(IdleSearch, BoundedSweepDoesNotUnwindAtEntryOrAccumulateTurns)
{
  IdleSearch search;
  Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  current[0] = 5.5;
  current[5] = 5.9;
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -6.18);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 6.18);
  search.start(current, Eigen::VectorXd::Zero(6), lower, upper, 0, {}, false, 0);
  EXPECT_DOUBLE_EQ(search.goal()[0], 5.5);
  EXPECT_EQ(search.phase(), "SEARCH_PLANNING");
  search.prepared(current, 0.0);
  for (int i = 0; i < 20; ++i) {
    const double endpoint = search.goal()[0];
    EXPECT_GE(endpoint, -6.18);
    EXPECT_LE(endpoint, 6.18);
    EXPECT_GE(endpoint, .58 - 1e-9);
    current = search.goal();
    search.update(current, i + 1);
    EXPECT_NE(search.goal()[0], endpoint);
  }
}

TEST(IdleSearch, LocalSearchTransitionsAtCurrentBaseAngle)
{
  IdleSearch search;
  Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  current[0] = .9;
  current[2] = .5;
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -6.18);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 6.18);
  search.start(current, Eigen::VectorXd::Zero(6), lower, upper, 0, {}, true, 10);
  EXPECT_EQ(search.phase(), "SEARCH_LOCAL");
  EXPECT_DOUBLE_EQ(search.goal()[2], .5);
  current[0] = 1.1;
  search.update(current, 16);
  EXPECT_TRUE(search.preparing());
  EXPECT_DOUBLE_EQ(search.goal()[0], 1.1);
  EXPECT_DOUBLE_EQ(search.goal()[2], 0.0);
}

TEST(IdleSearch, RejectsInvalidConfiguration)
{
  IdleSearch search;
  IdleSearchConfig config;
  config.speed_rad_s = -1;
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(6);
  EXPECT_THROW(search.start(q, q, q, q, 0, config, false, 0), std::invalid_argument);
}

TEST(IdleSearch, AcceptsMaximumConfiguredSearchSpeedAndRejectsAboveIt)
{
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -1.0);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 1.0);
  IdleSearchConfig config;
  config.speed_rad_s = 0.50;
  IdleSearch search;
  EXPECT_NO_THROW(search.start(q, q, lower, upper, 0, config, false, 0.0));

  config.speed_rad_s = 0.500001;
  EXPECT_THROW(search.start(q, q, lower, upper, 0, config, false, 0.0),
    std::invalid_argument);
}

TEST(IdleSearch, StalledEndpointRetriesTowardTheOppositeSide)
{
  IdleSearch search;
  Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -5.5);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 5.5);
  search.start(current, current, lower, upper, 0, {}, false, 0.0);
  search.prepared(current, 0.0);
  const double first_endpoint = search.goal()[0];

  current[0] = first_endpoint - 0.2;
  search.retryFrom(current, 1.0);

  EXPECT_LT(search.goal()[0], current[0]);
}

TEST(IdleSearch, StalledSweepReversesFromOffCenterPositionsInBothDirections)
{
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -5.5);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 5.5);
  for (const double initial_position : {-5.5, 5.5}) {
    IdleSearch search;
    Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
    current[0] = initial_position;
    search.start(current, current, lower, upper, 0, {}, false, 0.0);
    search.prepared(current, 0.0);

    if (initial_position > 0.0) {
      ASSERT_LT(search.goal()[0], initial_position);
      current[0] = 3.0;  // Blocked while already right of the bounded-range center.
      search.retryFrom(current, 1.0);
      EXPECT_GT(search.goal()[0], current[0]);
    } else {
      ASSERT_GT(search.goal()[0], initial_position);
      current[0] = -3.0;  // Blocked while already left of the bounded-range center.
      search.retryFrom(current, 1.0);
      EXPECT_LT(search.goal()[0], current[0]);
    }
  }
}

TEST(IdleSearch, DirectSweepRetriesAfterSustainedNoProgress)
{
  IdleSearch search;
  Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -5.5);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 5.5);
  search.start(current, current, lower, upper, 0, {}, false, 0.0);
  search.prepared(current, 0.0);
  const double initial_goal = search.goal()[0];

  search.update(current, 5.99);
  EXPECT_DOUBLE_EQ(search.goal()[0], initial_goal);
  search.update(current, 6.0);
  EXPECT_LT(search.goal()[0], current[0]);

  const double opposite_goal = search.goal()[0];
  search.update(current, 11.99);
  EXPECT_DOUBLE_EQ(search.goal()[0], opposite_goal);
  search.update(current, 12.0);
  EXPECT_DOUBLE_EQ(search.goal()[0], initial_goal);
}

TEST(IdleSearch, SlowContinuousSweepProgressDoesNotTriggerRetry)
{
  IdleSearch search;
  IdleSearchConfig config;
  config.speed_rad_s = 0.01;
  Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(6, -5.5);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(6, 5.5);
  search.start(current, current, lower, upper, 0, config, false, 0.0);
  search.prepared(current, 0.0);
  const double initial_goal = search.goal()[0];

  for (int second = 1; second <= 12; ++second) {
    current[0] = 0.0002 * second;
    search.update(current, static_cast<double>(second));
  }
  EXPECT_DOUBLE_EQ(search.goal()[0], initial_goal);
}

}  // namespace face_tracking_arm::control
