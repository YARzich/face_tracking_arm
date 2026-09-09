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
  search.prepared();
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

}  // namespace face_tracking_arm::control
