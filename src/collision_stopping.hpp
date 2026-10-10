// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COLLISION_STOPPING_HPP_
#define COLLISION_STOPPING_HPP_

#include <Eigen/Core>

#include <string>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control::collision_detail
{

inline constexpr double kComparisonTolerance = 1.0e-8;

[[nodiscard]] inline bool isFinite(const Eigen::VectorXd & vector)
{
  return vector.array().isFinite().all();
}

[[nodiscard]] inline bool isFinite(const Eigen::Vector3d & vector)
{
  return vector.array().isFinite().all();
}

struct ReachableVelocityInterval
{
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

struct ReachableApproachInterval
{
  Eigen::VectorXd lower_contribution;
  Eigen::VectorXd upper_contribution;
  double minimum_mps{0.0};
  double maximum_mps{0.0};
};

[[nodiscard]] bool computeReachableVelocityInterval(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const double period_sec,
  ReachableVelocityInterval & interval,
  std::string & error);

[[nodiscard]] ReachableApproachInterval projectApproachInterval(
  const Eigen::VectorXd & distance_gradient,
  const ReachableVelocityInterval & velocity_interval);

[[nodiscard]] double requiredCollisionDistanceForApproachCap(
  const double cap,
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const ReachableVelocityInterval & velocity_interval,
  const ReachableApproachInterval & approach_interval,
  const double period_sec,
  const double residual_latency_sec);

[[nodiscard]] bool validateMotionInput(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index expected_size,
  std::string & error);

[[nodiscard]] bool validateSegmentMotionState(
  const JointMotionState & state,
  const Eigen::Index expected_size,
  std::string & error);

}  // namespace face_tracking_arm::control::collision_detail

#endif  // COLLISION_STOPPING_HPP_
