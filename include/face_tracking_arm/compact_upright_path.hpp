// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_UPRIGHT_PATH_HPP_
#define FACE_TRACKING_ARM__COMPACT_UPRIGHT_PATH_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>

namespace face_tracking_arm::control
{

/// Joint indices and bounded work limits for the two known xArm compact legs.
struct CompactStagedPathConfig
{
  std::size_t yaw_joint_index{0};
  std::size_t reset_joint_index{3};
  double maximum_early_yaw_shift_rad{0.35};
  double maximum_joint_step_rad{0.01};
  std::size_t maximum_samples{8192};
};

/// Dense bounded samples plus the last sample before any J1/J4 reset begins.
struct CompactStagedPath
{
  std::vector<Eigen::VectorXd> waypoints;
  std::size_t fold_end_index{0};
};

/// Generates a bounded joint path with no screen-orientation constraints. Joint1
/// shifts toward the full-turn goal during the fold while joint4 stays at its
/// start value; the compact leg then resets both. Scene, collision and radius
/// checks remain the caller's responsibility for every returned segment.
[[nodiscard]] std::optional<CompactStagedPath> makeCompactStagedPath(
  const moveit::core::RobotState & start,
  const Eigen::VectorXd & goal_positions,
  const moveit::core::JointModelGroup * group,
  const CompactStagedPathConfig & config,
  const std::function<bool()> & should_cancel = {});

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_UPRIGHT_PATH_HPP_
