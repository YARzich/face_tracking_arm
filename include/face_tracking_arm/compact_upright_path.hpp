// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_UPRIGHT_PATH_HPP_
#define FACE_TRACKING_ARM__COMPACT_UPRIGHT_PATH_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>

namespace face_tracking_arm::control
{

/// Joint indices and bounded work limits for the two known xArm compact legs.
struct CompactStagedProjectionConfig
{
  std::size_t yaw_joint_index{0};
  std::size_t reset_joint_index{3};
  std::size_t wrist_pitch_joint_index{4};
  std::size_t wrist_roll_joint_index{5};
  double maximum_early_yaw_shift_rad{0.35};
  double maximum_abs_pitch_rad{0.0};
  double maximum_roll_error_rad{0.0};
  double maximum_joint_step_rad{0.01};
  std::size_t maximum_samples{8192};
};

/// Dense projected samples plus the last sample before any J1/J4 reset begins.
struct CompactStagedPath
{
  std::vector<Eigen::VectorXd> waypoints;
  std::size_t fold_end_index{0};
};

/// Generates a bounded, upright path from start to the exact committed goal.
/// Joint1 shifts toward the goal during the fold while joint4 stays at its start
/// value; the compact leg then resets both. Wrist joints are projected with the
/// analytic screen-pitch and gravity-roll Jacobian. Scene, collision and radius
/// checks remain the caller's responsibility for every returned segment.
[[nodiscard]] std::optional<CompactStagedPath> projectCompactStagedPath(
  const moveit::core::RobotState & start,
  const Eigen::VectorXd & goal_positions,
  const moveit::core::JointModelGroup * group,
  const std::string & screen_link_name,
  const CompactStagedProjectionConfig & config,
  const std::function<bool()> & should_cancel = {});

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_UPRIGHT_PATH_HPP_
