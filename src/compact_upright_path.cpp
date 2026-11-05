// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_upright_path.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace face_tracking_arm::control
{
namespace
{
bool finitePositive(const double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool insidePositionBounds(
  const Eigen::VectorXd & positions,
  const std::vector<std::string> & variable_names,
  const moveit::core::RobotModelConstPtr & model)
{
  if (positions.size() != static_cast<Eigen::Index>(variable_names.size()) ||
    !positions.allFinite())
  {
    return false;
  }
  for (Eigen::Index index = 0; index < positions.size(); ++index) {
    const auto & bounds = model->getVariableBounds(
      variable_names[static_cast<std::size_t>(index)]);
    if (bounds.position_bounded_ &&
      (positions[index] < bounds.min_position_ || positions[index] > bounds.max_position_))
    {
      return false;
    }
  }
  return true;
}

bool boundedIntervals(
  const double distance, const double maximum_step, const std::size_t maximum_samples,
  std::size_t & intervals)
{
  const double required = std::ceil(distance / maximum_step);
  if (!std::isfinite(required) || required < 0.0 ||
    required > static_cast<double>(maximum_samples) ||
    required >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
  {
    return false;
  }
  intervals = std::max<std::size_t>(1U, static_cast<std::size_t>(required));
  return intervals <= maximum_samples;
}
}  // namespace

std::optional<CompactStagedPath> makeCompactStagedPath(
  const moveit::core::RobotState & start,
  const Eigen::VectorXd & goal_positions,
  const moveit::core::JointModelGroup * group,
  const CompactStagedPathConfig & config,
  const std::function<bool()> & should_cancel)
{
  if (!group || !start.getRobotModel() || config.maximum_samples < 3U ||
    !finitePositive(config.maximum_joint_step_rad) ||
    !std::isfinite(config.maximum_early_yaw_shift_rad) ||
    config.maximum_early_yaw_shift_rad < 0.0)
  {
    return std::nullopt;
  }

  const auto & variable_names = group->getVariableNames();
  const auto variable_count = static_cast<std::size_t>(variable_names.size());
  if (variable_count != 6U ||
    goal_positions.size() != static_cast<Eigen::Index>(variable_count) ||
    !goal_positions.allFinite() || !start.satisfiesBounds(group) ||
    config.yaw_joint_index >= variable_count || config.reset_joint_index >= variable_count ||
    config.yaw_joint_index == config.reset_joint_index)
  {
    return std::nullopt;
  }

  const auto model = start.getRobotModel();
  Eigen::VectorXd start_positions;
  start.copyJointGroupPositions(group, start_positions);
  if (!insidePositionBounds(start_positions, variable_names, model) ||
    !insidePositionBounds(goal_positions, variable_names, model))
  {
    return std::nullopt;
  }

  moveit::core::RobotState goal_state(start);
  goal_state.setJointGroupPositions(group, goal_positions);
  goal_state.update();
  if (!goal_state.satisfiesBounds(group)) {
    return std::nullopt;
  }
  if (should_cancel && should_cancel()) {
    return std::nullopt;
  }

  const Eigen::Index yaw_index = static_cast<Eigen::Index>(config.yaw_joint_index);
  const Eigen::Index reset_index = static_cast<Eigen::Index>(config.reset_joint_index);
  Eigen::VectorXd prefix = goal_positions;
  prefix[yaw_index] = start_positions[yaw_index] + std::clamp(
    goal_positions[yaw_index] - start_positions[yaw_index],
    -config.maximum_early_yaw_shift_rad, config.maximum_early_yaw_shift_rad);
  prefix[reset_index] = start_positions[reset_index];

  std::size_t fold_intervals = 0U;
  std::size_t reset_intervals = 0U;
  if (!boundedIntervals(
      (prefix - start_positions).lpNorm<Eigen::Infinity>(), config.maximum_joint_step_rad,
      config.maximum_samples - 1U, fold_intervals) ||
    !boundedIntervals(
      (goal_positions - prefix).lpNorm<Eigen::Infinity>(), config.maximum_joint_step_rad,
      config.maximum_samples - 1U - fold_intervals, reset_intervals) ||
    fold_intervals + reset_intervals + 1U > config.maximum_samples)
  {
    return std::nullopt;
  }

  CompactStagedPath result;
  result.waypoints.reserve(1U + fold_intervals + reset_intervals);
  result.waypoints.push_back(start_positions);
  const auto append_leg = [&](const Eigen::VectorXd & from, const Eigen::VectorXd & to,
    const std::size_t intervals) {
      for (std::size_t step = 1U; step <= intervals; ++step) {
        if (should_cancel && should_cancel()) {
          return false;
        }
        const double fraction = static_cast<double>(step) / static_cast<double>(intervals);
        const Eigen::VectorXd positions = from + fraction * (to - from);
        if (!insidePositionBounds(positions, variable_names, model)) {
          return false;
        }
        result.waypoints.push_back(positions);
      }
      return true;
    };

  if (!append_leg(start_positions, prefix, fold_intervals)) {
    return std::nullopt;
  }
  result.fold_end_index = result.waypoints.size() - 1U;
  if (!append_leg(prefix, goal_positions, reset_intervals) ||
    result.waypoints.empty() || !result.waypoints.back().isApprox(goal_positions, 1.0e-12))
  {
    return std::nullopt;
  }
  result.waypoints.back() = goal_positions;
  return result;
}
}  // namespace face_tracking_arm::control
