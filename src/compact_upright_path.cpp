// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_upright_path.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "face_tracking_arm/compact_upright.hpp"

namespace face_tracking_arm::control
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kPoseToleranceRad = 1.0e-8;
constexpr double kMaximumNewtonStepRad = 0.20;
constexpr double kMaximumJointStepSlack = 1.0e-9;
constexpr std::size_t kMaximumNewtonIterations = 20;
constexpr std::size_t kMaximumSubdivisionDepth = 24;

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

std::optional<CompactStagedPath> projectCompactStagedPath(
  const moveit::core::RobotState & start, const Eigen::VectorXd & goal_positions,
  const moveit::core::JointModelGroup * group, const std::string & screen_link_name,
  const CompactStagedProjectionConfig & config,
  const std::function<bool()> & should_cancel)
{
  if (!group || !start.getRobotModel() || screen_link_name.empty() ||
    config.maximum_samples < 2U || !finitePositive(config.maximum_joint_step_rad) ||
    !std::isfinite(config.maximum_early_yaw_shift_rad) ||
    config.maximum_early_yaw_shift_rad < 0.0 ||
    !std::isfinite(config.maximum_abs_pitch_rad) ||
    config.maximum_abs_pitch_rad < 0.0 || config.maximum_abs_pitch_rad > 0.5 * kPi ||
    !std::isfinite(config.maximum_roll_error_rad) ||
    config.maximum_roll_error_rad < 0.0 || config.maximum_roll_error_rad > kPi)
  {
    return std::nullopt;
  }
  const auto & variable_names = group->getVariableNames();
  const auto variable_count = static_cast<std::size_t>(variable_names.size());
  const std::vector<std::size_t> indices{
    config.yaw_joint_index, config.reset_joint_index,
    config.wrist_pitch_joint_index, config.wrist_roll_joint_index};
  if (variable_count != 6U || goal_positions.size() != static_cast<Eigen::Index>(variable_count) ||
    !goal_positions.allFinite() || !start.satisfiesBounds(group) ||
    std::any_of(indices.begin(), indices.end(), [variable_count](const std::size_t index) {
      return index >= variable_count;
    }))
  {
    return std::nullopt;
  }
  for (std::size_t lhs = 0; lhs < indices.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1U; rhs < indices.size(); ++rhs) {
      if (indices[lhs] == indices[rhs]) {
        return std::nullopt;
      }
    }
  }
  const auto model = start.getRobotModel();
  const auto * screen_link = model->getLinkModel(screen_link_name);
  if (!screen_link) {
    return std::nullopt;
  }
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
  const Eigen::Matrix3d start_rotation = start.getGlobalLinkTransform(screen_link).linear();
  const Eigen::Matrix3d goal_rotation = goal_state.getGlobalLinkTransform(screen_link).linear();
  const auto start_pose = evaluateCompactUprightPose(start_rotation);
  const auto goal_pose = evaluateCompactUprightPose(goal_rotation);
  if (!start_pose || !goal_pose ||
    std::abs(start_pose->signed_pitch_rad) >
    config.maximum_abs_pitch_rad + kCompactPitchValidationToleranceRad ||
    std::abs(goal_pose->signed_pitch_rad) >
    config.maximum_abs_pitch_rad + kCompactPitchValidationToleranceRad ||
    std::abs(start_pose->signed_roll_error_rad) > config.maximum_roll_error_rad ||
    std::abs(goal_pose->signed_roll_error_rad) > config.maximum_roll_error_rad)
  {
    return std::nullopt;
  }
  if (should_cancel && should_cancel()) {
    return std::nullopt;
  }

  const Eigen::Index yaw_index = static_cast<Eigen::Index>(config.yaw_joint_index);
  const Eigen::Index reset_index = static_cast<Eigen::Index>(config.reset_joint_index);
  const Eigen::Index wrist_pitch_index =
    static_cast<Eigen::Index>(config.wrist_pitch_joint_index);
  const Eigen::Index wrist_roll_index =
    static_cast<Eigen::Index>(config.wrist_roll_joint_index);
  Eigen::VectorXd prefix = goal_positions;
  prefix[yaw_index] = start_positions[yaw_index] + std::clamp(
    goal_positions[yaw_index] - start_positions[yaw_index],
    -config.maximum_early_yaw_shift_rad, config.maximum_early_yaw_shift_rad);
  prefix[reset_index] = start_positions[reset_index];
  prefix[wrist_pitch_index] = start_positions[wrist_pitch_index];
  prefix[wrist_roll_index] = start_positions[wrist_roll_index];

  const auto coreDistance = [wrist_pitch_index, wrist_roll_index](
    const Eigen::VectorXd & from, const Eigen::VectorXd & to) {
      double distance = 0.0;
      for (Eigen::Index index = 0; index < from.size(); ++index) {
        if (index != wrist_pitch_index && index != wrist_roll_index) {
          distance = std::max(distance, std::abs(to[index] - from[index]));
        }
      }
      return distance;
    };
  const double fold_distance = coreDistance(start_positions, prefix);
  const double reset_distance = coreDistance(prefix, goal_positions);
  const double total_core_distance = fold_distance + reset_distance;
  const double fold_progress = total_core_distance > 1.0e-12 ?
    fold_distance / total_core_distance : 0.5;
  const double roll_delta = std::remainder(
    goal_pose->signed_roll_error_rad - start_pose->signed_roll_error_rad, 2.0 * kPi);

  moveit::core::RobotState sample_state(start);
  std::size_t projection_attempts = 0U;
  const auto cancelled = [&should_cancel]() {
      return should_cancel && should_cancel();
    };
  const auto solveWrist = [&](const Eigen::VectorXd & seed, const Eigen::VectorXd & from,
    const Eigen::VectorXd & to, const double local_progress,
    const double global_progress) -> std::optional<Eigen::VectorXd> {
      if (cancelled() || projection_attempts >= config.maximum_samples) {
        return std::nullopt;
      }
      ++projection_attempts;
      Eigen::VectorXd q = seed;
      for (Eigen::Index joint = 0; joint < q.size(); ++joint) {
        if (joint != wrist_pitch_index && joint != wrist_roll_index) {
          q[joint] = from[joint] + local_progress * (to[joint] - from[joint]);
        }
      }
      const double target_pitch = start_pose->signed_pitch_rad +
        global_progress * (goal_pose->signed_pitch_rad - start_pose->signed_pitch_rad);
      const double target_roll = start_pose->signed_roll_error_rad + global_progress * roll_delta;
      for (std::size_t iteration = 0; iteration < kMaximumNewtonIterations; ++iteration) {
        if (cancelled()) {
          return std::nullopt;
        }
        sample_state.setJointGroupPositions(group, q);
        sample_state.update(true);
        const Eigen::Matrix3d rotation =
          sample_state.getGlobalLinkTransform(screen_link).linear();
        const auto pose = evaluateCompactUprightPose(rotation);
        if (!pose) {
          return std::nullopt;
        }
        Eigen::Vector2d error(
          target_pitch - pose->signed_pitch_rad,
          std::remainder(target_roll - pose->signed_roll_error_rad, 2.0 * kPi));
        if (error.cwiseAbs().maxCoeff() <= kPoseToleranceRad) {
          if (std::abs(pose->signed_pitch_rad) >
            config.maximum_abs_pitch_rad + kCompactPitchValidationToleranceRad ||
            std::abs(pose->signed_roll_error_rad) > config.maximum_roll_error_rad + 1.0e-9)
          {
            return std::nullopt;
          }
          return q;
        }
        Eigen::MatrixXd spatial_jacobian;
        if (!sample_state.getJacobian(
            group, screen_link, Eigen::Vector3d::Zero(), spatial_jacobian) ||
          spatial_jacobian.rows() != 6 ||
          spatial_jacobian.cols() != static_cast<Eigen::Index>(variable_count))
        {
          return std::nullopt;
        }
        const auto * group_root = group->getJointModels().front()->getParentLinkModel();
        if (!group_root) {
          return std::nullopt;
        }
        const Eigen::Matrix3d root_rotation =
          sample_state.getGlobalLinkTransform(group_root).linear();
        const Eigen::MatrixXd angular_jacobian =
          (root_rotation * spatial_jacobian.bottomRows<3>()).eval();
        const auto pitch_gradient = signedScreenPitchGradient(
          pose->screen_normal, angular_jacobian);
        const auto roll_gradient = signedScreenRollErrorGradient(rotation, angular_jacobian);
        if (!pitch_gradient || !roll_gradient) {
          return std::nullopt;
        }
        Eigen::Matrix2d wrist_jacobian;
        wrist_jacobian << (*pitch_gradient)[wrist_pitch_index],
          (*pitch_gradient)[wrist_roll_index], (*roll_gradient)[wrist_pitch_index],
          (*roll_gradient)[wrist_roll_index];
        const double determinant = wrist_jacobian.determinant();
        if (!wrist_jacobian.allFinite() || !std::isfinite(determinant) ||
          std::abs(determinant) < 1.0e-8)
        {
          return std::nullopt;
        }
        Eigen::Vector2d delta = wrist_jacobian.fullPivLu().solve(error);
        if (!delta.allFinite()) {
          return std::nullopt;
        }
        const double largest = delta.cwiseAbs().maxCoeff();
        if (largest > kMaximumNewtonStepRad) {
          delta *= kMaximumNewtonStepRad / largest;
        }
        bool accepted = false;
        double scale = 1.0;
        for (int line_search = 0; line_search < 8; ++line_search) {
          Eigen::VectorXd candidate = q;
          candidate[wrist_pitch_index] += scale * delta[0];
          candidate[wrist_roll_index] += scale * delta[1];
          if (!insidePositionBounds(candidate, variable_names, model)) {
            scale *= 0.5;
            continue;
          }
          Eigen::Vector2d candidate_error;
          sample_state.setJointGroupPositions(group, candidate);
          sample_state.update(true);
          const auto candidate_pose = evaluateCompactUprightPose(
            sample_state.getGlobalLinkTransform(screen_link).linear());
          if (candidate_pose) {
            candidate_error[0] = target_pitch - candidate_pose->signed_pitch_rad;
            candidate_error[1] = std::remainder(
              target_roll - candidate_pose->signed_roll_error_rad, 2.0 * kPi);
            if (candidate_error.allFinite() &&
              candidate_error.squaredNorm() < error.squaredNorm())
            {
              q = std::move(candidate);
              accepted = true;
              break;
            }
          }
          scale *= 0.5;
        }
        if (!accepted) {
          return std::nullopt;
        }
      }
      return std::nullopt;
    };

  std::vector<Eigen::VectorXd> path;
  path.reserve(std::min<std::size_t>(config.maximum_samples, 2048U));
  path.push_back(start_positions);
  const auto appendLeg = [&](const Eigen::VectorXd & from, const Eigen::VectorXd & to,
    const double progress_begin, const double progress_end,
    std::vector<Eigen::VectorXd> & waypoints) {
      std::function<bool(
          double, const Eigen::VectorXd &, double, std::size_t, Eigen::VectorXd &)> advance;
      advance = [&](const double begin, const Eigen::VectorXd & begin_q,
        const double end, const std::size_t depth, Eigen::VectorXd & end_q) -> bool {
          if (cancelled() || waypoints.size() >= config.maximum_samples ||
            depth > kMaximumSubdivisionDepth)
          {
            return false;
          }
          const double global_progress = progress_begin + end * (progress_end - progress_begin);
          const auto projected = solveWrist(
            begin_q, from, to, end, global_progress);
          if (!projected) {
            return false;
          }
          const double step = (*projected - begin_q).lpNorm<Eigen::Infinity>();
          if (step <= config.maximum_joint_step_rad + kMaximumJointStepSlack) {
            waypoints.push_back(*projected);
            end_q = *projected;
            return waypoints.size() <= config.maximum_samples;
          }
          const double midpoint = 0.5 * (begin + end);
          if (midpoint <= begin || midpoint >= end) {
            return false;
          }
          Eigen::VectorXd midpoint_q;
          if (!advance(begin, begin_q, midpoint, depth + 1U, midpoint_q)) {
            return false;
          }
          return advance(midpoint, midpoint_q, end, depth + 1U, end_q);
        };

      const double major_distance = coreDistance(from, to);
      std::size_t intervals = 0U;
      if (!boundedIntervals(
          major_distance, config.maximum_joint_step_rad, config.maximum_samples, intervals))
      {
        return false;
      }
      Eigen::VectorXd previous = waypoints.empty() ? from : waypoints.back();
      for (std::size_t interval = 1; interval <= intervals; ++interval) {
        const double begin = static_cast<double>(interval - 1U) / intervals;
        const double end = static_cast<double>(interval) / intervals;
        Eigen::VectorXd next;
        if (!advance(begin, previous, end, 0U, next)) {
          return false;
        }
        previous = std::move(next);
      }
      return true;
    };

  const bool fold_ok = appendLeg(start_positions, prefix, 0.0, fold_progress, path);
  if (!fold_ok || path.empty() || cancelled()) {
    return std::nullopt;
  }
  const std::size_t fold_end_index = path.size() - 1U;
  Eigen::VectorXd projected_prefix = path.back();
  if (!appendLeg(projected_prefix, goal_positions, fold_progress, 1.0, path) ||
    path.empty() || cancelled())
  {
    return std::nullopt;
  }
  const Eigen::VectorXd endpoint_delta = goal_positions - path.back();
  const double endpoint_distance = endpoint_delta.lpNorm<Eigen::Infinity>();
  std::size_t endpoint_steps = 0U;
  if (endpoint_distance > 1.0e-12 && !boundedIntervals(
      endpoint_distance, config.maximum_joint_step_rad,
      config.maximum_samples - std::min(config.maximum_samples, path.size()), endpoint_steps))
  {
    return std::nullopt;
  }
  const Eigen::VectorXd endpoint_start = path.back();
  for (std::size_t step = 1; step <= endpoint_steps; ++step) {
    if (cancelled()) {
      return std::nullopt;
    }
    const double fraction = static_cast<double>(step) / endpoint_steps;
    const Eigen::VectorXd candidate = endpoint_start + fraction * (goal_positions - endpoint_start);
    if (!insidePositionBounds(candidate, variable_names, model)) {
      return std::nullopt;
    }
    sample_state.setJointGroupPositions(group, candidate);
    sample_state.update(true);
    const auto pose = evaluateCompactUprightPose(
      sample_state.getGlobalLinkTransform(screen_link).linear());
    if (!pose || std::abs(pose->signed_pitch_rad) >
      config.maximum_abs_pitch_rad + kCompactPitchValidationToleranceRad ||
      std::abs(pose->signed_roll_error_rad) > config.maximum_roll_error_rad + 1.0e-9)
    {
      return std::nullopt;
    }
    path.push_back(candidate);
  }
  if (path.back().size() != goal_positions.size()) {
    return std::nullopt;
  }
  if (endpoint_steps == 0U) {
    sample_state.setJointGroupPositions(group, goal_positions);
    sample_state.update(true);
    const auto pose = evaluateCompactUprightPose(
      sample_state.getGlobalLinkTransform(screen_link).linear());
    if (!pose || std::abs(pose->signed_pitch_rad) >
      config.maximum_abs_pitch_rad + kCompactPitchValidationToleranceRad ||
      std::abs(pose->signed_roll_error_rad) > config.maximum_roll_error_rad + 1.0e-9)
    {
      return std::nullopt;
    }
    path.back() = goal_positions;
  } else if (!path.back().isApprox(goal_positions, 1.0e-12)) {
    return std::nullopt;
  } else {
    path.back() = goal_positions;
  }
  return CompactStagedPath{std::move(path), fold_end_index};
}

}  // namespace face_tracking_arm::control
