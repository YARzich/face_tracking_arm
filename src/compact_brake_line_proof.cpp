// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "compact_brake_line_proof.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace face_tracking_arm::control
{
namespace
{

bool same_context(
  const CompactBrakeLineProofContext & left,
  const CompactBrakeLineProofContext & right)
{
  return left.scene_revision == right.scene_revision &&
         left.config_revision == right.config_revision &&
         left.model_identity == right.model_identity &&
         left.group_identity == right.group_identity &&
         left.fixed_positions.size() == right.fixed_positions.size() &&
         left.fixed_positions.allFinite() && right.fixed_positions.allFinite() &&
         (left.fixed_positions.array() == right.fixed_positions.array()).all();
}

bool corridor_safe(
  const std::vector<JointMotionState> & tail, const std::size_t protected_index,
  const PositionInterval & protected_interval, const Radians position_error)
{
  for (const auto & point : tail) {
    if (point.position.size() <= static_cast<Eigen::Index>(protected_index) ||
      point.velocity.size() != point.position.size() ||
      point.acceleration.size() != point.position.size() ||
      !point.position.allFinite() || !point.velocity.allFinite() ||
      !point.acceleration.allFinite() ||
      classifyPosition(
        point.position[static_cast<Eigen::Index>(protected_index)],
        protected_interval, position_error) != BoundaryClass::kInterior)
    {
      return false;
    }
  }
  return true;
}

bool point_on_interval(
  const Eigen::VectorXd & position, const Eigen::VectorXd & origin,
  const Eigen::VectorXd & direction, const double begin, const double end,
  const double tube_error, double & projected)
{
  if (position.size() != origin.size() || !position.allFinite()) {
    return false;
  }
  projected = (position - origin).dot(direction);
  if (!std::isfinite(projected)) {
    return false;
  }
  const double clamped = std::clamp(projected, begin, end);
  const Eigen::VectorXd nearest = origin + clamped * direction;
  return (position - nearest).lpNorm<Eigen::Infinity>() <= tube_error;
}

bool point_on_forward_line(
  const Eigen::VectorXd & position, const Eigen::VectorXd & origin,
  const Eigen::VectorXd & direction, const double tube_error, double & projected)
{
  if (position.size() != origin.size() || !position.allFinite()) {
    return false;
  }
  projected = (position - origin).dot(direction);
  if (!std::isfinite(projected)) {
    return false;
  }
  const Eigen::VectorXd nearest = origin + std::max(0.0, projected) * direction;
  return (position - nearest).lpNorm<Eigen::Infinity>() <= tube_error;
}

}  // namespace

CompactBrakeLinePlan CompactBrakeLineProof::plan(
  const std::vector<JointMotionState> & tail,
  const CompactBrakeLineProofContext & context, const double tube_error_rad,
  const std::size_t protected_joint_index, const PositionInterval & protected_interval,
  const Radians protected_position_error)
{
  CompactBrakeLinePlan plan;
  if (tail.size() < 2U || !std::isfinite(tube_error_rad) || tube_error_rad < 0.0 ||
    !context.fixed_positions.allFinite() || context.model_identity == nullptr ||
    context.group_identity == nullptr ||
    !corridor_safe(tail, protected_joint_index, protected_interval, protected_position_error))
  {
    clear();
    return plan;
  }

  const bool can_reuse = valid_ && same_context(context_, context) &&
    tube_error_rad_ == tube_error_rad;
  if (can_reuse) {
    double maximum_projection = 0.0;
    for (const auto & point : tail) {
      double projected = 0.0;
      if (!point_on_forward_line(
          point.position, origin_, direction_, tube_error_rad, projected))
      {
        clear();
        return plan;
      }
      if (projected < -tube_error_rad) {
        clear();
        return plan;
      }
      maximum_projection = std::max(maximum_projection, projected);
    }
    plan.origin = origin_;
    plan.direction = direction_;
    plan.tube_error_rad = tube_error_rad_;
    plan.context = context_;
    if (maximum_projection <= interval_end_) {
      plan.result = CompactBrakeLinePlan::Result::kCovered;
      plan.interval_end = interval_end_;
      return plan;
    }
    plan.start_position = origin_ + interval_end_ * direction_;
    plan.end_position = origin_ + maximum_projection * direction_;
    plan.interval_end = maximum_projection;
    plan.result = CompactBrakeLinePlan::Result::kValidate;
    return plan;
  }

  clear();
  const Eigen::VectorXd displacement = tail.back().position - tail.front().position;
  const double length = displacement.norm();
  if (!std::isfinite(length) || length <= tube_error_rad) {
    return plan;
  }
  const Eigen::VectorXd direction = displacement / length;
  for (const auto & point : tail) {
    double projected = 0.0;
    if (!point_on_interval(
        point.position, tail.front().position, direction, 0.0, length,
        tube_error_rad, projected))
    {
      return plan;
    }
  }
  plan.origin = tail.front().position;
  plan.direction = direction;
  plan.start_position = plan.origin;
  plan.end_position = plan.origin + length * direction;
  plan.interval_end = length;
  plan.tube_error_rad = tube_error_rad;
  plan.context = context;
  plan.result = CompactBrakeLinePlan::Result::kValidate;
  return plan;
}

void CompactBrakeLineProof::commit(const CompactBrakeLinePlan & plan)
{
  if (plan.result != CompactBrakeLinePlan::Result::kValidate ||
    !plan.origin.allFinite() || !plan.direction.allFinite() ||
    !std::isfinite(plan.interval_end) || plan.interval_end <= 0.0)
  {
    clear();
    return;
  }
  valid_ = true;
  context_ = plan.context;
  origin_ = plan.origin;
  direction_ = plan.direction;
  interval_end_ = plan.interval_end;
  tube_error_rad_ = plan.tube_error_rad;
}

void CompactBrakeLineProof::clear() noexcept
{
  valid_ = false;
  context_ = {};
  origin_.resize(0);
  direction_.resize(0);
  interval_end_ = 0.0;
  tube_error_rad_ = 0.0;
}

}  // namespace face_tracking_arm::control
