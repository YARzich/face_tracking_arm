// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/joint_path_follower.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace face_tracking_arm::control
{
namespace
{

constexpr double kLengthEpsilon = 1.0e-9;

struct Projection
{
  std::size_t segment{0};
  double fraction{0.0};
  double distance{std::numeric_limits<double>::infinity()};
  Eigen::VectorXd position;
};

Projection projectWithinWindow(
  const std::vector<Eigen::VectorXd> & path, const Eigen::VectorXd & position,
  const std::size_t first_segment, const std::size_t maximum_segments)
{
  Projection best;
  if (path.size() == 1U) {
    best.position = path.front();
    best.distance = (position - best.position).norm();
    return best;
  }
  const std::size_t count = std::min(maximum_segments, path.size() - 1U - first_segment);
  for (std::size_t segment = first_segment; segment < first_segment + count; ++segment) {
    const Eigen::VectorXd delta = path[segment + 1U] - path[segment];
    const double fraction = std::clamp(
      (position - path[segment]).dot(delta) / delta.squaredNorm(), 0.0, 1.0);
    const Eigen::VectorXd projected = path[segment] + fraction * delta;
    const double distance = (position - projected).norm();
    // At a crossing or a shared endpoint, keep the earliest equally close segment.
    if (distance + kLengthEpsilon < best.distance) {
      best = Projection{segment, fraction, distance, projected};
    }
  }
  return best;
}

}  // namespace

JointPathFollower::JointPathFollower(JointPathFollowerConfig config)
: config_(std::move(config))
{
  const auto positive = [](const double value) {
      return std::isfinite(value) && value > 0.0;
    };
  if (!positive(config_.lookahead_distance_rad) ||
    !positive(config_.maximum_attach_distance_rad) ||
    !positive(config_.maximum_corridor_distance_rad) ||
    !positive(config_.arrival_tolerance_rad) ||
    !positive(config_.minimum_progress_rad) ||
    config_.minimum_progress_rad > config_.lookahead_distance_rad ||
    !positive(config_.stall_timeout_sec) || config_.projection_search_segments == 0U)
  {
    throw std::invalid_argument("invalid joint path follower configuration");
  }
}

bool JointPathFollower::setPath(
  const std::vector<Eigen::VectorXd> & path,
  const Eigen::VectorXd & current_position, const double time_sec)
{
  if (path.empty() || current_position.size() == 0 || !current_position.allFinite() ||
    !std::isfinite(time_sec) || time_sec < 0.0)
  {
    return false;
  }
  std::vector<Eigen::VectorXd> cleaned;
  cleaned.reserve(path.size());
  for (const auto & position : path) {
    if (position.size() != current_position.size() || !position.allFinite()) {
      return false;
    }
    const double length = cleaned.empty() ? 0.0 : (position - cleaned.back()).norm();
    if (!std::isfinite(length)) {
      return false;
    }
    if (cleaned.empty() || length > kLengthEpsilon) {
      cleaned.push_back(position);
    }
  }
  const Projection attachment = projectWithinWindow(
    cleaned, current_position, 0U, config_.projection_search_segments);
  if (!std::isfinite(attachment.distance) ||
    attachment.distance > config_.maximum_attach_distance_rad)
  {
    return false;
  }

  std::vector<Eigen::VectorXd> attached{current_position};
  const auto append = [&attached](const Eigen::VectorXd & position) {
      if ((position - attached.back()).norm() > kLengthEpsilon) {
        attached.push_back(position);
      }
    };
  append(attachment.position);
  for (std::size_t index = attachment.segment + 1U; index < cleaned.size(); ++index) {
    append(cleaned[index]);
  }
  std::vector<double> cumulative{0.0};
  cumulative.reserve(attached.size());
  for (std::size_t index = 1U; index < attached.size(); ++index) {
    cumulative.push_back(cumulative.back() + (attached[index] - attached[index - 1U]).norm());
    if (!std::isfinite(cumulative.back())) {
      return false;
    }
  }

  path_ = std::move(attached);
  cumulative_length_ = std::move(cumulative);
  segment_ = 0U;
  progress_anchor_rad_ = 0.0;
  last_progress_time_sec_ = time_sec;
  last_update_time_sec_ = time_sec;
  last_update_ = JointPathUpdate{};
  last_update_.status = JointPathStatus::kTracking;
  last_update_.total_length_rad = cumulative_length_.back();
  last_update_.reference_position = positionAt(config_.lookahead_distance_rad);
  return true;
}

JointPathUpdate JointPathFollower::update(
  const Eigen::VectorXd & current_position, const double time_sec)
{
  if (!active()) {
    return last_update_;
  }
  if (current_position.size() != path_.front().size() || !current_position.allFinite() ||
    !std::isfinite(time_sec) || time_sec < last_update_time_sec_)
  {
    last_update_.status = JointPathStatus::kInvalid;
    return last_update_;
  }
  last_update_time_sec_ = time_sec;
  const Projection projection = projectWithinWindow(
    path_, current_position, segment_, config_.projection_search_segments);
  last_update_.distance_from_path_rad = projection.distance;
  if (!std::isfinite(projection.distance) ||
    projection.distance > config_.maximum_corridor_distance_rad)
  {
    last_update_.status = JointPathStatus::kInvalid;
    return last_update_;
  }
  double projected_progress = 0.0;
  if (path_.size() > 1U) {
    projected_progress = cumulative_length_[projection.segment] + projection.fraction *
      (cumulative_length_[projection.segment + 1U] - cumulative_length_[projection.segment]);
  }
  last_update_.progress_rad = std::max(last_update_.progress_rad, projected_progress);
  while (segment_ + 2U < path_.size() &&
    cumulative_length_[segment_ + 1U] <= last_update_.progress_rad + kLengthEpsilon)
  {
    ++segment_;
  }
  last_update_.reference_position = positionAt(
    last_update_.progress_rad + config_.lookahead_distance_rad);
  if (last_update_.total_length_rad - last_update_.progress_rad <= config_.arrival_tolerance_rad &&
    (current_position - path_.back()).norm() <= config_.arrival_tolerance_rad)
  {
    last_update_.status = JointPathStatus::kArrived;
    last_update_.reference_position = path_.back();
  } else if (last_update_.progress_rad - progress_anchor_rad_ >= config_.minimum_progress_rad) {
    progress_anchor_rad_ = last_update_.progress_rad;
    last_progress_time_sec_ = time_sec;
  } else if (time_sec - last_progress_time_sec_ >= config_.stall_timeout_sec) {
    last_update_.status = JointPathStatus::kStalled;
  }
  return last_update_;
}

Eigen::VectorXd JointPathFollower::positionAt(const double distance_rad) const
{
  if (path_.size() == 1U || distance_rad >= cumulative_length_.back()) {
    return path_.back();
  }
  const auto upper = std::upper_bound(cumulative_length_.begin(), cumulative_length_.end(),
      distance_rad);
  const std::size_t segment = static_cast<std::size_t>(
    std::distance(cumulative_length_.begin(), upper)) - 1U;
  const double fraction = (distance_rad - cumulative_length_[segment]) /
    (cumulative_length_[segment + 1U] - cumulative_length_[segment]);
  return path_[segment] + fraction * (path_[segment + 1U] - path_[segment]);
}

void JointPathFollower::reset() noexcept
{
  path_.clear();
  cumulative_length_.clear();
  segment_ = 0U;
  last_update_ = JointPathUpdate{};
}

bool JointPathFollower::active() const noexcept
{
  return last_update_.status == JointPathStatus::kTracking;
}

}  // namespace face_tracking_arm::control
