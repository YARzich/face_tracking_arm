// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMPACT_PATH_ATTACHMENT_HPP_
#define COMPACT_PATH_ATTACHMENT_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/compact_joint_profile.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"

namespace face_tracking_arm::control::detail
{

struct CompactAttachmentSelection
{
  bool valid{false};
  std::string failure_reason;
  bool used_certificate{false};
  std::size_t sample_index{0U};
  std::size_t segment_index{0U};
  double segment_fraction{std::numeric_limits<double>::quiet_NaN()};
  Eigen::VectorXd position;
  double maximum_radius_m{std::numeric_limits<double>::quiet_NaN()};
  double maximum_roll_error_rad{std::numeric_limits<double>::quiet_NaN()};
  double maximum_abs_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
};

struct CompactPublishedPrefix
{
  std::vector<JointMotionState> samples;
};

[[nodiscard]] CompactJointProfile compactProfileForPublishedPrefix(
  const std::vector<JointMotionState> & prefix, double period_sec);

[[nodiscard]] CompactAttachmentSelection selectCompactAttachment(
  const BackgroundPathResult & plan, const Eigen::VectorXd & actual,
  const Eigen::VectorXd & queue, double radius_cap_m);

[[nodiscard]] std::optional<CompactPublishedPrefix> findCompactPublishedPrefix(
  const JointMotionState & expected_start, double now_sec,
  const PublishedTrajectoryRecord * latest,
  const PublishedTrajectoryRecord * active,
  double period_sec, const JointMotionLimits & limits,
  std::string * failure_reason);

}  // namespace face_tracking_arm::control::detail

#endif  // COMPACT_PATH_ATTACHMENT_HPP_
