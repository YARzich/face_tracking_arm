// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__POSTURE_GEOMETRY_HPP_
#define FACE_TRACKING_ARM__POSTURE_GEOMETRY_HPP_

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <optional>

namespace face_tracking_arm::control
{

/// Signed elbow height relative to the infinite shoulder-wrist line in world Z.
/// A missing value means that the line is degenerate or its inputs are invalid.
[[nodiscard]] inline std::optional<double> evaluateUpwardBend(
  const Eigen::Vector3d & shoulder, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist)
{
  if (!shoulder.allFinite() || !elbow.allFinite() || !wrist.allFinite()) {
    return std::nullopt;
  }
  const Eigen::Vector3d line = wrist - shoulder;
  const double line_squared = line.squaredNorm();
  if (!std::isfinite(line_squared) ||
    line_squared <= std::numeric_limits<double>::epsilon())
  {
    return std::nullopt;
  }
  const Eigen::Vector3d offset = elbow - shoulder;
  const double height = (offset - line * (offset.dot(line) / line_squared)).z();
  return std::isfinite(height) ? std::optional<double>(height) : std::nullopt;
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__POSTURE_GEOMETRY_HPP_
