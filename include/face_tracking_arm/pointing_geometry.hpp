// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__POINTING_GEOMETRY_HPP_
#define FACE_TRACKING_ARM__POINTING_GEOMETRY_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

namespace face_tracking_arm
{

/// Express an optical camera (+Z forward, +X right, +Y down) as the screen
/// convention (+X forward, +Y left, +Z up), without changing its physical origin.
inline Eigen::Matrix3d opticalToPointingRotation()
{
  Eigen::Matrix3d rotation;
  rotation.col(0) = Eigen::Vector3d::UnitZ();
  rotation.col(1) = -Eigen::Vector3d::UnitX();
  rotation.col(2) = -Eigen::Vector3d::UnitY();
  return rotation;
}

/// +X points at the target. Roll is upright where defined; at the pole any
/// perpendicular up direction is valid. Callers must treat roll as a soft task.
inline std::optional<Eigen::Matrix3d> pointingRotation(
  const Eigen::Vector3d & ray, const Eigen::Vector3d & preferred_up = Eigen::Vector3d::UnitZ())
{
  if (!ray.allFinite() || !preferred_up.allFinite() || ray.norm() <= 1.0e-6) {
    return std::nullopt;
  }
  const Eigen::Vector3d direction = ray.normalized();
  Eigen::Vector3d up = preferred_up - preferred_up.dot(direction) * direction;
  if (up.norm() <= 1.0e-6) {
    up = direction.unitOrthogonal();
  }
  up.normalize();
  Eigen::Matrix3d rotation;
  rotation.col(0) = direction;
  rotation.col(1) = up.cross(direction);
  rotation.col(2) = up;
  return rotation;
}

}  // namespace face_tracking_arm

#endif  // FACE_TRACKING_ARM__POINTING_GEOMETRY_HPP_
