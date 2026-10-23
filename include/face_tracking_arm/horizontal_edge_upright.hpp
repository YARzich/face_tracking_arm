// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <optional>

namespace face_tracking_arm::control
{
struct HorizontalEdgeUprightPose
{
  Eigen::Vector3d horizontal_edge;
  Eigen::Vector3d vertical_edge;
  double signed_edge_sine{0.0};
};

inline bool validEdgeRotation(const Eigen::Matrix3d & rotation)
{
  constexpr double kRotationTolerance = 1.0e-9;
  return rotation.allFinite() &&
         (rotation.transpose() * rotation).isApprox(
    Eigen::Matrix3d::Identity(), kRotationTolerance) &&
         std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

inline std::optional<HorizontalEdgeUprightPose> evaluateHorizontalEdgeUprightWithSineCap(
  const Eigen::Matrix3d & rotation, double maximum_abs_edge_sine)
{
  if (!validEdgeRotation(rotation) || !std::isfinite(maximum_abs_edge_sine) ||
    maximum_abs_edge_sine < 0.0 || maximum_abs_edge_sine > 1.0)
  {
    return std::nullopt;
  }
  const Eigen::Vector3d h = rotation.col(1);
  const Eigen::Vector3d u = rotation.col(2);
  if (std::abs(h.z()) > maximum_abs_edge_sine || u.z() < 0.0) {return std::nullopt;}
  return HorizontalEdgeUprightPose{h, u, h.z()};
}

inline std::optional<HorizontalEdgeUprightPose> evaluateHorizontalEdgeUpright(
  const Eigen::Matrix3d & rotation, double maximum_tilt_rad)
{
  if (!std::isfinite(maximum_tilt_rad) || maximum_tilt_rad < 0.0 ||
    maximum_tilt_rad > 0.5 * M_PI)
  {
    return std::nullopt;
  }
  return evaluateHorizontalEdgeUprightWithSineCap(
    rotation, std::sin(maximum_tilt_rad));
}

}  // namespace face_tracking_arm::control
