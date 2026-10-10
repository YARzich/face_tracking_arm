// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_GEOMETRY_VALIDATOR_HPP_
#define FACE_TRACKING_ARM__COMPACT_GEOMETRY_VALIDATOR_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>

namespace face_tracking_arm::control
{

/// Two-sided distance from a bounded joint position to its physical limits.
struct BoundedJointReserve
{
  double lower_margin_rad{0.0};
  double upper_margin_rad{0.0};
  double nearest_bound_margin_rad{0.0};
  bool inside_bounds{false};
};

/// One active support row for a per-link radial envelope maximum.
struct CompactRadiusSupport
{
  double radius_m{0.0};
  /// Derivative of this support's radius with respect to group variables [m/rad].
  Eigen::VectorXd outward_gradient;
};

/// Co-active rows for every link with modeled visual or collision geometry.
struct CompactRadiusSupportSet
{
  double maximum_radius_m{0.0};
  std::vector<CompactRadiusSupport> supports;
};

/// Measure the distance to each physical bound without wrapping revolute joints.
[[nodiscard]] BoundedJointReserve measureBoundedJointReserve(
  double position_rad, double lower_bound_rad, double upper_bound_rad) noexcept;

/// Cached visual and collision geometry envelope used by compact-path admission.
///
/// The envelope is measured about the base link's local Z axis. Mesh vertex
/// extrema, box corners, and sphere extrema are exact; cylinders and cones use
/// conservative axial-endpoint plus radius bounds. Mesh resources and URDF
/// visual/collision geometry are cached at construction; queries perform FK
/// transforms and envelope evaluation only.
class CompactGeometryValidator final
{
public:
  CompactGeometryValidator(
    moveit::core::RobotModelConstPtr robot_model, std::string base_link_name);
  ~CompactGeometryValidator();

  CompactGeometryValidator(const CompactGeometryValidator &) = delete;
  CompactGeometryValidator & operator=(const CompactGeometryValidator &) = delete;

  /// Maximum radial distance of visual or collision geometry in the base XY plane.
  /// The caller must update FK after changing joint positions.
  [[nodiscard]] double maximumAssemblyRadius(
    const moveit::core::RobotState & state) const;

  /// Return co-active radial supports for each geometry-bearing link.
  ///
  /// Each row is the exact derivative of a support radius near that link's
  /// maximum; the caller applies its own global radius cap to every row.
  /// Returns nullopt when the model/state/Jacobian is invalid or the bounded
  /// support budget cannot retain the complete co-active set.
  [[nodiscard]] std::optional<CompactRadiusSupportSet> perLinkRadiusSupports(
    const moveit::core::RobotState & state,
    const moveit::core::JointModelGroup & group,
    double coactive_tolerance_m = 1.0e-6,
    std::size_t maximum_supports = 64) const;

private:
  struct CachedShape;

  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::LinkModel * base_link_{nullptr};
  std::vector<const moveit::core::LinkModel *> links_;
  std::vector<CachedShape> shapes_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_GEOMETRY_VALIDATOR_HPP_
