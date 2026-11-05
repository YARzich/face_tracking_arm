// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMPACT_MANEUVER_VALIDATION_HPP_
#define COMPACT_MANEUVER_VALIDATION_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/compact_geometry_validator.hpp"
#include "face_tracking_arm/compact_profile_worker.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"

namespace face_tracking_arm::control::detail
{

inline constexpr double kCompactRadiusFloorM{0.40};

struct CompactBridgeValidation
{
  double radius_cap{0.0};
  double maximum_radius{0.0};
  double maximum_abs_pitch{0.0};
  double maximum_abs_roll{0.0};
  std::string failure_reason;
  std::string failure_details;
};

/// Stateless compact path geometry and boundary validation.
///
/// Model-owned pointers remain valid through robot_model_; worker callbacks
/// returned by makeCompactProfileFinalGeometryValidator capture their own
/// immutable model, geometry, scene, and settings.
class CompactManeuverValidation final
{
public:
  CompactManeuverValidation(
    moveit::core::RobotModelConstPtr robot_model,
    const moveit::core::JointModelGroup * group,
    const moveit::core::LinkModel * command_link,
    const JointMotionLimits & limits,
    const CollisionConstraintBuilder & collision_validator,
    std::shared_ptr<const CompactGeometryValidator> compact_geometry_validator,
    std::size_t segment_validation_substeps);

  [[nodiscard]] CompactProfileFinalGeometryValidator makeCompactProfileFinalGeometryValidator(
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & state,
    std::uint64_t scene_revision, double radius_cap_m,
    std::optional<CompactReturnRegion> return_region = std::nullopt) const;

  [[nodiscard]] bool validateCompactPathPosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    double radius_cap_m, std::string * violation = nullptr) const;

  [[nodiscard]] bool validateCompactReturnRegion(
    const moveit::core::RobotState & state, const Eigen::VectorXd & positions,
    const CompactReturnRegion & region) const;

  [[nodiscard]] bool validateCompactBridgeSegment(
    const moveit::core::RobotState & state, const Eigen::VectorXd & from,
    const Eigen::VectorXd & to, std::size_t profile_sample_index,
    CompactBridgeValidation & validation) const;

  [[nodiscard]] bool includeCompactEntryEnvelopePosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    double * maximum_radius, double * maximum_abs_pitch_rad = nullptr,
    double * maximum_abs_roll_rad = nullptr) const;

  [[nodiscard]] bool includeCompactEntryEnvelope(
    const moveit::core::RobotState & state, double time_sec,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication,
    double * maximum_radius, double * maximum_abs_pitch_rad = nullptr,
    double * maximum_abs_roll_rad = nullptr) const;

private:
  const moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup * const group_;
  const moveit::core::LinkModel * const command_link_;
  const JointMotionLimits & limits_;
  const CollisionConstraintBuilder & collision_validator_;
  const std::shared_ptr<const CompactGeometryValidator> compact_geometry_validator_;
  const std::size_t segment_validation_substeps_;
};

}  // namespace face_tracking_arm::control::detail

#endif  // COMPACT_MANEUVER_VALIDATION_HPP_
