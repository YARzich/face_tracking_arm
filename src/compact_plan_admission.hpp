// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COMPACT_PLAN_ADMISSION_HPP_
#define COMPACT_PLAN_ADMISSION_HPP_

#include <optional>
#include <cstdint>
#include <string>
#include <vector>

#include "compact_maneuver_validation.hpp"
#include "compact_path_attachment.hpp"

namespace face_tracking_arm::control::detail
{

struct CompactPlanAdmissionEnvironment
{
  const moveit::core::RobotState & state;
  const planning_scene::PlanningScene & scene;
  const moveit::core::JointModelGroup & group;
  const CompactGeometryValidator * geometry{nullptr};
  const CollisionConstraintBuilder & collision_validator;
  const CompactManeuverValidation & validation;
  const JointMotionLimits & compact_limits;
  const JointMotionLimits & prefix_limits;
  const JointMotionState * actual_feedback{nullptr};
  const JointMotionState * command_queue_state{nullptr};
  const PublishedTrajectoryRecord * active_publication{nullptr};
  const PublishedTrajectoryRecord * latest_publication{nullptr};
  double time_sec{0.0};
  double control_period_sec{0.0};
  double compact_speed_scale{1.0};
  double compact_return_speed_scale{1.0};
  double compact_fold_speed_scale{1.0};
  double compact_turn_speed_scale{1.0};
  bool compact_active{false};
};

enum class CompactAdmissionDisposition { kReady, kWait, kReject };

struct CompactAdmissionMetadata
{
  bool phase_scales_computed{false};
  bool profile_tempo_limited{false};
};

struct CompactAdmittedPlan
{
  double entry_radius_cap_m{0.0};
  double entry_max_pitch_rad{0.0};
  double entry_max_roll_rad{0.0};
  double actual_to_attachment_rad{0.0};
  double queue_to_attachment_rad{0.0};
  CompactPublishedPrefix published_prefix;
  JointMotionState compact_start;
  CompactAttachmentSelection attachment;
  CompactBridgeValidation bridge_validation;
  std::vector<Eigen::VectorXd> path;
  std::vector<double> phase_scales;
  CompactJointProfile return_profile;
};

struct CompactPlanAdmissionResult
{
  CompactAdmissionDisposition disposition{CompactAdmissionDisposition::kReject};
  std::string reason;
  std::string additional_details;
  std::string prefix_failure;
  CompactAdmissionMetadata metadata;
  std::optional<CompactAdmittedPlan> admitted;
};

struct CompactPreparationAdmissionInput
{
  const BackgroundPathResult & plan;
  const CompactGoal & candidate;
  const Eigen::VectorXd & submitted_start_positions;
  const std::optional<CompactReturnRegion> & return_region;
  double submitted_entry_radius_cap_m{0.0};
  std::uint64_t scene_revision{0U};
  double planner_goal_tolerance_rad{0.0};
  bool fresh_actual_feedback{false};
  double minimum_entry_radius_cap_m{0.0};
};

struct CompactCommittedAdmissionInput
{
  const BackgroundPathResult & plan;
  const CompactGoal & goal;
  std::uint64_t scene_revision{0U};
  double effective_radius_cap_m{0.0};
  double return_radius_cap_m{0.0};
};

[[nodiscard]] CompactPlanAdmissionResult admitCompactPreparationPlan(
  const CompactPreparationAdmissionInput & input,
  const CompactPlanAdmissionEnvironment & environment);

[[nodiscard]] CompactPlanAdmissionResult admitCommittedCompactPlan(
  const CompactCommittedAdmissionInput & input,
  const CompactPlanAdmissionEnvironment & environment);

}  // namespace face_tracking_arm::control::detail

#endif  // COMPACT_PLAN_ADMISSION_HPP_
