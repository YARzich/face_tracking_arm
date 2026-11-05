// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__MOTION_REFERENCE_TYPES_HPP_
#define FACE_TRACKING_ARM__MOTION_REFERENCE_TYPES_HPP_

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"

namespace face_tracking_arm::control
{

struct CompactGoalReferenceSettings
{
  double gain{0.0};
  double maximum_velocity_radps{0.0};
};

struct MotionReferenceResult
{
  std::optional<HierarchicalVelocityTask> task;
  /// A prevalidated compact profile qdot for the next fixed-period command.
  /// The caller bypasses QP tracking only for this sample, then uses the shared
  /// candidate validation, braking, collision and queue-publication path.
  std::optional<Eigen::VectorXd> compact_profile_velocity;
  bool compact_profile_command{false};
  std::optional<TrackingVelocityTaskResult> tracking;
  bool follows_path{false};
  std::vector<LinearVelocityConstraint> compact_radius_constraints;
  // Physical screen-roll envelope rows are present for committed compact
  // motion and for FACE after its published stop envelope is proven in-band.
  std::vector<LinearVelocityConstraint> compact_roll_constraints;
  std::vector<LinearVelocityConstraint> compact_pitch_constraints;
  bool motion_reference_support_unavailable{false};
};


struct CompactEvent
{
  std::string event;
  std::int64_t stamp_ns{0};
  std::uint64_t generation_id{0};
  std::string reason;
  std::string details_json{"{}"};
};

struct MotionReferenceDiagnostics
{
  std::string state{"HOLD"};
  std::string tracking_phase{"FACE"};
  std::string recovery_mode{"compact"};
  std::string posture_mode{"unavailable"};
  std::string posture_unavailable_reason{"none"};
  std::string compact_phase{"INACTIVE"};
  std::string compact_reason;
  std::string last_plan_message{"not requested"};
  std::uint64_t plans_requested{0};
  std::uint64_t plans_accepted{0};
  std::uint64_t plans_rejected{0};
  std::uint64_t paths_completed{0};
  std::uint64_t paths_abandoned{0};
  std::uint64_t compact_generation{0};
  double last_plan_wall_ms{0.0};
  double compact_profile_duration_sec{std::numeric_limits<double>::quiet_NaN()};
  bool compact_profile_tempo_limited{false};
  double path_progress_rad{0.0};
  double path_distance_rad{0.0};
  double compact_entry_radius_cap_m{std::numeric_limits<double>::quiet_NaN()};
  double compact_effective_radius_cap_m{std::numeric_limits<double>::quiet_NaN()};
  double compact_pitch_cap_rad{std::numeric_limits<double>::quiet_NaN()};
  double planned_path_max_radius_m{std::numeric_limits<double>::quiet_NaN()};
  double planned_path_max_roll_error_rad{std::numeric_limits<double>::quiet_NaN()};
  double planned_path_max_abs_pitch_rad{std::numeric_limits<double>::quiet_NaN()};
  double compact_goal_reached{0.0};
  bool compact_radius_floor_latched{false};
  bool upright_roll_guard_armed{false};
  bool compact_actual_settled{false};
  bool disabled_limit_hold_latched{false};
  std::optional<double> compact_actual_goal_error_rad;
  std::optional<double> compact_actual_speed_max_rad_s;
  std::optional<double> compact_queue_tail_goal_error_rad;
  std::optional<double> compact_queue_tail_speed_max_rad_s;
  std::optional<double> compact_queue_tail_acceleration_max_rad_s2;
  std::uint64_t compact_event_overflow_count{0};
};


}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__MOTION_REFERENCE_TYPES_HPP_
