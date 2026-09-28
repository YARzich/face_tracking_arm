// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_
#define FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/collision_aware_parameters.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/joint_path_follower.hpp"
#include "face_tracking_arm/idle_search.hpp"
#include "face_tracking_arm/posture_recovery_policy.hpp"
#include "face_tracking_arm/msg/tracking_target.hpp"
#include "face_tracking_arm/target_motion_estimator.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"

namespace face_tracking_arm::control
{

struct MotionReferenceResult
{
  std::optional<HierarchicalVelocityTask> task;
  std::optional<TrackingVelocityTaskResult> tracking;
  bool follows_path{false};
};

struct MotionReferenceDiagnostics
{
  std::string state{"HOLD"};
  std::string last_plan_message{"not requested"};
  std::uint64_t plans_requested{0};
  std::uint64_t plans_accepted{0};
  std::uint64_t plans_rejected{0};
  std::uint64_t paths_completed{0};
  std::uint64_t paths_abandoned{0};
  double last_plan_wall_ms{0.0};
  double path_progress_rad{0.0};
  double path_distance_rad{0.0};
  unsigned int recovery_relaxation{0};
};

/// Thread-confined task selection; the planner alone owns a background worker.
/// Produces velocity objectives, never changes the executor's q/v/a or publishes.
class MotionReference final
{
public:
  MotionReference(
    const rclcpp::Node::SharedPtr & node,
    const planning_scene_monitor::PlanningSceneMonitorPtr & scene_monitor,
    ControllerParameters parameters, JointMotionLimits limits,
    CollisionConstraintConfig collision_config);

  [[nodiscard]] MotionReferenceResult update(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    const planning_scene::PlanningScene & scene,
    double time_sec, std::uint64_t scene_revision);

  void reset();
  void rejectPath();
  [[nodiscard]] const MotionReferenceDiagnostics & diagnostics() const noexcept;

private:
  ControllerParameters parameters_;
  JointMotionLimits limits_;
  CollisionConstraintBuilder collision_validator_;
  const moveit::core::JointModelGroup * group_;
  const moveit::core::LinkModel * command_link_;
  const moveit::core::LinkModel * gaze_link_{nullptr};
  Eigen::VectorXd rest_positions_;
  Eigen::VectorXd search_positions_;
  Eigen::Index search_base_index_{0};
  IdleSearch search_;
  IdleSearchConfig search_config_;
  std::string search_pattern_;
  std::unique_ptr<BackgroundPathPlanner> planner_;
  JointPathFollower follower_;
  TargetMotionEstimator target_motion_estimator_;
  MotionReferenceDiagnostics diagnostics_;
  std::optional<std::uint8_t> mode_;
  std::optional<std::uint64_t> pending_generation_;
  double next_plan_time_sec_{0.0};
  bool rest_completed_{false};
  bool rest_settling_{false};
  bool acquisition_planning_{false};
  bool recovery_requested_{false};
  unsigned int recovery_failures_{0};
  unsigned int pending_relaxation_{0};
  unsigned int path_relaxation_{0};
  std::uint64_t recovery_attempt_{0};
  PointingProgressMonitor pointing_progress_;
  std::optional<Eigen::Vector3d> planned_face_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_
