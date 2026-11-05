// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_
#define FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/clock.hpp>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/face_tracking_policy.hpp"
#include "face_tracking_arm/motion_reference_types.hpp"
#include "face_tracking_arm/collision_aware_parameters.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/idle_search.hpp"
#include "face_tracking_arm/joint_path_follower.hpp"
#include "face_tracking_arm/msg/tracking_target.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"
#include "face_tracking_arm/target_motion_estimator.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"
#include "face_tracking_arm/posture_task_builder.hpp"

namespace face_tracking_arm::control
{

class CompactManeuver;
struct CompactManeuverContext;

/// Project a planning-frame face direction into the joint-1 root coordinates.
[[nodiscard]] double preferredBaseJointAngle(
  const Eigen::Vector3d & face_in_planning_frame,
  const Eigen::Isometry3d & base_pose_in_planning_frame);

/// Thread-confined task selector. The planner owns its background worker;
/// this class selects tasks but never changes or publishes the executor state.
class MotionReference final
{
public:
  MotionReference(
    const rclcpp::Node::SharedPtr & node,
    const planning_scene_monitor::PlanningSceneMonitorPtr & scene_monitor,
    ControllerParameters parameters, JointMotionLimits limits,
    CollisionConstraintConfig collision_config);
  ~MotionReference();

  /// Borrowed publication records are read synchronously to arm the FACE roll
  /// guard and to certify a compact commitment's entry envelope.
  [[nodiscard]] MotionReferenceResult update(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    const planning_scene::PlanningScene & scene,
    double time_sec, std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback = std::nullopt,
    const std::optional<double> & actual_feedback_time_sec = std::nullopt,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr);

  /// Preserve the immutable compact goal across a controlled re-arm by default.
  /// Pass false only for a deliberate full reset/time-epoch change.
  void reset(bool preserve_compact_maneuver = true);
  void rejectPath();
  [[nodiscard]] const MotionReferenceDiagnostics & diagnostics() const noexcept;
  [[nodiscard]] std::vector<CompactEvent> drainCompactEvents();
  [[nodiscard]] bool compactManeuverActive() const noexcept;
  [[nodiscard]] bool hasActiveCompactProfile() const noexcept;
  /// Dynamic limits for the command branch selected by the current compact
  /// phase; ordinary FACE receives the unchanged configured envelope.
  [[nodiscard]] const JointMotionLimits & activeCommandLimits() const noexcept;
  /// Advance only after a compact profile command was actually published.
  /// A false report retains the sample; queue-state mismatch invalidates it.
  void recordCompactProfileCommandPublished(bool published) noexcept;
  /// Return a profile-compatible braking suffix only for the current direct
  /// PATH sample. Copied prefix samples use their exact published remainder;
  /// scalar profile samples use a projected existing braking rollout.
  [[nodiscard]] EmergencyBrakeTail compactProfileBrakingTail(
    const JointMotionState & candidate_state,
    const EmergencyBrakeTailConfig & config) const;
  [[nodiscard]] bool validateCommandPosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    std::string * violation = nullptr) const;
  [[nodiscard]] bool validateCommandTransition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & previous_position,
    const Eigen::VectorXd & position, std::string * violation = nullptr) const;
  [[nodiscard]] bool validateCompactPathPosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    std::string * violation = nullptr) const;

private:
  friend class MotionReferenceTestPeer;

  enum class RecoveryMode
  {
    kCompact,
    kDisabled
  };

  enum class RestMode
  {
    kNone,
    kRest,
    kSearch
  };

  void updateControlClock(double time_sec);
  [[nodiscard]] bool handleRestSearchMode(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    double time_sec, std::uint64_t scene_revision,
    const Eigen::VectorXd & position,
    bool settled, bool fresh_actual_state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    MotionReferenceResult & result);
  [[nodiscard]] bool motionSettled(
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state) const;
  void updateJ1BoundaryAndLimitHold(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    const TargetMotionEstimate & target_motion,
    double time_sec, bool fresh_actual_state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state);
  [[nodiscard]] bool appendUprightConstraints(
    const moveit::core::RobotState & state,
    const JointMotionState & motion_state,
    MotionReferenceResult & result);
  [[nodiscard]] bool appendFaceRollGuardConstraints(
    const moveit::core::RobotState & state,
    const std::optional<JointMotionState> & command_queue_state,
    MotionReferenceResult & result);
  [[nodiscard]] bool validateUprightRollPose(
    const moveit::core::RobotState & state, std::string * violation) const;
  [[nodiscard]] bool validateUprightRollPosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    std::string * violation) const;
  [[nodiscard]] bool maybeArmFaceUprightRollGuard(
    const moveit::core::RobotState & state, double time_sec,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication);
  [[nodiscard]] std::optional<HierarchicalVelocityTask> pathTask(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    const Eigen::VectorXd & waypoint, double gain, double speed,
    bool upright_roll_primary, double * measured_roll_error = nullptr,
    bool horizontal_edge_primary = false);
  [[nodiscard]] Eigen::VectorXd groupPosition(
    const moveit::core::RobotState & state) const;
  [[nodiscard]] std::optional<HierarchicalVelocityTask> faceTask(
    const moveit::core::RobotState & state, const msg::TrackingTarget & target,
    const TargetMotionEstimate & target_motion, MotionReferenceResult & result,
    bool face_tracking = true);

  [[nodiscard]] CompactManeuverContext compactContext(
    const moveit::core::RobotState & state,
    const planning_scene::PlanningScene & scene,
    const std::optional<msg::TrackingTarget> & target,
    const TargetMotionEstimate & target_motion, double time_sec,
    std::uint64_t scene_revision, const Eigen::VectorXd & position,
    bool settled, bool fresh_actual_state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication) const;

  rclcpp::Clock::SharedPtr clock_;
  ControllerParameters parameters_;
  RecoveryMode recovery_mode_{RecoveryMode::kCompact};
  JointMotionLimits limits_;
  CollisionConstraintBuilder collision_validator_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  const moveit::core::LinkModel * command_link_{nullptr};
  const moveit::core::LinkModel * base_link_{nullptr};
  std::optional<PostureTaskBuilder> posture_task_builder_;
  Eigen::VectorXd rest_positions_;
  Eigen::VectorXd search_positions_;
  Eigen::Index search_base_index_{0};
  std::optional<IdleSearch> search_;
  IdleSearchConfig search_config_;
  bool search_enabled_{true};
  std::string search_pattern_;
  bool search_paused_{false};
  std::unique_ptr<BackgroundPathPlanner> planner_;
  JointPathFollower follower_;
  TargetMotionEstimator target_motion_estimator_;
  MotionReferenceDiagnostics diagnostics_;
  RestMode rest_mode_{RestMode::kNone};
  bool rest_upright_start_active_{false};
  std::optional<std::uint64_t> pending_generation_;
  J1BoundaryAssessment j1_boundary_assessment_;
  bool disabled_limit_hold_latched_{false};
  std::optional<std::int64_t> disabled_limit_hold_source_stamp_ns_;
  std::optional<msg::TrackingTarget> last_real_face_target_;
  std::optional<TargetMotionEstimate> last_real_face_motion_;
  std::optional<double> last_control_time_sec_;
  bool upright_roll_guard_armed_{false};
  bool rest_completed_{false};
  bool rest_settling_{false};
  std::unique_ptr<CompactManeuver> compact_maneuver_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_
