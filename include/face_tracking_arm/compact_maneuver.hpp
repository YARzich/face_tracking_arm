// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__COMPACT_MANEUVER_HPP_
#define FACE_TRACKING_ARM__COMPACT_MANEUVER_HPP_

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/node.hpp>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/collision_aware_parameters.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/compact_geometry_validator.hpp"
#include "face_tracking_arm/compact_joint_profile.hpp"
#include "face_tracking_arm/compact_path_execution.hpp"
#include "face_tracking_arm/compact_profile_worker.hpp"
#include "face_tracking_arm/face_tracking_policy.hpp"
#include "face_tracking_arm/joint_path_follower.hpp"
#include "face_tracking_arm/motion_reference_types.hpp"
#include "face_tracking_arm/msg/tracking_target.hpp"
#include "face_tracking_arm/posture_task_builder.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"
#include "face_tracking_arm/target_motion_estimator.hpp"

namespace face_tracking_arm::control
{

struct CompactManeuverContext
{
  const moveit::core::RobotState & state;
  const planning_scene::PlanningScene & scene;
  const std::optional<msg::TrackingTarget> & target;
  const TargetMotionEstimate & target_motion;
  const J1BoundaryAssessment & j1_boundary_assessment;
  double time_sec{0.0};
  std::uint64_t scene_revision{0U};
  const Eigen::VectorXd & position;
  bool settled{false};
  bool fresh_actual_state{false};
  const std::optional<JointMotionState> & actual_feedback;
  const std::optional<double> & actual_feedback_time_sec;
  const std::optional<JointMotionState> & command_queue_state;
  const PublishedTrajectoryRecord * active_publication{nullptr};
  const PublishedTrajectoryRecord * latest_publication{nullptr};
  bool upright_roll_guard_armed{false};
};

enum class CompactFaceDirective
{
  kNone,
  kBuildFace,
  kBuildFaceAndGuard,
  kGuardOnly
};

struct CompactManeuverOutcome
{
  bool handled{false};
  CompactFaceDirective face_directive{CompactFaceDirective::kNone};
  MotionReferenceResult result;
  bool disarm_face_roll_guard{false};
};

class MotionReferenceTestPeer;

/// Owns compact preparation and execution state while borrowing the shared
/// planning, following, model, and diagnostics services from MotionReference.
class CompactManeuver final
{
public:
  CompactManeuver(
    const rclcpp::Node::SharedPtr & node, bool compact_recovery_enabled,
    const ControllerParameters & parameters, const JointMotionLimits & limits,
    const CollisionConstraintBuilder & collision_validator,
    moveit::core::RobotModelConstPtr robot_model,
    const moveit::core::JointModelGroup * group,
    const moveit::core::LinkModel * command_link,
    const moveit::core::LinkModel * base_link,
    const PostureTaskBuilder * posture_task_builder,
    const rclcpp::Clock::SharedPtr & clock,
    BackgroundPathPlanner & planner, JointPathFollower & follower,
    MotionReferenceDiagnostics & diagnostics);

  [[nodiscard]] FaceTrackingPhase phase() const noexcept;
  [[nodiscard]] bool active() const noexcept;
  void requestPhase();
  [[nodiscard]] bool updateTargetIdentity(
    const std::optional<msg::TrackingTarget> & target, double time_sec);
  void consumeCompactProfileWorkResult(double time_sec, std::uint64_t scene_revision);
  void beginTick(const CompactManeuverContext & context);
  [[nodiscard]] CompactManeuverOutcome update(const CompactManeuverContext & context);
  void requestCompactManeuver(
    const CompactManeuverContext & context, std::string reason);
  void onJ1Boundary(
    const CompactManeuverContext & context, std::string reason,
    std::string boundary_details);
  void cancelProfileWork();
  [[nodiscard]] bool takeFaceRollGuardDisarmRequest() noexcept;
  [[nodiscard]] bool reset(
    bool preserve_compact_maneuver,
    bool disabled_limit_hold_latched);
  void rejectPath();

  [[nodiscard]] std::vector<CompactEvent> drainEvents();
  void recordFaceRollGuardArmed(
    double time_sec, std::string reason, std::string details_json);
  [[nodiscard]] bool hasActiveProfile() const noexcept;
  [[nodiscard]] const JointMotionLimits & activeCommandLimits() const noexcept;
  void recordProfileCommandPublished(bool published) noexcept;
  [[nodiscard]] EmergencyBrakeTail profileBrakingTail(
    const JointMotionState & candidate_state,
    const EmergencyBrakeTailConfig & config) const;
  [[nodiscard]] bool validateCompactPathPosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    std::string * violation = nullptr) const;
  [[nodiscard]] double effectiveCompactRadiusCap() const noexcept;

private:
  friend class MotionReferenceTestPeer;
  static constexpr std::size_t kCompactEventCapacity{64U};

  enum class CompactPreparationSubmitOutcome
  {
    kNotSubmitted,
    kEntryEnvelopeUnsafe,
    kCandidateRejected,
    kSubmitted
  };

  enum class CompactProfileWorkKind { kInitial, kReadoption, kResync };

  struct CompactPreparation
  {
    std::vector<CompactGoal> candidates;
    std::size_t candidate_index{0U};
    Eigen::VectorXd submitted_start_positions;
    Eigen::VectorXd exhausted_start_positions;
    std::uint64_t scene_revision{0U};
    std::uint64_t generation{0U};
    std::uint64_t exhausted_scene_revision{0U};
    double entry_radius_cap_m{0.0};
    std::optional<CompactReturnRegion> return_region;
    bool exhausted{false};
  };

  struct PendingCompactProfileWork
  {
    std::uint64_t request_id{0U};
    std::uint64_t generation{0U};
    std::uint64_t scene_revision{0U};
    CompactProfileWorkKind kind{CompactProfileWorkKind::kInitial};
    std::vector<JointMotionState> fixed_prefix;
    std::size_t prefix_endpoint_sample_index{0U};
    std::size_t source_route_sample_count{0U};
  };

  [[nodiscard]] bool compactManeuverActive() const noexcept;
  [[nodiscard]] bool handleCompactPreparation(
    const CompactManeuverContext & context, CompactManeuverOutcome & outcome);
  void submitCompactPlan(const CompactManeuverContext & context);
  [[nodiscard]] CompactPreparationSubmitOutcome submitCompactPreparationPlan(
    const CompactManeuverContext & context);
  void consumeCompactPreparationResult(
    const BackgroundPathResult & plan, const CompactManeuverContext & context);
  void adoptCommittedCompactPlan(
    const BackgroundPathResult & plan, const CompactManeuverContext & context);
  void cancelCompactProfileWork();
  [[nodiscard]] CompactProfileFinalGeometryValidator makeCompactProfileFinalGeometryValidator(
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & state,
    std::uint64_t scene_revision, double radius_cap_m,
    std::optional<CompactReturnRegion> return_region = std::nullopt) const;
  [[nodiscard]] std::uint64_t submitCompactProfileWork(
    CompactProfileWorkRequest request,
    const moveit::core::RobotState & geometry_state,
    const planning_scene::PlanningScene & scene,
    double radius_cap_m, double return_radius_cap_m,
    std::optional<CompactReturnRegion> return_region,
    double time_sec, bool resync = false);
  [[nodiscard]] bool compactSettled(
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const Eigen::VectorXd & goal) const;
  void recordCompactSettledEvent(
    const CompactManeuverContext & context, bool settled_now);
  [[nodiscard]] bool motionSettled(
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state) const;
  void finishCompactReturnProfileStage(
    const CompactManeuverContext & context, MotionReferenceResult & result);
  [[nodiscard]] bool serviceCompactReturnProfileStage(
    const CompactManeuverContext & context, MotionReferenceResult & result);
  [[nodiscard]] bool compactGoalPositionReached(
    const JointMotionState & feedback, const Eigen::VectorXd & goal) const;
  [[nodiscard]] bool validateCompactReturnRegion(
    const moveit::core::RobotState & state, const Eigen::VectorXd & positions,
    const CompactReturnRegion & region) const;
  [[nodiscard]] bool includeCompactEntryEnvelopePosition(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    double * maximum_radius, double * maximum_abs_pitch_rad = nullptr,
    double * maximum_abs_roll_rad = nullptr) const;
  [[nodiscard]] bool includeCompactEntryEnvelope(
    const CompactManeuverContext & context,
    double * maximum_radius, double * maximum_abs_pitch_rad = nullptr,
    double * maximum_abs_roll_rad = nullptr) const;
  void maybeLatchCompactRadiusCap(const CompactManeuverContext & context);
  [[nodiscard]] std::vector<CompactGoal> selectDefaultCompactGoals(
    const moveit::core::RobotState & state,
    const planning_scene::PlanningScene & scene,
    double measured_joint6) const;
  [[nodiscard]] std::optional<HierarchicalVelocityTask> compactTask(
    const CompactManeuverContext & context, MotionReferenceResult & result);
  void appendCompactEvent(
    std::string event, double time_sec, std::string reason = {},
    std::string details_json = "{}");
  [[nodiscard]] bool appendCompactPathConstraints(
    const moveit::core::RobotState & state,
    const JointMotionState & motion_state, MotionReferenceResult & result);
  [[nodiscard]] Eigen::VectorXd groupPosition(
    const moveit::core::RobotState & state) const;

  const ControllerParameters & parameters_;
  const JointMotionLimits & limits_;
  const CollisionConstraintBuilder & collision_validator_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  const moveit::core::LinkModel * command_link_{nullptr};
  const moveit::core::LinkModel * base_link_{nullptr};
  const PostureTaskBuilder * posture_task_builder_{nullptr};
  const rclcpp::Clock::SharedPtr & clock_;
  BackgroundPathPlanner & planner_;
  JointPathFollower & follower_;
  MotionReferenceDiagnostics & diagnostics_;

  JointMotionLimits compact_limits_;
  JointMotionLimits compact_return_limits_;
  double compact_speed_scale_{1.0};
  double compact_fold_speed_scale_{1.0};
  double compact_turn_speed_scale_{1.0};
  double compact_return_speed_scale_{1.0};
  CompactPathExecution compact_path_execution_;
  std::optional<CompactJointProfile> compact_prepared_return_profile_;
  Eigen::VectorXd compact_return_endpoint_;
  bool compact_return_profile_active_{false};
  std::unique_ptr<CompactProfileWorker> compact_profile_worker_;
  std::optional<PendingCompactProfileWork> pending_compact_profile_work_;
  double planner_goal_tolerance_rad_{0.01};
  std::shared_ptr<CompactGeometryValidator> compact_geometry_validator_;
  CompactGoal compact_goal_;
  CompactGoalReferenceSettings compact_goal_settings_{3.0, 0.10};
  std::vector<CompactGoal> default_compact_goals_;
  std::optional<CompactPreparation> compact_preparation_;
  bool select_default_compact_goal_{false};
  FaceTrackingPolicy face_policy_;
  std::optional<std::uint64_t> pending_generation_;
  std::optional<std::uint64_t> active_scene_revision_;
  bool compact_plan_pending_{false};
  bool compact_path_adopted_{false};
  bool compact_goal_reached_{false};
  bool compact_radius_floor_latched_{false};
  bool compact_path_endpoint_reached_{false};
  bool compact_commit_initialized_{false};
  bool compact_return_path_pending_{false};
  bool compact_goal_event_emitted_{false};
  bool compact_path_started_event_emitted_{false};
  bool compact_profile_unavailable_reported_{false};
  bool face_roll_guard_disarm_requested_{false};
  int target_bearing_motion_sign_{0};
  bool target_bearing_moving_{false};
  std::optional<std::int64_t> target_id_;
  std::deque<CompactEvent> compact_events_;
  std::uint64_t compact_event_overflow_count_{0U};
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__COMPACT_MANEUVER_HPP_
