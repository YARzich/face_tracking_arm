// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_
#define FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_

#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "face_tracking_arm/background_path_planner.hpp"
#include "face_tracking_arm/collision_aware_parameters.hpp"
#include "face_tracking_arm/collision_constraints.hpp"
#include "face_tracking_arm/compact_geometry_validator.hpp"
#include "face_tracking_arm/face_tracking_policy.hpp"
#include "face_tracking_arm/idle_search.hpp"
#include "face_tracking_arm/joint_path_follower.hpp"
#include "face_tracking_arm/msg/tracking_target.hpp"
#include "face_tracking_arm/published_trajectory_history.hpp"
#include "face_tracking_arm/target_motion_estimator.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"

namespace face_tracking_arm::control
{

struct MotionReferenceResult
{
  std::optional<HierarchicalVelocityTask> task;
  std::optional<TrackingVelocityTaskResult> tracking;
  struct FaceAlignmentMeasurement
  {
    double pointing_error_rad{std::numeric_limits<double>::infinity()};
    double upright_roll_error_rad{std::numeric_limits<double>::infinity()};
    std::int64_t stamp_ns{0};
  };
  std::optional<FaceAlignmentMeasurement> actual_face_alignment;
  std::optional<double> j1_hold_position_rad;
  std::optional<std::size_t> j1_joint_index;
  int j1_hold_direction{0};
  bool follows_path{false};
  std::vector<LinearVelocityConstraint> compact_radius_constraints;
  // Physical screen-roll envelope rows are present for committed compact
  // motion and for FACE after its published stop envelope is proven in-band.
  std::vector<LinearVelocityConstraint> compact_roll_constraints;
  std::vector<LinearVelocityConstraint> compact_pitch_constraints;
  bool motion_reference_support_unavailable{false};
};

struct CompactPathReferenceSettings
{
  double gain{0.0};
  double maximum_velocity_radps{0.0};
};

/// Build the compact PATH velocity reference from measured path progress and
/// lookahead, before the existing maximum-velocity cap is applied.
[[nodiscard]] std::optional<Eigen::VectorXd> makeCompactPathVelocityReference(
  const Eigen::VectorXd & current_position, const JointPathUpdate & path,
  double lookahead_distance_rad, double speed_scale);

/// Apply one positive scale to the existing compact path gain and speed cap.
[[nodiscard]] CompactPathReferenceSettings scaleCompactPathReference(
  double gain, double maximum_velocity_radps, double speed_scale);

/// Project a planning-frame face direction into the joint-1 root coordinates.
[[nodiscard]] double preferredBaseJointAngle(
  const Eigen::Vector3d & face_in_planning_frame,
  const Eigen::Isometry3d & base_pose_in_planning_frame);

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
  bool compact_return_full_face_active{false};
  bool compact_return_face_qualified{false};
  bool disabled_limit_hold_latched{false};
  double compact_return_face_dwell_sec{0.0};
  std::optional<double> compact_actual_goal_error_rad;
  std::optional<double> compact_actual_speed_max_rad_s;
  std::optional<double> compact_queue_tail_goal_error_rad;
  std::optional<double> compact_queue_tail_speed_max_rad_s;
  std::optional<double> compact_queue_tail_acceleration_max_rad_s2;
  int j1_hold_direction{0};
  double j1_hold_q{0.0};
  std::uint64_t compact_event_overflow_count{0};
};

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

  /// Record evidence only after a FACE candidate was accepted and published.
  /// Failed QP/candidate/publication cycles must call with published=false.
  void recordFaceCommand(
    const TrackingVelocityTaskResult & tracking, const QpResult & qp_result,
    double sim_time_sec, bool published,
    const std::optional<MotionReferenceResult::FaceAlignmentMeasurement> &
    actual_face_alignment = std::nullopt);

  /// Preserve the immutable compact goal across a controlled re-arm by default.
  /// Pass false only for a deliberate full reset/time-epoch change.
  void reset(bool preserve_compact_maneuver = true);
  void rejectPath();
  [[nodiscard]] const MotionReferenceDiagnostics & diagnostics() const noexcept;
  [[nodiscard]] std::vector<CompactEvent> drainCompactEvents();
  [[nodiscard]] bool compactManeuverActive() const noexcept;
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

  enum class CompactPreparationSubmitOutcome
  {
    kNotSubmitted,
    kEntryEnvelopeUnsafe,
    kCandidateRejected,
    kSubmitted
  };

  static constexpr std::size_t kCompactEventCapacity{64U};

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
    double maximum_abs_screen_pitch_rad{0.0};
    bool exhausted{false};
  };

  void appendCompactEvent(
    std::string event, double time_sec, std::string reason = {},
    std::string details_json = "{}");
  void requestCompactManeuver(
    const moveit::core::RobotState & state, double time_sec,
    const planning_scene::PlanningScene & scene,
    std::uint64_t scene_revision, std::string reason,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication);
  [[nodiscard]] bool handleCompactPreparation(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    const TargetMotionEstimate & target_motion,
    const planning_scene::PlanningScene & scene,
    double time_sec, std::uint64_t scene_revision,
    const Eigen::VectorXd & position,
    bool fresh_actual_state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication,
    MotionReferenceResult & result);
  [[nodiscard]] bool handleRestSearchMode(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    double time_sec, std::uint64_t scene_revision,
    const Eigen::VectorXd & position,
    bool settled, bool fresh_actual_state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    MotionReferenceResult & result);
  void updateFaceTargetIdentity(
    const std::optional<msg::TrackingTarget> & target, double time_sec);
  void submitCompactPlan(
    const moveit::core::RobotState & state, std::uint64_t scene_revision,
    double time_sec);
  [[nodiscard]] CompactPreparationSubmitOutcome submitCompactPreparationPlan(
    const moveit::core::RobotState & state, double time_sec,
    std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication);
  void consumeCompactPreparationResult(
    const BackgroundPathResult & plan, const moveit::core::RobotState & state,
    double time_sec, const planning_scene::PlanningScene & scene,
    std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication);
  void adoptCommittedCompactPlan(
    const BackgroundPathResult & plan, const Eigen::VectorXd & current_position,
    double time_sec, std::uint64_t scene_revision);
  [[nodiscard]] bool compactSettled(
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const Eigen::VectorXd & goal) const;
  void recordCompactSettledEvent(
    double time_sec, bool settled_now,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state);
  [[nodiscard]] bool motionSettled(
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state) const;
  [[nodiscard]] bool compactGoalPositionReached(
    const JointMotionState & feedback, const Eigen::VectorXd & goal) const;
  [[nodiscard]] static bool hasAvailablePairYawResource(
    const PairYawResourceAssessment & assessment) noexcept;
  void updatePairYawResourceAndLimitHold(
    const moveit::core::RobotState & state,
    const std::optional<msg::TrackingTarget> & target,
    const TargetMotionEstimate & target_motion, double time_sec, bool fresh_actual_state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state);
  [[nodiscard]] bool appendCompactPathConstraints(
    const moveit::core::RobotState & state,
    const JointMotionState & motion_state,
    MotionReferenceResult & result) const;
  [[nodiscard]] bool appendUprightConstraints(
    const moveit::core::RobotState & state,
    const JointMotionState & motion_state,
    MotionReferenceResult & result, bool constrain_pitch) const;
  [[nodiscard]] bool appendFaceRollGuardConstraints(
    const moveit::core::RobotState & state,
    const std::optional<JointMotionState> & command_queue_state,
    MotionReferenceResult & result) const;
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
  void maybeLatchCompactRadiusCap(
    const moveit::core::RobotState & state, double time_sec,
    double publication_lookback_sec,
    const JointMotionState & actual_feedback,
    const JointMotionState & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication);
  [[nodiscard]] double effectiveCompactRadiusCap() const noexcept;
  [[nodiscard]] std::vector<CompactGoal> selectDefaultCompactGoals(
    const moveit::core::RobotState & state,
    const planning_scene::PlanningScene & scene,
    double measured_joint6) const;
  [[nodiscard]] std::optional<HierarchicalVelocityTask> compactTask(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    double time_sec);
  [[nodiscard]] std::optional<HierarchicalVelocityTask> pathTask(
    const moveit::core::RobotState & state, const Eigen::VectorXd & position,
    const Eigen::VectorXd & waypoint, double gain, double speed,
    bool upright_roll_primary, double * measured_roll_error = nullptr,
    bool horizontal_edge_primary = false);
  [[nodiscard]] Eigen::VectorXd groupPosition(
    const moveit::core::RobotState & state) const;
  void updateBaseHandoff(
    const moveit::core::RobotState & state,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & feedback_time_sec,
    const msg::TrackingTarget & target, const TargetMotionEstimate & target_motion,
    const std::optional<JointMotionState> & command_queue_state,
    MotionReferenceResult & result);
  [[nodiscard]] std::optional<HierarchicalVelocityTask> faceTask(
    const moveit::core::RobotState & state, const msg::TrackingTarget & target,
    const TargetMotionEstimate & target_motion, MotionReferenceResult & result,
    bool face_tracking = true,
    FaceReturnPointingTask * return_pointing = nullptr);
  [[nodiscard]] std::optional<MotionReferenceResult::FaceAlignmentMeasurement>
  measureActualFaceAlignment(
    const moveit::core::RobotState & state, const msg::TrackingTarget & target,
    double time_sec, const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec) const;
  [[nodiscard]] PairYawResourceAssessment evaluatePairYawResource(
    const moveit::core::RobotState & state, const msg::TrackingTarget & target,
    const TargetMotionEstimate & target_motion,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state) const;
  [[nodiscard]] static std::optional<Eigen::Vector2d> pairYawProbeEdgeDeltaInterval(
    double initial_edge_sine) noexcept;
  [[nodiscard]] std::optional<HierarchicalVelocityTask> faceReturnTask(
    const moveit::core::RobotState & state, const msg::TrackingTarget & target,
    const TargetMotionEstimate & target_motion, MotionReferenceResult & result,
    double time_sec, const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec, bool fresh_actual_state);

  ControllerParameters parameters_;
  RecoveryMode recovery_mode_{RecoveryMode::kCompact};
  JointMotionLimits limits_;
  CollisionConstraintBuilder collision_validator_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  const moveit::core::LinkModel * command_link_{nullptr};
  const moveit::core::LinkModel * base_link_{nullptr};
  Eigen::VectorXd rest_positions_;
  Eigen::VectorXd search_positions_;
  Eigen::Index search_base_index_{0};
  std::optional<IdleSearch> search_;
  IdleSearchConfig search_config_;
  double compact_speed_scale_{1.0};
  CompactPathReferenceSettings compact_path_settings_{6.0, 0.70};
  CompactPathReferenceSettings compact_goal_settings_{3.0, 0.10};
  bool search_enabled_{true};
  std::string search_pattern_;
  std::unique_ptr<BackgroundPathPlanner> planner_;
  double planner_goal_tolerance_rad_{0.01};
  JointPathFollower follower_;
  TargetMotionEstimator target_motion_estimator_;
  std::unique_ptr<CompactGeometryValidator> compact_geometry_validator_;
  CompactGoal compact_goal_;
  std::vector<CompactGoal> default_compact_goals_;
  std::optional<CompactPreparation> compact_preparation_;
  bool select_default_compact_goal_{false};
  FaceTrackingPolicy face_policy_;
  BaseJointHandoffPolicy base_handoff_policy_;
  MotionReferenceDiagnostics diagnostics_;
  RestMode rest_mode_{RestMode::kNone};
  bool rest_upright_start_active_{false};
  std::optional<std::uint64_t> pending_generation_;
  std::optional<std::uint64_t> active_scene_revision_;
  PairYawResourceAssessment pair_yaw_assessment_;
  bool disabled_limit_hold_latched_{false};
  std::optional<std::int64_t> disabled_limit_hold_source_stamp_ns_;
  std::optional<msg::TrackingTarget> last_real_face_target_;
  std::optional<TargetMotionEstimate> last_real_face_motion_;
  std::optional<double> last_control_time_sec_;
  std::optional<double> last_face_return_evidence_time_sec_;
  double face_return_dwell_sec_{0.0};
  bool compact_return_full_face_active_{false};
  bool compact_plan_pending_{false};
  bool compact_path_adopted_{false};
  bool compact_goal_reached_{false};
  bool compact_radius_floor_latched_{false};
  bool compact_path_endpoint_reached_{false};
  bool compact_commit_initialized_{false};
  bool compact_face_return_pending_{false};
  bool compact_goal_event_emitted_{false};
  bool compact_path_started_event_emitted_{false};
  bool compact_profile_unavailable_reported_{false};
  bool upright_roll_guard_armed_{false};
  int target_bearing_motion_sign_{0};
  bool target_bearing_moving_{false};
  bool rest_completed_{false};
  bool rest_settling_{false};
  std::optional<std::int64_t> target_id_;
  std::optional<std::int64_t> handoff_target_id_;
  std::deque<CompactEvent> compact_events_;
  std::uint64_t compact_event_overflow_count_{0};
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__MOTION_REFERENCE_HPP_
