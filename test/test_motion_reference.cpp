// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Geometry>
#include <geometric_shapes/shapes.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "collision_aware_servo_component.hpp"
#include "face_tracking_arm/face_tracking_policy.hpp"
#include "face_tracking_arm/motion_reference.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/horizontal_edge_upright.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"

namespace face_tracking_arm::control
{

/// Narrow friend adapter for deterministic preparation lifecycle tests.
/// Synthetic successful plans represent a worker-certified path only in tests
/// that exercise commit/admission state; the real planner integration test
/// still obtains and validates its path from BackgroundPathPlanner.
class MotionReferenceTestPeer final
{
public:
  static void setFaceRollGuardArmedForTest(MotionReference & reference)
  {
    reference.upright_roll_guard_armed_ = true;
    reference.diagnostics_.upright_roll_guard_armed = true;
  }

  [[nodiscard]] static bool hasPreparation(const MotionReference & reference)
  {
    return reference.compact_preparation_.has_value();
  }

  [[nodiscard]] static PairYawResourceAssessment pairYawAssessment(
    const MotionReference & reference)
  {
    return reference.pair_yaw_assessment_;
  }

  [[nodiscard]] static std::size_t candidateIndex(const MotionReference & reference)
  {
    return reference.compact_preparation_ ?
           reference.compact_preparation_->candidate_index : 0U;
  }

  [[nodiscard]] static std::size_t candidateCount(const MotionReference & reference)
  {
    return reference.compact_preparation_ ?
           reference.compact_preparation_->candidates.size() : 0U;
  }

  [[nodiscard]] static CompactGoal candidate(
    const MotionReference & reference, const std::size_t index)
  {
    if (!reference.compact_preparation_ ||
      index >= reference.compact_preparation_->candidates.size())
    {
      return {};
    }
    return reference.compact_preparation_->candidates[index];
  }

  [[nodiscard]] static bool preparationExhausted(const MotionReference & reference)
  {
    return reference.compact_preparation_ && reference.compact_preparation_->exhausted;
  }

  [[nodiscard]] static std::optional<std::uint64_t> pendingGeneration(
    const MotionReference & reference)
  {
    return reference.pending_generation_;
  }

  static void clearDefaultCandidatesForTest(MotionReference & reference)
  {
    reference.default_compact_goals_.clear();
  }

  static void activateResultGeneration(
    MotionReference & reference, const std::uint64_t generation)
  {
    if (!reference.compact_preparation_ || reference.compact_preparation_->exhausted ||
      reference.compact_preparation_->candidate_index >=
      reference.compact_preparation_->candidates.size())
    {
      return;
    }
    reference.compact_preparation_->generation = generation;
    reference.pending_generation_ = generation;
    reference.active_scene_revision_ = reference.compact_preparation_->scene_revision;
  }

  static void cancelPlannerForTest(MotionReference & reference)
  {
    reference.planner_->cancel();
  }

  static bool installRestFollowerPathForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const std::vector<Eigen::VectorXd> & path, const double time_sec)
  {
    Eigen::VectorXd position;
    state.copyJointGroupPositions(reference.group_, position);
    reference.rest_mode_ = MotionReference::RestMode::kRest;
    reference.rest_completed_ = false;
    reference.rest_settling_ = false;
    return reference.follower_.setPath(path, position, time_sec);
  }

  static double assemblyRadiusForTest(
    const MotionReference & reference, const moveit::core::RobotState & state)
  {
    return reference.compact_geometry_validator_->maximumAssemblyRadius(state);
  }

  static void beginPreparation(
    MotionReference & reference, const moveit::core::RobotState & state,
    const double time_sec, const planning_scene::PlanningScene & scene,
    const std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication)
  {
    reference.requestCompactManeuver(
      state, time_sec, scene, scene_revision, "test_preparation",
      actual_feedback, command_queue_state, active_publication, latest_publication);
  }


  /// Exercise preparation admission after a synthetic qualified pair resource
  /// assessment; public update gate behavior has separate real-FK tests.
  static MotionReferenceResult handleQualifiedPreparationForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const msg::TrackingTarget & target, const planning_scene::PlanningScene & scene,
    const double time_sec, const std::uint64_t scene_revision,
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const bool preserve_existing_roll_guard = false)
  {
    const bool roll_guard_was_armed = reference.upright_roll_guard_armed_;
    qualifyPairYawForIsolatedAdmissionTest(reference);
    reference.upright_roll_guard_armed_ = preserve_existing_roll_guard && roll_guard_was_armed;
    reference.diagnostics_.upright_roll_guard_armed = reference.upright_roll_guard_armed_;
    Eigen::VectorXd position;
    state.copyJointGroupPositions(reference.group_, position);
    MotionReferenceResult result;
    static_cast<void>(reference.handleCompactPreparation(
        state, target, TargetMotionEstimate{}, scene, time_sec, scene_revision,
        position, true, actual_feedback, time_sec, command_queue_state,
        nullptr, nullptr, result));
    return result;
  }

  static MotionReferenceResult handlePreparationWithAssessmentForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const msg::TrackingTarget & target, const planning_scene::PlanningScene & scene,
    const double time_sec, const std::uint64_t scene_revision,
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const PairYawResourceAssessment & assessment)
  {
    reference.pair_yaw_assessment_ = assessment;
    Eigen::VectorXd position;
    state.copyJointGroupPositions(reference.group_, position);
    MotionReferenceResult result;
    static_cast<void>(reference.handleCompactPreparation(
        state, target, TargetMotionEstimate{}, scene, time_sec, scene_revision,
        position, true, actual_feedback, time_sec, command_queue_state,
        nullptr, nullptr, result));
    return result;
  }

  static bool armFaceUprightRollGuardFromFeedbackForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const double time_sec, const JointMotionState & actual_feedback,
    const JointMotionState & command_queue_state)
  {
    return reference.maybeArmFaceUprightRollGuard(
      state, time_sec, actual_feedback, time_sec, command_queue_state, nullptr, nullptr);
  }

  /// Isolated planner/admission fixtures begin after the production pair gate.
  /// Real FK and public update gating are covered by dedicated tests below;
  /// this injected certificate keeps these cases focused on path admission.
  static void qualifyPairYawForIsolatedAdmissionTest(MotionReference & reference)
  {
    PairYawResourceInput input;
    input.target_bearing_rate_radps = 0.4;
    input.target_rate_deadband_radps = 0.01;
    input.j1 = PairYawJointResourceInput{
      -1.0, 1.9, 1.9, -2.0, 2.0, 0.10, BaseJointDirection::kNone};
    input.j4 = PairYawJointResourceInput{
      -1.0, 1.9, 1.9, -2.0, 2.0, 0.10, BaseJointDirection::kNone};
    reference.pair_yaw_assessment_ = assessPairYawResource(input);
    auto & assessment = reference.pair_yaw_assessment_;
    assessment.probe_valid = true;
    assessment.actual_probe_solved = true;
    assessment.relaxed_probe_solved = true;
    assessment.actual_available_yaw_rad = 0.0;
    assessment.pair_relaxed_available_yaw_rad = kPairYawProbeDisplacementRad;
    assessment.actual_roll_residual_rad = 0.0;
    assessment.relaxed_roll_residual_rad = 0.0;
    assessment.actual_elevation_residual_rad = 0.0;
    assessment.relaxed_elevation_residual_rad = 0.0;
    assessment.pair_bound_cause = pairYawProbeShowsPairBoundCause(assessment);
    assessment.exhausted = assessment.pair_bound_cause;
    reference.upright_roll_guard_armed_ = true;
    static_cast<void>(reference.face_policy_.update({true, false}));
  }

  static void removeCompactProfileForTest(MotionReference & reference)
  {
    reference.compact_goal_.profile_id.clear();
    reference.compact_geometry_validator_.reset();
    reference.default_compact_goals_.clear();
  }

  [[nodiscard]] static BackgroundPathResult resultForCurrentCandidate(
    const MotionReference & reference, const bool success,
    const double terminal_joint6_offset_rad = 0.0)
  {
    BackgroundPathResult result;
    if (!reference.compact_preparation_ ||
      reference.compact_preparation_->candidate_index >=
      reference.compact_preparation_->candidates.size())
    {
      return result;
    }
    const auto & preparation = *reference.compact_preparation_;
    const auto & candidate = preparation.candidates[preparation.candidate_index];
    result.generation = preparation.generation;
    result.scene_revision = preparation.scene_revision;
    result.start_positions = preparation.submitted_start_positions;
    result.goal_positions = candidate.bounded_positions;
    result.path = {result.start_positions, result.goal_positions};
    result.path.back()[5] += terminal_joint6_offset_rad;
    result.success = success;
    result.error_code = success ? 0 : -1;
    result.message = success ? "test worker-certified path" : "injected preparation failure";
    result.profile_id = candidate.profile_id;
    if (success) {
      PathValidationMetrics metrics;
      // The handler contract receives worker-certified metrics. Lifecycle-only
      // fixtures model that certificate; they do not claim the two-point path
      // above was collision-checked by MoveIt.
      metrics.maximum_assembly_radius_m =
        candidate.path_requirements.maximum_assembly_radius_m;
      metrics.maximum_roll_error_rad = candidate.path_requirements.maximum_roll_error_rad;
      metrics.maximum_abs_screen_pitch_rad =
        candidate.path_requirements.maximum_abs_screen_pitch_rad.value_or(0.0);
      result.validation_metrics = metrics;
    }
    return result;
  }

  static void consume(
    MotionReference & reference, const BackgroundPathResult & plan,
    const moveit::core::RobotState & state, const double time_sec,
    const planning_scene::PlanningScene & scene, const std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication)
  {
    reference.consumeCompactPreparationResult(
      plan, state, time_sec, scene, scene_revision, actual_feedback,
      actual_feedback_time_sec, command_queue_state, active_publication,
      latest_publication);
  }

  static CollisionSegmentValidationResult validatePose(
    const MotionReference & reference, const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & state, const Eigen::VectorXd & position)
  {
    const JointMotionState stationary{
      position, Eigen::VectorXd::Zero(position.size()),
      Eigen::VectorXd::Zero(position.size())};
    return reference.collision_validator_.validatePose(
      scene, state, *reference.group_, stationary);
  }

  [[nodiscard]] static PairYawResourceAssessment evaluatePairYawResource(
    const MotionReference & reference, const moveit::core::RobotState & state,
    const msg::TrackingTarget & target)
  {
    Eigen::VectorXd position;
    Eigen::VectorXd velocity;
    Eigen::VectorXd acceleration;
    state.copyJointGroupPositions(reference.group_, position);
    state.copyJointGroupVelocities(reference.group_, velocity);
    state.copyJointGroupAccelerations(reference.group_, acceleration);
    const JointMotionState feedback{position, velocity, acceleration};
    return reference.evaluatePairYawResource(
      state, target, TargetMotionEstimate{}, feedback, feedback);
  }

  [[nodiscard]] static std::optional<Eigen::Vector2d> pairYawProbeEdgeDeltaInterval(
    const double initial_edge_sine)
  {
    return MotionReference::pairYawProbeEdgeDeltaInterval(initial_edge_sine);
  }

  static void holdBaseJointForTest(
    MotionReference & reference, const double position,
    const BaseJointDirection direction)
  {
    BaseJointHandoffInput input;
    input.working_edge_reached = true;
    input.actual_base_position_rad = position;
    input.predicted_hold_position_rad = position;
    input.edge_direction = direction;
    input.tracking_yaw_direction = direction;
    static_cast<void>(reference.base_handoff_policy_.update(input));
  }

  [[nodiscard]] static bool hasAvailablePairYawResource(
    const PairYawResourceAssessment & assessment)
  {
    return MotionReference::hasAvailablePairYawResource(assessment);
  }

  static JointPathUpdate updateFollower(
    MotionReference & reference, const Eigen::VectorXd & position, const double time_sec)
  {
    return reference.follower_.update(position, time_sec);
  }

  [[nodiscard]] static bool followerActive(const MotionReference & reference)
  {
    return reference.follower_.active();
  }
};

class CollisionAwareServoComponentTestPeer final
{
public:
  static CollisionConstraintResult buildCollisionConstraints(
    CollisionAwareServoComponent & component,
    const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & robot_state,
    const JointMotionState & motion_state)
  {
    return component.collision_constraint_builder_->build(
      scene, robot_state, *component.joint_model_group_, motion_state,
      component.motion_limits_, component.parameters_.control_period_sec);
  }

  static QpResult solveTask(
    CollisionAwareServoComponent & component, const HierarchicalVelocityTask & task,
    const JointMotionState & motion_state,
    const std::vector<LinearVelocityConstraint> & constraints)
  {
    return component.velocity_qp_->solve(
      task, motion_state, component.motion_limits_, constraints);
  }

  static void configure(
    CollisionAwareServoComponent & component,
    const planning_scene_monitor::PlanningSceneMonitorPtr & monitor,
    const moveit::core::JointModelGroup * group,
    const std::vector<std::string> & joint_names,
    const JointMotionLimits & motion_limits,
    const CollisionConstraintConfig & collision_config,
    const std::vector<PositionBoundSet> & bounds)
  {
    component.commands_ = {};
    component.last_rearm_hardware_read_sequence_.reset();
    component.commands_.last_rearm_sample_time.reset();
    component.require_hardware_io_status_ = true;
    component.hardware_io_status_policy_ = std::make_unique<
      hardware::HardwareIoStatusPolicy>(2'000'000'000LL, 100'000'000LL, 2'000'000'000LL);
    component.planning_scene_monitor_ = monitor;
    component.joint_model_group_ = group;
    component.parameters_.joint_names = joint_names;
    component.parameters_.state_feedback_timeout_sec = 0.5;
    component.parameters_.following_position_tolerance_rad = 0.05;
    component.parameters_.following_velocity_tolerance_rad_s = 0.05;
    component.parameters_.collision_tracking_error_bound_rad = 0.05;
    component.parameters_.control_period_sec = 0.01;
    component.parameters_.trajectory_controller_period_sec = 0.01;
    component.motion_limits_ = motion_limits;
    component.physical_joint_velocity_limits_ = motion_limits.max_velocity;
    component.position_bound_sets_ = bounds;
    component.hardware_execution_error_rad_.assign(bounds.size(), 0.5);
    component.hardware_factory_position_limits_rad_.clear();
    component.hardware_position_limits_rad_.clear();
    for (const auto & bound : bounds) {
      component.hardware_factory_position_limits_rad_.push_back(bound.factory.lower_rad);
      component.hardware_factory_position_limits_rad_.push_back(bound.factory.upper_rad);
      component.hardware_position_limits_rad_.push_back(bound.hardware.lower_rad);
      component.hardware_position_limits_rad_.push_back(bound.hardware.upper_rad);
    }
    component.collision_constraint_builder_ =
      std::make_unique<CollisionConstraintBuilder>(collision_config);
    EmergencyBrakeTailConfig tail_config;
    tail_config.period_sec = 0.01;
    tail_config.terminal_hold_points = 4U;
    component.emergency_brake_tail_generator_ =
      std::make_unique<EmergencyBrakeTailGenerator>(tail_config);
  }

  static void installMotionReference(
    CollisionAwareServoComponent & component,
    std::unique_ptr<MotionReference> reference)
  {
    component.motion_reference_ = std::move(reference);
  }

  static void installStatusPublisher(CollisionAwareServoComponent & component)
  {
    component.status_publisher_ = component.create_publisher<moveit_msgs::msg::ServoStatus>(
      "/test/servo_status", rclcpp::QoS(1));
  }

  static void setControllerState(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const Eigen::VectorXd * feedback_positions = nullptr,
    const Eigen::VectorXd * command_positions = nullptr)
  {
    control_msgs::msg::JointTrajectoryControllerState state;
    const auto stamp = component.now();
    const auto nanoseconds = stamp.nanoseconds();
    state.header.stamp.sec = static_cast<std::int32_t>(nanoseconds / 1000000000LL);
    state.header.stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % 1000000000LL);
    state.joint_names = component.parameters_.joint_names;
    const std::vector<double> values(positions.data(), positions.data() + positions.size());
    const std::vector<double> command_values = command_positions == nullptr ? values :
      std::vector<double>(command_positions->data(),
      command_positions->data() + command_positions->size());
    state.reference.positions = command_values;
    state.reference.velocities.assign(values.size(), 0.0);
    state.feedback.positions = feedback_positions == nullptr ? values :
      std::vector<double>(feedback_positions->data(),
      feedback_positions->data() + feedback_positions->size());
    state.feedback.velocities.assign(values.size(), 0.0);
    state.output.positions = command_values;
    state.speed_scaling_factor = 1.0;
    std::lock_guard<std::mutex> lock(component.controller_state_mutex_);
    component.latest_controller_state_ = std::move(state);
  }

  static void publishHardwareSample(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const std::uint64_t epoch, const std::uint64_t sequence,
    const Eigen::VectorXd * controller_feedback = nullptr,
    const Eigen::VectorXd * controller_command = nullptr,
    const std::optional<std::uint64_t> read_epoch = std::nullopt)
  {
    msg::HardwareIoStatus status;
    status.source_kind = msg::HardwareIoStatus::SOURCE_HARDWARE;
    status.runtime_actions_enabled = true;
    status.hardware_active = true;
    status.epoch = epoch;
    status.read_epoch = read_epoch.value_or(epoch);
    status.read_attempt_sequence = sequence;
    status.read_sequence = sequence;
    status.read_ok = true;
    status.robot_msg_sequence = sequence;
    status.robot_mode = 1U;
    status.robot_state = 0U;
    status.robot_mt_able = 0x3FU;
    status.robot_mt_brake = 0x3FU;
    const auto stamp = component.now();
    const auto nanoseconds = stamp.nanoseconds();
    const auto set_stamp = [nanoseconds](builtin_interfaces::msg::Time & output) {
        output.sec = static_cast<std::int32_t>(nanoseconds / 1000000000LL);
        output.nanosec = static_cast<std::uint32_t>(nanoseconds % 1000000000LL);
      };
    set_stamp(status.header.stamp);
    set_stamp(status.read_stamp);
    set_stamp(status.robot_msg_stamp);
    for (std::size_t index = 0; index < 6U; ++index) {
      status.q[index] = static_cast<double>(static_cast<float>(
          positions[static_cast<Eigen::Index>(index)]));
      status.dq[index] = 0.0;
    }
    setControllerState(component, positions, controller_feedback, controller_command);
    component.receive_hardware_io_status(status);
    component.cycle_hardware_io_status_.reset();
    component.cycle_hardware_io_status_ = component.current_hardware_io_status();
  }

  static bool hardwareActiveSeen(const CollisionAwareServoComponent & component)
  {
    return component.hardware_active_seen_.load(std::memory_order_acquire);
  }

  static void clearHardwareInhibitPending(CollisionAwareServoComponent & component)
  {
    component.hardware_inhibit_pending_.store(false, std::memory_order_release);
  }

  static bool retainStationaryPublicationAndRecover(
    CollisionAwareServoComponent & component, const std::string & reason)
  {
    component.commands_.time_grid = std::make_unique<control::ControlTimeGrid>(10'000'000LL);
    const auto start = component.now();
    const auto publication_start = start + rclcpp::Duration::from_nanoseconds(100'000'000LL);
    const Eigen::VectorXd positions = Eigen::VectorXd::Zero(6);
    std::deque<moveit_servo::KinematicState> queue;
    for (std::int64_t step = 0; step < 4; ++step) {
      moveit_servo::KinematicState state;
      state.joint_names = component.parameters_.joint_names;
      state.positions = positions;
      state.velocities = Eigen::VectorXd::Zero(6);
      state.accelerations = Eigen::VectorXd::Zero(6);
      state.time_stamp = publication_start +
        rclcpp::Duration::from_nanoseconds(step * 10'000'000LL);
      queue.push_back(std::move(state));
    }
    const auto stationary_time = publication_start +
      rclcpp::Duration::from_nanoseconds(20'000'000LL);
    const auto publication_id = component.commands_.publications.append(
      std::move(queue), stationary_time);
    if (!publication_id) {
      return false;
    }
    component.commands_.active_publication_id = publication_id;
    component.commands_.epoch_active = true;
    component.commands_.epoch_has_published = true;
    component.commands_.startup_warmup_complete = true;
    component.handle_recoverable_control_gap(reason);
    return component.commands_.hasRetainedPublication() &&
           component.commands_.published_tail_rearm_pending;
  }

  static bool currentHardwareStatusValid(CollisionAwareServoComponent & component)
  {
    const auto status = component.current_hardware_io_status();
    return status && status->valid;
  }

  static void controlTick(CollisionAwareServoComponent & component)
  {
    if (!component.commands_.time_grid) {
      component.commands_.time_grid = std::make_unique<control::ControlTimeGrid>(10'000'000LL);
    }
    if (!component.trajectory_publisher_) {
      component.trajectory_publisher_ = component.create_publisher<
        trajectory_msgs::msg::JointTrajectory>("/test/command", rclcpp::QoS(1));
    }
    component.control_tick();
  }

  static void setContinuationTarget(
    CollisionAwareServoComponent & component, const msg::TrackingTarget & target)
  {
    std::lock_guard<std::mutex> lock(component.target_mutex_);
    component.latest_pose_ = target;
  }

  static void setPlanningSceneRevision(
    CollisionAwareServoComponent & component, const std::uint64_t revision)
  {
    component.planning_scene_collision_revision_.store(revision, std::memory_order_release);
  }

  static bool installPublishedBrakeSuffix(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const rclcpp::Time & publication_start, const std::uint64_t scene_revision,
    const std::size_t current_point_index = 0U,
    const std::size_t transmitted_points = 4U)
  {
    component.commands_ = {};
    component.commands_.time_grid = std::make_unique<control::ControlTimeGrid>(10'000'000LL);
    component.commands_.max_queue_points = 8U;
    if (transmitted_points < 3U || current_point_index >= transmitted_points ||
      !component.commands_.time_grid->commandStamp(
        publication_start.nanoseconds() +
        static_cast<std::int64_t>(current_point_index) * 10'000'000LL,
        0, 0))
    {
      return false;
    }
    std::deque<moveit_servo::KinematicState> published_queue;
    for (std::size_t index = 0; index < transmitted_points; ++index) {
      moveit_servo::KinematicState point;
      point.joint_names = component.parameters_.joint_names;
      point.positions = positions;
      point.velocities = Eigen::VectorXd::Zero(positions.size());
      point.accelerations = Eigen::VectorXd::Zero(positions.size());
      point.time_stamp = publication_start + rclcpp::Duration::from_nanoseconds(
        static_cast<std::int64_t>(index) * 10'000'000LL);
      published_queue.push_back(std::move(point));
    }
    const auto stationary_time = published_queue[2U].time_stamp;
    const auto publication_id = component.commands_.publications.append(
      published_queue, stationary_time);
    if (!publication_id) {
      return false;
    }
    component.commands_.queue.push_back(published_queue[current_point_index]);
    component.commands_.active_publication_id = publication_id;
    component.commands_.epoch_active = true;
    component.commands_.epoch_has_published = true;
    component.commands_.startup_warmup_complete = true;
    component.servo_parameters_.max_expected_latency = 0.01;
    component.last_validated_scene_revision_ = scene_revision;
    component.planning_scene_collision_revision_.store(
      scene_revision, std::memory_order_release);
    component.accept_pose_commands_.store(true, std::memory_order_release);
    component.trajectory_controller_active_.store(true, std::memory_order_release);
    return true;
  }

  static PublishedBrakeContinuationStatus tryContinuePublishedBrake(
    CollisionAwareServoComponent & component, const rclcpp::Time & current_time,
    const msg::TrackingTarget & target, const std::uint64_t scene_revision)
  {
    return component.tryContinuePublishedBrake(
      current_time, target, true, scene_revision);
  }

  static bool restFollowerActive(const CollisionAwareServoComponent & component)
  {
    return component.motion_reference_ &&
           MotionReferenceTestPeer::followerActive(*component.motion_reference_);
  }

  static std::size_t continuationQueueSize(const CollisionAwareServoComponent & component)
  {
    return component.commands_.queue.size();
  }

  static rclcpp::Time nextPublishedBrakeStamp(
    const CollisionAwareServoComponent & component, const std::size_t index)
  {
    const auto * publication = component.commands_.activePublication();
    if (publication == nullptr || index >= publication->execution_queue.size()) {
      return rclcpp::Time(0, 0, RCL_ROS_TIME);
    }
    return publication->execution_queue[index].time_stamp;
  }

  static rclcpp::Time continuationQueueTailStamp(
    const CollisionAwareServoComponent & component)
  {
    return component.commands_.queue.empty() ?
           rclcpp::Time(0, 0, RCL_ROS_TIME) : component.commands_.queue.back().time_stamp;
  }

  static RearmReadiness checkRearm(
    CollisionAwareServoComponent & component, std::string & reason)
  {
    return component.safe_rearm_ready(component.now(), reason);
  }

  static bool prepareClockRewindWithRetainedTail(
    CollisionAwareServoComponent & component)
  {
    component.commands_.time_grid = std::make_unique<control::ControlTimeGrid>(10'000'000LL);
    const auto current_time = component.now();
    const auto now_ns = current_time.nanoseconds();
    (void)component.commands_.time_grid->observe(now_ns + 100'000'000LL);
    const auto rewind = component.commands_.time_grid->observe(now_ns);
    component.commands_.epoch_has_published = false;
    component.commands_.published_tail_rearm_pending = true;
    const bool keep_running = component.prepare_control_cycle(
      current_time, std::chrono::steady_clock::now(), rewind);
    return !keep_running && component.hardware_inhibit_pending_.load(
      std::memory_order_acquire);
  }

  static void prepareExpiredRearmDeadline(CollisionAwareServoComponent & component)
  {
    component.commands_.time_grid = std::make_unique<control::ControlTimeGrid>(10'000'000LL);
    component.commands_.rearm_pending = true;
    component.commands_.startRearmDeadline(
      component.now(), std::chrono::steady_clock::now() - std::chrono::seconds(3));
  }

  static bool rearmDeadlineExpired(const CollisionAwareServoComponent & component)
  {
    return component.commands_.rearmDeadlineExpired(
      component.now() + rclcpp::Duration::from_seconds(3.0),
      std::chrono::steady_clock::now());
  }

  static std::optional<rclcpp::Time> rearmDeadline(
    const CollisionAwareServoComponent & component)
  {
    return component.commands_.rearm_deadline;
  }

  static void setStableRearmSamples(
    CollisionAwareServoComponent & component, const std::size_t samples)
  {
    component.commands_.stable_rearm_samples = samples;
  }

  static std::size_t stableRearmSamples(const CollisionAwareServoComponent & component)
  {
    return component.commands_.stable_rearm_samples;
  }

  static std::size_t rearmTimeoutCount(const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.rearm_timeout_count;
  }

  static std::string latchedHaltReason(const CollisionAwareServoComponent & component)
  {
    return component.latched_halt_reason_;
  }

  static bool safetyWaitLatched(const CollisionAwareServoComponent & component)
  {
    return component.controller_mode_ == ControllerMode::kWaitingForSafeState &&
           component.halt_latched_.load(std::memory_order_acquire);
  }

  static bool validateHardwareIoForControlTick(CollisionAwareServoComponent & component)
  {
    return component.validateHardwareIoForControlTick();
  }

  static bool hardwareInhibitPending(const CollisionAwareServoComponent & component)
  {
    return component.hardware_inhibit_pending_.load(std::memory_order_acquire);
  }

  static void handleRearmSettleTimeout(
    CollisionAwareServoComponent & component, std::string message)
  {
    component.handle_rearm_settle_timeout(std::move(message));
  }

  static std::string hardwareStatusFailure(CollisionAwareServoComponent & component)
  {
    const auto status = component.current_hardware_io_status();
    if (!status || status->valid) {
      return status ? "valid" : "missing";
    }
    const std::size_t joint = status->failure_joint_index;
    const auto & bound = component.position_bound_sets_.at(joint).working;
    return std::string(hardware::hardwareIoStatusFailureName(status->failure)) +
           " joint=" + std::to_string(joint) + " q=" +
           std::to_string(status->sample.position[joint]) + " working=[" +
           std::to_string(bound.lower_rad) + "," + std::to_string(bound.upper_rad) + "]";
  }

  static std::string rearmSequenceState(const CollisionAwareServoComponent & component)
  {
    const auto status = component.current_hardware_io_status();
    return std::string("current_read=") +
           (status ? std::to_string(status->sample.read_sequence) : "missing") +
           " last_rearm=" + (component.last_rearm_hardware_read_sequence_ ?
           std::to_string(*component.last_rearm_hardware_read_sequence_) : "unset");
  }

  static std::optional<moveit_servo::KinematicState> predict(
    CollisionAwareServoComponent & component, std::string & reason,
    bool & requires_latched_halt)
  {
    return component.predicted_or_measured_state(
      component.now(), reason, requires_latched_halt, nullptr, nullptr);
  }

  static std::optional<Eigen::VectorXd> provenHold(
    const CollisionAwareServoComponent & component)
  {
    return component.proven_q_hold_;
  }

  static void setHardwareExecutionError(
    CollisionAwareServoComponent & component, const double error_rad)
  {
    component.hardware_execution_error_rad_.assign(
      component.parameters_.joint_names.size(), error_rad);
  }

  static bool setSyntheticExecutionBudget(
    CollisionAwareServoComponent & component, const double total_error_rad,
    const double reaction_threshold_rad, const double post_threshold_growth_rad)
  {
    ExecutionErrorBudget budget;
    budget.source = ExecutionBudgetSource::kSyntheticTest;
    budget.identity = {
      "unit_test", "xarm6", 1303, "test_firmware", "test_position_servoj",
      "test_conditions", "synthetic-only", true, false};
    budget.total_error_rad = total_error_rad;
    budget.reaction_threshold_rad = reaction_threshold_rad;
    budget.post_threshold_growth_rad = post_threshold_growth_rad;
    const ExecutionBudgetExpectedIdentity expected{
      "unit_test", "xarm6", 1303, "test_firmware", "test_position_servoj",
      "test_conditions", false};
    const auto validation = validateExecutionErrorBudget(budget, expected, true);
    if (!validation.valid || validation.physical_profile) {
      return false;
    }
    component.hardware_execution_budget_ = budget;
    component.hardware_execution_error_rad_.assign(
      component.parameters_.joint_names.size(), total_error_rad);
    component.hardware_reaction_threshold_rad_ = reaction_threshold_rad;
    component.hardware_post_threshold_growth_rad_ = post_threshold_growth_rad;
    return true;
  }

  static void clearExecutionBudgetForTest(CollisionAwareServoComponent & component)
  {
    component.hardware_execution_budget_.reset();
    component.hardware_execution_error_rad_.assign(
      component.parameters_.joint_names.size(), 0.0);
    component.parameters_.collision_tracking_error_bound_rad = 0.0;
  }

  static std::optional<BootstrapCommand> stationaryBootstrap(
    CollisionAwareServoComponent & component,
    const moveit_servo::KinematicState & measured_state,
    const moveit::core::RobotState & robot_state,
    const planning_scene::PlanningScene & scene, std::string & reason)
  {
    BootstrapFailureKind failure_kind = BootstrapFailureKind::kRetryableState;
    return component.make_stationary_bootstrap(
      measured_state, robot_state, scene, reason, failure_kind);
  }

  static std::optional<ValidatedCommand> validateCandidate(
    CollisionAwareServoComponent & component,
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const JointMotionState & current_motion_state,
    const Eigen::VectorXd & joint_velocity,
    const planning_scene::PlanningScene & scene, std::string & reason)
  {
    CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
    return component.validated_candidate(
      robot_state, current_state, current_motion_state, joint_velocity, scene, true,
      CommandAttempt::kTracking, reason, CollisionConstraintDiagnostics{}, rejection_kind);
  }

  static std::optional<ValidatedCommand> validateCandidateWithKind(
    CollisionAwareServoComponent & component,
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const JointMotionState & current_motion_state,
    const Eigen::VectorXd & joint_velocity,
    const planning_scene::PlanningScene & scene,
    std::string & reason,
    CandidateRejectionKind & rejection_kind)
  {
    return component.validated_candidate(
      robot_state, current_state, current_motion_state, joint_velocity, scene, true,
      CommandAttempt::kTracking, reason, CollisionConstraintDiagnostics{}, rejection_kind);
  }

  static bool tryReferenceGuardBlend(
    CollisionAwareServoComponent & component,
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const JointMotionState & motion_state,
    const planning_scene::PlanningScene & scene,
    const QpResult & tracking_result,
    const QpResult & brake_result,
    const Eigen::MatrixXd & primary_matrix,
    const std::vector<LinearVelocityConstraint> & constraints,
    std::optional<ValidatedCommand> & accepted_command,
    QpResult & accepted_qp_result)
  {
    return component.try_reference_guard_blend(
      robot_state, current_state, motion_state, scene, true, tracking_result,
      brake_result, primary_matrix, constraints, CollisionConstraintDiagnostics{},
      accepted_command, accepted_qp_result);
  }

  static void installVelocityQp(
    CollisionAwareServoComponent & component,
    const HierarchicalVelocityQpConfig & config = {})
  {
    component.velocity_qp_ = std::make_unique<HierarchicalVelocityQp>(config);
  }

  static std::optional<moveit_servo::KinematicState> nextState(
    CollisionAwareServoComponent & component,
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const Eigen::VectorXd & joint_velocity, std::string & reason)
  {
    return component.make_next_state(robot_state, current_state, joint_velocity, reason);
  }

  static void resetStatusEpochForNewSession(CollisionAwareServoComponent & component)
  {
    component.hardware_io_status_policy_->reset();
  }

  static void seedProvenHoldForEpochTest(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & hold,
    const std::uint64_t epoch)
  {
    component.proven_q_hold_ = hold;
    component.proven_q_hold_epoch_ = epoch;
  }

  static bool hasPublishedCommand(const CollisionAwareServoComponent & component)
  {
    return component.commands_.epoch_has_published;
  }
};

namespace
{

using namespace std::chrono_literals;

constexpr std::uint64_t kSceneRevision = 11;
constexpr double kCompactRollCapTestRad{5.1 * 3.14159265358979323846 / 180.0};

/// Synthetic worker metadata for MotionReference admission tests only. These
/// fixtures exercise certificate mapping/selection; they are not planner or
/// collision-validation evidence. BackgroundPathPlanner tests cover real FK facts.
void installSyntheticCompactCertificate(
  BackgroundPathResult & result,
  const std::function<std::array<double, 3>(std::size_t, double)> & metrics_at)
{
  ASSERT_TRUE(result.validation_metrics);
  ASSERT_GE(result.path.size(), 2U);
  result.compact_path_samples.clear();
  result.compact_goal_metrics = PathValidationMetrics{};
  result.compact_goal_metrics->maximum_assembly_radius_m = 0.39;
  result.compact_goal_metrics->maximum_roll_error_rad = 0.01;
  result.compact_goal_metrics->maximum_abs_screen_pitch_rad = 0.20;
  result.compact_goal_metrics->checked_samples = 1U;

  for (std::size_t segment = 0U; segment + 1U < result.path.size(); ++segment) {
    const auto & from = result.path[segment];
    const auto & to = result.path[segment + 1U];
    const auto subdivisions = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(
        (to - from).lpNorm<Eigen::Infinity>() / 0.01)));
    const std::size_t first_step = segment == 0U ? 0U : 1U;
    for (std::size_t step = first_step; step <= subdivisions; ++step) {
      const double fraction = static_cast<double>(step) /
        static_cast<double>(subdivisions);
      const auto values = metrics_at(segment, fraction);
      CompactPathSampleCertificate sample;
      sample.positions = from + fraction * (to - from);
      sample.original_segment_index = segment;
      sample.fraction = fraction;
      sample.maximum_assembly_radius_m = values[0];
      sample.maximum_roll_error_rad = values[1];
      sample.maximum_abs_screen_pitch_rad = values[2];
      result.compact_path_samples.push_back(std::move(sample));
    }
  }

  double suffix_radius = result.compact_goal_metrics->maximum_assembly_radius_m;
  double suffix_roll = result.compact_goal_metrics->maximum_roll_error_rad;
  double suffix_pitch = result.compact_goal_metrics->maximum_abs_screen_pitch_rad;
  double whole_radius = suffix_radius;
  double whole_roll = suffix_roll;
  double whole_pitch = suffix_pitch;
  for (auto sample = result.compact_path_samples.rbegin();
    sample != result.compact_path_samples.rend(); ++sample)
  {
    suffix_radius = std::max(suffix_radius, sample->maximum_assembly_radius_m);
    suffix_roll = std::max(suffix_roll, sample->maximum_roll_error_rad);
    suffix_pitch = std::max(suffix_pitch, sample->maximum_abs_screen_pitch_rad);
    sample->suffix_maximum_assembly_radius_m = suffix_radius;
    sample->suffix_maximum_roll_error_rad = suffix_roll;
    sample->suffix_maximum_abs_screen_pitch_rad = suffix_pitch;
    whole_radius = std::max(whole_radius, sample->maximum_assembly_radius_m);
    whole_roll = std::max(whole_roll, sample->maximum_roll_error_rad);
    whole_pitch = std::max(whole_pitch, sample->maximum_abs_screen_pitch_rad);
  }
  result.validation_metrics->maximum_assembly_radius_m = whole_radius;
  result.validation_metrics->maximum_roll_error_rad = whole_roll;
  result.validation_metrics->maximum_abs_screen_pitch_rad = whole_pitch;
  result.validation_metrics->checked_samples = result.compact_path_samples.size();
}

std::array<double, 3> safeSyntheticCompactMetrics()
{
  return {0.39, 0.01, 0.20};
}

std::optional<std::size_t> triggeredAttachmentSampleIndex(
  const std::vector<CompactEvent> & events)
{
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  if (triggered == events.end()) {
    return std::nullopt;
  }
  constexpr std::string_view field{"\"sample_index\":"};
  const auto start = triggered->details_json.find(field);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  const auto value_start = start + field.size();
  if (triggered->details_json.compare(value_start, 4U, "null") == 0) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(
      triggered->details_json.substr(value_start)));
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

TEST(CompactPathReferenceSpeed, ScalingChangesCompactPathVelocityReference)
{
  const Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd distant_waypoint = Eigen::VectorXd::Constant(6, 1.0);
  for (const auto & [scale, expected_speed] : {
        std::pair<double, double>{0.5, 0.35}, {1.0, 0.70}, {1.5, 1.05}})
  {
    const auto settings = scaleCompactPathReference(6.0, 0.70, scale);
    const auto task = makeJointPathVelocityTask(
      current, distant_waypoint, settings.gain, settings.maximum_velocity_radps);
    ASSERT_TRUE(task);
    EXPECT_NEAR(
      task->primary_reference.lpNorm<Eigen::Infinity>(), expected_speed, 1.0e-12);
  }

  const Eigen::VectorXd nearby_waypoint = Eigen::VectorXd::Constant(6, 0.01);
  double previous_reference_norm = 0.0;
  for (const auto & [scale, expected_norm] : {
        std::pair<double, double>{0.5, 0.5}, {1.0, 1.0}, {1.5, 1.5}})
  {
    const auto settings = scaleCompactPathReference(6.0, 0.70, scale);
    const auto task = makeJointPathVelocityTask(
      current, nearby_waypoint, settings.gain, settings.maximum_velocity_radps);
    ASSERT_TRUE(task);
    const double reference_norm = task->primary_reference.norm();
    EXPECT_NEAR(reference_norm, expected_norm * 0.06 * std::sqrt(6.0), 1.0e-12);
    EXPECT_GT(reference_norm, previous_reference_norm);
    previous_reference_norm = reference_norm;
  }
}

TEST(CompactPathReferenceSpeed, ScaleOneMatchesExistingPathTaskGeometry)
{
  struct Case
  {
    Eigen::VectorXd current;
    Eigen::VectorXd projected;
    Eigen::VectorXd lookahead;
    double remaining;
    bool crosses_corner;
  };
  const std::vector<Case> cases{
    {Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{0.0, 0.0},
      Eigen::Vector2d{0.1, 0.0}, 0.4, false},
    {Eigen::Vector2d{0.0, 0.02}, Eigen::Vector2d{0.0, 0.0},
      Eigen::Vector2d{0.1, 0.0}, 0.4, false},
    {Eigen::Vector2d{0.01, 0.03}, Eigen::Vector2d{0.01, 0.0},
      Eigen::Vector2d{0.08, 0.03}, 0.4, true},
    {Eigen::Vector2d{0.0, 0.0}, Eigen::Vector2d{0.0, 0.0},
      Eigen::Vector2d{0.05, 0.0}, 0.05, false}};

  for (const auto & test_case : cases) {
    JointPathUpdate path;
    path.status = JointPathStatus::kTracking;
    path.projected_position = test_case.projected;
    path.reference_position = test_case.lookahead;
    path.remaining_distance_rad = test_case.remaining;
    path.lookahead_crosses_corner = test_case.crosses_corner;
    const auto velocity = makeCompactPathVelocityReference(
      test_case.current, path, 0.10, 1.0);
    ASSERT_TRUE(velocity);

    const auto old_task = makeJointPathVelocityTask(
      test_case.current, test_case.lookahead, 6.0, 0.70);
    const auto new_task = makeJointPathVelocityTask(
      test_case.current, test_case.current + *velocity / 6.0, 6.0, 0.70);
    ASSERT_TRUE(old_task);
    ASSERT_TRUE(new_task);
    EXPECT_TRUE(new_task->primary_reference.isApprox(old_task->primary_reference, 1.0e-12));
  }
}

TEST(CompactPathReferenceSpeed, CruiseScaleDoesNotScaleCrossTrackAndCornersLimitCruise)
{
  JointPathUpdate path;
  path.status = JointPathStatus::kTracking;
  path.projected_position = Eigen::Vector2d{0.0, 0.0};
  path.reference_position = Eigen::Vector2d{0.1, 0.0};
  path.remaining_distance_rad = 1.0;
  const Eigen::Vector2d off_path_position{0.0, 0.02};

  auto velocity = makeCompactPathVelocityReference(off_path_position, path, 0.10, 1.5);
  ASSERT_TRUE(velocity);
  EXPECT_TRUE(velocity->isApprox(Eigen::Vector2d{0.9, -0.12}, 1.0e-12));

  path.lookahead_crosses_corner = true;
  velocity = makeCompactPathVelocityReference(off_path_position, path, 0.10, 1.5);
  ASSERT_TRUE(velocity);
  EXPECT_TRUE(velocity->isApprox(Eigen::Vector2d{0.6, -0.12}, 1.0e-12));
}

TEST(CompactPathReferenceSpeed, TerminalFeedforwardCapIsContinuousAndLimitsEndpointSpeed)
{
  JointPathUpdate path;
  path.status = JointPathStatus::kTracking;
  path.projected_position = Eigen::Vector2d{0.0, 0.0};
  path.reference_position = Eigen::Vector2d{0.05, 0.0};
  path.remaining_distance_rad = 0.05;
  const Eigen::Vector2d current{0.0, 0.0};
  const auto velocity = makeCompactPathVelocityReference(current, path, 0.10, 1.5);
  ASSERT_TRUE(velocity);
  EXPECT_NEAR(velocity->norm(), 6.0 * path.remaining_distance_rad, 1.0e-12);

  path.reference_position = Eigen::Vector2d{0.10, 0.0};
  path.remaining_distance_rad = 0.100001;
  const auto before = makeCompactPathVelocityReference(current, path, 0.10, 1.5);
  path.remaining_distance_rad = 0.099999;
  const auto after = makeCompactPathVelocityReference(current, path, 0.10, 1.5);
  ASSERT_TRUE(before);
  ASSERT_TRUE(after);
  EXPECT_NEAR((*before - *after).norm(), 1.2e-5, 1.0e-10);
}

TEST(CompactPathReferenceSpeed, DirectionPreservingVelocityCapBoundsTheReference)
{
  JointPathUpdate path;
  path.status = JointPathStatus::kTracking;
  path.projected_position = Eigen::Vector2d{0.0, 0.0};
  path.reference_position = Eigen::Vector2d{0.2, 0.0};
  path.remaining_distance_rad = 1.0;
  const Eigen::Vector2d current{0.0, 0.0};
  const auto velocity = makeCompactPathVelocityReference(current, path, 0.10, 1.5);
  ASSERT_TRUE(velocity);
  const auto task = makeJointPathVelocityTask(
    current, current + *velocity / 6.0, 6.0, 0.70 * 1.5);
  ASSERT_TRUE(task);
  EXPECT_NEAR(task->primary_reference.lpNorm<Eigen::Infinity>(), 0.70 * 1.5, 1.0e-12);
  EXPECT_TRUE(task->primary_reference.normalized().isApprox(velocity->normalized(), 1.0e-12));
}

TEST(CompactPathReferenceSpeed, RejectsMalformedFollowerOutput)
{
  JointPathUpdate path;
  path.status = JointPathStatus::kTracking;
  path.projected_position = Eigen::Vector2d{0.0, 0.0};
  path.reference_position = Eigen::Vector2d{0.1, 0.0};
  path.remaining_distance_rad = 0.2;
  const Eigen::Vector2d current{0.0, 0.0};
  const auto valid = [&]() {
      return makeCompactPathVelocityReference(current, path, 0.10, 1.0);
    };
  EXPECT_TRUE(valid());
  path.status = JointPathStatus::kInvalid;
  EXPECT_FALSE(valid());
  path.status = JointPathStatus::kTracking;
  path.projected_position.resize(3);
  EXPECT_FALSE(valid());
  path.projected_position = Eigen::Vector2d{0.0, 0.0};
  path.remaining_distance_rad = -0.1;
  EXPECT_FALSE(valid());
  path.remaining_distance_rad = 0.2;
  EXPECT_FALSE(makeCompactPathVelocityReference(current, path, 0.0, 1.0));
  EXPECT_FALSE(makeCompactPathVelocityReference(
      current, path, 0.10, std::numeric_limits<double>::quiet_NaN()));
  path.reference_position[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(valid());
}

TEST(CompactPathReferenceSpeed, RejectsNonPositiveOrNonFiniteScale)
{
  for (const double scale : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max()})
  {
    EXPECT_THROW((void)scaleCompactPathReference(6.0, 0.70, scale), std::invalid_argument);
  }
}

std::optional<Eigen::VectorXd> goalVectorFromDetails(const std::string & details)
{
  constexpr std::string_view kField{"\"goal_joints_rad\":["};
  const auto start = details.find(kField);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  std::size_t cursor = start + kField.size();
  Eigen::VectorXd result(6);
  for (Eigen::Index index = 0; index < result.size(); ++index) {
    const auto end = details.find_first_of(",]", cursor);
    if (end == std::string::npos) {
      return std::nullopt;
    }
    try {
      result[index] = std::stod(details.substr(cursor, end - cursor));
    } catch (const std::exception &) {
      return std::nullopt;
    }
    cursor = end + 1;
  }
  return result;
}

std::string readModelFile(const std::string & path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot read test robot model: " + path);
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/// The real xArm6 description is expanded by CMake, never by a running test.
/// No controller, sensor, simulator or robot command publisher is started here.
class MotionReferenceTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    std::string urdf = readModelFile(TEST_ROBOT_URDF_FILE);
    const auto closing_robot = urdf.rfind("</robot>");
    ASSERT_NE(closing_robot, std::string::npos);
    urdf.insert(closing_robot,
          R"(
      <link name="test_camera_optical"/>
      <joint name="test_camera_mount" type="fixed">
        <parent link="monitor_control_frame"/><child link="test_camera_optical"/>
        <origin xyz="0 0 0.125" rpy="-1.57079632679 0 -1.57079632679"/>
      </joint>)");
    std::string semantic = readModelFile(TEST_ROBOT_SRDF_FILE);
    if (!searchEnabled()) {
      const auto start = semantic.find("  <group_state name=\"search\"");
      ASSERT_NE(start, std::string::npos);
      const auto end = semantic.find("  </group_state>", start);
      ASSERT_NE(end, std::string::npos);
      semantic.erase(start, end + std::string("  </group_state>\n").size() - start);
    }
    if (!compactProfileEnabled()) {
      for (const auto * profile : {
            "compact_turn_negative_wrist", "compact_turn_positive_wrist"})
      {
        const auto start = semantic.find(
          "  <group_state name=\"" + std::string(profile) + "\"");
        if (start == std::string::npos) {
          continue;
        }
        const auto end = semantic.find("  </group_state>", start);
        ASSERT_NE(end, std::string::npos);
        semantic.erase(start, end + std::string("  </group_state>\n").size() - start);
      }
    }
    options.parameter_overrides({
          rclcpp::Parameter("robot_description", urdf),
          rclcpp::Parameter("robot_description_semantic", semantic),
          rclcpp::Parameter(
        "robot_description_kinematics.xarm6.kinematics_solver",
        "kdl_kinematics_plugin/KDLKinematicsPlugin"),
          rclcpp::Parameter(
        "robot_description_kinematics.xarm6.kinematics_solver_timeout", 0.01),
          rclcpp::Parameter(
        "recovery_planner.planner_configs.RRTConnectkConfigDefault.type", "geometric::RRTConnect"),
          rclcpp::Parameter(
        "recovery_planner.planner_configs.RRTConnectkConfigDefault.range", 0.10),
          rclcpp::Parameter(
        "recovery_planner.xarm6.planner_configs",
        std::vector<std::string>{"RRTConnectkConfigDefault"}),
          rclcpp::Parameter("recovery_planner.xarm6.longest_valid_segment_fraction", 0.00019),
          rclcpp::Parameter("search_enabled", searchEnabled()),
          rclcpp::Parameter("recovery.mode", recoveryMode()),
          rclcpp::Parameter("search_speed_rad_s", searchEnabled() ? 0.30 : -1.0),
          rclcpp::Parameter("compact_maneuver.goal_joints_rad", compactGoalOverride()),
    });
    node_ = std::make_shared<rclcpp::Node>("motion_reference_test", options);
    monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node_, "robot_description");
    model_ = monitor_->getRobotModel();
    ASSERT_TRUE(model_);
    group_ = model_->getJointModelGroup("xarm6");
    ASSERT_NE(group_, nullptr);
    ASSERT_EQ(group_->getVariableCount(), 6U);
    parameters_.joint_names = group_->getVariableNames();
    parameters_.recovery_mode = recoveryMode();
    parameters_.gaze_frame = gazeFrame();
    parameters_.command_frame = commandFrame();
    limits_.lower_position.resize(6);
    limits_.upper_position.resize(6);
    limits_.position_margin = Eigen::VectorXd::Constant(6, 0.10);
    limits_.max_velocity = Eigen::VectorXd::Constant(6, 1.0);
    limits_.max_acceleration = Eigen::VectorXd::Constant(6, 2.0);
    limits_.max_jerk = Eigen::VectorXd::Constant(6, 20.0);
    for (Eigen::Index joint = 0; joint < 6; ++joint) {
      const auto & bounds = model_->getVariableBounds(parameters_.joint_names[joint]);
      limits_.lower_position[joint] = bounds.min_position_;
      limits_.upper_position[joint] = bounds.max_position_;
    }
    const auto numerical_policy = NumericalPolicy::fromDiscreteMotionLimits(
      PositionInterval{limits_.lower_position.minCoeff(), limits_.upper_position.maxCoeff()},
      limits_.max_velocity.maxCoeff(), limits_.max_acceleration.maxCoeff(),
      limits_.max_jerk.maxCoeff(), 0.01, SolverResidual{1.0e-6});
    ASSERT_TRUE(numerical_policy.has_value());
    limits_.numerical_policy = *numerical_policy;
    {
      planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
      Eigen::Isometry3d tabletop = Eigen::Isometry3d::Identity();
      tabletop.translation().z() = 0.725;
      scene->getWorldNonConst()->addToObject(
        "round_table", std::make_shared<shapes::Cylinder>(0.5, 0.05), tabletop);
      scene->getAllowedCollisionMatrixNonConst().setEntry("link_base", "round_table", true);
      ASSERT_TRUE(scene->getCurrentStateNonConst().setToDefaultValues(group_, "rest"));
      scene->getCurrentStateNonConst().update();
    }
    collision_config_.protected_joint_lower_rad = limits_.lower_position[4];
    collision_config_.protected_joint_upper_rad = limits_.upper_position[4];
    collision_config_.numerical_policy = numerical_policy;
    collision_config_.invariant_mount_neighbor_name = "link6";
    configureGeometryBounds(collision_config_, *model_, *group_);
    reference_ = std::make_unique<MotionReference>(
      node_, monitor_, parameters_, limits_, collision_config_);
    clock_origin_ = std::chrono::steady_clock::now();
  }

  void TearDown() override
  {
    // Stop the planner worker before its scene monitor and node are released.
    reference_.reset();
    monitor_.reset();
    model_.reset();
    group_ = nullptr;
    node_.reset();
  }

  virtual std::string gazeFrame() const {return "";}
  virtual std::string commandFrame() const {return "monitor_control_frame";}
  virtual bool searchEnabled() const {return true;}
  virtual std::string recoveryMode() const {return "compact";}
  virtual bool compactProfileEnabled() const {return true;}
  virtual std::vector<double> compactGoalOverride() const {return {};}

  void recordQualifyingFaceShortfall(const moveit::core::RobotState & robot_state)
  {
    auto selected = update(robot_state, target(msg::TrackingTarget::FACE));
    ASSERT_TRUE(selected.tracking);
    auto tracking = *selected.tracking;
    tracking.has_face_primary_roll = true;
    tracking.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
    tracking.face_primary_feedforward_radps.setZero();
    QpResult qp;
    qp.status = QpStatus::kSolved;
    qp.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
    for (int tick = 0; tick <= 8; ++tick) {
      reference_->recordFaceCommand(
        tracking, qp, 1.0 + 0.02 * static_cast<double>(tick), true);
    }
  }

  std::vector<CompactEvent> commitCompactWithLifecycleFixturePath(
    const moveit::core::RobotState & robot_state)
  {
    recordQualifyingFaceShortfall(robot_state);
    beginPreparationForResultTest(robot_state);
    if (MotionReferenceTestPeer::hasPreparation(*reference_)) {
      const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
      consumePreparationResult(robot_state, plan);
    }
    return reference_->drainCompactEvents();
  }

  void admitCurrentPreparationWithLifecycleFixturePath(
    const moveit::core::RobotState & robot_state,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::uint64_t scene_revision = kSceneRevision,
    const std::optional<JointMotionState> & actual_feedback_override = std::nullopt)
  {
    if (!MotionReferenceTestPeer::hasPreparation(*reference_)) {
      beginPreparationForResultTest(
        robot_state, command_queue_state, active_publication,
        latest_publication, scene_revision);
    }
    if (!MotionReferenceTestPeer::hasPreparation(*reference_)) {
      return;
    }
    MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
    MotionReferenceTestPeer::activateResultGeneration(
      *reference_, reference_->diagnostics().compact_generation + 100U);
    const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
    consumePreparationResult(
      robot_state, plan, command_queue_state, active_publication,
      latest_publication, scene_revision, actual_feedback_override);
  }

  moveit::core::RobotState state(const double base_angle = 0.0) const
  {
    moveit::core::RobotState result(model_);
    result.setToDefaultValues();
    if (!result.setToDefaultValues(group_, "rest")) {
      throw std::runtime_error("The real robot model must contain a named rest state");
    }
    result.setVariablePosition("joint1", base_angle);
    result.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    result.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    result.update();
    return result;
  }

  moveit::core::RobotState compactGoalState(const double joint1_offset = 0.0) const
  {
    moveit::core::RobotState result(model_);
    result.setToDefaultValues();
    if (!result.setToDefaultValues(group_, "compact_turn_negative_wrist")) {
      throw std::runtime_error("The xArm6 model must contain the compact goal state");
    }
    result.setVariablePosition("joint1", result.getVariablePosition("joint1") + joint1_offset);
    result.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    result.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    result.update();
    return result;
  }

  moveit::core::RobotState nearScreenPoleState() const
  {
    auto result = state();
    result.setVariablePosition("joint1", 0.2890392970728219);
    result.setVariablePosition("joint2", -0.3842942596277428);
    result.setVariablePosition("joint3", -3.025275867240606);
    result.setVariablePosition("joint4", 3.525770103294581);
    result.setVariablePosition("joint5", -0.26038283922245037);
    result.setVariablePosition("joint6", 4.5984044351267475);
    result.update();
    return result;
  }

  msg::TrackingTarget faceAlongLookDirection(const moveit::core::RobotState & robot_state) const
  {
    auto command = target(msg::TrackingTarget::FACE);
    const auto * look_link = parameters_.gaze_frame.empty() ? nullptr :
      model_->getLinkModel(parameters_.gaze_frame);
    Eigen::Isometry3d look_pose = look_link ?
      robot_state.getGlobalLinkTransform(look_link) :
      robot_state.getGlobalLinkTransform("monitor_control_frame");
    if (look_link) {
      look_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
    }
    const Eigen::Vector3d face = look_pose.translation() + 0.8 * look_pose.linear().col(0);
    command.face.x = face.x();
    command.face.y = face.y();
    command.face.z = face.z();
    return command;
  }

  msg::TrackingTarget target(const std::uint8_t mode) const
  {
    msg::TrackingTarget result;
    result.header.frame_id = "world";
    result.mode = mode;
    result.pose.orientation.w = 1.0;
    result.pose.position.x = 0.2;
    result.pose.position.z = 1.05;
    result.face.x = 0.85;
    result.face.z = 1.68;
    return result;
  }

  MotionReferenceResult update(
    const moveit::core::RobotState & robot_state,
    const std::optional<msg::TrackingTarget> & command,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const std::uint64_t scene_revision = kSceneRevision,
    const std::optional<double> & time_override_sec = std::nullopt)
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = time_override_sec.value_or(std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count());
    Eigen::VectorXd actual_position;
    Eigen::VectorXd actual_velocity;
    robot_state.copyJointGroupPositions(group_, actual_position);
    robot_state.copyJointGroupVelocities(group_, actual_velocity);
    const JointMotionState actual_feedback{
      actual_position, actual_velocity, Eigen::VectorXd::Zero(actual_position.size())};
    const std::optional<JointMotionState> effective_command_queue_state = command_queue_state ?
      command_queue_state : std::optional<JointMotionState>{actual_feedback};
    return reference_->update(
      robot_state, command, *planning_scene, now, scene_revision, actual_feedback, now,
      effective_command_queue_state, active_publication, latest_publication);
  }

  PairYawResourceAssessment pairYawAssessment(
    const moveit::core::RobotState & robot_state,
    const msg::TrackingTarget & command) const
  {
    return MotionReferenceTestPeer::evaluatePairYawResource(
      *reference_, robot_state, command);
  }

  msg::TrackingTarget targetWithYawOffset(
    const moveit::core::RobotState & robot_state, const double yaw_offset_rad) const
  {
    auto command = faceAlongLookDirection(robot_state);
    const auto * tracking_link = parameters_.gaze_frame.empty() ?
      model_->getLinkModel(parameters_.command_frame) :
      model_->getLinkModel(parameters_.gaze_frame);
    if (!tracking_link) {
      throw std::runtime_error("The pair-yaw test needs a valid tracking link");
    }
    Eigen::Isometry3d tracking_pose = robot_state.getGlobalLinkTransform(tracking_link);
    if (!parameters_.gaze_frame.empty()) {
      tracking_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
    }
    const Eigen::Isometry3d base_pose = robot_state.getGlobalLinkTransform("link_base");
    const Eigen::Vector3d face(command.face.x, command.face.y, command.face.z);
    Eigen::Vector3d ray_base = base_pose.linear().transpose() *
      (face - tracking_pose.translation());
    ray_base = Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ()) * ray_base;
    const Eigen::Vector3d offset_face = tracking_pose.translation() +
      base_pose.linear() * ray_base;
    command.face.x = offset_face.x();
    command.face.y = offset_face.y();
    command.face.z = offset_face.z();
    return command;
  }

  std::pair<moveit::core::RobotState, msg::TrackingTarget> criticalPairExhaustedCase() const
  {
    auto robot_state = state();
    constexpr std::array<double, 6> kJointAnglesRad{
      6.233, 0.261, -1.674, -0.056, -0.812, 6.283};
    for (std::size_t index = 0; index < kJointAnglesRad.size(); ++index) {
      robot_state.setVariablePosition(parameters_.joint_names[index], kJointAnglesRad[index]);
    }
    robot_state.update(true);
    auto command = target(msg::TrackingTarget::FACE);
    command.face.x = 0.850;
    command.face.y = 0.023;
    command.face.z = 1.686;
    return {std::move(robot_state), std::move(command)};
  }

  void consumePreparationResult(
    const moveit::core::RobotState & robot_state,
    const BackgroundPathResult & plan,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::uint64_t scene_revision = kSceneRevision,
    const std::optional<JointMotionState> & actual_feedback_override = std::nullopt)
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count();
    Eigen::VectorXd actual_position;
    Eigen::VectorXd actual_velocity;
    robot_state.copyJointGroupPositions(group_, actual_position);
    robot_state.copyJointGroupVelocities(group_, actual_velocity);
    const JointMotionState actual_feedback = actual_feedback_override.value_or(
      JointMotionState{
          actual_position, actual_velocity, Eigen::VectorXd::Zero(actual_position.size())});
    const std::optional<JointMotionState> effective_command_queue_state = command_queue_state ?
      command_queue_state : std::optional<JointMotionState>{actual_feedback};
    MotionReferenceTestPeer::consume(
      *reference_, plan, robot_state, now, *planning_scene, scene_revision,
      actual_feedback, now, effective_command_queue_state,
      active_publication, latest_publication);
  }

  void beginPreparationForResultTest(
    const moveit::core::RobotState & robot_state,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::uint64_t scene_revision = kSceneRevision)
  {
    MotionReferenceTestPeer::qualifyPairYawForIsolatedAdmissionTest(*reference_);
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count();
    Eigen::VectorXd actual_position;
    Eigen::VectorXd actual_velocity;
    robot_state.copyJointGroupPositions(group_, actual_position);
    robot_state.copyJointGroupVelocities(group_, actual_velocity);
    const JointMotionState actual_feedback{
      actual_position, actual_velocity, Eigen::VectorXd::Zero(actual_position.size())};
    const std::optional<JointMotionState> effective_command_queue_state = command_queue_state ?
      command_queue_state : std::optional<JointMotionState>{actual_feedback};
    MotionReferenceTestPeer::beginPreparation(
      *reference_, robot_state, now, *planning_scene, scene_revision,
      actual_feedback, effective_command_queue_state,
      active_publication, latest_publication);
    MotionReferenceTestPeer::activateResultGeneration(
      *reference_, reference_->diagnostics().compact_generation + 100U);
  }

  void admitWorkerCertifiedLifecycleFixturePath(
    const moveit::core::RobotState & robot_state,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr)
  {
    const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
    consumePreparationResult(
      robot_state, plan, command_queue_state,
      active_publication, latest_publication);
  }

  std::optional<MotionReferenceResult> waitForRestPath(
    const moveit::core::RobotState & robot_state)
  {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto result = update(robot_state, target(msg::TrackingTarget::REST));
      if (result.follows_path && reference_->diagnostics().plans_accepted > 0U) {
        return result;
      }
      std::this_thread::sleep_for(1ms);
    }
    return std::nullopt;
  }

  void expectPendingRestPreempted(const std::uint8_t new_mode)
  {
    const auto start = state(0.4);
    const auto waiting = update(start, target(msg::TrackingTarget::REST));
    EXPECT_FALSE(waiting.task);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
    // The first update submits but cannot adopt its newly submitted result.
    // Change mode immediately, then keep polling past the planner's 250 ms budget
    // so a late completion from REST has an opportunity to reach the selector.
    const auto deadline = std::chrono::steady_clock::now() + 350ms;
    do {
      const auto selected = update(start, target(new_mode));
      EXPECT_FALSE(selected.follows_path);
      EXPECT_EQ(reference_->diagnostics().plans_accepted, 0U);
      if (new_mode == msg::TrackingTarget::HOLD) {
        EXPECT_FALSE(selected.task);
        EXPECT_EQ(reference_->diagnostics().state, "HOLD");
      } else {
        EXPECT_TRUE(selected.tracking);
        EXPECT_EQ(reference_->diagnostics().state, "FACE");
      }
      std::this_thread::sleep_for(2ms);
    } while (std::chrono::steady_clock::now() < deadline);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  }

  rclcpp::Node::SharedPtr node_;
  planning_scene_monitor::PlanningSceneMonitorPtr monitor_;
  moveit::core::RobotModelConstPtr model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  ControllerParameters parameters_;
  JointMotionLimits limits_;
  CollisionConstraintConfig collision_config_;
  std::unique_ptr<MotionReference> reference_;
  std::chrono::steady_clock::time_point clock_origin_;
};

class MotionReferenceCameraTest : public MotionReferenceTest
{
protected:
  std::string gazeFrame() const override {return "test_camera_optical";}
};

class MotionReferenceSearchDisabledTest : public MotionReferenceTest
{
protected:
  bool searchEnabled() const override {return false;}
};

class MotionReferenceCompactDisabledTest : public MotionReferenceTest
{
protected:
  bool compactProfileEnabled() const override {return false;}
};

class MotionReferenceRecoveryDisabledTest : public MotionReferenceTest
{
protected:
  std::string recoveryMode() const override {return "disabled";}
};

class MotionReferenceRecoveryDisabledNoCompactProfileTest : public MotionReferenceTest
{
protected:
  std::string recoveryMode() const override {return "disabled";}
  bool compactProfileEnabled() const override {return false;}
};

class MotionReferenceCustomCompactGoalTest : public MotionReferenceTest
{
protected:
  std::vector<double> compactGoalOverride() const override
  {
    return {0.0, -0.2715944, -2.8699983, 0.0, -1.151386, 0.7};
  }
};

class MotionReferenceInvalidCompactMountTest : public MotionReferenceTest
{
protected:
  std::string commandFrame() const override {return "test_camera_optical";}
};

TEST_F(MotionReferenceTest, PairYawGateHasValidJacobianAtAlignedFeasiblePose)
{
  const auto robot_state = state();
  const auto aligned = faceAlongLookDirection(robot_state);
  const auto assessment = pairYawAssessment(robot_state, aligned);

  ASSERT_TRUE(assessment.valid);
  EXPECT_FALSE(assessment.demand_present);

  auto pure_vertical_error = aligned;
  pure_vertical_error.face.z += 0.20;
  const auto vertical_assessment = pairYawAssessment(robot_state, pure_vertical_error);
  ASSERT_TRUE(vertical_assessment.valid);
  EXPECT_FALSE(vertical_assessment.demand_present);
  EXPECT_FALSE(vertical_assessment.exhausted);
}

TEST_F(MotionReferenceTest, PairYawProbeRollIntervalClipsBothEdgeSignsOnlyNearCap)
{
  constexpr double kPi = 3.14159265358979323846;
  const double cap = std::sin(5.1 * kPi / 180.0);
  const double local_allowance = std::sin(kPi / 180.0);
  const double near_edge = 0.0858;

  const auto negative_edge = MotionReferenceTestPeer::pairYawProbeEdgeDeltaInterval(-near_edge);
  ASSERT_TRUE(negative_edge);
  EXPECT_NEAR((*negative_edge)[0], -cap + near_edge, 1.0e-15);
  EXPECT_DOUBLE_EQ((*negative_edge)[1], local_allowance);

  const auto positive_edge = MotionReferenceTestPeer::pairYawProbeEdgeDeltaInterval(near_edge);
  ASSERT_TRUE(positive_edge);
  EXPECT_DOUBLE_EQ((*positive_edge)[0], -local_allowance);
  EXPECT_NEAR((*positive_edge)[1], cap - near_edge, 1.0e-15);

  const auto interior = MotionReferenceTestPeer::pairYawProbeEdgeDeltaInterval(0.0);
  ASSERT_TRUE(interior);
  EXPECT_DOUBLE_EQ((*interior)[0], -local_allowance);
  EXPECT_DOUBLE_EQ((*interior)[1], local_allowance);
  EXPECT_FALSE(MotionReferenceTestPeer::pairYawProbeEdgeDeltaInterval(cap + 1.0e-6));
  EXPECT_FALSE(MotionReferenceTestPeer::pairYawProbeEdgeDeltaInterval(
      std::numeric_limits<double>::quiet_NaN()));
}

TEST_F(MotionReferenceTest, PairYawProbeNearNegativeCapKeepsOnlyCapValidResource)
{
  auto robot_state = state();
  Eigen::VectorXd captured_q(6);
  captured_q << 6.2088163036370609, 0.2772025522914156, -1.6536501053918486,
    -0.087558368682742596, -0.83288051421121145, 6.2831839078553147;
  robot_state.setJointGroupPositions(group_, captured_q);
  robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  robot_state.update(true);
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = 0.8405535239328289;
  command.face.y = 0.1263715688123846;
  command.face.z = 1.6757422070843495;
  MotionReferenceTestPeer::holdBaseJointForTest(
    *reference_, captured_q[0], BaseJointDirection::kPositive);

  const auto assessment = pairYawAssessment(robot_state, command);

  ASSERT_TRUE(assessment.pair_bound_candidate);
  ASSERT_TRUE(assessment.actual_probe_solved);
  ASSERT_TRUE(assessment.relaxed_probe_solved);
  EXPECT_LT(assessment.actual_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
  EXPECT_GE(assessment.pair_relaxed_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
  EXPECT_LE(assessment.actual_roll_residual_rad,
    std::asin(std::sin(5.1 * 3.14159265358979323846 / 180.0)) -
    std::asin(0.08580283244921869) + 1.0e-6);
  EXPECT_LE(assessment.relaxed_roll_residual_rad, 1.0 * 3.14159265358979323846 / 180.0 + 1.0e-4);
}

TEST_F(MotionReferenceTest, StaleInitialReadEpochWaitsUntilFreshActiveProvenance)
{
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));
  Eigen::VectorXd q(6);
  state().copyJointGroupPositions(group_, q);

  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, q, 11U, 1U, nullptr, nullptr, 10U);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hardwareActiveSeen(*component));
  EXPECT_NE(CollisionAwareServoComponentTestPeer::hardwareStatusFailure(*component), "valid");
  CollisionAwareServoComponentTestPeer::controlTick(*component);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hardwareInhibitPending(*component));
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hasPublishedCommand(*component));

  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, q, 11U, 2U);
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::hardwareActiveSeen(*component));
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::hardwareStatusFailure(*component), "valid");
  CollisionAwareServoComponentTestPeer::controlTick(*component);

  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, q, 11U, 3U, nullptr, nullptr, 10U);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::currentHardwareStatusValid(*component));
  CollisionAwareServoComponentTestPeer::controlTick(*component);
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::hardwareInhibitPending(*component));
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hasPublishedCommand(*component));
}

TEST_F(MotionReferenceTest, RecoverableTfGapKeepsCheckedTailWithoutHardwareInhibit)
{
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));

  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::retainStationaryPublicationAndRecover(
      *component, "TF sample was temporarily unavailable"));
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hardwareInhibitPending(*component));
}

TEST_F(MotionReferenceTest, RestBrakeContinuationKeepsPathOnlyForFreshMatchingTarget)
{
  const auto make_component = [this]() {
      auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
      std::vector<PositionBoundSet> bounds;
      for (const auto & name : parameters_.joint_names) {
        const auto & variable_bounds = model_->getVariableBounds(name);
        const PositionInterval physical{
          variable_bounds.min_position_, variable_bounds.max_position_};
        bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
      }
      CollisionAwareServoComponentTestPeer::configure(
        *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
      return component;
    };
  const auto install_rest_path = [this](CollisionAwareServoComponent & component) {
      auto rest_reference = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      const auto start_state = state();
      Eigen::VectorXd start_position;
      start_state.copyJointGroupPositions(group_, start_position);
      Eigen::VectorXd goal_position = start_position;
      goal_position[0] += 0.01;
      EXPECT_TRUE(MotionReferenceTestPeer::installRestFollowerPathForTest(
        *rest_reference, start_state, {start_position, goal_position}, 0.0));
      CollisionAwareServoComponentTestPeer::installMotionReference(
        component, std::move(rest_reference));
      return start_position;
    };

  {
    auto component = make_component();
    const Eigen::VectorXd position = install_rest_path(*component);
    const auto start = component->now();
    const auto publication_start = start + rclcpp::Duration::from_nanoseconds(100'000'000LL);
    auto rest_target = target(msg::TrackingTarget::REST);
    rest_target.target_id = 1;
    rest_target.header.stamp = start;
    CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, rest_target);
    ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installPublishedBrakeSuffix(
      *component, position, publication_start, 1U));
    const auto expected_next_stamp =
      CollisionAwareServoComponentTestPeer::nextPublishedBrakeStamp(*component, 1U);

    EXPECT_EQ(CollisionAwareServoComponentTestPeer::tryContinuePublishedBrake(
      *component, start, rest_target, 1U), PublishedBrakeContinuationStatus::kContinued);
    EXPECT_TRUE(CollisionAwareServoComponentTestPeer::restFollowerActive(*component));
    EXPECT_EQ(CollisionAwareServoComponentTestPeer::continuationQueueSize(*component), 2U);
    EXPECT_EQ(
      CollisionAwareServoComponentTestPeer::continuationQueueTailStamp(*component),
      expected_next_stamp);
  }

  for (const std::string & invalidation : {"mode", "generation", "stale", "scene"}) {
    auto component = make_component();
    const Eigen::VectorXd position = install_rest_path(*component);
    const auto start = component->now();
    const auto publication_start = start + rclcpp::Duration::from_nanoseconds(100'000'000LL);
    auto current_target = target(msg::TrackingTarget::REST);
    current_target.target_id = 1;
    current_target.header.stamp = start;
    auto requested_target = current_target;
    std::uint64_t expected_scene_revision = 1U;
    if (invalidation == "mode") {
      requested_target.mode = msg::TrackingTarget::SEARCH;
    } else if (invalidation == "generation") {
      requested_target.target_id = 2;
    } else if (invalidation == "stale") {
      current_target.header.stamp = start - rclcpp::Duration::from_seconds(1.0);
    } else {
      expected_scene_revision = 2U;
    }
    CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, current_target);
    ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installPublishedBrakeSuffix(
      *component, position, publication_start, 1U));
    if (invalidation == "scene") {
      CollisionAwareServoComponentTestPeer::setPlanningSceneRevision(*component, 2U);
    }

    EXPECT_EQ(CollisionAwareServoComponentTestPeer::tryContinuePublishedBrake(
      *component, start, requested_target, expected_scene_revision),
      PublishedBrakeContinuationStatus::kNotEligible) << invalidation;
    EXPECT_FALSE(CollisionAwareServoComponentTestPeer::restFollowerActive(*component))
      << invalidation;
    EXPECT_EQ(CollisionAwareServoComponentTestPeer::continuationQueueSize(*component), 1U)
      << invalidation;
  }
}

TEST_F(MotionReferenceTest, ExhaustedRestPublishedSuffixDropsFollowerPath)
{
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  auto rest_reference = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  const auto start_state = state();
  Eigen::VectorXd start_position;
  start_state.copyJointGroupPositions(group_, start_position);
  Eigen::VectorXd goal_position = start_position;
  goal_position[0] += 0.01;
  ASSERT_TRUE(MotionReferenceTestPeer::installRestFollowerPathForTest(
    *rest_reference, start_state, {start_position, goal_position}, 0.0));
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(rest_reference));

  const auto start = component->now();
  const auto publication_start = start + rclcpp::Duration::from_nanoseconds(100'000'000LL);
  auto rest_target = target(msg::TrackingTarget::REST);
  rest_target.target_id = 1;
  rest_target.header.stamp = start;
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, rest_target);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installPublishedBrakeSuffix(
    *component, start_position, publication_start, 1U, 2U, 4U));
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::restFollowerActive(*component));

  EXPECT_EQ(CollisionAwareServoComponentTestPeer::tryContinuePublishedBrake(
    *component, start, rest_target, 1U),
    PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::restFollowerActive(*component));
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::continuationQueueSize(*component), 0U);
}

TEST_F(MotionReferenceTest, HardwareQHoldSurvivesBootstrapAndIsInvalidatedByNewEpoch)
{
  monitor_->startStateMonitor("/joint_states");
  auto joint_state_publisher = node_->create_publisher<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::QoS(10).reliable());

  auto hold_robot_state = state();
  Eigen::VectorXd hold(static_cast<Eigen::Index>(group_->getVariableCount()));
  hold_robot_state.copyJointGroupPositions(group_, hold);
  hold[0] += 0.05;
  hold_robot_state.setJointGroupPositions(group_, hold);
  hold_robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(hold.size()));
  hold_robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(hold.size()));
  hold_robot_state.update();

  const auto publish_joint_state = [&](const Eigen::VectorXd & positions) {
      sensor_msgs::msg::JointState message;
      message.header.stamp = node_->now();
      message.name = parameters_.joint_names;
      message.position.assign(positions.data(), positions.data() + positions.size());
      message.velocity.assign(positions.size(), 0.0);
      joint_state_publisher->publish(message);
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

  rclcpp::NodeOptions component_options;
  auto component = std::make_shared<CollisionAwareServoComponent>(component_options);
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (std::size_t index = 0; index < parameters_.joint_names.size(); ++index) {
    const auto & joint_bounds = model_->getVariableBounds(parameters_.joint_names[index]);
    const PositionInterval physical{joint_bounds.min_position_, joint_bounds.max_position_};
    const double center = hold[static_cast<Eigen::Index>(index)];
    bounds.push_back(PositionBoundSet{
          physical, physical, physical, PositionInterval{center - 0.15, center + 0.15}});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);

  std::string reason;
  bool rearm_ready = false;
  for (std::uint64_t sequence = 1U; sequence <= 10U; ++sequence) {
    publish_joint_state(hold);
    CollisionAwareServoComponentTestPeer::publishHardwareSample(
      *component, hold, 11U, sequence);
    const RearmReadiness readiness =
      CollisionAwareServoComponentTestPeer::checkRearm(*component, reason);
    ASSERT_NE(readiness, RearmReadiness::kUnsafe) << reason;
    if (readiness == RearmReadiness::kReady) {
      rearm_ready = true;
      break;
    }
  }
  ASSERT_TRUE(rearm_ready) << reason << " / "
                           << CollisionAwareServoComponentTestPeer::hardwareStatusFailure(
        *component)
                           << " / " <<
        CollisionAwareServoComponentTestPeer::rearmSequenceState(*component)
                           << " / hold0=" << hold[0] << " working0=[" << bounds[0].working.lower_rad
                           << "," << bounds[0].working.upper_rad << "]";
  const auto retained = CollisionAwareServoComponentTestPeer::provenHold(*component);
  ASSERT_TRUE(retained.has_value());
  EXPECT_LT(((*retained) - hold).norm(), 1.0e-12);

  Eigen::VectorXd outside_wire = hold;
  outside_wire[0] = static_cast<double>(static_cast<float>(hold[0] + 0.4));
  ASSERT_TRUE(control::classifyMeasuredPosition(
      outside_wire[0], bounds[0].working) == control::BoundaryClass::kInterior);
  EXPECT_FALSE(control::isWireCommand(outside_wire[0], bounds[0].wire));
  publish_joint_state(outside_wire);
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, outside_wire, 11U, 4U, &outside_wire, &hold);
  bool requires_latched_halt = false;
  const auto bootstrap_state = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  ASSERT_TRUE(bootstrap_state.has_value()) << reason;
  EXPECT_FALSE(requires_latched_halt);
  EXPECT_LT((bootstrap_state->positions - hold).norm(), 1.0e-12);

  moveit::core::RobotState bootstrap_robot_state(hold_robot_state);
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
    ASSERT_TRUE(locked_scene);
    const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
    const auto bootstrap = CollisionAwareServoComponentTestPeer::stationaryBootstrap(
      *component, *bootstrap_state, bootstrap_robot_state, *planning_scene, reason);
    ASSERT_TRUE(bootstrap.has_value()) << reason;
    EXPECT_LT((bootstrap->seed_state.positions - hold).norm(), 1.0e-12);
    ASSERT_TRUE(bootstrap->first_command.braking_tail.command_available());
    for (const auto & point : bootstrap->first_command.braking_tail.points) {
      for (std::size_t index = 0; index < bounds.size(); ++index) {
        EXPECT_TRUE(control::isWireCommand(
            point.position[static_cast<Eigen::Index>(index)], bounds[index].wire));
      }
    }

    control::HierarchicalVelocityTask task;
    task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
    task.primary_reference = Eigen::VectorXd::Zero(6);
    task.primary_reference[0] = -0.05;
    task.primary_weights = Eigen::VectorXd::Ones(6);
    task.secondary_matrix = Eigen::MatrixXd(0, 6);
    task.secondary_reference = Eigen::VectorXd(0);
    task.secondary_weights = Eigen::VectorXd(0);
    HierarchicalVelocityQpConfig qp_config;
    qp_config.period_sec = 0.01;
    const HierarchicalVelocityQp solver(qp_config);
    const JointMotionState hold_motion{hold, Eigen::VectorXd::Zero(6),
      Eigen::VectorXd::Zero(6)};
    const QpResult inward = solver.solve(task, hold_motion, limits_);
    ASSERT_TRUE(inward.command_available());
    ASSERT_LT(inward.joint_velocity[0], 0.0);
    const auto accepted = CollisionAwareServoComponentTestPeer::validateCandidate(
      *component, std::make_shared<moveit::core::RobotState>(bootstrap_robot_state),
      *bootstrap_state, hold_motion, inward.joint_velocity, *planning_scene, reason);
    ASSERT_TRUE(accepted.has_value()) << reason;
    for (std::size_t index = 0; index < bounds.size(); ++index) {
      EXPECT_TRUE(control::isWireCommand(
          accepted->next_state.positions[static_cast<Eigen::Index>(index)], bounds[index].wire));
    }
    ASSERT_TRUE(accepted->braking_tail.command_available());
    for (const auto & point : accepted->braking_tail.points) {
      for (std::size_t index = 0; index < bounds.size(); ++index) {
        EXPECT_TRUE(control::isWireCommand(
            point.position[static_cast<Eigen::Index>(index)], bounds[index].wire));
      }
    }

    // Solve the most-negative jerk-limited command from a near-saturated
    // acceleration, then admit it through the component and validate its full
    // braking tail. This exercises the QP -> integrated candidate -> tail path.
    JointMotionState saturated_motion = hold_motion;
    saturated_motion.acceleration[0] = -1.8;
    moveit_servo::KinematicState saturated_current = *bootstrap_state;
    saturated_current.accelerations[0] = -1.8;
    control::HierarchicalVelocityTask saturated_task;
    saturated_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
    saturated_task.primary_reference = Eigen::VectorXd::Zero(6);
    saturated_task.primary_reference[0] = -1.0;
    saturated_task.primary_weights = Eigen::VectorXd::Ones(6);
    saturated_task.secondary_matrix = Eigen::MatrixXd(0, 6);
    saturated_task.secondary_reference = Eigen::VectorXd(0);
    saturated_task.secondary_weights = Eigen::VectorXd(0);
    const QpResult saturated_qp = solver.solve(
      saturated_task, saturated_motion, limits_);
    ASSERT_TRUE(saturated_qp.command_available());
    EXPECT_NEAR(saturated_qp.joint_velocity[0], -0.02, 1.0e-5);
    const auto saturated_admission = CollisionAwareServoComponentTestPeer::validateCandidate(
      *component, std::make_shared<moveit::core::RobotState>(bootstrap_robot_state),
      saturated_current, saturated_motion, saturated_qp.joint_velocity,
      *planning_scene, reason);
    ASSERT_TRUE(saturated_admission.has_value()) << reason;
    EXPECT_GT(std::abs(saturated_admission->next_state.accelerations[0]), 1.99);
    const double candidate_jerk =
      (saturated_admission->next_state.accelerations[0] -
      saturated_current.accelerations[0]) / parameters_.control_period_sec;
    EXPECT_GT(std::abs(candidate_jerk), 19.9);
    ASSERT_TRUE(saturated_admission->braking_tail.command_available());
    for (const auto & point : saturated_admission->braking_tail.points) {
      for (std::size_t index = 0; index < bounds.size(); ++index) {
        EXPECT_TRUE(control::isWireCommand(
            point.position[static_cast<Eigen::Index>(index)], bounds[index].wire));
      }
    }
  }

  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, outside_wire, 11U, 5U, &outside_wire, &outside_wire);
  const auto lost_ownership_state = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  EXPECT_FALSE(lost_ownership_state.has_value());
  EXPECT_TRUE(requires_latched_halt);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::provenHold(*component).has_value());
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hasPublishedCommand(*component));

  EXPECT_TRUE(
    CollisionAwareServoComponentTestPeer::prepareClockRewindWithRetainedTail(*component));

  // Same-epoch controller ownership can also change while the raw sample stays
  // representable. It must not turn that raw sample into a fresh bootstrap.
  CollisionAwareServoComponentTestPeer::setHardwareExecutionError(*component, 0.20);
  Eigen::VectorXd raw_inside_wire = hold;
  Eigen::VectorXd changed_reference = hold;
  raw_inside_wire[0] -= 0.10;
  changed_reference[0] += 0.05;
  ASSERT_TRUE(control::isWireCommand(raw_inside_wire[0], bounds[0].wire));
  ASSERT_TRUE(control::isWireCommand(changed_reference[0], bounds[0].wire));
  publish_joint_state(raw_inside_wire);
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, raw_inside_wire, 11U, 6U, &changed_reference, &changed_reference);
  const auto unowned_but_wire_valid = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  ASSERT_TRUE(unowned_but_wire_valid.has_value()) << reason;
  EXPECT_FALSE(requires_latched_halt);
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
    ASSERT_TRUE(locked_scene);
    const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
    const auto rejected_bootstrap = CollisionAwareServoComponentTestPeer::stationaryBootstrap(
      *component, *unowned_but_wire_valid, bootstrap_robot_state, *planning_scene, reason);
    EXPECT_FALSE(rejected_bootstrap.has_value());
    EXPECT_NE(reason.find("ownership"), std::string::npos) << reason;
  }
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hasPublishedCommand(*component));
  CollisionAwareServoComponentTestPeer::setHardwareExecutionError(*component, 0.5);

  // Model a new executor session receiving a valid active status with a new
  // driver epoch. A stale q_hold from the former epoch must not seed it.
  CollisionAwareServoComponentTestPeer::seedProvenHoldForEpochTest(
    *component, hold, 11U);
  CollisionAwareServoComponentTestPeer::resetStatusEpochForNewSession(*component);
  Eigen::VectorXd new_epoch_raw = hold;
  new_epoch_raw[0] += 0.08;
  publish_joint_state(new_epoch_raw);
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, new_epoch_raw, 12U, 1U, &new_epoch_raw, &hold);
  const auto unowned_state = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  ASSERT_TRUE(unowned_state.has_value()) << reason;
  EXPECT_FALSE(requires_latched_halt);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::provenHold(*component).has_value());
  EXPECT_LT((unowned_state->positions - new_epoch_raw).norm(), 1.0e-6);
  EXPECT_TRUE(control::isWireCommand(unowned_state->positions[0], bounds[0].wire));

  // With no proven hold, even a physically working raw state outside the
  // representable wire interval must stop before bootstrap/publication.
  CollisionAwareServoComponentTestPeer::resetStatusEpochForNewSession(*component);
  publish_joint_state(outside_wire);
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, outside_wire, 12U, 1U, &hold, &hold);
  const auto unheld_outside_wire = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  EXPECT_FALSE(unheld_outside_wire.has_value());
  EXPECT_TRUE(requires_latched_halt);
  EXPECT_NE(reason.find("wire interval"), std::string::npos) << reason;
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::hasPublishedCommand(*component));
}

TEST_F(MotionReferenceTest, ComponentAdmissionKeepsInclusiveQvAndSaturatedAjEndpoints)
{
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);

  auto make_inputs = [&]() {
      moveit::core::RobotState robot_state = state();
      Eigen::VectorXd positions(6);
      robot_state.copyJointGroupPositions(group_, positions);
      moveit_servo::KinematicState current_state(6);
      current_state.joint_names = parameters_.joint_names;
      current_state.positions = positions;
      current_state.velocities = Eigen::VectorXd::Zero(6);
      current_state.accelerations = Eigen::VectorXd::Zero(6);
      return std::make_pair(std::move(robot_state), std::move(current_state));
    };

  auto [robot_state, current_state] = make_inputs();
  const Eigen::Index joint = 0;
  const double upper = limits_.upper_position[joint];
  current_state.positions[joint] = upper;
  robot_state.setVariablePosition(parameters_.joint_names[joint], upper);
  robot_state.update();
  const Eigen::VectorXd zero_velocity = Eigen::VectorXd::Zero(6);
  std::string reason;
  const auto at_q_endpoint = CollisionAwareServoComponentTestPeer::nextState(
    *component, std::make_shared<moveit::core::RobotState>(robot_state),
    current_state, zero_velocity, reason);
  ASSERT_TRUE(at_q_endpoint.has_value()) << reason;
  EXPECT_DOUBLE_EQ(at_q_endpoint->positions[joint], upper);

  // Inclusive nominal position and zero velocity must survive the complete
  // component admission path, including generation and validation of the full
  // braking suffix. The solver is intentionally not required to choose zero.
  const JointMotionState endpoint_motion{
    current_state.positions, current_state.velocities, current_state.accelerations};
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
    ASSERT_TRUE(locked_scene);
    const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
    const auto zero_candidate = CollisionAwareServoComponentTestPeer::validateCandidate(
      *component, std::make_shared<moveit::core::RobotState>(robot_state),
      current_state, endpoint_motion, zero_velocity, *planning_scene, reason);
    ASSERT_TRUE(zero_candidate.has_value()) << reason;
    EXPECT_DOUBLE_EQ(zero_candidate->next_state.positions[joint], upper);
    EXPECT_DOUBLE_EQ(zero_candidate->next_state.velocities[joint], 0.0);
    ASSERT_TRUE(zero_candidate->braking_tail.command_available())
      << zero_candidate->braking_tail.failure_reason;
    ASSERT_FALSE(zero_candidate->braking_tail.points.empty());
    for (const auto & point : zero_candidate->braking_tail.points) {
      EXPECT_DOUBLE_EQ(point.position[joint], upper);
      EXPECT_DOUBLE_EQ(point.velocity[joint], 0.0);
      EXPECT_DOUBLE_EQ(point.acceleration[joint], 0.0);
    }
  }

  auto [velocity_robot_state, velocity_state] = make_inputs();
  const double max_velocity = limits_.max_velocity[joint];
  velocity_state.velocities[joint] = max_velocity;
  Eigen::VectorXd max_velocity_command = Eigen::VectorXd::Zero(6);
  max_velocity_command[joint] = max_velocity;
  const auto at_v_endpoint = CollisionAwareServoComponentTestPeer::nextState(
    *component, std::make_shared<moveit::core::RobotState>(velocity_robot_state),
    velocity_state, max_velocity_command, reason);
  ASSERT_TRUE(at_v_endpoint.has_value()) << reason;
  EXPECT_DOUBLE_EQ(at_v_endpoint->velocities[joint], max_velocity);

  auto [saturated_robot_state, saturated_state] = make_inputs();
  const double period = parameters_.control_period_sec;
  const double max_acceleration = limits_.max_acceleration[joint];
  const double max_jerk = limits_.max_jerk[joint];
  saturated_state.accelerations[joint] = max_acceleration - max_jerk * period;
  Eigen::VectorXd saturated_command = Eigen::VectorXd::Zero(6);
  saturated_command[joint] = max_acceleration * period;
  const auto saturated = CollisionAwareServoComponentTestPeer::nextState(
    *component, std::make_shared<moveit::core::RobotState>(saturated_robot_state),
    saturated_state, saturated_command, reason);
  ASSERT_TRUE(saturated.has_value()) << reason;
  EXPECT_NEAR(saturated->accelerations[joint], max_acceleration, 1.0e-12);
  EXPECT_NEAR(
    (saturated->accelerations[joint] - saturated_state.accelerations[joint]) / period,
    max_jerk, 1.0e-10);

  auto [outside_robot_state, outside_state] = make_inputs();
  const double epsilon = 1.0e-5;
  outside_state.positions[joint] = upper - epsilon;
  outside_robot_state.setVariablePosition(parameters_.joint_names[joint], upper - epsilon);
  outside_robot_state.update();
  Eigen::VectorXd outside_command = Eigen::VectorXd::Zero(6);
  outside_command[joint] = 2.0 * epsilon / period;
  const auto outside = CollisionAwareServoComponentTestPeer::nextState(
    *component, std::make_shared<moveit::core::RobotState>(outside_robot_state),
    outside_state, outside_command, reason);
  EXPECT_FALSE(outside.has_value());
  EXPECT_NE(reason.find("q/v/a/j"), std::string::npos) << reason;

  auto outside_acceleration_inputs = make_inputs();
  Eigen::VectorXd outside_acceleration_command = Eigen::VectorXd::Zero(6);
  outside_acceleration_command[joint] = (limits_.max_acceleration[joint] + 1.0e-3) *
    parameters_.control_period_sec;
  const auto outside_acceleration = CollisionAwareServoComponentTestPeer::nextState(
    *component,
    std::make_shared<moveit::core::RobotState>(outside_acceleration_inputs.first),
    outside_acceleration_inputs.second, outside_acceleration_command, reason);
  EXPECT_FALSE(outside_acceleration.has_value());

  auto outside_jerk_inputs = make_inputs();
  Eigen::VectorXd outside_jerk_command = Eigen::VectorXd::Zero(6);
  const double beyond_jerk_acceleration =
    limits_.max_jerk[joint] * parameters_.control_period_sec + 1.0e-4;
  outside_jerk_command[joint] = beyond_jerk_acceleration *
    parameters_.control_period_sec;
  const auto outside_jerk = CollisionAwareServoComponentTestPeer::nextState(
    *component, std::make_shared<moveit::core::RobotState>(outside_jerk_inputs.first),
    outside_jerk_inputs.second, outside_jerk_command, reason);
  EXPECT_FALSE(outside_jerk.has_value());
}

TEST_F(MotionReferenceTest, JointThreeLimitExamplesDoNotExhaustJ1J4YawResource)
{
  constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
  const std::array<std::array<double, 6>, 2> joint_angles_deg{{
    {{15.0, -9.0, -70.0, -16.0, -56.0, 23.0}},
    {{183.0, 82.0, -222.9, 26.5, 9.0, -25.0}}}};

  for (const auto & angles_deg : joint_angles_deg) {
    auto robot_state = state();
    for (std::size_t index = 0; index < angles_deg.size(); ++index) {
      robot_state.setVariablePosition(
        parameters_.joint_names[index], angles_deg[index] * kDegreesToRadians);
    }
    robot_state.update(true);
    const auto yaw_demand = targetWithYawOffset(robot_state, 0.30);
    const auto assessment = pairYawAssessment(robot_state, yaw_demand);

    ASSERT_TRUE(assessment.valid);
    ASSERT_TRUE(assessment.demand_present);
    EXPECT_FALSE(assessment.pair_bound_candidate);
    EXPECT_FALSE(assessment.exhausted);
  }
}

TEST_F(MotionReferenceTest, PublicUpdateDoesNotCompactFromShortfallOrOtherJointLimits)
{
  constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
  const std::array<std::array<double, 6>, 2> joint_angles_deg{{
    {{15.0, -9.0, -70.0, -16.0, -56.0, 23.0}},
    {{183.0, 82.0, -222.9, 26.5, 9.0, -25.0}}}};

  for (const auto & angles_deg : joint_angles_deg) {
    auto robot_state = state();
    for (std::size_t index = 0; index < angles_deg.size(); ++index) {
      robot_state.setVariablePosition(
        parameters_.joint_names[index], angles_deg[index] * kDegreesToRadians);
    }
    robot_state.update(true);
    const auto yaw_demand = targetWithYawOffset(robot_state, 0.30);
    const auto selected = update(robot_state, yaw_demand);
    ASSERT_TRUE(selected.tracking);

    auto shortfall = *selected.tracking;
    shortfall.has_face_primary_roll = true;
    shortfall.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
    shortfall.face_primary_feedforward_radps.setZero();
    QpResult shortfall_qp;
    shortfall_qp.status = QpStatus::kSolved;
    shortfall_qp.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
    for (int tick = 0; tick < 9; ++tick) {
      reference_->recordFaceCommand(shortfall, shortfall_qp, 1.0 + 0.02 * tick, true);
    }

    EXPECT_FALSE(reference_->compactManeuverActive());
    EXPECT_EQ(reference_->diagnostics().compact_generation, 0U);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
    const auto events = reference_->drainCompactEvents();
    EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
        return event.event == "pair_resource_exhausted";
    }));
  }
}

TEST_F(MotionReferenceTest, PublicUpdateTriggersForDirectionallyExhaustedJ1J4Pair)
{
  const auto [robot_state, yaw_demand] = criticalPairExhaustedCase();
  const auto assessment = pairYawAssessment(robot_state, yaw_demand);
  ASSERT_TRUE(assessment.valid);
  ASSERT_TRUE(assessment.demand_present);
  ASSERT_TRUE(assessment.pair_bound_candidate);
  ASSERT_TRUE(assessment.probe_valid);
  EXPECT_TRUE(assessment.actual_probe_solved);
  EXPECT_FALSE(assessment.actual_probe_infeasible);
  EXPECT_DOUBLE_EQ(assessment.probe_horizon_sec, kPairYawProbeHorizonSec);
  EXPECT_TRUE(assessment.pair_bound_cause);
  EXPECT_TRUE(assessment.exhausted);
  EXPECT_LT(assessment.actual_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
  EXPECT_GE(assessment.pair_relaxed_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
  EXPECT_LE(assessment.relaxed_roll_residual_rad, 1.0 * 3.14159265358979323846 / 180.0 + 1.0e-4);
  EXPECT_LE(assessment.relaxed_elevation_residual_rad,
        1.0 * 3.14159265358979323846 / 180.0 + 1.0e-4);
  const auto selected = update(robot_state, yaw_demand);

  ASSERT_TRUE(selected.tracking);
  EXPECT_EQ(reference_->diagnostics().state, "FACE_PAIR_YAW_RESOURCE_EXHAUSTED");
  EXPECT_EQ(reference_->diagnostics().compact_reason, "pair_yaw_resource_exhausted");
  const auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "pair_resource_exhausted";
  }));
}

TEST_F(MotionReferenceRecoveryDisabledTest, ExhaustionBrakesAndLatchesAcrossLossPredictionAndReset)
{
  const auto [robot_state, exhausted_target] = criticalPairExhaustedCase();
  auto observed_target = exhausted_target;
  observed_target.header.stamp = rclcpp::Time(0, 1'000'000, RCL_ROS_TIME);
  observed_target.face_stamp = observed_target.header.stamp;
  observed_target.target_id = 7;

  const auto exhausted = update(robot_state, observed_target);

  EXPECT_FALSE(exhausted.task);
  EXPECT_FALSE(exhausted.follows_path);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto lost = target(msg::TrackingTarget::HOLD);
  lost.allow_recovery = true;
  EXPECT_FALSE(update(robot_state, lost).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  reference_->reset(true);
  EXPECT_FALSE(update(robot_state, lost, nullptr, nullptr, std::nullopt,
    kSceneRevision, 10.0).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  auto rest = target(msg::TrackingTarget::REST);
  EXPECT_FALSE(update(robot_state, rest, nullptr, nullptr, std::nullopt,
    kSceneRevision, 30.0).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  EXPECT_FALSE(update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 60.0).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);

  auto prediction = observed_target;
  prediction.face_is_prediction = true;
  prediction.header.stamp = rclcpp::Time(0, 2, RCL_ROS_TIME);
  prediction.face_stamp = rclcpp::Time(0, 1, RCL_ROS_TIME);
  EXPECT_FALSE(update(robot_state, prediction).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
}

TEST_F(MotionReferenceRecoveryDisabledTest, OnlyNewRealAccessibleFaceReleasesLimitHold)
{
  const auto [robot_state, exhausted_target] = criticalPairExhaustedCase();
  auto blocked = exhausted_target;
  blocked.header.stamp = rclcpp::Time(0, 1'000'000, RCL_ROS_TIME);
  blocked.face_stamp = blocked.header.stamp;
  blocked.target_id = 7;
  ASSERT_FALSE(update(robot_state, blocked).task);
  ASSERT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto stale_accessible = faceAlongLookDirection(robot_state);
  stale_accessible.header.stamp = blocked.header.stamp;
  stale_accessible.face_stamp = blocked.face_stamp;
  stale_accessible.target_id = 99;
  EXPECT_FALSE(update(robot_state, stale_accessible).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto still_blocked = blocked;
  still_blocked.header.stamp = rclcpp::Time(0, 2'000'000, RCL_ROS_TIME);
  still_blocked.face_stamp = still_blocked.header.stamp;
  still_blocked.target_id = 100;
  EXPECT_FALSE(update(robot_state, still_blocked).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto accessible = faceAlongLookDirection(robot_state);
  accessible.header.stamp = rclcpp::Time(0, 3'000'000, RCL_ROS_TIME);
  accessible.face_stamp = accessible.header.stamp;
  accessible.target_id = 101;
  const auto resumed = update(robot_state, accessible);

  EXPECT_TRUE(resumed.task);
  EXPECT_TRUE(resumed.tracking);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
}

TEST_F(MotionReferenceRecoveryDisabledTest, PredictionStampCannotReleaseLimitHold)
{
  const auto [robot_state, exhausted_target] = criticalPairExhaustedCase();
  auto prediction = exhausted_target;
  prediction.face_is_prediction = true;
  prediction.header.stamp = rclcpp::Time(0, 2'000'000, RCL_ROS_TIME);
  prediction.face_stamp = rclcpp::Time(0, 1'000'000, RCL_ROS_TIME);
  ASSERT_FALSE(update(robot_state, prediction).task);
  ASSERT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto repeated_source_frame = faceAlongLookDirection(robot_state);
  repeated_source_frame.header.stamp = rclcpp::Time(0, 3'000'000, RCL_ROS_TIME);
  repeated_source_frame.face_stamp = prediction.face_stamp;
  repeated_source_frame.target_id = 12;
  EXPECT_FALSE(update(robot_state, repeated_source_frame).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  repeated_source_frame.header.stamp = rclcpp::Time(0, 4'000'000, RCL_ROS_TIME);
  repeated_source_frame.face_stamp = repeated_source_frame.header.stamp;
  EXPECT_TRUE(update(robot_state, repeated_source_frame).task);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
}

TEST_F(MotionReferenceRecoveryDisabledTest, FreshFeedbackOnFaceLossCanLatchExhaustion)
{
  const auto [critical_state, critical_face] = criticalPairExhaustedCase();
  auto last_face = critical_face;
  last_face.header.stamp = rclcpp::Time(3, 0, RCL_ROS_TIME);
  last_face.face_stamp = last_face.header.stamp;
  last_face.target_id = 31;
  auto initial_state = state();

  const auto initial = update(initial_state, last_face, nullptr, nullptr,
    std::nullopt, kSceneRevision, 3.0);
  EXPECT_TRUE(initial.tracking);
  EXPECT_NE(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto lost = target(msg::TrackingTarget::HOLD);
  lost.allow_recovery = true;
  const auto braking = update(initial_state, lost, nullptr, nullptr,
    std::nullopt, kSceneRevision, 3.01);
  EXPECT_FALSE(braking.task);
  EXPECT_NE(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  const auto held = update(critical_state, lost, nullptr, nullptr,
    std::nullopt, kSceneRevision, 3.02);
  EXPECT_FALSE(held.task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  EXPECT_TRUE(reference_->diagnostics().disabled_limit_hold_latched);
}

TEST_F(MotionReferenceTest, FailedPairYawProbeCannotProveResourceAvailable)
{
  PairYawResourceAssessment assessment;
  assessment.valid = true;
  assessment.demand_present = true;
  assessment.pair_bound_candidate = true;
  assessment.probe_valid = false;
  assessment.exhausted = false;

  EXPECT_FALSE(MotionReferenceTestPeer::hasAvailablePairYawResource(assessment));

  assessment.probe_valid = true;
  assessment.actual_probe_solved = true;
  assessment.actual_available_yaw_rad = kPairYawProbeDisplacementRad;
  EXPECT_TRUE(MotionReferenceTestPeer::hasAvailablePairYawResource(assessment));
}

TEST_F(MotionReferenceRecoveryDisabledTest, ClockRewindKeepsHoldUntilFreshFaceInNewEpoch)
{
  const auto [robot_state, exhausted_target] = criticalPairExhaustedCase();
  auto exhausted = exhausted_target;
  exhausted.header.stamp = rclcpp::Time(9, 500'000'000, RCL_ROS_TIME);
  exhausted.face_stamp = exhausted.header.stamp;
  ASSERT_FALSE(update(robot_state, exhausted, nullptr, nullptr, std::nullopt,
    kSceneRevision, 10.0).task);
  ASSERT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto lost = target(msg::TrackingTarget::HOLD);
  lost.allow_recovery = true;
  EXPECT_FALSE(update(robot_state, lost, nullptr, nullptr, std::nullopt,
    kSceneRevision, 1.0).task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");

  auto accessible = faceAlongLookDirection(robot_state);
  accessible.header.stamp = rclcpp::Time(1, 100'000'000, RCL_ROS_TIME);
  accessible.face_stamp = accessible.header.stamp;
  accessible.target_id = 203;
  const auto resumed = update(robot_state, accessible, nullptr, nullptr,
    std::nullopt, kSceneRevision, 1.2);

  EXPECT_TRUE(resumed.task);
  EXPECT_TRUE(resumed.tracking);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
}

TEST_F(MotionReferenceRecoveryDisabledTest, MissingCompactProfileDoesNotBlockDisabledMode)
{
  MotionReferenceTestPeer::removeCompactProfileForTest(*reference_);
  const auto [robot_state, exhausted_target] = criticalPairExhaustedCase();

  const auto result = update(robot_state, exhausted_target);

  EXPECT_FALSE(result.task);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceRecoveryDisabledNoCompactProfileTest, StartsWithoutCompactProfile)
{
  const auto [robot_state, face] = criticalPairExhaustedCase();
  const auto result = update(robot_state, face);

  EXPECT_FALSE(result.task);
  EXPECT_EQ(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceRecoveryDisabledTest, OrdinaryRestAndSearchRemainAvailableBeforeLimitHold)
{
  const auto robot_state = state(0.4);
  const auto rest = update(robot_state, target(msg::TrackingTarget::REST));
  (void)rest;
  EXPECT_NE(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  EXPECT_GE(reference_->diagnostics().plans_requested, 1U);

  reference_->reset(false);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  (void)update(robot_state, search);
  EXPECT_NE(reference_->diagnostics().state, "DISABLED_LIMIT_HOLD");
  EXPECT_FALSE(reference_->compactManeuverActive());
}

TEST_F(MotionReferenceRecoveryDisabledTest, AvailablePairKeepsNormalFaceTracking)
{
  auto robot_state = state();
  constexpr std::array<double, 6> kJointAnglesRad{
    6.233, -0.784, -1.010, 1.311, -1.097, 4.632};
  for (std::size_t index = 0; index < kJointAnglesRad.size(); ++index) {
    robot_state.setVariablePosition(parameters_.joint_names[index], kJointAnglesRad[index]);
  }
  robot_state.update(true);
  auto face = target(msg::TrackingTarget::FACE);
  face.face.x = 0.425;
  face.face.y = -0.736;
  face.face.z = 1.680;

  const auto result = update(robot_state, face);

  ASSERT_TRUE(result.task);
  ASSERT_TRUE(result.tracking);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
}

TEST_F(MotionReferenceTest, PairYawProbeDoesNotBlameJ4WhenTheOtherSwivelCanContinue)
{
  auto robot_state = state();
  constexpr std::array<double, 6> kJointAnglesRad{
    6.233, -0.784, -1.010, 1.311, -1.097, 4.632};
  for (std::size_t index = 0; index < kJointAnglesRad.size(); ++index) {
    robot_state.setVariablePosition(parameters_.joint_names[index], kJointAnglesRad[index]);
  }
  robot_state.update(true);
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = 0.425;
  command.face.y = -0.736;
  command.face.z = 1.680;

  const auto assessment = pairYawAssessment(robot_state, command);

  ASSERT_TRUE(assessment.valid);
  ASSERT_TRUE(assessment.demand_present);
  EXPECT_TRUE(assessment.probe_valid);
  EXPECT_FALSE(assessment.exhausted);
  EXPECT_GE(assessment.actual_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
}

TEST_F(MotionReferenceTest, PublicYawReversalReleasesTheLatchedJ1Hold)
{
  auto robot_state = state();
  robot_state.setVariablePosition("joint1", 6.20);
  robot_state.setVariablePosition("joint2", 0.233);
  robot_state.setVariablePosition("joint3", -1.71);
  robot_state.setVariablePosition("joint4", 0.0168);
  robot_state.setVariablePosition("joint5", -0.8338);
  robot_state.setVariablePosition("joint6", 6.27);
  robot_state.update(true);
  Eigen::VectorXd position;
  robot_state.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};

  const auto edge_update = update(
    robot_state, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(edge_update.j1_hold_position_rad);
  const auto held_direction = static_cast<BaseJointDirection>(
    reference_->diagnostics().j1_hold_direction);
  ASSERT_NE(held_direction, BaseJointDirection::kNone);

  std::optional<msg::TrackingTarget> reversed_target;
  for (const double offset : {0.30, -0.30}) {
    auto candidate = targetWithYawOffset(robot_state, offset);
    const auto assessment = pairYawAssessment(robot_state, candidate);
    if (assessment.valid && assessment.demand_present &&
      assessment.j1.required_direction != BaseJointDirection::kNone &&
      assessment.j1.required_direction != held_direction)
    {
      reversed_target = std::move(candidate);
      break;
    }
  }
  ASSERT_TRUE(reversed_target);

  const auto reversed_update = update(
    robot_state, reversed_target, nullptr, nullptr, settled);

  EXPECT_FALSE(reversed_update.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().j1_hold_q, 0.0);
}

TEST_F(MotionReferenceTest, FreshPairAvailabilityCancelsPreparationBeforeCommit)
{
  const auto robot_state = state();
  const auto aligned_target = faceAlongLookDirection(robot_state);
  beginPreparationForResultTest(robot_state);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
  MotionReferenceTestPeer::activateResultGeneration(
    *reference_, reference_->diagnostics().compact_generation + 100U);
  const auto result = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);

  const auto refreshed = update(robot_state, aligned_target);

  EXPECT_TRUE(refreshed.tracking);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARATION_CANCELLED");
  consumePreparationResult(robot_state, result);
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "preparation_cancelled";
  }));
}

TEST_F(MotionReferenceTest, UnknownPairProbePreservesPreparationGeneration)
{
  const auto robot_state = state();
  beginPreparationForResultTest(robot_state);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  const auto generation = MotionReferenceTestPeer::pendingGeneration(*reference_);
  ASSERT_TRUE(generation);

  PairYawResourceAssessment unknown;
  unknown.valid = true;
  unknown.demand_present = true;
  unknown.pair_bound_candidate = true;
  unknown.actual_probe_solved = false;
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  Eigen::VectorXd position;
  Eigen::VectorXd velocity;
  robot_state.copyJointGroupPositions(group_, position);
  robot_state.copyJointGroupVelocities(group_, velocity);
  const JointMotionState feedback{
    position, velocity, Eigen::VectorXd::Zero(position.size())};

  const auto result = MotionReferenceTestPeer::handlePreparationWithAssessmentForTest(
    *reference_, robot_state, faceAlongLookDirection(robot_state), *planning_scene,
    now, kSceneRevision, feedback, feedback, unknown);

  EXPECT_TRUE(result.tracking);
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::pendingGeneration(*reference_), generation);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  MotionReferenceTestPeer::qualifyPairYawForIsolatedAdmissionTest(*reference_);
  const auto exhausted = MotionReferenceTestPeer::pairYawAssessment(*reference_);
  ASSERT_EQ(pairYawResourceState(exhausted), PairYawResourceState::kExhausted);
  static_cast<void>(MotionReferenceTestPeer::handlePreparationWithAssessmentForTest(
      *reference_, robot_state, faceAlongLookDirection(robot_state), *planning_scene,
      now + parameters_.control_period_sec, kSceneRevision, feedback, feedback, exhausted));

  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::pendingGeneration(*reference_), generation);
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "preparation_cancelled";
  }));

  const auto lost_target = target(msg::TrackingTarget::HOLD);
  static_cast<void>(MotionReferenceTestPeer::handlePreparationWithAssessmentForTest(
      *reference_, robot_state, lost_target, *planning_scene,
      now + 2.0 * parameters_.control_period_sec, kSceneRevision,
      feedback, feedback, unknown));
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));

  beginPreparationForResultTest(robot_state);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  static_cast<void>(MotionReferenceTestPeer::handlePreparationWithAssessmentForTest(
      *reference_, robot_state, faceAlongLookDirection(robot_state), *planning_scene,
      now + 3.0 * parameters_.control_period_sec, kSceneRevision + 1U,
      feedback, feedback, unknown));
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(MotionReferenceTestPeer::pendingGeneration(*reference_));
}

TEST_F(MotionReferenceTest, DefaultCompactGoalUsesNearestValidatedBoundedWristBranch)
{
  const auto joint6 = std::find(
    group_->getVariableNames().begin(), group_->getVariableNames().end(), "joint6");
  ASSERT_NE(joint6, group_->getVariableNames().end());
  const auto joint6_index = std::distance(group_->getVariableNames().begin(), joint6);
  const std::vector<std::pair<double, std::string>> cases{
    {limits_.upper_position[joint6_index] - 0.001, "positive_upper"},
    {limits_.lower_position[joint6_index] + 0.001, "positive_lower"},
    {3.14159265358979323846, "negative_pi"}};

  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (index > 0U) {
      reference_->reset(false);
    }
    auto start = compactGoalState();
    if (cases[index].second != "negative_pi") {
      ASSERT_TRUE(start.setToDefaultValues(group_, "compact_turn_positive_wrist"));
      start.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
      start.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    }
    start.setVariablePosition("joint6", cases[index].first);
    start.update(true);
    const auto events = commitCompactWithLifecycleFixturePath(start);
    const auto triggered = std::find_if(
      events.begin(), events.end(), [](const CompactEvent & event) {
        return event.event == "triggered";
      });
    ASSERT_NE(triggered, events.end()) << cases[index].second;
    ASSERT_TRUE(reference_->compactManeuverActive()) << cases[index].second;
    const auto goal = goalVectorFromDetails(triggered->details_json);
    ASSERT_TRUE(goal) << triggered->details_json;
    if (cases[index].second == "negative_pi") {
      EXPECT_NE(
        triggered->details_json.find("\"goal_profile\":\"compact_turn_negative_wrist\""),
        std::string::npos);
      EXPECT_NE(triggered->details_json.find("\"staged_route_enabled\":true"),
        std::string::npos);
      EXPECT_NEAR((*goal)[joint6_index], 3.14159265358979323846, 1.0e-12);
    } else {
      EXPECT_NE(
        triggered->details_json.find("\"goal_profile\":\"compact_turn_positive_wrist\""),
        std::string::npos);
      EXPECT_NE(triggered->details_json.find("\"staged_route_enabled\":true"),
        std::string::npos);
      const double expected = cases[index].second == "positive_upper" ?
        limits_.upper_position[joint6_index] - 0.02 :
        limits_.lower_position[joint6_index] + 0.02;
      EXPECT_NEAR((*goal)[joint6_index], expected, 1.0e-9);
    }
    if (index == 0U) {
      reference_->reset(true);
      auto different_winding_start = compactGoalState();
      different_winding_start.setVariablePosition(
        "joint6", limits_.lower_position[joint6_index] + 0.001);
      different_winding_start.update(true);
      JointMotionState moving_queue;
      different_winding_start.copyJointGroupPositions(group_, moving_queue.position);
      moving_queue.velocity = Eigen::VectorXd::Zero(moving_queue.position.size());
      moving_queue.acceleration = Eigen::VectorXd::Zero(moving_queue.position.size());
      moving_queue.velocity[0] = 0.02;
      (void)update(
        different_winding_start, target(msg::TrackingTarget::FACE),
        nullptr, nullptr, moving_queue);
      const auto rearmed = reference_->drainCompactEvents();
      EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
      EXPECT_TRUE(reference_->compactManeuverActive());
      EXPECT_FALSE(std::any_of(rearmed.begin(), rearmed.end(), [](const CompactEvent & event) {
          return event.event == "triggered";
      }));
    }
  }
}

TEST_F(MotionReferenceTest, PreparationFailureKeepsFaceActiveWithoutTriggeredEvent)
{
  const auto [start, initial_target] = criticalPairExhaustedCase();
  auto face_target = initial_target;
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};

  // First FACE update qualifies the pair-resource transition. Preparation is
  // entered on the following update, after the policy transition is visible.
  const auto face = update(start, face_target, nullptr, nullptr, settled);
  ASSERT_TRUE(face.task);
  ASSERT_TRUE(face.tracking);
  EXPECT_FALSE(face.actual_face_alignment);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  face_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  const auto preparation = update(start, face_target, nullptr, nullptr, settled);
  ASSERT_TRUE(preparation.task);
  ASSERT_TRUE(preparation.tracking);
  ASSERT_TRUE(preparation.actual_face_alignment);
  EXPECT_TRUE(std::isfinite(preparation.actual_face_alignment->pointing_error_rad));
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_FALSE(reference_->compactManeuverActive());

  const auto preparation_events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(preparation_events.begin(), preparation_events.end(),
    [](const CompactEvent & event) {return event.event == "preparing";}));
  EXPECT_FALSE(std::any_of(preparation_events.begin(), preparation_events.end(),
    [](const CompactEvent & event) {return event.event == "triggered";}));

  MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
  MotionReferenceTestPeer::activateResultGeneration(
    *reference_, reference_->diagnostics().compact_generation + 100U);
  const auto failure = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, false);
  consumePreparationResult(start, failure, settled);
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto failure_events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(failure_events.begin(), failure_events.end(),
    [](const CompactEvent & event) {return event.event == "triggered";}));
}

TEST_F(MotionReferenceTest, FailedDefaultCandidateAdvancesToDistinctBoundedGoal)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const std::size_t count = MotionReferenceTestPeer::candidateCount(*reference_);
  ASSERT_GE(count, 2U);
  const CompactGoal first = MotionReferenceTestPeer::candidate(*reference_, 0U);
  ASSERT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 0U);

  const auto first_failure = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, false);
  consumePreparationResult(start, first_failure);
  ASSERT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 1U);
  EXPECT_FALSE(reference_->compactManeuverActive());
  const CompactGoal second = MotionReferenceTestPeer::candidate(*reference_, 1U);
  EXPECT_NE(first.profile_id, second.profile_id);
  ASSERT_EQ(first.bounded_positions.size(), limits_.lower_position.size());
  ASSERT_EQ(second.bounded_positions.size(), limits_.lower_position.size());
  EXPECT_GT(
    (first.bounded_positions - second.bounded_positions).
    lpNorm<Eigen::Infinity>(), 1.0e-3);
  EXPECT_TRUE((first.bounded_positions.array() >= limits_.lower_position.array()).all());
  EXPECT_TRUE((first.bounded_positions.array() <= limits_.upper_position.array()).all());
  EXPECT_TRUE((second.bounded_positions.array() >= limits_.lower_position.array()).all());
  EXPECT_TRUE((second.bounded_positions.array() <= limits_.upper_position.array()).all());

  MotionReferenceTestPeer::activateResultGeneration(*reference_, 1002U);
  const auto second_success = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);
  consumePreparationResult(start, second_success);
  EXPECT_TRUE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  const auto triggered = std::find_if(events.begin(), events.end(),
      [](const CompactEvent & event) {return event.event == "triggered";});
  ASSERT_NE(triggered, events.end());
  EXPECT_NE(triggered->details_json.find("\"goal_profile\":\"" + second.profile_id + "\""),
    std::string::npos);
}

TEST_F(MotionReferenceTest, AllPreparationCandidatesExhaustOnceForUnchangedStateAndScene)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const std::size_t candidate_count = MotionReferenceTestPeer::candidateCount(*reference_);
  ASSERT_GT(candidate_count, 0U);
  for (std::size_t index = 0U; index < candidate_count; ++index) {
    ASSERT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), index);
    MotionReferenceTestPeer::activateResultGeneration(
      *reference_, 1100U + static_cast<std::uint64_t>(index));
    const auto failure = MotionReferenceTestPeer::resultForCurrentCandidate(
      *reference_, false);
    consumePreparationResult(start, failure);
    EXPECT_FALSE(reference_->compactManeuverActive());
  }
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  const auto generation = reference_->diagnostics().compact_generation;
  const auto exhausted_plan_requests = reference_->diagnostics().plans_requested;

  for (int attempt = 0; attempt < 3; ++attempt) {
    const auto face = update(start, target(msg::TrackingTarget::FACE));
    ASSERT_TRUE(face.task);
    EXPECT_FALSE(reference_->compactManeuverActive());
    EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  }
  EXPECT_EQ(reference_->diagnostics().compact_generation, generation);
  EXPECT_EQ(reference_->diagnostics().plans_requested, exhausted_plan_requests);
  const auto events = reference_->drainCompactEvents();
  EXPECT_EQ(std::count_if(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "preparing";
  }), 1);
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, EmptyDefaultCandidateSetExhaustsUntilStateOrSceneChanges)
{
  const auto start = compactGoalState();
  MotionReferenceTestPeer::clearDefaultCandidatesForTest(*reference_);
  beginPreparationForResultTest(start);

  ASSERT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::candidateCount(*reference_), 0U);
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARATION_EXHAUSTED");
  const auto first_generation = reference_->diagnostics().compact_generation;
  const auto first_plan_requests = reference_->diagnostics().plans_requested;

  beginPreparationForResultTest(start);
  EXPECT_EQ(reference_->diagnostics().compact_generation, first_generation);
  EXPECT_EQ(reference_->diagnostics().plans_requested, first_plan_requests);
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));

  constexpr std::uint64_t kChangedSceneRevision{kSceneRevision + 1U};
  beginPreparationForResultTest(
    start, std::nullopt, nullptr, nullptr, kChangedSceneRevision);
  EXPECT_EQ(reference_->diagnostics().compact_generation, first_generation + 1U);
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));

  const auto changed_state = compactGoalState(0.06);
  const auto scene_retry_generation = reference_->diagnostics().compact_generation;
  recordQualifyingFaceShortfall(changed_state);
  beginPreparationForResultTest(
    changed_state, std::nullopt, nullptr, nullptr, kChangedSceneRevision);
  EXPECT_EQ(reference_->diagnostics().compact_generation, scene_retry_generation + 1U);
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
}

TEST_F(MotionReferenceTest, StalePreparationResultsCannotCommit)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));

  const auto stale_generation = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);
  MotionReferenceTestPeer::activateResultGeneration(*reference_, 2201U);
  consumePreparationResult(start, stale_generation);
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 0U);
  EXPECT_FALSE(reference_->compactManeuverActive());

  const auto stale_scene = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);
  consumePreparationResult(start, stale_scene, std::nullopt, nullptr, nullptr,
    kSceneRevision + 1U);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, PreparationResultWithStaleStartRejectsCandidate)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  ASSERT_GE(MotionReferenceTestPeer::candidateCount(*reference_), 2U);
  auto stale_start = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  stale_start.start_positions[0] += 0.01;
  stale_start.path.front() = stale_start.start_positions;

  consumePreparationResult(start, stale_start);
  EXPECT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 1U);
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, AlignedFaceAndLegacyShortfallNeverAdmitCompact)
{
  const auto start = state();
  recordQualifyingFaceShortfall(start);
  const auto face = faceAlongLookDirection(start);
  for (int tick = 0; tick < 16; ++tick) {
    const auto measured = update(start, face);
    ASSERT_TRUE(measured.tracking);
    EXPECT_LE(measured.tracking->pointing_error_rad, kFaceTrackingErrorGuardRad);
    EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
    EXPECT_FALSE(reference_->compactManeuverActive());
  }

  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "pair_resource_exhausted" || event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, NearbyAdmissionRejectsCollisionWithinAttachDistance)
{
  const auto submitted_start = compactGoalState();
  recordQualifyingFaceShortfall(submitted_start);
  beginPreparationForResultTest(submitted_start);
  ASSERT_GE(MotionReferenceTestPeer::candidateCount(*reference_), 2U);
  const auto worker_result = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);

  auto nearby_actual = submitted_start;
  const auto submitted_position = worker_result.start_positions;
  nearby_actual.setVariablePosition(
    "joint1", nearby_actual.getVariablePosition("joint1") + 0.12);
  nearby_actual.update(true);
  Eigen::VectorXd actual_position;
  nearby_actual.copyJointGroupPositions(group_, actual_position);
  ASSERT_LT((actual_position - submitted_position).norm(), 0.20);
  const Eigen::Isometry3d obstacle_pose =
    submitted_start.getGlobalLinkTransform("link6");
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
    scene->getWorldNonConst()->addToObject(
      "preparation_bridge_obstacle", std::make_shared<shapes::Sphere>(0.002), obstacle_pose);
  }

  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto start_validation = MotionReferenceTestPeer::validatePose(
      *reference_, *planning_scene, submitted_start, submitted_position);
    const auto end_validation = MotionReferenceTestPeer::validatePose(
      *reference_, *planning_scene, nearby_actual, actual_position);
    ASSERT_TRUE(start_validation.input_valid) << start_validation.failure_reason;
    ASSERT_TRUE(end_validation.input_valid) << end_validation.failure_reason;
    EXPECT_TRUE(start_validation.unsafe) << start_validation.failure_reason;
    EXPECT_TRUE(end_validation.unsafe) << end_validation.failure_reason;
  }

  const JointMotionState nearby_queue{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(nearby_actual, worker_result, nearby_queue);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_GT(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("attachment bridge"),
    std::string::npos);
  EXPECT_NE(MotionReferenceTestPeer::candidateIndex(*reference_), 0U);
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, CertifiedSafeSuffixRebasesPathAndFollowerDoesNotReplayPrefix)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd first_waypoint = plan.start_positions;
  first_waypoint[0] += 0.15;
  Eigen::VectorXd detour_waypoint = first_waypoint;
  detour_waypoint[1] += 0.50;
  plan.path = {
    plan.start_positions, first_waypoint, detour_waypoint, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](const std::size_t segment, const double fraction) {
      // A submitted-path prefix exceeds the current cap; the selected suffix is safe.
      if (segment == 0U && fraction < 0.5) {
        return std::array<double, 3>{0.90, 0.01, 0.20};
      }
      return safeSyntheticCompactMetrics();
    });

  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.10;
  const JointMotionState actual_feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  Eigen::VectorXd queue_position = actual_position;
  queue_position[0] += 0.20;
  const JointMotionState queue_feedback{
    queue_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto selected_index = [&]() {
      double nearest_distance = std::numeric_limits<double>::infinity();
      std::size_t nearest = 0U;
      for (std::size_t index = 0U; index < plan.compact_path_samples.size(); ++index) {
        const double distance =
          (actual_position - plan.compact_path_samples[index].positions).norm();
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest = index;
        }
      }
      return nearest;
    }();
  ASSERT_GT(selected_index, 0U);
  ASSERT_EQ(plan.compact_path_samples[selected_index].original_segment_index, 0U);
  ASSERT_LE(
    (actual_position - plan.compact_path_samples[selected_index].positions).norm(), 0.20);
  EXPECT_NEAR(
    (queue_position - plan.compact_path_samples[selected_index].positions).norm(), 0.20,
    1.0e-9);
  consumePreparationResult(start, plan, queue_feedback, nullptr, nullptr,
    kSceneRevision, actual_feedback);

  ASSERT_TRUE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  const auto emitted_sample_index = triggeredAttachmentSampleIndex(events);
  ASSERT_TRUE(emitted_sample_index);
  EXPECT_EQ(*emitted_sample_index, selected_index);
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(triggered, events.end());
  EXPECT_NE(triggered->details_json.find("\"used_certificate\":true"), std::string::npos);
  EXPECT_NE(triggered->details_json.find("\"worker_full_path_metrics\""), std::string::npos);
  EXPECT_NE(triggered->details_json.find("\"follower_seed\":\"command_queue_state\""),
    std::string::npos);
  EXPECT_NE(triggered->details_json.find("\"follower_seed_positions_rad\":"),
    std::string::npos);

  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto follower = MotionReferenceTestPeer::updateFollower(
    *reference_, queue_position, now + 0.01);
  ASSERT_EQ(follower.status, JointPathStatus::kTracking);
  EXPECT_NEAR(follower.distance_from_path_rad, 0.0, 1.0e-9);
  EXPECT_DOUBLE_EQ(follower.progress_rad, 0.0);
  const auto & attachment = plan.compact_path_samples[selected_index].positions;
  EXPECT_GE(follower.reference_position[0], attachment[0] - 1.0e-9);
  EXPECT_GT((follower.reference_position - plan.path.front()).norm(), 0.05);
}

TEST_F(MotionReferenceTest, CompactAdmissionRejectsCertificateGapOrMappingMismatch)
{
  const auto make_plan = [this](const moveit::core::RobotState & start) {
      recordQualifyingFaceShortfall(start);
      beginPreparationForResultTest(start);
      auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
      Eigen::VectorXd waypoint = plan.start_positions;
      waypoint[0] += 0.15;
      plan.path = {plan.start_positions, waypoint, plan.goal_positions};
      installSyntheticCompactCertificate(
        plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
      return plan;
    };

  auto start = compactGoalState();
  auto gap_plan = make_plan(start);
  const auto removed = std::find_if(
    gap_plan.compact_path_samples.begin() + 2,
    gap_plan.compact_path_samples.end() - 1,
    [](const CompactPathSampleCertificate & sample) {
      return sample.original_segment_index == 0U && sample.fraction > 0.25 &&
             sample.fraction < 0.75;
    });
  ASSERT_NE(removed, gap_plan.compact_path_samples.end() - 1);
  gap_plan.compact_path_samples.erase(removed);
  consumePreparationResult(start, gap_plan);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);

  reference_->reset(false);
  start = compactGoalState();
  auto malformed_plan = make_plan(start);
  const auto mapped = std::find_if(
    malformed_plan.compact_path_samples.begin() + 1,
    malformed_plan.compact_path_samples.end(),
    [](const CompactPathSampleCertificate & sample) {
      return sample.original_segment_index == 0U && sample.fraction > 0.25;
    });
  ASSERT_NE(mapped, malformed_plan.compact_path_samples.end());
  mapped->positions[1] += 0.002;
  consumePreparationResult(start, malformed_plan);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, CompactAdmissionRejectsUnsafeRemainingSuffix)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd waypoint = plan.start_positions;
  waypoint[0] += 0.15;
  plan.path = {plan.start_positions, waypoint, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](const std::size_t segment, const double fraction) {
      if ((segment == 0U && fraction >= 0.75) || segment > 0U) {
        return std::array<double, 3>{0.90, 0.01, 0.20};
      }
      return safeSyntheticCompactMetrics();
    });
  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.05;
  const JointMotionState feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(start, plan, feedback, nullptr, nullptr, kSceneRevision, feedback);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_GT(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, CompactCertificateAdmissionEnforcesWindowAndFirstFourSegments)
{
  auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto queue_plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd queue_waypoint = queue_plan.start_positions;
  queue_waypoint[0] += 0.15;
  queue_plan.path = {queue_plan.start_positions, queue_waypoint, queue_plan.goal_positions};
  installSyntheticCompactCertificate(
    queue_plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
  Eigen::VectorXd queue_position = queue_plan.start_positions;
  queue_position[0] += 0.37;
  const JointMotionState queue_outside_window{
    queue_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(
    start, queue_plan, queue_outside_window, nullptr, nullptr, kSceneRevision,
    JointMotionState{queue_plan.start_positions, Eigen::VectorXd::Zero(6),
      Eigen::VectorXd::Zero(6)});
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);

  reference_->reset(false);
  start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.50;
  auto p1 = plan.start_positions;
  auto p2 = plan.start_positions;
  auto p3 = plan.start_positions;
  auto p4 = plan.start_positions;
  p1[3] += 0.05;
  p2[3] += 0.10;
  p3[3] += 0.15;
  p4[3] += 0.20;
  auto p5 = actual_position;
  plan.path = {plan.start_positions, p1, p2, p3, p4, p5, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
  const JointMotionState feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(start, plan, feedback, nullptr, nullptr, kSceneRevision, feedback);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_GT(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, CompactCertificateAdmissionChoosesEarliestTiedSample)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd first_waypoint = plan.start_positions;
  first_waypoint[0] += 0.08;
  plan.path = {plan.start_positions, first_waypoint, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.025;
  const JointMotionState feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto expected_earliest = static_cast<std::size_t>(2U);
  ASSERT_NEAR(
    (actual_position - plan.compact_path_samples[expected_earliest].positions).norm(),
    (actual_position - plan.compact_path_samples[expected_earliest + 1U].positions).norm(),
    1.0e-12);
  consumePreparationResult(start, plan, feedback, nullptr, nullptr, kSceneRevision, feedback);

  ASSERT_TRUE(reference_->compactManeuverActive());
  const auto emitted_sample_index =
    triggeredAttachmentSampleIndex(reference_->drainCompactEvents());
  ASSERT_TRUE(emitted_sample_index);
  EXPECT_EQ(*emitted_sample_index, expected_earliest);
}

TEST_F(MotionReferenceTest, WorkerEndpointWithinGoalToleranceIsAdmitted)
{
  const auto start = compactGoalState(0.05);
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true, 0.005);
  ASSERT_NEAR(
    (plan.path.back() - plan.goal_positions).lpNorm<Eigen::Infinity>(), 0.005, 1.0e-12);

  consumePreparationResult(start, plan);

  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
}

TEST_F(MotionReferenceTest, WorkerEndpointOutsideGoalToleranceIsRejected)
{
  const auto start = compactGoalState(0.05);
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true, 0.1);
  ASSERT_NEAR(
    (plan.path.back() - plan.goal_positions).lpNorm<Eigen::Infinity>(), 0.1, 1.0e-12);

  consumePreparationResult(start, plan);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 1U);
  EXPECT_EQ(reference_->diagnostics().plans_rejected, 1U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("incomplete path"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, TargetLossOrChangeCancelsOnlyUncommittedPreparation)
{
  auto [start, first_target] = criticalPairExhaustedCase();
  first_target.target_id = 17;
  (void)update(start, first_target);
  first_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  (void)update(start, first_target);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  auto changed_target = first_target;
  changed_target.target_id = 18;
  (void)update(start, changed_target);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());

  reference_->reset(false);
  auto healthy_case = criticalPairExhaustedCase();
  start = std::move(healthy_case.first);
  first_target = std::move(healthy_case.second);
  first_target.target_id = 17;
  (void)update(start, first_target);
  first_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  (void)update(start, first_target);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  auto healthy_loss = target(msg::TrackingTarget::HOLD);
  healthy_loss.allow_recovery = true;
  (void)update(start, healthy_loss);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());

  reference_->reset(false);
  start = compactGoalState(0.05);
  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  ASSERT_TRUE(reference_->compactManeuverActive());
  (void)update(start, healthy_loss);
  EXPECT_TRUE(reference_->compactManeuverActive());
  changed_target.target_id = 19;
  (void)update(start, changed_target);
  EXPECT_TRUE(reference_->compactManeuverActive());
}

TEST_F(MotionReferenceCustomCompactGoalTest, ExplicitCompactGoalIsNotBranchSelectedOrStaged)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  ASSERT_EQ(MotionReferenceTestPeer::candidateCount(*reference_), 1U);
  const auto configured_candidate = MotionReferenceTestPeer::candidate(*reference_, 0U);
  EXPECT_EQ(configured_candidate.profile_id, "configured_goal_override");
  EXPECT_DOUBLE_EQ(configured_candidate.bounded_positions[5], 0.7);
  EXPECT_FALSE(configured_candidate.path_requirements.staged_route.has_value());

  const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  consumePreparationResult(start, plan);
  const auto events = reference_->drainCompactEvents();
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(triggered, events.end());
  EXPECT_NE(
    triggered->details_json.find("\"goal_profile\":\"configured_goal_override\""),
    std::string::npos);
  EXPECT_EQ(triggered->details_json.find("\"staged_route_enabled\":true"),
    std::string::npos);
  const auto goal = goalVectorFromDetails(triggered->details_json);
  ASSERT_TRUE(goal);
  EXPECT_DOUBLE_EQ((*goal)[5], 0.7);
}

TEST_F(MotionReferenceInvalidCompactMountTest, DefaultCandidatesRejectInvalidScreenMount)
{
  auto safe_mount_state = nearScreenPoleState();
  safe_mount_state.setVariablePosition(
    "joint5", safe_mount_state.getVariablePosition("joint5") + 0.15);
  safe_mount_state.update(true);
  const auto safe_mount_pose = evaluateCompactUprightPose(
    safe_mount_state.getGlobalLinkTransform("test_camera_optical").linear());
  ASSERT_TRUE(safe_mount_pose);
  ASSERT_LE(std::abs(safe_mount_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  const auto events = commitCompactWithLifecycleFixturePath(safe_mount_state);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "cancelled";
  }));
}

TEST(TrackingOrientationPriority, FaceTaskKeepsThreeExactOrientationRowsPrimary)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  TrackingVelocityTaskConfig config;
  config.position_deadband_m = 0.0;
  config.pointing_deadband_rad = 0.0;
  config.roll_deadband_rad = 0.0;
  const auto task = makeTrackingVelocityTask(
    current, jacobian, Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Vector3d(1.0, 0.3, 0.2), Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(task);
  EXPECT_EQ(task->task.primary_matrix.rows(), 3);
  EXPECT_EQ(task->task.primary_matrix.cols(), 6);
  EXPECT_TRUE(task->has_face_primary_roll);
}

TEST_F(MotionReferenceTest, HoldAndMissingTargetProduceNoTask)
{
  const auto robot_state = state();
  auto held = target(msg::TrackingTarget::HOLD);
  EXPECT_FALSE(update(robot_state, held).task);
  EXPECT_FALSE(update(robot_state, std::nullopt).task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, RestWithRecoveryDoesNotArmFaceRollGuard)
{
  const auto robot_state = state();
  auto rest_with_recovery = target(msg::TrackingTarget::REST);
  rest_with_recovery.allow_recovery = true;
  (void)update(robot_state, rest_with_recovery);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
}

TEST_F(MotionReferenceTest, RestCommandValidatorEnforcesEdgeWithoutFaceGuard)
{
  const auto robot_state = state();
  (void)update(robot_state, target(msg::TrackingTarget::REST));
  ASSERT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  Eigen::VectorXd position;
  robot_state.copyJointGroupPositions(group_, position);
  Eigen::VectorXd edge_invalid_position;
  bool found_edge_violation = false;
  for (Eigen::Index joint = 0; joint < position.size() && !found_edge_violation; ++joint) {
    for (const double offset : {-0.60, -0.30, 0.30, 0.60}) {
      auto candidate = robot_state;
      candidate.setVariablePosition(
        parameters_.joint_names[static_cast<std::size_t>(joint)], position[joint] + offset);
      candidate.update(true);
      if (!evaluateHorizontalEdgeUpright(
          candidate.getGlobalLinkTransform("monitor_control_frame").linear(),
          kCompactRollCapTestRad))
      {
        candidate.copyJointGroupPositions(group_, edge_invalid_position);
        found_edge_violation = true;
        break;
      }
    }
  }
  ASSERT_TRUE(found_edge_violation);
  std::string violation;
  EXPECT_FALSE(reference_->validateCommandPosition(
      robot_state, edge_invalid_position, &violation));
  EXPECT_EQ(violation, "screen roll");
}

TEST_F(MotionReferenceTest, ReferenceGuardBlendUsesValidatedBrakeFallback)
{
  constexpr double row_upper = 2.8454501015796202e-5;
  const double period = parameters_.control_period_sec;
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  const auto make_component = [&]() {
      auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
      std::vector<PositionBoundSet> bounds;
      bounds.reserve(parameters_.joint_names.size());
      for (const auto & name : parameters_.joint_names) {
        const auto & variable_bounds = model_->getVariableBounds(name);
        const PositionInterval physical{
          variable_bounds.min_position_, variable_bounds.max_position_};
        bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
      }
      CollisionAwareServoComponentTestPeer::configure(
        *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
      HierarchicalVelocityQpConfig qp_config;
      qp_config.period_sec = period;
      qp_config.solution_feasibility_tolerance = 2.0e-5;
      CollisionAwareServoComponentTestPeer::installVelocityQp(*component, qp_config);
      return component;
    };

  const auto make_result = [](const Eigen::VectorXd & velocity) {
      QpResult result;
      result.status = QpStatus::kSolved;
      result.primary_status = QpStageStatus::kSolved;
      result.joint_velocity = velocity;
      result.achieved_primary_velocity = Eigen::VectorXd::Constant(6, 42.0);
      return result;
    };

  const auto run_stationary_edge = [&](const bool mirrored) {
      Eigen::VectorXd q(6);
      q << 6.0850690097544176, 0.33651671407004213, -1.5722575822551443,
        mirrored ? 0.09426871707457246 : -0.094266091778429775,
        -1.0603541089415993, 6.2831852993649591;
      Eigen::VectorXd tracking_velocity(6);
      tracking_velocity << 7.6126117540317983e-12, -0.0019999984779851864,
        -0.0019999959400758623,
        mirrored ? -0.0001007755486928388 : 0.0001007755486928388,
        -0.002, 7.8146225091581575e-7;
      auto robot_state = state();
      robot_state.setJointGroupPositions(group_, q);
      robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
      robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
      robot_state.update(true);
      const auto & rotation = robot_state.getGlobalLinkTransform(
        "monitor_control_frame").linear();
      const auto initial_edge = evaluateHorizontalEdgeUpright(rotation, kCompactRollCapTestRad);
      ASSERT_TRUE(initial_edge);
      EXPECT_NEAR(initial_edge->signed_edge_sine,
        mirrored ? 0.08889359655317948 : -0.08889359655317948, 2.0e-12);

      JointMotionState motion{q, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
      moveit_servo::KinematicState current(6);
      current.joint_names = parameters_.joint_names;
      current.positions = q;
      current.velocities.setZero(6);
      current.accelerations.setZero(6);
      current.time_stamp = rclcpp::Time(1, 0, RCL_ROS_TIME);

      auto reference = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      MotionReferenceTestPeer::qualifyPairYawForIsolatedAdmissionTest(*reference);
      auto component = make_component();
      CollisionAwareServoComponentTestPeer::installMotionReference(
        *component, std::move(reference));

      const Eigen::VectorXd zero = Eigen::VectorXd::Zero(6);
      std::string reason;
      CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
      const auto brake_anchor = CollisionAwareServoComponentTestPeer::validateCandidateWithKind(
        *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
        motion, zero, *scene, reason, rejection_kind);
      ASSERT_TRUE(brake_anchor) << reason;
      ASSERT_TRUE(brake_anchor->braking_tail.command_available());
      for (const auto & point : brake_anchor->braking_tail.points) {
        auto tail_state = robot_state;
        tail_state.setJointGroupPositions(group_, point.position);
        tail_state.update(true);
        EXPECT_TRUE(evaluateHorizontalEdgeUpright(
          tail_state.getGlobalLinkTransform("monitor_control_frame").linear(),
          kCompactRollCapTestRad));
      }

      const QpResult tracking_result = make_result(tracking_velocity);
      const QpResult brake_result = make_result(zero);
      const auto actual_candidate =
        CollisionAwareServoComponentTestPeer::validateCandidateWithKind(
        *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
        motion, tracking_velocity, *scene, reason, rejection_kind);
      EXPECT_FALSE(actual_candidate);
      EXPECT_EQ(rejection_kind, CandidateRejectionKind::kReferenceGuard);
      EXPECT_NE(reason.find("screen roll"), std::string::npos) << reason;

      Eigen::VectorXd row(6);
      row << 0.0, -0.030948101920223375, -0.030948101920223375,
        mirrored ? 0.94019902711327696 : -0.94019902711327696,
        2.7462953445522853e-06, -0.74625291944103944;
      std::vector<LinearVelocityConstraint> constraints;
      constraints.push_back(LinearVelocityConstraint{row, -std::numeric_limits<double>::infinity(),
          row_upper});
      Eigen::VectorXd j1 = Eigen::VectorXd::Zero(6);
      j1[0] = 1.0;
      constraints.push_back(LinearVelocityConstraint{j1, 0.0, 0.0});
      const Eigen::MatrixXd primary = Eigen::MatrixXd::Identity(6, 6);

      std::optional<ValidatedCommand> selected_command = *brake_anchor;
      QpResult selected_qp = brake_result;
      ASSERT_TRUE(CollisionAwareServoComponentTestPeer::tryReferenceGuardBlend(
        *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
        motion, *scene, tracking_result, brake_result, primary, constraints,
        selected_command, selected_qp));
      ASSERT_TRUE(selected_command);
      EXPECT_NEAR(selected_command->next_state.velocities[1],
        0.5 * tracking_velocity[1], 1.0e-12);
      EXPECT_TRUE(selected_qp.joint_velocity.isApprox(
        selected_command->next_state.velocities, 1.0e-12));
      EXPECT_TRUE(selected_qp.achieved_primary_velocity.isApprox(
        primary * selected_qp.joint_velocity, 1.0e-12));
      ASSERT_TRUE(selected_command->braking_tail.command_available());
      for (const auto & point : selected_command->braking_tail.points) {
        auto tail_state = robot_state;
        tail_state.setJointGroupPositions(group_, point.position);
        tail_state.update(true);
        EXPECT_TRUE(evaluateHorizontalEdgeUpright(
          tail_state.getGlobalLinkTransform("monitor_control_frame").linear(),
          kCompactRollCapTestRad));
      }

      // A true arbitrary-row violation is rejected; an arithmetic-sized J1
      // equality residual is accepted by the configured QP feasibility budget.
      constraints.back().upper_bound = 0.0;
      constraints.back().lower_bound = 0.0;
      const auto row_checked = CollisionAwareServoComponentTestPeer::tryReferenceGuardBlend(
        *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
        motion, *scene, tracking_result, brake_result, primary, constraints,
        selected_command, selected_qp);
      EXPECT_TRUE(row_checked);
      constraints.back().upper_bound = -1.0e-3;
      constraints.back().lower_bound = -1.0e-3;
      const auto true_row_violation =
        CollisionAwareServoComponentTestPeer::tryReferenceGuardBlend(
        *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
        motion, *scene, tracking_result, brake_result, primary, constraints,
        selected_command, selected_qp);
      EXPECT_FALSE(true_row_violation);
      EXPECT_TRUE(selected_qp.joint_velocity.isApprox(
        selected_command->next_state.velocities, 1.0e-12));
    };

  run_stationary_edge(false);
  run_stationary_edge(true);

  // Moving captured case: neither .5 nor .25 blend passes the complete edge
  // validator, so the already validated braking command/result must be retained.
  Eigen::VectorXd q(6), source_velocity(6), source_acceleration(6);
  q << 6.0850690097615496, 0.26061321747638477, -1.672651119140903,
    -0.080434964759384275, -0.81012443513387056, 6.2831842066190635;
  source_velocity << 4.3563188096923914e-12, 0.010579037993794787,
    0.011621537267429661, -0.12853694962634898, -0.0040528061837056557,
    -0.00011005605197030206;
  source_acceleration << -2.057227541799855e-7, -0.18929180338341683,
    -0.081908906483246252, -0.087596144989413038, -0.086683141661974925,
    -0.011005605228716958;
  Eigen::VectorXd tracking_velocity(6), brake_velocity(6);
  tracking_velocity << 3.2164113269588461e-12, 0.0093151601777144571,
    0.012802448202597199, -0.1314127705737104, -0.006918970646780021,
    0.00011005605172901092;
  brake_velocity << 0.0, 0.0066861199599606187, 0.0088024482025971986,
    -0.12741291107624311, -0.0029196376003254049, 0.0;
  auto robot_state = state();
  robot_state.setJointGroupPositions(group_, q);
  robot_state.setJointGroupVelocities(group_, source_velocity);
  robot_state.setJointGroupAccelerations(group_, source_acceleration);
  robot_state.update(true);
  JointMotionState motion{q, source_velocity, source_acceleration};
  moveit_servo::KinematicState current(6);
  current.joint_names = parameters_.joint_names;
  current.positions = q;
  current.velocities = source_velocity;
  current.accelerations = source_acceleration;
  current.time_stamp = rclcpp::Time(1, 0, RCL_ROS_TIME);
  auto reference = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  MotionReferenceTestPeer::qualifyPairYawForIsolatedAdmissionTest(*reference);
  auto component = make_component();
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference));
  std::string reason;
  CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
  const auto brake_anchor = CollisionAwareServoComponentTestPeer::validateCandidateWithKind(
    *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, brake_velocity, *scene, reason, rejection_kind);
  ASSERT_TRUE(brake_anchor) << reason;
  const QpResult tracking_result = make_result(tracking_velocity);
  const QpResult brake_result = make_result(brake_velocity);
  const Eigen::MatrixXd primary = Eigen::MatrixXd::Identity(6, 6);
  std::vector<LinearVelocityConstraint> constraints;
  Eigen::VectorXd j1 = Eigen::VectorXd::Zero(6);
  j1[0] = 1.0;
  constraints.push_back(LinearVelocityConstraint{j1, 0.0, 0.0});
  std::optional<ValidatedCommand> selected_command = *brake_anchor;
  QpResult selected_qp = brake_result;
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::tryReferenceGuardBlend(
    *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, *scene, tracking_result, brake_result, primary, constraints,
    selected_command, selected_qp));
  ASSERT_TRUE(selected_command);
  EXPECT_TRUE(selected_command->next_state.velocities.isApprox(brake_velocity, 1.0e-12));
  EXPECT_TRUE(selected_qp.joint_velocity.isApprox(brake_velocity, 1.0e-12));
  EXPECT_TRUE(selected_qp.achieved_primary_velocity.isApprox(
    brake_result.achieved_primary_velocity, 1.0e-12));

  // An unavailable brake anchor or a non-finite tracking result cannot replace
  // that validated fallback, even if blend attempts are requested.
  QpResult unavailable_brake = brake_result;
  unavailable_brake.status = QpStatus::kInfeasible;
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::tryReferenceGuardBlend(
    *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, *scene, tracking_result, unavailable_brake, primary, constraints,
    selected_command, selected_qp));
  QpResult non_finite_tracking = tracking_result;
  non_finite_tracking.joint_velocity[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::tryReferenceGuardBlend(
    *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, *scene, non_finite_tracking, brake_result, primary, constraints,
    selected_command, selected_qp));
  EXPECT_TRUE(selected_command->next_state.velocities.isApprox(brake_velocity, 1.0e-12));
  EXPECT_TRUE(selected_qp.joint_velocity.isApprox(brake_velocity, 1.0e-12));
}

TEST_F(MotionReferenceTest, CapturedLowFaceEdgeRecoveryPassesReconstructedFullValidation)
{
  // Reconstruction from the low-target capture at sim 108.0 s. Runtime did not
  // save an exact candidate/constraint snapshot; task and geometry are rebuilt
  // from this q/v/a, the captured FACE target, and the test URDF/PlanningScene.
  Eigen::VectorXd q(6), velocity(6), acceleration(6);
  q << 6.105605521844887, -1.1560093238337685, 0.11119659727275859,
    6.180221095151599, -1.1289119056618604, 6.283185300429141;
  velocity << -1.376779616962115e-10, -0.0007937360876465148,
    -0.0004159785582391676, 0.00011015559800572937,
    -0.0005105456047977963, 1.4780985540102643e-07;
  acceleration << 3.3528666930505492e-09, -0.0233380009988892,
    0.01933955380949408, 0.00045073370370466283,
    0.012943688271018083, 5.694214006262612e-06;
  JointMotionState motion{q, velocity, acceleration};

  auto robot_state = state();
  robot_state.setJointGroupPositions(group_, q);
  robot_state.setJointGroupVelocities(group_, velocity);
  robot_state.setJointGroupAccelerations(group_, acceleration);
  robot_state.update(true);

  msg::TrackingTarget command;
  command.header.frame_id = "world";
  command.header.stamp.sec = 108;
  command.header.stamp.nanosec = 9'000'000U;
  command.face_stamp = command.header.stamp;
  command.mode = msg::TrackingTarget::FACE;
  command.target_id = 1U;
  command.face.x = 0.26974412934916037;
  command.face.y = 0.8060633378846003;
  command.face.z = 1.52;
  command.pose.position.x = 0.1605004414995133;
  command.pose.position.y = 0.4796157081127313;
  command.pose.position.z = 1.3162900059365332;
  command.pose.orientation.x = 0.1197094468216164;
  command.pose.orientation.y = -0.14000605955100084;
  command.pose.orientation.z = 0.6387445722811635;
  command.pose.orientation.w = 0.7470430529842425;

  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  auto reference = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  MotionReferenceTestPeer::setFaceRollGuardArmedForTest(*reference);
  const auto reference_result = reference->update(
    robot_state, command, *scene, 108.009, kSceneRevision, motion, 108.009, motion);
  ASSERT_TRUE(reference_result.task);
  ASSERT_TRUE(reference_result.tracking);
  ASSERT_TRUE(reference_result.tracking->has_face_primary_roll);
  ASSERT_FALSE(reference_result.follows_path);

  const auto recovery_task = makeFaceEdgeOnlyRecoveryTask(
    *reference_result.task, reference_result.tracking->has_face_primary_roll);
  ASSERT_TRUE(recovery_task);
  ASSERT_EQ(recovery_task->primary_matrix.rows(), 1);

  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  CollisionAwareServoComponentTestPeer::installVelocityQp(*component, qp_config);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference));

  const auto collision = CollisionAwareServoComponentTestPeer::buildCollisionConstraints(
    *component, *scene, robot_state, motion);
  ASSERT_TRUE(collision.diagnostics.valid) << collision.diagnostics.failure_reason;
  auto constraints = collision.constraints;
  constraints.insert(
    constraints.end(), reference_result.compact_radius_constraints.begin(),
    reference_result.compact_radius_constraints.end());
  constraints.insert(
    constraints.end(), reference_result.compact_roll_constraints.begin(),
    reference_result.compact_roll_constraints.end());
  constraints.insert(
    constraints.end(), reference_result.compact_pitch_constraints.begin(),
    reference_result.compact_pitch_constraints.end());

  const QpResult recovery = CollisionAwareServoComponentTestPeer::solveTask(
    *component, *recovery_task, motion, constraints);
  ASSERT_TRUE(recovery.command_available());
  moveit_servo::KinematicState current(6);
  current.joint_names = parameters_.joint_names;
  current.positions = q;
  current.velocities = velocity;
  current.accelerations = acceleration;
  current.time_stamp = rclcpp::Time(108, 9'000'000U, RCL_ROS_TIME);
  std::string reason;
  CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
  const auto accepted = CollisionAwareServoComponentTestPeer::validateCandidateWithKind(
    *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, recovery.joint_velocity, *scene, reason, rejection_kind);
  ASSERT_TRUE(accepted) << reason;
  EXPECT_TRUE(accepted->braking_tail.command_available());
  EXPECT_EQ(rejection_kind, CandidateRejectionKind::kOther);
}

TEST_F(MotionReferenceTest, SyntheticExecutionBudgetConstrainsCandidateAndFullTail)
{
  constexpr double execution_error_rad = 0.02;
  const auto robot_state = state();
  Eigen::VectorXd q;
  robot_state.copyJointGroupPositions(group_, q);
  const double working_upper = q[0] + execution_error_rad + 0.0015;

  const auto make_bounds = [&](const double error_rad) {
      std::vector<PositionBoundSet> bounds;
      bounds.reserve(parameters_.joint_names.size());
      for (std::size_t index = 0; index < parameters_.joint_names.size(); ++index) {
        const auto & variable_bounds = model_->getVariableBounds(parameters_.joint_names[index]);
        const PositionInterval physical{
          variable_bounds.min_position_, variable_bounds.max_position_};
        PositionInterval working = physical;
        if (index == 0U) {
          working = PositionInterval{q[0] - 0.5, working_upper};
        }
        const auto bound = makePositionBoundSet(
          physical, physical, working, Radians{error_rad});
        if (!bound.has_value()) {
          return std::vector<PositionBoundSet>{};
        }
        bounds.push_back(*bound);
      }
      return bounds;
    };
  const auto build_component = [&](const double error_rad) {
      const auto bounds = make_bounds(error_rad);
      if (bounds.size() != parameters_.joint_names.size()) {
        return std::shared_ptr<CollisionAwareServoComponent>{};
      }
      JointMotionLimits bounded_limits = limits_;
      for (std::size_t index = 0; index < bounds.size(); ++index) {
        const Eigen::Index axis = static_cast<Eigen::Index>(index);
        bounded_limits.lower_position[axis] = bounds[index].wire.lower_rad;
        bounded_limits.upper_position[axis] = bounds[index].wire.upper_rad;
      }
      CollisionConstraintConfig config = collision_config_;
      config.tracking_position_error_bound_rad = error_rad;
      auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
      CollisionAwareServoComponentTestPeer::configure(
        *component, monitor_, group_, parameters_.joint_names,
        bounded_limits, config, bounds);
      const bool synthetic_budget_set =
        CollisionAwareServoComponentTestPeer::setSyntheticExecutionBudget(
        *component, error_rad, error_rad * 0.25, error_rad * 0.25);
      if (error_rad == 0.0) {
        CollisionAwareServoComponentTestPeer::clearExecutionBudgetForTest(*component);
      } else if (!synthetic_budget_set) {
        return std::shared_ptr<CollisionAwareServoComponent>{};
      }
      return component;
    };

  auto no_budget = build_component(0.0);
  auto with_budget = build_component(execution_error_rad);
  ASSERT_TRUE(no_budget);
  ASSERT_TRUE(with_budget);
  const auto no_budget_bounds = make_bounds(0.0);
  const auto budget_bounds = make_bounds(execution_error_rad);
  ASSERT_EQ(no_budget_bounds.size(), 6U);
  ASSERT_EQ(budget_bounds.size(), 6U);
  EXPECT_GT(no_budget_bounds[0].wire.upper_rad, budget_bounds[0].wire.upper_rad);

  Eigen::VectorXd candidate_velocity = Eigen::VectorXd::Zero(6);
  candidate_velocity[0] = 0.1;
  Eigen::VectorXd source_velocity = Eigen::VectorXd::Zero(6);
  source_velocity[0] = 0.09;
  Eigen::VectorXd source_acceleration = Eigen::VectorXd::Zero(6);
  source_acceleration[0] = 0.8;
  moveit_servo::KinematicState current(6);
  current.joint_names = parameters_.joint_names;
  current.positions = q;
  current.velocities = source_velocity;
  current.accelerations = source_acceleration;
  current.time_stamp = rclcpp::Time(1, 0, RCL_ROS_TIME);
  const JointMotionState motion{q, source_velocity, source_acceleration};

  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  std::string reason;
  const auto accepted_without_budget =
    CollisionAwareServoComponentTestPeer::validateCandidate(
    *no_budget, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, candidate_velocity, *scene, reason);
  ASSERT_TRUE(accepted_without_budget) << reason;
  ASSERT_TRUE(accepted_without_budget->braking_tail.command_available());
  const double no_budget_upper = no_budget_bounds[0].wire.upper_rad;
  for (const auto & point : accepted_without_budget->braking_tail.points) {
    EXPECT_LE(point.position[0], no_budget_upper);
  }

  const auto accepted_candidate = CollisionAwareServoComponentTestPeer::validateCandidate(
    *with_budget, std::make_shared<moveit::core::RobotState>(robot_state), current,
    motion, candidate_velocity, *scene, reason);
  EXPECT_FALSE(accepted_candidate);
  EXPECT_NE(reason.find("pre-published braking tail is unavailable"), std::string::npos)
    << reason;

  // The same validated synthetic E leaves a stationary interior command and
  // its complete tail unchanged; only the reserved boundary region differs.
  const Eigen::VectorXd zero_velocity = Eigen::VectorXd::Zero(6);
  const JointMotionState stationary{q, zero_velocity, zero_velocity};
  current.velocities = zero_velocity;
  current.accelerations = zero_velocity;
  reason.clear();
  const auto interior_without_budget =
    CollisionAwareServoComponentTestPeer::validateCandidate(
    *no_budget, std::make_shared<moveit::core::RobotState>(robot_state), current,
    stationary, zero_velocity, *scene, reason);
  ASSERT_TRUE(interior_without_budget) << reason;
  reason.clear();
  const auto interior_with_budget =
    CollisionAwareServoComponentTestPeer::validateCandidate(
    *with_budget, std::make_shared<moveit::core::RobotState>(robot_state), current,
    stationary, zero_velocity, *scene, reason);
  ASSERT_TRUE(interior_with_budget) << reason;
  ASSERT_EQ(interior_without_budget->braking_tail.points.size(),
    interior_with_budget->braking_tail.points.size());
  for (std::size_t point_index = 0;
    point_index < interior_without_budget->braking_tail.points.size(); ++point_index)
  {
    EXPECT_TRUE(interior_without_budget->braking_tail.points[point_index].position.isApprox(
        interior_with_budget->braking_tail.points[point_index].position, 0.0));
  }
}

TEST_F(MotionReferenceTest, HighPitchRestCandidateAndBrakeTailPassComponentAdmission)
{
  // These measured xArm6 states from compact_v9_path_no_face_2s_v2 form a
  // moving REST_PATH segment across the old 84.26-degree cutoff. Its sampled
  // FK stays inside the B edge cap and the assembly radius remains below 0.40 m.
  Eigen::VectorXd q(6);
  q << -0.001407423105479485, -0.2396943388405636, -1.8878559336092842,
    -0.09467427676176186, -1.0717125957575526, 2.1930010470536283;
  Eigen::VectorXd rest_path_end(6);
  rest_path_end << -0.001402187836476492, -0.24318210725422562,
    -1.8592154327180823, -0.09275387139607832, -1.0749214159045504,
    2.183316122394598;
  auto robot_state = state();
  robot_state.setJointGroupPositions(group_, q);
  robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  robot_state.update(true);
  const auto & screen_rotation =
    robot_state.getGlobalLinkTransform("monitor_control_frame").linear();
  ASSERT_TRUE(evaluateHorizontalEdgeUpright(screen_rotation, kCompactRollCapTestRad));
  for (int step = 0; step <= 20; ++step) {
    const double fraction = static_cast<double>(step) / 20.0;
    auto sample = robot_state;
    sample.setJointGroupPositions(group_, q + fraction * (rest_path_end - q));
    sample.update(true);
    EXPECT_TRUE(evaluateHorizontalEdgeUpright(
      sample.getGlobalLinkTransform("monitor_control_frame").linear(),
      kCompactRollCapTestRad));
    EXPECT_LE(
      MotionReferenceTestPeer::assemblyRadiusForTest(*reference_, sample), 0.40 + 1.0e-6);
  }
  const double start_pitch = std::atan2(
    std::abs(screen_rotation.col(0).z()),
    std::hypot(screen_rotation.col(0).x(), screen_rotation.col(0).y()));
  EXPECT_GT(start_pitch, 84.26 * std::acos(-1.0) / 180.0);
  std::vector<Eigen::VectorXd> rest_path{q, rest_path_end};
  ASSERT_TRUE(MotionReferenceTestPeer::installRestFollowerPathForTest(
      *reference_, robot_state, rest_path, 1.0));
  const auto rest = update(
    robot_state, target(msg::TrackingTarget::REST), nullptr, nullptr, std::nullopt,
    kSceneRevision, 1.01);
  ASSERT_TRUE(rest.follows_path);
  ASSERT_TRUE(rest.task);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);

  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));
  CollisionAwareServoComponentTestPeer::publishHardwareSample(*component, q, 11U, 1U);

  moveit_servo::KinematicState current(6);
  current.joint_names = parameters_.joint_names;
  current.positions = q;
  current.velocities = Eigen::VectorXd::Zero(6);
  current.accelerations = Eigen::VectorXd::Zero(6);
  current.time_stamp = component->now();
  const JointMotionState current_motion{
    q, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  std::vector<control::LinearVelocityConstraint> constraints;
  constraints.insert(
    constraints.end(), rest.compact_radius_constraints.begin(),
    rest.compact_radius_constraints.end());
  constraints.insert(
    constraints.end(), rest.compact_roll_constraints.begin(),
    rest.compact_roll_constraints.end());
  constraints.insert(
    constraints.end(), rest.compact_pitch_constraints.begin(),
    rest.compact_pitch_constraints.end());
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult qp = solver.solve(*rest.task, current_motion, limits_, constraints);
  ASSERT_TRUE(qp.command_available());
  EXPECT_GT(qp.joint_velocity.norm(), 1.0e-5)
    << "expected a moving REST_PATH reference at the measured high-pitch state";

  std::string reason;
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
  const auto accepted = CollisionAwareServoComponentTestPeer::validateCandidate(
    *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
    current_motion, qp.joint_velocity, *planning_scene, reason);
  ASSERT_TRUE(accepted.has_value()) << reason;
  ASSERT_TRUE(accepted->braking_tail.command_available());
  ASSERT_FALSE(accepted->braking_tail.points.empty());
  for (const auto & point : accepted->braking_tail.points) {
    auto tail_state = robot_state;
    tail_state.setJointGroupPositions(group_, point.position);
    tail_state.update(true);
    EXPECT_TRUE(evaluateHorizontalEdgeUpright(
      tail_state.getGlobalLinkTransform("monitor_control_frame").linear(),
      kCompactRollCapTestRad));
  }
}

TEST_F(MotionReferenceTest, ExpiredRearmTimeoutDoesNotResetNewSettlingSamples)
{
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));
  CollisionAwareServoComponentTestPeer::installStatusPublisher(*component);
  CollisionAwareServoComponentTestPeer::prepareExpiredRearmDeadline(*component);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::rearmDeadlineExpired(*component));
  const auto deadline = CollisionAwareServoComponentTestPeer::rearmDeadline(*component);
  ASSERT_TRUE(deadline.has_value());

  std::string reason;
  monitor_->startStateMonitor("/joint_states");
  auto joint_state_publisher = node_->create_publisher<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::QoS(10).reliable());
  Eigen::VectorXd hold(static_cast<Eigen::Index>(group_->getVariableCount()));
  state().copyJointGroupPositions(group_, hold);
  const auto publish_joint_state = [&]() {
      sensor_msgs::msg::JointState message;
      message.header.stamp = node_->now();
      message.name = parameters_.joint_names;
      message.position.assign(hold.data(), hold.data() + hold.size());
      message.velocity.assign(hold.size(), 0.0);
      joint_state_publisher->publish(message);
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };
  const auto observe_fresh_sample = [&](const std::uint64_t sequence) {
      publish_joint_state();
      CollisionAwareServoComponentTestPeer::publishHardwareSample(
        *component, hold, 11U, sequence);
      return CollisionAwareServoComponentTestPeer::checkRearm(*component, reason);
    };

  EXPECT_EQ(observe_fresh_sample(1U), RearmReadiness::kWaiting) << reason;
  EXPECT_NE(reason.find("(1/3)"), std::string::npos) << reason;
  CollisionAwareServoComponentTestPeer::handleRearmSettleTimeout(*component, reason);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::rearmTimeoutCount(*component), 1U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::stableRearmSamples(*component), 0U);
  const std::string first_latched_reason =
    CollisionAwareServoComponentTestPeer::latchedHaltReason(*component);
  EXPECT_NE(first_latched_reason.find("(1/3)"), std::string::npos);

  // After the one-time timeout transition, a changing 1/3 → 2/3 diagnostic
  // must not restart settling or re-enter the safety wait.
  EXPECT_EQ(observe_fresh_sample(2U), RearmReadiness::kWaiting) << reason;
  EXPECT_NE(reason.find("(1/3)"), std::string::npos) << reason;
  CollisionAwareServoComponentTestPeer::clearHardwareInhibitPending(*component);
  CollisionAwareServoComponentTestPeer::handleRearmSettleTimeout(*component, reason);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::stableRearmSamples(*component), 1U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::rearmTimeoutCount(*component), 1U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::latchedHaltReason(*component),
    first_latched_reason);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::rearmDeadline(*component), deadline);
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::rearmDeadlineExpired(*component));
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::safetyWaitLatched(*component));
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::hardwareInhibitPending(*component));

  EXPECT_EQ(observe_fresh_sample(3U), RearmReadiness::kWaiting) << reason;
  EXPECT_NE(reason.find("(2/3)"), std::string::npos) << reason;
  CollisionAwareServoComponentTestPeer::handleRearmSettleTimeout(*component, reason);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::stableRearmSamples(*component), 2U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::rearmTimeoutCount(*component), 1U);

  EXPECT_EQ(observe_fresh_sample(4U), RearmReadiness::kReady) << reason;
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::provenHold(*component).has_value());
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::hardwareInhibitPending(*component));
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::safetyWaitLatched(*component));
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::validateHardwareIoForControlTick(*component));
}

TEST_F(MotionReferenceTest, FaceUsesExactUprightTrackingWithoutJointPathGuidance)
{
  auto face = target(msg::TrackingTarget::FACE);
  face.face_stamp = rclcpp::Time(1, 0, RCL_ROS_TIME);
  const auto result = update(state(), face);
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_EQ(result.tracking->task.primary_matrix.rows(), 3);
  EXPECT_EQ(result.tracking->task.primary_matrix.cols(), 6);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, FaceRollGuardArmsOnlyWithSettledOrCompleteStopEvidence)
{
  const auto robot_state = compactGoalState();
  auto face = target(msg::TrackingTarget::FACE);
  Eigen::VectorXd position;
  robot_state.copyJointGroupPositions(group_, position);
  const auto now_seconds = [this]() {
      return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_origin_).count();
    };
  const auto sample = [this](
    const Eigen::VectorXd & q, const double seconds) {
      moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
      point.joint_names = parameters_.joint_names;
      point.positions = q;
      point.velocities.setZero(q.size());
      point.accelerations.setZero(q.size());
      point.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return point;
    };

  JointMotionState moving_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  moving_queue.velocity[0] = 0.02;
  const auto moving_without_history = update(
    robot_state, face, nullptr, nullptr, moving_queue);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_TRUE(moving_without_history.compact_roll_constraints.empty());

  // A fresh, settled, matched actual/command state can arm without a retained
  // publication, because there is no in-flight suffix to certify.
  JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto settled_without_history = update(
    robot_state, face, nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  ASSERT_EQ(settled_without_history.compact_roll_constraints.size(), 3U);
  const auto arm_events = reference_->drainCompactEvents();
  const auto armed = std::find_if(
    arm_events.begin(), arm_events.end(), [](const CompactEvent & event) {
      return event.event == "upright_roll_guard_armed";
    });
  ASSERT_NE(armed, arm_events.end());
  EXPECT_NE(armed->details_json.find("\"fullstop_max_abs_roll_error_rad\":"),
    std::string::npos);

  reference_->reset(true);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  const auto after_controlled_rearm = update(robot_state, face);
  EXPECT_EQ(after_controlled_rearm.compact_roll_constraints.size(), 3U);

  reference_->reset(false);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  const auto after_full_reset = update(robot_state, face);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(after_full_reset.compact_roll_constraints.size(), 3U);

  Eigen::VectorXd unsafe_published_stop;
  bool found_edge_violation = false;
  for (Eigen::Index joint = 0; joint < position.size() && !found_edge_violation; ++joint) {
    for (const double offset : {-0.60, -0.30, 0.30, 0.60}) {
      auto candidate = robot_state;
      candidate.setVariablePosition(
        parameters_.joint_names[static_cast<std::size_t>(joint)], position[joint] + offset);
      candidate.update(true);
      if (!evaluateHorizontalEdgeUpright(
          candidate.getGlobalLinkTransform("monitor_control_frame").linear(),
          kCompactRollCapTestRad))
      {
        candidate.copyJointGroupPositions(group_, unsafe_published_stop);
        found_edge_violation = true;
        break;
      }
    }
  }
  ASSERT_TRUE(found_edge_violation);
  std::string violation;
  EXPECT_TRUE(reference_->validateCommandPosition(robot_state, position, &violation));
  EXPECT_TRUE(violation.empty());
  EXPECT_FALSE(reference_->validateCommandPosition(
      robot_state, unsafe_published_stop, &violation));
  EXPECT_EQ(violation, "screen roll");

  // A safe measured pose cannot arm when the newest publication omits its
  // certified stop, even though the active branch contains that endpoint.
  reference_->reset(false);
  const double now = now_seconds();
  PublishedTrajectoryRecord active;
  active.id = 10U;
  active.execution_queue.push_back(sample(position, now - 0.02));
  active.execution_queue.push_back(sample(position, now + 0.02));
  active.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.20) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord truncated_latest = active;
  truncated_latest.id = 11U;
  active.execution_queue.push_back(sample(position, now + 0.20));
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const auto incomplete_history = reference_->update(
    robot_state, face, *planning_scene, now, kSceneRevision, settled, now, settled,
    &active, &truncated_latest);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_TRUE(incomplete_history.compact_roll_constraints.empty());
}

TEST_F(MotionReferenceTest, FaceRollGuardUsesHorizontalEdgeForPublishedStop)
{
  const Eigen::VectorXd current_position = (Eigen::VectorXd(6) <<
    6.100528, -0.297505, -3.904281, 0.004713, 0.158596, 3.142188).finished();
  const Eigen::VectorXd unsafe_stop_position = (Eigen::VectorXd(6) <<
    6.100528, -0.305181, -3.926985, 0.231062, 0.277314, 2.760624).finished();
  auto current = state();
  current.setJointGroupPositions(group_, current_position);
  current.update(true);
  const auto current_pose = evaluateCompactUprightPose(
    current.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(current_pose);
  ASSERT_LE(std::abs(current_pose->signed_roll_error_rad), kCompactRollCapTestRad);

  auto stop = state();
  stop.setJointGroupPositions(group_, unsafe_stop_position);
  stop.update(true);
  const auto stop_pose = evaluateCompactUprightPose(
    stop.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(stop_pose);
  ASSERT_GT(std::abs(stop_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_TRUE(evaluateHorizontalEdgeUpright(
    current.getGlobalLinkTransform("monitor_control_frame").linear(),
    5.1 * std::acos(-1.0) / 180.0));
  ASSERT_TRUE(evaluateHorizontalEdgeUpright(
    stop.getGlobalLinkTransform("monitor_control_frame").linear(),
    5.1 * std::acos(-1.0) / 180.0));

  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto sample = [this](const Eigen::VectorXd & q, const double seconds) {
      moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
      point.joint_names = parameters_.joint_names;
      point.positions = q;
      point.velocities.setZero(q.size());
      point.accelerations.setZero(q.size());
      point.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return point;
    };
  PublishedTrajectoryRecord active;
  active.id = 21U;
  active.execution_queue.push_back(sample(current_position, now - 0.02));
  active.execution_queue.push_back(sample(current_position, now + 0.02));
  active.execution_queue.push_back(sample(unsafe_stop_position, now + 0.80));
  active.execution_queue.push_back(sample(unsafe_stop_position, now + 0.81));
  active.execution_queue.push_back(sample(unsafe_stop_position, now + 0.82));
  active.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.80) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord latest = active;
  latest.id = 22U;

  JointMotionState actual{
    current_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  JointMotionState command = actual;
  auto face = target(msg::TrackingTarget::FACE);
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const auto selected = reference_->update(
    current, face, *planning_scene, now, kSceneRevision, actual, now, command,
    &active, &latest);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(selected.compact_roll_constraints.size(), 3U);
  EXPECT_EQ(reference_->diagnostics().compact_generation, 0U);
}

TEST_F(MotionReferenceTest, FaceRemainsFiniteNearScreenUprightPole)
{
  const auto robot_state = nearScreenPoleState();
  const auto & screen = robot_state.getGlobalLinkTransform("monitor_control_frame");
  ASSERT_LT(screen.linear().col(0).head<2>().norm(), 0.11);

  const auto result = update(robot_state, faceAlongLookDirection(robot_state));
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_TRUE(result.task->primary_matrix.allFinite());
  EXPECT_TRUE(result.task->primary_reference.allFinite());
  EXPECT_GT(result.task->primary_matrix.row(2).norm(), 1.0e-6);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_TRUE(std::isfinite(result.tracking->roll_error_rad));
  EXPECT_LT(std::abs(result.tracking->roll_error_rad), 1.0e-3);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(result.compact_roll_constraints.size(), 3U);
  std::string edge_violation;
  Eigen::VectorXd candidate_position;
  robot_state.copyJointGroupPositions(group_, candidate_position);
  EXPECT_TRUE(reference_->validateCommandPosition(
      robot_state, candidate_position, &edge_violation)) << edge_violation;
}

TEST_F(MotionReferenceCameraTest, GazeFaceKeepsPhysicalScreenRollPrimary)
{
  const auto robot_state = state();
  auto face = target(msg::TrackingTarget::FACE);
  const auto * gaze_link = model_->getLinkModel(parameters_.gaze_frame);
  ASSERT_NE(gaze_link, nullptr);
  const Eigen::Isometry3d optical_pose = robot_state.getGlobalLinkTransform(gaze_link);
  const Eigen::Vector3d target_face =
    optical_pose.translation() + 0.8 * optical_pose.linear().col(2);
  face.face.x = target_face.x();
  face.face.y = target_face.y();
  face.face.z = target_face.z();
  const auto result = update(robot_state, face);
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_NEAR(result.tracking->pointing_error_rad, 0.0, 1.0e-8);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_EQ(result.task->secondary_matrix.rows(), 3);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(result.compact_roll_constraints.size(), 3U);

  Eigen::MatrixXd screen_jacobian;
  ASSERT_TRUE(robot_state.getJacobian(
      group_, model_->getLinkModel("monitor_control_frame"), Eigen::Vector3d::Zero(),
      screen_jacobian, false));
  const auto * root_link = group_->getJointModels().front()->getParentLinkModel();
  if (root_link) {
    const Eigen::Matrix3d root_rotation =
      robot_state.getGlobalLinkTransform(root_link).linear();
    screen_jacobian.topRows(3) = (root_rotation * screen_jacobian.topRows(3)).eval();
    screen_jacobian.bottomRows(3) =
      (root_rotation * screen_jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig config;
  config.orientation_gain = parameters_.orientation_gain;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  const auto edge_task = makeHorizontalEdgePathTask(
    robot_state.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::VectorXd::Zero(6), config);
  ASSERT_TRUE(edge_task);
  EXPECT_TRUE(result.task->primary_matrix.row(2).isApprox(
      edge_task->task.primary_matrix.row(0), 1.0e-12));

  const auto pair_resource = pairYawAssessment(robot_state, face);
  ASSERT_TRUE(pair_resource.valid);
  EXPECT_FALSE(pair_resource.demand_present);
  EXPECT_FALSE(pair_resource.exhausted);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_EQ(reference_->diagnostics().compact_generation, 0U);
}

TEST_F(MotionReferenceCameraTest, GazeFaceRemainsFiniteNearScreenUprightPole)
{
  const auto robot_state = nearScreenPoleState();
  const auto & screen = robot_state.getGlobalLinkTransform("monitor_control_frame");
  ASSERT_LT(screen.linear().col(0).head<2>().norm(), 0.11);

  const auto result = update(robot_state, faceAlongLookDirection(robot_state));
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_TRUE(result.task->primary_matrix.allFinite());
  EXPECT_TRUE(result.task->primary_reference.allFinite());
  EXPECT_GT(result.task->primary_matrix.row(2).norm(), 1.0e-6);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_TRUE(std::isfinite(result.tracking->roll_error_rad));
  EXPECT_LT(std::abs(result.tracking->roll_error_rad), 1.0e-3);
}

TEST_F(MotionReferenceCompactDisabledTest, MissingCompactProfileReportsOnceWithoutBreakingFace)
{
  const auto [robot_state, face_target] = criticalPairExhaustedCase();
  const auto transition = update(robot_state, face_target);
  ASSERT_TRUE(transition.tracking);
  auto preparation_target = face_target;
  preparation_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  const auto result = update(robot_state, preparation_target);
  ASSERT_TRUE(result.tracking);
  EXPECT_TRUE(result.task);
  EXPECT_FALSE(reference_->compactManeuverActive());
  auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "compact_unavailable";
  }));
  recordQualifyingFaceShortfall(robot_state);
  events = reference_->drainCompactEvents();
  EXPECT_TRUE(events.empty());
}

TEST_F(MotionReferenceTest, ExplicitHoldPausesButHealthyHoldContinuesCommittedCompactGoal)
{
  const auto robot_state = state();
  const auto upright = evaluateCompactUprightPose(
    robot_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(upright);
  ASSERT_LE(std::abs(upright->signed_roll_error_rad), kCompactRollCapTestRad);
  JointMotionState moving_queue;
  robot_state.copyJointGroupPositions(group_, moving_queue.position);
  moving_queue.velocity = Eigen::VectorXd::Zero(6);
  moving_queue.acceleration = Eigen::VectorXd::Zero(6);
  moving_queue.velocity[0] = 0.02;
  recordQualifyingFaceShortfall(robot_state);
  admitCurrentPreparationWithLifecycleFixturePath(robot_state, moving_queue);
  const auto committed = update(
    robot_state, target(msg::TrackingTarget::FACE), nullptr, nullptr, moving_queue);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PATH");

  auto explicit_hold = target(msg::TrackingTarget::HOLD);
  const auto paused = update(
    robot_state, explicit_hold, nullptr, nullptr, moving_queue);
  EXPECT_FALSE(paused.task);
  EXPECT_EQ(reference_->diagnostics().state, "COMPACT_COMMITTED_WAITING_FOR_HEALTHY_COMMAND");
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);

  auto healthy_hold = explicit_hold;
  healthy_hold.allow_recovery = true;
  const auto continued = update(
    robot_state, healthy_hold, nullptr, nullptr, moving_queue);
  EXPECT_TRUE(continued.task);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PATH");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_NE(reference_->diagnostics().state, "COMPACT_COMMITTED_WAITING_FOR_HEALTHY_COMMAND");
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
}

TEST_F(MotionReferenceTest, CompactEntryCapCoversMeasuredCommandAndPublishedStopEnvelope)
{
  const auto start = compactGoalState(0.05);
  moveit::core::RobotState geometry_state(model_);
  CompactGeometryValidator geometry(model_, "link_base");
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  const double start_radius = geometry.maximumAssemblyRadius(start);

  Eigen::VectorXd published_stop_position = start_position;
  double published_stop_radius = start_radius;
  for (int second = 0; second <= 8; ++second) {
    for (int third = 0; third <= 8; ++third) {
      for (int fourth = 0; fourth <= 4; ++fourth) {
        Eigen::VectorXd candidate_position = start_position;
        const auto interpolate_limit = [this](const std::size_t joint, const int step,
          const int count) {
            const double fraction = static_cast<double>(step) / static_cast<double>(count);
            return limits_.lower_position[static_cast<Eigen::Index>(joint)] + fraction *
                   (limits_.upper_position[static_cast<Eigen::Index>(joint)] -
                   limits_.lower_position[static_cast<Eigen::Index>(joint)]);
          };
        candidate_position[1] = interpolate_limit(1U, second, 8);
        candidate_position[2] = interpolate_limit(2U, third, 8);
        candidate_position[3] = interpolate_limit(3U, fourth, 4);
        geometry_state.setJointGroupPositions(group_, candidate_position);
        geometry_state.update(true);
        const auto upright = evaluateCompactUprightPose(
          geometry_state.getGlobalLinkTransform("monitor_control_frame").linear());
        if (!upright ||
          std::abs(upright->signed_roll_error_rad) > kCompactRollCapTestRad)
        {
          continue;
        }
        const double radius = geometry.maximumAssemblyRadius(geometry_state);
        if (radius > published_stop_radius) {
          published_stop_position = std::move(candidate_position);
          published_stop_radius = radius;
        }
      }
    }
  }
  ASSERT_GT(published_stop_radius, start_radius + 1.0e-5);
  const auto absolute_pitch = [this, &geometry_state](const Eigen::VectorXd & position) {
      geometry_state.setJointGroupPositions(group_, position);
      geometry_state.update(true);
      const auto pose = evaluateCompactUprightPose(
        geometry_state.getGlobalLinkTransform("monitor_control_frame").linear());
      return pose ? std::optional<double>{std::abs(pose->signed_pitch_rad)} : std::nullopt;
    };
  const auto start_pitch = absolute_pitch(start_position);
  const auto published_stop_pitch = absolute_pitch(published_stop_position);
  Eigen::VectorXd goal_position;
  const auto compact_goal_state = compactGoalState();
  compact_goal_state.copyJointGroupPositions(group_, goal_position);
  const auto goal_pitch = absolute_pitch(goal_position);
  ASSERT_TRUE(start_pitch);
  ASSERT_TRUE(published_stop_pitch);
  ASSERT_TRUE(goal_pitch);
  const double expected_pitch_cap = std::max({
        0.5235987755982988, *start_pitch, *published_stop_pitch, *goal_pitch});

  recordQualifyingFaceShortfall(start);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  PublishedTrajectoryRecord publication;
  publication.id = 1U;
  const auto make_sample = [this](
    const Eigen::VectorXd & position, const double seconds) {
      moveit_servo::KinematicState sample(static_cast<int>(group_->getVariableCount()));
      sample.joint_names = parameters_.joint_names;
      sample.positions = position;
      sample.velocities.setZero(position.size());
      sample.accelerations.setZero(position.size());
      sample.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return sample;
    };
  publication.execution_queue.push_back(make_sample(start_position, now - 0.01));
  publication.execution_queue.push_back(make_sample(start_position, now + 0.01));
  publication.execution_queue.push_back(make_sample(published_stop_position, now + 0.5));
  publication.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.5) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord latest_publication = publication;
  latest_publication.id = 2U;
  PublishedTrajectoryRecord active_publication = publication;
  active_publication.id = 1U;
  active_publication.execution_queue.back() =
    make_sample(start_position, now + 0.5);

  const auto committed = update(
    start, target(msg::TrackingTarget::FACE), &active_publication, &latest_publication,
    JointMotionState{start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)});
  (void)committed;
  admitCurrentPreparationWithLifecycleFixturePath(
    start, JointMotionState{
        start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)},
    &active_publication, &latest_publication);
  ASSERT_TRUE(reference_->compactManeuverActive());
  const double expected_cap = std::max({0.40, start_radius,
        published_stop_radius});
  EXPECT_NEAR(reference_->diagnostics().compact_entry_radius_cap_m, expected_cap, 1.0e-9);
  EXPECT_NEAR(reference_->diagnostics().compact_pitch_cap_rad, expected_pitch_cap, 1.0e-9);
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().planned_path_max_radius_m));

  reference_->reset(true);
  (void)update(start, target(msg::TrackingTarget::FACE), &publication, &publication);
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_NEAR(reference_->diagnostics().compact_entry_radius_cap_m, expected_cap, 1.0e-9);
  EXPECT_NEAR(reference_->diagnostics().compact_pitch_cap_rad, expected_pitch_cap, 1.0e-9);
}

TEST_F(MotionReferenceTest, CompactPitchCapIncludesFreshActualAndSurvivesRearm)
{
  auto command_state = compactGoalState(0.05);
  command_state.setVariablePosition("joint5", -0.8);
  command_state.update(true);
  auto measured_state = command_state;
  measured_state.setVariablePosition("joint5", -0.7);
  measured_state.update(true);
  Eigen::VectorXd command_position;
  command_state.copyJointGroupPositions(group_, command_position);
  Eigen::VectorXd measured_position;
  measured_state.copyJointGroupPositions(group_, measured_position);

  const auto measured_pose = evaluateCompactUprightPose(
    measured_state.getGlobalLinkTransform("monitor_control_frame").linear());
  const auto command_pose = evaluateCompactUprightPose(
    command_state.getGlobalLinkTransform("monitor_control_frame").linear());
  const auto goal_state = compactGoalState();
  const auto goal_pose = evaluateCompactUprightPose(
    goal_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(measured_pose);
  ASSERT_TRUE(command_pose);
  ASSERT_TRUE(goal_pose);
  const double measured_abs_pitch = std::abs(measured_pose->signed_pitch_rad);
  ASSERT_GT(measured_abs_pitch, 0.5235987755982988);
  ASSERT_LT(measured_abs_pitch, 1.3962634015954636);
  ASSERT_GT(std::abs(measured_abs_pitch - std::abs(command_pose->signed_pitch_rad)), 0.05);

  recordQualifyingFaceShortfall(command_state);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto sample = [this, &command_position](const double seconds) {
      moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
      point.joint_names = parameters_.joint_names;
      point.positions = command_position;
      point.velocities.setZero(command_position.size());
      point.accelerations.setZero(command_position.size());
      point.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return point;
    };
  PublishedTrajectoryRecord publication;
  publication.id = 1U;
  publication.execution_queue.push_back(sample(now - 0.02));
  publication.execution_queue.push_back(sample(now + 0.02));
  publication.execution_queue.push_back(sample(now + 0.20));
  publication.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.20) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord latest_publication = publication;
  latest_publication.id = 2U;
  const JointMotionState actual{
    measured_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const JointMotionState queued{
    command_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  (void)reference_->update(
    command_state, target(msg::TrackingTarget::FACE), *planning_scene, now,
    kSceneRevision, actual, now, queued, &publication, &latest_publication);
  admitCurrentPreparationWithLifecycleFixturePath(
    command_state, queued, &publication, &latest_publication, kSceneRevision, actual);
  ASSERT_TRUE(reference_->compactManeuverActive());

  const double expected_cap = std::max({
        0.5235987755982988, measured_abs_pitch,
        std::abs(command_pose->signed_pitch_rad),
        std::abs(goal_pose->signed_pitch_rad)});
  EXPECT_NEAR(reference_->diagnostics().compact_pitch_cap_rad, expected_cap, 1.0e-9);

  reference_->reset(true);
  const double rearm_time = now + 0.02;
  (void)reference_->update(
    command_state, target(msg::TrackingTarget::FACE), *planning_scene, rearm_time,
    kSceneRevision, actual, rearm_time, queued, &publication, &latest_publication);
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_pitch_cap_rad, expected_cap);
}

TEST_F(MotionReferenceTest, CompactCandidateValidationRejectsPitchBeyondCommittedCap)
{
  // Keep the known folded wrist (J4=0, J6=pi) upright while varying only J5.
  // Interpolating toward nearScreenPoleState() also changes screen roll and makes
  // an unrelated hard-roll guard mask this pitch-envelope regression.
  auto entry = compactGoalState();
  entry.setVariablePosition("joint5", -0.8);
  entry.update(true);
  auto candidate = entry;
  candidate.setVariablePosition("joint5", -0.7);
  candidate.update(true);
  Eigen::VectorXd entry_position;
  Eigen::VectorXd candidate_position;
  entry.copyJointGroupPositions(group_, entry_position);
  candidate.copyJointGroupPositions(group_, candidate_position);
  const auto entry_pose = evaluateCompactUprightPose(
    entry.getGlobalLinkTransform("monitor_control_frame").linear());
  const auto candidate_pose = evaluateCompactUprightPose(
    candidate.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(entry_pose);
  ASSERT_TRUE(candidate_pose);
  ASSERT_LE(std::abs(entry_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_LE(std::abs(candidate_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_GT(
    std::abs(candidate_pose->signed_pitch_rad),
    std::abs(entry_pose->signed_pitch_rad) + kCompactPitchValidationToleranceRad);
  ASSERT_LT(std::abs(candidate_pose->signed_pitch_rad), 1.3962634015954636);

  recordQualifyingFaceShortfall(entry);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const JointMotionState actual{
    entry_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const JointMotionState queued = actual;
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  (void)reference_->update(
    entry, target(msg::TrackingTarget::FACE), *planning_scene, now,
    kSceneRevision, actual, now, queued);
  admitCurrentPreparationWithLifecycleFixturePath(entry, queued);
  ASSERT_TRUE(reference_->compactManeuverActive());
  CompactGeometryValidator geometry(model_, "link_base");
  EXPECT_LE(
    geometry.maximumAssemblyRadius(candidate),
    reference_->diagnostics().compact_entry_radius_cap_m + 1.0e-6);
  ASSERT_GT(
    std::abs(candidate_pose->signed_pitch_rad),
    reference_->diagnostics().compact_pitch_cap_rad + kCompactPitchValidationToleranceRad);

  std::string violation;
  EXPECT_TRUE(reference_->validateCompactPathPosition(entry, entry_position, &violation));
  EXPECT_TRUE(violation.empty());
  EXPECT_FALSE(reference_->validateCompactPathPosition(
      entry, candidate_position, &violation));
  EXPECT_EQ(violation, "screen pitch");
}

TEST_F(MotionReferenceTest, CompactRadiusFloorLatchesOnlyAfterPublishedEnvelopeEnters)
{
  auto outside = state();
  outside.setVariablePosition("joint1", 6.20);
  outside.setVariablePosition("joint2", 0.233);
  outside.setVariablePosition("joint3", -1.71);
  outside.setVariablePosition("joint4", 0.0168);
  outside.setVariablePosition("joint5", -0.8338);
  outside.setVariablePosition("joint6", 6.27);
  outside.update();
  const auto inside = compactGoalState();
  CompactGeometryValidator geometry(model_, "link_base");
  Eigen::VectorXd outside_position;
  Eigen::VectorXd inside_position;
  outside.copyJointGroupPositions(group_, outside_position);
  inside.copyJointGroupPositions(group_, inside_position);
  ASSERT_GT(geometry.maximumAssemblyRadius(outside), 0.40);
  ASSERT_LE(geometry.maximumAssemblyRadius(inside), 0.40 + 1.0e-6);

  const auto now_seconds = [this]() {
      return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_origin_).count();
    };
  const auto make_publication = [this](
    const std::uint64_t id, const Eigen::VectorXd & position, const double now) {
      PublishedTrajectoryRecord publication;
      publication.id = id;
      const auto sample = [this, &position](const double seconds) {
          moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
          point.joint_names = parameters_.joint_names;
          point.positions = position;
          point.velocities.setZero(position.size());
          point.accelerations.setZero(position.size());
          point.time_stamp = rclcpp::Time(
            static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
          return point;
        };
      publication.execution_queue.push_back(sample(now - 0.02));
      publication.execution_queue.push_back(sample(now + 0.02));
      publication.execution_queue.push_back(sample(now + 0.20));
      publication.stationary_time = rclcpp::Time(
        static_cast<std::int64_t>(std::llround((now + 0.20) * 1.0e9)), RCL_ROS_TIME);
      return publication;
    };
  const auto update_with = [this](
    const moveit::core::RobotState & robot_state,
    const Eigen::VectorXd & actual_position,
    const Eigen::VectorXd & queued_position,
    const PublishedTrajectoryRecord & active,
    const PublishedTrajectoryRecord & latest,
    const double now) {
      const JointMotionState actual{
        actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
      const JointMotionState queued{
        queued_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(
        robot_state, target(msg::TrackingTarget::FACE), *planning_scene, now,
        kSceneRevision, actual, now, queued, &active, &latest);
    };

  recordQualifyingFaceShortfall(outside);
  double now = now_seconds();
  auto active = make_publication(1U, outside_position, now);
  auto latest = make_publication(2U, outside_position, now);
  (void)update_with(outside, outside_position, outside_position, active, latest, now);
  admitCurrentPreparationWithLifecycleFixturePath(
    outside, JointMotionState{
        outside_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)},
    &active, &latest);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_GT(reference_->diagnostics().compact_entry_radius_cap_m, 0.40);
  const double entry_cap = reference_->diagnostics().compact_entry_radius_cap_m;

  now = now_seconds();
  (void)update_with(inside, inside_position, inside_position, active, latest, now);
  EXPECT_TRUE(reference_->validateCompactPathPosition(inside, outside_position));
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_entry_radius_cap_m, entry_cap);

  now = now_seconds();
  active = make_publication(3U, inside_position, now);
  latest = make_publication(4U, inside_position, now);
  (void)update_with(inside, inside_position, inside_position, active, latest, now);
  EXPECT_FALSE(reference_->validateCompactPathPosition(inside, outside_position));
  EXPECT_TRUE(reference_->validateCompactPathPosition(inside, inside_position));
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_entry_radius_cap_m, entry_cap);
  reference_->reset(true);
  EXPECT_FALSE(reference_->validateCompactPathPosition(inside, outside_position));
}

TEST_F(MotionReferenceTest, CompactRadiusRowsDeflectOutwardCommandNearEntryCap)
{
  auto start = state();
  start.setVariablePosition("joint1", 6.20);
  start.setVariablePosition("joint2", 0.233);
  start.setVariablePosition("joint3", -1.71);
  start.setVariablePosition("joint4", 0.0168);
  start.setVariablePosition("joint5", -0.8338);
  start.setVariablePosition("joint6", 6.27);
  start.update();
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(position.size()),
    Eigen::VectorXd::Zero(position.size())};
  const auto edge_face = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(edge_face.tracking);
  ASSERT_TRUE(edge_face.j1_hold_position_rad);
  EXPECT_NE(reference_->diagnostics().j1_hold_direction, 0);

  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);

  const auto reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->compactManeuverActive());
  EXPECT_FALSE(reference.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().j1_hold_q, 0.0);
  ASSERT_GT(reference.compact_radius_constraints.size(), 0U);
  CompactGeometryValidator geometry(model_, "link_base");
  EXPECT_NEAR(
    reference_->diagnostics().compact_entry_radius_cap_m,
    geometry.maximumAssemblyRadius(start), 1.0e-6);

  const auto restrictive_row = std::max_element(
    reference.compact_radius_constraints.begin(),
    reference.compact_radius_constraints.end(), [](const auto & left, const auto & right) {
      const double left_rate = left.upper_bound / left.coefficients.norm();
      const double right_rate = right.upper_bound / right.coefficients.norm();
      return left_rate > right_rate;
    });
  ASSERT_NE(restrictive_row, reference.compact_radius_constraints.end());
  ASSERT_GT(restrictive_row->coefficients.norm(), 1.0e-8);

  HierarchicalVelocityTask hold_task;
  hold_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  hold_task.primary_reference = Eigen::VectorXd::Zero(6);
  hold_task.primary_weights = Eigen::VectorXd::Ones(6);
  hold_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  hold_task.secondary_reference = Eigen::VectorXd(0);
  hold_task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult stationary_hold = solver.solve(
    hold_task, settled, limits_, reference.compact_radius_constraints);
  ASSERT_TRUE(stationary_hold.command_available());
  EXPECT_NEAR(stationary_hold.joint_velocity.norm(), 0.0, 1.0e-5);
  for (const auto & row : reference.compact_radius_constraints) {
    EXPECT_GE(row.upper_bound, -1.0e-8);
  }

  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  task.primary_reference = 0.2 * restrictive_row->coefficients.normalized();
  task.primary_weights = Eigen::VectorXd::Ones(6);
  task.secondary_matrix = Eigen::MatrixXd(0, 6);
  task.secondary_reference = Eigen::VectorXd(0);
  task.secondary_weights = Eigen::VectorXd(0);
  const QpResult unguarded = solver.solve(task, settled, limits_);
  const QpResult guarded = solver.solve(
    task, settled, limits_, reference.compact_radius_constraints);
  ASSERT_TRUE(unguarded.command_available());
  ASSERT_TRUE(guarded.command_available());
  EXPECT_GT(
    restrictive_row->coefficients.dot(unguarded.joint_velocity),
    restrictive_row->upper_bound + 1.0e-6);
  EXPECT_LE(
    restrictive_row->coefficients.dot(guarded.joint_velocity),
    restrictive_row->upper_bound + 2.0e-5);
}

TEST_F(MotionReferenceTest, CompactRadiusRowsDoNotBindFarInsideFloorCap)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_GT(reference.compact_radius_constraints.size(), 0U);

  const auto strongest_row = std::max_element(
    reference.compact_radius_constraints.begin(),
    reference.compact_radius_constraints.end(), [](const auto & left, const auto & right) {
      return left.coefficients.norm() < right.coefficients.norm();
    });
  ASSERT_NE(strongest_row, reference.compact_radius_constraints.end());
  ASSERT_GT(strongest_row->coefficients.norm(), 1.0e-8);

  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  task.primary_reference = 0.001 * strongest_row->coefficients.normalized();
  task.primary_weights = Eigen::VectorXd::Ones(6);
  task.secondary_matrix = Eigen::MatrixXd(0, 6);
  task.secondary_reference = Eigen::VectorXd(0);
  task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult unguarded = solver.solve(task, settled, limits_);
  const QpResult guarded = solver.solve(
    task, settled, limits_, reference.compact_radius_constraints);
  ASSERT_TRUE(unguarded.command_available());
  ASSERT_TRUE(guarded.command_available());
  EXPECT_NEAR((guarded.joint_velocity - unguarded.joint_velocity).norm(), 0.0, 1.0e-5);
}

TEST_F(MotionReferenceTest, CompactPitchRowsHoldAtCapDeflectOutwardAndAllowInwardMotion)
{
  auto start = compactGoalState();
  start.setVariablePosition("joint5", -0.8);
  start.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  start.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  start.update(true);
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  const auto start_pose = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_pose);
  ASSERT_LE(std::abs(start_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_GT(std::abs(start_pose->signed_pitch_rad), 0.5235987755982988);
  ASSERT_LT(std::abs(start_pose->signed_pitch_rad), 1.3962634015954636);

  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  const JointMotionState settled{
    start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_EQ(reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(reference.compact_roll_constraints.size(), 2U);
  const auto boundary_row = std::min_element(
    reference.compact_pitch_constraints.begin(), reference.compact_pitch_constraints.end(),
    [](const auto & left, const auto & right) {return left.upper_bound < right.upper_bound;});
  ASSERT_NE(boundary_row, reference.compact_pitch_constraints.end());
  EXPECT_NEAR(boundary_row->upper_bound, 0.0, 1.0e-8);
  auto compact_constraints = reference.compact_radius_constraints;
  compact_constraints.insert(
    compact_constraints.end(), reference.compact_roll_constraints.begin(),
    reference.compact_roll_constraints.end());
  compact_constraints.insert(
    compact_constraints.end(), reference.compact_pitch_constraints.begin(),
    reference.compact_pitch_constraints.end());

  HierarchicalVelocityTask hold_task;
  hold_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  hold_task.primary_reference = Eigen::VectorXd::Zero(6);
  hold_task.primary_weights = Eigen::VectorXd::Ones(6);
  hold_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  hold_task.secondary_reference = Eigen::VectorXd(0);
  hold_task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult stationary_hold = solver.solve(
    hold_task, settled, limits_, compact_constraints);
  ASSERT_TRUE(stationary_hold.command_available());
  EXPECT_NEAR(stationary_hold.joint_velocity.norm(), 0.0, 1.0e-5);

  HierarchicalVelocityTask outward_task = hold_task;
  outward_task.primary_reference = 0.2 * boundary_row->coefficients.normalized();
  const QpResult unguarded_outward = solver.solve(outward_task, settled, limits_);
  const QpResult guarded_outward = solver.solve(
    outward_task, settled, limits_, compact_constraints);
  ASSERT_TRUE(unguarded_outward.command_available());
  ASSERT_TRUE(guarded_outward.command_available());
  EXPECT_GT(
    boundary_row->coefficients.dot(unguarded_outward.joint_velocity),
    boundary_row->upper_bound + 1.0e-6);
  EXPECT_LE(
    boundary_row->coefficients.dot(guarded_outward.joint_velocity),
    boundary_row->upper_bound + 2.0e-5);

  HierarchicalVelocityTask inward_task = hold_task;
  inward_task.primary_reference = -0.1 * boundary_row->coefficients.normalized();
  const QpResult guarded_inward = solver.solve(
    inward_task, settled, limits_, compact_constraints);
  ASSERT_TRUE(guarded_inward.command_available());
  EXPECT_LT(boundary_row->coefficients.dot(guarded_inward.joint_velocity), -1.0e-5);

  const Eigen::VectorXd outward_velocity =
    0.02 * boundary_row->coefficients.normalized();
  const JointMotionState moving_queue{
    start_position, outward_velocity, Eigen::VectorXd::Zero(6)};
  const auto moving_reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, moving_queue);
  ASSERT_EQ(moving_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(moving_reference.compact_roll_constraints.size(), 2U);
  const auto moving_boundary_row = std::find_if(
    moving_reference.compact_pitch_constraints.begin(),
    moving_reference.compact_pitch_constraints.end(),
    [&boundary_row](const auto & row) {
      return row.coefficients.isApprox(boundary_row->coefficients, 1.0e-12);
    });
  ASSERT_NE(moving_boundary_row, moving_reference.compact_pitch_constraints.end());
  const double current_outward_speed =
    boundary_row->coefficients.dot(moving_queue.velocity);
  EXPECT_LE(moving_boundary_row->upper_bound, current_outward_speed + 1.0e-6);

  const QpResult unguarded_moving = solver.solve(outward_task, moving_queue, limits_);
  auto moving_constraints = moving_reference.compact_radius_constraints;
  moving_constraints.insert(
    moving_constraints.end(), moving_reference.compact_roll_constraints.begin(),
    moving_reference.compact_roll_constraints.end());
  moving_constraints.insert(
    moving_constraints.end(), moving_reference.compact_pitch_constraints.begin(),
    moving_reference.compact_pitch_constraints.end());
  const QpResult guarded_moving = solver.solve(
    outward_task, moving_queue, limits_, moving_constraints);
  ASSERT_TRUE(unguarded_moving.command_available());
  ASSERT_TRUE(guarded_moving.command_available());
  EXPECT_GT(
    moving_boundary_row->coefficients.dot(unguarded_moving.joint_velocity),
    moving_boundary_row->upper_bound + 1.0e-6);
  EXPECT_LE(
    moving_boundary_row->coefficients.dot(guarded_moving.joint_velocity),
    moving_boundary_row->upper_bound + 2.0e-5);
}

TEST_F(MotionReferenceTest, CompactOrientationViableMotionHasCertifiedBrakingTail)
{
  const auto start = compactGoalState(0.05);
  const auto start_pose = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_pose);
  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  const auto stationary_reference = update(
    start, target(msg::TrackingTarget::FACE));
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_TRUE(std::isfinite(reference_->diagnostics().compact_pitch_cap_rad));
  ASSERT_GT(
    reference_->diagnostics().compact_pitch_cap_rad -
    std::abs(start_pose->signed_pitch_rad), 0.02);
  ASSERT_GT(kCompactRollCapTestRad - std::abs(start_pose->signed_roll_error_rad), 0.02);
  CompactGeometryValidator geometry(model_, "link_base");
  EXPECT_GT(
    reference_->diagnostics().compact_entry_radius_cap_m -
    geometry.maximumAssemblyRadius(start), 0.01);
  ASSERT_EQ(stationary_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(stationary_reference.compact_roll_constraints.size(), 2U);
  const auto outward_row = std::min_element(
    stationary_reference.compact_pitch_constraints.begin(),
    stationary_reference.compact_pitch_constraints.end(),
    [](const auto & left, const auto & right) {return left.upper_bound < right.upper_bound;});
  ASSERT_NE(outward_row, stationary_reference.compact_pitch_constraints.end());

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const auto roll_outward_row = std::min_element(
    stationary_reference.compact_roll_constraints.begin(),
    stationary_reference.compact_roll_constraints.end(),
    [](const auto & left, const auto & right) {return left.upper_bound < right.upper_bound;});
  ASSERT_NE(roll_outward_row, stationary_reference.compact_roll_constraints.end());
  const Eigen::VectorXd velocity = 0.03 * outward_row->coefficients.normalized() +
    0.03 * roll_outward_row->coefficients.normalized();
  const JointMotionState moving{position, velocity, Eigen::VectorXd::Zero(6)};
  const auto moving_reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, moving);
  ASSERT_EQ(moving_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(moving_reference.compact_roll_constraints.size(), 2U);

  HierarchicalVelocityTask outward_task;
  outward_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  outward_task.primary_reference = 0.2 * outward_row->coefficients.normalized() +
    0.2 * roll_outward_row->coefficients.normalized();
  outward_task.primary_weights = Eigen::VectorXd::Ones(6);
  outward_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  outward_task.secondary_reference = Eigen::VectorXd(0);
  outward_task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  auto constraints = moving_reference.compact_radius_constraints;
  constraints.insert(
    constraints.end(), moving_reference.compact_roll_constraints.begin(),
    moving_reference.compact_roll_constraints.end());
  constraints.insert(
    constraints.end(), moving_reference.compact_pitch_constraints.begin(),
    moving_reference.compact_pitch_constraints.end());
  const QpResult command = solver.solve(outward_task, moving, limits_, constraints);
  ASSERT_TRUE(command.command_available());

  const Eigen::VectorXd next_position = position +
    parameters_.control_period_sec * command.joint_velocity;
  const Eigen::VectorXd next_acceleration =
    (command.joint_velocity - velocity) / parameters_.control_period_sec;
  const JointMotionState branch{next_position, command.joint_velocity, next_acceleration};
  EmergencyBrakeTailConfig tail_config;
  tail_config.period_sec = parameters_.control_period_sec;
  const EmergencyBrakeTail tail = EmergencyBrakeTailGenerator(tail_config).generate(
    branch, limits_);
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  ASSERT_GT(tail.first_stationary_point, 0U);

  for (std::size_t index = 0U; index < tail.points.size(); ++index) {
    std::string violation;
    EXPECT_TRUE(reference_->validateCompactPathPosition(
        start, tail.points[index].position, &violation)) <<
      "braking tail point " << index << " violates " << violation;
  }
}

TEST_F(MotionReferenceTest, CompactQueueTailRowsAllowDumpedStateAndCertifyEmergencyBrake)
{
  auto start = state();
  const Eigen::VectorXd position = (Eigen::VectorXd(6) <<
    6.1149018406659783, 0.26252163795647276, -1.6730647452367475,
    -0.051544811767068766, -0.80646278725974319, 6.2830336538974318).finished();
  const Eigen::VectorXd velocity = (Eigen::VectorXd(6) <<
    -0.002, -0.0000944063747657, -0.00199843840694, -0.002, 0.002, -0.002).finished();
  const Eigen::VectorXd acceleration = (Eigen::VectorXd(6) <<
    -0.2, -0.0094406375, -0.1998438407, -0.2, 0.2, -0.2).finished();
  start.setJointGroupPositions(group_, position);
  start.setJointGroupVelocities(group_, velocity);
  start.setJointGroupAccelerations(group_, acceleration);
  start.update(true);
  const JointMotionState queue_tail{position, velocity, acceleration};

  EmergencyBrakeTailConfig prior_tail_config;
  prior_tail_config.period_sec = parameters_.control_period_sec;
  const EmergencyBrakeTail prior_tail = EmergencyBrakeTailGenerator(
    prior_tail_config).generate(queue_tail, limits_);
  ASSERT_TRUE(prior_tail.command_available()) << prior_tail.failure_reason;
  const double evidence_start = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto selected = update(start, target(msg::TrackingTarget::FACE));
  ASSERT_TRUE(selected.tracking);
  auto shortfall = *selected.tracking;
  shortfall.has_face_primary_roll = true;
  shortfall.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
  shortfall.face_primary_feedforward_radps.setZero();
  QpResult accepted_shortfall;
  accepted_shortfall.status = QpStatus::kSolved;
  accepted_shortfall.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
  for (int tick = 0; tick <= 8; ++tick) {
    reference_->recordFaceCommand(
      shortfall, accepted_shortfall,
      evidence_start + 0.02 * static_cast<double>(tick + 1), true);
  }
  const double update_time = evidence_start + 0.18;
  PublishedTrajectoryRecord publication;
  publication.id = 1U;
  for (std::size_t index = 0U; index < prior_tail.points.size(); ++index) {
    const auto & point = prior_tail.points[index];
    moveit_servo::KinematicState sample(static_cast<int>(group_->getVariableCount()));
    sample.joint_names = parameters_.joint_names;
    sample.positions = point.position;
    sample.velocities = point.velocity;
    sample.accelerations = point.acceleration;
    sample.time_stamp = rclcpp::Time(
      static_cast<std::int64_t>(std::llround(
        (update_time + static_cast<double>(index) * parameters_.control_period_sec) * 1.0e9)),
      RCL_ROS_TIME);
    publication.execution_queue.push_back(std::move(sample));
  }
  publication.stationary_time = publication.execution_queue.back().time_stamp;

  MotionReferenceResult compact_reference;
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    compact_reference = reference_->update(
      start, target(msg::TrackingTarget::FACE), *planning_scene, update_time,
      kSceneRevision, queue_tail, update_time, queue_tail, &publication, &publication);
  }
  admitCurrentPreparationWithLifecycleFixturePath(
    start, queue_tail, &publication, &publication, kSceneRevision, queue_tail);
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    compact_reference = reference_->update(
      start, target(msg::TrackingTarget::FACE), *planning_scene, update_time,
      kSceneRevision, queue_tail, update_time, queue_tail, &publication, &publication);
  }
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_EQ(compact_reference.compact_radius_constraints.size(), 6U);
  ASSERT_EQ(compact_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(compact_reference.compact_roll_constraints.size(), 2U);
  CompactGeometryValidator geometry(model_, "link_base");
  for (const auto & point : prior_tail.points) {
    auto queued_state = start;
    queued_state.setJointGroupPositions(group_, point.position);
    queued_state.update(true);
    EXPECT_LE(
      geometry.maximumAssemblyRadius(queued_state),
      reference_->diagnostics().compact_entry_radius_cap_m + 1.0e-6);
  }

  HierarchicalVelocityTask outward_task;
  outward_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  outward_task.primary_reference = Eigen::VectorXd::Zero(6);
  const auto & radius_boundary_row = compact_reference.compact_radius_constraints[5];
  const auto & pitch_boundary_row = compact_reference.compact_pitch_constraints[0];
  outward_task.primary_reference +=
    0.2 * radius_boundary_row.coefficients.normalized() +
    0.2 * pitch_boundary_row.coefficients.normalized();
  outward_task.primary_weights = Eigen::VectorXd::Ones(6);
  outward_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  outward_task.secondary_reference = Eigen::VectorXd(0);
  outward_task.secondary_weights = Eigen::VectorXd(0);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance =
    parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  auto constraints = compact_reference.compact_radius_constraints;
  constraints.insert(
    constraints.end(), compact_reference.compact_roll_constraints.begin(),
    compact_reference.compact_roll_constraints.end());
  constraints.insert(
    constraints.end(), compact_reference.compact_pitch_constraints.begin(),
    compact_reference.compact_pitch_constraints.end());
  const QpResult command = solver.solve(outward_task, queue_tail, limits_, constraints);
  ASSERT_TRUE(command.command_available()) << static_cast<int>(command.status);
  EXPECT_LE(
    radius_boundary_row.coefficients.dot(command.joint_velocity),
    radius_boundary_row.upper_bound + 2.0e-5);
  EXPECT_LE(
    pitch_boundary_row.coefficients.dot(command.joint_velocity),
    pitch_boundary_row.upper_bound + 2.0e-5);

  const Eigen::VectorXd branch_position =
    position + parameters_.control_period_sec * command.joint_velocity;
  const Eigen::VectorXd branch_acceleration =
    (command.joint_velocity - velocity) / parameters_.control_period_sec;
  const EmergencyBrakeTailGenerator tail_generator(
    EmergencyBrakeTailConfig{parameters_.control_period_sec});
  const EmergencyBrakeTail tail = tail_generator.generate(
    JointMotionState{branch_position, command.joint_velocity, branch_acceleration}, limits_);
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  bool outward_tail_rejected_by_radius = false;
  for (std::size_t index = 0U; index < tail.points.size(); ++index) {
    std::string violation;
    if (!reference_->validateCompactPathPosition(
        start, tail.points[index].position, &violation))
    {
      EXPECT_EQ(violation, "assembly radius") << "tail point " << index;
      outward_tail_rejected_by_radius = true;
      break;
    }
  }
  EXPECT_TRUE(outward_tail_rejected_by_radius)
    << "the outward probe should remain covered by the hard compact radius guard";

  const Eigen::VectorXd goal = (Eigen::VectorXd(6) <<
    0.0, -0.2715944, -2.8699983, 0.0, 1.5707963267948966,
    limits_.upper_position[5] - 0.02).finished();
  Eigen::VectorXd prefix = goal;
  prefix[0] = position[0] + std::clamp(goal[0] - position[0], -0.35, 0.35);
  prefix[3] = position[3];
  const auto planned_fold_task = makeJointPathVelocityTask(position, prefix, 6.0, 0.70);
  ASSERT_TRUE(planned_fold_task);
  const QpResult planned_fold_command = solver.solve(
    *planned_fold_task, queue_tail, limits_, constraints);
  ASSERT_TRUE(planned_fold_command.command_available()) <<
    static_cast<int>(planned_fold_command.status);
  const Eigen::VectorXd fold_branch_position = position +
    parameters_.control_period_sec * planned_fold_command.joint_velocity;
  const Eigen::VectorXd fold_branch_acceleration =
    (planned_fold_command.joint_velocity - velocity) / parameters_.control_period_sec;
  const EmergencyBrakeTail fold_tail = tail_generator.generate(
    JointMotionState{
        fold_branch_position, planned_fold_command.joint_velocity, fold_branch_acceleration},
    limits_);
  ASSERT_TRUE(fold_tail.command_available()) << fold_tail.failure_reason;
  for (std::size_t index = 0U; index < fold_tail.points.size(); ++index) {
    std::string violation;
    EXPECT_TRUE(reference_->validateCompactPathPosition(
      start, fold_tail.points[index].position, &violation)) <<
      "planned fold tail point " << index << " violates " << violation;
  }
}

TEST_F(MotionReferenceTest, CompactCommitAdoptsCheckedPathAndPreservesGoalAcrossRearm)
{
  const auto [start, face_target] = criticalPairExhaustedCase();
  const auto start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  ASSERT_LE(std::abs(start_upright->signed_roll_error_rad), kCompactRollCapTestRad);
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  const JointMotionState settled_queue{
    start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto initial_face = update(
    start, face_target, nullptr, nullptr, settled_queue);
  ASSERT_TRUE(initial_face.task);
  auto preparation_target = face_target;
  preparation_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  const auto waiting = update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  EXPECT_TRUE(waiting.task);
  EXPECT_TRUE(waiting.tracking);
  EXPECT_FALSE(waiting.follows_path);
  // Entry-envelope caps are committed only after a worker result is admitted.
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_effective_radius_cap_m));
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_entry_radius_cap_m));
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_pitch_cap_rad));

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  MotionReferenceResult planned;
  do {
    planned = update(
      start, preparation_target, nullptr, nullptr, settled_queue);
    if (reference_->diagnostics().plans_accepted > 0U) {
      break;
    }
    std::this_thread::sleep_for(2ms);
  } while (std::chrono::steady_clock::now() < deadline);
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U)
    << reference_->diagnostics().last_plan_message;
  EXPECT_TRUE(planned.follows_path);
  ASSERT_TRUE(planned.task);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PATH");
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_entry_radius_cap_m));
  EXPECT_DOUBLE_EQ(
    reference_->diagnostics().compact_effective_radius_cap_m,
    reference_->diagnostics().compact_entry_radius_cap_m);
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_pitch_cap_rad));

  const auto events = reference_->drainCompactEvents();
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(triggered, events.end());
  EXPECT_EQ(triggered->generation_id, 1U);
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "path_started";
  }));

  // A controlled executor re-arm drops the follower but keeps the immutable
  // compact goal. The next healthy FACE update must request a fresh path from
  // current q and must not expose the goal as an unplanned direct joint task.
  reference_->reset(true);
  ASSERT_TRUE(reference_->compactManeuverActive());
  const auto rearmed = update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
  if (rearmed.task) {
    EXPECT_EQ(rearmed.task->primary_matrix.rows(), 1);
    EXPECT_EQ(rearmed.task->secondary_reference.norm(), 0.0);
  }
  const auto replan_deadline = std::chrono::steady_clock::now() + 3s;
  while (reference_->diagnostics().plans_requested < 2U &&
    std::chrono::steady_clock::now() < replan_deadline)
  {
    (void)update(start, preparation_target, nullptr, nullptr, settled_queue);
    std::this_thread::sleep_for(2ms);
  }
  EXPECT_GE(reference_->diagnostics().plans_requested, 2U);
  EXPECT_LE(reference_->diagnostics().plans_accepted,
    reference_->diagnostics().plans_requested);
  const auto rearm_events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(
      rearm_events.begin(), rearm_events.end(), [](const CompactEvent & event) {
        return event.event == "path_started" && event.generation_id == 1U;
      }));
}

TEST_F(MotionReferenceTest, CompactPlanningKeepsUprightCorrectionOutsideRollAdmission)
{
  auto start = compactGoalState(0.05);
  auto start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  auto probe = start;
  probe.setVariablePosition("joint6", probe.getVariablePosition("joint6") + 1.0e-4);
  probe.update(true);
  const auto probe_upright = evaluateCompactUprightPose(
    probe.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(probe_upright);
  const double joint6_direction =
    std::abs(probe_upright->signed_roll_error_rad) >
    std::abs(start_upright->signed_roll_error_rad) ? 1.0 : -1.0;
  start.setVariablePosition("joint6", start.getVariablePosition("joint6") +
    joint6_direction * 0.15);
  start.update(true);
  start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  ASSERT_GT(std::abs(start_upright->signed_roll_error_rad), kCompactRollCapTestRad);

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  JointMotionState moving_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  moving_queue.velocity[0] = 0.02;
  const JointMotionState actual_feedback{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto correction = MotionReferenceTestPeer::handleQualifiedPreparationForTest(
    *reference_, start, target(msg::TrackingTarget::FACE), *planning_scene,
    now, kSceneRevision, actual_feedback, moving_queue);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_phase, "UPRIGHT_START");
  EXPECT_EQ(reference_->diagnostics().state, "COMPACT_UPRIGHT_START");
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  ASSERT_TRUE(correction.task);
  EXPECT_EQ(correction.task->primary_matrix.rows(), 1);
  EXPECT_EQ(correction.task->secondary_matrix.rows(), 6);
  EXPECT_GT(correction.task->primary_reference.norm(), 1.0e-6);

  // Once feedback and the command queue have recovered inside the roll
  // admission envelope and are settled, the same pending policy may commit.
  const auto recovered = compactGoalState(0.05);
  Eigen::VectorXd recovered_position;
  recovered.copyJointGroupPositions(group_, recovered_position);
  const JointMotionState recovered_queue{
    recovered_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const JointMotionState recovered_feedback{
    recovered_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  ASSERT_TRUE(MotionReferenceTestPeer::armFaceUprightRollGuardFromFeedbackForTest(
      *reference_, recovered, now + parameters_.control_period_sec,
      recovered_feedback, recovered_queue));
  const auto recovered_update = MotionReferenceTestPeer::handleQualifiedPreparationForTest(
    *reference_, recovered, target(msg::TrackingTarget::FACE), *planning_scene,
    now + parameters_.control_period_sec, kSceneRevision,
    recovered_feedback, recovered_queue, true);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_GT(reference_->diagnostics().compact_generation, 0U);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  EXPECT_FALSE(recovered_update.follows_path);
}

TEST_F(
  MotionReferenceTest,
  ExhaustedCompactPreparationUsesOldRollRecoveryWithArmedEdgeGuard)
{
  auto start = state();
  constexpr std::array<double, 6> kRuntimeFailurePositionRad{
    6.251909451703674, -1.1826500946802403, 0.015470642634827407,
    6.186385077427225, -1.016831617343016, 6.283185304763757};
  constexpr std::array<double, 6> kRuntimeFailureVelocityRadps{
    -1.172418499090222e-12, 0.16218074334456303, -0.041916194886367476,
    -0.005253809546751862, -0.13616195992908112, 1.1045216113817272e-7};
  Eigen::VectorXd runtime_velocity(6);
  for (std::size_t index = 0; index < kRuntimeFailureVelocityRadps.size(); ++index) {
    runtime_velocity[static_cast<Eigen::Index>(index)] =
      kRuntimeFailureVelocityRadps[index];
  }
  for (std::size_t index = 0; index < kRuntimeFailurePositionRad.size(); ++index) {
    start.setVariablePosition(parameters_.joint_names[index],
      kRuntimeFailurePositionRad[index]);
  }
  start.setJointGroupVelocities(group_, runtime_velocity);
  start.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  start.update(true);

  const auto edge_pose = evaluateHorizontalEdgeUprightWithSineCap(
    start.getGlobalLinkTransform("monitor_control_frame").linear(),
    std::sin(kCompactRollCapTestRad));
  const auto compact_pose = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(edge_pose);
  ASSERT_TRUE(compact_pose);
  ASSERT_LE(std::abs(edge_pose->signed_edge_sine),
    std::sin(kCompactRollCapTestRad));
  ASSERT_GT(std::abs(compact_pose->signed_roll_error_rad), kCompactRollCapTestRad);

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState feedback{
    position, runtime_velocity, Eigen::VectorXd::Zero(6)};
  // The capture has no q/v/a queue snapshot at this timestamp. Keep the
  // recorded measured q/dq exact and use an explicitly synthetic stationary
  // queue state for the admission/candidate rollout fixture.
  const JointMotionState command_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();

  // Keep the independently armed B edge guard while proving compact admission
  // still uses its original gravity-roll envelope.
  MotionReferenceTestPeer::qualifyPairYawForIsolatedAdmissionTest(*reference_);
  const auto recovery = MotionReferenceTestPeer::handleQualifiedPreparationForTest(
    *reference_, start, target(msg::TrackingTarget::FACE), *planning_scene,
    now, kSceneRevision, feedback, command_queue, true);

  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(reference_->diagnostics().state, "COMPACT_UPRIGHT_START");
  EXPECT_EQ(reference_->diagnostics().compact_phase, "UPRIGHT_START");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_FALSE(reference_->compactManeuverActive());
  ASSERT_TRUE(recovery.task);
  EXPECT_GT(recovery.task->primary_reference.norm(), 1.0e-6);
  EXPECT_FALSE(recovery.motion_reference_support_unavailable);
  EXPECT_FALSE(recovery.compact_roll_constraints.empty());

  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, position, 11U, 1U);

  moveit_servo::KinematicState current(6);
  current.joint_names = parameters_.joint_names;
  current.positions = position;
  current.velocities = command_queue.velocity;
  current.accelerations = Eigen::VectorXd::Zero(6);
  current.time_stamp = component->now();
  const JointMotionState current_motion = command_queue;
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  const HierarchicalVelocityQp solver(qp_config);
  std::vector<LinearVelocityConstraint> recovery_constraints;
  recovery_constraints.insert(recovery_constraints.end(),
    recovery.compact_radius_constraints.begin(), recovery.compact_radius_constraints.end());
  recovery_constraints.insert(recovery_constraints.end(),
    recovery.compact_roll_constraints.begin(), recovery.compact_roll_constraints.end());
  recovery_constraints.insert(recovery_constraints.end(),
    recovery.compact_pitch_constraints.begin(), recovery.compact_pitch_constraints.end());
  const QpResult qp = solver.solve(
    *recovery.task, current_motion, limits_, recovery_constraints);
  ASSERT_TRUE(qp.command_available()) << static_cast<int>(qp.status);
  EXPECT_GT(qp.joint_velocity.norm(), 1.0e-5);

  std::string candidate_reason;
  const auto accepted = CollisionAwareServoComponentTestPeer::validateCandidate(
    *component, std::make_shared<moveit::core::RobotState>(start), current,
    current_motion, qp.joint_velocity, *planning_scene, candidate_reason);
  ASSERT_TRUE(accepted) << candidate_reason;
  ASSERT_TRUE(accepted->braking_tail.command_available());
  ASSERT_FALSE(accepted->braking_tail.points.empty());
}

TEST_F(MotionReferenceTest, CompactPreparationKeepsFaceActiveWhilePlanRuns)
{
  const auto [start, face_target] = criticalPairExhaustedCase();

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  Eigen::VectorXd moving_velocity = Eigen::VectorXd::Zero(6);
  moving_velocity[0] = 0.02;
  const JointMotionState settled_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  // Let a stationary real-FK update arm the roll guard before requesting the
  // plan. Motion is introduced only after preparation has started.
  (void)update(
    start, face_target, nullptr, nullptr, settled_queue);
  auto preparation_target = face_target;
  preparation_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  (void)update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  ASSERT_GE(reference_->diagnostics().plans_requested, 1U);
  const auto requests_while_preparing = reference_->diagnostics().plans_requested;
  // Hold the worker result pending deterministically so this test checks the
  // FACE behavior while preparation is running, independent of planner speed.
  MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
  MotionReferenceTestPeer::activateResultGeneration(
    *reference_, reference_->diagnostics().compact_generation + 100U);

  const JointMotionState moving_queue{
    position, moving_velocity, Eigen::VectorXd::Zero(6)};
  preparation_target.face_stamp = rclcpp::Time(1, 40'000'000, RCL_ROS_TIME);
  const auto moving_face = update(
    start, preparation_target, nullptr, nullptr, moving_queue);
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_while_preparing);
  EXPECT_TRUE(moving_face.tracking);
  EXPECT_TRUE(moving_face.task);
  EXPECT_FALSE(moving_face.follows_path);

  Eigen::VectorXd moving_acceleration = Eigen::VectorXd::Zero(6);
  moving_acceleration[0] = 0.04;
  const JointMotionState accelerating_queue{
    position, Eigen::VectorXd::Zero(6), moving_acceleration};
  preparation_target.face_stamp = rclcpp::Time(1, 60'000'000, RCL_ROS_TIME);
  (void)update(
    start, preparation_target, nullptr, nullptr, accelerating_queue);
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_while_preparing);

  preparation_target.face_stamp = rclcpp::Time(1, 80'000'000, RCL_ROS_TIME);
  const auto settled = update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_while_preparing);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  EXPECT_FALSE(settled.follows_path);
}

TEST_F(MotionReferenceCameraTest, CompactLifecycleCompletesAndShortfallAloneDoesNotRestart)
{
  auto start = compactGoalState(0.05);
  start.setVariablePosition("joint6", start.getVariablePosition("joint6") + 0.04);
  start.update(true);
  const auto start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  ASSERT_GT(std::abs(start_upright->signed_roll_error_rad), 0.03);
  ASSERT_LT(std::abs(start_upright->signed_roll_error_rad), kCompactRollCapTestRad);
  // This lifecycle test starts after pair-gate qualification; the real FK gate
  // is exercised by dedicated public-update cases above.
  admitCurrentPreparationWithLifecycleFixturePath(start);
  const auto plan_requests_before_lifecycle = reference_->diagnostics().plans_requested;
  const auto committed_generation = reference_->diagnostics().compact_generation;

  double simulated_time = 1.18;
  const auto motion_state = [this](const moveit::core::RobotState & state) {
      JointMotionState result;
      state.copyJointGroupPositions(group_, result.position);
      state.copyJointGroupVelocities(group_, result.velocity);
      state.copyJointGroupAccelerations(group_, result.acceleration);
      return result;
    };
  const auto aligned_face = [this](const moveit::core::RobotState & state) {
      auto command = target(msg::TrackingTarget::FACE);
      const auto & screen = state.getGlobalLinkTransform("monitor_control_frame");
      const Eigen::Quaterniond orientation(screen.linear());
      command.pose.position.x = screen.translation().x();
      command.pose.position.y = screen.translation().y();
      command.pose.position.z = screen.translation().z();
      command.pose.orientation.x = orientation.x();
      command.pose.orientation.y = orientation.y();
      command.pose.orientation.z = orientation.z();
      command.pose.orientation.w = orientation.w();
      const auto * gaze_link = parameters_.gaze_frame.empty() ? nullptr :
        model_->getLinkModel(parameters_.gaze_frame);
      Eigen::Isometry3d look_frame = gaze_link ?
        state.getGlobalLinkTransform(gaze_link) : screen;
      if (gaze_link) {
        look_frame.linear() *= face_tracking_arm::opticalToPointingRotation();
      }
      const Eigen::Vector3d face = look_frame.translation() +
        0.8 * look_frame.linear().col(0);
      command.face.x = face.x();
      command.face.y = face.y();
      command.face.z = face.z();
      return command;
    };
  const auto misaligned_face = [&aligned_face](const moveit::core::RobotState & state) {
      auto command = aligned_face(state);
      const auto & screen = state.getGlobalLinkTransform("monitor_control_frame");
      const Eigen::Vector3d face(
        command.face.x, command.face.y, command.face.z);
      const Eigen::Vector3d offset = 0.4 * screen.linear().col(1);
      command.face.x = face.x() + offset.x();
      command.face.y = face.y() + offset.y();
      command.face.z = face.z() + offset.z();
      return command;
    };
  const auto update_at = [this, &simulated_time, &motion_state](
    const moveit::core::RobotState & state, msg::TrackingTarget command,
    const moveit::core::RobotState * actual_state_override = nullptr)
    {
      simulated_time += parameters_.control_period_sec;
      command.face_stamp.sec = static_cast<std::int32_t>(simulated_time);
      command.face_stamp.nanosec = static_cast<std::uint32_t>(
        (simulated_time - static_cast<double>(command.face_stamp.sec)) * 1.0e9);
      const JointMotionState current = motion_state(state);
      const JointMotionState actual = actual_state_override ?
        motion_state(*actual_state_override) : current;
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(
        state, command, *planning_scene, simulated_time, kSceneRevision,
        actual, simulated_time, current);
    };
  const auto solved_qp = []() {
      QpResult result;
      result.status = QpStatus::kSolved;
      result.achieved_primary_velocity = Eigen::VectorXd::Zero(1);
      return result;
    };
  const auto solved_full_face_qp = []() {
      QpResult result;
      result.status = QpStatus::kSolved;
      result.achieved_primary_velocity = Eigen::VectorXd::Zero(3);
      return result;
    };
  const auto record_aligned_full_face = [this, &solved_full_face_qp](
    const MotionReferenceResult & measured, const double time_sec) {
      ASSERT_TRUE(measured.tracking);
      auto stable = *measured.tracking;
      stable.signed_face_primary_error_rad.setZero();
      stable.face_primary_feedforward_radps.setZero();
      reference_->recordFaceCommand(
        stable, solved_full_face_qp(), time_sec, true, measured.actual_face_alignment);
    };

  MotionReferenceResult first_path;
  const auto first_plan_deadline = std::chrono::steady_clock::now() + 3s;
  do {
    first_path = update_at(start, aligned_face(start));
    if (reference_->diagnostics().plans_accepted >= 1U) {
      break;
    }
    std::this_thread::sleep_for(2ms);
  } while (std::chrono::steady_clock::now() < first_plan_deadline);
  ASSERT_EQ(reference_->diagnostics().compact_generation, 1U);
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U)
    << reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(first_path.follows_path);
  ASSERT_TRUE(first_path.task);
  EXPECT_EQ(first_path.task->primary_matrix.rows(), 6);
  EXPECT_TRUE(first_path.task->primary_matrix.isApprox(Eigen::MatrixXd::Identity(6, 6)));
  ASSERT_GE(
    reference_->diagnostics().planned_path_max_roll_error_rad,
    std::abs(start_upright->signed_roll_error_rad));
  EXPECT_LE(
    reference_->diagnostics().planned_path_max_roll_error_rad,
    kCompactRollCapTestRad + kCompactPitchValidationToleranceRad);
  ASSERT_EQ(first_path.compact_roll_constraints.size(), 2U);
  ASSERT_EQ(first_path.compact_pitch_constraints.size(), 2U);
  const auto joint6 = std::find(
    group_->getVariableNames().begin(), group_->getVariableNames().end(), "joint6");
  ASSERT_NE(joint6, group_->getVariableNames().end());
  const auto joint6_index = std::distance(group_->getVariableNames().begin(), joint6);
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  auto roll_probe = start;
  roll_probe.setVariablePosition("joint6", roll_probe.getVariablePosition("joint6") + 1.0e-4);
  roll_probe.update(true);
  const auto probe_upright = evaluateCompactUprightPose(
    roll_probe.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(probe_upright);
  const double q6_direction =
    std::abs(probe_upright->signed_roll_error_rad) >
    std::abs(start_upright->signed_roll_error_rad) ? 1.0 : -1.0;
  Eigen::VectorXd beyond_roll_cap = start_position;
  beyond_roll_cap[joint6_index] += q6_direction * 0.06;
  auto beyond_roll_state = start;
  beyond_roll_state.setJointGroupPositions(group_, beyond_roll_cap);
  beyond_roll_state.update(true);
  const auto beyond_roll_pose = evaluateCompactUprightPose(
    beyond_roll_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(beyond_roll_pose);
  ASSERT_GT(std::abs(beyond_roll_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  std::string compact_violation;
  EXPECT_FALSE(reference_->validateCompactPathPosition(
      start, beyond_roll_cap, &compact_violation));
  EXPECT_EQ(compact_violation, "screen roll");

  auto commit_events = reference_->drainCompactEvents();
  const auto trigger = std::find_if(
    commit_events.begin(), commit_events.end(), [](const auto & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(trigger, commit_events.end());
  const auto committed_goal = goalVectorFromDetails(trigger->details_json);
  ASSERT_TRUE(committed_goal);
  std::string endpoint_violation;
  EXPECT_TRUE(reference_->validateCompactPathPosition(start, *committed_goal, &endpoint_violation))
    << endpoint_violation;

  // Advance synthetic feedback in small steps along the follower's checked-path
  // reference. This exercises lifecycle policy without simulating the QP or a
  // physical controller, and respects the follower's bounded projection window.
  moveit::core::RobotState current(start);
  MotionReferenceResult return_step;
  constexpr double kSyntheticFeedbackStepRad{0.005};
  for (int tick = 0; tick < 100; ++tick) {
    return_step = update_at(current, misaligned_face(current));
    if (reference_->diagnostics().compact_phase == "RETURN_FACE") {
      break;
    }
    EXPECT_EQ(reference_->diagnostics().paths_abandoned, 0U)
      << "checked path lost during incremental feedback, tick=" << tick
      << ", state=" << reference_->diagnostics().state
      << ", progress=" << reference_->diagnostics().path_progress_rad
      << ", distance=" << reference_->diagnostics().path_distance_rad;
    ASSERT_TRUE(return_step.task);
    ASSERT_EQ(return_step.task->primary_matrix.rows(), 6);
    ASSERT_TRUE(return_step.task->primary_matrix.isApprox(
        Eigen::MatrixXd::Identity(6, 6)));

    Eigen::VectorXd feedback_step = return_step.task->primary_reference;
    ASSERT_EQ(feedback_step.size(), 6);
    ASSERT_TRUE(feedback_step.allFinite());
    const double largest_step = feedback_step.cwiseAbs().maxCoeff();
    ASSERT_GT(largest_step, 1.0e-9)
      << "follower produced no reference step before lifecycle completion";
    feedback_step *= std::min(1.0, kSyntheticFeedbackStepRad / largest_step);
    Eigen::VectorXd current_position;
    current.copyJointGroupPositions(group_, current_position);
    Eigen::VectorXd next_position = current_position + feedback_step;
    std::string feedback_violation;
    ASSERT_TRUE(reference_->validateCompactPathPosition(
        start, next_position, &feedback_violation))
      << feedback_violation;
    current.setJointGroupPositions(group_, next_position);
    current.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    current.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    current.update(true);
    if (reference_->diagnostics().compact_phase == "SETTLING" &&
      (next_position - *committed_goal).lpNorm<Eigen::Infinity>() <= kSyntheticFeedbackStepRad)
    {
      current.setJointGroupPositions(group_, *committed_goal);
      current.update(true);
    }
  }
  ASSERT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE")
    << "state=" << reference_->diagnostics().state
    << ", plan='" << reference_->diagnostics().last_plan_message
    << "', paths_abandoned=" << reference_->diagnostics().paths_abandoned
    << ", path_progress=" << reference_->diagnostics().path_progress_rad
    << ", path_distance=" << reference_->diagnostics().path_distance_rad;
  EXPECT_EQ(reference_->diagnostics().paths_completed, 1U);
  EXPECT_EQ(reference_->diagnostics().paths_abandoned, 0U);
  EXPECT_TRUE(return_step.compact_pitch_constraints.empty());
  EXPECT_EQ(return_step.compact_roll_constraints.size(), 2U);
  EXPECT_FALSE(return_step.compact_radius_constraints.empty());
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_pitch_cap_rad));
  auto actual_misaligned = current;
  actual_misaligned.setVariablePosition(
    "joint1", actual_misaligned.getVariablePosition("joint1") + 0.25);
  actual_misaligned.update(true);
  const auto command_aligned_actual_misaligned = update_at(
    current, aligned_face(current), &actual_misaligned);
  ASSERT_TRUE(command_aligned_actual_misaligned.tracking);
  ASSERT_TRUE(command_aligned_actual_misaligned.task);
  EXPECT_EQ(command_aligned_actual_misaligned.task->primary_matrix.rows(), 1);
  EXPECT_LE(command_aligned_actual_misaligned.tracking->pointing_error_rad,
    kFaceTrackingErrorGuardRad);
  ASSERT_TRUE(command_aligned_actual_misaligned.actual_face_alignment);
  EXPECT_GT(command_aligned_actual_misaligned.actual_face_alignment->pointing_error_rad,
    kFaceTrackingErrorGuardRad);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);
  reference_->recordFaceCommand(
    *command_aligned_actual_misaligned.tracking, solved_full_face_qp(), simulated_time,
    true, command_aligned_actual_misaligned.actual_face_alignment);
  EXPECT_FALSE(reference_->diagnostics().compact_return_face_qualified);

  const auto first_goal = update_at(current, misaligned_face(current));
  ASSERT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  ASSERT_TRUE(reference_->compactManeuverActive());
  EXPECT_TRUE(reference_->diagnostics().compact_actual_settled);
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_effective_radius_cap_m));
  ASSERT_TRUE(first_goal.tracking);
  ASSERT_TRUE(first_goal.task);
  EXPECT_EQ(first_goal.task->primary_matrix.rows(), 1);
  EXPECT_EQ(first_goal.task->secondary_matrix.rows(), 2);
  EXPECT_TRUE(first_goal.task->primary_matrix.isApprox(
      first_goal.tracking->task.primary_matrix.row(2), 1.0e-12));
  EXPECT_GT(first_goal.tracking->pointing_error_rad, kFaceTrackingErrorGuardRad);
  EXPECT_LE(first_goal.tracking->roll_error_rad, kFaceTrackingErrorGuardRad);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);

  Eigen::MatrixXd screen_jacobian;
  ASSERT_TRUE(current.getJacobian(
      group_, model_->getLinkModel("monitor_control_frame"), Eigen::Vector3d::Zero(),
      screen_jacobian, false));
  const auto * root_link = group_->getJointModels().front()->getParentLinkModel();
  if (root_link) {
    const Eigen::Matrix3d root_rotation =
      current.getGlobalLinkTransform(root_link).linear();
    screen_jacobian.topRows(3) = (root_rotation * screen_jacobian.topRows(3)).eval();
    screen_jacobian.bottomRows(3) =
      (root_rotation * screen_jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig upright_config;
  upright_config.orientation_gain = parameters_.orientation_gain;
  upright_config.maximum_angular_reference_radps =
    parameters_.maximum_angular_reference_radps;
  const auto gravity_upright = makeUprightRollPathTask(
    current.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::VectorXd::Zero(6), upright_config);
  ASSERT_TRUE(gravity_upright);
  const auto * gaze_link = model_->getLinkModel(parameters_.gaze_frame);
  ASSERT_NE(gaze_link, nullptr);
  GazeKinematics gaze;
  gaze.pose = current.getGlobalLinkTransform(gaze_link);
  gaze.pose.linear() *= face_tracking_arm::opticalToPointingRotation();
  ASSERT_TRUE(current.getJacobian(
      group_, gaze_link, Eigen::Vector3d::Zero(), gaze.jacobian, false));
  if (root_link) {
    const Eigen::Matrix3d root_rotation =
      current.getGlobalLinkTransform(root_link).linear();
    gaze.jacobian.topRows(3) = (root_rotation * gaze.jacobian.topRows(3)).eval();
    gaze.jacobian.bottomRows(3) =
      (root_rotation * gaze.jacobian.bottomRows(3)).eval();
  }
  const auto first_goal_command = misaligned_face(current);
  const auto expected_return_pointing = makeFaceReturnPointingTask(
    current.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::Vector3d(first_goal_command.face.x, first_goal_command.face.y,
      first_goal_command.face.z), Eigen::Vector3d::Zero(), upright_config,
    first_goal.tracking->task.primary_matrix.topRows(2),
    first_goal.tracking->task.primary_reference.head(2), gaze);
  ASSERT_TRUE(expected_return_pointing);
  EXPECT_FALSE(expected_return_pointing->used_normal_fallback);
  EXPECT_TRUE(first_goal.task->secondary_matrix.isApprox(
      expected_return_pointing->matrix, 1.0e-12));
  const auto horizontal_edge = makeHorizontalEdgePathTask(
    current.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::VectorXd::Zero(6), upright_config);
  ASSERT_TRUE(horizontal_edge);
  EXPECT_TRUE(first_goal.task->primary_matrix.row(0).isApprox(
      horizontal_edge->task.primary_matrix.row(0), 1.0e-12));
  EXPECT_NEAR(first_goal.tracking->roll_error_rad,
    std::abs(horizontal_edge->edge_tilt_error_rad), 1.0e-12);
  EXPECT_TRUE(first_goal.tracking->task.primary_matrix.row(2).isApprox(
      horizontal_edge->task.primary_matrix.row(0), 1.0e-12));

  constexpr double kOldPoleCutoffRad{84.26 * 3.14159265358979323846 / 180.0};
  // Reuse a measured intermediate from the recorded NO_FACE→REST_PATH run
  // compact_v9_path_no_face_2s_v2 (stamp 39.65 s). Its xArm6 FK is 85.015°
  // compact pitch, while the horizontal edge remains within the 5.1° cap and
  // the assembly radius remains below the committed 0.40 m limit.
  auto return_pitch_state = current;
  const Eigen::VectorXd recorded_high_pitch_position = (Eigen::VectorXd(6) <<
    -0.0014021878365071967, -0.24328809639189844, -1.8555171377707995,
    -0.09275389103830937, -1.0749213995985931, 2.185574412337391).finished();
  return_pitch_state.setJointGroupPositions(group_, recorded_high_pitch_position);
  return_pitch_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  return_pitch_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  return_pitch_state.update(true);
  const Eigen::Vector3d return_normal =
    return_pitch_state.getGlobalLinkTransform("monitor_control_frame").linear().col(0);
  const double return_abs_pitch = std::atan2(
    std::abs(return_normal.z()), std::hypot(return_normal.x(), return_normal.y()));
  EXPECT_GT(return_abs_pitch, kOldPoleCutoffRad);
  const double committed_pitch_cap = reference_->diagnostics().compact_pitch_cap_rad;
  const double committed_radius_cap = reference_->diagnostics().compact_entry_radius_cap_m;
  ASSERT_GT(return_abs_pitch, committed_pitch_cap + 0.05);
  Eigen::VectorXd return_position;
  return_pitch_state.copyJointGroupPositions(group_, return_position);
  std::string return_radius_violation;
  ASSERT_TRUE(reference_->validateCompactPathPosition(
    return_pitch_state, return_position, &return_radius_violation)) <<
    return_radius_violation;
  ASSERT_TRUE(reference_->validateCommandPosition(
    return_pitch_state, return_position, &return_radius_violation)) <<
    return_radius_violation;

  const auto return_before_rearm = update_at(
    return_pitch_state, misaligned_face(return_pitch_state));
  ASSERT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  ASSERT_TRUE(return_before_rearm.tracking);
  ASSERT_TRUE(return_before_rearm.task);
  EXPECT_TRUE(return_before_rearm.compact_pitch_constraints.empty());
  EXPECT_EQ(return_before_rearm.compact_roll_constraints.size(), 3U);
  EXPECT_FALSE(return_before_rearm.compact_radius_constraints.empty());
  const std::uint64_t plans_before_return_rearm =
    reference_->diagnostics().plans_requested;

  reference_->reset(true);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_goal_reached, 1.0);
  EXPECT_FALSE(reference_->diagnostics().compact_actual_settled);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_return_face_dwell_sec, 0.0);
  EXPECT_DOUBLE_EQ(
    reference_->diagnostics().compact_pitch_cap_rad, committed_pitch_cap);
  EXPECT_DOUBLE_EQ(
    reference_->diagnostics().compact_entry_radius_cap_m, committed_radius_cap);
  const auto return_after_rearm = update_at(
    return_pitch_state, misaligned_face(return_pitch_state));
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, plans_before_return_rearm);
  EXPECT_TRUE(reference_->compactManeuverActive());
  ASSERT_TRUE(return_after_rearm.tracking);
  ASSERT_TRUE(return_after_rearm.task);
  EXPECT_EQ(return_after_rearm.task->primary_matrix.rows(), 1);
  EXPECT_TRUE(return_after_rearm.compact_pitch_constraints.empty());
  EXPECT_EQ(return_after_rearm.compact_roll_constraints.size(), 3U);
  EXPECT_FALSE(return_after_rearm.compact_radius_constraints.empty());

  reference_->recordFaceCommand(
    *first_goal.tracking, solved_qp(), simulated_time, true,
    first_goal.actual_face_alignment);
  auto lost_reacquisition = *first_goal.tracking;
  lost_reacquisition.pointing_error_rad = kFaceTrackingErrorGuardRad + 0.01;
  reference_->recordFaceCommand(
    lost_reacquisition, solved_qp(), simulated_time, true,
    first_goal.actual_face_alignment);
  const auto first_aligned_after_gap = update_at(current, aligned_face(current));
  ASSERT_TRUE(first_aligned_after_gap.tracking);
  ASSERT_TRUE(first_aligned_after_gap.task);
  EXPECT_EQ(first_aligned_after_gap.task->primary_matrix.rows(), 3);
  EXPECT_TRUE(reference_->diagnostics().compact_return_full_face_active);
  record_aligned_full_face(first_aligned_after_gap, simulated_time);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  for (int tick = 0; tick < 5; ++tick) {
    const auto returned_face = update_at(current, aligned_face(current));
    ASSERT_TRUE(returned_face.tracking);
    ASSERT_TRUE(returned_face.task);
    EXPECT_EQ(returned_face.task->primary_matrix.rows(), 3);
    EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
    record_aligned_full_face(returned_face, simulated_time);
  }
  EXPECT_NEAR(
    reference_->diagnostics().compact_return_face_dwell_sec,
    5.0 * parameters_.control_period_sec, 1.0e-9);

  const auto failed_publication = update_at(current, aligned_face(current));
  ASSERT_TRUE(failed_publication.tracking);
  reference_->recordFaceCommand(
    *failed_publication.tracking, solved_full_face_qp(), simulated_time, false,
    failed_publication.actual_face_alignment);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_return_face_dwell_sec, 0.0);

  const auto forecast_crossing = update_at(current, aligned_face(current));
  ASSERT_TRUE(forecast_crossing.tracking);
  auto crossing_tracking = *forecast_crossing.tracking;
  crossing_tracking.pointing_error_rad = 0.06;
  crossing_tracking.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
  crossing_tracking.face_primary_feedforward_radps = Eigen::Vector3d(0.2, 0.0, 0.0);
  reference_->recordFaceCommand(
    crossing_tracking, solved_full_face_qp(), simulated_time, true,
    forecast_crossing.actual_face_alignment);
  EXPECT_FALSE(reference_->diagnostics().compact_return_face_qualified);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_return_face_dwell_sec, 0.0);

  record_aligned_full_face(forecast_crossing, simulated_time);
  const int dwell_samples = static_cast<int>(std::ceil(
      0.30 / parameters_.control_period_sec));
  for (int tick = 0; tick < dwell_samples - 1; ++tick) {
    const auto returned_face = update_at(current, aligned_face(current));
    ASSERT_TRUE(returned_face.tracking);
    ASSERT_TRUE(returned_face.task);
    EXPECT_EQ(returned_face.task->primary_matrix.rows(), 3);
    EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
    record_aligned_full_face(returned_face, simulated_time);
  }
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  EXPECT_NEAR(
    reference_->diagnostics().compact_return_face_dwell_sec,
    0.30 - parameters_.control_period_sec, 1.0e-9);
  const auto settled_face = update_at(current, aligned_face(current));
  ASSERT_TRUE(settled_face.tracking);
  record_aligned_full_face(settled_face, simulated_time);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_phase, "COMPLETE");
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_goal_reached, 0.0);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);
  EXPECT_FALSE(reference_->diagnostics().compact_radius_floor_latched);
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_effective_radius_cap_m));

  const auto face_again = update_at(current, misaligned_face(current));
  ASSERT_TRUE(face_again.tracking);
  EXPECT_FALSE(face_again.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  auto shortfall = *face_again.tracking;
  shortfall.has_face_primary_roll = true;
  shortfall.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
  shortfall.face_primary_feedforward_radps.setZero();
  QpResult shortfall_qp;
  shortfall_qp.status = QpStatus::kSolved;
  shortfall_qp.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
  reference_->recordFaceCommand(shortfall, shortfall_qp, simulated_time, true);
  for (int tick = 0; tick < 8; ++tick) {
    const auto shortfall_sample = update_at(current, misaligned_face(current));
    ASSERT_TRUE(shortfall_sample.tracking);
    reference_->recordFaceCommand(
      shortfall, shortfall_qp,
      simulated_time, true);
  }
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_requested, plan_requests_before_lifecycle);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "COMPLETE");
  EXPECT_EQ(reference_->diagnostics().paths_completed, 1U);
  EXPECT_EQ(reference_->diagnostics().compact_generation, committed_generation);

  const auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "completed" && event.generation_id == 1U;
  }));
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "compact_goal_settled" && event.generation_id == 1U;
  }));
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "face_return_full_face_started" && event.generation_id == 1U;
  }));
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "pair_resource_exhausted" && event.generation_id == 2U;
  }));
}

TEST_F(MotionReferenceTest, NamedRestPostureHoldsWithoutPlanning)
{
  const auto robot_state = state();
  const auto result = update(robot_state, target(msg::TrackingTarget::REST));
  EXPECT_FALSE(result.task);
  EXPECT_EQ(reference_->diagnostics().state, "REST_HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, SearchIsPreemptedByHold)
{
  const auto robot_state = state(0.4);
  const auto search = update(robot_state, target(msg::TrackingTarget::SEARCH));
  EXPECT_FALSE(search.task);
  auto held = update(robot_state, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(held.task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
}

TEST_F(MotionReferenceTest, PreferredBaseAngleUsesJointRootFrame)
{
  Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
  base_pose.linear() = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d face = base_pose.translation() +
    base_pose.linear() * Eigen::Vector3d(0.0, 1.0, 0.0);
  EXPECT_NEAR(preferredBaseJointAngle(face, base_pose), 0.5 * M_PI, 1.0e-12);
}

TEST_F(MotionReferenceSearchDisabledTest, ExternalSearchIsRejectedWithDiagnostic)
{
  const auto result = update(state(), target(msg::TrackingTarget::SEARCH));
  EXPECT_FALSE(result.task);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_UNAVAILABLE");
}

TEST_F(MotionReferenceTest, RealClearancePredicateAllowsRestPlanningAndPathAdoption)
{
  const auto start = state(0.4);
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState motion{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto validity = CollisionConstraintBuilder(collision_config_).validateSegment(
      *planning_scene, start, *group_, motion, motion, 2);
    ASSERT_TRUE(validity.input_valid) << validity.failure_reason;
    ASSERT_FALSE(validity.unsafe) << validity.failure_reason;
  }
  const auto result = waitForRestPath(start);
  ASSERT_TRUE(result) << reference_->diagnostics().state << ": " <<
    reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(result->task);
  EXPECT_TRUE(result->follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "REST_PATH");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_EQ(result->task->primary_matrix.rows(), 1);
  ASSERT_EQ(result->task->secondary_matrix.rows(), 6);
  EXPECT_GT(result->task->secondary_reference.norm(), 0.0);
  EXPECT_LE(result->task->secondary_reference.cwiseAbs().maxCoeff(), 0.70 + 1.0e-9);
  EXPECT_DOUBLE_EQ(start.getVariablePosition("joint1"), 0.4);
}

TEST_F(MotionReferenceTest, HoldCancelsPendingRestWithoutAdoptingItsLateResult)
{
  expectPendingRestPreempted(msg::TrackingTarget::HOLD);
}

TEST_F(MotionReferenceTest, FaceCancelsPendingRestWithoutAdoptingItsLateResult)
{
  expectPendingRestPreempted(msg::TrackingTarget::FACE);
}

TEST_F(MotionReferenceTest, HoldImmediatelyPreemptsAnAdoptedRestPath)
{
  const auto start = state(0.4);
  ASSERT_TRUE(waitForRestPath(start)) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  const auto result = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(result.task);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}

TEST_F(MotionReferenceTest, FaceImmediatelyPreemptsAnAdoptedRestPath)
{
  const auto start = state(0.4);
  ASSERT_TRUE(waitForRestPath(start)) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  const auto result = update(start, target(msg::TrackingTarget::FACE));
  EXPECT_TRUE(result.tracking);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}


TEST_F(MotionReferenceTest, SearchUsesTheSameExecutorAndIsPreemptedByHoldAndFace)
{
  auto start = state();
  ASSERT_TRUE(start.setToDefaultValues(group_, "search"));
  start.update();
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  const auto scanned = update(start, search);
  ASSERT_TRUE(scanned.task);
  EXPECT_TRUE(scanned.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_SWEEP");
  EXPECT_EQ(scanned.task->primary_matrix.rows(), 1);
  ASSERT_EQ(scanned.task->secondary_matrix.rows(), 6);
  EXPECT_LE(scanned.task->secondary_reference.norm(), .30001);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  velocity[0] = .2;
  start.setJointGroupVelocities(group_, velocity);
  const auto held = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(held.task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_DOUBLE_EQ(start.getVariableVelocity("joint1"), .2);
  auto face = target(msg::TrackingTarget::FACE);
  face.face.x = 2.0;
  const auto tracked = update(start, face);
  EXPECT_FALSE(tracked.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  ASSERT_TRUE(tracked.tracking);
  EXPECT_EQ(tracked.tracking->task.primary_matrix.rows(), 3);
  EXPECT_DOUBLE_EQ(start.getVariableVelocity("joint1"), .2);
}

TEST_F(MotionReferenceTest, NearbySelectorIdentitySwitchClearsFaceVelocityHistory)
{
  const auto start = state();
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  auto command = target(msg::TrackingTarget::FACE);
  command.target_id = 17;
  command.pose.position.x = tcp.translation().x();
  command.pose.position.y = tcp.translation().y();
  command.pose.position.z = tcp.translation().z();
  command.pose.orientation.w = 1.0;
  command.face.x = tcp.translation().x() + 0.8;
  command.face.y = tcp.translation().y();
  command.face.z = tcp.translation().z();
  double simulated_time = 1.0;
  const auto update_at = [this, &simulated_time](
    const moveit::core::RobotState & state, const msg::TrackingTarget & target)
    {
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(state, target, *planning_scene, simulated_time, kSceneRevision);
    };
  command.face_stamp = rclcpp::Time(1'000'000'000);
  (void)update_at(start, command);
  simulated_time += 0.02;
  command.face_stamp = rclcpp::Time(1'020'000'000);
  command.face.y += 0.01;
  const auto moving_target = update_at(start, command);
  ASSERT_TRUE(moving_target.tracking);
  const auto inherited_feedforward = moving_target.tracking->task.primary_reference;
  ASSERT_GT(inherited_feedforward.norm(), 0.0);

  command.target_id = 18;
  simulated_time += 0.02;
  command.face_stamp = rclcpp::Time(1'040'000'000);
  const auto switched_target = update_at(start, command);
  ASSERT_TRUE(switched_target.tracking);

  MotionReference fresh_reference(
    node_, monitor_, parameters_, limits_, collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const auto fresh_target = fresh_reference.update(
    start, command, *planning_scene, simulated_time, kSceneRevision);
  ASSERT_TRUE(fresh_target.tracking);
  EXPECT_GT((inherited_feedforward - fresh_target.tracking->task.primary_reference).norm(), 1.0e-4);
  EXPECT_TRUE(switched_target.tracking->task.primary_reference.isApprox(
      fresh_target.tracking->task.primary_reference, 1.0e-12));
}

TEST_F(MotionReferenceTest, LocalSearchKeepsTheLastPostureAndStartsWithoutPlanning)
{
  auto start = state(.7);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "local_then_sweep";
  const auto scanned = update(start, search);
  ASSERT_TRUE(scanned.task);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_LOCAL");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_EQ(scanned.task->primary_matrix.rows(), 1);
  ASSERT_EQ(scanned.task->secondary_matrix.rows(), 6);
  EXPECT_LT(scanned.task->secondary_reference.tail(5).norm(), 1e-9);
}

TEST_F(MotionReferenceTest, FoldedRestAcquisitionChecksEveryEdgeAndBrakeTailStep)
{
  constexpr double kPi = 3.14159265358979323846;
  const Eigen::VectorXd folded_position = (Eigen::VectorXd(6) <<
    6.1837015398159085, -1.1781137192602429, -0.26176528921742126,
    -1.7523105690035137, -1.047197551196452, 2.0559978588493215).finished();
  auto folded = state();
  folded.setJointGroupPositions(group_, folded_position);
  folded.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  folded.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  folded.update(true);
  const Eigen::Matrix3d folded_rotation =
    folded.getGlobalLinkTransform("monitor_control_frame").linear();
  ASSERT_GT(std::abs(folded_rotation(2, 1)), std::sin(5.1 * kPi / 180.0));
  ASSERT_GT(folded_rotation(2, 2), 0.0);

  const auto acquisition = update(folded, target(msg::TrackingTarget::REST));
  ASSERT_TRUE(acquisition.task);
  EXPECT_EQ(reference_->diagnostics().state, "REST_UPRIGHT_START");
  EXPECT_EQ(acquisition.task->primary_matrix.rows(), 1);
  ASSERT_EQ(acquisition.task->primary_reference.size(), 1);
  EXPECT_GT(std::abs(acquisition.task->primary_reference[0]), 1.0e-6);

  Eigen::VectorXd rest_position;
  auto upright_rest = state();
  upright_rest.copyJointGroupPositions(group_, rest_position);
  const Eigen::VectorXd improving_position =
    0.5 * (folded_position + rest_position);
  // Acquisition remains active across moving ticks; settled evidence is only
  // required when entering from an invalid stationary pose.
  auto moving_acquisition_state = state();
  moving_acquisition_state.setJointGroupPositions(group_, improving_position);
  Eigen::VectorXd moving_velocity = rest_position - improving_position;
  moving_velocity.normalize();
  moving_velocity *= 0.02;
  moving_acquisition_state.setJointGroupVelocities(group_, moving_velocity);
  moving_acquisition_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  moving_acquisition_state.update(true);
  const auto moving_acquisition = update(
    moving_acquisition_state, target(msg::TrackingTarget::REST));
  ASSERT_TRUE(moving_acquisition.task);
  EXPECT_EQ(reference_->diagnostics().state, "REST_UPRIGHT_START");

  std::string violation;
  ASSERT_TRUE(reference_->validateCommandTransition(
      folded, folded_position, improving_position, &violation)) << violation;
  ASSERT_TRUE(reference_->validateCommandTransition(
      folded, improving_position, rest_position, &violation)) << violation;

  // Model a brake tail that first improves, enters the cap, and then rolls
  // back outside it. Every point is checked against its immediate predecessor.
  EXPECT_FALSE(reference_->validateCommandTransition(
      folded, improving_position, folded_position, &violation));
  EXPECT_EQ(violation, "screen roll acquisition must not worsen");
  EXPECT_FALSE(reference_->validateCommandTransition(
      folded, rest_position, improving_position, &violation));
  EXPECT_EQ(violation, "screen roll");

  const Eigen::VectorXd less_improved_position =
    0.75 * folded_position + 0.25 * rest_position;
  EXPECT_FALSE(reference_->validateCommandTransition(
      folded, improving_position, less_improved_position, &violation));
  EXPECT_EQ(violation, "screen roll acquisition must not worsen");

  const Eigen::VectorXd upside_down_position = (Eigen::VectorXd(6) <<
    -2.8666980890966247, -0.03227592388764711, -3.589730953877382,
    -2.0265347701430265, -1.3432677139218883, -6.144498228217451).finished();
  EXPECT_FALSE(reference_->validateCommandTransition(
      folded, folded_position, upside_down_position, &violation));
  EXPECT_EQ(violation, "screen roll");
  EXPECT_FALSE(reference_->validateCommandTransition(
      folded, upside_down_position, improving_position, &violation));
  EXPECT_EQ(violation, "screen roll");

  // Exercise the real QP -> fixed-period candidate -> complete emergency-tail
  // admission chain from a moving acquisition state.
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  bounds.reserve(parameters_.joint_names.size());
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{
      variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, improving_position, 11U, 1U);

  const Eigen::VectorXd toward_rest_velocity = moving_velocity;
  moveit_servo::KinematicState current(6);
  current.joint_names = parameters_.joint_names;
  current.positions = improving_position;
  current.velocities = toward_rest_velocity;
  current.accelerations = Eigen::VectorXd::Zero(6);
  current.time_stamp = component->now();
  const JointMotionState current_motion{
    improving_position, toward_rest_velocity, Eigen::VectorXd::Zero(6)};
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult qp = solver.solve(*moving_acquisition.task, current_motion, limits_);
  ASSERT_TRUE(qp.command_available()) << static_cast<int>(qp.status);
  EXPECT_GT(qp.joint_velocity.norm(), 1.0e-5);

  std::string candidate_reason;
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
  const auto accepted = CollisionAwareServoComponentTestPeer::validateCandidate(
    *component, std::make_shared<moveit::core::RobotState>(moving_acquisition_state),
    current, current_motion, qp.joint_velocity, *planning_scene, candidate_reason);
  ASSERT_TRUE(accepted) << candidate_reason;
  ASSERT_TRUE(accepted->braking_tail.command_available());
  ASSERT_FALSE(accepted->braking_tail.points.empty());
}

}  // namespace
}  // namespace face_tracking_arm::control
