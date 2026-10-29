// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <geometric_shapes/shapes.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
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

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "collision_aware_servo_component.hpp"
#include "collision_stopping.hpp"
#include "face_tracking_arm/face_tracking_policy.hpp"
#include "face_tracking_arm/motion_reference.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"
#include "face_tracking_arm/horizontal_edge_upright.hpp"
#include "face_tracking_arm/qp_failure_snapshot_io.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"
#include "face_tracking_arm/posture_preference.hpp"
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"

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

  [[nodiscard]] static double comfortableJoint3Lower(const MotionReference & reference)
  {
    return reference.posture_preference_config_.comfortable_joint3_lower;
  }

  [[nodiscard]] static double comfortableJoint3Upper(const MotionReference & reference)
  {
    return reference.posture_preference_config_.comfortable_joint3_upper;
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

  static bool installFaceFollowerPathForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const std::vector<Eigen::VectorXd> & path, const double time_sec)
  {
    Eigen::VectorXd position;
    state.copyJointGroupPositions(reference.group_, position);
    return reference.follower_.setPath(path, position, time_sec);
  }

  static bool installSearchFollowerPathForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const std::vector<Eigen::VectorXd> & path, const double time_sec)
  {
    Eigen::VectorXd position;
    state.copyJointGroupPositions(reference.group_, position);
    reference.planner_->cancel();
    reference.pending_generation_.reset();
    reference.rest_mode_ = MotionReference::RestMode::kSearch;
    return reference.follower_.setPath(path, position, time_sec);
  }

  [[nodiscard]] static Eigen::VectorXd searchGoal(const MotionReference & reference)
  {
    return reference.search_->goal();
  }

  [[nodiscard]] static bool plannerBusy(const MotionReference & reference)
  {
    return reference.planner_->isBusy();
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

  static void installTrajectoryPublisher(CollisionAwareServoComponent & component)
  {
    component.trajectory_publisher_ = component.create_publisher<
      trajectory_msgs::msg::JointTrajectory>("/test/command", rclcpp::QoS(1));
  }

  static std::size_t trajectorySubscriberCount(const CollisionAwareServoComponent & component)
  {
    return component.trajectory_publisher_ ?
           component.trajectory_publisher_->get_subscription_count() : 0U;
  }

  static void prepareStartupControlTick(
    CollisionAwareServoComponent & component, const bool warmup_complete)
  {
    component.commands_.startup_warmup_complete = warmup_complete;
    component.accept_pose_commands_.store(true, std::memory_order_release);
    component.trajectory_controller_active_.store(true, std::memory_order_release);
  }

  static std::size_t qpSampleCount(const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.qp_statistics.count();
  }

  static std::size_t pointingSampleCount(const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.pointing_error_statistics.count();
  }

  static std::uint64_t actualStateValidationCount(
    const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.actual_state_validation_count;
  }

  static std::uint64_t bootstrapSegmentValidationCount(
    const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.bootstrap_segment_validation_count;
  }

  static std::size_t bootstrapSourceStableSamples(
    const CollisionAwareServoComponent & component)
  {
    return component.commands_.bootstrap_source_stable_samples;
  }

  static bool epochActive(const CollisionAwareServoComponent & component)
  {
    return component.commands_.epoch_active;
  }

  static bool rearmPending(const CollisionAwareServoComponent & component)
  {
    return component.commands_.rearm_pending ||
           component.commands_.published_tail_rearm_pending;
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

  static bool setControllerTerminalTimeline(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const rclcpp::Time & trajectory_origin, const rclcpp::Time & reference_time,
    const rclcpp::Time & header_stamp)
  {
    control_msgs::msg::JointTrajectoryControllerState state;
    const auto set_stamp = [](
      const rclcpp::Time & value, builtin_interfaces::msg::Time & output) {
        output.sec = static_cast<std::int32_t>(value.nanoseconds() / 1000000000LL);
        output.nanosec = static_cast<std::uint32_t>(value.nanoseconds() % 1000000000LL);
      };
    const auto set_duration = [](
      const std::int64_t nanoseconds, builtin_interfaces::msg::Duration & output) {
        output.sec = static_cast<std::int32_t>(nanoseconds / 1000000000LL);
        output.nanosec = static_cast<std::uint32_t>(nanoseconds % 1000000000LL);
      };
    if (trajectory_origin.get_clock_type() != header_stamp.get_clock_type() ||
      reference_time.get_clock_type() != header_stamp.get_clock_type() ||
      trajectory_origin > reference_time || reference_time > header_stamp)
    {
      return false;
    }
    set_stamp(header_stamp, state.header.stamp);
    state.joint_names = component.parameters_.joint_names;
    const std::vector<double> values(positions.data(), positions.data() + positions.size());
    state.reference.positions = values;
    state.reference.velocities.assign(values.size(), 0.0);
    state.feedback.positions = values;
    state.feedback.velocities.assign(values.size(), 0.0);
    state.output.positions = values;
    state.output.velocities.assign(values.size(), 0.0);
    state.speed_scaling_factor = 1.0;
    set_duration((reference_time - trajectory_origin).nanoseconds(),
      state.reference.time_from_start);
    set_duration((header_stamp - trajectory_origin).nanoseconds(),
      state.feedback.time_from_start);
    std::lock_guard<std::mutex> lock(component.controller_state_mutex_);
    component.latest_controller_state_ = std::move(state);
    return true;
  }

  static bool setControllerHybridTimeline(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & reference_positions,
    const Eigen::VectorXd & output_positions, const Eigen::VectorXd & output_velocities,
    const rclcpp::Time & trajectory_origin, const rclcpp::Time & reference_time,
    const rclcpp::Time & header_stamp)
  {
    control_msgs::msg::JointTrajectoryControllerState state;
    const auto set_stamp = [](
      const rclcpp::Time & value, builtin_interfaces::msg::Time & output) {
        output.sec = static_cast<std::int32_t>(value.nanoseconds() / 1000000000LL);
        output.nanosec = static_cast<std::uint32_t>(value.nanoseconds() % 1000000000LL);
      };
    const auto set_duration = [](
      const std::int64_t nanoseconds, builtin_interfaces::msg::Duration & output) {
        std::int64_t seconds = nanoseconds / 1000000000LL;
        std::int64_t subsecond = nanoseconds % 1000000000LL;
        if (subsecond < 0) {
          --seconds;
          subsecond += 1000000000LL;
        }
        output.sec = static_cast<std::int32_t>(seconds);
        output.nanosec = static_cast<std::uint32_t>(subsecond);
      };
    if (trajectory_origin.get_clock_type() != header_stamp.get_clock_type() ||
      reference_time.get_clock_type() != header_stamp.get_clock_type() ||
      output_positions.size() != reference_positions.size() ||
      output_velocities.size() != reference_positions.size())
    {
      return false;
    }
    set_stamp(header_stamp, state.header.stamp);
    state.joint_names = component.parameters_.joint_names;
    const std::vector<double> reference_values(
      reference_positions.data(), reference_positions.data() + reference_positions.size());
    const std::vector<double> output_values(
      output_positions.data(), output_positions.data() + output_positions.size());
    const std::vector<double> velocity_values(
      output_velocities.data(), output_velocities.data() + output_velocities.size());
    state.reference.positions = reference_values;
    state.reference.velocities.assign(reference_values.size(), 0.0);
    state.feedback.positions = reference_values;
    state.feedback.velocities.assign(reference_values.size(), 0.0);
    state.output.positions = output_values;
    state.output.velocities = velocity_values;
    state.speed_scaling_factor = 1.0;
    set_duration((reference_time - trajectory_origin).nanoseconds(),
      state.reference.time_from_start);
    set_duration((header_stamp - trajectory_origin).nanoseconds(),
      state.feedback.time_from_start);
    std::lock_guard<std::mutex> lock(component.controller_state_mutex_);
    component.latest_controller_state_ = std::move(state);
    return true;
  }

  static std::optional<std::uint64_t> appendStationaryPublication(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const rclcpp::Time & start, const bool buffer_as_current)
  {
    std::deque<moveit_servo::KinematicState> queue;
    for (std::int64_t step = 0; step < 4; ++step) {
      moveit_servo::KinematicState point;
      point.joint_names = component.parameters_.joint_names;
      point.positions = positions;
      point.velocities = Eigen::VectorXd::Zero(positions.size());
      point.accelerations = Eigen::VectorXd::Zero(positions.size());
      point.time_stamp = start + rclcpp::Duration::from_nanoseconds(step * 10'000'000LL);
      queue.push_back(std::move(point));
    }
    const auto stationary_time = queue[2U].time_stamp;
    const auto id = component.commands_.publications.append(queue, stationary_time);
    if (id && buffer_as_current) {
      component.commands_.queue.clear();
      component.commands_.queue.push_back(queue.front());
      component.commands_.active_publication_id = id;
      component.commands_.epoch_active = true;
      component.commands_.epoch_has_published = true;
      component.commands_.startup_warmup_complete = true;
    }
    return id;
  }

  static std::optional<std::uint64_t> appendFinitePublication(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & start_positions,
    const Eigen::VectorXd & next_positions, const rclcpp::Time & start,
    const bool buffer_as_current)
  {
    std::deque<moveit_servo::KinematicState> queue;
    const double interval = 0.01;
    const Eigen::VectorXd interval_velocity = (next_positions - start_positions) / interval;
    for (std::int64_t step = 0; step < 4; ++step) {
      moveit_servo::KinematicState point;
      point.joint_names = component.parameters_.joint_names;
      point.positions = step == 0 ? start_positions : next_positions;
      if (step == 1) {
        point.velocities = interval_velocity;
      } else {
        point.velocities = Eigen::VectorXd::Zero(start_positions.size());
      }
      point.accelerations = Eigen::VectorXd::Zero(start_positions.size());
      point.time_stamp = start + rclcpp::Duration::from_nanoseconds(step * 10'000'000LL);
      queue.push_back(std::move(point));
    }
    const auto stationary_time = queue[2U].time_stamp;
    const auto id = component.commands_.publications.append(queue, stationary_time);
    if (id && buffer_as_current) {
      component.commands_.queue.clear();
      component.commands_.queue.push_back(queue.front());
      component.commands_.active_publication_id = id;
      component.commands_.epoch_active = true;
      component.commands_.epoch_has_published = true;
      component.commands_.startup_warmup_complete = true;
    }
    return id;
  }

  static std::uint64_t olderPublicationRecoveryCount(
    const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.older_publication_recovery_count;
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

  static bool installRecoveredCapacityQueue(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const rclcpp::Time & now)
  {
    constexpr std::int64_t period_ns = 10'000'000LL;
    constexpr std::size_t max_points = 13U;
    constexpr std::uint64_t recovered_periods = 12U;
    component.commands_ = {};
    component.commands_.time_grid = std::make_unique<control::ControlTimeGrid>(period_ns);
    component.commands_.max_queue_points = max_points;
    component.parameters_.residual_command_latency_sec = 0.0;
    component.servo_parameters_.max_expected_latency = 0.02;

    const rclcpp::Time anchor = now + rclcpp::Duration::from_nanoseconds(60'000'000LL);
    const rclcpp::Time first_stamp = anchor - rclcpp::Duration::from_nanoseconds(
      static_cast<std::int64_t>(recovered_periods) * period_ns);
    std::deque<moveit_servo::KinematicState> published_queue;
    for (std::size_t index = 0U; index < 20U; ++index) {
      moveit_servo::KinematicState point(static_cast<int>(positions.size()));
      point.joint_names = component.parameters_.joint_names;
      point.positions = positions;
      point.velocities = Eigen::VectorXd::Zero(positions.size());
      point.accelerations = Eigen::VectorXd::Zero(positions.size());
      point.time_stamp = first_stamp + rclcpp::Duration::from_nanoseconds(
        static_cast<std::int64_t>(index) * period_ns);
      published_queue.push_back(std::move(point));
    }
    if (!component.commands_.time_grid->commandStamp(
        first_stamp.nanoseconds() - period_ns, period_ns, period_ns))
    {
      return false;
    }
    const auto publication_id = component.commands_.publications.append(
      published_queue, anchor);
    if (!publication_id) {
      return false;
    }
    component.commands_.queue.push_back(published_queue.front());
    component.commands_.active_publication_id = publication_id;
    component.commands_.epoch_active = true;
    component.commands_.epoch_has_published = true;
    std::string reason;
    if (component.commands_.recoverTimeline(
        recovered_periods, now, reason, 0.05, positions.size()) !=
      TimelineRecoveryStatus::kSuccess)
    {
      return false;
    }
    const auto * recovered_publication = component.commands_.activePublication();
    if (recovered_publication == nullptr ||
      component.commands_.queue.size() != recovered_periods ||
      component.commands_.queue.back().time_stamp != anchor)
    {
      return false;
    }
    for (std::size_t index = 0U; index < recovered_periods; ++index) {
      const auto & actual = component.commands_.queue[index];
      const auto & expected = recovered_publication->execution_queue[index + 1U];
      if (actual.time_stamp != expected.time_stamp ||
        !actual.positions.isApprox(expected.positions) ||
        !actual.velocities.isApprox(expected.velocities) ||
        !actual.accelerations.isApprox(expected.accelerations))
      {
        return false;
      }
    }
    return true;
  }

  static CommandPublicationStatus appendStationaryCommand(
    CollisionAwareServoComponent & component, const Eigen::VectorXd & positions,
    const rclcpp::Time & now, std::string & reason)
  {
    if (!component.trajectory_publisher_) {
      component.trajectory_publisher_ = component.create_publisher<
        trajectory_msgs::msg::JointTrajectory>("/test/queue_capacity", rclcpp::QoS(1));
    }
    moveit_servo::KinematicState next_state(static_cast<int>(positions.size()));
    next_state.joint_names = component.parameters_.joint_names;
    next_state.positions = positions;
    next_state.velocities = Eigen::VectorXd::Zero(positions.size());
    next_state.accelerations = Eigen::VectorXd::Zero(positions.size());
    control::EmergencyBrakeTail tail;
    tail.status = control::EmergencyBrakeTailStatus::kSuccess;
    tail.first_stationary_point = 0U;
    for (std::size_t index = 0U; index < 3U; ++index) {
      tail.points.push_back(control::JointMotionState{
          positions, Eigen::VectorXd::Zero(positions.size()),
          Eigen::VectorXd::Zero(positions.size())});
    }
    const moveit_servo::KinematicState current_state = next_state;
    return component.enqueue_and_publish(
      current_state, ValidatedCommand{std::move(next_state), std::move(tail)}, now, reason);
  }

  static CommandPublicationStatus appendMismatchedTailCommand(
    CollisionAwareServoComponent & component,
    const Eigen::VectorXd & positions,
    const rclcpp::Time & now,
    std::string & reason)
  {
    moveit_servo::KinematicState next_state(static_cast<int>(positions.size()));
    next_state.joint_names = component.parameters_.joint_names;
    next_state.positions = positions;
    next_state.velocities = Eigen::VectorXd::Zero(positions.size());
    next_state.accelerations = Eigen::VectorXd::Zero(positions.size());
    control::EmergencyBrakeTail tail;
    tail.status = control::EmergencyBrakeTailStatus::kSuccess;
    tail.first_stationary_point = 0U;
    for (std::size_t index = 0U; index < 3U; ++index) {
      Eigen::VectorXd point_position = positions;
      if (index == 0U) {
        point_position[0] += 0.01;
      }
      tail.points.push_back(control::JointMotionState{
          point_position, Eigen::VectorXd::Zero(positions.size()),
          Eigen::VectorXd::Zero(positions.size())});
    }
    return component.enqueue_and_publish(
      next_state, ValidatedCommand{std::move(next_state), std::move(tail)}, now, reason);
  }

  static std::size_t commandQueueSize(const CollisionAwareServoComponent & component)
  {
    return component.commands_.queue.size();
  }

  static std::optional<control::PublicationId> activePublicationId(
    const CollisionAwareServoComponent & component)
  {
    return component.commands_.active_publication_id;
  }

  static std::size_t publicationHistorySize(
    const CollisionAwareServoComponent & component)
  {
    return component.commands_.publications.size();
  }

  static rclcpp::Time commandQueueBackStamp(const CollisionAwareServoComponent & component)
  {
    return component.commands_.queue.back().time_stamp;
  }

  static std::optional<std::int64_t> nextGridStamp(
    const CollisionAwareServoComponent & component, const rclcpp::Time & now)
  {
    auto grid = *component.commands_.time_grid;
    return grid.commandStamp(now.nanoseconds(), 0, 0);
  }

  static std::optional<std::uint64_t> periodsUntilGridTarget(
    const CollisionAwareServoComponent & component, const rclcpp::Time & now,
    const std::int64_t target_lead_ns)
  {
    return component.commands_.time_grid ?
           component.commands_.time_grid->periodsUntilCommandLead(
      now.nanoseconds(), target_lead_ns) : std::nullopt;
  }

  static rclcpp::Time commandQueueStampAt(
    const CollisionAwareServoComponent & component, const std::size_t index)
  {
    return component.commands_.queue[index].time_stamp;
  }

  static void setContinuationTarget(
    CollisionAwareServoComponent & component, const msg::TrackingTarget & target)
  {
    std::lock_guard<std::mutex> lock(component.target_mutex_);
    component.latest_pose_ = target;
  }

  static void setGazeboFeedbackForTest(CollisionAwareServoComponent & component)
  {
    component.require_hardware_io_status_ = false;
  }

  static TimelineRecoveryStatus resumeVerifiedTerminalHold(
    CollisionAwareServoComponent & component, const rclcpp::Time & current_time,
    const std::int64_t target_lead_ns, std::string & reason)
  {
    return component.commands_.resumeVerifiedTerminalHold(
      *component.commands_.active_publication_id, current_time, target_lead_ns,
      component.parameters_.joint_names.size(), true, reason);
  }

  static bool prepareVerifiedTerminalHoldPublication(
    CollisionAwareServoComponent & component, const rclcpp::Time & tick_time,
    const msg::TrackingTarget & target, const std::uint64_t scene_revision)
  {
    const auto expected_publication_id = component.commands_.active_publication_id;
    std::optional<planning_scene_monitor::LockedPlanningSceneRO> locked_scene;
    locked_scene.emplace(component.planning_scene_monitor_);
    planning_scene::PlanningSceneConstPtr scene = *locked_scene;
    const std::optional<std::uint64_t> hardware_epoch =
      component.cycle_hardware_io_status_.has_value() ?
      std::optional<std::uint64_t>{
      component.cycle_hardware_io_status_->sample.epoch} : std::nullopt;
    const std::optional<std::uint64_t> proven_hold_epoch =
      component.proven_q_hold_.has_value() ? component.proven_q_hold_epoch_ : std::nullopt;
    return component.prepare_verified_terminal_hold_publication(
      tick_time, target, expected_publication_id,
      component.commands_.queue.empty() ? std::nullopt :
      std::optional<rclcpp::Time>{component.commands_.queue.back().time_stamp},
      hardware_epoch, proven_hold_epoch, scene_revision, locked_scene, scene);
  }

  static std::string lastServoStatusMessage(
    const CollisionAwareServoComponent & component)
  {
    return component.telemetry_.last_servo_status_message;
  }

  static bool hasVerifiedTerminalHold(const CollisionAwareServoComponent & component)
  {
    return component.commands_.hasVerifiedTerminalHold(
      component.parameters_.joint_names.size());
  }

  static void setPlanningSceneRevision(
    CollisionAwareServoComponent & component, const std::uint64_t revision)
  {
    component.planning_scene_collision_revision_.store(revision, std::memory_order_release);
  }

  static void setMaximumExpectedLatency(
    CollisionAwareServoComponent & component, const double seconds)
  {
    component.servo_parameters_.max_expected_latency = seconds;
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

  static bool referenceFollowerActive(const CollisionAwareServoComponent & component)
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

  static std::optional<ValidatedCommand> validateCapturedCandidate(
    CollisionAwareServoComponent & component,
    const moveit::core::RobotStatePtr & robot_state,
    const moveit_servo::KinematicState & current_state,
    const JointMotionState & current_motion_state,
    const Eigen::VectorXd & joint_velocity,
    const planning_scene::PlanningScene & scene,
    const CollisionConstraintDiagnostics & diagnostics,
    std::string & reason,
    CandidateRejectionKind & rejection_kind)
  {
    return component.validated_candidate(
      robot_state, current_state, current_motion_state, joint_velocity, scene, true,
      CommandAttempt::kTracking, reason, diagnostics, rejection_kind);
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


  static Eigen::VectorXd exactBrakeAnchor(
    const CollisionAwareServoComponent & component, const JointMotionState & motion_state)
  {
    const auto tail = component.emergency_brake_tail_generator_->generate(
      motion_state, component.motion_limits_);
    if (!tail.command_available() || tail.points.size() < 2U) {
      return Eigen::VectorXd{};
    }
    return tail.points[1U].velocity;
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

struct Posture1CFixture
{
  int id{};
  bool good{};
  Eigen::VectorXd joints;
  Eigen::Vector3d shoulder;
  Eigen::Vector3d elbow;
  Eigen::Vector3d wrist;
  Eigen::Vector3d face;
};

std::vector<Posture1CFixture> readPosture1CFixtures()
{
  boost::property_tree::ptree document;
  std::istringstream json(readModelFile(TEST_POSTURE1C_POSES_FILE));
  boost::property_tree::read_json(json, document);
  const auto vector_field = [](const boost::property_tree::ptree & object,
    const std::string & key) {
      Eigen::Vector3d result;
      Eigen::Index index = 0;
      for (const auto & value : object.get_child(key)) {
        if (index >= result.size()) {
          throw std::runtime_error("Expected 3D posture fixture vector: " + key);
        }
        result[index++] = value.second.get_value<double>();
      }
      if (index != result.size()) {
        throw std::runtime_error("Expected 3D posture fixture vector: " + key);
      }
      return result;
    };
  std::vector<Posture1CFixture> result;
  for (const auto & entry : document.get_child("poses")) {
    const auto & object = entry.second;
    Posture1CFixture fixture;
    fixture.id = object.get<int>("id");
    const auto label = object.get<std::string>("label");
    if (label != "GOOD" && label != "BAD") {
      throw std::runtime_error("Invalid posture fixture label: " + label);
    }
    fixture.good = label == "GOOD";
    fixture.joints.resize(6);
    for (Eigen::Index joint = 0; joint < 6; ++joint) {
      const std::string key = "joint" + std::to_string(joint + 1);
      fixture.joints[joint] = object.get<double>("joints." + key);
    }
    const auto & mainchain = object.get_child("mainchain");
    fixture.shoulder = vector_field(mainchain, "S_link2_m");
    fixture.elbow = vector_field(mainchain, "E_link3_m");
    fixture.wrist = vector_field(mainchain, "W_link4_m");
    fixture.face = vector_field(object, "face_target_m");
    result.push_back(std::move(fixture));
  }
  return result;
}

double independentElbowFaceSide(
  const Eigen::Vector3d & origin, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist, const Eigen::Vector3d & face)
{
  const Eigen::Vector3d w = wrist - origin;
  const Eigen::Vector3d e = elbow - origin;
  const Eigen::Vector3d f = face - origin;
  return e.dot(f) - (e.dot(w) * f.dot(w)) / w.squaredNorm();
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
        <origin xyz="0.0095 0 0.12525" rpy="-1.57079632679 0 -1.57079632679"/>
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

TEST_F(MotionReferenceTest, InactiveWarmEpochSkipsTrackingWorkAndKeepsBootstrapGuards)
{
  monitor_->startStateMonitor("/joint_states");
  auto joint_state_publisher = node_->create_publisher<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::QoS(10).reliable());
  auto command_subscriber = node_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    "/test/command", rclcpp::QoS(1), [](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr) {});

  const auto hold = state();
  Eigen::VectorXd positions;
  hold.copyJointGroupPositions(group_, positions);
  const auto publish_joint_state = [&]() {
      sensor_msgs::msg::JointState message;
      message.header.stamp = node_->now();
      message.name = parameters_.joint_names;
      message.position.assign(positions.data(), positions.data() + positions.size());
      message.velocity.assign(positions.size(), 0.0);
      joint_state_publisher->publish(message);
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

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
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  CollisionAwareServoComponentTestPeer::installVelocityQp(*component, qp_config);
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(reference_));
  CollisionAwareServoComponentTestPeer::installStatusPublisher(*component);
  CollisionAwareServoComponentTestPeer::installTrajectoryPublisher(*component);
  CollisionAwareServoComponentTestPeer::prepareStartupControlTick(*component, false);
  for (std::uint64_t sequence = 1U; sequence <= 3U; ++sequence) {
    publish_joint_state();
    CollisionAwareServoComponentTestPeer::publishHardwareSample(
      *component, positions, 11U, sequence);
  }
  for (std::size_t attempt = 0U; attempt < 100U &&
    CollisionAwareServoComponentTestPeer::trajectorySubscriberCount(*component) == 0U;
    ++attempt)
  {
    rclcpp::spin_some(node_);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_GT(CollisionAwareServoComponentTestPeer::trajectorySubscriberCount(*component), 0U);

  auto face = faceAlongLookDirection(hold);
  const auto set_fresh_face = [&]() {
      const auto stamp = component->now();
      face.header.stamp = stamp;
      face.face_stamp = stamp;
      CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, face);
    };
  set_fresh_face();

  const auto qp_samples_before =
    CollisionAwareServoComponentTestPeer::qpSampleCount(*component);
  const auto pointing_samples_before =
    CollisionAwareServoComponentTestPeer::pointingSampleCount(*component);
  const auto actual_validations_before =
    CollisionAwareServoComponentTestPeer::actualStateValidationCount(*component);
  const auto bootstrap_segments_before =
    CollisionAwareServoComponentTestPeer::bootstrapSegmentValidationCount(*component);

  CollisionAwareServoComponentTestPeer::controlTick(*component);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::qpSampleCount(*component),
    qp_samples_before + 1U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::pointingSampleCount(*component),
    pointing_samples_before + 1U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::actualStateValidationCount(*component),
    actual_validations_before + 1U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::bootstrapSourceStableSamples(*component), 0U);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::epochActive(*component));

  CollisionAwareServoComponentTestPeer::prepareStartupControlTick(*component, true);
  ASSERT_FALSE(CollisionAwareServoComponentTestPeer::epochActive(*component));
  publish_joint_state();
  CollisionAwareServoComponentTestPeer::publishHardwareSample(
    *component, positions, 11U, 4U);
  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  set_fresh_face();
  const auto fast_qp_samples_before =
    CollisionAwareServoComponentTestPeer::qpSampleCount(*component);
  const auto fast_pointing_samples_before =
    CollisionAwareServoComponentTestPeer::pointingSampleCount(*component);
  const auto fast_actual_validations_before =
    CollisionAwareServoComponentTestPeer::actualStateValidationCount(*component);

  CollisionAwareServoComponentTestPeer::controlTick(*component);
  (void)command_subscriber;

  EXPECT_EQ(CollisionAwareServoComponentTestPeer::qpSampleCount(*component),
    fast_qp_samples_before);
  EXPECT_EQ(
    CollisionAwareServoComponentTestPeer::pointingSampleCount(*component),
    fast_pointing_samples_before);
  EXPECT_EQ(
    CollisionAwareServoComponentTestPeer::actualStateValidationCount(*component),
    fast_actual_validations_before + 1U);
  EXPECT_EQ(
    CollisionAwareServoComponentTestPeer::bootstrapSourceStableSamples(*component), 1U);
  EXPECT_EQ(
    CollisionAwareServoComponentTestPeer::bootstrapSegmentValidationCount(*component),
    bootstrap_segments_before);
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

TEST_F(MotionReferenceTest, FreshAppendPrunesOnlyExpiredRecoveredQueuePrefix)
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
  Eigen::VectorXd positions(6);
  state().copyJointGroupPositions(group_, positions);
  const rclcpp::Time now = component->now();

  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installRecoveredCapacityQueue(
      *component, positions, now));
  const rclcpp::Time recovered_branch =
    CollisionAwareServoComponentTestPeer::commandQueueBackStamp(*component);
  ASSERT_EQ(CollisionAwareServoComponentTestPeer::commandQueueSize(*component), 12U);

  std::string failure_reason;
  const auto status = CollisionAwareServoComponentTestPeer::appendStationaryCommand(
    *component, positions, now, failure_reason);

  ASSERT_EQ(status, CommandPublicationStatus::kSuccess) << failure_reason;
  const std::size_t queue_size =
    CollisionAwareServoComponentTestPeer::commandQueueSize(*component);
  ASSERT_LE(queue_size, 13U);
  ASSERT_GE(queue_size, 2U);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::commandQueueStampAt(
      *component, queue_size - 2U), recovered_branch);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::commandQueueBackStamp(*component),
    recovered_branch + rclcpp::Duration::from_nanoseconds(10'000'000LL));
}

TEST_F(MotionReferenceTest, FailedPostEnqueueValidationPreservesPublishedSuffixAndGrid)
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
  Eigen::VectorXd positions(6);
  state().copyJointGroupPositions(group_, positions);
  auto rest_reference = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  Eigen::VectorXd goal_position = positions;
  goal_position[0] += 0.01;
  ASSERT_TRUE(MotionReferenceTestPeer::installRestFollowerPathForTest(
      *rest_reference, state(), {positions, goal_position}, 0.0));
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(rest_reference));
  const rclcpp::Time start = component->now();
  const rclcpp::Time publication_start =
    start + rclcpp::Duration::from_nanoseconds(100'000'000LL);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installPublishedBrakeSuffix(
      *component, positions, publication_start, kSceneRevision));
  CollisionAwareServoComponentTestPeer::setMaximumExpectedLatency(*component, 0.03);
  const std::size_t queue_size =
    CollisionAwareServoComponentTestPeer::commandQueueSize(*component);
  const rclcpp::Time queue_back =
    CollisionAwareServoComponentTestPeer::commandQueueBackStamp(*component);
  const auto next_grid_stamp =
    CollisionAwareServoComponentTestPeer::nextGridStamp(*component, start);
  const auto active_publication_id =
    CollisionAwareServoComponentTestPeer::activePublicationId(*component);
  const std::size_t publication_history_size =
    CollisionAwareServoComponentTestPeer::publicationHistorySize(*component);
  const auto expected_continuation_stamp =
    CollisionAwareServoComponentTestPeer::nextPublishedBrakeStamp(*component, 1U);

  std::string failure_reason;
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::appendMismatchedTailCommand(
      *component, positions, start, failure_reason),
    CommandPublicationStatus::kInvariantFailure) << failure_reason;
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::commandQueueSize(*component), queue_size);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::commandQueueBackStamp(*component), queue_back);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::nextGridStamp(*component, start),
    next_grid_stamp);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::activePublicationId(*component),
    active_publication_id);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::publicationHistorySize(*component),
    publication_history_size);

  auto rest_target = target(msg::TrackingTarget::REST);
  rest_target.target_id = 1;
  rest_target.header.stamp = start;
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, rest_target);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::tryContinuePublishedBrake(
      *component, start, rest_target, kSceneRevision),
    PublishedBrakeContinuationStatus::kContinued);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::continuationQueueTailStamp(*component),
    expected_continuation_stamp);
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

TEST_F(MotionReferenceTest, ExhaustedFacePublishedSuffixPreservesFollowerUntilFreshProof)
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
  auto face_reference = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  const auto start_state = state();
  Eigen::VectorXd start_position;
  start_state.copyJointGroupPositions(group_, start_position);
  Eigen::VectorXd goal_position = start_position;
  goal_position[0] += 0.01;
  ASSERT_TRUE(MotionReferenceTestPeer::installFaceFollowerPathForTest(
    *face_reference, start_state, {start_position, goal_position}, 0.0));
  CollisionAwareServoComponentTestPeer::installMotionReference(
    *component, std::move(face_reference));

  const rclcpp::Time start = component->now();
  auto face_target = target(msg::TrackingTarget::FACE);
  face_target.target_id = 1;
  face_target.header.stamp = start;
  face_target.face_stamp = start;
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, face_target);
  const rclcpp::Time publication_start =
    start - rclcpp::Duration::from_nanoseconds(100'000'000LL);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installPublishedBrakeSuffix(
    *component, start_position, publication_start, kSceneRevision, 2U, 4U));
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::referenceFollowerActive(*component));

  const std::size_t queue_size =
    CollisionAwareServoComponentTestPeer::continuationQueueSize(*component);
  const auto queue_tail =
    CollisionAwareServoComponentTestPeer::continuationQueueTailStamp(*component);
  const auto grid_periods_to_fixed_target =
    CollisionAwareServoComponentTestPeer::periodsUntilGridTarget(
    *component, start, 1'000'000'000LL);
  const auto active_publication_id =
    CollisionAwareServoComponentTestPeer::activePublicationId(*component);
  ASSERT_TRUE(active_publication_id);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::epochActive(*component));
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::rearmPending(*component));

  EXPECT_EQ(CollisionAwareServoComponentTestPeer::tryContinuePublishedBrake(
    *component, start, face_target, kSceneRevision),
    PublishedBrakeContinuationStatus::kContinued);

  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::referenceFollowerActive(*component));
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::continuationQueueSize(*component), queue_size);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::continuationQueueTailStamp(*component),
    queue_tail);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::periodsUntilGridTarget(
      *component, start, 1'000'000'000LL), grid_periods_to_fixed_target);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::activePublicationId(*component),
    active_publication_id);
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::epochActive(*component));
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::rearmPending(*component));
}

TEST_F(MotionReferenceTest, VerifiedFaceTerminalHoldWaitsWithoutMutatingPublishedPath)
{
  monitor_->startStateMonitor("/joint_states");
  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
  std::vector<PositionBoundSet> bounds;
  for (const auto & name : parameters_.joint_names) {
    const auto & variable_bounds = model_->getVariableBounds(name);
    const PositionInterval physical{variable_bounds.min_position_, variable_bounds.max_position_};
    bounds.push_back(PositionBoundSet{physical, physical, physical, physical});
  }
  CollisionAwareServoComponentTestPeer::configure(
    *component, monitor_, group_, parameters_.joint_names, limits_, collision_config_, bounds);
  CollisionAwareServoComponentTestPeer::setGazeboFeedbackForTest(*component);
  CollisionAwareServoComponentTestPeer::installStatusPublisher(*component);

  const rclcpp::Time tick_time = component->now();
  auto original_target = target(msg::TrackingTarget::FACE);
  original_target.target_id = 1U;
  original_target.header.stamp = tick_time;
  original_target.face_stamp = tick_time;
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, original_target);

  const auto robot_state = state();
  Eigen::VectorXd positions;
  robot_state.copyJointGroupPositions(group_, positions);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::installPublishedBrakeSuffix(
    *component, positions,
    tick_time - rclcpp::Duration::from_nanoseconds(100'000'000LL), kSceneRevision, 2U, 4U));
  std::string resume_reason;
  ASSERT_EQ(CollisionAwareServoComponentTestPeer::resumeVerifiedTerminalHold(
      *component, tick_time - rclcpp::Duration::from_nanoseconds(50'000'000LL), 0,
      resume_reason), TimelineRecoveryStatus::kSuccess) << resume_reason;
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::hasVerifiedTerminalHold(*component));

  const std::size_t original_queue_size =
    CollisionAwareServoComponentTestPeer::commandQueueSize(*component);
  const auto original_queue_back =
    CollisionAwareServoComponentTestPeer::commandQueueBackStamp(*component);
  const auto original_publication_id =
    CollisionAwareServoComponentTestPeer::activePublicationId(*component);
  const std::size_t original_publication_count =
    CollisionAwareServoComponentTestPeer::publicationHistorySize(*component);
  const auto original_grid_stamp =
    CollisionAwareServoComponentTestPeer::nextGridStamp(*component, tick_time);
  ASSERT_TRUE(original_publication_id);

  // The full candidate and its braking tail still receive ordinary collision certification.
  moveit_servo::KinematicState current(static_cast<int>(positions.size()));
  current.joint_names = parameters_.joint_names;
  current.positions = positions;
  current.velocities = Eigen::VectorXd::Zero(positions.size());
  current.accelerations = Eigen::VectorXd::Zero(positions.size());
  current.time_stamp = tick_time;
  const JointMotionState stationary{
    positions, current.velocities, current.accelerations};
  std::string validation_reason;
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
    ASSERT_TRUE(locked_scene);
    const planning_scene::PlanningSceneConstPtr scene = locked_scene;
    const auto certified = CollisionAwareServoComponentTestPeer::validateCandidate(
      *component, std::make_shared<moveit::core::RobotState>(robot_state), current,
      stationary, current.velocities, *scene, validation_reason);
    ASSERT_TRUE(certified) << validation_reason;
    ASSERT_TRUE(certified->braking_tail.command_available());
  }

  auto stale_target = original_target;
  stale_target.header.stamp = tick_time - rclcpp::Duration::from_seconds(1.0);
  stale_target.face_stamp = stale_target.header.stamp;
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::prepareVerifiedTerminalHoldPublication(
    *component, tick_time, stale_target, kSceneRevision));
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::activePublicationId(*component),
    original_publication_id);

  auto changed_target = original_target;
  changed_target.target_id = 2U;
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, changed_target);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::prepareVerifiedTerminalHoldPublication(
    *component, tick_time, original_target, kSceneRevision));
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, original_target);

  // Same FACE identity with a newer accepted sample remains eligible. Missing fresh
  // controller proof waits while preserving the already-published stationary branch.
  auto refreshed_target = original_target;
  refreshed_target.header.stamp = component->now();
  refreshed_target.face_stamp = refreshed_target.header.stamp;
  CollisionAwareServoComponentTestPeer::setContinuationTarget(*component, refreshed_target);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::prepareVerifiedTerminalHoldPublication(
    *component, tick_time, original_target, kSceneRevision));
  EXPECT_NE(CollisionAwareServoComponentTestPeer::lastServoStatusMessage(*component).find(
      "Safe re-arm is waiting"), std::string::npos);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::commandQueueSize(*component),
    original_queue_size);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::commandQueueBackStamp(*component),
    original_queue_back);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::activePublicationId(*component),
    original_publication_id);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::publicationHistorySize(*component),
    original_publication_count);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::nextGridStamp(*component, tick_time),
    original_grid_stamp);
  EXPECT_TRUE(CollisionAwareServoComponentTestPeer::hasVerifiedTerminalHold(*component));
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
    qp_config.period_sec = parameters_.control_period_sec;
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

TEST_F(MotionReferenceTest, PredictKeepsLatestBufferedBranchDuringPendingTerminalReplacement)
{
  monitor_->startStateMonitor("/joint_states");
  auto joint_state_publisher = node_->create_publisher<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::QoS(10).reliable());
  auto hold_robot_state = state();
  Eigen::VectorXd hold;
  hold_robot_state.copyJointGroupPositions(group_, hold);
  hold_robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(hold.size()));
  hold_robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(hold.size()));
  hold_robot_state.update();

  sensor_msgs::msg::JointState joint_state;
  joint_state.header.stamp = node_->now();
  joint_state.name = parameters_.joint_names;
  joint_state.position.assign(hold.data(), hold.data() + hold.size());
  joint_state.velocity.assign(hold.size(), 0.0);
  joint_state_publisher->publish(joint_state);
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const auto state_monitor = monitor_->getStateMonitor();
  ASSERT_TRUE(state_monitor);
  ASSERT_TRUE(state_monitor->haveCompleteState(
      node_->now() - rclcpp::Duration::from_seconds(0.5)));

  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
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
  CollisionAwareServoComponentTestPeer::publishHardwareSample(*component, hold, 21U, 1U);

  const rclcpp::Time header_stamp = component->now();
  const rclcpp::Time old_origin = header_stamp - rclcpp::Duration::from_nanoseconds(200'000'000LL);
  const rclcpp::Time replacement_start = header_stamp +
    rclcpp::Duration::from_nanoseconds(1'000'000'000LL);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::appendStationaryPublication(
      *component, hold, old_origin, false).has_value());
  const auto latest_id = CollisionAwareServoComponentTestPeer::appendStationaryPublication(
    *component, hold, replacement_start, true);
  ASSERT_TRUE(latest_id.has_value());
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::setControllerTerminalTimeline(
      *component, hold, old_origin, header_stamp, header_stamp));

  std::string reason;
  bool requires_latched_halt = false;
  const auto predicted = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  ASSERT_TRUE(predicted.has_value()) << reason;
  EXPECT_FALSE(requires_latched_halt);
  EXPECT_EQ(predicted->time_stamp.nanoseconds(), replacement_start.nanoseconds());
  EXPECT_LT((predicted->positions - hold).norm(), 1.0e-12);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::activePublicationId(*component), latest_id);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::olderPublicationRecoveryCount(*component), 0U);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::rearmPending(*component));
}

TEST_F(MotionReferenceTest, PredictAcceptsFutureOriginHybridPendingTerminalHandoff)
{
  monitor_->startStateMonitor("/joint_states");
  auto joint_state_publisher = node_->create_publisher<sensor_msgs::msg::JointState>(
    "/joint_states", rclcpp::QoS(10).reliable());
  auto hold_robot_state = state();
  Eigen::VectorXd hold;
  hold_robot_state.copyJointGroupPositions(group_, hold);
  hold_robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(hold.size()));
  hold_robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(hold.size()));
  hold_robot_state.update();

  sensor_msgs::msg::JointState joint_state;
  joint_state.header.stamp = node_->now();
  joint_state.name = parameters_.joint_names;
  joint_state.position.assign(hold.data(), hold.data() + hold.size());
  joint_state.velocity.assign(hold.size(), 0.0);
  joint_state_publisher->publish(joint_state);
  rclcpp::spin_some(node_);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const auto state_monitor = monitor_->getStateMonitor();
  ASSERT_TRUE(state_monitor);
  ASSERT_TRUE(state_monitor->haveCompleteState(
      node_->now() - rclcpp::Duration::from_seconds(0.5)));

  auto component = std::make_shared<CollisionAwareServoComponent>(rclcpp::NodeOptions{});
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
  CollisionAwareServoComponentTestPeer::publishHardwareSample(*component, hold, 21U, 1U);

  const rclcpp::Time header_stamp = component->now();
  const rclcpp::Time old_origin =
    header_stamp - rclcpp::Duration::from_nanoseconds(200'000'000LL);
  const rclcpp::Time replacement_start =
    header_stamp + rclcpp::Duration::from_nanoseconds(2'000'000LL);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::appendStationaryPublication(
      *component, hold, old_origin, false).has_value());
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::appendStationaryPublication(
      *component, hold, replacement_start, false).has_value());
  const Eigen::VectorXd next_positions = hold +
    Eigen::VectorXd::Constant(hold.size(), 0.001);
  const Eigen::VectorXd interval_velocity =
    (next_positions - hold) / 0.01;
  const auto latest_id = CollisionAwareServoComponentTestPeer::appendFinitePublication(
    *component, hold, next_positions, replacement_start, true);
  ASSERT_TRUE(latest_id.has_value());

  const Eigen::VectorXd output_positions = hold +
    0.8 * (next_positions - hold);
  ASSERT_TRUE(CollisionAwareServoComponentTestPeer::setControllerHybridTimeline(
      *component, hold, output_positions, interval_velocity, replacement_start,
      header_stamp, header_stamp));

  std::string reason;
  bool requires_latched_halt = false;
  const auto predicted = CollisionAwareServoComponentTestPeer::predict(
    *component, reason, requires_latched_halt);
  ASSERT_TRUE(predicted.has_value()) << reason;
  EXPECT_FALSE(requires_latched_halt);
  EXPECT_LT((predicted->positions - hold).norm(), 1.0e-12);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::activePublicationId(*component), latest_id);
  EXPECT_EQ(CollisionAwareServoComponentTestPeer::olderPublicationRecoveryCount(*component), 0U);
  EXPECT_FALSE(CollisionAwareServoComponentTestPeer::rearmPending(*component));
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
  ASSERT_TRUE(result.tracking) << reference_->diagnostics().state;
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

  auto bad_state = state();
  bad_state.setVariablePosition("joint1", 6.20);
  bad_state.setVariablePosition("joint2", 0.233);
  bad_state.setVariablePosition("joint3", -2.30);
  bad_state.setVariablePosition("joint4", 0.0168);
  bad_state.setVariablePosition("joint5", -0.8338);
  bad_state.setVariablePosition("joint6", 6.27);
  bad_state.update(true);
  ASSERT_TRUE(bad_state.satisfiesBounds(group_));
  Eigen::VectorXd bad_position;
  bad_state.copyJointGroupPositions(group_, bad_position);
  const JointMotionState bad_settled{
    bad_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  reference_ = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  const auto bad_edge = update(
    bad_state, target(msg::TrackingTarget::FACE), nullptr, nullptr, bad_settled);
  ASSERT_TRUE(bad_edge.task);
  ASSERT_TRUE(bad_edge.j1_hold_position_rad);
  ASSERT_EQ(reference_->diagnostics().posture_mode, "recover");
  const auto bad_held_direction = static_cast<BaseJointDirection>(
    reference_->diagnostics().j1_hold_direction);
  std::optional<msg::TrackingTarget> bad_reversed_target;
  for (const double offset : {0.30, -0.30}) {
    auto candidate = targetWithYawOffset(bad_state, offset);
    const auto assessment = pairYawAssessment(bad_state, candidate);
    if (assessment.valid && assessment.demand_present &&
      assessment.j1.required_direction != BaseJointDirection::kNone &&
      assessment.j1.required_direction != bad_held_direction)
    {
      bad_reversed_target = std::move(candidate);
      break;
    }
  }
  ASSERT_TRUE(bad_reversed_target);
  const auto bad_reversed = update(
    bad_state, bad_reversed_target, nullptr, nullptr, bad_settled);
  ASSERT_TRUE(bad_reversed.task);
  EXPECT_FALSE(bad_reversed.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  EXPECT_EQ(reference_->diagnostics().posture_mode, "recover");

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  const auto collision = collision_builder.build(
    *scene, bad_state, *group_, bad_settled, limits_, parameters_.control_period_sec);
  auto constraints = collision.constraints;
  constraints.insert(
    constraints.end(), bad_reversed.compact_roll_constraints.begin(),
    bad_reversed.compact_roll_constraints.end());
  const auto bad_qp = solver.solve(
    *bad_reversed.task, bad_settled, limits_, constraints);
  ASSERT_TRUE(bad_qp.command_available());
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
  const Eigen::Vector3d monitor_position =
    robot_state.getGlobalLinkTransform("monitor_control_frame").translation();
  const double measured_monitor_face_distance = (monitor_position - target_face).norm();
  ASSERT_GT(measured_monitor_face_distance, parameters_.face_distance_max_m);
  face.face.x = target_face.x();
  face.face.y = target_face.y();
  face.face.z = target_face.z();
  const auto result = update(robot_state, face);
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_NEAR(result.tracking->pointing_error_rad, 0.0, 1.0e-8);
  ASSERT_TRUE(result.tracking->face_distance_interval);
  EXPECT_TRUE(result.tracking->face_distance_outside_band);
  EXPECT_EQ(result.task->secondary_matrix.rows(), 0);
  ASSERT_EQ(result.task->distance_matrix.rows(), 3)
    << "Out-of-band distance follows the face ray as a final soft task";
  ASSERT_EQ(result.task->preferred_primary_constraints.size(), 1U)
    << "A GOOD J3 comfort bound remains active while distance is out of band";
  EXPECT_NEAR(result.tracking->position_error_m,
    measured_monitor_face_distance - parameters_.face_distance_max_m, 1.0e-8);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_EQ(result.task->secondary_matrix.rows(), 0);
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
  EXPECT_TRUE(result.task->distance_matrix.isApprox(screen_jacobian.topRows(3)));
  const Eigen::Vector3d radial_direction =
    (monitor_position - target_face).normalized();
  const Eigen::Vector3d expected_linear_reference =
    radial_direction * result.tracking->face_distance_reference[0];
  EXPECT_TRUE(result.task->distance_reference.isApprox(expected_linear_reference));
  EXPECT_NEAR(radial_direction.dot(result.task->distance_reference),
    result.tracking->face_distance_reference[0], 1.0e-8);
  EXPECT_NEAR(((Eigen::Matrix3d::Identity() -
    radial_direction * radial_direction.transpose()) * result.task->distance_reference,
    Eigen::Vector3d::Zero()).norm(), 0.0, 1.0e-8);
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

TEST_F(MotionReferenceTest, Posture1CDistanceIntervalRemainsDiagnosticOnly)
{
  parameters_.position_deadband_m = 0.03;

  const auto robot_state = state();
  const Eigen::Vector3d screen =
    robot_state.getGlobalLinkTransform("monitor_control_frame").translation();
  auto target = faceAlongLookDirection(robot_state);
  const Eigen::Vector3d initial_face(target.face.x, target.face.y, target.face.z);
  const Eigen::Vector3d ray = (initial_face - screen).normalized();
  const double delta = parameters_.maximum_linear_reference_mps *
    parameters_.control_period_sec;
  ASSERT_GT(delta, 0.0);
  ASSERT_LT(2.0 * delta,
    parameters_.face_distance_max_m - parameters_.face_distance_min_m);

  const auto make_target_at_distance = [&](const double distance, const double stamp_sec) {
      auto command = target;
      const Eigen::Vector3d face = screen + distance * ray;
      command.face.x = face.x();
      command.face.y = face.y();
      command.face.z = face.z();
      const auto stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(stamp_sec * 1.0e9)), RCL_ROS_TIME);
      command.header.stamp = stamp;
      command.face_stamp = stamp;
      return command;
    };
  const auto update_at_distance = [&](const double distance) {
      reference_.reset();
      reference_ = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      constexpr double kStampSec = 100.0;
      return update(robot_state, make_target_at_distance(distance, kStampSec),
        nullptr, nullptr, std::nullopt, kSceneRevision, kStampSec);
    };
  const auto expect_no_distance_constraint = [](const auto & result) {
      ASSERT_TRUE(result.tracking);
      ASSERT_TRUE(result.task);
      for (const auto & constraint : result.task->secondary_constraints) {
        EXPECT_FALSE(constraint.coefficients.isApprox(
            result.tracking->face_distance_matrix.row(0).transpose(), 1.0e-8));
      }
    };
  const auto below = update_at_distance(
    parameters_.face_distance_min_m - 0.5 * delta);
  ASSERT_TRUE(below.tracking);
  ASSERT_TRUE(below.task);
  ASSERT_TRUE(below.tracking->face_distance_interval);
  ASSERT_TRUE(below.tracking->face_distance_outside_band);
  expect_no_distance_constraint(below);
  EXPECT_TRUE(below.task->primary_matrix.isApprox(
      below.tracking->task.primary_matrix, 1.0e-12));
  EXPECT_TRUE(below.task->primary_reference.isApprox(
      below.tracking->task.primary_reference, 1.0e-12));
  EXPECT_TRUE(below.task->primary_weights.isApprox(
      below.tracking->task.primary_weights, 1.0e-12));

  const auto above = update_at_distance(
    parameters_.face_distance_max_m + 0.5 * delta);
  ASSERT_TRUE(above.tracking);
  ASSERT_TRUE(above.task);
  ASSERT_TRUE(above.tracking->face_distance_interval);
  ASSERT_TRUE(above.tracking->face_distance_outside_band);
  expect_no_distance_constraint(above);
  EXPECT_TRUE(above.task->primary_matrix.isApprox(
      above.tracking->task.primary_matrix, 1.0e-12));
  EXPECT_TRUE(above.task->primary_reference.isApprox(
      above.tracking->task.primary_reference, 1.0e-12));
  EXPECT_TRUE(above.task->primary_weights.isApprox(
      above.tracking->task.primary_weights, 1.0e-12));

  const double in_band_distance = 0.5 * (
    parameters_.face_distance_min_m + parameters_.face_distance_max_m);
  const auto inside = update_at_distance(in_band_distance);
  ASSERT_TRUE(inside.tracking);
  ASSERT_TRUE(inside.task);
  ASSERT_TRUE(inside.tracking->face_distance_interval);
  ASSERT_FALSE(inside.tracking->face_distance_outside_band);
  expect_no_distance_constraint(inside);
  bool inside_has_radial_row = false;
  for (Eigen::Index row = 0; row < inside.task->secondary_matrix.rows(); ++row) {
    inside_has_radial_row = inside_has_radial_row ||
      inside.task->secondary_matrix.row(row).isApprox(
      inside.tracking->face_distance_matrix.row(0), 1.0e-8);
  }
  EXPECT_FALSE(inside_has_radial_row)
    << "Inside the band, posture stays active without an extra radial hold objective";

  const auto outside_shell = update_at_distance(
    parameters_.face_distance_min_m - 2.0 * delta);
  ASSERT_TRUE(outside_shell.tracking);
  ASSERT_TRUE(outside_shell.task);
  ASSERT_TRUE(outside_shell.tracking->face_distance_interval);
  EXPECT_TRUE(outside_shell.tracking->face_distance_outside_band);
  expect_no_distance_constraint(outside_shell);
  EXPECT_EQ(outside_shell.task->distance_matrix.rows(), 3);
  const double outside_distance = parameters_.face_distance_min_m - 2.0 * delta;
  const Eigen::Vector3d outside_face = screen + outside_distance * ray;
  const Eigen::Vector3d outside_radial = (screen - outside_face).normalized();
  EXPECT_NEAR(outside_radial.dot(outside_shell.task->distance_reference),
    outside_shell.tracking->face_distance_reference[0], 1.0e-8);
  EXPECT_NEAR(((Eigen::Matrix3d::Identity() -
    outside_radial * outside_radial.transpose()) *
    outside_shell.task->distance_reference).norm(), 0.0, 1.0e-8);
}

TEST_F(MotionReferenceTest, Posture1CApprovedFixturesUseWorldBaseOrigin)
{
  const auto fixtures = readPosture1CFixtures();
  ASSERT_EQ(fixtures.size(), 10U);
  const auto named_bend = [this](const std::string & name) {
      moveit::core::RobotState state(model_);
      if (!state.setToDefaultValues(group_, name)) {
        return std::optional<std::pair<double, double>>{};
      }
      state.update();
      Eigen::VectorXd q;
      state.copyJointGroupPositions(group_, q);
      const auto * joint_model = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint_model);
      const auto * parent = joint_model ? joint_model->getParentLinkModel() : nullptr;
      const auto * child = joint_model ? joint_model->getChildLinkModel() : nullptr;
      if (!revolute || !parent || !child || q.size() != 6) {
        return std::optional<std::pair<double, double>>{};
      }
      const Eigen::Vector3d shoulder = state.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow = state.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist = state.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = state.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      const Eigen::Vector3d upper_arm = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      return std::optional<std::pair<double, double>>{{q[2], std::atan2(
          axis.normalized().dot(upper_arm.cross(forearm)), upper_arm.dot(forearm))}};
    };
  const auto rest = named_bend("rest");
  const auto search = named_bend("search");
  ASSERT_TRUE(rest);
  ASSERT_TRUE(search);
  const auto calibration = calibratePostureBend(
    rest->first, rest->second, search->first, search->second,
    limits_.lower_position[2], limits_.upper_position[2]);
  ASSERT_TRUE(calibration.valid);
  int base_matches = 0;
  int shoulder_matches = 0;
  int bend_matches = 0;
  for (const auto & fixture : fixtures) {
    SCOPED_TRACE("POSTURE-1C fixture " + std::to_string(fixture.id));
    moveit::core::RobotState robot_state(model_);
    robot_state.setToDefaultValues();
    for (Eigen::Index joint = 0; joint < fixture.joints.size(); ++joint) {
      robot_state.setVariablePosition("joint" + std::to_string(joint + 1), fixture.joints[joint]);
    }
    robot_state.update();
    ASSERT_TRUE(robot_state.satisfiesBounds(group_));

    const Eigen::Vector3d base =
      robot_state.getGlobalLinkTransform("link_base").translation();
    const Eigen::Vector3d shoulder =
      robot_state.getGlobalLinkTransform("link2").translation();
    const Eigen::Vector3d elbow =
      robot_state.getGlobalLinkTransform("link3").translation();
    const Eigen::Vector3d wrist =
      robot_state.getGlobalLinkTransform("link4").translation();
    const Eigen::MatrixXd base_jacobian = Eigen::MatrixXd::Zero(3, 6);
    const auto world_linear_jacobian = [this, &robot_state](
      const std::string & link_name, Eigen::MatrixXd & output) {
        Eigen::MatrixXd full;
        if (!robot_state.getJacobian(
            group_, model_->getLinkModel(link_name), Eigen::Vector3d::Zero(), full, false) ||
          full.rows() != 6 || full.cols() != 6)
        {
          return false;
        }
        const auto * root = group_->getJointModels().front()->getParentLinkModel();
        output = full.topRows(3);
        if (root) {
          output = (robot_state.getGlobalLinkTransform(root).linear() * output).eval();
        }
        return output.allFinite();
      };
    Eigen::MatrixXd elbow_jacobian;
    Eigen::MatrixXd wrist_jacobian;
    ASSERT_TRUE(world_linear_jacobian("link3", elbow_jacobian));
    ASSERT_TRUE(world_linear_jacobian("link4", wrist_jacobian));
    EXPECT_TRUE(shoulder.isApprox(fixture.shoulder, 2.0e-6));
    EXPECT_TRUE(elbow.isApprox(fixture.elbow, 2.0e-6));
    EXPECT_TRUE(wrist.isApprox(fixture.wrist, 2.0e-6));
    EXPECT_NEAR(base.x(), 0.0, 1.0e-9);
    EXPECT_NEAR(base.y(), 0.0, 1.0e-9);
    EXPECT_NEAR(base.z(), 0.75, 1.0e-9);

    // The approved labels are an independent oracle. This scalar is evaluated
    // directly from the fixture FK points, without calling the production helper.
    const double base_side = independentElbowFaceSide(
      base, fixture.elbow, fixture.wrist, fixture.face);
    const double shoulder_side = independentElbowFaceSide(
      fixture.shoulder, fixture.elbow, fixture.wrist, fixture.face);
    const bool base_classifies_good = base_side < 0.0;
    const bool shoulder_classifies_good = shoulder_side < 0.0;
    EXPECT_EQ(base_classifies_good, fixture.good)
      << "base-side C=" << base_side << " (world base z=" << base.z() << ")";
    base_matches += base_classifies_good == fixture.good ? 1 : 0;
    shoulder_matches += shoulder_classifies_good == fixture.good ? 1 : 0;

    Eigen::VectorXd q;
    robot_state.copyJointGroupPositions(group_, q);
    const auto * joint3_model = model_->getJointModel("joint3");
    const auto * joint3 = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint3_model);
    ASSERT_NE(joint3, nullptr);
    const auto * joint3_parent = joint3_model->getParentLinkModel();
    const auto * joint3_child = joint3_model->getChildLinkModel();
    ASSERT_NE(joint3_parent, nullptr);
    ASSERT_NE(joint3_child, nullptr);
    const Eigen::Vector3d axis = robot_state.getGlobalLinkTransform(joint3_parent).linear() *
      joint3_child->getJointOriginTransform().linear() * joint3->getAxis();
    const Eigen::Vector3d upper_arm = elbow - shoulder;
    const Eigen::Vector3d forearm = wrist - elbow;
    const double bend_angle = std::atan2(
      axis.normalized().dot(upper_arm.cross(forearm)), upper_arm.dot(forearm));
    const double signed_bend = calibration.preferred_sign * std::sin(bend_angle);
    EXPECT_EQ(signed_bend >= calibration.minimum_comfortable_sin, fixture.good)
      << "actual FK bend sin=" << signed_bend << " threshold="
      << calibration.minimum_comfortable_sin;
    PosturePreferenceConfig config;
    config.joint3_angle_offset_rad = calibration.angle_offset_rad;
    config.preferred_bend_sign = calibration.preferred_sign;
    config.comfortable_joint3_lower = calibration.joint3_lower;
    config.comfortable_joint3_upper = calibration.joint3_upper;
    config.position_margin_j5_rad = 0.0;
    config.position_margin_j6_rad = 0.0;
    const auto production = makePosturePreferenceObjective(
      base, elbow, wrist, fixture.face, base_jacobian, elbow_jacobian,
      wrist_jacobian, q, config);
    ASSERT_TRUE(production.valid);
    const bool bend_classifies_good = q[2] >= calibration.joint3_lower &&
      q[2] <= calibration.joint3_upper;
    EXPECT_EQ(bend_classifies_good, fixture.good)
      << "calibrated J3 interval [" << calibration.joint3_lower << ", "
      << calibration.joint3_upper << "] q3=" << q[2];
    bend_matches += bend_classifies_good == fixture.good ? 1 : 0;
    EXPECT_EQ(production.branch_comfortable, fixture.good)
      << "Production branch classification disagrees with approved label; independent "
      << "C(link_base)=" << base_side << " C(link2)=" << shoulder_side;
    if (!fixture.good) {
      ASSERT_GT(production.matrix.rows(), 0);
      EXPECT_GT(production.reference[0], 0.0)
        << "Outside the comfort branch, the soft objective must progress toward it";
    }
    // The branch label is anatomical and must survive polar J1 rotation and
    // an arbitrary change of F. Rotate the known FK geometry about link_base,
    // then try face points on both sides and near the gaze-axis pole.
    for (const double polar_delta : {-1.1, 0.7}) {
      const Eigen::Matrix3d polar = Eigen::AngleAxisd(polar_delta,
        Eigen::Vector3d::UnitZ()).toRotationMatrix();
      Eigen::VectorXd polar_q = q;
      polar_q[0] += polar_delta;
      const Eigen::Vector3d polar_elbow = base + polar * (elbow - base);
      const Eigen::Vector3d polar_wrist = base + polar * (wrist - base);
      const Eigen::Vector3d polar_face = base + polar * (fixture.face - base);
      const std::array<Eigen::Vector3d, 3> face_variants{
        polar_face,
        base + polar * Eigen::Vector3d(0.35, 0.22, 0.08),
        base + polar * Eigen::Vector3d(0.35, -0.22, -0.08)};
      for (std::size_t variant = 0; variant < face_variants.size(); ++variant) {
        const auto stable = makePosturePreferenceObjective(
          base, polar_elbow, polar_wrist, face_variants[variant], base_jacobian,
          elbow_jacobian, wrist_jacobian, polar_q, config);
        ASSERT_TRUE(stable.valid);
        EXPECT_EQ(stable.branch_comfortable, fixture.good)
          << "label changed under polar delta=" << polar_delta << " F variant=" << variant;
      }
    }
  }

  EXPECT_EQ(base_matches, 10)
    << "All approved poses classify only when link_base's world transform is used";
  EXPECT_EQ(shoulder_matches, 6)
    << "The link2-origin counterexample proves the shoulder substitution changes approved labels";
  EXPECT_EQ(bend_matches, 10)
    << "Named-pose calibrated anatomical bend must reproduce all ten approved labels";
}

TEST_F(MotionReferenceTest, Posture1CBadFixturesHaveSafeLocalTrackingCorrection)
{
  const auto fixtures = readPosture1CFixtures();
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  CollisionConstraintBuilder collision_builder(collision_config_);
  const double side_reserve = parameters_.maximum_linear_reference_mps *
    parameters_.control_period_sec;
  int checked_bad_fixtures = 0;
  int corrected_bad_fixtures = 0;
  std::ostringstream analysis;

  for (const auto & fixture : fixtures) {
    if (fixture.good) {
      continue;
    }
    SCOPED_TRACE("POSTURE-1C BAD fixture " + std::to_string(fixture.id));
    ++checked_bad_fixtures;
    moveit::core::RobotState robot_state(model_);
    robot_state.setToDefaultValues();
    for (Eigen::Index joint = 0; joint < fixture.joints.size(); ++joint) {
      robot_state.setVariablePosition("joint" + std::to_string(joint + 1), fixture.joints[joint]);
    }
    robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    robot_state.update();
    ASSERT_TRUE(robot_state.satisfiesBounds(group_)) << "fixture q violates URDF limits";
    Eigen::VectorXd q;
    robot_state.copyJointGroupPositions(group_, q);
    const JointMotionState motion{
      q, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
    const auto actual = collision_builder.validateActualState(*scene, robot_state, *group_);
    analysis << "fixture=" << fixture.id << " actual_valid=" << actual.input_valid
             << " actual_unsafe=" << actual.unsafe << " actual_collision="
             << actual.collision_detected << " actual_min_distance_m="
             << actual.minimum_distance_m << " closest_pair=" << actual.closest_pair
             << " actual_reason=" << actual.failure_reason;
    if (!actual.input_valid || actual.unsafe) {
      ADD_FAILURE() << "Unsafe POSTURE-1C BAD seed " << fixture.id
                    << "; refusing to assess its correction. " << actual.failure_reason;
      analysis << " correction=skipped\n";
      continue;
    }
    const auto robust = collision_builder.validatePose(*scene, robot_state, *group_, motion);
    analysis << " robust_valid=" << robust.input_valid << " robust_unsafe=" << robust.unsafe
             << " robust_clearance_violated=" << robust.robust_clearance_violated
             << " robust_min_distance_m=" << robust.minimum_distance_m
             << " robust_pair=" << robust.closest_pair << " robust_reason="
             << robust.failure_reason;
    if (!robust.input_valid || robust.unsafe) {
      ADD_FAILURE() << "POSTURE-1C BAD seed " << fixture.id
                    << " fails robust stationary clearance; refusing correction claim. "
                    << robust.failure_reason;
      analysis << " correction=skipped\n";
      continue;
    }

    msg::TrackingTarget face = faceAlongLookDirection(robot_state);
    face.face.x = fixture.face.x();
    face.face.y = fixture.face.y();
    face.face.z = fixture.face.z();
    const auto result = update(robot_state, face);
    ASSERT_TRUE(result.tracking) << reference_->diagnostics().state;
    ASSERT_TRUE(result.task);
    ASSERT_FALSE(result.tracking->face_distance_outside_band);
    ASSERT_TRUE(result.tracking->face_distance_interval);
    ASSERT_GT(result.task->secondary_matrix.rows(), 0);

    auto full_task = *result.task;
    const auto collision = collision_builder.build(
      *scene, robot_state, *group_, motion, limits_, parameters_.control_period_sec);
    const QpResult qp = solver.solve(full_task, motion, limits_, collision.constraints);
    ASSERT_TRUE(qp.command_available()) << "QP status=" << static_cast<int>(qp.status);
    ASSERT_EQ(qp.joint_velocity.size(), 6);

    const Eigen::Vector3d base = robot_state.getGlobalLinkTransform("link_base").translation();
    const double c_before = independentElbowFaceSide(
      base, fixture.elbow, fixture.wrist, fixture.face);
    const Eigen::Vector3d w = fixture.wrist - base;
    const Eigen::Vector3d f = fixture.face - base;
    const double f_perp_norm =
      (f - w * (f.dot(w) / w.squaredNorm())).norm();
    ASSERT_GT(f_perp_norm, 1.0e-6);
    const double clearance_before = -c_before / f_perp_norm;
    ASSERT_GT(c_before, 0.0) << "BAD fixture must start on the target-side of link_base SW";
    ASSERT_LT(clearance_before, -side_reserve);

    auto next_state = robot_state;
    const Eigen::VectorXd next_q = q + parameters_.control_period_sec * qp.joint_velocity;
    next_state.setJointGroupPositions(group_, next_q);
    next_state.update(true);
    const Eigen::Vector3d next_base =
      next_state.getGlobalLinkTransform("link_base").translation();
    const Eigen::Vector3d next_elbow =
      next_state.getGlobalLinkTransform("link3").translation();
    const Eigen::Vector3d next_wrist =
      next_state.getGlobalLinkTransform("link4").translation();
    const double c_after = independentElbowFaceSide(
      next_base, next_elbow, next_wrist, fixture.face);
    const Eigen::Vector3d next_w = next_wrist - next_base;
    const Eigen::Vector3d next_f = fixture.face - next_base;
    const double clearance_after = -(c_after) /
      (next_f - next_w * (next_f.dot(next_w) / next_w.squaredNorm())).norm();
    analysis << " clearance_m=" << clearance_before << " -> " << clearance_after
             << " qp_status=" << static_cast<int>(qp.status)
             << " qdot=" << qp.joint_velocity.transpose();
    EXPECT_GT(clearance_after, clearance_before)
      << "fixture=" << fixture.id << " c=" << c_before << " -> " << c_after
      << " qdot=" << qp.joint_velocity.transpose();

    const double feasibility_tolerance =
      solver.config().solution_feasibility_tolerance +
      4.0 * solver.config().solver_absolute_tolerance;
    const auto & distance = *result.tracking->face_distance_interval;
    const double distance_rate = distance.coefficients.dot(qp.joint_velocity);
    EXPECT_GE(distance_rate, distance.lower_bound - feasibility_tolerance);
    EXPECT_LE(distance_rate, distance.upper_bound + feasibility_tolerance);
    ASSERT_EQ(qp.achieved_primary_velocity.size(), result.task->primary_matrix.rows());
    EXPECT_LE((result.task->primary_matrix * qp.joint_velocity -
      qp.achieved_primary_velocity).norm(),
      solver.config().primary_preservation_tolerance + feasibility_tolerance);
    const auto next_actual = collision_builder.validateActualState(*scene, next_state, *group_);
    EXPECT_TRUE(next_actual.input_valid) << next_actual.failure_reason;
    EXPECT_FALSE(next_actual.unsafe) << next_actual.failure_reason;
    const JointMotionState next_motion{
      next_q, qp.joint_velocity,
      Eigen::VectorXd::Zero(qp.joint_velocity.size())};
    const auto next_robust = collision_builder.validatePose(
      *scene, next_state, *group_, next_motion);
    EXPECT_TRUE(next_robust.input_valid) << next_robust.failure_reason;
    EXPECT_FALSE(next_robust.unsafe) << next_robust.failure_reason;
    analysis << " next_actual_unsafe=" << next_actual.unsafe
             << " next_robust_unsafe=" << next_robust.unsafe
             << " next_robust_reason=" << next_robust.failure_reason << '\n';
    ++corrected_bad_fixtures;
  }
  EXPECT_EQ(checked_bad_fixtures, 5);
  RecordProperty("posture1c_bad_seed_model_analysis", analysis.str());
  EXPECT_EQ(corrected_bad_fixtures, 5)
    << "Every BAD fixture must have a safe start before its local QP correction is assessed";
}

TEST_F(MotionReferenceTest, Posture1DFixtureGoodSeedsHoldAndBadSeedsMakeSafeBendProgress)
{
  constexpr double kComfortLower = -1.95514;
  constexpr double kComfortUpper = -0.3696309078954824;
  constexpr double kOneDegreeRad = 3.14159265358979323846 / 180.0;
  constexpr int kGoodHoldSteps = 500;
  constexpr int kBadMaximumSteps = 3000;
  constexpr int kRecoveredHoldSteps = 500;
  const double dt = parameters_.control_period_sec;
  const auto fixtures = readPosture1CFixtures();
  ASSERT_EQ(fixtures.size(), 10U);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = dt;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  std::ostringstream report;
  double time_sec = 300.0;
  int checked_good = 0;
  int checked_bad = 0;

  for (const auto & fixture : fixtures) {
    if ((fixture.good && (fixture.id < 2 || fixture.id > 5)) ||
      (!fixture.good && (fixture.id < 6 || fixture.id > 9)))
    {
      continue;
    }
    SCOPED_TRACE("POSTURE-1D fixture " + std::to_string(fixture.id));
    reference_.reset();
    reference_ = std::make_unique<MotionReference>(
      node_, monitor_, parameters_, limits_, collision_config_);
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    for (Eigen::Index joint = 0; joint < fixture.joints.size(); ++joint) {
      state.setVariablePosition("joint" + std::to_string(joint + 1), fixture.joints[joint]);
    }
    state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    state.update();
    ASSERT_TRUE(state.satisfiesBounds(group_));

    auto face = fixture.good ? faceAlongLookDirection(state) : target(msg::TrackingTarget::FACE);
    if (!fixture.good) {
      face.face.x = fixture.face.x();
      face.face.y = fixture.face.y();
      face.face.z = fixture.face.z();
    } else {
      const Eigen::Vector3d screen =
        state.getGlobalLinkTransform("monitor_control_frame").translation();
      const Eigen::Vector3d nominal_face(face.face.x, face.face.y, face.face.z);
      const Eigen::Vector3d ray = (nominal_face - screen).normalized();
      const Eigen::Vector3d fixed_face = screen + 0.40 * ray;
      face.face.x = fixed_face.x();
      face.face.y = fixed_face.y();
      face.face.z = fixed_face.z();
    }
    face.target_id = static_cast<std::uint32_t>(100 + fixture.id);
    Eigen::VectorXd initial_q;
    state.copyJointGroupPositions(group_, initial_q);
    const int maximum_steps = fixture.good ? kGoodHoldSteps : kBadMaximumSteps;
    int comfortable_streak = 0;
    int first_comfortable_step = -1;
    double maximum_comfort_excursion = 0.0;
    double maximum_q3 = initial_q[2];
    int completed_steps = 0;

    for (int step = 0; step < maximum_steps; ++step) {
      time_sec += dt;
      const auto stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
      face.header.stamp = stamp;
      face.face_stamp = stamp;
      const auto result = update(
        state, face, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
      ASSERT_TRUE(result.tracking) << "step=" << step;
      ASSERT_TRUE(result.task) << "step=" << step;
      if (step == 0) {
        EXPECT_EQ(reference_->diagnostics().posture_mode, fixture.good ? "preserve" : "recover");
        if (fixture.good) {
          EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
          EXPECT_FALSE(result.compact_roll_constraints.empty());
        }
      }

      Eigen::VectorXd q, qd, qdd;
      state.copyJointGroupPositions(group_, q);
      state.copyJointGroupVelocities(group_, qd);
      state.copyJointGroupAccelerations(group_, qdd);
      const JointMotionState motion{q, qd, qdd};
      const auto collision = collision_builder.build(
        *scene, state, *group_, motion, limits_, dt);
      const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
      ASSERT_TRUE(qp.command_available()) << "step=" << step
                                          << " status=" << static_cast<int>(qp.status);
      ASSERT_EQ(qp.joint_velocity.size(), 6);
      const Eigen::VectorXd next_q = q + dt * qp.joint_velocity;
      const Eigen::VectorXd next_qdd = (qp.joint_velocity - qd) / dt;
      state.setJointGroupPositions(group_, next_q);
      state.setJointGroupVelocities(group_, qp.joint_velocity);
      state.setJointGroupAccelerations(group_, next_qdd);
      state.update(true);
      ASSERT_TRUE(state.satisfiesBounds(group_)) << "step=" << step;
      EXPECT_FALSE(scene->isStateColliding(state, group_->getName())) << "step=" << step;

      Eigen::VectorXd current_q;
      state.copyJointGroupPositions(group_, current_q);
      maximum_q3 = std::max(maximum_q3, current_q[2]);
      const double excursion = std::max({
            0.0, kComfortLower - current_q[2], current_q[2] - kComfortUpper});
      maximum_comfort_excursion = std::max(maximum_comfort_excursion, excursion);
      if (current_q[2] >= kComfortLower && current_q[2] <= kComfortUpper) {
        ++comfortable_streak;
        if (first_comfortable_step < 0) {
          first_comfortable_step = step;
        }
      } else {
        comfortable_streak = 0;
        first_comfortable_step = -1;
      }
      completed_steps = step + 1;
      if (!fixture.good && comfortable_streak >= kRecoveredHoldSteps) {
        break;
      }
    }

    if (fixture.good) {
      ++checked_good;
      EXPECT_LE(maximum_comfort_excursion, kOneDegreeRad)
        << "GOOD fixture must remain in the calibrated branch within the practical 1-degree "
        << "allowance; raw excursion=" << maximum_comfort_excursion;
      Eigen::VectorXd final_q;
      state.copyJointGroupPositions(group_, final_q);
      EXPECT_GE(final_q[2], kComfortLower - kOneDegreeRad);
      EXPECT_LE(final_q[2], kComfortUpper + kOneDegreeRad);
      if (fixture.id == 2) {
        auto lost_target = target(msg::TrackingTarget::HOLD);
        lost_target.target_id = face.target_id;
        time_sec += dt;
        auto stamp = rclcpp::Time(
          static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
        lost_target.header.stamp = stamp;
        const auto lost = update(
          state, lost_target, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
        EXPECT_FALSE(lost.task);
        Eigen::VectorXd before_loss_q, before_loss_qd, before_loss_qdd;
        state.copyJointGroupPositions(group_, before_loss_q);
        state.copyJointGroupVelocities(group_, before_loss_qd);
        state.copyJointGroupAccelerations(group_, before_loss_qdd);

        face.target_id += 1000U;
        time_sec += dt;
        stamp = rclcpp::Time(
          static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
        face.header.stamp = stamp;
        face.face_stamp = stamp;
        const auto returned = update(
          state, face, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
        ASSERT_TRUE(returned.tracking);
        ASSERT_TRUE(returned.task);
        EXPECT_EQ(reference_->diagnostics().posture_mode, "preserve");
        EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
        Eigen::VectorXd after_return_q, after_return_qd, after_return_qdd;
        state.copyJointGroupPositions(group_, after_return_q);
        state.copyJointGroupVelocities(group_, after_return_qd);
        state.copyJointGroupAccelerations(group_, after_return_qdd);
        EXPECT_TRUE(after_return_q.isApprox(before_loss_q, 1.0e-12));
        EXPECT_TRUE(after_return_qd.isApprox(before_loss_qd, 1.0e-12));
        EXPECT_TRUE(after_return_qdd.isApprox(before_loss_qdd, 1.0e-12));
        EXPECT_GE(after_return_q[2], kComfortLower - kOneDegreeRad);
        EXPECT_LE(after_return_q[2], kComfortUpper + kOneDegreeRad);
      }
      report << "GOOD" << fixture.id << " steps=" << completed_steps
             << " q3=" << final_q[2] << " max_excursion_rad="
             << maximum_comfort_excursion << '\n';
    } else {
      ++checked_bad;
      EXPECT_GT(maximum_q3, initial_q[2] + 0.005)
        << "BAD fixture must make measurable inward bend progress without sacrificing safe "
        << "commands; initial_q3=" << initial_q[2] << " max_q3=" << maximum_q3;
      const bool recovered = comfortable_streak >= kRecoveredHoldSteps;
      EXPECT_TRUE(recovered)
        << "Recorded BAD fixture must recover within the 30-second model budget and remain "
        << "comfortable for five seconds; fixture=" << fixture.id
        << " steps=" << completed_steps << " q3=" << maximum_q3;
      report << "BAD" << fixture.id << " steps=" << completed_steps
             << " max_q3=" << maximum_q3 << " recovered=" << recovered;
      if (recovered) {
        report << " recovery_sec=" << first_comfortable_step * dt
               << " held_sec=" << kRecoveredHoldSteps * dt;
      }
      report << '\n';
    }
  }

  EXPECT_EQ(checked_good, 4);
  EXPECT_EQ(checked_bad, 4);
  RecordProperty("posture1d_fixture_seed_rollouts", report.str());
  std::cout << "POSTURE-1D fixture seed rollouts\n" << report.str();
}

TEST_F(MotionReferenceTest, Posture1CQuietFaceHoldClampsResidualNullspaceMotion)
{
  const auto fixtures = readPosture1CFixtures();
  const auto good = std::find_if(fixtures.begin(), fixtures.end(),
      [](const Posture1CFixture & fixture) {return fixture.id == 1 && fixture.good;});
  ASSERT_NE(good, fixtures.end());

  parameters_.position_deadband_m = 0.03;
  reference_.reset();
  reference_ = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  moveit::core::RobotState robot_state(model_);
  robot_state.setToDefaultValues();
  for (Eigen::Index joint = 0; joint < good->joints.size(); ++joint) {
    robot_state.setVariablePosition("joint" + std::to_string(joint + 1), good->joints[joint]);
  }
  robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  robot_state.update();
  ASSERT_TRUE(robot_state.satisfiesBounds(group_));
  const Eigen::Vector3d screen =
    robot_state.getGlobalLinkTransform("monitor_control_frame").translation();
  auto target = faceAlongLookDirection(robot_state);
  Eigen::Vector3d face(target.face.x, target.face.y, target.face.z);
  const Eigen::Vector3d ray = (face - screen).normalized();

  const auto set_face_range = [&target](const Eigen::Vector3d & origin,
    const Eigen::Vector3d & direction, const double range) {
      const Eigen::Vector3d value = origin + range * direction;
      target.face.x = value.x();
      target.face.y = value.y();
      target.face.z = value.z();
    };
  const auto stamp_target = [&target](const double time_sec) {
      const auto stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
      target.header.stamp = stamp;
      target.face_stamp = stamp;
    };

  // A stationary compatible GOOD pose is a zero-motion control for the same
  // calibrated branch and actual FACE geometry used by the residual cases.
  constexpr double kFixedFaceDistance = 0.40;
  set_face_range(screen, ray, kFixedFaceDistance);
  stamp_target(1.0);
  auto stationary = update(robot_state, target, nullptr, nullptr, std::nullopt,
      kSceneRevision, 1.0);
  ASSERT_TRUE(stationary.tracking);
  ASSERT_TRUE(stationary.task);
  ASSERT_TRUE(stationary.tracking->face_distance_interval);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd still_q;
  robot_state.copyJointGroupPositions(group_, still_q);
  const JointMotionState still_motion{still_q, zero, zero};
  const auto still_collision = collision_builder.build(
    *scene, robot_state, *group_, still_motion, limits_, parameters_.control_period_sec);
  const QpResult still_qp = solver.solve(
    *stationary.task, still_motion, limits_, still_collision.constraints);
  ASSERT_TRUE(still_qp.command_available());
  EXPECT_LE(still_qp.joint_velocity.norm(), 1.0e-3)
    << "A zero-velocity, aligned GOOD pose should remain still";

  const std::array<double, 2> residual_magnitudes{0.01, 0.03};
  const int settling_steps = static_cast<int>(std::ceil(
      5.0 / parameters_.control_period_sec));
  const int observation_steps = static_cast<int>(std::ceil(
      30.0 / parameters_.control_period_sec));
  const double distance_quiet_limit = 0.01;
  const TrackingVelocityTaskConfig tracking_config;
  const double angular_allowance = parameters_.maximum_angular_reference_radps *
    parameters_.control_period_sec;
  std::ostringstream residual_report;
  const Eigen::Vector3d fixed_face = screen + kFixedFaceDistance * ray;
  const double fixture_distance = (fixed_face - screen).norm();
  ASSERT_GE(fixture_distance, parameters_.face_distance_min_m);
  ASSERT_LE(fixture_distance, parameters_.face_distance_max_m);

  int scenario = 0;
  for (const double magnitude : residual_magnitudes) {
    for (const double sign : {-1.0, 1.0}) {
      reference_.reset();
      reference_ = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      moveit::core::RobotState seed(model_);
      seed.setToDefaultValues();
      for (Eigen::Index joint = 0; joint < good->joints.size(); ++joint) {
        seed.setVariablePosition("joint" + std::to_string(joint + 1), good->joints[joint]);
      }
      seed.setJointGroupVelocities(group_, zero);
      seed.setJointGroupAccelerations(group_, zero);
      seed.update();
      auto fixed_target = faceAlongLookDirection(seed);
      fixed_target.face.x = fixed_face.x();
      fixed_target.face.y = fixed_face.y();
      fixed_target.face.z = fixed_face.z();
      const double scenario_start_time = 2.0 + 40.0 * static_cast<double>(scenario++);
      auto stamp_fixed_target = [&fixed_target](const double time_sec) {
          const auto stamp = rclcpp::Time(
            static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
          fixed_target.header.stamp = stamp;
          fixed_target.face_stamp = stamp;
        };
      stamp_fixed_target(scenario_start_time);
      auto initial = update(seed, fixed_target, nullptr, nullptr, std::nullopt,
          kSceneRevision, scenario_start_time);
      ASSERT_TRUE(initial.tracking);
      ASSERT_TRUE(initial.task);
      ASSERT_TRUE(initial.tracking->face_distance_interval);
      ASSERT_FALSE(initial.tracking->face_distance_outside_band);

      Eigen::FullPivLU<Eigen::MatrixXd> primary_decomposition(
        initial.task->primary_matrix);
      const Eigen::MatrixXd nullspace = primary_decomposition.kernel();
      ASSERT_GT(nullspace.cols(), 0);
      Eigen::VectorXd preferred = Eigen::VectorXd::Zero(6);
      preferred[1] = 1.0;
      preferred[2] = 1.0;
      Eigen::VectorXd residual_direction = nullspace * (nullspace.transpose() * preferred);
      ASSERT_GT(residual_direction.norm(), 1.0e-8);
      residual_direction.normalize();
      Eigen::VectorXd residual_velocity = sign * magnitude * residual_direction;
      ASSERT_LE(residual_velocity.cwiseAbs().maxCoeff(), limits_.max_velocity.maxCoeff());
      seed.setJointGroupVelocities(group_, residual_velocity);
      seed.update();

      double minimum_distance = fixture_distance;
      double maximum_distance = fixture_distance;
      double maximum_band_excursion = 0.0;
      double maximum_settled_band_excursion = 0.0;
      double maximum_pointing_error = 0.0;
      double maximum_roll_error = 0.0;
      int primary_only_after_settle = 0;
      double settled_distance_min = std::numeric_limits<double>::infinity();
      double settled_distance_max = 0.0;
      double maximum_screen_path_10s = 0.0;
      double maximum_elbow_path_10s = 0.0;
      double maximum_wrist_path_10s = 0.0;
      double maximum_distance_span_10s = 0.0;
      struct HoldSample
      {
        double distance;
        Eigen::Vector3d screen;
        Eigen::Vector3d elbow;
        Eigen::Vector3d wrist;
      };
      std::vector<HoldSample> hold_samples;
      const int total_steps = settling_steps + observation_steps;
      for (int step = 0; step < total_steps; ++step) {
        Eigen::VectorXd current_q;
        Eigen::VectorXd current_velocity;
        Eigen::VectorXd current_acceleration;
        seed.copyJointGroupPositions(group_, current_q);
        seed.copyJointGroupVelocities(group_, current_velocity);
        seed.copyJointGroupAccelerations(group_, current_acceleration);
        const JointMotionState motion{current_q, current_velocity, current_acceleration};
        const double time_sec = scenario_start_time +
          static_cast<double>(step + 1) * parameters_.control_period_sec;
        stamp_fixed_target(time_sec);
        const auto result = update(seed, fixed_target, nullptr, nullptr, std::nullopt,
            kSceneRevision, time_sec);
        ASSERT_TRUE(result.tracking) << "scenario=" << scenario << " step=" << step;
        ASSERT_TRUE(result.task);
        ASSERT_TRUE(result.tracking->face_distance_interval);
        maximum_pointing_error = std::max(
          maximum_pointing_error, std::abs(result.tracking->pointing_error_rad));
        maximum_roll_error = std::max(
          maximum_roll_error, std::abs(result.tracking->roll_error_rad));
        const auto collision = collision_builder.build(
          *scene, seed, *group_, motion, limits_, parameters_.control_period_sec);
        const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
        ASSERT_TRUE(qp.command_available()) << "scenario=" << scenario << " step=" << step
                                            << " QP=" << static_cast<int>(qp.status);
        if (step >= settling_steps && qp.status == QpStatus::kPrimaryOnly) {
          ++primary_only_after_settle;
        }
        const Eigen::VectorXd next_q = current_q +
          parameters_.control_period_sec * qp.joint_velocity;
        const Eigen::VectorXd next_acceleration =
          (qp.joint_velocity - current_velocity) / parameters_.control_period_sec;
        seed.setJointGroupPositions(group_, next_q);
        seed.setJointGroupVelocities(group_, qp.joint_velocity);
        seed.setJointGroupAccelerations(group_, next_acceleration);
        seed.update(true);
        ASSERT_TRUE(seed.satisfiesBounds(group_)) << "scenario=" << scenario << " step=" << step;
        const auto actual = collision_builder.validateActualState(*scene, seed, *group_);
        ASSERT_TRUE(actual.input_valid) << actual.failure_reason;
        ASSERT_FALSE(actual.unsafe) << actual.failure_reason;
        const JointMotionState next_motion{next_q, qp.joint_velocity, next_acceleration};
        const auto robust = collision_builder.validatePose(*scene, seed, *group_, next_motion);
        ASSERT_TRUE(robust.input_valid) << robust.failure_reason;
        ASSERT_FALSE(robust.unsafe) << robust.failure_reason;

        const Eigen::Vector3d next_screen =
          seed.getGlobalLinkTransform("monitor_control_frame").translation();
        const double next_distance = (next_screen - fixed_face).norm();
        minimum_distance = std::min(minimum_distance, next_distance);
        maximum_distance = std::max(maximum_distance, next_distance);
        const double band_excursion = std::max({
              0.0, parameters_.face_distance_min_m - next_distance,
              next_distance - parameters_.face_distance_max_m});
        maximum_band_excursion = std::max(maximum_band_excursion, band_excursion);
        if (step >= settling_steps) {
          const double settled_excursion = std::max({
                0.0,
                parameters_.face_distance_min_m -
                parameters_.maximum_linear_reference_mps * parameters_.control_period_sec -
                next_distance,
                next_distance - parameters_.face_distance_max_m -
                parameters_.maximum_linear_reference_mps * parameters_.control_period_sec});
          maximum_settled_band_excursion = std::max(
            maximum_settled_band_excursion, settled_excursion);
        }
        if (step >= settling_steps && step % 5 == 0) {
          const Eigen::Vector3d elbow = seed.getGlobalLinkTransform("link3").translation();
          const Eigen::Vector3d wrist = seed.getGlobalLinkTransform("link4").translation();
          hold_samples.push_back({next_distance, next_screen, elbow, wrist});
          settled_distance_min = std::min(settled_distance_min, next_distance);
          settled_distance_max = std::max(settled_distance_max, next_distance);
          const std::size_t samples_per_10s = static_cast<std::size_t>(std::ceil(
              10.0 / (5.0 * parameters_.control_period_sec))) + 1U;
          if (hold_samples.size() >= samples_per_10s) {
            const std::size_t begin = hold_samples.size() - samples_per_10s;
            double screen_path = 0.0;
            double elbow_path = 0.0;
            double wrist_path = 0.0;
            double window_distance_min = std::numeric_limits<double>::infinity();
            double window_distance_max = 0.0;
            for (std::size_t i = begin; i < hold_samples.size(); ++i) {
              window_distance_min = std::min(window_distance_min, hold_samples[i].distance);
              window_distance_max = std::max(window_distance_max, hold_samples[i].distance);
              if (i > begin) {
                screen_path += (hold_samples[i].screen - hold_samples[i - 1].screen).norm();
                elbow_path += (hold_samples[i].elbow - hold_samples[i - 1].elbow).norm();
                wrist_path += (hold_samples[i].wrist - hold_samples[i - 1].wrist).norm();
              }
            }
            maximum_screen_path_10s = std::max(maximum_screen_path_10s, screen_path);
            maximum_elbow_path_10s = std::max(maximum_elbow_path_10s, elbow_path);
            maximum_wrist_path_10s = std::max(maximum_wrist_path_10s, wrist_path);
            maximum_distance_span_10s = std::max(
              maximum_distance_span_10s, window_distance_max - window_distance_min);
          }
        }
      }
      residual_report << "magnitude=" << magnitude << " sign=" << sign
                      << " residual_qdot=" << residual_velocity.transpose()
                      << " distance_range=" << minimum_distance << ".." << maximum_distance
                      << " settled_span=" << settled_distance_max - settled_distance_min
                      << " max_transient_band_excursion=" << maximum_band_excursion
                      << " settled_band_excursion=" << maximum_settled_band_excursion
                      << " quiet_path10s(screen/E/W)=" << maximum_screen_path_10s << '/'
                      << maximum_elbow_path_10s << '/' << maximum_wrist_path_10s
                      << " dist_span10s=" << maximum_distance_span_10s
                      << " max_gaze_roll=" << maximum_pointing_error << '/'
                      << maximum_roll_error << " primary_only_after_settle="
                      << primary_only_after_settle << '\n';
      EXPECT_LE(maximum_band_excursion, parameters_.position_deadband_m)
        << "Transient face-distance excursion exceeds the configured deadband; "
        << residual_report.str();
      EXPECT_LE(maximum_pointing_error,
        parameters_.pointing_deadband_rad + angular_allowance) << residual_report.str();
      EXPECT_LE(maximum_roll_error,
        tracking_config.roll_deadband_rad + angular_allowance) << residual_report.str();
      ASSERT_GE(hold_samples.size(), 2U);
      EXPECT_GE(hold_samples.size() * 5.0 * parameters_.control_period_sec, 29.0)
        << "Need a full 30 s post-settle hold; " << residual_report.str();
      EXPECT_LE(settled_distance_max - settled_distance_min, distance_quiet_limit)
        << residual_report.str();
      EXPECT_LE(maximum_screen_path_10s, distance_quiet_limit) << residual_report.str();
      EXPECT_LE(maximum_elbow_path_10s, distance_quiet_limit) << residual_report.str();
      EXPECT_LE(maximum_wrist_path_10s, distance_quiet_limit) << residual_report.str();
      EXPECT_LE(maximum_distance_span_10s, distance_quiet_limit) << residual_report.str();
      EXPECT_LE(maximum_settled_band_excursion, 0.0) << residual_report.str();
      EXPECT_EQ(primary_only_after_settle, 0) << residual_report.str();
    }
  }
  RecordProperty("posture1c_residual_nullspace_hold", residual_report.str());
}

TEST_F(MotionReferenceTest, Posture1DGoodRestKeepsComfortWhileTrackingFarFixedFace)
{
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  const auto named_bend = [this](const std::string & name) {
      moveit::core::RobotState named_state(model_);
      if (!named_state.setToDefaultValues(group_, name)) {
        return std::optional<std::pair<double, double>>{};
      }
      named_state.update();
      Eigen::VectorXd q;
      named_state.copyJointGroupPositions(group_, q);
      const auto * joint = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint);
      if (!joint || !revolute) {
        return std::optional<std::pair<double, double>>{};
      }
      const auto * parent = joint->getParentLinkModel();
      const auto * child = joint->getChildLinkModel();
      if (!parent || !child) {
        return std::optional<std::pair<double, double>>{};
      }
      const Eigen::Vector3d shoulder =
        named_state.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow =
        named_state.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist =
        named_state.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = named_state.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      const Eigen::Vector3d upper = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      const double angle = std::atan2(
        axis.normalized().dot(upper.cross(forearm)), upper.dot(forearm));
      return std::optional<std::pair<double, double>>{{q[2], angle}};
    };
  const auto rest = named_bend("rest");
  const auto search = named_bend("search");
  ASSERT_TRUE(rest);
  ASSERT_TRUE(search);
  // This oracle uses the measured xArm6 rest/search joint values and signed
  // bend angles directly. The interval is the calibrated model interval
  // recorded in ROADMAP, independent of the production calibration helper.
  constexpr double kComfortableJoint3Lower = -1.95514;
  constexpr double kComfortableJoint3Upper = -0.3696309078954824;
  const double minimum_comfortable_bend_sin =
    std::min(std::sin(rest->second), std::sin(search->second));

  moveit::core::RobotState state(model_);
  ASSERT_TRUE(state.setToDefaultValues(group_, "rest"));
  state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  state.update();
  ASSERT_TRUE(state.satisfiesBounds(group_));
  msg::TrackingTarget fixed_target = target(msg::TrackingTarget::FACE);
  fixed_target.face.x = 3.052973146;
  fixed_target.face.y = 1.907709351;
  fixed_target.face.z = 1.75;

  double maximum_comfort_excursion_rad = 0.0;
  double final_pointing_error = 0.0;
  const double dt = parameters_.control_period_sec;
  constexpr int kRolloutSteps = 1000;
  for (int step = 0; step < kRolloutSteps; ++step) {
    const double time_sec = 1.0 + static_cast<double>(step) * dt;
    const auto stamp = rclcpp::Time(
      static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
    fixed_target.header.stamp = stamp;
    fixed_target.face_stamp = stamp;
    const auto result = update(
      state, fixed_target, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
    ASSERT_TRUE(result.tracking) << "step=" << step;
    ASSERT_TRUE(result.task) << "step=" << step;
    final_pointing_error = result.tracking->pointing_error_rad;

    Eigen::VectorXd q;
    Eigen::VectorXd qd;
    Eigen::VectorXd qdd;
    state.copyJointGroupPositions(group_, q);
    state.copyJointGroupVelocities(group_, qd);
    state.copyJointGroupAccelerations(group_, qdd);
    const JointMotionState motion{q, qd, qdd};
    const auto collision = collision_builder.build(
      *scene, state, *group_, motion, limits_, dt);
    const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
    ASSERT_TRUE(qp.command_available()) << "step=" << step
                                        << " qp=" << static_cast<int>(qp.status);
    const Eigen::VectorXd next_q = q + dt * qp.joint_velocity;
    const Eigen::VectorXd next_qdd = (qp.joint_velocity - qd) / dt;
    state.setJointGroupPositions(group_, next_q);
    state.setJointGroupVelocities(group_, qp.joint_velocity);
    state.setJointGroupAccelerations(group_, next_qdd);
    state.update(true);
    ASSERT_TRUE(state.satisfiesBounds(group_)) << "step=" << step;
    moveit::core::RobotState next_state(model_);
    next_state.setJointGroupPositions(group_, next_q);
    next_state.update();
    const auto next_bend_joint = model_->getJointModel("joint3");
    const auto * next_revolute =
      dynamic_cast<const moveit::core::RevoluteJointModel *>(next_bend_joint);
    ASSERT_NE(next_revolute, nullptr);
    const auto * next_parent = next_bend_joint->getParentLinkModel();
    const auto * next_child = next_bend_joint->getChildLinkModel();
    ASSERT_NE(next_parent, nullptr);
    ASSERT_NE(next_child, nullptr);
    const Eigen::Vector3d next_shoulder =
      next_state.getGlobalLinkTransform("link2").translation();
    const Eigen::Vector3d next_elbow =
      next_state.getGlobalLinkTransform("link3").translation();
    const Eigen::Vector3d next_wrist =
      next_state.getGlobalLinkTransform("link4").translation();
    const Eigen::Vector3d next_axis = next_state.getGlobalLinkTransform(next_parent).linear() *
      next_child->getJointOriginTransform().linear() * next_revolute->getAxis();
    const double next_bend_angle = std::atan2(
      next_axis.normalized().dot((next_elbow - next_shoulder).cross(next_wrist - next_elbow)),
      (next_elbow - next_shoulder).dot(next_wrist - next_elbow));
    const double joint3_excursion = std::max({
          0.0, kComfortableJoint3Lower - next_q[2], next_q[2] - kComfortableJoint3Upper});
    const double bend_sine_excursion = std::max(
      0.0, minimum_comfortable_bend_sin - std::sin(next_bend_angle));
    // The QP solves rates; its documented feasibility/absolute residual is
    // integrated over one controller step. Use that shared command-space
    // allowance for the independent geometric oracle too.
    const double command_tolerance = std::max(1.0e-5, dt * (
          qp_config.solution_feasibility_tolerance + 4.0 * qp_config.solver_absolute_tolerance));
    maximum_comfort_excursion_rad = std::max(
      maximum_comfort_excursion_rad,
      std::max(joint3_excursion, bend_sine_excursion));
    EXPECT_LE(joint3_excursion, command_tolerance) << "step=" << step;
    EXPECT_LE(bend_sine_excursion, command_tolerance) << "step=" << step;
  }

  EXPECT_LE(final_pointing_error,
    parameters_.pointing_deadband_rad +
    parameters_.maximum_angular_reference_radps * dt)
    << "The gaze should settle after the initial 31-degree target change";
  EXPECT_LE(maximum_comfort_excursion_rad, std::max(1.0e-5, dt * (
      qp_config.solution_feasibility_tolerance + 4.0 * qp_config.solver_absolute_tolerance)))
    << "The GOOD rest posture must remain comfortable within one integrated solver residual; "
    << "model-derived q3 interval=" << kComfortableJoint3Lower << ".."
    << kComfortableJoint3Upper << " minimum signed bend sine="
    << minimum_comfortable_bend_sin;
}

TEST_F(MotionReferenceTest, Posture1DGoodComfortSurvivesDistanceAndFaceTransitions)
{
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  const double kComfortLower = MotionReferenceTestPeer::comfortableJoint3Lower(*reference_);
  const double kComfortUpper = MotionReferenceTestPeer::comfortableJoint3Upper(*reference_);
  moveit::core::RobotState state(model_);
  ASSERT_TRUE(state.setToDefaultValues(group_, "rest"));
  state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  state.update();
  ASSERT_TRUE(state.satisfiesBounds(group_));

  const auto signed_bend_sine = [&](const moveit::core::RobotState & sample) {
      const auto * joint = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint);
      const auto * parent = joint->getParentLinkModel();
      const auto * child = joint->getChildLinkModel();
      const Eigen::Vector3d shoulder = sample.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow = sample.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist = sample.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = sample.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      return std::sin(std::atan2(
        axis.normalized().dot((elbow - shoulder).cross(wrist - elbow)),
                 (elbow - shoulder).dot(wrist - elbow)));
    };
  const auto fixed_face = [this](const Eigen::Vector3d & point) {
      auto value = target(msg::TrackingTarget::FACE);
      value.face.x = point.x();
      value.face.y = point.y();
      value.face.z = point.z();
      return value;
    };
  const double dt = parameters_.control_period_sec;
  double time_sec = 20.0;
  double maximum_q3_excursion = 0.0;
  double minimum_signed_bend = std::numeric_limits<double>::infinity();
  double maximum_q3_speed = 0.0;
  double maximum_excursion_time = -1.0;
  double q3_at_maximum_excursion = 0.0;
  std::string maximum_excursion_phase;
  int preserve_mode_steps = 0;
  int recover_mode_steps = 0;
  std::string first_excursion_phase;
  double first_excursion_time = -1.0;
  double qdot_at_first_excursion = 0.0;
  double first_recover_time = -1.0;
  double last_recover_time = -1.0;
  double maximum_speed_time = -1.0;
  std::string maximum_speed_phase;
  std::string first_excursion_qp_context;
  std::string first_recover_qp_context;
  std::string maximum_excursion_qp_context;
  std::ostringstream phase_report;
  const auto run_phase = [&](const std::string & label, const msg::TrackingTarget & fixed_target,
    const int steps) {
      double final_pointing_error = std::numeric_limits<double>::infinity();
      for (int step = 0; step < steps; ++step) {
        time_sec += dt;
        auto stamped_target = fixed_target;
        const auto stamp = rclcpp::Time(
          static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
        stamped_target.header.stamp = stamp;
        stamped_target.face_stamp = stamp;
        const auto result = update(
          state, stamped_target, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
        if (!result.tracking || !result.task) {
          ADD_FAILURE() << label << " missing tracking task at step=" << step;
          return std::numeric_limits<double>::quiet_NaN();
        }
        const Eigen::Index distance_rows = result.task->distance_matrix.rows();
        if (distance_rows != 0 && distance_rows != 3) {
          ADD_FAILURE() << label << " invalid distance rows=" << distance_rows;
          return std::numeric_limits<double>::quiet_NaN();
        }
        const bool recovering = reference_->diagnostics().posture_mode == "recover";
        if (!recovering) {
          ++preserve_mode_steps;
        } else {
          ++recover_mode_steps;
          last_recover_time = time_sec;
        }
        Eigen::VectorXd q, qd, qdd;
        state.copyJointGroupPositions(group_, q);
        state.copyJointGroupVelocities(group_, qd);
        state.copyJointGroupAccelerations(group_, qdd);
        const JointMotionState motion{q, qd, qdd};
        const auto collision = collision_builder.build(
          *scene, state, *group_, motion, limits_, dt);
        const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
        if (!qp.command_available()) {
          ADD_FAILURE() << label << " step=" << step
                        << " qp=" << static_cast<int>(qp.status);
          return std::numeric_limits<double>::quiet_NaN();
        }
        const Eigen::VectorXd next_q = q + dt * qp.joint_velocity;
        if (recovering && first_recover_time < 0.0) {
          first_recover_time = time_sec;
          const Eigen::VectorXd primary_error = result.task->primary_matrix *
            qp.joint_velocity - result.task->primary_reference;
          const Eigen::VectorXd preserved_primary_error = result.task->primary_matrix *
            qp.joint_velocity - qp.achieved_primary_velocity;
          std::ostringstream context;
          context << label << " t=" << time_sec << " q3_before=" << q[2]
                  << " q3_after=" << next_q[2] << " qdot3_before=" << qd[2]
                  << " qddot3_before=" << qdd[2] << " qdot3_command="
                  << qp.joint_velocity[2]
                  << " qp=" << static_cast<int>(qp.status) << " ranks="
                  << static_cast<int>(qp.primary_status) << '/'
                  << static_cast<int>(qp.secondary_status) << '/'
                  << static_cast<int>(qp.tertiary_status) << '/'
                  << static_cast<int>(qp.distance_status) << " primary_rows="
                  << result.task->primary_matrix.rows() << " primary_q3_norm="
                  << result.task->primary_matrix.col(2).norm() << " primary_ref_norm="
                  << result.task->primary_reference.norm() << " primary_error_norm="
                  << primary_error.norm() << " preserved_primary_error_norm="
                  << preserved_primary_error.norm() << " secondary_rows="
                  << result.task->secondary_matrix.rows() << " secondary_ref="
                  << result.task->secondary_reference.transpose() << " secondary_q3="
                  << (result.task->secondary_matrix.rows() > 0 ?
          result.task->secondary_matrix(0, 2) : 0.0) << " constraints="
                  << result.task->secondary_constraints.size();
          first_recover_qp_context = context.str();
        }
        const Eigen::VectorXd next_qdd = (qp.joint_velocity - qd) / dt;
        state.setJointGroupPositions(group_, next_q);
        state.setJointGroupVelocities(group_, qp.joint_velocity);
        state.setJointGroupAccelerations(group_, next_qdd);
        state.update(true);
        if (!state.satisfiesBounds(group_)) {
          ADD_FAILURE() << label << " joint bounds at step=" << step;
          return std::numeric_limits<double>::quiet_NaN();
        }
        const double q3_excursion = std::max({
            0.0, kComfortLower - next_q[2], next_q[2] - kComfortUpper});
        const double bend = signed_bend_sine(state);
        if (std::abs(qp.joint_velocity[2]) > maximum_q3_speed) {
          maximum_q3_speed = std::abs(qp.joint_velocity[2]);
          maximum_speed_time = time_sec;
          maximum_speed_phase = label;
        }
        if (q3_excursion > 0.0 && first_excursion_time < 0.0) {
          first_excursion_phase = label;
          first_excursion_time = time_sec;
          qdot_at_first_excursion = qp.joint_velocity[2];
          std::ostringstream context;
          context << label << " t=" << time_sec << " q3_before=" << q[2]
                  << " q3_after=" << next_q[2] << " qdot3_before=" << qd[2]
                  << " qddot3_before=" << qdd[2] << " qdot3_command="
                  << qp.joint_velocity[2]
                  << " posture_mode=" << reference_->diagnostics().posture_mode
                  << " qp=" << static_cast<int>(qp.status) << " ranks="
                  << static_cast<int>(qp.primary_status) << '/'
                  << static_cast<int>(qp.secondary_status) << '/'
                  << static_cast<int>(qp.tertiary_status) << '/'
                  << static_cast<int>(qp.distance_status) << " task_rows="
                  << result.task->primary_matrix.rows() << '/'
                  << result.task->secondary_matrix.rows() << '/'
                  << result.task->tertiary_matrix.rows() << '/'
                  << result.task->distance_matrix.rows() << " constraints="
                  << result.task->secondary_constraints.size();
          for (const auto & constraint : result.task->secondary_constraints) {
            if (constraint.coefficients.size() > 2 &&
              std::abs(constraint.coefficients[2]) > 1.0e-9)
            {
              context << " q3_constraint=" << constraint.coefficients[2] << ':'
                      << constraint.lower_bound << ".." << constraint.upper_bound;
            }
          }
          first_excursion_qp_context = context.str();
        }
        if (q3_excursion > maximum_q3_excursion) {
          maximum_q3_excursion = q3_excursion;
          maximum_excursion_time = time_sec;
          q3_at_maximum_excursion = next_q[2];
          maximum_excursion_phase = label;
          std::ostringstream context;
          context << label << " t=" << time_sec << " q3_before=" << q[2]
                  << " q3_after=" << next_q[2] << " qdot3=" << qp.joint_velocity[2]
                  << " posture_mode=" << reference_->diagnostics().posture_mode
                  << " qp=" << static_cast<int>(qp.status) << " ranks="
                  << static_cast<int>(qp.primary_status) << '/'
                  << static_cast<int>(qp.secondary_status) << '/'
                  << static_cast<int>(qp.tertiary_status) << '/'
                  << static_cast<int>(qp.distance_status) << " task_rows="
                  << result.task->primary_matrix.rows() << '/'
                  << result.task->secondary_matrix.rows() << '/'
                  << result.task->tertiary_matrix.rows() << '/'
                  << result.task->distance_matrix.rows() << " constraints="
                  << result.task->secondary_constraints.size();
          for (const auto & constraint : result.task->secondary_constraints) {
            if (constraint.coefficients.size() > 2 &&
              std::abs(constraint.coefficients[2]) > 1.0e-9)
            {
              context << " q3_constraint=" << constraint.coefficients[2] << ':'
                      << constraint.lower_bound << ".." << constraint.upper_bound;
            }
          }
          maximum_excursion_qp_context = context.str();
        }
        minimum_signed_bend = std::min(minimum_signed_bend, bend);
        final_pointing_error = std::abs(result.tracking->pointing_error_rad);
      }
      EXPECT_LE(final_pointing_error,
        parameters_.pointing_deadband_rad + parameters_.maximum_angular_reference_radps * dt)
        << label << " gaze did not settle";
      const Eigen::Vector3d screen =
        state.getGlobalLinkTransform("monitor_control_frame").translation();
      const Eigen::Vector3d face(
        fixed_target.face.x, fixed_target.face.y, fixed_target.face.z);
      const double distance = (screen - face).norm();
      Eigen::VectorXd final_q;
      state.copyJointGroupPositions(group_, final_q);
      phase_report << label << " d=" << distance << " q3=" << final_q[2]
                   << " bend_sin=" << signed_bend_sine(state)
                   << " pointing_final=" << final_pointing_error << '\n';
      return distance;
    };

  const Eigen::Isometry3d start_screen =
    state.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d start_face = start_screen.translation() +
    0.40 * start_screen.linear().col(0);
  const double initial_distance = run_phase("reachable", fixed_face(start_face), 100);
  EXPECT_GE(initial_distance, parameters_.face_distance_min_m - 0.01);
  EXPECT_LE(initial_distance, parameters_.face_distance_max_m + 0.01);

  const double far_distance = run_phase("far", fixed_face(
      Eigen::Vector3d(3.052973146, 1.907709351, 1.75)), 1000);
  EXPECT_GT(far_distance, parameters_.face_distance_max_m);
  const double side_up_distance = run_phase("far-side-up", fixed_face(
      Eigen::Vector3d(3.052973146, 2.007709351, 1.80)), 500);
  EXPECT_GT(side_up_distance, parameters_.face_distance_max_m);

  const Eigen::Isometry3d return_screen =
    state.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d return_face = return_screen.translation() +
    0.40 * return_screen.linear().col(0);
  const double returned_distance = run_phase("return-reachable", fixed_face(return_face), 1000);
  EXPECT_GE(returned_distance, parameters_.face_distance_min_m - 0.01);
  EXPECT_LE(returned_distance, parameters_.face_distance_max_m + 0.01);

  const Eigen::Isometry3d upper_screen =
    state.getGlobalLinkTransform("monitor_control_frame");
  const Eigen::Vector3d upper_face = upper_screen.translation() +
    parameters_.face_distance_max_m * upper_screen.linear().col(0);
  const double upper_distance = run_phase("upper-boundary", fixed_face(upper_face), 300);
  EXPECT_NEAR(upper_distance, parameters_.face_distance_max_m, 0.01);
  Eigen::VectorXd final_q;
  state.copyJointGroupPositions(group_, final_q);
  EXPECT_LE(maximum_q3_excursion, 3.14159265358979323846 / 180.0)
    << "Small gaze-driven transient must not lose the GOOD branch; " << phase_report.str();
  EXPECT_GE(final_q[2], kComfortLower);
  EXPECT_LE(final_q[2], kComfortUpper);
  EXPECT_GE(signed_bend_sine(state), 0.701883982289872 - 0.003)
    << phase_report.str();
  EXPECT_GT(preserve_mode_steps, 0);
  EXPECT_GT(maximum_q3_speed, 0.0);
  RecordProperty("posture1d_good_transition_phases", phase_report.str());
  std::cout << "POSTURE-1D GOOD transitions max_q3_excursion=" << maximum_q3_excursion
            << " excursion_phase=" << maximum_excursion_phase << " excursion_time=" <<
    maximum_excursion_time << " q3_at_peak=" << q3_at_maximum_excursion
            << " first_phase=" << first_excursion_phase << " first_time=" <<
    first_excursion_time << " qdot_at_first=" << qdot_at_first_excursion
            << " first_recover=" << first_recover_time << " last_recover=" <<
    last_recover_time << " max_abs_q3_speed=" << maximum_q3_speed << " max_speed_phase=" <<
    maximum_speed_phase << " max_speed_time=" << maximum_speed_time << " preserve_steps=" <<
    preserve_mode_steps << " recover_steps=" << recover_mode_steps
            << " minimum_bend_sin=" << minimum_signed_bend << " first_excursion_qp={"
            << first_excursion_qp_context << "} first_recover={" << first_recover_qp_context
            << "} maximum_excursion_qp={" << maximum_excursion_qp_context << "}\n"
            << phase_report.str();
}

TEST_F(MotionReferenceCameraTest, GoodUpperComfortBoundaryKeepsExactOneTickGuard)
{
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  moveit::core::RobotState state(model_);
  ASSERT_TRUE(state.setToDefaultValues(group_, "rest"));
  state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  state.update();
  ASSERT_TRUE(state.satisfiesBounds(group_));
  Eigen::VectorXd q, qd, qdd;
  state.copyJointGroupPositions(group_, q);
  state.copyJointGroupVelocities(group_, qd);
  state.copyJointGroupAccelerations(group_, qdd);
  const double comfortable_upper = MotionReferenceTestPeer::comfortableJoint3Upper(*reference_);
  ASSERT_NEAR(q[2], comfortable_upper, 1.0e-9);

  const Eigen::Isometry3d screen = state.getGlobalLinkTransform("monitor_control_frame");
  auto face = target(msg::TrackingTarget::FACE);
  const Eigen::Vector3d face_position = screen.translation() + 0.40 * screen.linear().col(0);
  face.face.x = face_position.x();
  face.face.y = face_position.y();
  face.face.z = face_position.z();
  const double time_sec = 20.01;
  const auto stamp = rclcpp::Time(
    static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
  face.header.stamp = stamp;
  face.face_stamp = stamp;
  const auto reference = update(
    state, face, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
  ASSERT_TRUE(reference.tracking);
  ASSERT_TRUE(reference.task);
  EXPECT_EQ(reference_->diagnostics().posture_mode, "preserve");
  ASSERT_EQ(reference.task->preferred_primary_constraints.size(), 1U);

  const JointMotionState motion{q, qd, qdd};
  const auto collision = collision_builder.build(
    *scene, state, *group_, motion, limits_, parameters_.control_period_sec);
  const QpResult qp = solver.solve(*reference.task, motion, limits_, collision.constraints);
  ASSERT_TRUE(qp.command_available()) << "status=" << static_cast<int>(qp.status);
  EXPECT_FALSE(qp.preferred_primary_fallback_used);
  const Eigen::VectorXd next_q = q + parameters_.control_period_sec * qp.joint_velocity;
  EXPECT_LE(next_q[2], comfortable_upper)
    << "Camera gaze primary must respect the exact GOOD comfort guard for this tick; "
    << "q3=" << q[2] << " qdot3=" << qp.joint_velocity[2]
    << " pointing_error=" << reference.tracking->pointing_error_rad
    << " qp=" << static_cast<int>(qp.status) << " ranks="
    << static_cast<int>(qp.primary_status) << '/'
    << static_cast<int>(qp.secondary_status);
}

TEST_F(MotionReferenceCameraTest, Posture1DFastTailCapturedJitterAndMotionControls)
{
  // The fixture preserves camera producer stamps and FACE/HOLD availability. HOLD
  // rows reuse the last valid target; qdd is an adjacent-sample approximation.
  struct Sample
  {
    double sim_sec;
    double source_sec;
    Eigen::Vector3d face;
    Eigen::VectorXd q;
    Eigen::VectorXd qd;
    Eigen::VectorXd qdd;
    bool used;
  };
  std::ifstream capture(TEST_POSTURE1D_FAST_TAIL_CAPTURE_FILE);
  ASSERT_TRUE(capture.good());
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(capture, line)));
  std::vector<Sample> samples;
  while (std::getline(capture, line)) {
    std::stringstream row(line);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(row, field, ',')) {
      fields.push_back(field);
    }
    ASSERT_EQ(fields.size(), 27U) << line;
    Sample sample;
    sample.sim_sec = std::stod(fields[0]);
    sample.source_sec = std::stod(fields[2]);
    sample.used = std::stoi(fields[4]) != 0;
    sample.face = Eigen::Vector3d(
      std::stod(fields[6]), std::stod(fields[7]), std::stod(fields[8]));
    sample.q.resize(6);
    sample.qd.resize(6);
    sample.qdd.resize(6);
    for (Eigen::Index joint = 0; joint < 6; ++joint) {
      sample.q[joint] = std::stod(fields[9 + joint]);
      sample.qd[joint] = std::stod(fields[15 + joint]);
      sample.qdd[joint] = std::stod(fields[21 + joint]);
    }
    samples.push_back(std::move(sample));
  }
  ASSERT_EQ(samples.size(), 351U);
  constexpr double dt = 0.01;
  constexpr int replay_ticks = 710;
  // The piecewise fast/slow target stops at 0.55 s; this ends 30.0 s later.
  constexpr int overspeed_ticks = 3056;
  constexpr double first_sim_sec = 99.0;
  const Eigen::Vector3d fixed_face(1.8902364177247293, -0.0011634283727742813,
    1.7674253491322234);
  const Sample & seed = samples.front();

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = dt;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  struct Metrics
  {
    double pointing_error_max{0.0};
    double pointing_error_final{0.0};
    double q3_min{std::numeric_limits<double>::infinity()};
    double q3_max{-std::numeric_limits<double>::infinity()};
    double q4_min{std::numeric_limits<double>::infinity()};
    double q4_max{-std::numeric_limits<double>::infinity()};
    double q6_min{std::numeric_limits<double>::infinity()};
    double q6_max{-std::numeric_limits<double>::infinity()};
    std::array<double, 3> path_m{{0.0, 0.0, 0.0}};
    std::array<double, 3> late_path_m{{0.0, 0.0, 0.0}};
    double late_q4_min{std::numeric_limits<double>::infinity()};
    double late_q4_max{-std::numeric_limits<double>::infinity()};
    double late_q6_min{std::numeric_limits<double>::infinity()};
    double late_q6_max{-std::numeric_limits<double>::infinity()};
    int late_q4_reversals{0};
    int late_q6_reversals{0};
    int distance_rows{0};
    int qp_failures{0};
    double target_speed_max_mps{0.0};
    double primary_feedforward_max_radps{0.0};
    double slow_feedforward_max_radps{0.0};
  };
  const auto run = [&](const int mode) {
      // Recorded target jitter, fixed-target stop, and 20/150 mm/s motion controls.
      // HOLD rows reuse the last producer-stamped target; this model has no queue/gap replay.
      reference_ = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      moveit::core::RobotState state(model_);
      state.setToDefaultValues();
      state.setJointGroupPositions(group_, seed.q);
      state.setJointGroupVelocities(group_, seed.qd);
      state.setJointGroupAccelerations(group_, seed.qdd);
      state.update();
      EXPECT_TRUE(state.satisfiesBounds(group_));
      Metrics metrics;
      std::size_t sample_index = 0U;
      std::size_t last_used_index = 0U;
      bool have_previous_frames = false;
      int previous_q4_sign = 0;
      int previous_q6_sign = 0;
      bool have_q4_turn = false;
      bool have_q6_turn = false;
      double q4_turn_position = 0.0;
      double q6_turn_position = 0.0;
      std::array<Eigen::Vector3d, 3> previous_frames;
      const auto accumulate_frames = [&](const bool late_window) {
          const std::array<Eigen::Vector3d, 3> frames{{
            state.getGlobalLinkTransform("monitor_control_frame").translation(),
            state.getGlobalLinkTransform("link3").translation(),
            state.getGlobalLinkTransform("link4").translation()}};
          if (have_previous_frames) {
            for (std::size_t i = 0; i < frames.size(); ++i) {
              metrics.path_m[i] += (frames[i] - previous_frames[i]).norm();
              if (late_window) {
                metrics.late_path_m[i] += (frames[i] - previous_frames[i]).norm();
              }
            }
          }
          previous_frames = frames;
          have_previous_frames = true;
        };
      accumulate_frames(false);
      const int ticks = mode == 7 ? overspeed_ticks : replay_ticks;
      for (int step = 0; step < ticks; ++step) {
        const double time_sec = first_sim_sec + static_cast<double>(step) * dt;
        while (sample_index + 1U < samples.size() &&
          samples[sample_index + 1U].sim_sec <= time_sec + 1.0e-9)
        {
          ++sample_index;
        }
        if (samples[sample_index].used) {
          last_used_index = sample_index;
        }
        const Sample & sample = samples[last_used_index];
        auto face = target(msg::TrackingTarget::FACE);
        face.target_id = 8U;
        const double elapsed_sec = time_sec - first_sim_sec;
        double synthetic_speed_mps = mode == 5 ? 0.02 : (mode == 6 ? 0.15 : 0.0);
        Eigen::Vector3d point = mode == 2 ? fixed_face : sample.face;
        if (mode == 7) {
          double lateral_offset_m = 0.0;
          if (elapsed_sec < 0.25) {
            synthetic_speed_mps = 1.20;
            lateral_offset_m = synthetic_speed_mps * elapsed_sec;
          } else if (elapsed_sec < 0.55) {
            synthetic_speed_mps = 0.15;
            lateral_offset_m = 1.20 * 0.25 + synthetic_speed_mps * (elapsed_sec - 0.25);
          } else {
            lateral_offset_m = 1.20 * 0.25 + 0.15 * 0.30;
          }
          point = fixed_face + Eigen::Vector3d(0.0, lateral_offset_m, 0.0);
        } else if (synthetic_speed_mps > 0.0) {
          point = fixed_face + Eigen::Vector3d(0.0, synthetic_speed_mps * elapsed_sec, 0.0);
        }
        face.face.x = point.x();
        face.face.y = point.y();
        face.face.z = point.z();
        face.header.stamp = rclcpp::Time(
          static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
        const double source_sec = synthetic_speed_mps > 0.0 ? time_sec : sample.source_sec;
        face.face_stamp = rclcpp::Time(
          static_cast<std::int64_t>(std::llround(source_sec * 1.0e9)), RCL_ROS_TIME);
        auto reference = update(
          state, face, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
        if (!reference.task || !reference.tracking) {
          ADD_FAILURE() << "Missing FACE task at step=" << step;
          break;
        }
        metrics.pointing_error_max = std::max(metrics.pointing_error_max,
          std::abs(reference.tracking->pointing_error_rad));
        metrics.pointing_error_final = std::abs(reference.tracking->pointing_error_rad);
        metrics.target_speed_max_mps = std::max(
          metrics.target_speed_max_mps, synthetic_speed_mps);
        metrics.primary_feedforward_max_radps = std::max(
          metrics.primary_feedforward_max_radps,
          reference.tracking->face_primary_feedforward_radps.norm());
        if (mode == 7 && elapsed_sec >= 0.35 && elapsed_sec < 0.55) {
          metrics.slow_feedforward_max_radps = std::max(
            metrics.slow_feedforward_max_radps,
            reference.tracking->face_primary_feedforward_radps.norm());
        }
        HierarchicalVelocityTask task = *reference.task;
        Eigen::VectorXd q, qd, qdd;
        state.copyJointGroupPositions(group_, q);
        state.copyJointGroupVelocities(group_, qd);
        state.copyJointGroupAccelerations(group_, qdd);
        const JointMotionState motion{q, qd, qdd};
        const auto collision = collision_builder.build(
          *scene, state, *group_, motion, limits_, dt);
        const QpResult qp = solver.solve(task, motion, limits_, collision.constraints);
        if (!qp.command_available()) {
          ++metrics.qp_failures;
          ADD_FAILURE() << "No safe command at step=" << step
                        << " status=" << static_cast<int>(qp.status);
          break;
        }
        metrics.distance_rows += task.distance_matrix.rows() > 0 ? 1 : 0;
        const Eigen::VectorXd next_q = q + dt * qp.joint_velocity;
        const Eigen::VectorXd next_qdd = (qp.joint_velocity - qd) / dt;
        state.setJointGroupPositions(group_, next_q);
        state.setJointGroupVelocities(group_, qp.joint_velocity);
        state.setJointGroupAccelerations(group_, next_qdd);
        state.update(true);
        EXPECT_TRUE(state.satisfiesBounds(group_)) << "step=" << step;
        metrics.q3_min = std::min(metrics.q3_min, next_q[2]);
        metrics.q3_max = std::max(metrics.q3_max, next_q[2]);
        metrics.q4_min = std::min(metrics.q4_min, next_q[3]);
        metrics.q4_max = std::max(metrics.q4_max, next_q[3]);
        metrics.q6_min = std::min(metrics.q6_min, next_q[5]);
        metrics.q6_max = std::max(metrics.q6_max, next_q[5]);
        const bool late_window = step >= 410;
        if (late_window) {
          metrics.late_q4_min = std::min(metrics.late_q4_min, next_q[3]);
          metrics.late_q4_max = std::max(metrics.late_q4_max, next_q[3]);
          metrics.late_q6_min = std::min(metrics.late_q6_min, next_q[5]);
          metrics.late_q6_max = std::max(metrics.late_q6_max, next_q[5]);
          const auto count_reversal = [](const double velocity, const double position,
            int & previous_sign, bool & have_turn, double & turn_position, int & count) {
              if (std::abs(velocity) < 0.01) {
                return;
              }
              const int sign = velocity > 0.0 ? 1 : -1;
              if (previous_sign != 0 && sign != previous_sign) {
                if (have_turn && std::abs(position - turn_position) > 0.0174533) {
                  ++count;
                }
                turn_position = position;
                have_turn = true;
              }
              previous_sign = sign;
            };
          count_reversal(qp.joint_velocity[3], next_q[3], previous_q4_sign, have_q4_turn,
            q4_turn_position, metrics.late_q4_reversals);
          count_reversal(qp.joint_velocity[5], next_q[5], previous_q6_sign, have_q6_turn,
            q6_turn_position, metrics.late_q6_reversals);
        }
        accumulate_frames(late_window);
      }
      return metrics;
    };

  const std::array<Metrics, 5> metrics{{run(0), run(2), run(5), run(6), run(7)}};
  std::ostringstream report;
  const std::array<const char *, 5> names{{
    "recorded-target", "fixed-target-stop", "moving-20mmps", "moving-150mmps",
    "overspeed-slow-stop"}};
  for (std::size_t i = 0; i < metrics.size(); ++i) {
    report << names[i] << " P/E/W_m=" << metrics[i].path_m[0] << '/'
           << metrics[i].path_m[1] << '/' << metrics[i].path_m[2]
           << " q3=" << metrics[i].q3_min << ".." << metrics[i].q3_max
           << " q4=" << metrics[i].q4_min << ".." << metrics[i].q4_max
           << " q6=" << metrics[i].q6_min << ".." << metrics[i].q6_max
           << " pointing_max_rad=" << metrics[i].pointing_error_max
           << " pointing_final_rad=" << metrics[i].pointing_error_final
           << " late_P/E/W_m=" << metrics[i].late_path_m[0] << '/'
           << metrics[i].late_path_m[1] << '/' << metrics[i].late_path_m[2]
           << " late_q4=" << metrics[i].late_q4_min << ".." << metrics[i].late_q4_max
           << " late_q6=" << metrics[i].late_q6_min << ".." << metrics[i].late_q6_max
           << " late_reversals_q4/q6=" << metrics[i].late_q4_reversals << '/'
           << metrics[i].late_q6_reversals
           << " target_speed_max_mps=" << metrics[i].target_speed_max_mps
           << " primary_ff_max_radps=" << metrics[i].primary_feedforward_max_radps
           << " slow_ff_max_radps=" << metrics[i].slow_feedforward_max_radps
           << " distance_rows=" << metrics[i].distance_rows
           << " qp_failures=" << metrics[i].qp_failures << '\n';
  }
  // The fixed-target stop control should remain quiet; moving controls should follow.
  EXPECT_LT(metrics[1].path_m[2], 0.05);
  EXPECT_LT(metrics[0].pointing_error_max, 0.0174533);
  EXPECT_LT(metrics[0].late_path_m[2], 0.01);
  EXPECT_LT(metrics[0].late_q4_max - metrics[0].late_q4_min, 0.03);
  EXPECT_LT(metrics[0].late_q6_max - metrics[0].late_q6_min, 0.03);
  EXPECT_LE(metrics[0].late_q4_reversals, 1);
  EXPECT_LE(metrics[0].late_q6_reversals, 1);
  EXPECT_GT(metrics[0].q3_min, -1.95514);
  EXPECT_LT(metrics[2].pointing_error_max, 0.0174533);
  EXPECT_LT(metrics[3].pointing_error_max, 0.0174533);
  EXPECT_GT(metrics[3].path_m[2], 0.15);
  EXPECT_GT(metrics[3].path_m[2], 2.0 * metrics[1].path_m[2]);
  EXPECT_GT(metrics[4].target_speed_max_mps, 0.18);
  EXPECT_GT(metrics[4].primary_feedforward_max_radps, 0.70);
  EXPECT_GT(metrics[4].slow_feedforward_max_radps, 0.05);
  EXPECT_LT(metrics[4].slow_feedforward_max_radps, 0.70);
  EXPECT_LT(metrics[4].pointing_error_final, 0.0174533);
  EXPECT_LE(metrics[4].late_q4_reversals, 2);
  EXPECT_LE(metrics[4].late_q6_reversals, 2);
  for (const Metrics & metric : metrics) {
    EXPECT_EQ(metric.qp_failures, 0);
  }
  RecordProperty("posture1d_fast_tail_no_hold_replay", report.str());
  std::cout << "POSTURE-1D fast-tail 7.1s no-HOLD model approximation\n" << report.str();
}

TEST_F(MotionReferenceTest, Posture1DFarDistanceSoftTaskSettlesAtReachLimit)
{
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  constexpr double kFaceX = 3.052973146;
  constexpr double kFaceY = 1.907709351;
  constexpr double kFaceZ = 1.75;
  const Eigen::Vector3d face(kFaceX, kFaceY, kFaceZ);
  constexpr int kSteps = 6000;
  constexpr int kSampleStride = 5;
  struct Metrics
  {
    double monitor_span{0.0};
    double elbow_span{0.0};
    double wrist_span{0.0};
    double monitor_path{0.0};
    double elbow_path{0.0};
    double wrist_path{0.0};
    double distance_min{std::numeric_limits<double>::infinity()};
    double distance_max{0.0};
    double maximum_pointing_error{0.0};
    double mean_radial_rate{0.0};
    double mean_tangential_speed{0.0};
    double maximum_joint_speed{0.0};
    std::size_t rate_samples{0U};
  };
  const auto run = [&]() {
      reference_.reset();
      reference_ = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      moveit::core::RobotState state(model_);
      if (!state.setToDefaultValues(group_, "rest")) {
        ADD_FAILURE() << "Could not load named rest witness";
        return Metrics{};
      }
      state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
      state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
      state.update();
      const msg::TrackingTarget fixed_target = [&]() {
          auto value = target(msg::TrackingTarget::FACE);
          value.face.x = face.x();
          value.face.y = face.y();
          value.face.z = face.z();
          return value;
        }();
      std::array<Eigen::Vector3d, 3> minimum;
      std::array<Eigen::Vector3d, 3> maximum;
      for (auto & value : minimum) {
        value.setConstant(std::numeric_limits<double>::infinity());
      }
      for (auto & value : maximum) {
        value.setConstant(-std::numeric_limits<double>::infinity());
      }
      std::array<Eigen::Vector3d, 3> previous;
      bool have_previous = false;
      Metrics metrics;
      for (int step = 0; step < kSteps; ++step) {
        const double time_sec = 200.0 + (static_cast<double>(step) + 1.0) *
          parameters_.control_period_sec;
        auto stamped_target = fixed_target;
        const auto stamp = rclcpp::Time(
          static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
        stamped_target.header.stamp = stamp;
        stamped_target.face_stamp = stamp;
        const auto result = update(
          state, stamped_target, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
        if (!result.tracking || !result.task) {
          ADD_FAILURE() << "No tracking task at step=" << step;
          return Metrics{};
        }
        const auto & task = *result.task;
        Eigen::Vector3d tangent1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d tangent2 = Eigen::Vector3d::Zero();
        Eigen::MatrixXd linear_jacobian;
        if (task.distance_matrix.rows() == 3) {
          Eigen::MatrixXd screen_jacobian;
          if (!state.getJacobian(
              group_, model_->getLinkModel("monitor_control_frame"),
              Eigen::Vector3d::Zero(), screen_jacobian, false))
          {
            ADD_FAILURE() << "Could not evaluate screen Jacobian at step=" << step;
            return Metrics{};
          }
          linear_jacobian = screen_jacobian.topRows(3);
          const auto * root_link = group_->getJointModels().front()->getParentLinkModel();
          if (root_link) {
            linear_jacobian = (state.getGlobalLinkTransform(root_link).linear() *
              linear_jacobian).eval();
          }
          const Eigen::Vector3d screen =
            state.getGlobalLinkTransform("monitor_control_frame").translation();
          const Eigen::Vector3d radial = (screen - face).normalized();
          tangent1 = radial.unitOrthogonal().normalized();
          tangent2 = radial.cross(tangent1).normalized();
        }
        Eigen::VectorXd q, qd, qdd;
        state.copyJointGroupPositions(group_, q);
        state.copyJointGroupVelocities(group_, qd);
        state.copyJointGroupAccelerations(group_, qdd);
        const JointMotionState motion{q, qd, qdd};
        const auto collision = collision_builder.build(
          *scene, state, *group_, motion, limits_, parameters_.control_period_sec);
        const QpResult qp = solver.solve(task, motion, limits_, collision.constraints);
        if (!qp.command_available()) {
          ADD_FAILURE() << "No QP command at step=" << step;
          return Metrics{};
        }
        if (step >= 2000 && linear_jacobian.rows() == 3) {
          metrics.mean_radial_rate +=
            result.tracking->face_distance_matrix.row(0).dot(qp.joint_velocity);
          metrics.mean_tangential_speed += std::hypot(
            (tangent1.transpose() * linear_jacobian * qp.joint_velocity)[0],
            (tangent2.transpose() * linear_jacobian * qp.joint_velocity)[0]);
          metrics.maximum_joint_speed = std::max(
            metrics.maximum_joint_speed, qp.joint_velocity.lpNorm<Eigen::Infinity>());
          ++metrics.rate_samples;
        }
        const Eigen::VectorXd next_q = q + parameters_.control_period_sec * qp.joint_velocity;
        const Eigen::VectorXd next_qdd = (qp.joint_velocity - qd) /
          parameters_.control_period_sec;
        state.setJointGroupPositions(group_, next_q);
        state.setJointGroupVelocities(group_, qp.joint_velocity);
        state.setJointGroupAccelerations(group_, next_qdd);
        state.update(true);
        if (!state.satisfiesBounds(group_)) {
          ADD_FAILURE() << "Joint bounds violated at step=" << step;
          return Metrics{};
        }
        if (step < 2000 || step % kSampleStride != 0) {
          continue;
        }
        const std::array<Eigen::Vector3d, 3> points{
          state.getGlobalLinkTransform("monitor_control_frame").translation(),
          state.getGlobalLinkTransform("link3").translation(),
          state.getGlobalLinkTransform("link4").translation()};
        for (std::size_t index = 0; index < points.size(); ++index) {
          minimum[index] = minimum[index].cwiseMin(points[index]);
          maximum[index] = maximum[index].cwiseMax(points[index]);
          if (have_previous) {
            if (index == 0) {metrics.monitor_path += (points[index] - previous[index]).norm();}
            if (index == 1) {metrics.elbow_path += (points[index] - previous[index]).norm();}
            if (index == 2) {metrics.wrist_path += (points[index] - previous[index]).norm();}
          }
          previous[index] = points[index];
        }
        have_previous = true;
        metrics.distance_min = std::min(metrics.distance_min, (face - points[0]).norm());
        metrics.distance_max = std::max(metrics.distance_max, (face - points[0]).norm());
        metrics.maximum_pointing_error = std::max(
          metrics.maximum_pointing_error, std::abs(result.tracking->pointing_error_rad));
      }
      metrics.monitor_span = (maximum[0] - minimum[0]).norm();
      metrics.elbow_span = (maximum[1] - minimum[1]).norm();
      metrics.wrist_span = (maximum[2] - minimum[2]).norm();
      if (metrics.rate_samples > 0U) {
        metrics.mean_radial_rate /= static_cast<double>(metrics.rate_samples);
        metrics.mean_tangential_speed /= static_cast<double>(metrics.rate_samples);
      }
      return metrics;
    };
  const Metrics production = run();
  const auto report = [](const Metrics & value) {
      std::ostringstream stream;
      stream << "span(P/E/W)=" << value.monitor_span << '/' << value.elbow_span << '/' <<
        value.wrist_span << " path(P/E/W)=" << value.monitor_path << '/' <<
        value.elbow_path << '/' << value.wrist_path << " distance=" <<
        value.distance_min << ".." << value.distance_max << " pointing=" <<
        value.maximum_pointing_error << " mean_radial_rate=" << value.mean_radial_rate <<
        " mean_tangent_speed=" << value.mean_tangential_speed << " max_joint_speed=" <<
        value.maximum_joint_speed;
      return stream.str();
    };
  RecordProperty("far_distance_settled", report(production));
  EXPECT_LE(production.monitor_span, 0.01);
  EXPECT_LE(production.elbow_span, 0.01);
  EXPECT_LE(production.wrist_span, 0.01);
  EXPECT_LE(production.monitor_path, 0.04);
  EXPECT_LE(production.elbow_path, 0.04);
  EXPECT_LE(production.wrist_path, 0.04);
  std::cout << "POSTURE-1D far-distance production=" << report(production) << '\n';
}

TEST_F(MotionReferenceTest, Posture1CQualifiedShellHoldSettlesWithoutPrimaryFallback)
{
  parameters_.position_deadband_m = 0.03;
  reference_.reset();
  reference_ = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);

  // Frozen from runs/lower_narrow120/model_red_seed.json at t=32.93 s, the
  // start of the first qualified W-path failure. qdd is an estimate from
  // adjacent JointState velocity samples, not measured controller feedback.
  const Eigen::VectorXd seed_q = (Eigen::VectorXd(6) <<
    0.3431167536524256, -0.6451215104369273, -0.4131614728305029,
    0.8832054107062707, -0.49377890199444197, -0.7513625911319638).finished();
  const Eigen::VectorXd seed_qd = (Eigen::VectorXd(6) <<
    0.008830894583498557, 0.0027550561414891265, -0.007382714269534428,
    0.02533001325449499, -0.003092048824205024, -0.02217623385184244).finished();
  const Eigen::VectorXd seed_qdd_estimate = (Eigen::VectorXd(6) <<
    -0.0010566198674565928, 0.00012360055593418068, 0.00046335123802957796,
    -0.0030232341735942267, -3.5287786790724834e-06, 0.0026979210581949504).finished();
  const Eigen::Vector3d fixed_face(
    0.6720479502409709, -7.21194357188413e-07, 1.1499836707827846);
  constexpr double kCapturedDistanceM = 0.3688014620111739;
  constexpr double kQuietPathLimitM = 0.01;
  constexpr double kSettlingSec = 5.0;
  constexpr double kHoldSec = 30.0;
  constexpr double kSamplePeriodSec = 0.05;

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  const double dt = parameters_.control_period_sec;
  const int settling_steps = static_cast<int>(std::ceil(kSettlingSec / dt));
  const int hold_steps = static_cast<int>(std::ceil(kHoldSec / dt));
  const int total_steps = settling_steps + hold_steps;
  const int sample_stride = static_cast<int>(std::llround(kSamplePeriodSec / dt));
  ASSERT_GT(sample_stride, 0);
  ASSERT_NEAR(sample_stride * dt, kSamplePeriodSec, 1.0e-9);
  const double one_step_margin = parameters_.maximum_linear_reference_mps * dt;
  ASSERT_LE(one_step_margin, 0.5 * (
      parameters_.face_distance_max_m - parameters_.face_distance_min_m));
  const TrackingVelocityTaskConfig tracking_config;

  const auto named_bend = [this](const std::string & name) {
      moveit::core::RobotState named_state(model_);
      if (!named_state.setToDefaultValues(group_, name)) {
        return std::optional<std::pair<double, double>>{};
      }
      named_state.update();
      Eigen::VectorXd named_q;
      named_state.copyJointGroupPositions(group_, named_q);
      const auto * joint = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint);
      if (!joint || !revolute || named_q.size() != 6) {
        return std::optional<std::pair<double, double>>{};
      }
      const auto * parent = joint->getParentLinkModel();
      const auto * child = joint->getChildLinkModel();
      if (!parent || !child) {
        return std::optional<std::pair<double, double>>{};
      }
      const Eigen::Vector3d shoulder =
        named_state.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow =
        named_state.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist =
        named_state.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = named_state.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      const Eigen::Vector3d upper = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      const double angle = std::atan2(
        axis.normalized().dot(upper.cross(forearm)), upper.dot(forearm));
      return std::optional<std::pair<double, double>>{{named_q[2], angle}};
    };
  const auto rest = named_bend("rest");
  const auto search = named_bend("search");
  ASSERT_TRUE(rest);
  ASSERT_TRUE(search);
  const auto bend = calibratePostureBend(
    rest->first, rest->second, search->first, search->second,
    limits_.lower_position[2], limits_.upper_position[2]);
  ASSERT_TRUE(bend.valid);
  ASSERT_GE(seed_q[2], bend.joint3_lower);
  ASSERT_LE(seed_q[2], bend.joint3_upper);

  struct HoldSample
  {
    Eigen::Vector3d screen;
    Eigen::Vector3d elbow;
    Eigen::Vector3d wrist;
  };
  struct CaseMetrics
  {
    double screen_range{0.0};
    double elbow_range{0.0};
    double wrist_range{0.0};
    double screen_path_10s{0.0};
    double elbow_path_10s{0.0};
    double wrist_path_10s{0.0};
    double distance_min{std::numeric_limits<double>::infinity()};
    double distance_max{0.0};
    double max_pointing_error{0.0};
    double max_roll_error{0.0};
    double q3_min{std::numeric_limits<double>::infinity()};
    double q3_max{-std::numeric_limits<double>::infinity()};
    double final_q3{0.0};
    double final_distance{0.0};
    int primary_only_after_settle{0};
  };

  const auto run_hold = [&](const std::string & label, const Eigen::VectorXd & initial_qd,
    const Eigen::VectorXd & initial_qdd, const double start_time,
    CaseMetrics & metrics) {
      reference_.reset();
      reference_ = std::make_unique<MotionReference>(
        node_, monitor_, parameters_, limits_, collision_config_);
      moveit::core::RobotState seed(model_);
      seed.setToDefaultValues();
      for (Eigen::Index joint = 0; joint < seed_q.size(); ++joint) {
        seed.setVariablePosition("joint" + std::to_string(joint + 1), seed_q[joint]);
      }
      seed.setJointGroupVelocities(group_, initial_qd);
      seed.setJointGroupAccelerations(group_, initial_qdd);
      seed.update();
      ASSERT_TRUE(seed.satisfiesBounds(group_)) << label;
      auto fixed_target = faceAlongLookDirection(seed);
      fixed_target.face.x = fixed_face.x();
      fixed_target.face.y = fixed_face.y();
      fixed_target.face.z = fixed_face.z();
      const auto stamp_target = [&fixed_target](const double time_sec) {
          const auto stamp = rclcpp::Time(
            static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
          fixed_target.header.stamp = stamp;
          fixed_target.face_stamp = stamp;
        };
      stamp_target(start_time);
      const auto initial = update(seed, fixed_target, nullptr, nullptr, std::nullopt,
          kSceneRevision, start_time);
      ASSERT_TRUE(initial.tracking) << label;
      ASSERT_TRUE(initial.task) << label;
      ASSERT_TRUE(initial.tracking->face_distance_interval) << label;
      ASSERT_TRUE(initial.tracking->face_distance_outside_band) << label;
      const Eigen::Vector3d initial_screen =
        seed.getGlobalLinkTransform("monitor_control_frame").translation();
      const double initial_distance = (initial_screen - fixed_face).norm();
      ASSERT_NEAR(initial_distance, kCapturedDistanceM, 1.0e-5) << label;
      ASSERT_EQ(initial.task->secondary_matrix.rows(), 0) << label;
      ASSERT_EQ(initial.task->distance_matrix.rows(), 3) << label
                                                         <<
        "Out-of-band distance is soft and lower priority than posture";

      std::vector<HoldSample> samples;
      const double angular_allowance =
        parameters_.maximum_angular_reference_radps * dt;
      metrics.distance_min = initial_distance;
      metrics.distance_max = initial_distance;
      double minimum_distance_all = initial_distance;
      double maximum_distance_all = initial_distance;
      std::ostringstream case_report;
      case_report << label << " start_q=" << seed_q.transpose()
                  << " start_qd=" << initial_qd.transpose()
                  << " start_qdd=" << initial_qdd.transpose()
                  << " fixed_F=" << fixed_face.transpose()
                  << " initial_d=" << initial_distance
                  << " q3_comfort=" << bend.joint3_lower << ".."
                  << bend.joint3_upper << " secondary_rows="
                  << initial.task->secondary_matrix.rows();

      for (int step = 0; step < total_steps; ++step) {
        Eigen::VectorXd current_q;
        Eigen::VectorXd current_qd;
        Eigen::VectorXd current_qdd;
        seed.copyJointGroupPositions(group_, current_q);
        seed.copyJointGroupVelocities(group_, current_qd);
        seed.copyJointGroupAccelerations(group_, current_qdd);
        const JointMotionState motion{current_q, current_qd, current_qdd};
        const double time_sec = start_time + static_cast<double>(step + 1) * dt;
        stamp_target(time_sec);
        const auto result = update(seed, fixed_target, nullptr, nullptr, std::nullopt,
            kSceneRevision, time_sec);
        ASSERT_TRUE(result.tracking) << label << " step=" << step;
        ASSERT_TRUE(result.task) << label << " step=" << step;
        ASSERT_TRUE(result.tracking->face_distance_interval) << label << " step=" << step;
        const auto collision = collision_builder.build(
          *scene, seed, *group_, motion, limits_, dt);
        const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
        ASSERT_TRUE(qp.command_available()) << label << " step=" << step
                                            << " QP=" << static_cast<int>(qp.status);
        const Eigen::VectorXd next_q = current_q + dt * qp.joint_velocity;
        const Eigen::VectorXd next_qdd = (qp.joint_velocity - current_qd) / dt;
        seed.setJointGroupPositions(group_, next_q);
        seed.setJointGroupVelocities(group_, qp.joint_velocity);
        seed.setJointGroupAccelerations(group_, next_qdd);
        seed.update(true);
        ASSERT_TRUE(seed.satisfiesBounds(group_)) << label << " step=" << step;
        const auto actual = collision_builder.validateActualState(*scene, seed, *group_);
        ASSERT_TRUE(actual.input_valid) << label << " step=" << step
                                        << ' ' << actual.failure_reason;
        ASSERT_FALSE(actual.unsafe) << label << " step=" << step
                                    << ' ' << actual.failure_reason;
        const JointMotionState next_motion{next_q, qp.joint_velocity, next_qdd};
        const auto robust = collision_builder.validatePose(
          *scene, seed, *group_, next_motion);
        ASSERT_TRUE(robust.input_valid) << label << " step=" << step
                                        << ' ' << robust.failure_reason;
        ASSERT_FALSE(robust.unsafe) << label << " step=" << step
                                    << ' ' << robust.failure_reason;

        const Eigen::Vector3d screen =
          seed.getGlobalLinkTransform("monitor_control_frame").translation();
        const Eigen::Vector3d elbow = seed.getGlobalLinkTransform("link3").translation();
        const Eigen::Vector3d wrist = seed.getGlobalLinkTransform("link4").translation();
        const double distance = (screen - fixed_face).norm();
        minimum_distance_all = std::min(minimum_distance_all, distance);
        maximum_distance_all = std::max(maximum_distance_all, distance);
        EXPECT_GE(distance, parameters_.face_distance_min_m - one_step_margin)
          << label << " step=" << step;
        EXPECT_LE(distance, parameters_.face_distance_max_m + one_step_margin)
          << label << " step=" << step;
        if (step >= settling_steps) {
          metrics.distance_min = std::min(metrics.distance_min, distance);
          metrics.distance_max = std::max(metrics.distance_max, distance);
          metrics.max_pointing_error = std::max(
            metrics.max_pointing_error, std::abs(result.tracking->pointing_error_rad));
          metrics.max_roll_error = std::max(
            metrics.max_roll_error, std::abs(result.tracking->roll_error_rad));
          metrics.q3_min = std::min(metrics.q3_min, next_q[2]);
          metrics.q3_max = std::max(metrics.q3_max, next_q[2]);
          metrics.final_q3 = next_q[2];
          metrics.final_distance = distance;
          if (qp.status == QpStatus::kPrimaryOnly) {
            ++metrics.primary_only_after_settle;
          }
        }
        if (step >= settling_steps - 1 &&
          (step - (settling_steps - 1)) % sample_stride == 0)
        {
          samples.push_back({screen, elbow, wrist});
        }
      }

      const auto full_range = [](const std::vector<HoldSample> & values, const int point) {
          Eigen::Vector3d minimum = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::infinity());
          Eigen::Vector3d maximum = Eigen::Vector3d::Constant(
            -std::numeric_limits<double>::infinity());
          for (const HoldSample & sample : values) {
            const Eigen::Vector3d & value = point == 0 ? sample.screen :
              (point == 1 ? sample.elbow : sample.wrist);
            minimum = minimum.cwiseMin(value);
            maximum = maximum.cwiseMax(value);
          }
          return (maximum - minimum).norm();
        };
      const auto max_sliding_path = [&](const int point) {
          const std::size_t window = static_cast<std::size_t>(std::llround(
              10.0 / kSamplePeriodSec)) + 1U;
          if (samples.size() < window) {
            return std::numeric_limits<double>::infinity();
          }
          double maximum = 0.0;
          for (std::size_t end = window - 1U; end < samples.size(); ++end) {
            double path = 0.0;
            for (std::size_t index = end - window + 2U; index <= end; ++index) {
              const Eigen::Vector3d & left = point == 0 ? samples[index - 1U].screen :
                (point == 1 ? samples[index - 1U].elbow : samples[index - 1U].wrist);
              const Eigen::Vector3d & right = point == 0 ? samples[index].screen :
                (point == 1 ? samples[index].elbow : samples[index].wrist);
              path += (right - left).norm();
            }
            maximum = std::max(maximum, path);
          }
          return maximum;
        };
      metrics.screen_range = full_range(samples, 0);
      metrics.elbow_range = full_range(samples, 1);
      metrics.wrist_range = full_range(samples, 2);
      metrics.screen_path_10s = max_sliding_path(0);
      metrics.elbow_path_10s = max_sliding_path(1);
      metrics.wrist_path_10s = max_sliding_path(2);
      case_report << " hold_d=" << metrics.distance_min << ".." << metrics.distance_max
                  << " all_d=" << minimum_distance_all << ".." << maximum_distance_all
                  << " final_q3/d=" << metrics.final_q3 << '/' << metrics.final_distance
                  << " q3_hold=" << metrics.q3_min << ".." << metrics.q3_max
                  << " primary_only=" << metrics.primary_only_after_settle
                  << " gaze_roll_max=" << metrics.max_pointing_error << '/'
                  << metrics.max_roll_error
                  << " fullspan(P/E/W)=" << metrics.screen_range << '/'
                  << metrics.elbow_range << '/' << metrics.wrist_range
                  << " path10s(P/E/W)=" << metrics.screen_path_10s << '/'
                  << metrics.elbow_path_10s << '/' << metrics.wrist_path_10s
                  << " samples=" << samples.size();
      RecordProperty(label, case_report.str());
      SCOPED_TRACE(case_report.str());
      EXPECT_EQ(samples.size(), 601U);
      EXPECT_LE(metrics.screen_range, kQuietPathLimitM);
      EXPECT_LE(metrics.elbow_range, kQuietPathLimitM);
      EXPECT_LE(metrics.wrist_range, kQuietPathLimitM);
      EXPECT_LE(metrics.screen_path_10s, kQuietPathLimitM);
      EXPECT_LE(metrics.elbow_path_10s, kQuietPathLimitM);
      EXPECT_LE(metrics.wrist_path_10s, kQuietPathLimitM);
      EXPECT_LE(metrics.distance_max - metrics.distance_min, kQuietPathLimitM);
      EXPECT_LE(metrics.max_pointing_error,
        parameters_.pointing_deadband_rad + angular_allowance);
      EXPECT_LE(metrics.max_roll_error, tracking_config.roll_deadband_rad + angular_allowance);
    };

  CaseMetrics measured_state;
  run_hold("posture1c_lower120_seed_qd_qdd_estimated", seed_qd,
    seed_qdd_estimate, 100.0, measured_state);
  EXPECT_EQ(measured_state.primary_only_after_settle, 0)
    << "The fixed-target hold must not hide secondary fallback after the 5 s settle";

  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(6);
  CaseMetrics zero_motion_control;
  run_hold("posture1c_lower120_seed_zero_qd_qdd_control", zero, zero,
    140.0, zero_motion_control);
  // This control validates that the path/quietness oracle itself passes from
  // rest. The pre-fix shell guard may still select primary-only here; record
  // that separately, but do not make it a second required RED condition.
  EXPECT_LE(zero_motion_control.screen_path_10s, kQuietPathLimitM);
  EXPECT_LE(zero_motion_control.elbow_path_10s, kQuietPathLimitM);
  EXPECT_LE(zero_motion_control.wrist_path_10s, kQuietPathLimitM);
}

TEST_F(MotionReferenceTest, Posture1CLowerBoundaryRampRecoversWithoutGateCycling)
{
  reference_.reset();
  parameters_.position_deadband_m = 0.03;
  reference_ = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);

  // Captured just before the lower-bound target transition in
  // runs/boundaries_fixed_lower50 (sim time 8.6 s). The demo target then uses
  // the recorded quintic smoothstep transition from the original midpoint
  // FACE point to the fixed lower target over 5 s.
  constexpr double kSeedQ[6]{
    6.070616327715556e-07, -0.2510680819273041, -0.37663322650400705,
    1.0392998105627849e-06, -0.9426496677996122, -4.795719809185696e-07};
  const Eigen::Vector3d initial_face(
    0.7499981110465981, -1.017556868225869e-06, 1.149978488041211);
  const Eigen::Vector3d lower_face(
    0.6720479502409709, -7.21194357188413e-07, 1.1499836707827846);
  constexpr double kTargetRampSec = 5.0;
  constexpr double kSettlingSec = 5.0;
  constexpr double kHoldSec = 60.0;
  constexpr double kDistanceQuietLimitM = 0.01;
  const double dt = parameters_.control_period_sec;
  const int ramp_steps = static_cast<int>(std::ceil(kTargetRampSec / dt));
  const int settle_steps = static_cast<int>(std::ceil(kSettlingSec / dt));
  const int hold_steps = static_cast<int>(std::ceil(kHoldSec / dt));
  const int recovery_deadline_hold_steps = static_cast<int>(std::ceil(30.0 / dt));
  const int recovery_deadline_step =
    ramp_steps + settle_steps + recovery_deadline_hold_steps - 1;
  const int total_steps = ramp_steps + settle_steps + hold_steps;
  const double start_time = 80.0;

  moveit::core::RobotState state(model_);
  state.setToDefaultValues();
  Eigen::VectorXd q(6);
  for (Eigen::Index joint = 0; joint < q.size(); ++joint) {
    q[joint] = kSeedQ[joint];
    state.setVariablePosition("joint" + std::to_string(joint + 1), q[joint]);
  }
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd acceleration = Eigen::VectorXd::Zero(6);
  state.setJointGroupVelocities(group_, velocity);
  state.setJointGroupAccelerations(group_, acceleration);
  state.update();
  ASSERT_TRUE(state.satisfiesBounds(group_));

  const auto named_bend = [this](const std::string & name) {
      moveit::core::RobotState named_state(model_);
      if (!named_state.setToDefaultValues(group_, name)) {
        return std::optional<std::pair<double, double>>{};
      }
      named_state.update();
      Eigen::VectorXd named_q;
      named_state.copyJointGroupPositions(group_, named_q);
      const auto * joint = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint);
      if (!joint || !revolute || named_q.size() != 6) {
        return std::optional<std::pair<double, double>>{};
      }
      const auto * parent = joint->getParentLinkModel();
      const auto * child = joint->getChildLinkModel();
      if (!parent || !child) {
        return std::optional<std::pair<double, double>>{};
      }
      const Eigen::Vector3d shoulder =
        named_state.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow =
        named_state.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist =
        named_state.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = named_state.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      const Eigen::Vector3d upper = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      const double angle = std::atan2(
        axis.normalized().dot(upper.cross(forearm)), upper.dot(forearm));
      return std::optional<std::pair<double, double>>{{named_q[2], angle}};
    };
  const auto rest = named_bend("rest");
  const auto search = named_bend("search");
  ASSERT_TRUE(rest);
  ASSERT_TRUE(search);
  const auto bend_calibration = calibratePostureBend(
    rest->first, rest->second, search->first, search->second,
    limits_.lower_position[2], limits_.upper_position[2]);
  ASSERT_TRUE(bend_calibration.valid);
  ASSERT_GE(q[2], bend_calibration.joint3_lower);
  ASSERT_LE(q[2], bend_calibration.joint3_upper)
    << "Captured pre-transition human seed should lie in the calibrated comfort interval";
  const double initial_distance = (
    state.getGlobalLinkTransform("monitor_control_frame").translation() - initial_face).norm();
  ASSERT_GE(initial_distance, parameters_.face_distance_min_m);
  ASSERT_LE(initial_distance, parameters_.face_distance_max_m)
    << "Recorded pre-transition state must begin within the configured FACE band";

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = dt;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;

  auto target = faceAlongLookDirection(state);
  const auto stamp_target = [&target](const double time, const Eigen::Vector3d & face) {
      const auto stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(time * 1.0e9)), RCL_ROS_TIME);
      target.header.stamp = stamp;
      target.face_stamp = stamp;
      target.face.x = face.x();
      target.face.y = face.y();
      target.face.z = face.z();
    };
  stamp_target(start_time, initial_face);
  const auto initial = update(state, target, nullptr, nullptr, std::nullopt,
      kSceneRevision, start_time);
  ASSERT_TRUE(initial.tracking);
  ASSERT_TRUE(initial.task);
  ASSERT_TRUE(initial.tracking->face_distance_interval);

  double minimum_distance = std::numeric_limits<double>::infinity();
  double maximum_distance = 0.0;
  double final_distance = initial_distance;
  double maximum_settled_band_excursion = 0.0;
  std::size_t distance_band_transitions_in_fixed_hold = 0U;
  std::size_t distance_band_transitions_before_qualification = 0U;
  std::size_t posture_mode_transitions_after_qualification = 0U;
  std::size_t good_guard_transitions_after_qualification = 0U;
  int gate_inside_steps_in_fixed_hold = 0;
  int gate_outside_steps_in_fixed_hold = 0;
  int primary_only_steps_in_fixed_hold = 0;
  std::array<int, 7> secondary_status_counts{};
  std::optional<QpFailureSnapshot> first_secondary_failure_snapshot;
  std::optional<QpFailureSnapshot> first_secondary_infeasible_snapshot;
  double q3_total_variation_in_fixed_hold = 0.0;
  std::size_t q3_direction_reversals_in_fixed_hold = 0U;
  double q3_min_after_ramp = std::numeric_limits<double>::infinity();
  double q3_max_after_ramp = -std::numeric_limits<double>::infinity();
  double q3_at_ramp_end = q[2];
  double q3_at_hold_end = q[2];
  double q3_at_recovery_deadline = q[2];
  double distance_at_recovery_deadline = initial_distance;
  int first_comfortable_step = -1;
  int quiet_observation_steps = 0;
  int maximum_qualified_quiet_steps = 0;
  int continuous_qualified_steps = 0;
  int previous_gate_after_qualification = -1;
  std::string previous_posture_mode_after_qualification;
  int previous_good_guard_after_qualification = -1;
  bool quiet_qualification_seen = false;
  bool quiet_steps_preserved_good_mode = true;
  bool quiet_steps_kept_good_guard = true;
  int previous_q3_direction = 0;
  Eigen::VectorXd previous_q = q;
  std::vector<Eigen::Vector3d> screen_samples;
  std::vector<Eigen::Vector3d> elbow_samples;
  std::vector<Eigen::Vector3d> wrist_samples;
  double maximum_screen_path_10s = 0.0;
  double maximum_elbow_path_10s = 0.0;
  double maximum_wrist_path_10s = 0.0;
  std::ostringstream report;

  for (int step = 0; step < total_steps; ++step) {
    const double elapsed = static_cast<double>(step) * dt;
    const double phase = std::clamp(elapsed / kTargetRampSec, 0.0, 1.0);
    const double smooth = phase * phase * phase *
      (phase * (phase * 6.0 - 15.0) + 10.0);
    const Eigen::Vector3d face = initial_face + smooth * (lower_face - initial_face);
    const double now = start_time + elapsed + dt;
    stamp_target(now, face);
    const auto result = update(state, target, nullptr, nullptr, std::nullopt,
        kSceneRevision, now);
    ASSERT_TRUE(result.tracking) << "step=" << step;
    ASSERT_TRUE(result.task) << "step=" << step;
    ASSERT_TRUE(result.tracking->face_distance_interval) << "step=" << step;

    Eigen::VectorXd current_q;
    Eigen::VectorXd current_velocity;
    Eigen::VectorXd current_acceleration;
    state.copyJointGroupPositions(group_, current_q);
    state.copyJointGroupVelocities(group_, current_velocity);
    state.copyJointGroupAccelerations(group_, current_acceleration);
    const JointMotionState motion{current_q, current_velocity, current_acceleration};
    const auto collision = collision_builder.build(
      *scene, state, *group_, motion, limits_, dt);
    const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
    ASSERT_TRUE(qp.command_available()) << "step=" << step
                                        << " QP=" << static_cast<int>(qp.status);
    const auto secondary_status_index = static_cast<std::size_t>(qp.secondary_status);
    if (secondary_status_index < secondary_status_counts.size()) {
      ++secondary_status_counts[secondary_status_index];
    }
    if (!first_secondary_failure_snapshot && qp.secondary_failure_snapshot) {
      first_secondary_failure_snapshot = qp.secondary_failure_snapshot;
    }
    if (!first_secondary_infeasible_snapshot && qp.secondary_failure_snapshot &&
      qp.secondary_status == QpStageStatus::kInfeasible)
    {
      first_secondary_infeasible_snapshot = qp.secondary_failure_snapshot;
    }

    const Eigen::VectorXd next_q = current_q + dt * qp.joint_velocity;
    const Eigen::VectorXd next_acceleration = (qp.joint_velocity - current_velocity) / dt;
    state.setJointGroupPositions(group_, next_q);
    state.setJointGroupVelocities(group_, qp.joint_velocity);
    state.setJointGroupAccelerations(group_, next_acceleration);
    state.update(true);
    ASSERT_TRUE(state.satisfiesBounds(group_)) << "step=" << step;
    const auto actual = collision_builder.validateActualState(*scene, state, *group_);
    ASSERT_TRUE(actual.input_valid) << actual.failure_reason;
    ASSERT_FALSE(actual.unsafe) << actual.failure_reason;

    const Eigen::Vector3d screen =
      state.getGlobalLinkTransform("monitor_control_frame").translation();
    const Eigen::Vector3d elbow = state.getGlobalLinkTransform("link3").translation();
    const Eigen::Vector3d wrist = state.getGlobalLinkTransform("link4").translation();
    const double distance = (screen - face).norm();
    minimum_distance = std::min(minimum_distance, distance);
    maximum_distance = std::max(maximum_distance, distance);
    final_distance = distance;
    if (step == ramp_steps - 1) {
      q3_at_ramp_end = next_q[2];
    }
    const bool fixed_target_hold = step >= ramp_steps;
    const int gate = result.tracking->face_distance_outside_band ? 0 : 1;
    if (fixed_target_hold) {
      if (quiet_qualification_seen) {
        if (previous_gate_after_qualification >= 0 &&
          gate != previous_gate_after_qualification)
        {
          ++distance_band_transitions_in_fixed_hold;
        }
        previous_gate_after_qualification = gate;
      } else {
        if (previous_gate_after_qualification >= 0 &&
          gate != previous_gate_after_qualification)
        {
          ++distance_band_transitions_before_qualification;
        }
        previous_gate_after_qualification = gate;
      }
      gate_inside_steps_in_fixed_hold += gate == 1 ? 1 : 0;
      gate_outside_steps_in_fixed_hold += gate == 0 ? 1 : 0;
      if (qp.status == QpStatus::kPrimaryOnly) {
        ++primary_only_steps_in_fixed_hold;
      }
      q3_min_after_ramp = std::min(q3_min_after_ramp, next_q[2]);
      q3_max_after_ramp = std::max(q3_max_after_ramp, next_q[2]);
      q3_at_hold_end = next_q[2];
      if (step == recovery_deadline_step) {
        q3_at_recovery_deadline = next_q[2];
        distance_at_recovery_deadline = distance;
      }
      const double q3_delta = next_q[2] - previous_q[2];
      q3_total_variation_in_fixed_hold += std::abs(q3_delta);
      const int q3_direction = std::abs(q3_delta) > 1.0e-4 ?
        (q3_delta > 0.0 ? 1 : -1) : 0;
      if (q3_direction != 0) {
        if (previous_q3_direction != 0 && q3_direction != previous_q3_direction) {
          ++q3_direction_reversals_in_fixed_hold;
        }
        previous_q3_direction = q3_direction;
      }
      const bool branch_comfortable =
        next_q[2] >= bend_calibration.joint3_lower &&
        next_q[2] <= bend_calibration.joint3_upper;
      const std::string posture_mode = reference_->diagnostics().posture_mode;
      const bool good_guard_active = !result.task->preferred_primary_constraints.empty();
      const double one_step = parameters_.maximum_linear_reference_mps * dt;
      const bool distance_settled =
        distance >= parameters_.face_distance_min_m - one_step &&
        distance <= parameters_.face_distance_max_m + one_step;
      if (branch_comfortable && first_comfortable_step < 0) {
        first_comfortable_step = step;
      }
      continuous_qualified_steps = branch_comfortable && distance_settled ?
        continuous_qualified_steps + 1 : 0;
      const bool quiet_measurement = continuous_qualified_steps >= settle_steps;
      if (quiet_measurement && !quiet_qualification_seen) {
        quiet_qualification_seen = true;
        previous_gate_after_qualification = gate;
        previous_posture_mode_after_qualification = posture_mode;
        previous_good_guard_after_qualification = good_guard_active ? 1 : 0;
      }
      if (quiet_measurement) {
        if (posture_mode != previous_posture_mode_after_qualification) {
          ++posture_mode_transitions_after_qualification;
        }
        const int good_guard = good_guard_active ? 1 : 0;
        if (good_guard != previous_good_guard_after_qualification) {
          ++good_guard_transitions_after_qualification;
        }
        previous_posture_mode_after_qualification = posture_mode;
        previous_good_guard_after_qualification = good_guard;
        quiet_steps_preserved_good_mode = quiet_steps_preserved_good_mode &&
          posture_mode == "preserve";
        quiet_steps_kept_good_guard = quiet_steps_kept_good_guard && good_guard_active;
        ++quiet_observation_steps;
        maximum_qualified_quiet_steps = std::max(
          maximum_qualified_quiet_steps, quiet_observation_steps);
        maximum_settled_band_excursion = std::max(maximum_settled_band_excursion,
          std::max({0.0, parameters_.face_distance_min_m - one_step - distance,
              distance - parameters_.face_distance_max_m - one_step}));
      } else {
        // Quiet-path evidence starts only after 5 s of simultaneous branch and
        // range qualification; any interruption starts a new quiet window.
        quiet_observation_steps = 0;
        screen_samples.clear();
        elbow_samples.clear();
        wrist_samples.clear();
      }
      if (quiet_measurement && step % 5 == 0) {
        screen_samples.push_back(screen);
        elbow_samples.push_back(elbow);
        wrist_samples.push_back(wrist);
        const std::size_t window_samples = static_cast<std::size_t>(std::ceil(
            10.0 / (5.0 * dt))) + 1U;
        if (screen_samples.size() >= window_samples) {
          const std::size_t begin = screen_samples.size() - window_samples;
          double screen_path = 0.0;
          double elbow_path = 0.0;
          double wrist_path = 0.0;
          for (std::size_t i = begin + 1; i < screen_samples.size(); ++i) {
            screen_path += (screen_samples[i] - screen_samples[i - 1]).norm();
            elbow_path += (elbow_samples[i] - elbow_samples[i - 1]).norm();
            wrist_path += (wrist_samples[i] - wrist_samples[i - 1]).norm();
          }
          maximum_screen_path_10s = std::max(maximum_screen_path_10s, screen_path);
          maximum_elbow_path_10s = std::max(maximum_elbow_path_10s, elbow_path);
          maximum_wrist_path_10s = std::max(maximum_wrist_path_10s, wrist_path);
        }
      }
    }
    previous_q = next_q;
  }

  report << "target_ramp_sec=" << kTargetRampSec
         << " post_ramp_settle_sec=" << kSettlingSec << " fixed_hold_sec=" << kHoldSec
         << " initial_distance=" << initial_distance
         << " distance_range=" << minimum_distance << ".." << maximum_distance
         << " final_distance=" << final_distance
         << " settled_band_excursion=" << maximum_settled_band_excursion
         << " distance_band_transitions=" << distance_band_transitions_in_fixed_hold
         << " gate_inside/outside=" << gate_inside_steps_in_fixed_hold << '/'
         << gate_outside_steps_in_fixed_hold
         << " posture_mode_transitions_after_qualification=" <<
         posture_mode_transitions_after_qualification
         << " good_guard_transitions_after_qualification=" <<
         good_guard_transitions_after_qualification
         << " quiet_good_mode=" << quiet_steps_preserved_good_mode
         << " quiet_good_guard=" << quiet_steps_kept_good_guard
         << " q3_fixed_hold_range=" << q3_min_after_ramp << ".." << q3_max_after_ramp
         << " q3_ramp_end/final=" << q3_at_ramp_end << '/' << q3_at_hold_end
         << " recovery_deadline_sec=" <<
         static_cast<double>(recovery_deadline_hold_steps) * dt
         << " q3_distance_at_recovery_deadline=" << q3_at_recovery_deadline << '/'
         << distance_at_recovery_deadline
         << " q3_total_variation=" << q3_total_variation_in_fixed_hold
         << " q3_direction_reversals=" << q3_direction_reversals_in_fixed_hold
         << " q3_comfort_interval=" << bend_calibration.joint3_lower << ".."
         << bend_calibration.joint3_upper
         << " first_comfortable_step=" << first_comfortable_step
         << " quiet_coverage_sec=" << maximum_qualified_quiet_steps * dt
         << " quiet_current_streak_sec=" << quiet_observation_steps * dt
         << " distance_band_transitions_prequalification=" <<
         distance_band_transitions_before_qualification
         << " primary_only_fixed_hold=" << primary_only_steps_in_fixed_hold
         << " path10s(P/E/W)=" << maximum_screen_path_10s << '/'
         << maximum_elbow_path_10s << '/' << maximum_wrist_path_10s;
  report << " secondary_status_counts(kNotRun,kSolved,kSolvedInaccurate,kInfeasible,"
            "kMaxIterations,kTimeLimit,kSolverError)=";
  for (const int count : secondary_status_counts) {
    report << count << ',';
  }
  const auto append_snapshot_values = [&report](
    const std::string_view name, const auto & values) {
      report << ' ' << name << '[';
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) {
          report << ',';
        }
        report << values[index];
      }
      report << ']';
    };
  const auto append_snapshot = [&report, &append_snapshot_values](
    const std::string_view label, const QpFailureSnapshot & snapshot,
    const char * path_environment) {
      report << ' ' << label << "(stage=" << static_cast<int>(snapshot.stage)
             << ",status=" << static_cast<int>(snapshot.classified_status)
             << ",n=" << snapshot.variable_count << ",m=" << snapshot.constraint_count
             << ",setup=" << snapshot.setup_return_code
             << ",solve=" << snapshot.solve_return_code << ")";
      append_snapshot_values("P_col", snapshot.p_column_pointers);
      append_snapshot_values("P_row", snapshot.p_row_indices);
      append_snapshot_values("P", snapshot.p_values);
      append_snapshot_values("A_col", snapshot.a_column_pointers);
      append_snapshot_values("A_row", snapshot.a_row_indices);
      append_snapshot_values("A", snapshot.a_values);
      append_snapshot_values("l", snapshot.lower);
      append_snapshot_values("u", snapshot.upper);
      if (const char * snapshot_path = std::getenv(path_environment)) {
        std::ofstream snapshot_file(snapshot_path);
        const bool saved = snapshot_file.good() &&
          writeQpFailureSnapshot(snapshot_file, snapshot);
        report << " snapshot_file=" << snapshot_path << ":" << (saved ? "saved" : "failed");
      }
    };
  if (first_secondary_failure_snapshot) {
    append_snapshot("first_secondary_failure", *first_secondary_failure_snapshot,
      "POSTURE1C_QP_SNAPSHOT_PATH");
  } else {
    report << " first_secondary_failure=none";
  }
  if (first_secondary_infeasible_snapshot) {
    append_snapshot("first_secondary_infeasible", *first_secondary_infeasible_snapshot,
      "POSTURE1C_QP_INFEASIBLE_SNAPSHOT_PATH");
  } else {
    report << " first_secondary_infeasible=none";
  }
  RecordProperty("posture1c_lower_boundary_ramp", report.str());

  // The distance-band bit is diagnostic only. Acceptance follows actual posture
  // mode and GOOD guard continuity after five seconds in the calibrated branch.
  EXPECT_LE(posture_mode_transitions_after_qualification, 2U) << report.str();
  EXPECT_LE(good_guard_transitions_after_qualification, 2U) << report.str();
  EXPECT_TRUE(quiet_steps_preserved_good_mode) << report.str();
  EXPECT_TRUE(quiet_steps_kept_good_guard) << report.str();
  EXPECT_GE(q3_at_recovery_deadline, bend_calibration.joint3_lower) << report.str();
  EXPECT_LE(q3_at_recovery_deadline, bend_calibration.joint3_upper) << report.str();
  EXPECT_GE(distance_at_recovery_deadline,
    parameters_.face_distance_min_m - parameters_.maximum_linear_reference_mps * dt)
    << report.str();
  EXPECT_LE(distance_at_recovery_deadline,
    parameters_.face_distance_max_m + parameters_.maximum_linear_reference_mps * dt)
    << report.str();
  EXPECT_GE(q3_at_hold_end, bend_calibration.joint3_lower) << report.str();
  EXPECT_LE(q3_at_hold_end, bend_calibration.joint3_upper) << report.str();
  const double final_band_margin = parameters_.maximum_linear_reference_mps * dt;
  EXPECT_GE(final_distance, parameters_.face_distance_min_m - final_band_margin)
    << report.str();
  EXPECT_LE(final_distance, parameters_.face_distance_max_m + final_band_margin)
    << report.str();
  EXPECT_GE(maximum_qualified_quiet_steps * dt, 30.0) << report.str();
  EXPECT_LE(maximum_settled_band_excursion, 0.0) << report.str();
  EXPECT_LE(maximum_screen_path_10s, kDistanceQuietLimitM) << report.str();
  EXPECT_LE(maximum_elbow_path_10s, kDistanceQuietLimitM) << report.str();
  EXPECT_LE(maximum_wrist_path_10s, kDistanceQuietLimitM) << report.str();
}

TEST_F(MotionReferenceTest, Posture1CBad10RecordedEndpointRollsIntoNamedBendBranch)
{
  // Match the production FACE tracking deadband used by runtime captures.
  reference_.reset();
  parameters_.position_deadband_m = 0.03;
  reference_ = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  // This is the actual final joint state and fixed face target from
  // runs/bad10_final, where the previous scalar elbow-side check said “good”
  // although the rendered arm still had the wrong shape.
  constexpr double kRecordedQ[6]{
    -5.379036931941633e-06, 1.7178940940293195, -3.6019600429057217,
    1.3069379082891277e-06, 0.2890940863373446, 2.934930766309014e-06};
  const Eigen::Vector3d face(
    1.0510895434142848, -8.73348787353692e-07, 1.1892379220020002);

  const auto named_bend = [this](const std::string & name) {
      moveit::core::RobotState state(model_);
      if (!state.setToDefaultValues(group_, name)) {
        return std::optional<std::pair<double, double>>{};
      }
      state.update();
      Eigen::VectorXd q;
      state.copyJointGroupPositions(group_, q);
      const auto * joint_model = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint_model);
      if (!revolute || q.size() != 6) {
        return std::optional<std::pair<double, double>>{};
      }
      const auto * parent = joint_model->getParentLinkModel();
      const auto * child = joint_model->getChildLinkModel();
      if (!parent || !child) {
        return std::optional<std::pair<double, double>>{};
      }
      const Eigen::Vector3d shoulder = state.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow = state.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist = state.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = state.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      const Eigen::Vector3d upper_arm = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      const double angle = std::atan2(
        axis.normalized().dot(upper_arm.cross(forearm)), upper_arm.dot(forearm));
      return std::optional<std::pair<double, double>>{{q[2], angle}};
    };
  const auto rest = named_bend("rest");
  const auto search = named_bend("search");
  ASSERT_TRUE(rest);
  ASSERT_TRUE(search);
  const auto calibration = calibratePostureBend(
    rest->first, rest->second, search->first, search->second,
    limits_.lower_position[2], limits_.upper_position[2]);
  ASSERT_TRUE(calibration.valid);

  moveit::core::RobotState robot_state(model_);
  robot_state.setToDefaultValues();
  Eigen::VectorXd q(6);
  for (Eigen::Index i = 0; i < q.size(); ++i) {
    q[i] = kRecordedQ[i];
    robot_state.setVariablePosition("joint" + std::to_string(i + 1), q[i]);
  }
  robot_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  robot_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  robot_state.update();
  ASSERT_TRUE(robot_state.satisfiesBounds(group_));
  EXPECT_LT(q[2], calibration.joint3_lower);

  const auto bend_sin = [this, &calibration](const moveit::core::RobotState & state) {
      const auto * joint_model = model_->getJointModel("joint3");
      const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint_model);
      const auto * parent = joint_model->getParentLinkModel();
      const auto * child = joint_model->getChildLinkModel();
      const Eigen::Vector3d shoulder = state.getGlobalLinkTransform("link2").translation();
      const Eigen::Vector3d elbow = state.getGlobalLinkTransform("link3").translation();
      const Eigen::Vector3d wrist = state.getGlobalLinkTransform("link4").translation();
      const Eigen::Vector3d axis = state.getGlobalLinkTransform(parent).linear() *
        child->getJointOriginTransform().linear() * revolute->getAxis();
      const Eigen::Vector3d upper_arm = elbow - shoulder;
      const Eigen::Vector3d forearm = wrist - elbow;
      const double angle = std::atan2(
        axis.normalized().dot(upper_arm.cross(forearm)), upper_arm.dot(forearm));
      return calibration.preferred_sign * std::sin(angle);
    };
  EXPECT_LT(bend_sin(robot_state), calibration.minimum_comfortable_sin);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  auto target = faceAlongLookDirection(robot_state);
  target.face.x = face.x();
  target.face.y = face.y();
  target.face.z = face.z();
  std::ostringstream rollout;
  double minimum_distance = std::numeric_limits<double>::infinity();
  double maximum_distance = 0.0;
  double maximum_pointing_error = 0.0;
  double maximum_roll_error = 0.0;
  double maximum_distance_rate_violation = 0.0;
  double maximum_unconstrained_outward_rate = 0.0;
  int maximum_unconstrained_rate_step = -1;
  double maximum_unconstrained_rate_distance = 0.0;
  int primary_only_steps = 0;
  int strict_band_excursion_steps = 0;
  int expanded_band_excursion_steps = 0;
  std::array<int, 7> secondary_status_counts{};
  int steps_taken = 0;
  int continuous_comfort_steps = 0;
  int continuous_settled_steps = 0;
  int first_comfort_hold_step = -1;
  int settled_step = -1;
  double maximum_strict_band_excursion = 0.0;
  double final_pointing_error = std::numeric_limits<double>::quiet_NaN();
  double final_roll_error = std::numeric_limits<double>::quiet_NaN();
  std::ostringstream fallback_steps;
  std::ostringstream detailed_status_steps;
  bool reached_comfort = false;
  struct QuietSample
  {
    double time_sec;
    double distance_m;
    Eigen::Vector3d screen;
    Eigen::Vector3d elbow;
    Eigen::Vector3d wrist;
  };
  std::vector<QuietSample> quiet_samples;
  double maximum_screen_path_10s = 0.0;
  double maximum_elbow_path_10s = 0.0;
  double maximum_wrist_path_10s = 0.0;
  double maximum_distance_span_10s = 0.0;
  double maximum_settled_band_excursion = 0.0;
  int primary_only_after_settling = 0;
  double settled_distance_min = std::numeric_limits<double>::infinity();
  double settled_distance_max = 0.0;
  const int required_comfort_hold_steps = static_cast<int>(std::ceil(
      2.0 / parameters_.control_period_sec));
  const int required_settling_steps = static_cast<int>(std::ceil(
      5.0 / parameters_.control_period_sec));
  const int required_observation_steps = static_cast<int>(std::ceil(
      30.0 / parameters_.control_period_sec));
  const double initial_distance = (
    robot_state.getGlobalLinkTransform("monitor_control_frame").translation() - face).norm();
  minimum_distance = initial_distance;
  maximum_distance = initial_distance;
  if (initial_distance < parameters_.face_distance_min_m ||
    initial_distance > parameters_.face_distance_max_m)
  {
    ++strict_band_excursion_steps;
    maximum_strict_band_excursion = std::max({
          0.0, parameters_.face_distance_min_m - initial_distance,
          initial_distance - parameters_.face_distance_max_m});
  }
  const double distance_activation_margin = parameters_.maximum_linear_reference_mps *
    parameters_.control_period_sec;
  const double initial_expanded_excursion = std::max({
        0.0, parameters_.face_distance_min_m - distance_activation_margin - initial_distance,
        initial_distance - parameters_.face_distance_max_m - distance_activation_margin});
  if (initial_expanded_excursion > 0.0) {
    ++expanded_band_excursion_steps;
  }
  // The recorded BAD10 model recovery takes about 12.4 s; allow 15 s to reach
  // the branch, then five seconds to settle and a full 30 s fixed-F hold.
  const int recovery_budget_steps = static_cast<int>(std::ceil(
      15.0 / parameters_.control_period_sec));
  const int maximum_steps = recovery_budget_steps + required_settling_steps +
    required_observation_steps;
  const double settling_band_margin = distance_activation_margin;
  for (int step = 0; step < maximum_steps; ++step) {
    steps_taken = step + 1;
    Eigen::VectorXd current_q;
    robot_state.copyJointGroupPositions(group_, current_q);
    Eigen::VectorXd current_velocity;
    robot_state.copyJointGroupVelocities(group_, current_velocity);
    Eigen::VectorXd current_acceleration;
    robot_state.copyJointGroupAccelerations(group_, current_acceleration);
    const JointMotionState motion{
      current_q, current_velocity, current_acceleration};
    const double simulation_time = static_cast<double>(step + 1) *
      parameters_.control_period_sec;
    const auto target_stamp = rclcpp::Time(
      static_cast<std::int64_t>(std::llround(simulation_time * 1.0e9)), RCL_ROS_TIME);
    target.header.stamp = target_stamp;
    target.face_stamp = target_stamp;
    const auto result = update(
      robot_state, target, nullptr, nullptr, std::nullopt, kSceneRevision, simulation_time);
    ASSERT_TRUE(result.tracking) << reference_->diagnostics().state;
    ASSERT_TRUE(result.task);
    const Eigen::Vector3d monitor_position =
      robot_state.getGlobalLinkTransform("monitor_control_frame").translation();
    const double current_distance = (monitor_position - face).norm();
    minimum_distance = std::min(minimum_distance, current_distance);
    maximum_distance = std::max(maximum_distance, current_distance);
    maximum_pointing_error = std::max(
      maximum_pointing_error, std::abs(result.tracking->pointing_error_rad));
    maximum_roll_error = std::max(maximum_roll_error, std::abs(result.tracking->roll_error_rad));
    final_pointing_error = result.tracking->pointing_error_rad;
    final_roll_error = result.tracking->roll_error_rad;
    ASSERT_TRUE(result.tracking->face_distance_interval);
    const auto collision = collision_builder.build(
      *scene, robot_state, *group_, motion, limits_, parameters_.control_period_sec);
    const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
    ASSERT_TRUE(qp.command_available()) << "step=" << step
                                        << " status=" << static_cast<int>(qp.status)
                                        << " q=" << current_q.transpose();
    ASSERT_EQ(qp.joint_velocity.size(), 6);
    const double feasibility_tolerance = qp_config.solution_feasibility_tolerance +
      4.0 * qp_config.solver_absolute_tolerance;
    const auto & distance = *result.tracking->face_distance_interval;
    const double distance_rate = distance.coefficients.dot(qp.joint_velocity);
    const bool distance_interval_applied = std::any_of(
      result.task->secondary_constraints.begin(), result.task->secondary_constraints.end(),
      [&distance](const LinearVelocityConstraint & constraint) {
        return constraint.coefficients.size() == distance.coefficients.size() &&
               constraint.coefficients.isApprox(distance.coefficients, 1.0e-12) &&
               std::abs(constraint.lower_bound - distance.lower_bound) <= 1.0e-12 &&
               std::abs(constraint.upper_bound - distance.upper_bound) <= 1.0e-12;
      });
    EXPECT_FALSE(distance_interval_applied)
      << "step=" << step << " outside_band="
      << result.tracking->face_distance_outside_band << " secondary_rows="
      << result.task->secondary_constraints.size();
    const double rate_violation = std::max({
          0.0, distance.lower_bound - feasibility_tolerance - distance_rate,
          distance_rate - distance.upper_bound - feasibility_tolerance});
    if (distance_interval_applied) {
      maximum_distance_rate_violation = std::max(maximum_distance_rate_violation, rate_violation);
    } else {
      const double outward_rate = current_distance > parameters_.face_distance_max_m ?
        std::max(0.0, distance_rate) :
        (current_distance < parameters_.face_distance_min_m ? std::max(0.0, -distance_rate) : 0.0);
      if (outward_rate > maximum_unconstrained_outward_rate) {
        maximum_unconstrained_outward_rate = outward_rate;
        maximum_unconstrained_rate_step = step;
        maximum_unconstrained_rate_distance = current_distance;
      }
    }
    const auto secondary_status_index = static_cast<std::size_t>(qp.secondary_status);
    if (secondary_status_index < secondary_status_counts.size()) {
      ++secondary_status_counts[secondary_status_index];
    }
    if (qp.status == QpStatus::kPrimaryOnly) {
      // A failed secondary stage retains the higher-ranked gaze command. The
      // distance objective remains a lower-priority soft request.
      ++primary_only_steps;
      if (settled_step >= 0) {
        ++primary_only_after_settling;
      }
      if (primary_only_steps <= 12 || result.tracking->face_distance_outside_band ||
        (step >= 515 && step <= 530))
      {
        detailed_status_steps << step << ':' << static_cast<int>(qp.secondary_status)
                              << ":rows=" << result.task->secondary_matrix.rows()
                              << ":rate=" << distance_rate << '/' << distance.lower_bound
                              << ".." << distance.upper_bound << ":q3=" << current_q[2]
                              << ":d=" << current_distance << ":gaze="
                              << result.tracking->pointing_error_rad << '/'
                              << result.tracking->roll_error_rad << ' ';
      }
      fallback_steps << step << ':' << static_cast<int>(qp.secondary_status) << ' ';
    } else if (distance_interval_applied) {
      EXPECT_LE(rate_violation, 0.0) << "step=" << step << " status="
                                     << static_cast<int>(qp.status) << " secondary="
                                     << static_cast<int>(qp.secondary_status) << " rows="
                                     << result.task->secondary_matrix.rows() << " rate=" <<
        distance_rate
                                     << " interval=" << distance.lower_bound << ".." <<
        distance.upper_bound;
    }
    ASSERT_GT(result.task->primary_matrix.rows(), 0);

    const Eigen::VectorXd next_q = current_q + parameters_.control_period_sec *
      qp.joint_velocity;
    const Eigen::VectorXd next_acceleration =
      (qp.joint_velocity - current_velocity) / parameters_.control_period_sec;
    robot_state.setJointGroupPositions(group_, next_q);
    robot_state.setJointGroupVelocities(group_, qp.joint_velocity);
    robot_state.setJointGroupAccelerations(group_, next_acceleration);
    robot_state.update(true);
    ASSERT_TRUE(robot_state.satisfiesBounds(group_)) << "step=" << step;
    const auto actual = collision_builder.validateActualState(*scene, robot_state, *group_);
    ASSERT_TRUE(actual.input_valid) << "step=" << step << " " << actual.failure_reason;
    ASSERT_FALSE(actual.unsafe) << "step=" << step << " " << actual.failure_reason;
    const Eigen::Vector3d next_monitor_position =
      robot_state.getGlobalLinkTransform("monitor_control_frame").translation();
    const double next_distance = (next_monitor_position - face).norm();
    minimum_distance = std::min(minimum_distance, next_distance);
    maximum_distance = std::max(maximum_distance, next_distance);
    const double next_band_excursion = std::max({
          0.0, parameters_.face_distance_min_m - next_distance,
          next_distance - parameters_.face_distance_max_m});
    if (next_band_excursion > 0.0) {
      ++strict_band_excursion_steps;
      maximum_strict_band_excursion = std::max(
        maximum_strict_band_excursion, next_band_excursion);
    }
    const double next_expanded_excursion = std::max({
          0.0, parameters_.face_distance_min_m - distance_activation_margin - next_distance,
          next_distance - parameters_.face_distance_max_m - distance_activation_margin});
    if (next_expanded_excursion > 0.0) {
      ++expanded_band_excursion_steps;
    }
    rollout << step << " q=" << next_q.transpose()
            << " bend_sin=" << bend_sin(robot_state)
            << " d=" << current_distance << "->" << next_distance
            << " pointing_roll=" << result.tracking->pointing_error_rad << '/'
            << result.tracking->roll_error_rad
            << " qp=" << static_cast<int>(qp.status) << '/'
            << static_cast<int>(qp.secondary_status)
            << " interval_applied=" << distance_interval_applied
            << " d_rate=" << distance_rate << " bounds=" << distance.lower_bound
            << ".." << distance.upper_bound << " collision_clearance="
            << actual.minimum_distance_m << " qdot=" << qp.joint_velocity.transpose() << '\n';
    const bool in_comfort = next_q[2] >= calibration.joint3_lower &&
      next_q[2] <= calibration.joint3_upper &&
      bend_sin(robot_state) >= calibration.minimum_comfortable_sin;
    const bool in_settling_band =
      next_distance >= parameters_.face_distance_min_m - settling_band_margin &&
      next_distance <= parameters_.face_distance_max_m + settling_band_margin;
    continuous_comfort_steps = in_comfort && in_settling_band ?
      continuous_comfort_steps + 1 : 0;
    if (!reached_comfort && continuous_comfort_steps >= required_comfort_hold_steps) {
      reached_comfort = true;
      first_comfort_hold_step = step;
    }
    if (reached_comfort) {
      continuous_settled_steps = in_comfort && in_settling_band ?
        continuous_settled_steps + 1 : 0;
      if (settled_step < 0 && continuous_settled_steps >= required_settling_steps) {
        settled_step = step;
      }
    }
    if (settled_step >= 0) {
      const double settled_excursion = std::max({
            0.0, parameters_.face_distance_min_m - settling_band_margin - next_distance,
            next_distance - parameters_.face_distance_max_m - settling_band_margin});
      maximum_settled_band_excursion = std::max(
        maximum_settled_band_excursion, settled_excursion);
    }
    if (settled_step >= 0 && step >= settled_step && step % 5 == 0) {
      quiet_samples.push_back({
            simulation_time, next_distance, next_monitor_position,
            robot_state.getGlobalLinkTransform("link3").translation(),
            robot_state.getGlobalLinkTransform("link4").translation()});
      settled_distance_min = std::min(settled_distance_min, next_distance);
      settled_distance_max = std::max(settled_distance_max, next_distance);
      const std::size_t samples_per_10s = static_cast<std::size_t>(std::ceil(
          10.0 / (5.0 * parameters_.control_period_sec))) + 1U;
      if (quiet_samples.size() >= samples_per_10s) {
        const std::size_t begin = quiet_samples.size() - samples_per_10s;
        double screen_path = 0.0;
        double elbow_path = 0.0;
        double wrist_path = 0.0;
        double window_distance_min = std::numeric_limits<double>::infinity();
        double window_distance_max = 0.0;
        for (std::size_t i = begin; i < quiet_samples.size(); ++i) {
          window_distance_min = std::min(window_distance_min, quiet_samples[i].distance_m);
          window_distance_max = std::max(window_distance_max, quiet_samples[i].distance_m);
          if (i > begin) {
            screen_path += (quiet_samples[i].screen - quiet_samples[i - 1].screen).norm();
            elbow_path += (quiet_samples[i].elbow - quiet_samples[i - 1].elbow).norm();
            wrist_path += (quiet_samples[i].wrist - quiet_samples[i - 1].wrist).norm();
          }
        }
        maximum_screen_path_10s = std::max(maximum_screen_path_10s, screen_path);
        maximum_elbow_path_10s = std::max(maximum_elbow_path_10s, elbow_path);
        maximum_wrist_path_10s = std::max(maximum_wrist_path_10s, wrist_path);
        maximum_distance_span_10s = std::max(
          maximum_distance_span_10s, window_distance_max - window_distance_min);
      }
    }
  }
  RecordProperty("posture1c_bad10_bounded_model_rollout", rollout.str());
  std::ostringstream summary;
  summary << "POSTURE-1C BAD10 rollout steps=" << steps_taken
          << " reached_branch_and_settled_distance=" << reached_comfort << " q=";
  Eigen::VectorXd final_q;
  robot_state.copyJointGroupPositions(group_, final_q);
  const double final_distance = (
    robot_state.getGlobalLinkTransform("monitor_control_frame").translation() - face).norm();
  summary << final_q.transpose()
          << " E=" << robot_state.getGlobalLinkTransform("link3").translation().transpose()
          << " W=" << robot_state.getGlobalLinkTransform("link4").translation().transpose()
          << " F=" << face.transpose() << " bend_sin=" << bend_sin(robot_state)
          << " comfort_sin=" << calibration.minimum_comfortable_sin
          << " distance_range_m=" << minimum_distance << ".." << maximum_distance
          << " final_distance_m=" << final_distance
          << " allowed_m=" << parameters_.face_distance_min_m << ".."
          << parameters_.face_distance_max_m << " strict_band_excursion_samples="
          << strict_band_excursion_steps << " strict_excursion_sampled_duration_s="
          << strict_band_excursion_steps * parameters_.control_period_sec
          << " strict_excursion_max_m=" << maximum_strict_band_excursion
          << " transient_tolerance_m=" << parameters_.position_deadband_m
          << " settling_margin_m=" << settling_band_margin
          << " required_settling_hold_s=" << required_comfort_hold_steps *
          parameters_.control_period_sec
          << " first_comfort_hold_time_s=" << (first_comfort_hold_step >= 0 ?
  (first_comfort_hold_step + 1) * parameters_.control_period_sec : -1.0)
          << " settled_time_s=" << (settled_step >= 0 ?
  (settled_step + 1) * parameters_.control_period_sec : -1.0)
          << " post_settle_observation_s=" << (quiet_samples.empty() ? 0.0 :
  quiet_samples.back().time_sec - quiet_samples.front().time_sec)
          << " settled_distance_span_m=" << (settled_distance_max - settled_distance_min)
          << " quiet_max_screen_path_10s_m=" << maximum_screen_path_10s
          << " quiet_max_elbow_path_10s_m=" << maximum_elbow_path_10s
          << " quiet_max_wrist_path_10s_m=" << maximum_wrist_path_10s
          << " quiet_max_distance_span_10s_m=" << maximum_distance_span_10s
          << " max_settled_band_excursion_m=" << maximum_settled_band_excursion
          << " outside_activation_margin_samples=" << expanded_band_excursion_steps
          << " outside_activation_margin_duration_s="
          << expanded_band_excursion_steps * parameters_.control_period_sec
          << " max_pointing_error_rad="
          << maximum_pointing_error << " max_roll_error_rad=" << maximum_roll_error
          << " final_pointing_error_rad=" << final_pointing_error
          << " final_roll_error_rad=" << final_roll_error
          << " primary_only_steps=" << primary_only_steps
          << " primary_only_after_settling=" << primary_only_after_settling
          << " max_applied_interval_rate_violation_mps=" << maximum_distance_rate_violation
          << " max_unconstrained_outward_rate_mps=" << maximum_unconstrained_outward_rate
          << " max_unconstrained_outward_rate_step=" << maximum_unconstrained_rate_step
          << " max_unconstrained_outward_rate_distance_m="
          << maximum_unconstrained_rate_distance
          << " secondary_status_counts(kNotRun,kSolved,kSolvedInaccurate,kInfeasible,"
             "kMaxIterations,kTimeLimit,kSolverError)=";
  for (const int count : secondary_status_counts) {
    summary << count << ',';
  }
  summary << " primary_only_step_ids(stage2_status)=" << fallback_steps.str()
          << " fallback_details(step:stage2:rows:rate/interval:q3:d:gaze/roll)="
          << detailed_status_steps.str();
  RCLCPP_INFO(node_->get_logger(), "%s", summary.str().c_str());
  // BAD10 bend recovery has higher priority than distance. Keep recording the
  // transient excursion in the rollout report, but require distance to return
  // to the settled band below instead of imposing the former 3 cm transient cap.
  const TrackingVelocityTaskConfig tracking_config;
  const double one_cycle_angular_allowance =
    parameters_.maximum_angular_reference_radps * parameters_.control_period_sec;
  EXPECT_LE(maximum_pointing_error,
    parameters_.pointing_deadband_rad + one_cycle_angular_allowance)
       << "Pointing error exceeded its configured deadband plus one control-cycle allowance; "
       << summary.str();
  EXPECT_LE(maximum_roll_error,
    tracking_config.roll_deadband_rad + one_cycle_angular_allowance)
       << "Roll error exceeded its configured deadband plus one control-cycle allowance; "
       << summary.str();
  EXPECT_TRUE(reached_comfort)
    << "BAD10 must reach the calibrated rest/search bend branch and hold it inside the "
       "settling band; final q3="
    << robot_state.getVariablePosition("joint3")
    << " FK signed-bend sin=" << bend_sin(robot_state)
    << " comfort sin=" << calibration.minimum_comfortable_sin;
  EXPECT_GE(settled_step, 0)
    << "BAD10 needs five continuous seconds in the calibrated branch and settling band; "
    << summary.str();
  EXPECT_GE(quiet_samples.size(), 2U)
    << "BAD10 needs a full ten-second settled observation window; " << summary.str();
  EXPECT_GE(quiet_samples.empty() ? 0.0 :
    quiet_samples.back().time_sec - quiet_samples.front().time_sec, 29.0)
    << "The fixed-F hold must continue for 30 seconds after settling; " << summary.str();
  EXPECT_LE(settled_distance_max - settled_distance_min, 0.01)
    << "Settled face-distance span exceeded 1 cm; " << summary.str();
  EXPECT_LE(maximum_screen_path_10s, 0.01)
    << "Screen-center path exceeded 1 cm in a settled ten-second window; " << summary.str();
  EXPECT_LE(maximum_elbow_path_10s, 0.01)
    << "Elbow path exceeded 1 cm in a settled ten-second window; " << summary.str();
  EXPECT_LE(maximum_wrist_path_10s, 0.01)
    << "Wrist path exceeded 1 cm in a settled ten-second window; " << summary.str();
  EXPECT_LE(maximum_distance_span_10s, 0.01)
    << "Face-distance span exceeded 1 cm in a settled ten-second window; " << summary.str();
  EXPECT_LE(maximum_settled_band_excursion, 0.0)
    << "Face distance left the existing one-step settled band; " << summary.str();
  EXPECT_EQ(primary_only_after_settling, 0)
    << "The settled fixed-F hold must not fall back to primary-only commands; " << summary.str();
}

TEST_F(MotionReferenceTest, Posture1DBadStartRecoversAcrossNearBandAndFarTargets)
{
  constexpr double kRecordedQ[6]{
    -5.379036931941633e-06, 1.7178940940293195, -3.6019600429057217,
    1.3069379082891277e-06, 0.2890940863373446, 2.934930766309014e-06};
  constexpr double kComfortLower = -1.95514;
  constexpr double kComfortUpper = -0.3696309078954824;
  constexpr double kMinimumBendSin = 0.701883982289872;
  parameters_.position_deadband_m = 0.03;
  const double dt = parameters_.control_period_sec;
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = dt;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  CollisionConstraintBuilder collision_builder(collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  const std::array<double, 3> requested_distances{
    parameters_.face_distance_min_m - 0.02,
    0.5 * (parameters_.face_distance_min_m + parameters_.face_distance_max_m),
    parameters_.face_distance_max_m + 0.35};
  const std::array<double, 2> safe_bad_distances{
    parameters_.face_distance_min_m - 0.02,
    parameters_.face_distance_max_m + 0.35};
  for (const double requested_distance : safe_bad_distances) {
    reference_ = std::make_unique<MotionReference>(
      node_, monitor_, parameters_, limits_, collision_config_);
    moveit::core::RobotState safe_bad_state(model_);
    safe_bad_state.setToDefaultValues();
    safe_bad_state.setVariablePosition("joint1", kRecordedQ[0]);
    safe_bad_state.setVariablePosition("joint2", kRecordedQ[1]);
    safe_bad_state.setVariablePosition("joint3", kRecordedQ[2]);
    safe_bad_state.setVariablePosition("joint4", kRecordedQ[3]);
    safe_bad_state.setVariablePosition("joint5", kRecordedQ[4]);
    safe_bad_state.setVariablePosition("joint6", 6.20);
    safe_bad_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    safe_bad_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    safe_bad_state.update(true);
    ASSERT_TRUE(safe_bad_state.satisfiesBounds(group_));
    Eigen::VectorXd q, qd, qdd;
    safe_bad_state.copyJointGroupPositions(group_, q);
    safe_bad_state.copyJointGroupVelocities(group_, qd);
    safe_bad_state.copyJointGroupAccelerations(group_, qdd);
    ASSERT_GT(q[4] - limits_.lower_position[4], limits_.position_margin[4]);
    ASSERT_GT(limits_.upper_position[4] - q[4], limits_.position_margin[4]);
    ASSERT_LT(limits_.upper_position[5] - q[5], limits_.position_margin[5]);
    const JointMotionState motion{q, qd, qdd};
    auto face_target = faceAlongLookDirection(safe_bad_state);
    const Eigen::Vector3d screen = safe_bad_state.getGlobalLinkTransform(
      "monitor_control_frame").translation();
    const Eigen::Vector3d initial_face(face_target.face.x, face_target.face.y,
      face_target.face.z);
    const Eigen::Vector3d face = screen + requested_distance *
      (initial_face - screen).normalized();
    face_target.face.x = face.x();
    face_target.face.y = face.y();
    face_target.face.z = face.z();
    const auto stamp = rclcpp::Time(81'000'000'000LL, RCL_ROS_TIME);
    face_target.header.stamp = stamp;
    face_target.face_stamp = stamp;
    const auto result = update(
      safe_bad_state, face_target, nullptr, nullptr, motion, kSceneRevision, 81.0);
    ASSERT_TRUE(result.task);
    EXPECT_EQ(reference_->diagnostics().posture_mode, "recover");
    ASSERT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
    ASSERT_FALSE(result.compact_roll_constraints.empty());
    const auto has_inward_j6_row = [](
      const Eigen::MatrixXd & matrix, const Eigen::VectorXd & reference) {
        for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
          const Eigen::RowVectorXd coefficients = matrix.row(row);
          if (coefficients.size() == 6 && std::abs(coefficients[5] - 1.0) < 1.0e-9 &&
            coefficients.head(5).norm() < 1.0e-9 && reference[row] < 0.0)
          {
            return true;
          }
        }
        return false;
      };
    const bool has_inward_j6_margin_objective =
      has_inward_j6_row(result.task->secondary_matrix, result.task->secondary_reference) ||
      has_inward_j6_row(result.task->tertiary_matrix, result.task->tertiary_reference);
    ASSERT_TRUE(has_inward_j6_margin_objective)
      << "The BAD witness must exercise the active J6 soft margin";
    CollisionConstraintBuilder safe_bad_collision_builder(collision_config_);
    planning_scene_monitor::LockedPlanningSceneRO safe_bad_scene(monitor_);
    ASSERT_TRUE(safe_bad_scene);
    const planning_scene::PlanningSceneConstPtr safe_bad_planning_scene = safe_bad_scene;
    const auto collision = safe_bad_collision_builder.build(
      *safe_bad_planning_scene, safe_bad_state, *group_, motion, limits_, dt);
    ASSERT_TRUE(collision.diagnostics.valid);
    ASSERT_FALSE(collision.diagnostics.collision_detected)
      << collision.diagnostics.closest_pair << " clearance="
      << collision.diagnostics.minimum_distance_m;
    auto constraints = collision.constraints;
    constraints.insert(
      constraints.end(), result.compact_roll_constraints.begin(),
      result.compact_roll_constraints.end());
    const QpResult qp = solver.solve(*result.task, motion, limits_, constraints);
    ASSERT_TRUE(qp.command_available())
      << "distance=" << requested_distance
      << " qp_status=" << static_cast<int>(qp.status)
      << " ranks=" << static_cast<int>(qp.primary_status) << '/'
      << static_cast<int>(qp.secondary_status) << '/'
      << static_cast<int>(qp.tertiary_status) << '/'
      << static_cast<int>(qp.distance_status)
      << " collision_valid=" << collision.diagnostics.valid
      << " collision_pair=" << collision.diagnostics.closest_pair
      << " min_distance_m=" << collision.diagnostics.minimum_distance_m
      << " roll_rows=" << result.compact_roll_constraints.size();
    EXPECT_GT(qp.joint_velocity[2], 0.0)
      << "The BAD pose has available inward q3 recovery";
    const Eigen::VectorXd next_q = q + dt * qp.joint_velocity;
    safe_bad_state.setJointGroupPositions(group_, next_q);
    safe_bad_state.update(true);
    EXPECT_TRUE(safe_bad_state.satisfiesBounds(group_));
  }

  std::ostringstream summary;
  double time_sec = 80.0;

  for (const double requested_distance : requested_distances) {
    reference_.reset();
    reference_ = std::make_unique<MotionReference>(
      node_, monitor_, parameters_, limits_, collision_config_);
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    for (Eigen::Index joint = 0; joint < 6; ++joint) {
      state.setVariablePosition("joint" + std::to_string(joint + 1), kRecordedQ[joint]);
    }
    state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    state.update();
    ASSERT_TRUE(state.satisfiesBounds(group_));
    auto face_target = faceAlongLookDirection(state);
    const Eigen::Vector3d screen =
      state.getGlobalLinkTransform("monitor_control_frame").translation();
    const Eigen::Vector3d initial_face(face_target.face.x, face_target.face.y, face_target.face.z);
    const Eigen::Vector3d ray = (initial_face - screen).normalized();
    const Eigen::Vector3d face = screen + requested_distance * ray;
    face_target.face.x = face.x();
    face_target.face.y = face.y();
    face_target.face.z = face.z();

    double initial_distance = 0.0;
    double final_distance = 0.0;
    for (int step = 0; step < 1700; ++step) {
      time_sec += dt;
      const auto stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(time_sec * 1.0e9)), RCL_ROS_TIME);
      face_target.header.stamp = stamp;
      face_target.face_stamp = stamp;
      const auto result = update(
        state, face_target, nullptr, nullptr, std::nullopt, kSceneRevision, time_sec);
      ASSERT_TRUE(result.tracking) << "requested_distance=" << requested_distance
                                   << " step=" << step;
      ASSERT_TRUE(result.task) << "requested_distance=" << requested_distance
                               << " step=" << step;
      if (step == 0) {
        initial_distance = (state.getGlobalLinkTransform(
            "monitor_control_frame").translation() - face).norm();
        const bool outside = initial_distance<parameters_.face_distance_min_m ||
            initial_distance> parameters_.face_distance_max_m;
        EXPECT_EQ(result.task->distance_matrix.rows(), outside ? 3 : 0)
          << "Distance remains soft only when outside the exact band";
        EXPECT_EQ(reference_->diagnostics().posture_mode, "recover");
      }
      const double actual_distance = (state.getGlobalLinkTransform(
          "monitor_control_frame").translation() - face).norm();
      final_distance = actual_distance;
      Eigen::VectorXd q, qd, qdd;
      state.copyJointGroupPositions(group_, q);
      state.copyJointGroupVelocities(group_, qd);
      state.copyJointGroupAccelerations(group_, qdd);
      const JointMotionState motion{q, qd, qdd};
      const auto collision = collision_builder.build(
        *scene, state, *group_, motion, limits_, dt);
      const QpResult qp = solver.solve(*result.task, motion, limits_, collision.constraints);
      ASSERT_TRUE(qp.command_available()) << "requested_distance=" << requested_distance
                                          << " step=" << step;
      const Eigen::VectorXd next_q = q + dt * qp.joint_velocity;
      const Eigen::VectorXd next_qdd = (qp.joint_velocity - qd) / dt;
      state.setJointGroupPositions(group_, next_q);
      state.setJointGroupVelocities(group_, qp.joint_velocity);
      state.setJointGroupAccelerations(group_, next_qdd);
      state.update(true);
      ASSERT_TRUE(state.satisfiesBounds(group_)) << "requested_distance=" << requested_distance
                                                 << " step=" << step;
    }
    Eigen::VectorXd final_q;
    state.copyJointGroupPositions(group_, final_q);
    const auto * joint = model_->getJointModel("joint3");
    const auto * revolute = dynamic_cast<const moveit::core::RevoluteJointModel *>(joint);
    const auto * parent = joint->getParentLinkModel();
    const auto * child = joint->getChildLinkModel();
    const Eigen::Vector3d shoulder = state.getGlobalLinkTransform("link2").translation();
    const Eigen::Vector3d elbow = state.getGlobalLinkTransform("link3").translation();
    const Eigen::Vector3d wrist = state.getGlobalLinkTransform("link4").translation();
    const Eigen::Vector3d axis = state.getGlobalLinkTransform(parent).linear() *
      child->getJointOriginTransform().linear() * revolute->getAxis();
    const double bend_sin = std::sin(std::atan2(
      axis.normalized().dot((elbow - shoulder).cross(wrist - elbow)),
          (elbow - shoulder).dot(wrist - elbow)));
    EXPECT_GE(final_q[2], kComfortLower) << "requested_distance=" << requested_distance;
    EXPECT_LE(final_q[2], kComfortUpper) << "requested_distance=" << requested_distance;
    EXPECT_GE(bend_sin, kMinimumBendSin - 1.0e-4)
      << "requested_distance=" << requested_distance;
    summary << "requested=" << requested_distance << " initial=" << initial_distance
            << " final=" << final_distance << " q3=" << final_q[2]
            << " bend_sin=" << bend_sin << '\n';
  }
  RecordProperty("posture1d_bad_start_multiple_distances", summary.str());
  std::cout << "POSTURE-1D BAD-start distance witnesses\n" << summary.str();
}

TEST_F(MotionReferenceTest, FaceRelativeElbowSideUsesActualModelAndPreservesTracking)
{
  const auto signed_clearance = [](const moveit::core::RobotState & robot_state,
    const Eigen::Vector3d & face_point) {
      const Eigen::Vector3d base =
        robot_state.getGlobalLinkTransform("link_base").translation();
      const Eigen::Vector3d e =
        robot_state.getGlobalLinkTransform("link3").translation() - base;
      const Eigen::Vector3d w =
        robot_state.getGlobalLinkTransform("link4").translation() - base;
      const Eigen::Vector3d f = face_point - base;
      const Eigen::Vector3d e_perp = e - w * (e.dot(w) / w.squaredNorm());
      const Eigen::Vector3d f_perp = f - w * (f.dot(w) / w.squaredNorm());
      const double c = e_perp.dot(f_perp);
      return std::pair<double, double>{-c / f_perp.norm(), c};
    };
  const auto make_face_target = [this](const moveit::core::RobotState & robot_state) {
      auto face = faceAlongLookDirection(robot_state);
      const auto * look_link = parameters_.gaze_frame.empty() ? nullptr :
        model_->getLinkModel(parameters_.gaze_frame);
      Eigen::Isometry3d look_pose = look_link ?
        robot_state.getGlobalLinkTransform(look_link) :
        robot_state.getGlobalLinkTransform("monitor_control_frame");
      if (look_link) {
        look_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
      }
      const Eigen::Vector3d face_point = look_pose.translation() +
        0.40 * look_pose.linear().col(0);
      face.face.x = face_point.x();
      face.face.y = face_point.y();
      face.face.z = face_point.z();
      return face;
    };

  const auto configure_qp = [this]() {
      HierarchicalVelocityQpConfig config;
      config.period_sec = parameters_.control_period_sec;
      config.residual_command_latency_sec = 0.0;
      config.velocity_regularization = parameters_.velocity_regularization_weight;
      config.primary_preservation_tolerance = parameters_.primary_orientation_tolerance_radps;
      config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
      config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
      config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
      config.solver_max_iterations = parameters_.solver_max_iterations;
      return config;
    };
  const auto qp_config = configure_qp();
  const HierarchicalVelocityQp solver(qp_config);

  const double side_reserve = parameters_.maximum_linear_reference_mps *
    parameters_.control_period_sec;
  planning_scene_monitor::LockedPlanningSceneRO locked_scene(monitor_);
  ASSERT_TRUE(locked_scene);
  const planning_scene::PlanningSceneConstPtr scene = locked_scene;
  CollisionConstraintBuilder collision_builder(collision_config_);
  moveit::core::RobotState bad_pose(model_);
  ASSERT_TRUE(bad_pose.setToDefaultValues(group_, "rest"));
  // Keep this pose inside the FK-calibrated rest/search bend interval so the
  // face-side preference is active as a secondary preference. Search a small,
  // deterministic set of named joint values for a genuine negative F-side
  // case under the model FK and the actual look-ray target.
  constexpr std::array<double, 3> kJoint1Candidates{0.0, 1.0, -1.0};
  constexpr std::array<double, 5> kJoint2Candidates{-1.5, -1.0, -0.5, 0.0, 0.5};
  constexpr std::array<double, 5> kJoint4Candidates{-1.0, -0.5, 0.0, 0.5, 1.0};
  constexpr std::array<double, 5> kJoint5Candidates{-1.5, -0.75, 0.0, 0.75, 1.5};
  constexpr std::array<double, 7> kJoint6Candidates{-3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0};
  const TrackingVelocityTaskConfig tracking_config;
  Eigen::VectorXd seed_q;
  bool found_negative_face_seed = false;
  for (const double q1 : kJoint1Candidates) {
    for (const double q2 : kJoint2Candidates) {
      for (const double q4 : kJoint4Candidates) {
        for (const double q5 : kJoint5Candidates) {
          for (const double q6 : kJoint6Candidates) {
            moveit::core::RobotState candidate(model_);
            ASSERT_TRUE(candidate.setToDefaultValues(group_, "rest"));
            candidate.setVariablePosition("joint1", q1);
            candidate.setVariablePosition("joint2", q2);
            candidate.setVariablePosition("joint3", -0.9);
            candidate.setVariablePosition("joint4", q4);
            candidate.setVariablePosition("joint5", q5);
            candidate.setVariablePosition("joint6", q6);
            candidate.update();
            if (!candidate.satisfiesBounds(group_)) {
              continue;
            }
            const auto candidate_face = make_face_target(candidate);
            const Eigen::Vector3d candidate_target(
              candidate_face.face.x, candidate_face.face.y, candidate_face.face.z);
            const auto * look_link = parameters_.gaze_frame.empty() ? nullptr :
              model_->getLinkModel(parameters_.gaze_frame);
            Eigen::Isometry3d look_pose = look_link ?
              candidate.getGlobalLinkTransform(look_link) :
              candidate.getGlobalLinkTransform("monitor_control_frame");
            if (look_link) {
              look_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
            }
            const Eigen::Vector3d candidate_ray = candidate_target - look_pose.translation();
            const double candidate_pointing_error = std::acos(std::clamp(
                look_pose.linear().col(0).dot(candidate_ray.normalized()), -1.0, 1.0));
            const auto candidate_edge_task = makeHorizontalEdgePathTask(
              candidate.getGlobalLinkTransform("monitor_control_frame"),
              Eigen::MatrixXd::Identity(6, 6), Eigen::VectorXd::Zero(6), tracking_config);
            const double candidate_clearance =
              signed_clearance(candidate, candidate_target).first;
            const double candidate_distance = (
              candidate.getGlobalLinkTransform("monitor_control_frame").translation() -
              candidate_target).norm();
            if (candidate_clearance >= -side_reserve ||
              candidate_pointing_error > parameters_.pointing_deadband_rad ||
              !candidate_edge_task ||
              std::abs(candidate_edge_task->edge_tilt_error_rad) >
              tracking_config.roll_deadband_rad ||
              candidate_distance < parameters_.face_distance_min_m ||
              candidate_distance > parameters_.face_distance_max_m ||
              scene->isStateColliding(candidate, group_->getName()))
            {
              continue;
            }
            seed_q.resize(6);
            candidate.copyJointGroupPositions(group_, seed_q);
            found_negative_face_seed = true;
            break;
          }
          if (found_negative_face_seed) {break;}
        }
        if (found_negative_face_seed) {break;}
      }
      if (found_negative_face_seed) {break;}
    }
    if (found_negative_face_seed) {break;}
  }
  ASSERT_TRUE(found_negative_face_seed)
    << "No collision-free in-band face target with negative F-side clearance was found";
  for (Eigen::Index index = 0; index < seed_q.size(); ++index) {
    bad_pose.setVariablePosition("joint" + std::to_string(index + 1), seed_q[index]);
  }
  bad_pose.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(seed_q.size()));
  bad_pose.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(seed_q.size()));
  bad_pose.update();
  ASSERT_TRUE(bad_pose.satisfiesBounds(group_));
  EXPECT_FALSE(scene->isStateColliding(bad_pose, group_->getName()));
  Eigen::VectorXd q;
  bad_pose.copyJointGroupPositions(group_, q);
  const JointMotionState motion{q, Eigen::VectorXd::Zero(q.size()),
    Eigen::VectorXd::Zero(q.size())};
  const auto bad_actual = collision_builder.validateActualState(*scene, bad_pose, *group_);
  ASSERT_TRUE(bad_actual.input_valid) << bad_actual.failure_reason;
  ASSERT_FALSE(bad_actual.unsafe) << bad_actual.failure_reason;
  const auto bad_pose_validation = collision_builder.validatePose(*scene, bad_pose, *group_,
        motion);
  ASSERT_TRUE(bad_pose_validation.input_valid) << bad_pose_validation.failure_reason;
  ASSERT_FALSE(bad_pose_validation.unsafe) << bad_pose_validation.failure_reason;
  const auto bad_face = make_face_target(bad_pose);
  const Eigen::Vector3d fixed_face(bad_face.face.x, bad_face.face.y, bad_face.face.z);
  const double initial_clearance = signed_clearance(bad_pose, fixed_face).first;
  const double initial_c = signed_clearance(bad_pose, fixed_face).second;
  ASSERT_LT(initial_clearance, -side_reserve);
  const auto result = update(bad_pose, bad_face);
  ASSERT_TRUE(result.tracking) << reference_->diagnostics().state;
  EXPECT_LE(result.tracking->pointing_error_rad, parameters_.pointing_deadband_rad);
  EXPECT_LE(result.tracking->roll_error_rad, tracking_config.roll_deadband_rad);
  ASSERT_TRUE(result.task);
  ASSERT_FALSE(result.tracking->face_distance_outside_band);
  ASSERT_TRUE(result.tracking->face_distance_interval);
  EXPECT_EQ(result.task->distance_matrix.rows(), 0)
    << "Distance inside its allowed band must not hold the current radius";
  EXPECT_EQ(result.task->secondary_matrix.rows(), 0)
    << "The in-branch main-bend objective is empty";
  ASSERT_GT(result.task->tertiary_matrix.rows(), 0)
    << "Elbow-side preference is below the main bend";
  ASSERT_EQ(result.task->preferred_primary_constraints.size(), 1U)
    << "Only the GOOD J3 braking-aware preservation row is preferred by primary";
  EXPECT_EQ((result.task->preferred_primary_constraints.front().coefficients.array() !=
    0.0).count(), 1);
  EXPECT_DOUBLE_EQ(result.task->preferred_primary_constraints.front().coefficients[2], 1.0);
  const double active_distance = (fixed_face -
    bad_pose.getGlobalLinkTransform("monitor_control_frame").translation()).norm();
  EXPECT_NEAR(result.tracking->face_distance_interval->lower_bound,
    parameters_.position_gain * (parameters_.face_distance_min_m - active_distance), 1.0e-8);
  EXPECT_NEAR(result.tracking->face_distance_interval->upper_bound,
    parameters_.position_gain * (parameters_.face_distance_max_m - active_distance), 1.0e-8);

  auto full_task = *result.task;
  const auto collision = collision_builder.build(
    *scene, bad_pose, *group_, motion, limits_, parameters_.control_period_sec);
  full_task.secondary_constraints.insert(
    full_task.secondary_constraints.end(),
    collision.constraints.begin(), collision.constraints.end());
  const QpResult comfort_qp = solver.solve(full_task, motion, limits_);
  ASSERT_TRUE(comfort_qp.command_available()) << static_cast<int>(comfort_qp.status);
  EXPECT_GT(full_task.tertiary_matrix.row(0).dot(comfort_qp.joint_velocity), 0.0)
    << "q=" << q.transpose() << " s=" << initial_clearance
    << " qdot=" << comfort_qp.joint_velocity.transpose()
    << " comfort_ref=" << result.task->secondary_reference.transpose();
  auto next_pose = bad_pose;
  next_pose.setJointGroupPositions(
    group_, q + parameters_.control_period_sec * comfort_qp.joint_velocity);
  next_pose.update(true);
  const double next_clearance = signed_clearance(next_pose, fixed_face).first;
  EXPECT_GT(next_clearance, initial_clearance);
  EXPECT_GT(initial_c, 0.0);

  const double feasibility_tolerance = qp_config.solution_feasibility_tolerance +
    4.0 * qp_config.solver_absolute_tolerance;
  const auto & posture_bound = full_task.preferred_primary_constraints.front();
  const double joint3_rate = posture_bound.coefficients.dot(comfort_qp.joint_velocity);
  EXPECT_GE(joint3_rate, posture_bound.lower_bound - feasibility_tolerance);
  EXPECT_LE(joint3_rate, posture_bound.upper_bound + feasibility_tolerance);
  ASSERT_EQ(comfort_qp.achieved_primary_velocity.size(),
    result.task->primary_matrix.rows());
  EXPECT_LE((full_task.primary_matrix * comfort_qp.joint_velocity -
    comfort_qp.achieved_primary_velocity).norm(),
    qp_config.primary_preservation_tolerance + feasibility_tolerance);

  auto perturbed_face = bad_face;
  perturbed_face.face.y += 0.005;
  const auto blocked_result = update(bad_pose, perturbed_face);
  ASSERT_TRUE(blocked_result.tracking);
  ASSERT_TRUE(blocked_result.task);
  ASSERT_FALSE(blocked_result.tracking->face_distance_outside_band);
  ASSERT_GT(blocked_result.task->primary_reference.norm(), 1.0e-5);
  auto blocked_task = *blocked_result.task;
  blocked_task.secondary_constraints.insert(
    blocked_task.secondary_constraints.end(),
    collision.constraints.begin(), collision.constraints.end());
  const QpResult blocked_qp = solver.solve(blocked_task, motion, limits_);
  ASSERT_TRUE(blocked_qp.command_available()) << static_cast<int>(blocked_qp.status);
  EXPECT_TRUE(blocked_qp.fully_solved() || blocked_qp.status == QpStatus::kPrimaryOnly ||
    blocked_qp.status == QpStatus::kPartial);
  EXPECT_GT(blocked_qp.achieved_primary_velocity.norm(), 1.0e-5);
  EXPECT_LE((blocked_task.primary_matrix * blocked_qp.joint_velocity -
    blocked_qp.achieved_primary_velocity).norm(),
    qp_config.primary_preservation_tolerance + feasibility_tolerance);

  moveit::core::RobotState good_pose(model_);
  ASSERT_TRUE(good_pose.setToDefaultValues(group_, "rest"));
  good_pose.setVariablePosition("joint1", 0.0);
  good_pose.setVariablePosition("joint2", -1.178097245096172);
  good_pose.setVariablePosition("joint3", -0.9);
  good_pose.setVariablePosition("joint4", 0.0);
  good_pose.setVariablePosition("joint5", 0.0);
  good_pose.setVariablePosition("joint6", 0.0);
  good_pose.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  good_pose.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  good_pose.update();
  ASSERT_TRUE(good_pose.satisfiesBounds(group_));
  EXPECT_FALSE(scene->isStateColliding(good_pose, group_->getName()));
  Eigen::VectorXd good_q;
  good_pose.copyJointGroupPositions(group_, good_q);
  const JointMotionState good_motion{
    good_q, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto good_actual = collision_builder.validateActualState(*scene, good_pose, *group_);
  ASSERT_TRUE(good_actual.input_valid) << good_actual.failure_reason;
  ASSERT_FALSE(good_actual.unsafe) << good_actual.failure_reason;
  const auto good_pose_validation = collision_builder.validatePose(
    *scene, good_pose, *group_, good_motion);
  ASSERT_TRUE(good_pose_validation.input_valid) << good_pose_validation.failure_reason;
  ASSERT_FALSE(good_pose_validation.unsafe) << good_pose_validation.failure_reason;
  const auto good_face = make_face_target(good_pose);
  const Eigen::Vector3d good_target(good_face.face.x, good_face.face.y, good_face.face.z);
  ASSERT_GT(signed_clearance(good_pose, good_target).first, side_reserve);
  // Use a fresh reference so this one stationary face sample has zero radial
  // feed-forward when checking the idle-only allowed envelope.
  reference_.reset();
  reference_ = std::make_unique<MotionReference>(
    node_, monitor_, parameters_, limits_, collision_config_);
  const auto good_result = update(good_pose, good_face);
  ASSERT_TRUE(good_result.tracking);
  ASSERT_TRUE(good_result.task);
  ASSERT_FALSE(good_result.tracking->face_distance_outside_band);
  ASSERT_EQ(good_result.task->preferred_primary_constraints.size(), 1U)
    << "A GOOD J3 pose keeps its braking-aware comfort bound at every distance";
  EXPECT_EQ(good_result.task->distance_matrix.rows(), 0)
    << "Inside the band the final distance level is empty, not a radial hold";
  EXPECT_EQ(good_result.task->secondary_matrix.rows(), 0);
  EXPECT_EQ(good_result.task->tertiary_matrix.rows(), 0);
  EXPECT_DOUBLE_EQ(good_result.task->preferred_primary_constraints.front().coefficients[2], 1.0);
  EXPECT_GE(good_result.task->preferred_primary_constraints.front().lower_bound,
        -limits_.max_velocity[2]);
  EXPECT_LE(good_result.task->preferred_primary_constraints.front().upper_bound,
        limits_.max_velocity[2]);
  JointMotionState different_queue_state = good_motion;
  different_queue_state.velocity[2] = 0.2;
  different_queue_state.acceleration[2] = 0.1;
  const auto same_control_state = update(
    good_pose, good_face, nullptr, nullptr, different_queue_state);
  ASSERT_TRUE(same_control_state.task);
  ASSERT_EQ(same_control_state.task->preferred_primary_constraints.size(), 1U);
  EXPECT_DOUBLE_EQ(
    same_control_state.task->preferred_primary_constraints.front().lower_bound,
    good_result.task->preferred_primary_constraints.front().lower_bound);
  EXPECT_DOUBLE_EQ(
    same_control_state.task->preferred_primary_constraints.front().upper_bound,
    good_result.task->preferred_primary_constraints.front().upper_bound)
    << "Comfort bounds use the state shared with QP linearization, not a separate queue sample";
  moveit::core::RobotState position_only(model_);
  ASSERT_TRUE(position_only.setToDefaultValues(group_, "rest"));
  position_only.update();
  ASSERT_FALSE(position_only.hasVelocities());
  ASSERT_FALSE(position_only.hasAccelerations());
  const auto position_only_result = update(position_only, good_face);
  ASSERT_TRUE(position_only_result.task);
  EXPECT_TRUE(position_only_result.task->preferred_primary_constraints.empty())
    << "A position-only RobotState must not fabricate braking bounds from absent motion data";
  auto full_good_task = *good_result.task;
  const auto good_collision = collision_builder.build(
    *scene, good_pose, *group_, good_motion, limits_, parameters_.control_period_sec);
  full_good_task.secondary_constraints.insert(
    full_good_task.secondary_constraints.end(),
    good_collision.constraints.begin(), good_collision.constraints.end());
  const QpResult settled_qp = solver.solve(full_good_task, good_motion, limits_);
  ASSERT_TRUE(settled_qp.command_available()) << static_cast<int>(settled_qp.status);
  EXPECT_LT(settled_qp.joint_velocity.norm(), 1.0e-5);
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
  const double committed_radius_cap =
    reference_->diagnostics().compact_entry_radius_cap_m;
  for (std::size_t index = 0U; index < tail.points.size(); ++index) {
    std::string violation;
    EXPECT_TRUE(reference_->validateCompactPathPosition(
        start, tail.points[index].position, &violation))
      << "braking tail point " << index << " violates " << violation;
    auto tail_state = start;
    tail_state.setJointGroupPositions(group_, tail.points[index].position);
    tail_state.update(true);
    const double tail_radius = geometry.maximumAssemblyRadius(tail_state);
    EXPECT_LE(tail_radius, committed_radius_cap + 1.0e-6)
      << "braking tail point " << index << ", radius=" << tail_radius
      << ", committed cap=" << committed_radius_cap;
  }

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

TEST_F(MotionReferenceTest, CameraHoldPreservesConsumedLocalSearchTime)
{
  auto robot_state = state(.7);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "local_then_sweep";
  ASSERT_EQ(update(robot_state, search, nullptr, nullptr, std::nullopt,
      kSceneRevision, 1.0).task.has_value(), true);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_LOCAL");

  // Five seconds of local search elapse before the camera watchdog's HOLD.
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 6.0);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_LOCAL");
  auto camera_hold = target(msg::TrackingTarget::HOLD);
  // The controller publishes unavailable camera HOLD with no active target.
  // This sentinel is availability state, not a new search request.
  camera_hold.target_id = -1;
  camera_hold.allow_recovery = false;
  EXPECT_EQ(camera_hold.target_id, -1);
  EXPECT_FALSE(camera_hold.allow_recovery);
  EXPECT_FALSE(update(robot_state, camera_hold, nullptr, nullptr, std::nullopt,
      kSceneRevision, 6.01).task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");

  // Resume one second after the pause. Only one active second remained in the
  // six-second local phase, so this call must advance to preparation.
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 20.0);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 21.02);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_PLANNING");
}

TEST_F(MotionReferenceTest, RepeatedCameraHoldsPreserveSweepGoalAndDirection)
{
  auto robot_state = state();
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 113.0);
  const Eigen::VectorXd original_goal = MotionReferenceTestPeer::searchGoal(*reference_);
  ASSERT_EQ(original_goal.size(), 6);

  auto camera_hold = target(msg::TrackingTarget::HOLD);
  camera_hold.target_id = -1;
  camera_hold.allow_recovery = false;
  EXPECT_EQ(camera_hold.target_id, -1);
  EXPECT_FALSE(camera_hold.allow_recovery);
  EXPECT_FALSE(update(robot_state, camera_hold, nullptr, nullptr, std::nullopt,
      kSceneRevision, 113.683).task);
  robot_state = state(.8);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 114.0);
  EXPECT_TRUE(MotionReferenceTestPeer::searchGoal(*reference_).isApprox(original_goal, 0.001));

  EXPECT_FALSE(update(robot_state, camera_hold, nullptr, nullptr, std::nullopt,
      kSceneRevision, 114.333).task);
  robot_state = state(.9);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 120.0);
  EXPECT_TRUE(MotionReferenceTestPeer::searchGoal(*reference_).isApprox(original_goal, 0.001));

  EXPECT_FALSE(update(robot_state, camera_hold, nullptr, nullptr, std::nullopt,
      kSceneRevision, 121.531).task);
  robot_state = state(1.0);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 122.0);
  EXPECT_TRUE(MotionReferenceTestPeer::searchGoal(*reference_).isApprox(original_goal, 0.001));
}

TEST_F(MotionReferenceTest, SearchRestSearchStartsANewLocalPhase)
{
  auto robot_state = state(.3);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "local_then_sweep";
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 1.0);
  (void)update(robot_state, target(msg::TrackingTarget::REST), nullptr, nullptr,
    std::nullopt, kSceneRevision, 2.0);
  EXPECT_EQ(reference_->diagnostics().state, "REST_PLANNING");

  robot_state = state(.8);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 3.0);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_LOCAL");
  EXPECT_NEAR(MotionReferenceTestPeer::searchGoal(*reference_)[0], 1.15, 0.001);
}

TEST_F(MotionReferenceTest, SearchClockResetStartsANewSector)
{
  auto robot_state = state(.3);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 10.0);
  const Eigen::VectorXd old_goal = MotionReferenceTestPeer::searchGoal(*reference_);

  robot_state = state(.8);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 0.5);
  const Eigen::VectorXd new_goal = MotionReferenceTestPeer::searchGoal(*reference_);
  ASSERT_EQ(new_goal.size(), old_goal.size());
  EXPECT_GT(std::abs(new_goal[0] - old_goal[0]), 0.1);
}

TEST_F(MotionReferenceTest, SearchArrivalDoesNotRequestAnotherPlanWhileSettling)
{
  auto robot_state = state();
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "local_then_sweep";
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 1.0);
  (void)update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 8.0);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_PLANNING");

  Eigen::VectorXd position = MotionReferenceTestPeer::searchGoal(*reference_);
  position[0] -= 0.03;
  robot_state.setJointGroupPositions(group_, position);
  robot_state.update();
  std::vector<Eigen::VectorXd> zero_distance_path{position, position};
  ASSERT_TRUE(MotionReferenceTestPeer::installSearchFollowerPathForTest(
      *reference_, robot_state, zero_distance_path, 9.0));
  const auto planner_deadline = std::chrono::steady_clock::now() + 500ms;
  while (MotionReferenceTestPeer::plannerBusy(*reference_) &&
    std::chrono::steady_clock::now() < planner_deadline)
  {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_FALSE(MotionReferenceTestPeer::plannerBusy(*reference_));
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  velocity[0] = 0.02;
  robot_state.setJointGroupVelocities(group_, velocity);
  robot_state.update();
  const auto before = reference_->diagnostics().plans_requested;
  const auto settling = update(robot_state, search, nullptr, nullptr, std::nullopt,
    kSceneRevision, 10.0);
  EXPECT_TRUE(settling.task);
  EXPECT_TRUE(settling.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_SETTLING");
  EXPECT_EQ(reference_->diagnostics().paths_completed, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_requested, before);
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
