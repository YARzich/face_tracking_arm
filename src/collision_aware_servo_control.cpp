// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "face_tracking_arm/detail/scalar_braking_policy.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr double kStoppedAccelerationRadps2 = 2.0e-2;

constexpr std::chrono::milliseconds kTelemetryPublishPeriod{100};

bool same_numerical_policy(
  const std::optional<control::NumericalPolicy> & first,
  const std::optional<control::NumericalPolicy> & second) noexcept
{
  if (first.has_value() != second.has_value()) {
    return false;
  }
  if (!first.has_value()) {
    return true;
  }
  return first->position_error.value == second->position_error.value &&
         first->velocity_error.value == second->velocity_error.value &&
         first->acceleration_error.value == second->acceleration_error.value &&
         first->jerk_error.value == second->jerk_error.value &&
         first->solver_residual.value == second->solver_residual.value;
}

bool same_dynamic_limits(
  const control::JointMotionLimits & first,
  const control::JointMotionLimits & second) noexcept
{
  return first.max_velocity.size() == second.max_velocity.size() &&
         first.max_acceleration.size() == second.max_acceleration.size() &&
         first.max_jerk.size() == second.max_jerk.size() &&
         (first.max_velocity.array() == second.max_velocity.array()).all() &&
         (first.max_acceleration.array() == second.max_acceleration.array()).all() &&
         (first.max_jerk.array() == second.max_jerk.array()).all() &&
         same_numerical_policy(first.numerical_policy, second.numerical_policy);
}

struct VelocityConstraintViolation
{
  std::size_t row{0U};
  double projected_velocity{0.0};
  double lower_bound{0.0};
  double upper_bound{0.0};
};

bool satisfies_velocity_constraints(
  const Eigen::VectorXd & velocity,
  const std::vector<control::LinearVelocityConstraint> & constraints,
  const double tolerance,
  const std::size_t first_constraint,
  std::optional<VelocityConstraintViolation> * const violation = nullptr) noexcept
{
  if (violation != nullptr) {
    violation->reset();
  }
  if (!velocity.allFinite() || !std::isfinite(tolerance) || tolerance < 0.0 ||
    first_constraint > constraints.size())
  {
    return false;
  }
  for (std::size_t index = 0U; index < constraints.size(); ++index) {
    const auto & constraint = constraints[index];
    if (constraint.coefficients.size() != velocity.size() ||
      !constraint.coefficients.allFinite() || std::isnan(constraint.lower_bound) ||
      std::isnan(constraint.upper_bound) || constraint.lower_bound > constraint.upper_bound)
    {
      return false;
    }
    if (index < first_constraint) {
      continue;
    }
    const double value = constraint.coefficients.dot(velocity);
    if (!std::isfinite(value) || value < constraint.lower_bound - tolerance ||
      value > constraint.upper_bound + tolerance)
    {
      if (violation != nullptr) {
        *violation = VelocityConstraintViolation{
          index, value, constraint.lower_bound, constraint.upper_bound};
      }
      return false;
    }
  }
  return true;
}

class CompactProfilePublicationOutcome
{
public:
  CompactProfilePublicationOutcome(
    control::MotionReference * const reference, const bool attempted) noexcept
  : reference_(attempted ? reference : nullptr) {}

  ~CompactProfilePublicationOutcome()
  {
    if (reference_ != nullptr) {
      reference_->recordCompactProfileCommandPublished(published_);
    }
  }

  void markPublished(const bool published) noexcept
  {
    published_ = published;
  }

  CompactProfilePublicationOutcome(const CompactProfilePublicationOutcome &) = delete;
  CompactProfilePublicationOutcome & operator=(
    const CompactProfilePublicationOutcome &) = delete;

private:
  control::MotionReference * reference_{nullptr};
  bool published_{false};
};

}  // namespace

std::optional<control::QpResult> CollisionAwareServoComponent::reference_guard_blend(
  const control::QpResult & tracking_result,
  const control::QpResult & brake_result,
  const Eigen::MatrixXd & primary_matrix,
  const std::vector<control::LinearVelocityConstraint> & constraints,
  const double tracking_weight) const
{
  if (!tracking_result.command_available() || !brake_result.command_available() ||
    tracking_result.joint_velocity.size() == 0 ||
    tracking_result.joint_velocity.size() != brake_result.joint_velocity.size() ||
    !tracking_result.joint_velocity.allFinite() ||
    !brake_result.joint_velocity.allFinite() ||
    !std::isfinite(tracking_weight) || tracking_weight <= 0.0 || tracking_weight >= 1.0 ||
    primary_matrix.cols() != tracking_result.joint_velocity.size() ||
    !primary_matrix.allFinite())
  {
    return std::nullopt;
  }
  if (!velocity_qp_) {
    return std::nullopt;
  }
  const double feasibility_tolerance =
    velocity_qp_->config().solution_feasibility_tolerance;

  control::QpResult blended = tracking_result;
  blended.joint_velocity = brake_result.joint_velocity + tracking_weight *
    (tracking_result.joint_velocity - brake_result.joint_velocity);
  if (!blended.joint_velocity.allFinite()) {
    return std::nullopt;
  }
  for (const auto & constraint : constraints) {
    if (constraint.coefficients.size() != blended.joint_velocity.size() ||
      !constraint.coefficients.allFinite() || std::isnan(constraint.lower_bound) ||
      std::isnan(constraint.upper_bound) || constraint.lower_bound > constraint.upper_bound)
    {
      return std::nullopt;
    }
    const double value = constraint.coefficients.dot(blended.joint_velocity);
    if (!std::isfinite(value) || value < constraint.lower_bound ||
      value > constraint.upper_bound)
    {
      const double violation = value < constraint.lower_bound ?
        constraint.lower_bound - value : value - constraint.upper_bound;
      if (!std::isfinite(violation) || violation > feasibility_tolerance) {
        return std::nullopt;
      }
    }
  }
  blended.achieved_primary_velocity = primary_matrix * blended.joint_velocity;
  if (!blended.achieved_primary_velocity.allFinite()) {
    return std::nullopt;
  }
  return blended;
}

bool CollisionAwareServoComponent::try_reference_guard_blend(
  const moveit::core::RobotStatePtr & robot_state,
  const moveit_servo::KinematicState & current_state,
  const control::JointMotionState & motion_state,
  const planning_scene::PlanningScene & planning_scene,
  const bool segment_validation_required,
  const control::QpResult & tracking_result,
  const control::QpResult & brake_result,
  const Eigen::MatrixXd & primary_matrix,
  const std::vector<control::LinearVelocityConstraint> & constraints,
  const control::CollisionConstraintDiagnostics & collision_diagnostics,
  std::optional<ValidatedCommand> & accepted_command,
  control::QpResult & accepted_qp_result,
  control::CollisionValidationBudget * const budget)
{
  for (const double tracking_weight : {0.5, 0.25}) {
    if (budget != nullptr && budget->expired()) {
      return false;
    }
    auto blended_result = reference_guard_blend(
      tracking_result, brake_result, primary_matrix, constraints, tracking_weight);
    if (!blended_result) {
      continue;
    }
    std::string blend_failure;
    CandidateRejectionKind blend_rejection_kind = CandidateRejectionKind::kOther;
    auto blended_command = validated_candidate(
      robot_state, current_state, motion_state, blended_result->joint_velocity,
      planning_scene, segment_validation_required, CommandAttempt::kTracking,
      blend_failure, collision_diagnostics, blend_rejection_kind, budget);
    if (blended_command) {
      accepted_command = std::move(blended_command);
      accepted_qp_result = std::move(*blended_result);
      return true;
    }
  }
  return false;
}

bool CollisionAwareServoComponent::try_reference_guard_recovery(
  const moveit::core::RobotStatePtr & robot_state,
  const moveit_servo::KinematicState & current_state,
  const control::JointMotionState & motion_state,
  const planning_scene::PlanningScene & planning_scene,
  const bool segment_validation_required,
  const control::HierarchicalVelocityTask & face_task,
  const bool has_face_primary_roll,
  const std::vector<control::LinearVelocityConstraint> & constraints,
  const control::CollisionConstraintDiagnostics & collision_diagnostics,
  std::optional<ValidatedCommand> & accepted_command,
  control::QpResult & accepted_qp_result,
  control::CollisionValidationBudget * const budget)
{
  const auto recovery_task = control::makeFaceEdgeOnlyRecoveryTask(
    face_task, has_face_primary_roll);
  if (!recovery_task) {
    return false;
  }
  if (budget != nullptr && budget->expired()) {
    return false;
  }
  control::QpResult recovery_result =
    solve_with_telemetry(*recovery_task, motion_state, constraints);
  if (budget != nullptr && budget->expired()) {
    return false;
  }
  if (!recovery_result.command_available()) {
    log_qp_failure(
      CommandAttempt::kTracking, recovery_result, motion_state, constraints,
      collision_diagnostics);
    return false;
  }

  std::string recovery_failure;
  CandidateRejectionKind recovery_rejection_kind = CandidateRejectionKind::kOther;
  auto recovery_command = validated_candidate(
    robot_state, current_state, motion_state, recovery_result.joint_velocity,
    planning_scene, segment_validation_required, CommandAttempt::kTracking,
    recovery_failure, collision_diagnostics, recovery_rejection_kind, budget);
  if (!recovery_command) {
    return false;
  }
  accepted_command = std::move(recovery_command);
  accepted_qp_result = std::move(recovery_result);
  return true;
}

void CollisionAwareServoComponent::update_execution_target_context(
  const std::optional<msg::TrackingTarget> & target)
{
  if (!target.has_value()) {
    return;
  }
  current_tick_context_.target_available = true;
  current_tick_context_.target_mode = target->mode;
  current_tick_context_.target_id = target->target_id;
  current_tick_context_.target_source_stamp_ns =
    static_cast<std::int64_t>(target->header.stamp.sec) * 1'000'000'000LL +
    target->header.stamp.nanosec;
  current_tick_context_.face_stamp_ns =
    static_cast<std::int64_t>(target->face_stamp.sec) * 1'000'000'000LL +
    target->face_stamp.nanosec;
  current_tick_context_.target_position[0] = target->pose.position.x;
  current_tick_context_.target_position[1] = target->pose.position.y;
  current_tick_context_.target_position[2] = target->pose.position.z;
  current_tick_context_.face_position[0] = target->face.x;
  current_tick_context_.face_position[1] = target->face.y;
  current_tick_context_.face_position[2] = target->face.z;
  const std::size_t frame_size = std::min(
    target->header.frame_id.size(), sizeof(current_tick_context_.target_frame) - 1U);
  std::copy_n(target->header.frame_id.data(), frame_size,
    current_tick_context_.target_frame);
  current_tick_context_.target_frame[frame_size] = '\0';
}

void CollisionAwareServoComponent::publish_control_status(
  const bool collision_avoidance_active,
  const bool reference_guard_blend_accepted,
  const bool reference_guard_recovery_accepted,
  const std::optional<std::string> & mitigated_failure,
  const control::QpResult & accepted_qp_result,
  const std::int8_t mitigated_status)
{
  if (mitigated_failure.has_value()) {
    const std::string status_message =
      reference_guard_blend_accepted || reference_guard_recovery_accepted ?
      *mitigated_failure : *mitigated_failure + "; safe jerk-limited braking command published";
    publish_status(mitigated_status, status_message);
  } else if (collision_avoidance_active) {
    publish_status(
      moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
      accepted_qp_result.fully_solved() ? "Collision avoidance constraint active" :
      "Collision avoidance constraint active; lower-priority objective not completed");
  } else {
    publish_status(
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      accepted_qp_result.fully_solved() ? "No warnings" :
      "Safe higher-priority tracking command; lower-priority objective not completed");
  }
}

void CollisionAwareServoComponent::update_pause_after_publication()
{
  const bool terminal_pause = pause_requested_.load(std::memory_order_acquire) &&
    commands_.queue.back().velocities.cwiseAbs().maxCoeff() <= kStoppedVelocityRadps &&
    commands_.queue.back().accelerations.cwiseAbs().maxCoeff() <=
    kStoppedAccelerationRadps2;
  if (!terminal_pause) {
    pause_terminal_pending_.store(false, std::memory_order_release);
    return;
  }
  if (!pause_terminal_pending_.exchange(true, std::memory_order_acq_rel)) {
    // composeTrajectoryMessage withholds queue.back(); one more stopped point
    // makes the first terminal point visible to JTC.
    return;
  }
  paused_.store(true, std::memory_order_release);
  commands_.discardBranch();
  commands_.rearm_pending = false;
  commands_.published_tail_rearm_pending = false;
  commands_.restartSettling();
  commands_.clearRearmDeadlines();
  pause_terminal_pending_.store(false, std::memory_order_release);
}

void CollisionAwareServoComponent::control_loop()
{
  if (!realtime_tools::configure_sched_fifo(
      static_cast<int>(servo_parameters_.thread_priority)))
  {
    RCLCPP_WARN(
      get_logger(),
      "Could not enable SCHED_FIFO priority %d; continuing with normal scheduling",
      static_cast<int>(servo_parameters_.thread_priority));
  }
  rclcpp::WallRate rate(1.0 / parameters_.control_period_sec);
  auto next_telemetry_publish = std::chrono::steady_clock::now() +
    kTelemetryPublishPeriod;
  while (rclcpp::ok() && !stop_control_.load(std::memory_order_acquire)) {
    const auto tick_start = std::chrono::steady_clock::now();
    try {
      control_tick();
    } catch (const std::exception & error) {
      if (rclcpp::ok()) {
        if (require_hardware_io_status_ && !commands_.published_tail_rearm_pending) {
          request_hardware_inhibit();
        }
        enter_safety_wait(
          std::string("Collision-aware control exception: ") + error.what());
      }
    }
    const auto tick_finished = std::chrono::steady_clock::now();
    const double tick_duration_ms = std::chrono::duration<double, std::milli>(
      tick_finished - tick_start).count();
    (void)telemetry_.control_tick_statistics.add(tick_duration_ms);
    telemetry_.lifetime_maximum_control_tick_ms = std::max(
      telemetry_.lifetime_maximum_control_tick_ms, tick_duration_ms);
    telemetry_.observeMode(controller_mode_);

    if (tick_finished >= next_telemetry_publish && rclcpp::ok() &&
      !stop_control_.load(std::memory_order_acquire))
    {
      publish_runtime_diagnostics();
      do {
        next_telemetry_publish += kTelemetryPublishPeriod;
      } while (next_telemetry_publish <= tick_finished);
    }

    const auto cycle_duration = std::chrono::steady_clock::now() - tick_start;
    const double cycle_duration_ms = std::chrono::duration<double, std::milli>(
      cycle_duration).count();
    current_tick_context_.cycle_duration_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
      cycle_duration).count());
    (void)telemetry_.control_cycle_statistics.add(cycle_duration_ms);
    if (cycle_duration_ms > parameters_.control_period_sec * 1000.0) {
      ++telemetry_.missed_control_deadlines;
    }
    record_execution_context();

    // WallRate reaches into the ROS context. Do not enter it after shutdown;
    // also handle the narrow race where the context becomes invalid after this check.
    if (!rclcpp::ok() || stop_control_.load(std::memory_order_acquire)) {
      break;
    }
    try {
      rate.sleep();
    } catch (const std::exception & error) {
      if (rclcpp::ok() && !stop_control_.load(std::memory_order_acquire)) {
        RCLCPP_ERROR(get_logger(), "Control rate sleep failed: %s", error.what());
      }
      break;
    }
  }
}

bool CollisionAwareServoComponent::validateHardwareIoForControlTick()
{
  if (!require_hardware_io_status_) {
    return true;
  }
  cycle_hardware_io_status_ = *hardware_io_status_buffer_.readFromRT();
  current_tick_context_.hardware_epoch = cycle_hardware_io_status_->sample.epoch;
  current_tick_context_.hardware_read_sequence =
    cycle_hardware_io_status_->sample.read_sequence;
  const auto hardware_status = current_hardware_io_status();
  if (!hardware_status || !hardware_status->valid) {
    const auto failure = hardware_status ? hardware_status->failure :
      hardware::HardwareIoStatusFailure::kInvalidEpochOrSequence;
    const bool awaiting_first_active_sample =
      !hardware_active_seen_.load(std::memory_order_acquire) &&
      (failure == hardware::HardwareIoStatusFailure::kHardwareInactive ||
      failure == hardware::HardwareIoStatusFailure::kInvalidEpochOrSequence);
    if (awaiting_first_active_sample) {
      if (!hardware_io_halt_failure_.has_value() ||
        *hardware_io_halt_failure_ != failure)
      {
        hardware_io_halt_failure_ = failure;
        enter_startup_retry("Waiting for the first active hardware provenance snapshot");
      }
      return false;
    }
    if (hardware_status &&
      (failure == hardware::HardwareIoStatusFailure::kJointBoundaryUnresolved ||
      failure == hardware::HardwareIoStatusFailure::kJointOutsideLimit))
    {
      hardware_io_failure_joint_index_ = hardware_status->failure_joint_index;
    }
    if (!hardware_io_halt_failure_.has_value() || *hardware_io_halt_failure_ != failure) {
      hardware_io_halt_failure_ = failure;
      request_hardware_inhibit();
      enter_safety_wait("Hardware I/O provenance failed validation");
    }
    return false;
  }
  if (hardware_inhibit_pending_.load(std::memory_order_acquire) ||
    hardware_io_inhibit_requested_.load(std::memory_order_acquire))
  {
    return false;
  }
  hardware_io_halt_failure_.reset();
  hardware_io_failure_joint_index_.reset();
  return true;
}

bool CollisionAwareServoComponent::build_and_record_collision_context(
  const planning_scene::PlanningScene & planning_scene,
  const moveit::core::RobotState & robot_state,
  const control::JointMotionState & motion_state,
  const control::JointMotionLimits & limits,
  control::CollisionConstraintResult & result)
{
  if (!limits.numerical_policy.has_value()) {
    RCLCPP_ERROR(get_logger(), "Active motion limits have no collision numerical policy");
    enter_safety_wait("Active motion limits have no collision numerical policy");
    return false;
  }
  if (!same_numerical_policy(
      collision_constraint_config_.numerical_policy, limits.numerical_policy))
  {
    compact_brake_line_proof_.clear();
    compact_brake_line_validator_.reset();
    ++collision_config_revision_;
    control::CollisionConstraintConfig selected_config = collision_constraint_config_;
    selected_config.numerical_policy = limits.numerical_policy;
    collision_constraint_builder_ =
      std::make_unique<control::CollisionConstraintBuilder>(std::move(selected_config));
    collision_constraint_config_.numerical_policy = limits.numerical_policy;
  }
  const auto collision_build_start = std::chrono::steady_clock::now();
  result = collision_constraint_builder_->build(
    planning_scene, robot_state, *joint_model_group_, motion_state, limits,
    parameters_.control_period_sec);
  (void)telemetry_.collision_build_statistics.add(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - collision_build_start).count());
  (void)telemetry_.collision_self_query_statistics.add(result.diagnostics.self_query_ms);
  (void)telemetry_.collision_world_query_statistics.add(result.diagnostics.world_query_ms);
  telemetry_.last_minimum_self_distance_m = result.diagnostics.minimum_self_distance_m;
  telemetry_.last_minimum_world_distance_m = result.diagnostics.minimum_world_distance_m;
  telemetry_.observeClearances(
    result.diagnostics.minimum_self_distance_m, result.diagnostics.minimum_world_distance_m);
  for (const auto & pair : result.diagnostics.pairs) {
    const double headroom = pair.distance_m - pair.required_clearance_m;
    const std::string pair_name = pair.first_body + " <-> " + pair.second_body;
    if (std::isfinite(headroom) &&
      (!telemetry_.last_minimum_robust_headroom_m.has_value() ||
      headroom < *telemetry_.last_minimum_robust_headroom_m))
    {
      telemetry_.last_minimum_robust_headroom_m = headroom;
      telemetry_.last_minimum_robust_headroom_pair = pair_name;
    }
    if (std::isfinite(headroom) &&
      (!telemetry_.minimum_observed_robust_headroom_m.has_value() ||
      headroom < *telemetry_.minimum_observed_robust_headroom_m))
    {
      telemetry_.minimum_observed_robust_headroom_m = headroom;
      telemetry_.minimum_observed_robust_headroom_pair = pair_name;
    }
  }
  telemetry_.last_closest_pair = result.diagnostics.closest_pair;
  telemetry_.last_active_collision_rows =
    result.diagnostics.active_distance_constraints;
  current_tick_context_.active_collision_rows = static_cast<std::uint32_t>(
    result.diagnostics.active_distance_constraints);
  current_tick_context_.collision_distance_available = std::isfinite(
    result.diagnostics.minimum_world_distance_m);
  if (current_tick_context_.collision_distance_available) {
    current_tick_context_.minimum_world_distance_m = result.diagnostics.minimum_world_distance_m;
  }
  const std::size_t closest_length = std::min(
    result.diagnostics.closest_pair.size(), sizeof(current_tick_context_.closest_pair) - 1U);
  std::copy_n(
    result.diagnostics.closest_pair.data(), closest_length, current_tick_context_.closest_pair);
  current_tick_context_.closest_pair[closest_length] = '\0';
  for (const auto & pair : result.diagnostics.pairs) {
    if (pair.kind != control::CollisionPairKind::kWorld || !pair.constraint_added ||
      !std::isfinite(pair.distance_m))
    {
      continue;
    }
    if (!current_tick_context_.active_world_pair_available ||
      pair.distance_m < current_tick_context_.active_world_pair_distance_m)
    {
      current_tick_context_.active_world_pair_available = true;
      current_tick_context_.active_world_pair_distance_m = pair.distance_m;
      constexpr std::size_t capacity = sizeof(current_tick_context_.active_world_pair);
      std::size_t offset = 0U;
      const std::size_t first_length = std::min(pair.first_body.size(), capacity - 1U);
      std::copy_n(
        pair.first_body.data(), first_length, current_tick_context_.active_world_pair);
      offset += first_length;
      for (const char character : std::array<char, 5>{' ', '<', '-', '>', ' '}) {
        if (offset >= capacity - 1U) {
          break;
        }
        current_tick_context_.active_world_pair[offset++] = character;
      }
      const std::size_t second_length = std::min(
        pair.second_body.size(), capacity - 1U - offset);
      std::copy_n(
        pair.second_body.data(), second_length,
        current_tick_context_.active_world_pair + offset);
      offset += second_length;
      current_tick_context_.active_world_pair[offset] = '\0';
    }
  }
  telemetry_.last_outside_collision_viability_contacts =
    result.diagnostics.outside_viability_contacts;
  if (!result.diagnostics.valid) {
    RCLCPP_ERROR(
      get_logger(),
      "Collision constraint build failed: reason='%s', min_distance=%.6f m, pair='%s'",
      result.diagnostics.failure_reason.c_str(), result.diagnostics.minimum_distance_m,
      result.diagnostics.closest_pair.c_str());
    enter_safety_wait(
      "Collision constraints unavailable: " + result.diagnostics.failure_reason);
    return false;
  }
  return true;
}

const control::JointMotionLimits & CollisionAwareServoComponent::active_motion_limits()
const noexcept
{
  return motion_reference_ ? motion_reference_->activeCommandLimits() : motion_limits_;
}

bool CollisionAwareServoComponent::recover_command_timeline(
  const rclcpp::Time & current_time, const std::uint64_t skipped_periods)
{
  // Actual recovery retains max_expected_latency. Only an active timed compact
  // profile gets a hysteresis trigger at send minimum plus one full calculation
  // period; ordinary control keeps the existing recovery decision unchanged.
  const std::int64_t retry_target_lead_ns = seconds_to_nanoseconds(
    servo_parameters_.max_expected_latency);
  std::optional<std::int64_t> recovery_suppression_lead_ns;
  if (motion_reference_ && motion_reference_->hasActiveCompactProfile()) {
    const std::int64_t residual_latency_ns = seconds_to_nanoseconds(
      parameters_.residual_command_latency_sec);
    const std::int64_t control_period_ns = seconds_to_nanoseconds(
      parameters_.control_period_sec);
    recovery_suppression_lead_ns = residual_latency_ns + control_period_ns + control_period_ns;
  }
  const auto timeline_recovery_periods = commands_.timelineRecoveryPeriods(
    skipped_periods, current_time.nanoseconds(), retry_target_lead_ns,
    recovery_suppression_lead_ns);
  if (!timeline_recovery_periods.has_value()) {
    enter_controlled_rearm("Deferred command deadline has no active command-grid anchor");
    return false;
  }
  std::string timeline_recovery_failure;
  TimelineRecoveryStatus timeline_recovery_status = commands_.recoverTimeline(
    *timeline_recovery_periods, current_time, timeline_recovery_failure,
    servo_parameters_.max_expected_latency, parameters_.joint_names.size());
  if (timeline_recovery_status == TimelineRecoveryStatus::kExhaustedPublishedSuffix) {
    const auto target = current_target(current_time);
    const auto expected_publication_id = commands_.active_publication_id;
    const bool expected_epoch_active = commands_.epoch_active;
    const bool expected_epoch_published = commands_.epoch_has_published;
    const std::uint64_t expected_scene_revision =
      planning_scene_collision_revision_.load(std::memory_order_acquire);
    if (target && target->mode == msg::TrackingTarget::FACE &&
      expected_publication_id.has_value() && expected_epoch_active &&
      expected_epoch_published && expected_scene_revision == last_validated_scene_revision_)
    {
      std::string terminal_hold_reason;
      const RearmReadiness readiness = safe_rearm_ready(
        current_time, terminal_hold_reason);
      const std::uint64_t current_scene_revision =
        planning_scene_collision_revision_.load(std::memory_order_acquire);
      const bool identity_unchanged = commands_.epoch_active &&
        commands_.epoch_has_published &&
        commands_.active_publication_id == expected_publication_id;
      if (current_scene_revision != expected_scene_revision ||
        current_scene_revision != last_validated_scene_revision_ || !identity_unchanged)
      {
        ++telemetry_.planning_scene_invalidation_count;
        if (require_hardware_io_status_) {
          request_hardware_inhibit();
        }
        enter_safety_wait(
          "Terminal hold recovery lost its published command or validated collision scene");
        return false;
      }
      if (readiness == RearmReadiness::kUnsafe) {
        if (require_hardware_io_status_) {
          request_hardware_inhibit();
        }
        enter_safety_wait(std::move(terminal_hold_reason));
        return false;
      }
      if (readiness == RearmReadiness::kWaiting) {
        publish_status(
          moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
          std::move(terminal_hold_reason));
        return false;
      }

      timeline_recovery_failure.clear();
      timeline_recovery_status = commands_.resumeVerifiedTerminalHold(
        *expected_publication_id, current_time, retry_target_lead_ns,
        parameters_.joint_names.size(), true,
        timeline_recovery_failure);
    }
  }
  if (timeline_recovery_status != TimelineRecoveryStatus::kSuccess) {
    ++telemetry_.timeline_recovery_failures;
    std::string message = "Published command timeline recovery failed: " +
      timeline_recovery_failure;
    if (timeline_recovery_status == TimelineRecoveryStatus::kExhaustedPublishedSuffix) {
      enter_controlled_rearm(std::move(message));
    } else {
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait(std::move(message));
    }
    return false;
  }
  commands_.completeDeferredCommandDeadlineRetry();
  return true;
}

bool CollisionAwareServoComponent::prepare_verified_terminal_hold_publication(
  const rclcpp::Time & tick_time, const msg::TrackingTarget & target,
  const std::optional<control::PublicationId> & expected_publication_id,
  const std::optional<rclcpp::Time> & expected_anchor_stamp,
  const std::optional<std::uint64_t> & expected_hardware_epoch,
  const std::optional<std::uint64_t> & expected_proven_hold_epoch,
  const std::uint64_t expected_scene_revision,
  std::optional<planning_scene_monitor::LockedPlanningSceneRO> & locked_scene,
  planning_scene::PlanningSceneConstPtr & planning_scene)
{
  const auto command_identity_is_current = [this, &expected_publication_id,
      &expected_anchor_stamp]() {
      return expected_publication_id.has_value() && expected_anchor_stamp.has_value() &&
             commands_.epoch_active && commands_.epoch_has_published &&
             commands_.active_publication_id == expected_publication_id &&
             commands_.hasVerifiedTerminalHold(parameters_.joint_names.size()) &&
             !commands_.queue.empty() &&
             commands_.queue.back().time_stamp == *expected_anchor_stamp;
    };
  const auto scene_is_current = [this, expected_scene_revision]() {
      const std::uint64_t revision =
        planning_scene_collision_revision_.load(std::memory_order_acquire);
      return revision == expected_scene_revision &&
             revision == last_validated_scene_revision_;
    };
  const auto target_is_current = [this, &target, &tick_time](const rclcpp::Time & at_time) {
      const auto current = current_target(at_time);
      return target_snapshot_is_current(target, tick_time, at_time) && current.has_value() &&
             current->mode == target.mode && current->target_id == target.target_id &&
             accept_pose_commands_.load(std::memory_order_acquire) &&
             !pause_requested_.load(std::memory_order_acquire);
    };
  const auto fail_invariant = [this](std::string reason) {
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait(std::move(reason));
    };

  if (!scene_is_current() || !planning_scene || !command_identity_is_current()) {
    fail_invariant("Verified terminal hold lost its publication or validated collision scene");
    return false;
  }
  const rclcpp::Time check_time = now();
  if (check_time < tick_time) {
    fail_invariant("ROS time rewound during verified terminal-hold validation");
    return false;
  }
  if (!target_is_current(check_time)) {
    return false;
  }
  const double anchor_lead_sec =
    (commands_.queue.back().time_stamp - check_time).seconds();
  if (!std::isfinite(anchor_lead_sec)) {
    fail_invariant("Verified terminal-hold anchor has a non-finite send lead");
    return false;
  }
  if (anchor_lead_sec + kFiniteEpsilon >= parameters_.residual_command_latency_sec) {
    return true;
  }

  locked_scene.reset();
  planning_scene.reset();
  if (!validateHardwareIoForControlTick()) {
    return false;
  }
  if ((expected_hardware_epoch.has_value() &&
    (!cycle_hardware_io_status_.has_value() ||
    cycle_hardware_io_status_->sample.epoch != *expected_hardware_epoch)) ||
    (expected_proven_hold_epoch.has_value() &&
    proven_q_hold_epoch_ != expected_proven_hold_epoch))
  {
    fail_invariant("Hardware I/O epoch changed during late terminal-hold validation");
    return false;
  }

  std::string terminal_hold_reason;
  const RearmReadiness readiness = safe_rearm_ready(check_time, terminal_hold_reason);
  locked_scene.emplace(planning_scene_monitor_);
  planning_scene = *locked_scene;
  const rclcpp::Time proof_time = now();
  if (!planning_scene || !scene_is_current() || !command_identity_is_current()) {
    fail_invariant("Verified terminal hold or collision scene changed during late proof");
    return false;
  }
  if (readiness == RearmReadiness::kUnsafe) {
    fail_invariant(std::move(terminal_hold_reason));
    return false;
  }
  if (proof_time < check_time) {
    fail_invariant("ROS time rewound during late terminal-hold proof");
    return false;
  }
  if (!target_is_current(proof_time)) {
    return false;
  }
  if (readiness == RearmReadiness::kWaiting) {
    publish_status(
      moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
      std::move(terminal_hold_reason));
    return false;
  }

  const std::int64_t target_lead_ns = seconds_to_nanoseconds(
    servo_parameters_.max_expected_latency);
  std::string resume_failure;
  if (commands_.resumeVerifiedTerminalHold(
      *expected_publication_id, proof_time, target_lead_ns,
      parameters_.joint_names.size(), true, resume_failure) != TimelineRecoveryStatus::kSuccess)
  {
    fail_invariant(std::move(resume_failure));
    return false;
  }
  return true;
}

std::optional<control::CollisionValidationBudget>
CollisionAwareServoComponent::make_collision_validation_budget(
  const bool ordinary_face_budget_scope, const bool verified_terminal_hold_candidate,
  const rclcpp::Time & current_time,
  const control::CollisionValidationBudget::Clock::time_point current_wall_time) const
{
  if (!ordinary_face_budget_scope || verified_terminal_hold_candidate ||
    !commands_.epoch_active || !commands_.epoch_has_published ||
    commands_.activePublication() == nullptr || commands_.queue.empty())
  {
    return std::nullopt;
  }

  const double replacement_lead_sec =
    (commands_.queue.back().time_stamp - current_time).seconds();
  const double available_budget_sec = std::isfinite(replacement_lead_sec) ?
    std::max(0.0, replacement_lead_sec - parameters_.residual_command_latency_sec) : 0.0;
  const auto budget_duration = std::chrono::duration_cast<
    control::CollisionValidationBudget::Clock::duration>(
    std::chrono::duration<double>(available_budget_sec));
  return control::CollisionValidationBudget(current_wall_time + budget_duration);
}

bool CollisionAwareServoComponent::stop_for_expired_collision_validation_budget(
  const control::CollisionValidationBudget * const budget)
{
  if (budget == nullptr || !budget->expired()) {
    return false;
  }
  ++telemetry_.collision_validation_budget_exhaustion_count;
  controller_mode_ = ControllerMode::kBraking;
  publish_status(
    moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
    "Collision validation budget exhausted; continuing the published safe trajectory");
  return true;
}

bool CollisionAwareServoComponent::validate_actual_collision_state(
  const planning_scene::PlanningScene & planning_scene,
  const moveit::core::RobotState & robot_state)
{
  auto actual_robot_state = std::make_shared<moveit::core::RobotState>(robot_state);
  control::ActualPositionEncoding actual_position_encoding =
    control::ActualPositionEncoding::kDouble;
  if (require_hardware_io_status_) {
    const auto hardware_status = current_hardware_io_status();
    if (!hardware_status || !hardware_status->valid) {
      request_hardware_inhibit();
      enter_safety_wait("Hardware read is unavailable for actual collision validation");
      return false;
    }
    actual_robot_state->setJointGroupPositions(
      joint_model_group_, hardware_status->sample.position.data());
    actual_robot_state->update(true);
    actual_position_encoding = control::ActualPositionEncoding::kSdkFloatPromotedToDouble;
  }
  const control::CollisionActualStateValidationResult actual_state_validation =
    collision_constraint_builder_->validateActualState(
    planning_scene, *actual_robot_state, *joint_model_group_, actual_position_encoding);
  ++telemetry_.actual_state_validation_count;
  telemetry_.last_actual_minimum_self_distance_m =
    std::isfinite(actual_state_validation.minimum_self_distance_m) ?
    std::optional<double>{actual_state_validation.minimum_self_distance_m} : std::nullopt;
  telemetry_.last_actual_minimum_world_distance_m =
    std::isfinite(actual_state_validation.minimum_world_distance_m) ?
    std::optional<double>{actual_state_validation.minimum_world_distance_m} : std::nullopt;
  telemetry_.last_actual_closest_pair = actual_state_validation.closest_pair.empty() ?
    "outside hard-clearance query" : actual_state_validation.closest_pair;
  telemetry_.observeClearances(
    actual_state_validation.minimum_self_distance_m,
    actual_state_validation.minimum_world_distance_m);
  if (!actual_state_validation.input_valid || actual_state_validation.unsafe) {
    ++telemetry_.actual_state_validation_failure_count;
    const std::string reason = !actual_state_validation.input_valid ?
      actual_state_validation.failure_reason :
      "hard-clearance/corridor violation for " + telemetry_.last_actual_closest_pair;
    if (require_hardware_io_status_) {
      request_hardware_inhibit();
    }
    enter_safety_wait("Measured robot state is unsafe: " + reason);
    return false;
  }
  return true;
}

void CollisionAwareServoComponent::control_tick()
{
  const rclcpp::Time current_time = now();
  current_tick_context_ = diagnostics::ServoExecutionContextRecord{};
  current_tick_context_.control_stamp_ns = current_time.nanoseconds();
  current_tick_context_.tick_sequence = ++execution_context_tick_sequence_;
  const auto current_wall_time = std::chrono::steady_clock::now();
  if (!validateHardwareIoForControlTick()) {
    return;
  }
  const control::ControlTimeObservation time_observation =
    commands_.time_grid->observe(current_time.nanoseconds());
  if (!prepare_control_cycle(current_time, current_wall_time, time_observation)) {
    return;
  }

  if (!recover_command_timeline(current_time, time_observation.skipped_periods)) {
    return;
  }

  std::string state_failure_reason;
  bool requires_latched_halt = false;
  std::optional<control::JointMotionState> actual_feedback;
  std::optional<rclcpp::Time> actual_feedback_stamp;
  const auto state = predicted_or_measured_state(
    current_time, state_failure_reason, requires_latched_halt,
    &actual_feedback, &actual_feedback_stamp);
  if ((commands_.bootstrap_ack_pending || commands_.bootstrap_reset_ack_pending) &&
    commands_.rearmDeadlineExpired(current_time, current_wall_time))
  {
    if (commands_.claimRearmTimeoutReport()) {
      ++telemetry_.rearm_timeout_count;
    }
    if (require_hardware_io_status_) {
      request_hardware_inhibit();
    }
    enter_safety_wait(
      commands_.bootstrap_reset_ack_pending ?
      "Trajectory controller did not acknowledge the actual-state reset in time" :
      "Trajectory controller did not acknowledge the stationary bootstrap in time");
    return;
  }
  if (!state.has_value()) {
    if (requires_latched_halt) {
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait(std::move(state_failure_reason));
    } else {
      if (state_failure_reason.empty()) {
        state_failure_reason = initial_state_received_ ?
          "Fresh complete finite robot feedback is unavailable" :
          "Waiting for the first complete finite robot state";
      }
      handle_recoverable_control_gap(std::move(state_failure_reason));
    }
    return;
  }
  auto robot_state = planning_scene_monitor_->getStateMonitor()->getCurrentState();
  if (!robot_state) {
    handle_recoverable_control_gap("PlanningSceneMonitor RobotState is unavailable");
    return;
  }
  robot_state = std::make_shared<moveit::core::RobotState>(*robot_state);
  robot_state->update(true);

  std::optional<planning_scene_monitor::LockedPlanningSceneRO> locked_scene;
  locked_scene.emplace(planning_scene_monitor_);
  planning_scene::PlanningSceneConstPtr planning_scene = *locked_scene;
  const std::uint64_t collision_scene_revision = planning_scene_collision_revision_.load(
    std::memory_order_acquire);
  if (commands_.epoch_active && collision_scene_revision != last_validated_scene_revision_) {
    ++telemetry_.planning_scene_invalidation_count;
    if (require_hardware_io_status_) {
      request_hardware_inhibit();
    }
    enter_safety_wait(
      "PlanningScene collision geometry changed while commands were buffered");
    return;
  }
  last_validated_scene_revision_ = collision_scene_revision;
  if (!validate_actual_collision_state(*planning_scene, *robot_state)) {
    return;
  }

  robot_state->setJointGroupPositions(joint_model_group_, state->positions);
  robot_state->setJointGroupVelocities(joint_model_group_, state->velocities);
  robot_state->setJointGroupAccelerations(joint_model_group_, state->accelerations);
  robot_state->update();

  control::JointMotionState motion_state;
  motion_state.position = state->positions;
  motion_state.velocity = state->velocities;
  motion_state.acceleration = state->accelerations;
  telemetry_.observeMotionState(motion_state);

  if (commands_.startup_warmup_complete && !commands_.epoch_active) {
    process_startup_cycle(*state, *robot_state, *planning_scene);
    return;
  }

  control::CollisionConstraintResult collision_result;
  const control::JointMotionLimits collision_context_limits = active_motion_limits();
  if (!build_and_record_collision_context(
    *planning_scene, *robot_state, motion_state, collision_context_limits, collision_result))
  {
    return;
  }

  std::vector<control::LinearVelocityConstraint> constraints = collision_result.constraints;
  bool segment_validation_required =
    collision_constraint_builder_->shouldValidateSegment(
    collision_result.diagnostics, parameters_.segment_validation_distance_m);
  // Collision evaluation can take several milliseconds while the ROS
  // executor keeps receiving newer latest-only targets. Compare their stamp
  // with the clock at consumption time, not with the stale tick-start time.
  const auto target = current_target(current_time);
  const bool tracking_requested = target.has_value() &&
    accept_pose_commands_.load(std::memory_order_acquire) &&
    !pause_requested_.load(std::memory_order_acquire);
  update_execution_target_context(target);

  std::optional<double> actual_feedback_time_sec;
  if (actual_feedback.has_value() && actual_feedback_stamp.has_value()) {
    actual_feedback_time_sec = actual_feedback_stamp->seconds();
  }
  const auto * active_publication = commands_.activePublication();
  const auto * latest_publication = commands_.publications.latest();

  std::optional<ValidatedCommand> accepted_command;
  control::QpResult accepted_qp_result;
  std::optional<control::QpResult> guard_rejected_tracking_result;
  std::optional<std::string> mitigated_failure;
  bool reference_guard_blend_accepted = false;
  bool reference_guard_recovery_accepted = false;
  std::int8_t mitigated_status = moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION;

  const auto reference = motion_reference_->update(
    *robot_state, tracking_requested ? target : std::nullopt, *planning_scene,
    current_time.seconds(), collision_scene_revision, actual_feedback,
    actual_feedback_time_sec, motion_state, active_publication, latest_publication);
  CompactProfilePublicationOutcome compact_profile_publication(
    motion_reference_.get(), reference.compact_profile_command);
  bool compact_profile_candidate_accepted = false;
  const control::JointMotionLimits & command_limits = active_motion_limits();
  if (!same_dynamic_limits(collision_context_limits, command_limits)) {
    if (!build_and_record_collision_context(
        *planning_scene, *robot_state, motion_state, command_limits, collision_result))
    {
      return;
    }
    constraints = collision_result.constraints;
    segment_validation_required = collision_constraint_builder_->shouldValidateSegment(
      collision_result.diagnostics, parameters_.segment_validation_distance_m);
  }
  current_tick_context_.target_used_for_reference =
    tracking_requested && reference.tracking.has_value();
  const bool ordinary_face_budget_scope = target &&
    target->mode == msg::TrackingTarget::FACE && tracking_requested && reference.tracking &&
    reference.tracking->has_face_primary_roll && !reference.follows_path &&
    !motion_reference_->compactManeuverActive();
  const bool verified_terminal_hold_candidate = ordinary_face_budget_scope &&
    commands_.hasVerifiedTerminalHold(parameters_.joint_names.size());
  const auto verified_terminal_publication_id = verified_terminal_hold_candidate ?
    commands_.active_publication_id : std::optional<control::PublicationId>{};
  const auto verified_terminal_anchor_stamp = verified_terminal_hold_candidate ?
    std::optional<rclcpp::Time>{commands_.queue.back().time_stamp} :
  std::optional<rclcpp::Time>{};
  const std::optional<std::uint64_t> verified_terminal_hardware_epoch =
    require_hardware_io_status_ && cycle_hardware_io_status_.has_value() ?
    std::optional<std::uint64_t>{cycle_hardware_io_status_->sample.epoch} :
  std::nullopt;
  const auto verified_terminal_proven_hold_epoch = proven_q_hold_.has_value() ?
    proven_q_hold_epoch_ : std::optional<std::uint64_t>{};
  auto collision_validation_budget = make_collision_validation_budget(
    ordinary_face_budget_scope, verified_terminal_hold_candidate, current_time,
    current_wall_time);
  control::CollisionValidationBudget * const collision_budget =
    collision_validation_budget ? &*collision_validation_budget : nullptr;
  if (stop_for_expired_collision_validation_budget(collision_budget)) {
    return;
  }
  if (reference.motion_reference_support_unavailable) {
    mitigated_failure =
      "Motion-reference safety support is unavailable; using the validated braking command";
  } else {
    constraints.insert(
      constraints.end(), reference.compact_radius_constraints.begin(),
      reference.compact_radius_constraints.end());
    constraints.insert(
      constraints.end(), reference.compact_roll_constraints.begin(),
      reference.compact_roll_constraints.end());
    constraints.insert(
      constraints.end(), reference.compact_pitch_constraints.begin(),
      reference.compact_pitch_constraints.end());
  }
  if (reference.tracking) {
    telemetry_.last_position_error_m = reference.tracking->position_error_m;
    telemetry_.last_screen_normal_pointing_error_rad = reference.tracking->pointing_error_rad;
    telemetry_.last_roll_error_rad = reference.tracking->roll_error_rad;
    (void)telemetry_.pointing_error_statistics.add(reference.tracking->pointing_error_rad);
  }
  if (reference.compact_profile_command &&
    !reference.motion_reference_support_unavailable)
  {
    std::string candidate_failure;
    CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
    std::optional<VelocityConstraintViolation> hard_velocity_violation;
    if (reference.compact_profile_velocity) {
      const double tolerance = velocity_qp_->config().solution_feasibility_tolerance;
      // Collision rows are the conservative QP viability screen. This exact
      // profile is still admitted only after full candidate and brake-tail validation.
      if (satisfies_velocity_constraints(
          *reference.compact_profile_velocity, constraints, tolerance,
          collision_result.constraints.size(), &hard_velocity_violation))
      {
        accepted_command = validated_candidate(
          robot_state, *state, motion_state, *reference.compact_profile_velocity,
          *planning_scene, segment_validation_required, CommandAttempt::kTracking,
          candidate_failure, collision_result.diagnostics, rejection_kind, collision_budget,
          true);
      } else {
        candidate_failure = "Compact profile command violates an active hard velocity constraint";
      }
    } else {
      candidate_failure = "Compact profile command has no joint velocity sample";
    }
    if (stop_for_expired_collision_validation_budget(collision_budget)) {
      return;
    }
    if (accepted_command.has_value()) {
      compact_profile_candidate_accepted = true;
      accepted_qp_result.status = control::QpStatus::kSolved;
      accepted_qp_result.joint_velocity = *reference.compact_profile_velocity;
      controller_mode_ = ControllerMode::kTracking;
    } else {
      if (hard_velocity_violation) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Compact profile candidate rejected: %s; hard velocity row=%zu, "
          "projected_qdot=%.9g rad/s, bounds=[%.9g, %.9g] rad/s",
          candidate_failure.c_str(), hard_velocity_violation->row,
          hard_velocity_violation->projected_velocity,
          hard_velocity_violation->lower_bound, hard_velocity_violation->upper_bound);
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Compact profile candidate rejected: %s", candidate_failure.c_str());
      }
      mitigated_failure = "Compact profile candidate rejected: " + candidate_failure;
      // Keep the already validated braking suffix when it is still eligible.
      const auto continuation_status = tryContinuePublishedBrake(
        current_time, target, tracking_requested, collision_scene_revision);
      if (continuation_status != PublishedBrakeContinuationStatus::kNotEligible) {
        return;
      }
    }
  } else if (reference.task && !reference.motion_reference_support_unavailable) {
    auto task = *reference.task;
    if (stop_for_expired_collision_validation_budget(collision_budget)) {
      return;
    }
    control::QpResult tracking_result = solve_with_telemetry(task, motion_state, constraints);
    if (stop_for_expired_collision_validation_budget(collision_budget)) {
      return;
    }
    if (!tracking_result.command_available()) {
      log_qp_failure(
        CommandAttempt::kTracking, tracking_result, motion_state, constraints,
        collision_result.diagnostics);
      mitigated_failure = "Tracking QP command is unavailable";
    } else {
      std::string candidate_failure;
      CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
      accepted_command = validated_candidate(
        robot_state, *state, motion_state, tracking_result.joint_velocity,
        *planning_scene, segment_validation_required, CommandAttempt::kTracking,
        candidate_failure, collision_result.diagnostics, rejection_kind, collision_budget);
      if (stop_for_expired_collision_validation_budget(collision_budget)) {
        return;
      }
      if (accepted_command.has_value()) {
        accepted_qp_result = tracking_result;
        controller_mode_ = ControllerMode::kTracking;
      } else {
        const bool transient_compact_stop_rejection = reference.follows_path &&
          motion_reference_->compactManeuverActive() &&
          rejection_kind == CandidateRejectionKind::kCompactStopEnvelope;
        const bool retryable_reference_guard =
          rejection_kind == CandidateRejectionKind::kReferenceGuard;
        if (retryable_reference_guard) {
          guard_rejected_tracking_result = tracking_result;
        } else if (!transient_compact_stop_rejection) {
          motion_reference_->rejectPath();
        }
        mitigated_failure = "Tracking candidate rejected: " + candidate_failure;
      }
    }
  }

  if (!accepted_command.has_value()) {
    if (stop_for_expired_collision_validation_budget(collision_budget)) {
      return;
    }
    const control::QpResult braking_result =
      brake_with_telemetry(motion_state, collision_result.constraints);
    if (stop_for_expired_collision_validation_budget(collision_budget)) {
      return;
    }
    if (!braking_result.command_available()) {
      log_qp_failure(
        CommandAttempt::kBraking, braking_result, motion_state,
        collision_result.constraints,
        collision_result.diagnostics);
      const auto continuation_status = tryContinuePublishedBrake(
        current_time, target, tracking_requested, collision_scene_revision);
      if (continuation_status != PublishedBrakeContinuationStatus::kNotEligible) {
        return;
      }
      enter_controlled_rearm("No feasible jerk-limited collision-safe braking command");
      return;
    }

    std::string braking_candidate_failure;
    CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
    accepted_command = validated_candidate(
      robot_state, *state, motion_state, braking_result.joint_velocity,
      *planning_scene, segment_validation_required, CommandAttempt::kBraking,
      braking_candidate_failure, collision_result.diagnostics, rejection_kind,
      collision_budget);
    if (stop_for_expired_collision_validation_budget(collision_budget)) {
      return;
    }
    if (!accepted_command.has_value()) {
      const auto continuation_status = tryContinuePublishedBrake(
        current_time, target, tracking_requested, collision_scene_revision);
      if (continuation_status != PublishedBrakeContinuationStatus::kNotEligible) {
        return;
      }
      enter_controlled_rearm(
        "Collision-safe braking candidate rejected: " + braking_candidate_failure);
      return;
    }
    accepted_qp_result = braking_result;
    controller_mode_ = ControllerMode::kBraking;

    if (guard_rejected_tracking_result.has_value()) {
      if (reference.task) {
        reference_guard_blend_accepted = try_reference_guard_blend(
          robot_state, *state, motion_state, *planning_scene,
          segment_validation_required, *guard_rejected_tracking_result, braking_result,
          reference.task->primary_matrix, constraints, collision_result.diagnostics,
          accepted_command, accepted_qp_result, collision_budget);
      }
      if (stop_for_expired_collision_validation_budget(collision_budget)) {
        return;
      }
      if (reference_guard_blend_accepted) {
        controller_mode_ = ControllerMode::kTracking;
        mitigated_failure =
          "Tracking velocity was blended toward a validated braking step to preserve "
          "the motion-reference guard";
      } else {
        const bool ordinary_fresh_face = target &&
          target->mode == msg::TrackingTarget::FACE && reference.tracking &&
          !motion_reference_->compactManeuverActive() && !reference.follows_path &&
          reference.tracking->has_face_primary_roll;
        if (ordinary_fresh_face && reference.task) {
          reference_guard_recovery_accepted = try_reference_guard_recovery(
            robot_state, *state, motion_state, *planning_scene,
            segment_validation_required, *reference.task,
            reference.tracking->has_face_primary_roll, constraints,
            collision_result.diagnostics, accepted_command, accepted_qp_result,
            collision_budget);
          if (stop_for_expired_collision_validation_budget(collision_budget)) {
            return;
          }
        }
        if (reference_guard_recovery_accepted) {
          controller_mode_ = ControllerMode::kTracking;
          mitigated_failure =
            "Screen-edge inward recovery accepted by the full motion-reference guard";
        }
        if (!reference_guard_recovery_accepted) {
          motion_reference_->rejectPath();
        }
      }
    }
  }

  if (!commands_.startup_warmup_complete || !commands_.epoch_active) {
    process_startup_cycle(*state, *robot_state, *planning_scene);
    return;
  }

  if (verified_terminal_hold_candidate && !prepare_verified_terminal_hold_publication(
      current_time, *target, verified_terminal_publication_id,
      verified_terminal_anchor_stamp, verified_terminal_hardware_epoch,
      verified_terminal_proven_hold_epoch, collision_scene_revision,
      locked_scene, planning_scene))
  {
    return;
  }

  std::string publication_failure;
  if (stop_for_expired_collision_validation_budget(collision_budget)) {
    return;
  }
  const CommandPublicationStatus publication_status = enqueue_and_publish(
    *state, std::move(*accepted_command), current_time, publication_failure, collision_budget);
  if (publication_status == CommandPublicationStatus::kCollisionValidationBudgetExhausted) {
    (void)stop_for_expired_collision_validation_budget(collision_budget);
    return;
  }
  if (publication_status != CommandPublicationStatus::kSuccess) {
    if (publication_status == CommandPublicationStatus::kRecoverableCommandDeadline ||
      publication_status == CommandPublicationStatus::kRecoverableTimingFailure)
    {
      const bool continuation_mode = target &&
        (target->mode == msg::TrackingTarget::FACE || target->mode == msg::TrackingTarget::REST);
      if (tracking_requested && continuation_mode) {
        const auto continuation_status = tryContinuePublishedBrake(
          current_time, target, tracking_requested, collision_scene_revision);
        if (continuation_status != PublishedBrakeContinuationStatus::kNotEligible) {
          return;
        }
      }
      if (commands_.deferCommandDeadlineRetry()) {
        // Drop this stale candidate. The next cycle may only resume from an
        // exact point in the already-published, validated trajectory suffix.
        return;
      }
    }
    if (publication_status == CommandPublicationStatus::kRecoverableCommandDeadline ||
      publication_status == CommandPublicationStatus::kRecoverableTimingFailure)
    {
      enter_controlled_rearm(std::move(publication_failure));
    } else {
      enter_safety_wait(std::move(publication_failure));
    }
    return;
  }
  compact_profile_publication.markPublished(compact_profile_candidate_accepted);
  telemetry_.observeCommandTransition(
    motion_state, commands_.queue.back(), parameters_.control_period_sec);
  commands_.recordHealthyPublication();
  if (reference.tracking) {
    current_tick_context_.face_tracking_published =
      controller_mode_ == ControllerMode::kTracking && !reference_guard_recovery_accepted;
  }

  update_pause_after_publication();

  const bool collision_avoidance_active =
    collision_result.diagnostics.active_distance_constraints > 0U;
  publish_control_status(
    collision_avoidance_active, reference_guard_blend_accepted,
    reference_guard_recovery_accepted,
    mitigated_failure, accepted_qp_result,
    mitigated_status);
}

PublishedBrakeContinuationStatus CollisionAwareServoComponent::tryContinuePublishedBrake(
  const rclcpp::Time & current_time,
  const std::optional<msg::TrackingTarget> & target,
  const bool tracking_requested,
  const std::uint64_t collision_scene_revision)
{
  const rclcpp::Time decision_time = now();
  const auto reject_reference_path = [this](
    const PublishedBrakeContinuationStatus status)
    {
      if (status != PublishedBrakeContinuationStatus::kContinued && motion_reference_) {
        motion_reference_->rejectPath();
      }
      return status;
    };
  if (decision_time < current_time || !tracking_requested || !target.has_value() ||
    (target->mode != msg::TrackingTarget::FACE &&
    target->mode != msg::TrackingTarget::REST) || !commands_.epoch_active ||
    !commands_.epoch_has_published || commands_.rearm_pending ||
    commands_.published_tail_rearm_pending || commands_.bootstrap_ack_pending ||
    !commands_.startup_warmup_complete ||
    !trajectory_controller_active_.load(std::memory_order_acquire))
  {
    return reject_reference_path(PublishedBrakeContinuationStatus::kNotEligible);
  }
  const auto current_control_target = current_target(decision_time);
  if (!current_control_target.has_value() ||
    current_control_target->mode != target->mode ||
    current_control_target->target_id != target->target_id)
  {
    return reject_reference_path(PublishedBrakeContinuationStatus::kNotEligible);
  }
  const std::uint64_t current_scene_revision =
    planning_scene_collision_revision_.load(std::memory_order_acquire);
  if (current_scene_revision != collision_scene_revision ||
    current_scene_revision != last_validated_scene_revision_)
  {
    return reject_reference_path(PublishedBrakeContinuationStatus::kNotEligible);
  }
  if (!accept_pose_commands_.load(std::memory_order_acquire) ||
    pause_requested_.load(std::memory_order_acquire) ||
    paused_.load(std::memory_order_acquire) ||
    halt_latched_.load(std::memory_order_acquire) ||
    stop_control_.load(std::memory_order_acquire) ||
    reset_queue_requested_.load(std::memory_order_acquire))
  {
    return reject_reference_path(PublishedBrakeContinuationStatus::kNotEligible);
  }

  std::string continuation_failure;
  const auto continuation_status = commands_.continuePublishedBrake(
    decision_time, seconds_to_nanoseconds(servo_parameters_.max_expected_latency),
    continuation_failure, servo_parameters_.max_expected_latency,
    parameters_.joint_names.size());
  switch (continuation_status) {
    case PublishedBrakeContinuationStatus::kContinued:
      controller_mode_ = ControllerMode::kBraking;
      publish_status(
        moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
        "Continuing exact samples from the validated published braking suffix");
      return PublishedBrakeContinuationStatus::kContinued;
    case PublishedBrakeContinuationStatus::kNotEligible:
      return reject_reference_path(PublishedBrakeContinuationStatus::kNotEligible);
    case PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix:
      if (target->mode == msg::TrackingTarget::FACE) {
        controller_mode_ = ControllerMode::kBraking;
        publish_status(
          moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
          "Published FACE braking suffix is exhausted; keeping it for next-tick proof");
        // kContinued means only that this tick keeps the existing publication unchanged.
        return PublishedBrakeContinuationStatus::kContinued;
      }
      enter_controlled_rearm(
        "Published braking continuation has no transmitted future sample: " +
        continuation_failure);
      return reject_reference_path(
        PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix);
    case PublishedBrakeContinuationStatus::kInvariantFailure:
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait(
        "Published braking continuation violated its timeline invariant: " +
        continuation_failure);
      return reject_reference_path(PublishedBrakeContinuationStatus::kInvariantFailure);
  }
  return reject_reference_path(PublishedBrakeContinuationStatus::kInvariantFailure);
}

control::QpResult CollisionAwareServoComponent::solve_with_telemetry(
  const control::HierarchicalVelocityTask & task,
  const control::JointMotionState & state,
  const std::vector<control::LinearVelocityConstraint> & constraints)
{
  const auto start = std::chrono::steady_clock::now();
  control::QpResult result = velocity_qp_->solve(
    task, state, active_motion_limits(), constraints);
  telemetry_.recordQpResult(start, result);
  control::QpResult handoff;
  handoff.primary_snapshot_disposition = result.primary_snapshot_disposition;
  handoff.secondary_snapshot_disposition = result.secondary_snapshot_disposition;
  handoff.tertiary_snapshot_disposition = result.tertiary_snapshot_disposition;
  handoff.distance_snapshot_disposition = result.distance_snapshot_disposition;
  handoff.primary_failure_snapshot = std::move(result.primary_failure_snapshot);
  handoff.secondary_failure_snapshot = std::move(result.secondary_failure_snapshot);
  handoff.tertiary_failure_snapshot = std::move(result.tertiary_failure_snapshot);
  handoff.distance_failure_snapshot = std::move(result.distance_failure_snapshot);
  control::QpFailureSnapshotContext context;
  context.ros_stamp_ns = now().nanoseconds();
  context.steady_stamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  if (cycle_hardware_io_status_.has_value()) {
    context.hardware_epoch = cycle_hardware_io_status_->sample.epoch;
    context.hardware_read_sequence = cycle_hardware_io_status_->sample.read_sequence;
  }
  if (commands_.active_publication_id.has_value()) {
    context.active_publication_id = *commands_.active_publication_id;
  }
  if (const auto * latest = commands_.publications.latest(); latest != nullptr) {
    context.latest_publication_id = latest->id;
  }
  (void)qp_failure_snapshot_mailbox_.trySubmit(std::move(handoff), context);
  return result;
}

control::QpResult CollisionAwareServoComponent::brake_with_telemetry(
  const control::JointMotionState & state,
  const std::vector<control::LinearVelocityConstraint> & constraints)
{
  const auto start = std::chrono::steady_clock::now();
  control::QpResult result = velocity_qp_->brake(
    state, active_motion_limits(), constraints);
  telemetry_.recordQpResult(start, result);
  control::QpResult handoff;
  handoff.primary_snapshot_disposition = result.primary_snapshot_disposition;
  handoff.secondary_snapshot_disposition = result.secondary_snapshot_disposition;
  handoff.tertiary_snapshot_disposition = result.tertiary_snapshot_disposition;
  handoff.distance_snapshot_disposition = result.distance_snapshot_disposition;
  handoff.primary_failure_snapshot = std::move(result.primary_failure_snapshot);
  handoff.secondary_failure_snapshot = std::move(result.secondary_failure_snapshot);
  handoff.tertiary_failure_snapshot = std::move(result.tertiary_failure_snapshot);
  handoff.distance_failure_snapshot = std::move(result.distance_failure_snapshot);
  control::QpFailureSnapshotContext context;
  context.ros_stamp_ns = now().nanoseconds();
  context.steady_stamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  if (cycle_hardware_io_status_.has_value()) {
    context.hardware_epoch = cycle_hardware_io_status_->sample.epoch;
    context.hardware_read_sequence = cycle_hardware_io_status_->sample.read_sequence;
  }
  if (commands_.active_publication_id.has_value()) {
    context.active_publication_id = *commands_.active_publication_id;
  }
  if (const auto * latest = commands_.publications.latest(); latest != nullptr) {
    context.latest_publication_id = latest->id;
  }
  (void)qp_failure_snapshot_mailbox_.trySubmit(std::move(handoff), context);
  return result;
}

}  // namespace face_tracking_arm
