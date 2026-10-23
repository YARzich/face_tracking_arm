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
  control::QpResult & accepted_qp_result)
{
  for (const double tracking_weight : {0.5, 0.25}) {
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
      blend_failure, collision_diagnostics, blend_rejection_kind);
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
  control::QpResult & accepted_qp_result)
{
  const auto recovery_task = control::makeFaceEdgeOnlyRecoveryTask(
    face_task, has_face_primary_roll);
  if (!recovery_task) {
    return false;
  }
  control::QpResult recovery_result =
    solve_with_telemetry(*recovery_task, motion_state, constraints);
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
    recovery_failure, collision_diagnostics, recovery_rejection_kind);
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
      "Safe primary orientation command only");
  } else {
    publish_status(moveit_msgs::msg::ServoStatus::NO_WARNING, "No warnings");
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

std::optional<control::LinearVelocityConstraint> CollisionAwareServoComponent::j1_hold_constraint(
  const control::MotionReferenceResult & reference,
  const control::JointMotionState & motion_state) const
{
  if (!reference.j1_hold_position_rad || !reference.j1_joint_index ||
    *reference.j1_joint_index >= static_cast<std::size_t>(motion_state.position.size()))
  {
    return std::nullopt;
  }
  const auto index = static_cast<Eigen::Index>(*reference.j1_joint_index);
  control::detail::ScalarBrakingOptions braking_options;
  braking_options.period_sec = parameters_.control_period_sec;
  const auto braking_step = control::detail::scalar_braking_policy::nextStep(
    motion_state.velocity[index], motion_state.acceleration[index],
    control::detail::ScalarBrakingLimits{
      motion_limits_.max_velocity[index], motion_limits_.max_acceleration[index],
      motion_limits_.max_jerk[index]}, braking_options);
  if (!braking_step.valid) {
    return std::nullopt;
  }
  control::LinearVelocityConstraint hold;
  hold.coefficients = Eigen::VectorXd::Zero(motion_state.position.size());
  hold.coefficients[index] = 1.0;
  hold.lower_bound = braking_step.velocity;
  hold.upper_bound = braking_step.velocity;
  return hold;
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

  // Recover to the epoch's initial lead, which includes one control period
  // beyond the send-time minimum for the next computation to complete.
  const std::int64_t retry_target_lead_ns = seconds_to_nanoseconds(
    servo_parameters_.max_expected_latency);
  const auto timeline_recovery_periods = commands_.timelineRecoveryPeriods(
    time_observation.skipped_periods, current_time.nanoseconds(), retry_target_lead_ns);
  if (!timeline_recovery_periods.has_value()) {
    enter_controlled_rearm("Deferred command deadline has no active command-grid anchor");
    return;
  }
  std::string timeline_recovery_failure;
  const TimelineRecoveryStatus timeline_recovery_status = commands_.recoverTimeline(
    *timeline_recovery_periods, current_time, timeline_recovery_failure,
    servo_parameters_.max_expected_latency, parameters_.joint_names.size());
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
    return;
  }
  commands_.completeDeferredCommandDeadlineRetry();

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

  planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
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
  auto actual_robot_state = std::make_shared<moveit::core::RobotState>(*robot_state);
  control::ActualPositionEncoding actual_position_encoding =
    control::ActualPositionEncoding::kDouble;
  if (require_hardware_io_status_) {
    const auto hardware_status = current_hardware_io_status();
    if (!hardware_status || !hardware_status->valid) {
      request_hardware_inhibit();
      enter_safety_wait("Hardware read is unavailable for actual collision validation");
      return;
    }
    actual_robot_state->setJointGroupPositions(
      joint_model_group_, hardware_status->sample.position.data());
    actual_robot_state->update(true);
    actual_position_encoding = control::ActualPositionEncoding::kSdkFloatPromotedToDouble;
  }
  const control::CollisionActualStateValidationResult actual_state_validation =
    collision_constraint_builder_->validateActualState(
    *planning_scene, *actual_robot_state, *joint_model_group_, actual_position_encoding);
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

  const auto collision_build_start = std::chrono::steady_clock::now();
  const control::CollisionConstraintResult collision_result =
    collision_constraint_builder_->build(
    *planning_scene, *robot_state, *joint_model_group_, motion_state, motion_limits_,
    parameters_.control_period_sec);
  (void)telemetry_.collision_build_statistics.add(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - collision_build_start).count());
  (void)telemetry_.collision_self_query_statistics.add(collision_result.diagnostics.self_query_ms);
  (void)telemetry_.collision_world_query_statistics.add(
      collision_result.diagnostics.world_query_ms);
  telemetry_.last_minimum_self_distance_m =
    collision_result.diagnostics.minimum_self_distance_m;
  telemetry_.last_minimum_world_distance_m =
    collision_result.diagnostics.minimum_world_distance_m;
  telemetry_.observeClearances(
    collision_result.diagnostics.minimum_self_distance_m,
    collision_result.diagnostics.minimum_world_distance_m);
  for (const auto & pair : collision_result.diagnostics.pairs) {
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
  telemetry_.last_closest_pair = collision_result.diagnostics.closest_pair;
  telemetry_.last_active_collision_rows =
    collision_result.diagnostics.active_distance_constraints;
  current_tick_context_.active_collision_rows = static_cast<std::uint32_t>(
    collision_result.diagnostics.active_distance_constraints);
  current_tick_context_.collision_distance_available = std::isfinite(
    collision_result.diagnostics.minimum_world_distance_m);
  if (current_tick_context_.collision_distance_available) {
    current_tick_context_.minimum_world_distance_m =
      collision_result.diagnostics.minimum_world_distance_m;
  }
  const std::size_t closest_length = std::min(
    collision_result.diagnostics.closest_pair.size(),
    sizeof(current_tick_context_.closest_pair) - 1U);
  std::copy_n(
    collision_result.diagnostics.closest_pair.data(), closest_length,
    current_tick_context_.closest_pair);
  current_tick_context_.closest_pair[closest_length] = '\0';
  for (const auto & pair : collision_result.diagnostics.pairs) {
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
    collision_result.diagnostics.outside_viability_contacts;
  if (!collision_result.diagnostics.valid) {
    RCLCPP_ERROR(
      get_logger(),
      "Collision constraint build failed: reason='%s', min_distance=%.6f m, pair='%s'",
      collision_result.diagnostics.failure_reason.c_str(),
      collision_result.diagnostics.minimum_distance_m,
      collision_result.diagnostics.closest_pair.c_str());
    enter_safety_wait(
      "Collision constraints unavailable: " + collision_result.diagnostics.failure_reason);
    return;
  }

  std::vector<control::LinearVelocityConstraint> constraints = collision_result.constraints;
  const bool segment_validation_required =
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
  current_tick_context_.target_used_for_reference =
    tracking_requested && reference.tracking.has_value();
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
  const auto j1_hold = j1_hold_constraint(reference, motion_state);
  if (j1_hold) {
    constraints.push_back(*j1_hold);
  }
  if (reference.tracking) {
    telemetry_.last_position_error_m = reference.tracking->position_error_m;
    telemetry_.last_screen_normal_pointing_error_rad = reference.tracking->pointing_error_rad;
    telemetry_.last_roll_error_rad = reference.tracking->roll_error_rad;
    (void)telemetry_.pointing_error_statistics.add(reference.tracking->pointing_error_rad);
  }
  if (reference.task && !reference.motion_reference_support_unavailable) {
    const auto & task = reference.task;
    control::QpResult tracking_result =
      solve_with_telemetry(*task, motion_state, constraints);
    if (!tracking_result.command_available()) {
      log_qp_failure(
        CommandAttempt::kTracking, tracking_result, motion_state, constraints,
        collision_result.diagnostics);
      mitigated_failure = "Tracking QP command is unavailable";
      if (reference.tracking) {
        motion_reference_->recordFaceCommand(
            *reference.tracking, tracking_result, current_time.seconds(), false,
            reference.actual_face_alignment);
      }
    } else {
      std::string candidate_failure;
      CandidateRejectionKind rejection_kind = CandidateRejectionKind::kOther;
      accepted_command = validated_candidate(
        robot_state, *state, motion_state, tracking_result.joint_velocity,
        *planning_scene, segment_validation_required, CommandAttempt::kTracking,
        candidate_failure, collision_result.diagnostics, rejection_kind);
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
        if (!retryable_reference_guard && reference.tracking) {
          motion_reference_->recordFaceCommand(
            *reference.tracking, tracking_result, current_time.seconds(), false,
            reference.actual_face_alignment);
        }
      }
    }
  }

  if (!accepted_command.has_value()) {
    const control::QpResult braking_result =
      brake_with_telemetry(motion_state, collision_result.constraints);
    if (!braking_result.command_available()) {
      log_qp_failure(
        CommandAttempt::kBraking, braking_result, motion_state,
        collision_result.constraints,
        collision_result.diagnostics);
      const auto continuation_status = tryContinuePublishedBrake(
        current_time, target, tracking_requested, collision_scene_revision);
      if (continuation_status != PublishedBrakeContinuationStatus::kContinued) {
        if (guard_rejected_tracking_result && reference.tracking) {
          motion_reference_->recordFaceCommand(
            *reference.tracking, *guard_rejected_tracking_result, current_time.seconds(),
            false, reference.actual_face_alignment);
        }
      }
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
      braking_candidate_failure, collision_result.diagnostics, rejection_kind);
    if (!accepted_command.has_value()) {
      const auto continuation_status = tryContinuePublishedBrake(
        current_time, target, tracking_requested, collision_scene_revision);
      if (continuation_status != PublishedBrakeContinuationStatus::kContinued) {
        if (guard_rejected_tracking_result && reference.tracking) {
          motion_reference_->recordFaceCommand(
            *reference.tracking, *guard_rejected_tracking_result, current_time.seconds(),
            false, reference.actual_face_alignment);
        }
      }
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
          accepted_command, accepted_qp_result);
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
            collision_result.diagnostics, accepted_command, accepted_qp_result);
        }
        if (reference_guard_recovery_accepted) {
          controller_mode_ = ControllerMode::kTracking;
          mitigated_failure =
            "Screen-edge inward recovery accepted by the full motion-reference guard";
        }
        if (!reference_guard_recovery_accepted) {
          motion_reference_->rejectPath();
          if (reference.tracking) {
            motion_reference_->recordFaceCommand(
              *reference.tracking, *guard_rejected_tracking_result, current_time.seconds(),
              false, reference.actual_face_alignment);
          }
        }
      }
    }
  }

  if (!commands_.startup_warmup_complete || !commands_.epoch_active) {
    if (reference.tracking) {
      motion_reference_->recordFaceCommand(
        *reference.tracking, accepted_qp_result, current_time.seconds(), false,
        reference.actual_face_alignment);
    }
    process_startup_cycle(*state, *robot_state, *planning_scene);
    return;
  }

  std::string publication_failure;
  const CommandPublicationStatus publication_status = enqueue_and_publish(
    *state, std::move(*accepted_command), current_time, publication_failure);
  if (publication_status != CommandPublicationStatus::kSuccess) {
    if (reference.tracking) {
      motion_reference_->recordFaceCommand(
        *reference.tracking, accepted_qp_result, current_time.seconds(), false,
        reference.actual_face_alignment);
    }
    if (publication_status == CommandPublicationStatus::kRecoverableCommandDeadline &&
      commands_.deferCommandDeadlineRetry())
    {
      // Drop this stale candidate. The next cycle may only resume from an
      // exact point in the already-published, validated trajectory suffix.
      return;
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
  telemetry_.observeCommandTransition(
    motion_state, commands_.queue.back(), parameters_.control_period_sec);
  commands_.recordHealthyPublication();
  if (reference.tracking) {
    motion_reference_->recordFaceCommand(
      *reference.tracking, accepted_qp_result, current_time.seconds(),
      controller_mode_ == ControllerMode::kTracking && !reference_guard_recovery_accepted,
      reference.actual_face_alignment);
    current_tick_context_.face_tracking_published =
      controller_mode_ == ControllerMode::kTracking && !reference_guard_recovery_accepted;
  }

  update_pause_after_publication();

  const bool collision_avoidance_active =
    collision_result.diagnostics.active_distance_constraints > 0U ||
    !accepted_qp_result.fully_solved();
  publish_control_status(
    collision_avoidance_active, reference_guard_blend_accepted,
    reference_guard_recovery_accepted, mitigated_failure, accepted_qp_result,
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
        "Continuing one exact sample from the validated published braking suffix");
      return PublishedBrakeContinuationStatus::kContinued;
    case PublishedBrakeContinuationStatus::kNotEligible:
      return reject_reference_path(PublishedBrakeContinuationStatus::kNotEligible);
    case PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix:
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
    task, state, motion_limits_, constraints);
  telemetry_.recordQpResult(start, result);
  control::QpResult handoff;
  handoff.primary_snapshot_disposition = result.primary_snapshot_disposition;
  handoff.secondary_snapshot_disposition = result.secondary_snapshot_disposition;
  handoff.primary_failure_snapshot = std::move(result.primary_failure_snapshot);
  handoff.secondary_failure_snapshot = std::move(result.secondary_failure_snapshot);
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
  control::QpResult result = velocity_qp_->brake(state, motion_limits_, constraints);
  telemetry_.recordQpResult(start, result);
  control::QpResult handoff;
  handoff.primary_snapshot_disposition = result.primary_snapshot_disposition;
  handoff.secondary_snapshot_disposition = result.secondary_snapshot_disposition;
  handoff.primary_failure_snapshot = std::move(result.primary_failure_snapshot);
  handoff.secondary_failure_snapshot = std::move(result.secondary_failure_snapshot);
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
