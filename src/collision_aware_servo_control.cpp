// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr double kStoppedAccelerationRadps2 = 2.0e-2;

constexpr std::chrono::milliseconds kTelemetryPublishPeriod{100};

}  // namespace

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

    const double cycle_duration_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - tick_start).count();
    (void)telemetry_.control_cycle_statistics.add(cycle_duration_ms);
    if (cycle_duration_ms > parameters_.control_period_sec * 1000.0) {
      ++telemetry_.missed_control_deadlines;
    }

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

void CollisionAwareServoComponent::control_tick()
{
  const rclcpp::Time current_time = now();
  const auto current_wall_time = std::chrono::steady_clock::now();
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
    ++telemetry_.rearm_timeout_count;
    enter_safety_wait(
      commands_.bootstrap_reset_ack_pending ?
      "Trajectory controller did not acknowledge the actual-state reset in time" :
      "Trajectory controller did not acknowledge the stationary bootstrap in time");
    return;
  }
  if (!state.has_value()) {
    if (requires_latched_halt) {
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
    enter_safety_wait(
      "PlanningScene collision geometry changed while commands were buffered");
    return;
  }
  last_validated_scene_revision_ = collision_scene_revision;
  const control::CollisionActualStateValidationResult actual_state_validation =
    collision_constraint_builder_->validateActualState(
    *planning_scene, *robot_state, *joint_model_group_);
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

  std::optional<double> actual_feedback_time_sec;
  if (actual_feedback.has_value() && actual_feedback_stamp.has_value()) {
    actual_feedback_time_sec = actual_feedback_stamp->seconds();
  }
  const auto * active_publication = commands_.activePublication();
  const auto * latest_publication = commands_.publications.latest();

  std::optional<ValidatedCommand> accepted_command;
  control::QpResult accepted_qp_result;
  std::optional<std::string> mitigated_failure;
  std::int8_t mitigated_status = moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION;

  const auto reference = motion_reference_->update(
    *robot_state, tracking_requested ? target : std::nullopt, *planning_scene,
    current_time.seconds(), collision_scene_revision, actual_feedback,
    actual_feedback_time_sec, motion_state, active_publication, latest_publication);
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
  if (reference.j1_hold_position_rad && reference.j1_joint_index &&
    *reference.j1_joint_index < static_cast<std::size_t>(motion_state.position.size()))
  {
    const auto index = static_cast<Eigen::Index>(*reference.j1_joint_index);
    control::detail::ScalarBrakingOptions braking_options;
    braking_options.period_sec = parameters_.control_period_sec;
    const auto braking_step = control::detail::scalar_braking_policy::nextStep(
      motion_state.velocity[index], motion_state.acceleration[index],
      control::detail::ScalarBrakingLimits{
        motion_limits_.max_velocity[index], motion_limits_.max_acceleration[index],
        motion_limits_.max_jerk[index]}, braking_options);
    if (braking_step.valid) {
      control::LinearVelocityConstraint hold;
      hold.coefficients = Eigen::VectorXd::Zero(motion_state.position.size());
      hold.coefficients[index] = 1.0;
      hold.lower_bound = braking_step.velocity;
      hold.upper_bound = braking_step.velocity;
      constraints.push_back(std::move(hold));
    }
  }
  if (reference.tracking) {
    telemetry_.last_position_error_m = reference.tracking->position_error_m;
    telemetry_.last_screen_normal_pointing_error_rad = reference.tracking->pointing_error_rad;
    telemetry_.last_roll_error_rad = reference.tracking->roll_error_rad;
    (void)telemetry_.pointing_error_statistics.add(reference.tracking->pointing_error_rad);
  }
  const auto try_continue_published_brake = [&]() {
      const rclcpp::Time decision_time = now();
      if (decision_time < current_time || !tracking_requested || !target.has_value() ||
        target->mode != msg::TrackingTarget::FACE ||
        !commands_.epoch_active ||
        !commands_.epoch_has_published || commands_.rearm_pending ||
        commands_.published_tail_rearm_pending || commands_.bootstrap_ack_pending ||
        !commands_.startup_warmup_complete ||
        !trajectory_controller_active_.load(std::memory_order_acquire))
      {
        return false;
      }
      const auto current_face_target = current_target(decision_time);
      if (!current_face_target.has_value() ||
        current_face_target->mode != msg::TrackingTarget::FACE)
      {
        return false;
      }
      const std::uint64_t current_scene_revision =
        planning_scene_collision_revision_.load(std::memory_order_acquire);
      if (current_scene_revision != collision_scene_revision ||
        current_scene_revision != last_validated_scene_revision_)
      {
        return false;
      }
      if (!accept_pose_commands_.load(std::memory_order_acquire) ||
        pause_requested_.load(std::memory_order_acquire) ||
        paused_.load(std::memory_order_acquire) ||
        halt_latched_.load(std::memory_order_acquire) ||
        stop_control_.load(std::memory_order_acquire) ||
        reset_queue_requested_.load(std::memory_order_acquire))
      {
        return false;
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
          return true;
        case PublishedBrakeContinuationStatus::kNotEligible:
          return false;
        case PublishedBrakeContinuationStatus::kExhaustedPublishedSuffix:
          enter_controlled_rearm(
            "Published braking continuation has no transmitted future sample: " +
            continuation_failure);
          return true;
        case PublishedBrakeContinuationStatus::kInvariantFailure:
          enter_safety_wait(
            "Published braking continuation violated its timeline invariant: " +
            continuation_failure);
          return true;
      }
      return false;
    };
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
        if (!transient_compact_stop_rejection) {
          motion_reference_->rejectPath();
        }
        mitigated_failure = "Tracking candidate rejected: " + candidate_failure;
        if (reference.tracking) {
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
      if (try_continue_published_brake()) {
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
      if (try_continue_published_brake()) {
        return;
      }
      enter_controlled_rearm(
        "Collision-safe braking candidate rejected: " + braking_candidate_failure);
      return;
    }
    accepted_qp_result = braking_result;
    controller_mode_ = ControllerMode::kBraking;
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
      controller_mode_ == ControllerMode::kTracking, reference.actual_face_alignment);
  }

  if (pause_requested_.load(std::memory_order_acquire) &&
    commands_.queue.back().velocities.cwiseAbs().maxCoeff() <= kStoppedVelocityRadps &&
    commands_.queue.back().accelerations.cwiseAbs().maxCoeff() <=
    kStoppedAccelerationRadps2)
  {
    if (pause_terminal_pending_.exchange(true, std::memory_order_acq_rel)) {
      paused_.store(true, std::memory_order_release);
      commands_.discardBranch();
      commands_.rearm_pending = false;
      commands_.published_tail_rearm_pending = false;
      commands_.restartSettling();
      commands_.clearRearmDeadlines();
      pause_terminal_pending_.store(false, std::memory_order_release);
    } else {
      // composeTrajectoryMessage intentionally withholds queue.back(). One
      // additional stopped point makes the first terminal point visible to JTC.
    }
  } else {
    pause_terminal_pending_.store(false, std::memory_order_release);
  }

  const bool collision_avoidance_active =
    collision_result.diagnostics.active_distance_constraints > 0U ||
    !accepted_qp_result.fully_solved();
  if (mitigated_failure.has_value()) {
    publish_status(
      mitigated_status, *mitigated_failure + "; safe jerk-limited braking command published");
  } else if (collision_avoidance_active) {
    publish_status(
      moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION,
      accepted_qp_result.fully_solved() ? "Collision avoidance constraint active" :
      "Safe primary orientation command only");
  } else {
    publish_status(moveit_msgs::msg::ServoStatus::NO_WARNING, "No warnings");
  }
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
  return result;
}

control::QpResult CollisionAwareServoComponent::brake_with_telemetry(
  const control::JointMotionState & state,
  const std::vector<control::LinearVelocityConstraint> & constraints)
{
  const auto start = std::chrono::steady_clock::now();
  control::QpResult result = velocity_qp_->brake(state, motion_limits_, constraints);
  telemetry_.recordQpResult(start, result);
  return result;
}

}  // namespace face_tracking_arm
