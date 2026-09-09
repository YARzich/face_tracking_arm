// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

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
        enter_latched_halt(
          std::string("Collision-aware control exception: ") + error.what());
      }
    }

    const auto tick_finished = std::chrono::steady_clock::now();
    const double tick_duration_ms = std::chrono::duration<double, std::milli>(
      tick_finished - tick_start).count();
    (void)control_tick_statistics_.add(tick_duration_ms);
    lifetime_maximum_control_tick_ms_ = std::max(
      lifetime_maximum_control_tick_ms_, tick_duration_ms);
    observe_controller_mode();

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
    (void)control_cycle_statistics_.add(cycle_duration_ms);
    if (cycle_duration_ms > parameters_.control_period_sec * 1000.0) {
      ++missed_control_deadlines_;
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
    control_time_grid_->observe(current_time.nanoseconds());
  if (time_observation.status == control::ControlTimeStatus::kRewind) {
    ++clock_rewind_count_;
    motion_reference_->reset();
    command_queue_.clear();
    published_trajectory_history_.clear();
    active_publication_id_.reset();
    control_time_grid_->resetCommandEpoch();
    command_epoch_active_ = false;
    command_epoch_has_published_ = false;
    clear_bootstrap_ack_state();
    rearm_pending_ = false;
    published_tail_rearm_pending_ = false;
    stable_rearm_samples_ = 0;
    last_rearm_sample_time_.reset();
    clear_rearm_deadlines();
    last_feedback_sample_time_.reset();
    last_feedback_velocity_.reset();
    last_feedback_acceleration_.reset();
    last_feedback_jerk_.reset();
    initial_state_received_ = false;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_pose_.reset();
    }
    {
      std::lock_guard<std::mutex> lock(controller_state_mutex_);
      latest_controller_state_.reset();
    }
    controller_mode_ = ControllerMode::kLatchedHalt;
    halt_latched_.store(true, std::memory_order_release);
    latched_halt_reason_ =
      "ROS time rewound; the published safety timeline is invalid and requires re-arm";
    publish_status(
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION, latched_halt_reason_);
    return;
  }
  const bool rearm_requested = reset_queue_requested_.exchange(
    false, std::memory_order_acq_rel);
  if (rearm_requested) {
    motion_reference_->reset();
    const auto * published_record = active_published_record();
    published_tail_rearm_pending_ = published_record != nullptr &&
      current_time < published_record->stationary_time;
    command_queue_.clear();
    control_time_grid_->resetCommandEpoch();
    command_epoch_active_ = false;
    command_epoch_has_published_ = false;
    clear_bootstrap_ack_state();
    rearm_pending_ = true;
    stable_rearm_samples_ = 0;
    last_rearm_sample_time_.reset();
    clear_rearm_deadlines();
    consecutive_automatic_rearms_ = 0U;
    healthy_publications_since_rearm_ = 0U;
    reset_feedback_derivative_history();
    if (controller_mode_ != ControllerMode::kLatchedHalt) {
      controller_mode_ = ControllerMode::kBraking;
    }
  }
  if (rearm_pending_) {
    ensure_rearm_deadline(current_time);
  }
  if (time_observation.status == control::ControlTimeStatus::kNotAdvancing) {
    ++clock_not_advancing_observations_;
    if ((rearm_pending_ || bootstrap_ack_pending_) &&
      rearm_deadline_expired(current_time, current_wall_time))
    {
      ++rearm_timeout_count_;
      enter_latched_halt(
        bootstrap_ack_pending_ ?
        "Trajectory controller bootstrap acknowledgement timed out while ROS time was frozen" :
        "Safe re-arm timed out while ROS time was frozen");
      return;
    }
    publish_status(
      controller_mode_ == ControllerMode::kLatchedHalt ?
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION :
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      controller_mode_ == ControllerMode::kLatchedHalt ?
      (latched_halt_reason_.empty() ? "Collision-aware Servo is latched" :
      latched_halt_reason_) : "ROS time is not advancing");
    return;
  }
  if (time_observation.status == control::ControlTimeStatus::kWait) {
    if ((rearm_pending_ || bootstrap_ack_pending_) &&
      rearm_deadline_expired(current_time, current_wall_time))
    {
      ++rearm_timeout_count_;
      enter_latched_halt(
        bootstrap_ack_pending_ ?
        "Trajectory controller bootstrap acknowledgement timed out in wall time" :
        "Safe re-arm timed out in wall time");
    }
    return;
  }
  skipped_control_periods_ += time_observation.skipped_periods;

  reset_tick_telemetry();
  if (!rearm_pending_ && controller_mode_ == ControllerMode::kLatchedHalt) {
    publish_status(
      moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION,
      latched_halt_reason_.empty() ? "Collision-aware Servo is latched" :
      latched_halt_reason_);
    return;
  }
  if (paused_.load(std::memory_order_acquire) &&
    pause_requested_.load(std::memory_order_acquire))
  {
    publish_status(moveit_msgs::msg::ServoStatus::NO_WARNING, "Servoing paused");
    return;
  }

  if (trajectory_publisher_->get_subscription_count() == 0U ||
    !trajectory_controller_active_.load(std::memory_order_acquire))
  {
    if (rearm_pending_) {
      if (published_tail_rearm_pending_) {
        enter_latched_halt(
          "Trajectory controller was lost while a published braking tail was in flight");
      } else if (rearm_deadline_expired(current_time, current_wall_time)) {
        ++rearm_timeout_count_;
        enter_latched_halt(
          "Safe re-arm timed out waiting for an active subscribed trajectory controller");
      } else {
        publish_status(
          halt_latched_.load(std::memory_order_acquire) ?
          moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION :
          moveit_msgs::msg::ServoStatus::NO_WARNING,
          "Safe re-arm is waiting for an active subscribed trajectory controller");
      }
    } else {
      enter_latched_halt("Trajectory controller is not active and subscribed");
    }
    return;
  }

  if (rearm_pending_) {
    if (published_tail_rearm_pending_ &&
      planning_scene_collision_revision_.load(std::memory_order_acquire) !=
      last_validated_scene_revision_)
    {
      ++planning_scene_invalidation_count_;
      enter_latched_halt(
        "PlanningScene collision geometry changed while a published braking tail was in flight");
      return;
    }
    std::string rearm_reason;
    const RearmReadiness rearm_readiness = safe_rearm_ready(
      current_time, rearm_reason);
    if (rearm_readiness == RearmReadiness::kUnsafe) {
      enter_latched_halt(std::move(rearm_reason));
      return;
    }
    if (rearm_readiness == RearmReadiness::kWaiting) {
      if (rearm_deadline_expired(current_time, current_wall_time)) {
        ++rearm_timeout_count_;
        enter_latched_halt(
          "Safe re-arm timed out after its bounded settle window: " + rearm_reason);
        return;
      }
      publish_status(
        halt_latched_.load(std::memory_order_acquire) ?
        moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION :
        (published_tail_rearm_pending_ ?
        moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION :
        moveit_msgs::msg::ServoStatus::NO_WARNING),
        std::move(rearm_reason));
      return;
    }
    rearm_pending_ = false;
    published_tail_rearm_pending_ = false;
    stable_rearm_samples_ = 0;
    last_rearm_sample_time_.reset();
    published_trajectory_history_.clear();
    active_publication_id_.reset();
    clear_bootstrap_ack_state();
    halt_latched_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
    controller_mode_ = ControllerMode::kBraking;
    latched_halt_reason_.clear();
  }

  std::string timeline_recovery_failure;
  const TimelineRecoveryStatus timeline_recovery_status = recover_published_timeline(
    time_observation.skipped_periods, current_time, timeline_recovery_failure);
  if (timeline_recovery_status != TimelineRecoveryStatus::kSuccess) {
    ++timeline_recovery_failures_;
    std::string message = "Published command timeline recovery failed: " +
      timeline_recovery_failure;
    if (timeline_recovery_status == TimelineRecoveryStatus::kExhaustedPublishedSuffix) {
      enter_controlled_rearm(std::move(message));
    } else {
      enter_latched_halt(std::move(message));
    }
    return;
  }

  std::string state_failure_reason;
  bool requires_latched_halt = false;
  const auto state = predicted_or_measured_state(
    current_time, state_failure_reason, requires_latched_halt);
  if (bootstrap_ack_pending_ &&
    rearm_deadline_expired(current_time, current_wall_time))
  {
    ++rearm_timeout_count_;
    enter_latched_halt(
      "Trajectory controller did not acknowledge the stationary bootstrap in time");
    return;
  }
  if (!state.has_value()) {
    if (requires_latched_halt) {
      enter_latched_halt(std::move(state_failure_reason));
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
  if (command_epoch_active_ && collision_scene_revision != last_validated_scene_revision_) {
    ++planning_scene_invalidation_count_;
    enter_latched_halt(
      "PlanningScene collision geometry changed while commands were buffered");
    return;
  }
  last_validated_scene_revision_ = collision_scene_revision;
  const control::CollisionActualStateValidationResult actual_state_validation =
    collision_constraint_builder_->validateActualState(
    *planning_scene, *robot_state, *joint_model_group_);
  ++actual_state_validation_count_;
  last_actual_minimum_self_distance_m_ =
    std::isfinite(actual_state_validation.minimum_self_distance_m) ?
    std::optional<double>{actual_state_validation.minimum_self_distance_m} : std::nullopt;
  last_actual_minimum_world_distance_m_ =
    std::isfinite(actual_state_validation.minimum_world_distance_m) ?
    std::optional<double>{actual_state_validation.minimum_world_distance_m} : std::nullopt;
  last_actual_closest_pair_ = actual_state_validation.closest_pair.empty() ?
    "outside hard-clearance query" : actual_state_validation.closest_pair;
  observe_clearances(
    actual_state_validation.minimum_self_distance_m,
    actual_state_validation.minimum_world_distance_m);
  if (!actual_state_validation.input_valid || actual_state_validation.unsafe) {
    ++actual_state_validation_failure_count_;
    const std::string reason = !actual_state_validation.input_valid ?
      actual_state_validation.failure_reason :
      "hard-clearance/corridor violation for " + last_actual_closest_pair_;
    enter_latched_halt("Measured robot state is unsafe: " + reason);
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
  observe_motion_state(motion_state);

  const auto collision_build_start = std::chrono::steady_clock::now();
  const control::CollisionConstraintResult collision_result =
    collision_constraint_builder_->build(
    *planning_scene, *robot_state, *joint_model_group_, motion_state, motion_limits_,
    parameters_.control_period_sec);
  (void)collision_build_statistics_.add(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - collision_build_start).count());
  (void)collision_self_query_statistics_.add(collision_result.diagnostics.self_query_ms);
  (void)collision_world_query_statistics_.add(collision_result.diagnostics.world_query_ms);
  last_minimum_self_distance_m_ =
    collision_result.diagnostics.minimum_self_distance_m;
  last_minimum_world_distance_m_ =
    collision_result.diagnostics.minimum_world_distance_m;
  observe_clearances(
    collision_result.diagnostics.minimum_self_distance_m,
    collision_result.diagnostics.minimum_world_distance_m);
  for (const auto & pair : collision_result.diagnostics.pairs) {
    const double headroom = pair.distance_m - pair.required_clearance_m;
    const std::string pair_name = pair.first_body + " <-> " + pair.second_body;
    if (std::isfinite(headroom) &&
      (!last_minimum_robust_headroom_m_.has_value() ||
      headroom < *last_minimum_robust_headroom_m_))
    {
      last_minimum_robust_headroom_m_ = headroom;
      last_minimum_robust_headroom_pair_ = pair_name;
    }
    if (std::isfinite(headroom) &&
      (!minimum_observed_robust_headroom_m_.has_value() ||
      headroom < *minimum_observed_robust_headroom_m_))
    {
      minimum_observed_robust_headroom_m_ = headroom;
      minimum_observed_robust_headroom_pair_ = pair_name;
    }
  }
  last_closest_pair_ = collision_result.diagnostics.closest_pair;
  last_active_collision_rows_ =
    collision_result.diagnostics.active_distance_constraints;
  last_outside_collision_viability_contacts_ =
    collision_result.diagnostics.outside_viability_contacts;
  if (!collision_result.diagnostics.valid) {
    RCLCPP_ERROR(
      get_logger(),
      "Collision constraint build failed: reason='%s', min_distance=%.6f m, pair='%s'",
      collision_result.diagnostics.failure_reason.c_str(),
      collision_result.diagnostics.minimum_distance_m,
      collision_result.diagnostics.closest_pair.c_str());
    enter_latched_halt(
      "Collision constraints unavailable: " + collision_result.diagnostics.failure_reason);
    return;
  }

  const std::vector<control::LinearVelocityConstraint> & constraints =
    collision_result.constraints;
  const double condition_number = 0.0;
  const bool segment_validation_required =
    collision_constraint_builder_->shouldValidateSegment(
    collision_result.diagnostics, parameters_.segment_validation_distance_m);
  // Collision evaluation can take several milliseconds while the ROS
  // executor keeps receiving newer latest-only targets. Compare their stamp
  // with the clock at consumption time, not with the stale tick-start time.
  const auto target = current_target(now());
  const bool tracking_requested = target.has_value() &&
    accept_pose_commands_.load(std::memory_order_acquire) &&
    !pause_requested_.load(std::memory_order_acquire);

  std::optional<ValidatedCommand> accepted_command;
  control::QpResult accepted_qp_result;
  std::optional<std::string> mitigated_failure;
  std::int8_t mitigated_status = moveit_msgs::msg::ServoStatus::DECELERATE_FOR_COLLISION;

  const auto reference = motion_reference_->update(
    *robot_state, tracking_requested ? target : std::nullopt, *planning_scene,
    current_time.seconds(), collision_scene_revision);
  if (reference.tracking) {
    last_position_error_m_ = reference.tracking->position_error_m;
    last_screen_normal_pointing_error_rad_ = reference.tracking->pointing_error_rad;
    last_roll_error_rad_ = reference.tracking->roll_error_rad;
    (void)pointing_error_statistics_.add(reference.tracking->pointing_error_rad);
  }
  if (reference.task) {
    const auto & task = reference.task;
    if (!task.has_value()) {
      ++task_failure_count_;
      mitigated_failure = "Tracking task construction failed";
    } else {
      control::QpResult tracking_result =
        solve_with_telemetry(*task, motion_state, constraints);
      if (!tracking_result.command_available()) {
        log_qp_failure(
          CommandAttempt::kTracking, tracking_result, motion_state, constraints,
          collision_result.diagnostics, condition_number);
        mitigated_failure = "Tracking QP command is unavailable";
      } else {
        std::string candidate_failure;
        accepted_command = validated_candidate(
          robot_state, *state, motion_state, tracking_result.joint_velocity,
          *planning_scene, segment_validation_required, CommandAttempt::kTracking,
          candidate_failure, reference.follows_path ? &collision_result.diagnostics : nullptr);
        if (accepted_command.has_value()) {
          accepted_qp_result = tracking_result;
          controller_mode_ = ControllerMode::kTracking;
        } else {
          motion_reference_->rejectPath();
          mitigated_failure = "Tracking candidate rejected: " + candidate_failure;
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
        collision_result.diagnostics, condition_number);
      enter_controlled_rearm("No feasible jerk-limited collision-safe braking command");
      return;
    }

    std::string braking_candidate_failure;
    accepted_command = validated_candidate(
      robot_state, *state, motion_state, braking_result.joint_velocity,
      *planning_scene, segment_validation_required, CommandAttempt::kBraking,
      braking_candidate_failure, reference.follows_path ? &collision_result.diagnostics : nullptr);
    if (!accepted_command.has_value()) {
      enter_controlled_rearm(
        "Collision-safe braking candidate rejected: " + braking_candidate_failure);
      return;
    }
    accepted_qp_result = braking_result;
    controller_mode_ = ControllerMode::kBraking;
  }

  if (!startup_warmup_complete_ || !command_epoch_active_) {
    process_startup_cycle(*state, *robot_state, *planning_scene);
    return;
  }

  observe_command_transition(motion_state, accepted_command->next_state);
  std::string publication_failure;
  const CommandPublicationStatus publication_status = enqueue_and_publish(
    *state, std::move(*accepted_command), current_time, publication_failure);
  if (publication_status != CommandPublicationStatus::kSuccess) {
    if (publication_status == CommandPublicationStatus::kRecoverableTimingFailure) {
      enter_controlled_rearm(std::move(publication_failure));
    } else {
      enter_latched_halt(std::move(publication_failure));
    }
    return;
  }
  record_healthy_publication();

  if (pause_requested_.load(std::memory_order_acquire) &&
    command_queue_.back().velocities.cwiseAbs().maxCoeff() <= kStoppedVelocityRadps &&
    command_queue_.back().accelerations.cwiseAbs().maxCoeff() <=
    kStoppedAccelerationRadps2)
  {
    if (pause_terminal_pending_.exchange(true, std::memory_order_acq_rel)) {
      paused_.store(true, std::memory_order_release);
      command_queue_.clear();
      control_time_grid_->resetCommandEpoch();
      command_epoch_active_ = false;
      command_epoch_has_published_ = false;
      clear_bootstrap_ack_state();
      rearm_pending_ = false;
      published_tail_rearm_pending_ = false;
      stable_rearm_samples_ = 0;
      last_rearm_sample_time_.reset();
      clear_rearm_deadlines();
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
  record_qp_result(start, result);
  return result;
}

control::QpResult CollisionAwareServoComponent::brake_with_telemetry(
  const control::JointMotionState & state,
  const std::vector<control::LinearVelocityConstraint> & constraints)
{
  const auto start = std::chrono::steady_clock::now();
  control::QpResult result = velocity_qp_->brake(state, motion_limits_, constraints);
  record_qp_result(start, result);
  return result;
}

}  // namespace face_tracking_arm
