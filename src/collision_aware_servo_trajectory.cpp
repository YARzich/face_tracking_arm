// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr std::size_t kRequiredStartupWarmupCycles = 5;

control::JointMotionState to_motion_state(const moveit_servo::KinematicState & state)
{
  control::JointMotionState result;
  result.position = state.positions;
  result.velocity = state.velocities;
  result.acceleration = state.accelerations;
  return result;
}

}  // namespace

const control::PublishedTrajectoryRecord * CollisionAwareServoComponent::active_published_record()
const noexcept
{
  return active_publication_id_.has_value() ?
         published_trajectory_history_.find(*active_publication_id_) : nullptr;
}

std::optional<ExpectedCommandState> CollisionAwareServoComponent::expected_command_state_at(
  const rclcpp::Time & sample_time) const
{
  const auto * published_record = active_published_record();
  if (published_record == nullptr ||
    published_record->execution_queue.size() < 3U ||
    (!command_epoch_has_published_ && !published_tail_rearm_pending_))
  {
    return std::nullopt;
  }
  const auto & queue = published_record->execution_queue;

  // composeTrajectoryMessage intentionally withholds the final queue entry.
  const auto last_state = std::prev(queue.end(), 2);
  if (sample_time < queue.front().time_stamp ||
    sample_time > last_state->time_stamp)
  {
    return std::nullopt;
  }

  auto end = std::find_if(
    queue.begin(), std::next(last_state),
    [&sample_time](const moveit_servo::KinematicState & state) {
      return state.time_stamp >= sample_time;
    });
  if (end == queue.begin()) {
    ++end;
  } else if (end == std::next(last_state)) {
    end = last_state;
  }
  const auto start = std::prev(end);
  const double duration_sec = (end->time_stamp - start->time_stamp).seconds();
  if (!std::isfinite(duration_sec) || duration_sec <= 0.0 ||
    start->positions.size() != end->positions.size())
  {
    return std::nullopt;
  }
  const double elapsed_sec = (sample_time - start->time_stamp).seconds();
  const double ratio = std::clamp(elapsed_sec / duration_sec, 0.0, 1.0);

  ExpectedCommandState expected;
  expected.positions = start->positions + ratio * (end->positions - start->positions);
  expected.velocities = (end->positions - start->positions) / duration_sec;
  if (!expected.positions.allFinite() || !expected.velocities.allFinite()) {
    return std::nullopt;
  }
  return expected;
}

TimelineRecoveryStatus CollisionAwareServoComponent::recover_published_timeline(
  const std::uint64_t skipped_periods, const rclcpp::Time & current_time,
  std::string & failure_reason)
{
  if (!command_epoch_active_ || skipped_periods == 0U) {
    return TimelineRecoveryStatus::kSuccess;
  }
  const auto * published_record = active_published_record();
  if (!command_epoch_has_published_ || command_queue_.empty() ||
    published_record == nullptr || published_record->execution_queue.size() < 3U)
  {
    failure_reason = "active epoch has no recoverable published suffix";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const rclcpp::Duration retained_history = rclcpp::Duration::from_seconds(
    servo_parameters_.max_expected_latency);
  while (command_queue_.size() > 1U &&
    command_queue_.front().time_stamp < current_time - retained_history)
  {
    command_queue_.pop_front();
  }

  // The final execution-queue entry is intentionally withheld by MoveIt
  // Servo's trajectory composer, so it must never become a recovery anchor.
  const auto & published_queue = published_record->execution_queue;
  const auto transmitted_end = std::prev(published_queue.end());
  const auto current_branch = std::find_if(
    published_queue.begin(), transmitted_end,
    [this](const moveit_servo::KinematicState & state) {
      return state.time_stamp == command_queue_.back().time_stamp;
    });
  if (current_branch == transmitted_end ||
    !current_branch->positions.isApprox(command_queue_.back().positions, kFiniteEpsilon) ||
    !current_branch->velocities.isApprox(command_queue_.back().velocities, kFiniteEpsilon) ||
    !current_branch->accelerations.isApprox(
      command_queue_.back().accelerations, kFiniteEpsilon))
  {
    failure_reason = "current branch is not an exact prefix of the transmitted trajectory";
    return TimelineRecoveryStatus::kInvariantFailure;
  }

  const auto available_periods = static_cast<std::uint64_t>(
    std::distance(std::next(current_branch), transmitted_end));
  if (skipped_periods > available_periods) {
    failure_reason = "transmitted braking suffix is too short for the delayed callback";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }
  const auto recovery_anchor = std::next(
    current_branch, static_cast<std::ptrdiff_t>(skipped_periods));

  const std::size_t recovered_periods = static_cast<std::size_t>(skipped_periods);
  // Retained past points are useful for matching a replacement, but they are
  // not part of the future safety horizon. Drop only the oldest history when
  // a delayed callback needs room for already-published future samples.
  while (command_queue_.size() > 1U &&
    command_queue_.size() + recovered_periods > max_command_queue_points_)
  {
    command_queue_.pop_front();
  }
  if (command_queue_.size() + recovered_periods > max_command_queue_points_) {
    failure_reason = "published recovery span exceeds the command queue hard bound";
    return TimelineRecoveryStatus::kExhaustedPublishedSuffix;
  }

  auto previous_stamp = current_branch->time_stamp;
  for (auto point = std::next(current_branch); point != std::next(recovery_anchor); ++point) {
    if (!finite_state(*point, parameters_.joint_names.size()) ||
      point->time_stamp - previous_stamp !=
      rclcpp::Duration::from_nanoseconds(control_time_grid_->periodNs()))
    {
      failure_reason = "published recovery prefix left the fixed command grid";
      return TimelineRecoveryStatus::kInvariantFailure;
    }
    previous_stamp = point->time_stamp;
  }
  if (!control_time_grid_->adoptPublishedStamp(recovery_anchor->time_stamp.nanoseconds())) {
    failure_reason = "recovery anchor was not on the active command grid";
    return TimelineRecoveryStatus::kInvariantFailure;
  }
  command_queue_.insert(
    command_queue_.end(), std::next(current_branch), std::next(recovery_anchor));
  ++timeline_recovery_events_;
  timeline_recovered_periods_ += recovered_periods;
  return TimelineRecoveryStatus::kSuccess;
}

std::optional<BootstrapCommand> CollisionAwareServoComponent::make_stationary_bootstrap(
  const moveit_servo::KinematicState & measured_state,
  const moveit::core::RobotState & robot_state,
  const planning_scene::PlanningScene & planning_scene,
  std::string & failure_reason, BootstrapFailureKind & failure_kind)
{
  failure_kind = BootstrapFailureKind::kSafetyOrInvariant;
  const auto controller_state = current_controller_state();
  if (!controller_state.has_value()) {
    failure_kind = BootstrapFailureKind::kRetryableState;
    failure_reason = "fresh controller state is unavailable for stationary bootstrap";
    return std::nullopt;
  }

  const auto maximum_absolute = [](const std::vector<double> & values) {
      double maximum = 0.0;
      for (const double value : values) {
        maximum = std::max(maximum, std::abs(value));
      }
      return maximum;
    };
  const auto maximum_difference = [](
    const std::vector<double> & first, const std::vector<double> & second) {
      double maximum = 0.0;
      for (std::size_t index = 0; index < first.size(); ++index) {
        maximum = std::max(maximum, std::abs(first[index] - second[index]));
      }
      return maximum;
    };
  if (std::abs(controller_state->speed_scaling_factor - 1.0) >
    kControllerSpeedScalingTolerance)
  {
    failure_reason = "trajectory controller speed scaling changed before bootstrap";
    return std::nullopt;
  }
  if (maximum_difference(
      controller_state->reference.positions,
      controller_state->output.positions) > kControllerCommandAlignmentToleranceRad ||
    maximum_difference(
      controller_state->reference.positions,
      controller_state->feedback.positions) >
    parameters_.collision_tracking_error_bound_rad ||
    maximum_absolute(controller_state->reference.velocities) > kStoppedVelocityRadps ||
    maximum_absolute(controller_state->feedback.velocities) > kStoppedVelocityRadps)
  {
    failure_kind = BootstrapFailureKind::kRetryableState;
    failure_reason = "trajectory controller left its settled hold before bootstrap";
    return std::nullopt;
  }

  moveit_servo::KinematicState stationary_state(
    static_cast<int>(parameters_.joint_names.size()));
  stationary_state.joint_names = parameters_.joint_names;
  stationary_state.positions = Eigen::Map<const Eigen::VectorXd>(
    controller_state->reference.positions.data(),
    static_cast<Eigen::Index>(controller_state->reference.positions.size()));
  stationary_state.velocities.setZero();
  stationary_state.accelerations.setZero();

  const control::JointMotionState measured_motion_state = to_motion_state(measured_state);
  const control::JointMotionState stationary_motion_state = to_motion_state(stationary_state);
  ++segment_validation_count_;
  ++bootstrap_segment_validation_count_;
  const control::CollisionSegmentValidationResult bootstrap_segment =
    collision_constraint_builder_->validateSegment(
    planning_scene, robot_state, *joint_model_group_, measured_motion_state,
    stationary_motion_state,
    static_cast<std::size_t>(parameters_.segment_validation_substeps));
  observe_clearances(
    bootstrap_segment.minimum_self_distance_m,
    bootstrap_segment.minimum_world_distance_m);
  if (!bootstrap_segment.input_valid || bootstrap_segment.unsafe) {
    record_segment_rejection(CommandAttempt::kBraking, bootstrap_segment);
    failure_reason = "stationary bootstrap rejected: " +
      last_segment_rejection_reason_;
    return std::nullopt;
  }

  control::EmergencyBrakeTail stationary_tail =
    emergency_brake_tail_generator_->generate(stationary_motion_state, motion_limits_);
  if (!stationary_tail.command_available()) {
    ++braking_tail_generation_failure_count_;
    failure_reason = "stationary bootstrap tail is unavailable: " +
      stationary_tail.failure_reason;
    return std::nullopt;
  }

  BootstrapCommand bootstrap;
  bootstrap.seed_state = stationary_state;
  bootstrap.first_command.next_state = std::move(stationary_state);
  bootstrap.first_command.braking_tail = std::move(stationary_tail);
  return bootstrap;
}

std::optional<ValidatedCommand> CollisionAwareServoComponent::validated_candidate(
  const moveit::core::RobotStatePtr & robot_state,
  const moveit_servo::KinematicState & current_state,
  const control::JointMotionState & current_motion_state,
  const Eigen::VectorXd & joint_velocity,
  const planning_scene::PlanningScene & planning_scene,
  const bool segment_validation_required,
  const CommandAttempt attempt,
  std::string & failure_reason,
  const control::CollisionConstraintDiagnostics * path_clearance)
{
  auto next_state = make_next_state(
    robot_state, current_state, joint_velocity, failure_reason);
  if (!next_state.has_value()) {
    ++moveit_validation_failure_count_;
    RCLCPP_WARN(
      get_logger(), "%s MoveIt Servo validation rejected the command: %s",
      command_attempt_name(attempt), failure_reason.c_str());
    return std::nullopt;
  }

  const control::JointMotionState candidate_motion_state = to_motion_state(*next_state);
  // The QP already enforces a jerk-aware stopping envelope at every control
  // tick. Re-running mesh FCL over the candidate and its complete braking
  // suffix in free space used more than one 10 ms period and starved the JTC
  // command buffer. Keep the expensive sampled check as a last-line guard at
  // the configured hard-boundary activation distance only.
  const bool path_requires_sampling = path_clearance &&
    !collision_constraint_builder_->canCertifySegment(
    *path_clearance, current_motion_state.position, candidate_motion_state.position);
  if (segment_validation_required || path_requires_sampling) {
    ++segment_validation_count_;
    const auto segment_validation_start = std::chrono::steady_clock::now();
    const control::CollisionSegmentValidationResult segment_result =
      collision_constraint_builder_->validateSegment(
      planning_scene, *robot_state, *joint_model_group_, current_motion_state,
      candidate_motion_state,
      static_cast<std::size_t>(parameters_.segment_validation_substeps));
    (void)segment_validation_statistics_.add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - segment_validation_start).count());
    observe_clearances(
      segment_result.minimum_self_distance_m, segment_result.minimum_world_distance_m);
    if (!segment_result.input_valid || segment_result.unsafe) {
      record_segment_rejection(attempt, segment_result);
      failure_reason = last_segment_rejection_reason_;
      return std::nullopt;
    }
  }

  control::EmergencyBrakeTail braking_tail =
    emergency_brake_tail_generator_->generate(candidate_motion_state, motion_limits_);
  if (!braking_tail.command_available()) {
    ++braking_tail_generation_failure_count_;
    RCLCPP_WARN(
      get_logger(), "%s braking-tail candidate rejected: %s",
      command_attempt_name(attempt), braking_tail.failure_reason.c_str());
    for (Eigen::Index index = 0; index < candidate_motion_state.position.size(); ++index) {
      RCLCPP_WARN(
        get_logger(),
        "Rejected tail state %s: q=%.9f, qdot=%.9f, qddot=%.9f, "
        "limits=[qdot %.9f, qddot %.9f, jerk %.9f]",
        parameters_.joint_names[static_cast<std::size_t>(index)].c_str(),
        candidate_motion_state.position[index], candidate_motion_state.velocity[index],
        candidate_motion_state.acceleration[index], motion_limits_.max_velocity[index],
        motion_limits_.max_acceleration[index], motion_limits_.max_jerk[index]);
    }
    failure_reason = "pre-published braking tail is unavailable: " +
      braking_tail.failure_reason;
    return std::nullopt;
  }

  if (segment_validation_required) {
    moveit::core::RobotState tail_robot_state(*robot_state);
    const auto & tail_start = braking_tail.points.front();
    tail_robot_state.setJointGroupPositions(joint_model_group_, tail_start.position);
    tail_robot_state.setJointGroupVelocities(joint_model_group_, tail_start.velocity);
    tail_robot_state.setJointGroupAccelerations(
      joint_model_group_, tail_start.acceleration);
    tail_robot_state.update();
    const std::size_t tail_segment_count = braking_tail.points.size() - 1U;
    segment_validation_count_ += tail_segment_count;
    braking_tail_segment_validation_count_ += tail_segment_count;
    const auto path_validation_start = std::chrono::steady_clock::now();
    const control::CollisionSegmentValidationResult tail_path =
      collision_constraint_builder_->validatePath(
      planning_scene, tail_robot_state, *joint_model_group_, braking_tail.points,
      static_cast<std::size_t>(parameters_.segment_validation_substeps));
    (void)path_validation_statistics_.add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - path_validation_start).count());
    observe_clearances(tail_path.minimum_self_distance_m, tail_path.minimum_world_distance_m);
    if (!tail_path.input_valid || tail_path.unsafe) {
      ++braking_tail_collision_rejection_count_;
      record_segment_rejection(attempt, tail_path);
      failure_reason = "pre-published braking tail rejected: " +
        last_segment_rejection_reason_;
      return std::nullopt;
    }
  }

  return ValidatedCommand{std::move(*next_state), std::move(braking_tail)};
}

std::optional<moveit_servo::KinematicState> CollisionAwareServoComponent::make_next_state(
  const moveit::core::RobotStatePtr & robot_state,
  const moveit_servo::KinematicState & current_state,
  const Eigen::VectorXd & joint_velocity,
  std::string & failure_reason)
{
  if (joint_velocity.size() != static_cast<Eigen::Index>(parameters_.joint_names.size()) ||
    !joint_velocity.allFinite())
  {
    failure_reason = "joint velocity has invalid dimensions or non-finite values";
    return std::nullopt;
  }
  moveit_servo::KinematicState next_state = current_state;
  next_state.joint_names = parameters_.joint_names;
  next_state.positions = current_state.positions +
    joint_velocity * parameters_.control_period_sec;
  next_state.velocities = joint_velocity;
  next_state.accelerations =
    (joint_velocity - current_state.velocities) / parameters_.control_period_sec;
  if (!finite_state(next_state, parameters_.joint_names.size())) {
    failure_reason = "direct fixed-period integration produced an invalid state";
    return std::nullopt;
  }

  const double tolerance = 2.0 * velocity_qp_->config().solution_feasibility_tolerance;
  const Eigen::VectorXd safe_lower =
    motion_limits_.lower_position;
  const Eigen::VectorXd safe_upper =
    motion_limits_.upper_position;
  if ((next_state.positions.array() < safe_lower.array() - tolerance).any() ||
    (next_state.positions.array() > safe_upper.array() + tolerance).any() ||
    (next_state.velocities.cwiseAbs().array() >
    motion_limits_.max_velocity.array() + tolerance).any() ||
    (next_state.accelerations.cwiseAbs().array() >
    motion_limits_.max_acceleration.array() + tolerance).any())
  {
    failure_reason = "direct fixed-period integration exceeded a checked motion bound";
    return std::nullopt;
  }

  moveit::core::RobotState bound_check(*robot_state);
  bound_check.setJointGroupPositions(joint_model_group_, next_state.positions);
  if (!bound_check.satisfiesBounds(joint_model_group_, tolerance)) {
    failure_reason = "integrated state violates the MoveIt robot model bounds";
    return std::nullopt;
  }
  return next_state;
}

CommandPublicationStatus CollisionAwareServoComponent::enqueue_and_publish(
  const moveit_servo::KinematicState & current_state,
  ValidatedCommand command,
  const rclcpp::Time & current_time,
  std::string & failure_reason)
{
  const std::int64_t initial_lead_ns = seconds_to_nanoseconds(
    servo_parameters_.max_expected_latency);
  const std::int64_t minimum_lead_ns = seconds_to_nanoseconds(
    parameters_.residual_command_latency_sec + parameters_.control_period_sec);
  const rclcpp::Time scheduling_time = now();
  if (scheduling_time < current_time) {
    failure_reason = "ROS time rewound during a control calculation";
    return CommandPublicationStatus::kInvariantFailure;
  }
  const auto make_stamp = [this, &scheduling_time, initial_lead_ns, minimum_lead_ns]() {
      const auto stamp_ns = control_time_grid_->commandStamp(
        scheduling_time.nanoseconds(), initial_lead_ns, minimum_lead_ns);
      if (!stamp_ns.has_value()) {
        return std::optional<rclcpp::Time>{};
      }
      return std::optional<rclcpp::Time>{rclcpp::Time(
          *stamp_ns, scheduling_time.get_clock_type())};
    };

  if (!command_queue_.empty() && !command_epoch_active_) {
    failure_reason = "Inactive command epoch retained an online branch";
    return CommandPublicationStatus::kInvariantFailure;
  }
  if (command_queue_.empty()) {
    if (command_epoch_active_) {
      failure_reason = "Command epoch lost its buffered timeline";
      return CommandPublicationStatus::kInvariantFailure;
    }
    moveit_servo::KinematicState seed = current_state;
    const auto seed_stamp = make_stamp();
    if (!seed_stamp.has_value()) {
      failure_reason = "Could not start the fixed-period command timeline";
      return CommandPublicationStatus::kInvariantFailure;
    }
    seed.time_stamp = *seed_stamp;
    command_queue_.push_back(std::move(seed));
    command_epoch_active_ = true;
    command_epoch_has_published_ = false;
  }

  if (command_queue_.size() + 1U > max_command_queue_points_) {
    failure_reason = "Command queue exceeded its calibrated hard size bound";
    return CommandPublicationStatus::kInvariantFailure;
  }
  const auto next_stamp = make_stamp();
  if (!next_stamp.has_value()) {
    failure_reason = "Command buffer fell below the minimum safe lead";
    return CommandPublicationStatus::kRecoverableTimingFailure;
  }
  const rclcpp::Duration interval = *next_stamp - command_queue_.back().time_stamp;
  if (interval.nanoseconds() != control_time_grid_->periodNs()) {
    failure_reason = "Command timeline interval left the fixed control grid";
    return CommandPublicationStatus::kInvariantFailure;
  }
  command.next_state.time_stamp = *next_stamp;
  command_queue_.push_back(std::move(command.next_state));

  if (command.braking_tail.points.size() < 3U ||
    command.braking_tail.first_stationary_point >= command.braking_tail.points.size() ||
    command.braking_tail.points.size() -
    command.braking_tail.first_stationary_point < 3U ||
    !command.braking_tail.points.front().position.isApprox(
      command_queue_.back().positions, kFiniteEpsilon) ||
    !command.braking_tail.points.front().velocity.isApprox(
      command_queue_.back().velocities, kFiniteEpsilon) ||
    !command.braking_tail.points.front().acceleration.isApprox(
      command_queue_.back().accelerations, kFiniteEpsilon))
  {
    failure_reason = "Validated braking tail does not start at the command branch";
    return CommandPublicationStatus::kInvariantFailure;
  }

  const auto * previous_publication = active_published_record();
  if (previous_publication != nullptr && command_queue_.size() >= 2U) {
    const auto & previous_queue = previous_publication->execution_queue;
    const auto new_branch = std::prev(command_queue_.end());
    for (auto branch = command_queue_.begin(); branch != new_branch; ++branch) {
      const auto published_match = std::find_if(
        previous_queue.begin(), previous_queue.end(),
        [&branch](const moveit_servo::KinematicState & published_state) {
          return published_state.time_stamp == branch->time_stamp;
        });
      if (published_match == previous_queue.end() ||
        !published_match->positions.isApprox(branch->positions, kFiniteEpsilon) ||
        !published_match->velocities.isApprox(branch->velocities, kFiniteEpsilon) ||
        !published_match->accelerations.isApprox(branch->accelerations, kFiniteEpsilon))
      {
        failure_reason = "New command branch does not share the published execution prefix";
        return CommandPublicationStatus::kInvariantFailure;
      }
    }
  }

  std::deque<moveit_servo::KinematicState> publication_queue = command_queue_;
  const rclcpp::Duration period = rclcpp::Duration::from_nanoseconds(
    control_time_grid_->periodNs());
  for (std::size_t index = 1; index < command.braking_tail.points.size(); ++index) {
    const control::JointMotionState & tail_point = command.braking_tail.points[index];
    moveit_servo::KinematicState state(
      static_cast<int>(parameters_.joint_names.size()));
    state.joint_names = parameters_.joint_names;
    state.positions = tail_point.position;
    state.velocities = tail_point.velocity;
    state.accelerations = tail_point.acceleration;
    state.time_stamp = publication_queue.back().time_stamp + period;
    publication_queue.push_back(std::move(state));
  }

  if (const auto trajectory = moveit_servo::composeTrajectoryMessage(
      servo_parameters_, publication_queue))
  {
    const rclcpp::Time send_time = now();
    // The replacement starts changing the piecewise-linear command at the
    // previous branch point, one period before the new waypoint itself.
    const rclcpp::Time replacement_commit_time =
      previous_publication == nullptr ?
      command_queue_.front().time_stamp :
      std::prev(command_queue_.end(), 2)->time_stamp;
    const double replacement_lead_sec =
      (replacement_commit_time - send_time).seconds();
    if (!std::isfinite(replacement_lead_sec) ||
      replacement_lead_sec + kFiniteEpsilon < parameters_.residual_command_latency_sec)
    {
      failure_reason =
        "Checked branch reached publication with less than the residual replacement lead";
      return CommandPublicationStatus::kRecoverableTimingFailure;
    }
    const rclcpp::Time published_end = std::prev(publication_queue.end(), 2)->time_stamp;
    const double published_horizon_sec = (published_end - send_time).seconds();
    if (!std::isfinite(published_horizon_sec) ||
      published_horizon_sec + kFiniteEpsilon < parameters_.residual_command_latency_sec)
    {
      failure_reason =
        "Checked trajectory reached publication with less than the residual safety horizon";
      return CommandPublicationStatus::kRecoverableTimingFailure;
    }
    last_published_horizon_sec_ = published_horizon_sec;
    if (!minimum_published_horizon_sec_.has_value() ||
      published_horizon_sec < *minimum_published_horizon_sec_)
    {
      minimum_published_horizon_sec_ = published_horizon_sec;
    }
    const rclcpp::Time stationary_time = command_queue_.back().time_stamp +
      rclcpp::Duration::from_nanoseconds(
      control_time_grid_->periodNs() * static_cast<std::int64_t>(
        command.braking_tail.first_stationary_point));
    const auto publication_id = published_trajectory_history_.append(
      std::move(publication_queue), stationary_time);
    if (!publication_id.has_value()) {
      failure_reason = "Published trajectory history rejected a checked command";
      return CommandPublicationStatus::kInvariantFailure;
    }
    trajectory_publisher_->publish(*trajectory);
    active_publication_id_ = *publication_id;
    last_replacement_lead_sec_ = replacement_lead_sec;
    if (!minimum_replacement_lead_sec_.has_value() ||
      replacement_lead_sec < *minimum_replacement_lead_sec_)
    {
      minimum_replacement_lead_sec_ = replacement_lead_sec;
    }
    last_braking_tail_points_ = command.braking_tail.points.size() - 1U;
    maximum_braking_tail_points_ = std::max(
      maximum_braking_tail_points_, last_braking_tail_points_);
    ++published_command_count_;
    command_epoch_has_published_ = true;
    return CommandPublicationStatus::kSuccess;
  }
  failure_reason = "MoveIt Servo could not compose the buffered trajectory";
  return CommandPublicationStatus::kInvariantFailure;
}

void CollisionAwareServoComponent::process_startup_cycle(
  const moveit_servo::KinematicState & state,
  const moveit::core::RobotState & robot_state,
  const planning_scene::PlanningScene & planning_scene)
{
  if (!startup_warmup_complete_) {
    ++startup_warmup_cycles_;
    controller_mode_ = ControllerMode::kBraking;
    if (startup_warmup_cycles_ >= kRequiredStartupWarmupCycles) {
      startup_warmup_complete_ = true;
    }
    publish_status(
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      "Priming collision and solver caches before starting the command timeline (" +
      std::to_string(startup_warmup_cycles_) + "/" +
      std::to_string(kRequiredStartupWarmupCycles) + ")");
    return;
  }

  if (!command_epoch_active_) {
    std::string bootstrap_failure;
    BootstrapFailureKind bootstrap_failure_kind =
      BootstrapFailureKind::kSafetyOrInvariant;
    const rclcpp::Time bootstrap_time = now();
    auto bootstrap = make_stationary_bootstrap(
      state, robot_state, planning_scene, bootstrap_failure,
      bootstrap_failure_kind);
    if (!bootstrap.has_value()) {
      if (bootstrap_failure_kind == BootstrapFailureKind::kRetryableState) {
        enter_startup_retry("Command timeline bootstrap deferred: " + bootstrap_failure);
      } else {
        enter_safety_wait("Command timeline bootstrap failed: " + bootstrap_failure);
      }
      return;
    }
    const CommandPublicationStatus publication_status = enqueue_and_publish(
      bootstrap->seed_state, std::move(bootstrap->first_command),
      bootstrap_time, bootstrap_failure);
    if (publication_status != CommandPublicationStatus::kSuccess) {
      if (publication_status == CommandPublicationStatus::kRecoverableTimingFailure) {
        enter_startup_retry("Command timeline bootstrap retry: " + bootstrap_failure);
      } else {
        enter_safety_wait("Command timeline bootstrap failed: " + bootstrap_failure);
      }
      return;
    }
    ++bootstrap_publication_count_;
    const auto * bootstrap_record = active_published_record();
    if (bootstrap_record == nullptr || bootstrap_record->execution_queue.empty() ||
      !active_publication_id_.has_value())
    {
      enter_safety_wait("Published bootstrap record is unavailable");
      return;
    }
    bootstrap_ack_pending_ = true;
    bootstrap_publication_id_ = *active_publication_id_;
    bootstrap_trajectory_origin_ =
      bootstrap_record->execution_queue.front().time_stamp;
    bootstrap_seed_positions_ =
      bootstrap_record->execution_queue.front().positions;
    // A bootstrap acknowledgement is a new bounded phase. Do not inherit a
    // nearly expired settle deadline from the preceding controlled re-arm.
    start_rearm_deadline(bootstrap_time);
    controller_mode_ = ControllerMode::kBraking;
    publish_status(
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      "Stationary checked command timeline primed");
    return;
  }
}

}  // namespace face_tracking_arm
