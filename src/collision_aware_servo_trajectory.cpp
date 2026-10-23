// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

#include <cstring>

#include <builtin_interfaces/msg/duration.hpp>
#include <builtin_interfaces/msg/time.hpp>

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr std::size_t kRequiredStartupWarmupCycles = 5;
constexpr std::uint64_t kBootstrapResetIdPrefix = std::uint64_t{1} << 63U;
constexpr std::uint64_t kBootstrapResetSequenceMask = ~kBootstrapResetIdPrefix;

builtin_interfaces::msg::Time to_builtin_time(const rclcpp::Time & time)
{
  const auto ns = time.nanoseconds();
  builtin_interfaces::msg::Time result;
  result.sec = static_cast<std::int32_t>(ns / 1'000'000'000LL);
  result.nanosec = static_cast<std::uint32_t>(ns % 1'000'000'000LL);
  return result;
}

struct TrajectoryHeaderView
{
  builtin_interfaces::msg::Time stamp;
  const std::string & frame_id;
};

struct TrajectoryPublicationView
{
  TrajectoryHeaderView header;
  std::uint64_t publication_id;
  std::uint8_t publication_kind;
  builtin_interfaces::msg::Time stationary_time;
  std::uint32_t tail_start_index;
  std::uint64_t hardware_epoch;
  std::uint64_t hardware_read_sequence;
  std::uint64_t control_tick_sequence;
  const trajectory_msgs::msg::JointTrajectory & trajectory;
};

control::JointMotionState to_motion_state(const moveit_servo::KinematicState & state)
{
  control::JointMotionState result;
  result.position = state.positions;
  result.velocity = state.velocities;
  result.acceleration = state.accelerations;
  return result;
}

}  // namespace

void CollisionAwareServoComponent::record_trajectory_publication(
  const trajectory_msgs::msg::JointTrajectory & trajectory,
  const std::uint64_t publication_id, const std::uint8_t publication_kind,
  const rclcpp::Time & stationary_time, const std::uint32_t tail_start_index)
{
  const TrajectoryHeaderView header{to_builtin_time(now()), trajectory.header.frame_id};
  const auto hardware_epoch = cycle_hardware_io_status_.has_value() ?
    cycle_hardware_io_status_->sample.epoch : 0U;
  const auto hardware_read_sequence = cycle_hardware_io_status_.has_value() ?
    cycle_hardware_io_status_->sample.read_sequence : 0U;
  const TrajectoryPublicationView publication{
    header,
    publication_id,
    publication_kind,
    to_builtin_time(stationary_time),
    tail_start_index,
    hardware_epoch,
    hardware_read_sequence,
    execution_context_tick_sequence_,
    trajectory};
  (void)trajectory_publication_ring_.tryEncodeAndPush(publication);
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
    maximum_absolute(controller_state->reference.velocities) > kStoppedVelocityRadps ||
    maximum_absolute(controller_state->feedback.velocities) > kStoppedVelocityRadps ||
    measured_state.velocities.cwiseAbs().maxCoeff() > kStoppedVelocityRadps)
  {
    failure_kind = BootstrapFailureKind::kRetryableState;
    failure_reason =
      "trajectory controller or measured state left its stationary hold before bootstrap";
    return std::nullopt;
  }

  if (require_hardware_io_status_) {
    const std::size_t joint_count = parameters_.joint_names.size();
    const auto hardware_status = current_hardware_io_status();
    if (measured_state.positions.size() != static_cast<Eigen::Index>(joint_count) ||
      hardware_execution_error_rad_.size() != joint_count ||
      !motion_limits_.numerical_policy.has_value() || !hardware_status ||
      !hardware_status->valid)
    {
      failure_kind = BootstrapFailureKind::kRetryableState;
      failure_reason = !hardware_status ?
        "hardware stationary bootstrap lacks a fresh status" :
        (!hardware_status->valid ? "hardware stationary bootstrap status is invalid: " +
        std::string(hardware::hardwareIoStatusFailureName(hardware_status->failure)) :
        "hardware stationary bootstrap lacks a complete execution proof");
      return std::nullopt;
    }
    const double arithmetic_error = motion_limits_.numerical_policy->position_error.value;
    for (std::size_t index = 0; index < joint_count; ++index) {
      const double reference = controller_state->reference.positions[index];
      const double output = controller_state->output.positions[index];
      const double seed = measured_state.positions[static_cast<Eigen::Index>(index)];
      const double actual = hardware_status->sample.position[index];
      const auto measurement_error = control::floatMeasurementHalfUlp(actual);
      const double execution_error = hardware_execution_error_rad_[index];
      if (!measurement_error || !std::isfinite(execution_error) || execution_error < 0.0 ||
        std::abs(reference - output) > kControllerCommandAlignmentToleranceRad ||
        std::abs(seed - reference) > kControllerCommandAlignmentToleranceRad ||
        std::abs(actual - reference) >
        execution_error + measurement_error->value + arithmetic_error)
      {
        failure_kind = BootstrapFailureKind::kRetryableState;
        failure_reason = "hardware stationary bootstrap lost controller ownership or "
          "actual-to-command proof for " + parameters_.joint_names[index] +
          " seed=" + std::to_string(seed) + " reference=" + std::to_string(reference) +
          " raw=" + std::to_string(actual) + " E=" + std::to_string(execution_error);
        return std::nullopt;
      }
    }
  }

  moveit_servo::KinematicState stationary_state(
    static_cast<int>(parameters_.joint_names.size()));
  stationary_state.joint_names = parameters_.joint_names;
  stationary_state.positions = measured_state.positions;
  stationary_state.velocities.setZero();
  stationary_state.accelerations.setZero();

  const control::JointMotionState measured_motion_state = to_motion_state(measured_state);
  const control::JointMotionState stationary_motion_state = to_motion_state(stationary_state);
  ++telemetry_.segment_validation_count;
  ++telemetry_.bootstrap_segment_validation_count;
  const control::CollisionSegmentValidationResult bootstrap_segment =
    collision_constraint_builder_->validateSegment(
    planning_scene, robot_state, *joint_model_group_, measured_motion_state,
    stationary_motion_state,
    static_cast<std::size_t>(parameters_.segment_validation_substeps));
  telemetry_.observeClearances(
    bootstrap_segment.minimum_self_distance_m,
    bootstrap_segment.minimum_world_distance_m);
  if (!bootstrap_segment.input_valid || bootstrap_segment.unsafe) {
    record_segment_rejection(CommandAttempt::kBraking, bootstrap_segment);
    failure_reason = "stationary bootstrap rejected: " +
      telemetry_.last_segment_rejection_reason;
    return std::nullopt;
  }

  control::EmergencyBrakeTail stationary_tail =
    emergency_brake_tail_generator_->generate(stationary_motion_state, motion_limits_);
  if (!stationary_tail.command_available()) {
    ++telemetry_.braking_tail_generation_failure_count;
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
  const control::CollisionConstraintDiagnostics & collision_diagnostics,
  CandidateRejectionKind & rejection_kind)
{
  rejection_kind = CandidateRejectionKind::kOther;
  auto next_state = make_next_state(
    robot_state, current_state, joint_velocity, failure_reason);
  if (!next_state.has_value()) {
    ++telemetry_.moveit_validation_failure_count;
    RCLCPP_WARN(
      get_logger(), "%s MoveIt Servo validation rejected the command: %s",
      command_attempt_name(attempt), failure_reason.c_str());
    return std::nullopt;
  }

  const control::JointMotionState candidate_motion_state = to_motion_state(*next_state);
  if (motion_reference_) {
    std::string motion_reference_violation;
    if (!motion_reference_->validateCommandTransition(
        *robot_state, current_motion_state.position, candidate_motion_state.position,
        &motion_reference_violation))
    {
      failure_reason = motion_reference_->compactManeuverActive() ?
        "Compact command violates its committed " + motion_reference_violation +
        " constraint" :
        "FACE command violates its armed " + motion_reference_violation + " guard";
      if (!motion_reference_->compactManeuverActive()) {
        rejection_kind = CandidateRejectionKind::kReferenceGuard;
      }
      ++telemetry_.moveit_validation_failure_count;
      return std::nullopt;
    }
  }
  // The QP enforces a jerk-aware stopping envelope at every control tick.
  // Validate the next segment when its distance reserve is active or its
  // conservative distance certificate is inconclusive. The complete braking
  // suffix is separately certified before publication, using a whole-path
  // bound when possible and sampled checks for unresolved blocks otherwise.
  const bool path_requires_sampling =
    !collision_constraint_builder_->canCertifySegment(
    collision_diagnostics, current_motion_state.position, candidate_motion_state.position);
  if (segment_validation_required || path_requires_sampling) {
    ++telemetry_.segment_validation_count;
    const auto segment_validation_start = std::chrono::steady_clock::now();
    const control::CollisionSegmentValidationResult segment_result =
      collision_constraint_builder_->validateSegment(
      planning_scene, *robot_state, *joint_model_group_, current_motion_state,
      candidate_motion_state,
      static_cast<std::size_t>(parameters_.segment_validation_substeps));
    (void)telemetry_.segment_validation_statistics.add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - segment_validation_start).count());
    telemetry_.observeClearances(
      segment_result.minimum_self_distance_m, segment_result.minimum_world_distance_m);
    if (!segment_result.input_valid || segment_result.unsafe) {
      record_segment_rejection(attempt, segment_result);
      failure_reason = telemetry_.last_segment_rejection_reason;
      return std::nullopt;
    }
  }

  control::EmergencyBrakeTail braking_tail =
    emergency_brake_tail_generator_->generate(candidate_motion_state, motion_limits_);
  if (!braking_tail.command_available()) {
    ++telemetry_.braking_tail_generation_failure_count;
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
  if (motion_reference_) {
    const Eigen::VectorXd * previous_guarded_position = &candidate_motion_state.position;
    for (std::size_t tail_index = 0U; tail_index < braking_tail.points.size(); ++tail_index) {
      const auto & point = braking_tail.points[tail_index];
      std::string motion_reference_violation;
      if (!motion_reference_->validateCommandTransition(
          *robot_state, *previous_guarded_position, point.position,
          &motion_reference_violation))
      {
        failure_reason = motion_reference_->compactManeuverActive() ?
          "Compact emergency-braking suffix violates its committed " +
          motion_reference_violation + " constraint" :
          "FACE emergency-braking suffix violates its armed " +
          motion_reference_violation + " guard";
        if (motion_reference_->compactManeuverActive()) {
          rejection_kind = CandidateRejectionKind::kCompactStopEnvelope;
        } else {
          rejection_kind = CandidateRejectionKind::kReferenceGuard;
        }
        ++telemetry_.braking_tail_generation_failure_count;
        return std::nullopt;
      }
      previous_guarded_position = &point.position;
    }
  }

  const bool braking_tail_certified = collision_constraint_builder_->canCertifyPath(
    collision_diagnostics, braking_tail.points);
  telemetry_.last_braking_tail_certified_segment_count = 0U;
  telemetry_.last_braking_tail_uncertified_segment_count = 0U;
  for (std::size_t index = 0U; index + 1U < braking_tail.points.size(); ++index) {
    if (collision_constraint_builder_->canCertifySegment(
        collision_diagnostics, braking_tail.points[index].position,
        braking_tail.points[index + 1U].position))
    {
      ++telemetry_.last_braking_tail_certified_segment_count;
    } else {
      ++telemetry_.last_braking_tail_uncertified_segment_count;
    }
  }
  if (!braking_tail_certified) {
    moveit::core::RobotState tail_robot_state(*robot_state);
    const auto & tail_start = braking_tail.points.front();
    tail_robot_state.setJointGroupPositions(joint_model_group_, tail_start.position);
    tail_robot_state.setJointGroupVelocities(joint_model_group_, tail_start.velocity);
    tail_robot_state.setJointGroupAccelerations(
      joint_model_group_, tail_start.acceleration);
    tail_robot_state.update();
    const std::size_t tail_segment_count = braking_tail.points.size() - 1U;
    telemetry_.segment_validation_count += tail_segment_count;
    telemetry_.braking_tail_segment_validation_count += tail_segment_count;
    const auto path_validation_start = std::chrono::steady_clock::now();
    const control::CollisionSegmentValidationResult tail_path =
      collision_constraint_builder_->validatePath(
      planning_scene, tail_robot_state, *joint_model_group_, braking_tail.points,
      static_cast<std::size_t>(parameters_.segment_validation_substeps));
    telemetry_.braking_tail_distance_query_count += tail_path.distance_query_count;
    telemetry_.braking_tail_certified_path_block_count +=
      tail_path.certified_path_block_count;
    telemetry_.braking_tail_inconclusive_path_block_count +=
      tail_path.inconclusive_path_block_count;
    (void)telemetry_.path_validation_statistics.add(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - path_validation_start).count());
    telemetry_.observeClearances(tail_path.minimum_self_distance_m,
        tail_path.minimum_world_distance_m);
    if (!tail_path.input_valid || tail_path.unsafe) {
      ++telemetry_.braking_tail_collision_rejection_count;
      record_segment_rejection(attempt, tail_path);
      failure_reason = "pre-published braking tail rejected: " +
        telemetry_.last_segment_rejection_reason;
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

  if (!motion_limits_.numerical_policy.has_value()) {
    failure_reason = "Direct integration has no shared numerical policy";
    return std::nullopt;
  }
  const auto & numerical = *motion_limits_.numerical_policy;
  const double dt = parameters_.control_period_sec;
  for (std::size_t index = 0; index < parameters_.joint_names.size(); ++index) {
    const Eigen::Index axis = static_cast<Eigen::Index>(index);
    const control::PositionInterval position_interval{
      motion_limits_.lower_position[axis], motion_limits_.upper_position[axis]};
    const double jerk =
      (next_state.accelerations[axis] - current_state.accelerations[axis]) / dt;
    const auto acceleration_class = control::classifyMagnitude(
      control::RadiansPerSecondSquared{next_state.accelerations[axis]},
      control::RadiansPerSecondSquared{motion_limits_.max_acceleration[axis]},
      numerical.acceleration_error);
    const auto jerk_class = control::classifyMagnitude(
      control::RadiansPerSecondCubed{jerk},
      control::RadiansPerSecondCubed{motion_limits_.max_jerk[axis]},
      numerical.jerk_error);
    if (!control::isNominalPosition(next_state.positions[axis], position_interval) ||
      std::abs(next_state.velocities[axis]) > motion_limits_.max_velocity[axis] ||
      acceleration_class == control::BoundaryClass::kOutside ||
      acceleration_class == control::BoundaryClass::kInvalid ||
      jerk_class == control::BoundaryClass::kOutside ||
      jerk_class == control::BoundaryClass::kInvalid)
    {
      failure_reason = "Direct fixed-period integration is outside the shared q/v/a/j "
        "command policy for " + parameters_.joint_names[index];
      return std::nullopt;
    }
  }

  moveit::core::RobotState bound_check(*robot_state);
  bound_check.setJointGroupPositions(joint_model_group_, next_state.positions);
  if (!bound_check.satisfiesBounds(joint_model_group_, 0.0)) {
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
      const auto stamp_ns = commands_.time_grid->commandStamp(
        scheduling_time.nanoseconds(), initial_lead_ns, minimum_lead_ns);
      if (!stamp_ns.has_value()) {
        return std::optional<rclcpp::Time>{};
      }
      return std::optional<rclcpp::Time>{rclcpp::Time(
          *stamp_ns, scheduling_time.get_clock_type())};
    };

  if (!commands_.queue.empty() && !commands_.epoch_active) {
    failure_reason = "Inactive command epoch retained an online branch";
    return CommandPublicationStatus::kInvariantFailure;
  }
  if (commands_.queue.empty()) {
    if (commands_.epoch_active) {
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
    commands_.queue.push_back(std::move(seed));
    commands_.epoch_active = true;
    commands_.epoch_has_published = false;
  }

  if (commands_.queue.size() + 1U > commands_.max_queue_points) {
    failure_reason = "Command queue exceeded its calibrated hard size bound";
    return CommandPublicationStatus::kInvariantFailure;
  }
  const auto next_stamp = make_stamp();
  if (!next_stamp.has_value()) {
    failure_reason = "Command buffer fell below the minimum safe lead";
    return CommandPublicationStatus::kRecoverableCommandDeadline;
  }
  const rclcpp::Duration interval = *next_stamp - commands_.queue.back().time_stamp;
  if (interval.nanoseconds() != commands_.time_grid->periodNs()) {
    failure_reason = "Command timeline interval left the fixed control grid";
    return CommandPublicationStatus::kInvariantFailure;
  }
  command.next_state.time_stamp = *next_stamp;
  commands_.queue.push_back(std::move(command.next_state));

  if (command.braking_tail.points.size() < 3U ||
    command.braking_tail.first_stationary_point >= command.braking_tail.points.size() ||
    command.braking_tail.points.size() -
    command.braking_tail.first_stationary_point < 3U ||
    !command.braking_tail.points.front().position.isApprox(
      commands_.queue.back().positions, kFiniteEpsilon) ||
    !command.braking_tail.points.front().velocity.isApprox(
      commands_.queue.back().velocities, kFiniteEpsilon) ||
    !command.braking_tail.points.front().acceleration.isApprox(
      commands_.queue.back().accelerations, kFiniteEpsilon))
  {
    failure_reason = "Validated braking tail does not start at the command branch";
    return CommandPublicationStatus::kInvariantFailure;
  }

  const auto * previous_publication = commands_.activePublication();
  if (previous_publication != nullptr && commands_.queue.size() >= 2U) {
    const auto & previous_queue = previous_publication->execution_queue;
    const auto new_branch = std::prev(commands_.queue.end());
    for (auto branch = commands_.queue.begin(); branch != new_branch; ++branch) {
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

  std::deque<moveit_servo::KinematicState> publication_queue = commands_.queue;
  const rclcpp::Duration period = rclcpp::Duration::from_nanoseconds(
    commands_.time_grid->periodNs());
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

  for (std::size_t point_index = 0; point_index < publication_queue.size(); ++point_index) {
    const auto & point = publication_queue[point_index];
    if (point.positions.size() != static_cast<Eigen::Index>(position_bound_sets_.size())) {
      failure_reason = "Published trajectory point has the wrong joint count";
      return CommandPublicationStatus::kInvariantFailure;
    }
    for (std::size_t joint_index = 0; joint_index < position_bound_sets_.size(); ++joint_index) {
      const Eigen::Index axis = static_cast<Eigen::Index>(joint_index);
      const double nominal = point.positions[axis];
      const control::PositionInterval working_interval{
        motion_limits_.lower_position[axis], motion_limits_.upper_position[axis]};
      if (!control::isNominalPosition(nominal, working_interval)) {
        failure_reason = "Published point " + std::to_string(point_index) +
          " leaves nominal q bounds for " + parameters_.joint_names[joint_index];
        return CommandPublicationStatus::kInvariantFailure;
      }
      if (!require_hardware_io_status_) {
        continue;
      }
      const double transmitted = static_cast<double>(static_cast<float>(nominal));
      if (!control::isWireCommand(nominal, position_bound_sets_[joint_index].wire)) {
        failure_reason = "Published point " + std::to_string(point_index) +
          " violates the strict float wire interval for " +
          parameters_.joint_names[joint_index];
        return CommandPublicationStatus::kInvariantFailure;
      }
      if (std::abs(transmitted - nominal) >
        execution_error_bound_rad())
      {
        failure_reason = "Float wire quantization exceeds the collision uncertainty budget for " +
          parameters_.joint_names[joint_index];
        return CommandPublicationStatus::kInvariantFailure;
      }
    }
  }

  if (const auto trajectory = moveit_servo::composeTrajectoryMessage(
      servo_parameters_, publication_queue))
  {
    const rclcpp::Time send_time = now();
    // The replacement starts changing the piecewise-linear command at the
    // previous branch point, one period before the new waypoint itself.
    const rclcpp::Time replacement_commit_time =
      previous_publication == nullptr ?
      commands_.queue.front().time_stamp :
      std::prev(commands_.queue.end(), 2)->time_stamp;
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
    telemetry_.last_published_horizon_sec = published_horizon_sec;
    if (!telemetry_.minimum_published_horizon_sec.has_value() ||
      published_horizon_sec < *telemetry_.minimum_published_horizon_sec)
    {
      telemetry_.minimum_published_horizon_sec = published_horizon_sec;
    }
    const rclcpp::Time stationary_time = commands_.queue.back().time_stamp +
      rclcpp::Duration::from_nanoseconds(
      commands_.time_grid->periodNs() * static_cast<std::int64_t>(
        command.braking_tail.first_stationary_point));
    const auto publication_id = commands_.publications.append(
      std::move(publication_queue), stationary_time);
    if (!publication_id.has_value()) {
      failure_reason = "Published trajectory history rejected a checked command";
      return CommandPublicationStatus::kInvariantFailure;
    }
    if ((*publication_id & kBootstrapResetIdPrefix) != 0U) {
      failure_reason = "Published trajectory identity exhausted its diagnostic ID range";
      return CommandPublicationStatus::kInvariantFailure;
    }
    const std::size_t tail_start_index = commands_.queue.size() - 1U +
      command.braking_tail.first_stationary_point;
    record_trajectory_publication(
      *trajectory, *publication_id, msg::ServoTrajectoryPublication::KIND_SERVO_TRAJECTORY,
      stationary_time, static_cast<std::uint32_t>(tail_start_index));
    trajectory_publisher_->publish(*trajectory);
    commands_.active_publication_id = *publication_id;
    telemetry_.last_replacement_lead_sec = replacement_lead_sec;
    if (!telemetry_.minimum_replacement_lead_sec.has_value() ||
      replacement_lead_sec < *telemetry_.minimum_replacement_lead_sec)
    {
      telemetry_.minimum_replacement_lead_sec = replacement_lead_sec;
    }
    telemetry_.last_braking_tail_points = command.braking_tail.points.size() - 1U;
    telemetry_.maximum_braking_tail_points = std::max(
      telemetry_.maximum_braking_tail_points, telemetry_.last_braking_tail_points);
    ++telemetry_.published_command_count;
    commands_.epoch_has_published = true;
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
  if (!commands_.startup_warmup_complete) {
    ++commands_.startup_warmup_cycles;
    controller_mode_ = ControllerMode::kBraking;
    if (commands_.startup_warmup_cycles >= kRequiredStartupWarmupCycles) {
      commands_.startup_warmup_complete = true;
    }
    publish_status(
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      "Priming collision and solver caches before starting the command timeline (" +
      std::to_string(commands_.startup_warmup_cycles) + "/" +
      std::to_string(kRequiredStartupWarmupCycles) + ")");
    return;
  }

  if (commands_.bootstrap_reset_ack_pending) {
    const auto controller_state = current_controller_state();
    if (!controller_state.has_value()) {
      publish_status(
        moveit_msgs::msg::ServoStatus::NO_WARNING,
        "Stationary controller reset is waiting for fresh feedback");
      return;
    }
    std::string feedback_failure;
    if (std::abs(controller_state->speed_scaling_factor - 1.0) >
      kControllerSpeedScalingTolerance ||
      !feedback_motion_within_limits(*controller_state, feedback_failure))
    {
      enter_safety_wait(
        "Stationary controller reset feedback is unsafe" +
        (feedback_failure.empty() ? std::string{} : ": " + feedback_failure));
      return;
    }
    if (!commands_.bootstrap_reset_positions.has_value()) {
      enter_safety_wait("Stationary controller reset lost its measured target");
      return;
    }
    const auto map_values = [](const std::vector<double> & values) {
        return Eigen::Map<const Eigen::VectorXd>(
          values.data(), static_cast<Eigen::Index>(values.size())).eval();
      };
    const Eigen::VectorXd measured_positions = state.positions;
    const bool measured_state_matches_reset = measured_positions.size() ==
      commands_.bootstrap_reset_positions->size() && measured_positions.allFinite() &&
      (measured_positions - *commands_.bootstrap_reset_positions).cwiseAbs().maxCoeff() <=
      execution_error_bound_rad();
    const bool acknowledged = commands_.observeBootstrapResetAcknowledgement(
      rclcpp::Time(controller_state->header.stamp, get_clock()->get_clock_type()),
      *commands_.bootstrap_reset_positions,
      map_values(controller_state->reference.positions),
      map_values(controller_state->output.positions),
      map_values(controller_state->feedback.positions),
      map_values(controller_state->reference.velocities),
      map_values(controller_state->feedback.velocities),
      kControllerCommandAlignmentToleranceRad,
      execution_error_bound_rad(), kStoppedVelocityRadps,
      measured_state_matches_reset);
    if (!acknowledged) {
      publish_status(
        moveit_msgs::msg::ServoStatus::NO_WARNING,
        "Stationary controller reset is waiting for three fresh settled acknowledgements");
      return;
    }
    commands_.bootstrap_reset_ack_pending = false;
    commands_.bootstrap_reset_published_time.reset();
  }

  if (!commands_.epoch_active) {
    if (!commands_.bootstrap_reset_positions.has_value()) {
      const auto controller_state = current_controller_state();
      if (!controller_state.has_value()) {
        publish_status(
          moveit_msgs::msg::ServoStatus::NO_WARNING,
          "Stationary controller hold is waiting for fresh feedback");
        return;
      }
      std::string feedback_failure;
      if (std::abs(controller_state->speed_scaling_factor - 1.0) >
        kControllerSpeedScalingTolerance ||
        !feedback_motion_within_limits(*controller_state, feedback_failure))
      {
        enter_safety_wait(
          "Stationary controller hold is unsafe" +
          (feedback_failure.empty() ? std::string{} : ": " + feedback_failure));
        return;
      }
      const auto map_values = [](const std::vector<double> & values) {
          return Eigen::Map<const Eigen::VectorXd>(
            values.data(), static_cast<Eigen::Index>(values.size())).eval();
        };
      const bool source_hold_ready = commands_.observeBootstrapSourceHold(
        rclcpp::Time(controller_state->header.stamp, get_clock()->get_clock_type()),
        map_values(controller_state->reference.positions),
        map_values(controller_state->output.positions),
        map_values(controller_state->feedback.positions),
        map_values(controller_state->reference.velocities),
        map_values(controller_state->feedback.velocities),
        kControllerCommandAlignmentToleranceRad,
        execution_error_bound_rad(), kStoppedVelocityRadps, true);
      if (!source_hold_ready) {
        publish_status(
          moveit_msgs::msg::ServoStatus::NO_WARNING,
          "Stationary controller hold is confirming three fresh stable samples");
        return;
      }
    }

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

    if (!commands_.bootstrap_reset_positions.has_value()) {
      // Replace only a stationary controller hold. A zero-stamped t=0 point
      // makes Jazzy JTC sample this measured hold immediately instead of
      // blending from its previous desired state to the fresh actual state.
      trajectory_msgs::msg::JointTrajectory reset_trajectory;
      reset_trajectory.joint_names = parameters_.joint_names;
      reset_trajectory.header.stamp = builtin_interfaces::msg::Time{};
      trajectory_msgs::msg::JointTrajectoryPoint reset_point;
      reset_point.positions.assign(
        bootstrap->seed_state.positions.data(),
        bootstrap->seed_state.positions.data() + bootstrap->seed_state.positions.size());
      for (std::size_t joint_index = 0; require_hardware_io_status_ &&
        joint_index < position_bound_sets_.size();
        ++joint_index)
      {
        if (!control::isWireCommand(
            bootstrap->seed_state.positions[static_cast<Eigen::Index>(joint_index)],
            position_bound_sets_[joint_index].wire))
        {
          enter_safety_wait(
            "Bootstrap hold violates the strict float wire interval for " +
            parameters_.joint_names[joint_index]);
          return;
        }
      }
      reset_point.velocities.assign(parameters_.joint_names.size(), 0.0);
      reset_point.accelerations.assign(parameters_.joint_names.size(), 0.0);
      reset_point.time_from_start = builtin_interfaces::msg::Duration{};
      reset_trajectory.points.push_back(std::move(reset_point));
      commands_.bootstrap_reset_published_time = bootstrap_time;
      if (bootstrap_reset_event_sequence_ >= kBootstrapResetSequenceMask) {
        enter_safety_wait("Bootstrap reset diagnostic event ID space exhausted");
        return;
      }
      const std::uint64_t reset_event_id = kBootstrapResetIdPrefix |
        ++bootstrap_reset_event_sequence_;
      record_trajectory_publication(
        reset_trajectory, reset_event_id,
        msg::ServoTrajectoryPublication::KIND_BOOTSTRAP_RESET, bootstrap_time, 0U);
      trajectory_publisher_->publish(reset_trajectory);
      commands_.bootstrap_reset_positions = bootstrap->seed_state.positions;
      commands_.bootstrap_reset_ack_pending = true;
      commands_.bootstrap_reset_stable_samples = 0U;
      commands_.last_bootstrap_reset_sample_time.reset();
      commands_.startRearmDeadline(bootstrap_time, std::chrono::steady_clock::now());
      controller_mode_ = ControllerMode::kBraking;
      publish_status(
        moveit_msgs::msg::ServoStatus::NO_WARNING,
        "Published stationary actual-state reset; waiting for controller acknowledgement");
      return;
    }

    const CommandPublicationStatus publication_status = enqueue_and_publish(
      bootstrap->seed_state, std::move(bootstrap->first_command),
      bootstrap_time, bootstrap_failure);
    if (publication_status != CommandPublicationStatus::kSuccess) {
      if (publication_status == CommandPublicationStatus::kRecoverableCommandDeadline ||
        publication_status == CommandPublicationStatus::kRecoverableTimingFailure)
      {
        enter_startup_retry("Command timeline bootstrap retry: " + bootstrap_failure);
      } else {
        enter_safety_wait("Command timeline bootstrap failed: " + bootstrap_failure);
      }
      return;
    }
    ++telemetry_.bootstrap_publication_count;
    const auto * bootstrap_record = commands_.activePublication();
    if (bootstrap_record == nullptr || bootstrap_record->execution_queue.empty() ||
      !commands_.active_publication_id.has_value())
    {
      if (require_hardware_io_status_) {
        request_hardware_inhibit();
      }
      enter_safety_wait("Published bootstrap record is unavailable");
      return;
    }
    commands_.bootstrap_ack_pending = true;
    commands_.bootstrap_publication_id = *commands_.active_publication_id;
    commands_.bootstrap_trajectory_origin =
      bootstrap_record->execution_queue.front().time_stamp;
    commands_.bootstrap_seed_positions =
      bootstrap_record->execution_queue.front().positions;
    // A bootstrap acknowledgement is a new bounded phase. Do not inherit a
    // nearly expired settle deadline from the preceding controlled re-arm.
    commands_.startRearmDeadline(bootstrap_time, std::chrono::steady_clock::now());
    controller_mode_ = ControllerMode::kBraking;
    publish_status(
      moveit_msgs::msg::ServoStatus::NO_WARNING,
      "Stationary checked command timeline primed");
    return;
  }
}

}  // namespace face_tracking_arm
