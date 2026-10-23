// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

#include "face_tracking_arm/snapshot_timing.hpp"

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr std::size_t kRequiredStableRearmSamples = 3;

constexpr double kFeedbackLimitTolerance = 1.0e-6;

builtin_interfaces::msg::Time to_builtin_time(const rclcpp::Time & time)
{
  constexpr std::int64_t nanoseconds_per_second = 1000000000LL;
  const std::int64_t total = time.nanoseconds();
  builtin_interfaces::msg::Time message;
  message.sec = static_cast<std::int32_t>(total / nanoseconds_per_second);
  message.nanosec = static_cast<std::uint32_t>(total % nanoseconds_per_second);
  return message;
}

}  // namespace

std::optional<control_msgs::msg::JointTrajectoryControllerState> CollisionAwareServoComponent::
current_controller_state() const
{
  std::lock_guard<std::mutex> lock(controller_state_mutex_);
  if (!latest_controller_state_.has_value()) {
    return std::nullopt;
  }
  const auto & state = *latest_controller_state_;
  const rclcpp::Time current_time = now();
  const rclcpp::Time stamp(state.header.stamp, current_time.get_clock_type());
  const rclcpp::Time oldest_allowed = current_time -
    rclcpp::Duration::from_seconds(parameters_.state_feedback_timeout_sec);
  const std::size_t size = parameters_.joint_names.size();
  const auto finite_vector = [](const std::vector<double> & values) {
      return std::all_of(values.begin(), values.end(), [](const double value) {
                 return std::isfinite(value);
        });
    };
  if (stamp.nanoseconds() <= 0 || stamp > current_time || stamp < oldest_allowed ||
    state.joint_names != parameters_.joint_names ||
    state.reference.positions.size() != size || state.reference.velocities.size() != size ||
    state.feedback.positions.size() != size || state.feedback.velocities.size() != size ||
    state.output.positions.size() != size ||
    !finite_vector(state.reference.positions) || !finite_vector(state.reference.velocities) ||
    !finite_vector(state.feedback.positions) || !finite_vector(state.feedback.velocities) ||
    !finite_vector(state.output.positions) || !std::isfinite(state.speed_scaling_factor))
  {
    return std::nullopt;
  }
  return state;
}

ActualFeedbackSafety CollisionAwareServoComponent::validate_controller_feedback_collision_state(
  const control_msgs::msg::JointTrajectoryControllerState & controller_state,
  const std::string & context, std::string & reason)
{
  if ((commands_.epoch_active || commands_.published_tail_rearm_pending) &&
    planning_scene_collision_revision_.load(std::memory_order_acquire) !=
    last_validated_scene_revision_)
  {
    ++telemetry_.planning_scene_invalidation_count;
    reason = context + " detected changed collision geometry for a published trajectory";
    return ActualFeedbackSafety::kUnsafe;
  }
  auto measured_robot_state =
    planning_scene_monitor_->getStateMonitor()->getCurrentState();
  if (!measured_robot_state) {
    reason = context + " is waiting for the measured MoveIt robot state";
    return ActualFeedbackSafety::kWaiting;
  }
  measured_robot_state = std::make_shared<moveit::core::RobotState>(*measured_robot_state);
  auto actual_positions = controller_state.feedback.positions;
  if (require_hardware_io_status_) {
    const auto hardware_status = current_hardware_io_status();
    if (!hardware_status || !hardware_status->valid) {
      reason = context + " is waiting for valid coherent hardware feedback";
      return ActualFeedbackSafety::kWaiting;
    }
    actual_positions.assign(
      hardware_status->sample.position.begin(), hardware_status->sample.position.end());
  }
  measured_robot_state->setJointGroupPositions(
    joint_model_group_, actual_positions);
  measured_robot_state->update(true);

  planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
  if (!planning_scene) {
    reason = context + " lost the PlanningScene";
    return ActualFeedbackSafety::kUnsafe;
  }
  const control::CollisionActualStateValidationResult validation =
    collision_constraint_builder_->validateActualState(
    *planning_scene, *measured_robot_state, *joint_model_group_,
    require_hardware_io_status_ ?
    control::ActualPositionEncoding::kSdkFloatPromotedToDouble :
    control::ActualPositionEncoding::kDouble);
  ++telemetry_.actual_state_validation_count;
  telemetry_.last_actual_minimum_self_distance_m =
    std::isfinite(validation.minimum_self_distance_m) ?
    std::optional<double>{validation.minimum_self_distance_m} : std::nullopt;
  telemetry_.last_actual_minimum_world_distance_m =
    std::isfinite(validation.minimum_world_distance_m) ?
    std::optional<double>{validation.minimum_world_distance_m} : std::nullopt;
  telemetry_.last_actual_closest_pair = validation.closest_pair.empty() ?
    "outside hard-clearance query" : validation.closest_pair;
  telemetry_.observeClearances(
    validation.minimum_self_distance_m, validation.minimum_world_distance_m);
  if (!validation.input_valid || validation.unsafe) {
    ++telemetry_.actual_state_validation_failure_count;
    reason = !validation.input_valid ?
      context + " actual-state validation failed: " + validation.failure_reason :
      context + " measured a hard-clearance/corridor violation for " +
      telemetry_.last_actual_closest_pair;
    return ActualFeedbackSafety::kUnsafe;
  }
  return ActualFeedbackSafety::kSafe;
}

RearmReadiness CollisionAwareServoComponent::safe_rearm_ready(
  const rclcpp::Time & current_time, std::string & reason)
{
  const auto state_monitor = planning_scene_monitor_->getStateMonitor();
  const auto [measured_state, state_validation_time] =
    control::readSnapshotAtCurrentTime(
    [state_monitor]() {return state_monitor->getCurrentStateAndTime();},
    [this]() {return now();});
  const auto & state_time = measured_state.second;
  const auto feedback_timeout = rclcpp::Duration::from_seconds(
    parameters_.state_feedback_timeout_sec);
  const rclcpp::Time oldest_allowed_state = state_validation_time -
    feedback_timeout;
  if (!measured_state.first ||
    !control::isInSameClockEpoch(
      current_time.nanoseconds(), state_validation_time.nanoseconds()) ||
    !control::isFreshSnapshotTimestamp(
      state_time.nanoseconds(), state_validation_time.nanoseconds(),
      feedback_timeout.nanoseconds()) ||
    !state_monitor->haveCompleteState(oldest_allowed_state))
  {
    commands_.stable_rearm_samples = 0;
    reason = "Safe re-arm is waiting for fresh complete robot feedback";
    return RearmReadiness::kWaiting;
  }

  const auto state = current_controller_state();
  if (!state.has_value()) {
    commands_.stable_rearm_samples = 0;
    reason = "Safe re-arm is waiting for fresh finite controller feedback";
    return RearmReadiness::kWaiting;
  }
  auto actual_state = *state;
  std::optional<hardware::HardwareIoStatusAssessment> hardware_status;
  if (require_hardware_io_status_) {
    hardware_status = current_hardware_io_status();
    if (!hardware_status || !hardware_status->valid) {
      commands_.stable_rearm_samples = 0;
      reason = "Safe re-arm is waiting for valid coherent hardware feedback";
      return RearmReadiness::kWaiting;
    }
    if (last_rearm_hardware_read_sequence_.has_value() &&
      hardware_status->sample.read_sequence <= *last_rearm_hardware_read_sequence_)
    {
      reason = "Safe re-arm is waiting for the next successful hardware read";
      return RearmReadiness::kWaiting;
    }
    last_rearm_hardware_read_sequence_ = hardware_status->sample.read_sequence;
    actual_state.feedback.positions.assign(
      hardware_status->sample.position.begin(), hardware_status->sample.position.end());
    actual_state.feedback.velocities.assign(
      hardware_status->sample.velocity.begin(), hardware_status->sample.velocity.end());
  }
  const rclcpp::Time sample_time = require_hardware_io_status_ ?
    rclcpp::Time(hardware_status->sample.read_stamp_ns, current_time.get_clock_type()) :
    rclcpp::Time(state->header.stamp, current_time.get_clock_type());
  if (require_hardware_io_status_) {
    actual_state.header.stamp = to_builtin_time(sample_time);
  }
  if (commands_.last_rearm_sample_time.has_value() &&
    sample_time <= *commands_.last_rearm_sample_time)
  {
    reason = "Safe re-arm is waiting for the next controller feedback sample";
    return RearmReadiness::kWaiting;
  }
  commands_.last_rearm_sample_time = sample_time;
  observe_controller_feedback(actual_state, current_time.get_clock_type());

  const auto maximum_absolute_difference = [](
    const std::vector<double> & first, const std::vector<double> & second) {
      if (first.size() != second.size()) {
        return std::numeric_limits<double>::infinity();
      }
      double maximum = 0.0;
      for (std::size_t index = 0; index < first.size(); ++index) {
        maximum = std::max(maximum, std::abs(first[index] - second[index]));
      }
      return maximum;
    };
  const auto maximum_expected_difference = [](
    const std::vector<double> & actual, const Eigen::VectorXd & expected) {
      if (actual.size() != static_cast<std::size_t>(expected.size())) {
        return std::numeric_limits<double>::infinity();
      }
      double maximum = 0.0;
      for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(
          maximum,
          std::abs(actual[index] - expected[static_cast<Eigen::Index>(index)]));
      }
      return maximum;
    };
  const auto maximum_absolute = [](const std::vector<double> & values) {
      double maximum = 0.0;
      for (const double value : values) {
        maximum = std::max(maximum, std::abs(value));
      }
      return maximum;
    };
  const double reference_output_error = maximum_absolute_difference(
    state->reference.positions, state->output.positions);
  const double following_position_error = maximum_absolute_difference(
    state->reference.positions, actual_state.feedback.positions);
  const double reference_velocity = maximum_absolute(state->reference.velocities);
  const double feedback_velocity = maximum_absolute(actual_state.feedback.velocities);

  if (std::abs(state->speed_scaling_factor - 1.0) >
    kControllerSpeedScalingTolerance)
  {
    ++telemetry_.controller_speed_scaling_failure_count;
    reason = "Safe re-arm detected trajectory-controller speed scaling";
    return RearmReadiness::kUnsafe;
  }

  std::string feedback_failure;
  if (require_hardware_io_status_ ?
    !hardware_feedback_motion_within_limits(
      hardware_status->sample, feedback_failure) :
    !feedback_motion_within_limits(*state, feedback_failure))
  {
    ++telemetry_.feedback_motion_limit_failure_count;
    reason = "Safe re-arm measured unsafe motion: " + feedback_failure;
    return RearmReadiness::kUnsafe;
  }
  const ActualFeedbackSafety feedback_safety =
    validate_controller_feedback_collision_state(*state, "Safe re-arm", reason);
  if (feedback_safety == ActualFeedbackSafety::kWaiting) {
    commands_.stable_rearm_samples = 0;
    return RearmReadiness::kWaiting;
  }
  if (feedback_safety == ActualFeedbackSafety::kUnsafe) {
    return RearmReadiness::kUnsafe;
  }

  // On startup/retry, a three-sample stationary reference/output pair is the
  // controller-owned hold. Once this component has published, ownership must
  // be proven by the retained trajectory and its acknowledged terminal point.
  const bool retained_publication = commands_.hasRetainedPublication();
  if (commands_.epoch_has_published && !retained_publication) {
    commands_.stable_rearm_samples = 0;
    reason = "Safe re-arm lost ownership of its previously published command";
    return RearmReadiness::kUnsafe;
  }
  bool stationary_hold_owned = !commands_.epoch_has_published && !retained_publication;
  if (commands_.epoch_has_published || retained_publication) {
    const rclcpp::Duration controller_period = rclcpp::Duration::from_nanoseconds(
      seconds_to_nanoseconds(parameters_.trajectory_controller_period_sec));
    const control::PublicationMatch publication_match =
      commands_.publications.match(
      *state, current_time.get_clock_type(), controller_period,
      kControllerCommandAlignmentToleranceRad);
    if (!publication_match.matched()) {
      ++telemetry_.command_alignment_failure_count;
      commands_.stable_rearm_samples = 0;
      reason =
        "Safe re-arm is waiting for the controller to acknowledge its checked publication";
      return RearmReadiness::kWaiting;
    } else {
      const bool publication_changed = !commands_.active_publication_id.has_value() ||
        *commands_.active_publication_id != publication_match.publication_id;
      if (publication_changed) {
        commands_.stable_rearm_samples = 0;
        if (publication_match.kind == control::PublicationMatchKind::kOlder) {
          ++telemetry_.older_publication_recovery_count;
        }
        commands_.active_publication_id = publication_match.publication_id;
      }
      const auto * published_record =
        commands_.publications.find(publication_match.publication_id);
      if (published_record == nullptr || published_record->execution_queue.size() < 3U) {
        reason = "Safe re-arm lost the published braking-tail record";
        return RearmReadiness::kUnsafe;
      }
      if (publication_changed) {
        const rclcpp::Time required_ros_deadline = published_record->stationary_time +
          rclcpp::Duration::from_seconds(kSafeRearmSettleBudgetSec);
        if (!commands_.rearm_deadline.has_value() ||
          required_ros_deadline > *commands_.rearm_deadline)
        {
          commands_.rearm_deadline = required_ros_deadline;
        }
        const double remaining_tail_sec = std::max(
          0.0, (published_record->stationary_time - current_time).seconds());
        const auto required_wall_deadline = std::chrono::steady_clock::now() +
          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(
            remaining_tail_sec + kSafeRearmSettleBudgetSec));
        if (!commands_.rearm_wall_deadline.has_value() ||
          required_wall_deadline > *commands_.rearm_wall_deadline)
        {
          commands_.rearm_wall_deadline = required_wall_deadline;
        }
      }
      const auto & published_queue = published_record->execution_queue;
      const control::ControllerSampleTimes & corrected_times =
        publication_match.sample_times;
      telemetry_.last_controller_reference_alignment_error_rad =
        publication_match.reference_error_rad;
      telemetry_.last_controller_output_alignment_error_rad =
        publication_match.output_error_rad;
      telemetry_.max_controller_reference_alignment_error_rad = std::max(
        telemetry_.max_controller_reference_alignment_error_rad,
        publication_match.reference_error_rad);
      telemetry_.max_controller_output_alignment_error_rad = std::max(
        telemetry_.max_controller_output_alignment_error_rad,
        publication_match.output_error_rad);
      telemetry_.command_alignment_check_count += 2U;
      const rclcpp::Time transmitted_end =
        std::prev(published_queue.end(), 2)->time_stamp;
      if (corrected_times.reference < published_queue.front().time_stamp ||
        (corrected_times.output > transmitted_end &&
        corrected_times.reference < published_record->stationary_time))
      {
        ++telemetry_.command_alignment_failure_count;
        reason = "Safe re-arm controller sample left the published braking-tail horizon";
        commands_.stable_rearm_samples = 0;
        return RearmReadiness::kWaiting;
      }

      if (corrected_times.reference < published_record->stationary_time) {
        commands_.stable_rearm_samples = 0;
        reason = "Safe re-arm is monitoring the pre-published braking tail";
        return RearmReadiness::kWaiting;
      }

      const double terminal_reference_error = maximum_expected_difference(
        state->reference.positions, published_queue.back().positions);
      const double terminal_output_error = maximum_expected_difference(
        state->output.positions, published_queue.back().positions);
      if (terminal_reference_error > kControllerCommandAlignmentToleranceRad ||
        terminal_output_error > kControllerCommandAlignmentToleranceRad ||
        reference_output_error > kControllerCommandAlignmentToleranceRad ||
        reference_velocity > kStoppedVelocityRadps)
      {
        ++telemetry_.command_alignment_failure_count;
        commands_.stable_rearm_samples = 0;
        reason =
          "Safe re-arm is waiting for the checked braking hold to finish and be acknowledged";
        return RearmReadiness::kWaiting;
      }
      stationary_hold_owned = true;
    }
  }

  const bool stopped_and_settled = settled_rearm_state(
    reference_output_error, following_position_error, reference_velocity, feedback_velocity,
    kControllerCommandAlignmentToleranceRad, execution_error_bound_rad(),
    true, true, stationary_hold_owned);
  if (!stopped_and_settled) {
    commands_.stable_rearm_samples = 0;
    reason =
      "Safe re-arm requires a settled hold: command=" +
      std::to_string(reference_output_error) + " rad, following=" +
      std::to_string(following_position_error) + " rad, reference_velocity=" +
      std::to_string(reference_velocity) + " rad/s, feedback_velocity=" +
      std::to_string(feedback_velocity) + " rad/s";
    return RearmReadiness::kWaiting;
  }

  ++commands_.stable_rearm_samples;
  if (commands_.stable_rearm_samples < kRequiredStableRearmSamples) {
    reason = "Safe re-arm is confirming a settled hold (" +
      std::to_string(commands_.stable_rearm_samples) + "/" +
      std::to_string(kRequiredStableRearmSamples) + ")";
    return RearmReadiness::kWaiting;
  }
  if (require_hardware_io_status_ && stationary_hold_owned) {
    // Preserve the command that was actually proven stationary. A fresh
    // hardware read is the measured state, not a replacement command seed.
    const Eigen::VectorXd hold = Eigen::Map<const Eigen::VectorXd>(
      state->reference.positions.data(),
      static_cast<Eigen::Index>(state->reference.positions.size()));
    if (hold.size() != static_cast<Eigen::Index>(position_bound_sets_.size())) {
      reason = "Safe re-arm cannot retain a stationary hold with the wrong joint count";
      return RearmReadiness::kUnsafe;
    }
    for (std::size_t index = 0; index < position_bound_sets_.size(); ++index) {
      if (!control::isWireCommand(
          hold[static_cast<Eigen::Index>(index)], position_bound_sets_[index].wire))
      {
        reason = "Safe re-arm stationary command is outside the strict wire interval for " +
          parameters_.joint_names[index];
        return RearmReadiness::kUnsafe;
      }
    }
    proven_q_hold_ = hold;
    proven_q_hold_epoch_ = hardware_status->sample.epoch;
  }
  return RearmReadiness::kReady;
}

std::optional<moveit_servo::KinematicState> CollisionAwareServoComponent::
predicted_or_measured_state(
  const rclcpp::Time & current_time, std::string & failure_reason,
  bool & requires_latched_halt,
  std::optional<control::JointMotionState> * actual_feedback,
  std::optional<rclcpp::Time> * actual_feedback_stamp)
{
  requires_latched_halt = false;
  if (actual_feedback != nullptr) {
    actual_feedback->reset();
  }
  if (actual_feedback_stamp != nullptr) {
    actual_feedback_stamp->reset();
  }
  const auto state_monitor = planning_scene_monitor_->getStateMonitor();
  const auto [measured_state, state_validation_time] =
    control::readSnapshotAtCurrentTime(
    [state_monitor]() {return state_monitor->getCurrentStateAndTime();},
    [this]() {return now();});
  const auto & measured_robot_state = measured_state.first;
  const auto & state_time = measured_state.second;
  const auto feedback_timeout = rclcpp::Duration::from_seconds(
    parameters_.state_feedback_timeout_sec);
  const rclcpp::Time oldest_allowed_state = state_validation_time -
    feedback_timeout;
  if (!measured_robot_state ||
    !control::isInSameClockEpoch(
      current_time.nanoseconds(), state_validation_time.nanoseconds()) ||
    !control::isFreshSnapshotTimestamp(
      state_time.nanoseconds(), state_validation_time.nanoseconds(),
      feedback_timeout.nanoseconds()) ||
    !state_monitor->haveCompleteState(oldest_allowed_state))
  {
    return std::nullopt;
  }
  moveit_servo::KinematicState measured(parameters_.joint_names.size());
  measured.joint_names = parameters_.joint_names;
  measured_robot_state->copyJointGroupPositions(joint_model_group_, measured.positions);
  measured_robot_state->copyJointGroupVelocities(joint_model_group_, measured.velocities);
  std::optional<hardware::HardwareIoStatusAssessment> hardware_status;
  auto actual_controller_state = current_controller_state();
  if (!actual_controller_state.has_value()) {
    failure_reason = "Fresh finite trajectory controller state is unavailable";
    requires_latched_halt = require_hardware_io_status_ && commands_.epoch_has_published;
    return std::nullopt;
  }
  if (require_hardware_io_status_) {
    hardware_status = current_hardware_io_status();
    if (!hardware_status || !hardware_status->valid) {
      failure_reason = "Fresh coherent hardware read provenance is unavailable";
      requires_latched_halt = true;
      return std::nullopt;
    }
    measured.positions = Eigen::Map<const Eigen::VectorXd>(
      hardware_status->sample.position.data(),
      static_cast<Eigen::Index>(hardware_status->sample.position.size()));
    measured.velocities = Eigen::Map<const Eigen::VectorXd>(
      hardware_status->sample.velocity.data(),
      static_cast<Eigen::Index>(hardware_status->sample.velocity.size()));
    measured.time_stamp = rclcpp::Time(
      hardware_status->sample.read_stamp_ns, current_time.get_clock_type());
  }
  measured.accelerations.setZero();
  if (!finite_state(measured, parameters_.joint_names.size())) {
    return std::nullopt;
  }
  if (!require_hardware_io_status_) {
    measured.time_stamp = state_time;
  }

  const auto & controller_state = actual_controller_state;
  auto hardware_observation_state = *controller_state;
  if (require_hardware_io_status_) {
    hardware_observation_state.header.stamp = to_builtin_time(measured.time_stamp);
    hardware_observation_state.feedback.positions.assign(
      hardware_status->sample.position.begin(), hardware_status->sample.position.end());
    hardware_observation_state.feedback.velocities.assign(
      hardware_status->sample.velocity.begin(), hardware_status->sample.velocity.end());
    observe_controller_feedback(hardware_observation_state, current_time.get_clock_type());
  } else {
    observe_controller_feedback(*controller_state, current_time.get_clock_type());
  }
  if (require_hardware_io_status_ ?
    !hardware_feedback_motion_within_limits(hardware_status->sample, failure_reason) :
    !feedback_motion_within_limits(*controller_state, failure_reason))
  {
    ++telemetry_.feedback_motion_limit_failure_count;
    requires_latched_halt = true;
    return std::nullopt;
  }
  telemetry_.last_controller_speed_scaling_factor = controller_state->speed_scaling_factor;
  if (std::abs(controller_state->speed_scaling_factor - 1.0) >
    kControllerSpeedScalingTolerance)
  {
    ++telemetry_.controller_speed_scaling_failure_count;
    failure_reason =
      "Trajectory controller speed scaling changed the checked command timeline: factor=" +
      std::to_string(controller_state->speed_scaling_factor);
    requires_latched_halt = true;
    return std::nullopt;
  }

  if (actual_feedback != nullptr) {
    control::JointMotionState feedback_motion_state;
    rclcpp::Time feedback_stamp(
      controller_state->header.stamp, current_time.get_clock_type());
    if (require_hardware_io_status_) {
      if (!hardware_status || !hardware_status->valid ||
        hardware_status->sample.position.size() != parameters_.joint_names.size())
      {
        failure_reason = "Fresh coherent hardware read provenance is unavailable";
        requires_latched_halt = true;
        return std::nullopt;
      }
      feedback_motion_state.position = Eigen::Map<const Eigen::VectorXd>(
        hardware_status->sample.position.data(),
        static_cast<Eigen::Index>(hardware_status->sample.position.size()));
      feedback_motion_state.velocity = Eigen::Map<const Eigen::VectorXd>(
        hardware_status->sample.velocity.data(),
        static_cast<Eigen::Index>(hardware_status->sample.velocity.size()));
      feedback_stamp = rclcpp::Time(
        hardware_status->sample.read_stamp_ns, current_time.get_clock_type());
    } else {
      feedback_motion_state.position = Eigen::Map<const Eigen::VectorXd>(
        controller_state->feedback.positions.data(),
        static_cast<Eigen::Index>(controller_state->feedback.positions.size()));
      feedback_motion_state.velocity = Eigen::Map<const Eigen::VectorXd>(
        controller_state->feedback.velocities.data(),
        static_cast<Eigen::Index>(controller_state->feedback.velocities.size()));
    }
    // JTC state and hardware read snapshots both lack measured acceleration.
    feedback_motion_state.acceleration.setZero(feedback_motion_state.position.size());
    *actual_feedback = std::move(feedback_motion_state);
    if (actual_feedback_stamp != nullptr) {
      *actual_feedback_stamp = feedback_stamp;
    }
  }

  bool following_window_active = false;
  std::optional<control::ExpectedPositionState> expected_hardware_sample;
  if (commands_.epoch_has_published) {
    const rclcpp::Duration controller_period = rclcpp::Duration::from_nanoseconds(
      seconds_to_nanoseconds(parameters_.trajectory_controller_period_sec));
    const auto corrected_times =
      control::PublishedTrajectoryHistory::correctedSampleTimes(
      *controller_state, current_time.get_clock_type(), controller_period);
    const auto * latest_publication = commands_.publications.latest();
    if (latest_publication == nullptr) {
      failure_reason =
        "Published command epoch has no retained checked trajectory";
      ++telemetry_.command_alignment_failure_count;
      requires_latched_halt = true;
      return std::nullopt;
    }
    if (!corrected_times.has_value()) {
      failure_reason =
        "Trajectory controller state has no valid checked publication phase";
      ++telemetry_.command_alignment_failure_count;
      if (require_hardware_io_status_) {
        requires_latched_halt = true;
        return std::nullopt;
      }
      std::string collision_failure;
      const ActualFeedbackSafety feedback_safety =
        validate_controller_feedback_collision_state(
        *controller_state, "Unmatched controller phase", collision_failure);
      if (feedback_safety == ActualFeedbackSafety::kUnsafe) {
        failure_reason += "; " + collision_failure;
        requires_latched_halt = true;
      }
      return std::nullopt;
    }

    const control::PublicationMatch publication_match =
      commands_.publications.match(
      *controller_state, current_time.get_clock_type(), controller_period,
      kControllerCommandAlignmentToleranceRad);
    if (!publication_match.matched()) {
      const auto maximum_expected_difference = [](
        const std::vector<double> & actual, const Eigen::VectorXd & expected) {
          if (actual.size() != static_cast<std::size_t>(expected.size())) {
            return std::numeric_limits<double>::infinity();
          }
          double maximum = 0.0;
          for (std::size_t index = 0; index < actual.size(); ++index) {
            maximum = std::max(
              maximum,
              std::abs(actual[index] - expected[static_cast<Eigen::Index>(index)]));
          }
          return maximum;
        };
      const auto maximum_absolute = [](const std::vector<double> & values) {
          double maximum = 0.0;
          for (const double value : values) {
            maximum = std::max(maximum, std::abs(value));
          }
          return maximum;
        };
      // Several safe future-dated replacements can be published before JTC
      // reaches the first command timestamp. During that bounded phase its
      // old hold is valid only while it still equals the immutable bootstrap
      // seed, not whichever replacement happens to be latest in history.
      const bool preserving_validated_bootstrap_hold =
        commands_.bootstrap_ack_pending &&
        commands_.bootstrap_trajectory_origin.has_value() &&
        commands_.bootstrap_seed_positions.has_value() &&
        corrected_times->reference < *commands_.bootstrap_trajectory_origin &&
        maximum_expected_difference(
          controller_state->reference.positions, *commands_.bootstrap_seed_positions) <=
        kControllerCommandAlignmentToleranceRad &&
        maximum_expected_difference(
          controller_state->output.positions, *commands_.bootstrap_seed_positions) <=
        kControllerCommandAlignmentToleranceRad &&
        maximum_expected_difference(
          controller_state->feedback.positions, *commands_.bootstrap_seed_positions) <=
        execution_error_bound_rad() &&
        maximum_absolute(controller_state->reference.velocities) <=
        kStoppedVelocityRadps &&
        maximum_absolute(controller_state->feedback.velocities) <=
        kStoppedVelocityRadps;
      if (preserving_validated_bootstrap_hold) {
        if (require_hardware_io_status_) {
          const rclcpp::Time read_time(
            hardware_status->sample.read_stamp_ns, current_time.get_clock_type());
          if (read_time >= *commands_.bootstrap_trajectory_origin ||
            hardware_status->sample.position.size() != static_cast<std::size_t>(
              commands_.bootstrap_seed_positions->size()))
          {
            failure_reason =
              "Hardware read cannot be aligned to the verified pre-publication bootstrap hold";
            ++telemetry_.command_alignment_failure_count;
            requires_latched_halt = true;
            return std::nullopt;
          }
          expected_hardware_sample = control::ExpectedPositionState{
            *commands_.bootstrap_seed_positions,
            Eigen::VectorXd::Zero(commands_.bootstrap_seed_positions->size()), true};
          double bootstrap_error = 0.0;
          double bootstrap_speed = 0.0;
          for (std::size_t index = 0; index < hardware_status->sample.position.size(); ++index) {
            const Eigen::Index axis = static_cast<Eigen::Index>(index);
            bootstrap_error = std::max(bootstrap_error, std::abs(
              hardware_status->sample.position[index] -
              expected_hardware_sample->positions[axis]));
            bootstrap_speed = std::max(
              bootstrap_speed, std::abs(hardware_status->sample.velocity[index]));
          }
          if (bootstrap_error > reaction_threshold_rad() ||
            bootstrap_speed > parameters_.following_velocity_tolerance_rad_s)
          {
            failure_reason =
              "Hardware sample does not follow the verified stationary bootstrap hold";
            ++telemetry_.following_error_count;
            requires_latched_halt = true;
            return std::nullopt;
          }
        }
        following_window_active = false;
      } else {
        failure_reason =
          "Trajectory controller reference/output do not match any retained checked "
          "publication";
        ++telemetry_.command_alignment_failure_count;
        if (require_hardware_io_status_) {
          requires_latched_halt = true;
          return std::nullopt;
        }
        std::string collision_failure;
        const ActualFeedbackSafety feedback_safety =
          validate_controller_feedback_collision_state(
          *controller_state, "Unmatched controller phase", collision_failure);
        if (feedback_safety == ActualFeedbackSafety::kUnsafe) {
          failure_reason += "; " + collision_failure;
          requires_latched_halt = true;
        }
        return std::nullopt;
      }
    } else {
      telemetry_.last_controller_reference_alignment_error_rad =
        publication_match.reference_error_rad;
      telemetry_.last_controller_output_alignment_error_rad = publication_match.output_error_rad;
      telemetry_.max_controller_reference_alignment_error_rad = std::max(
        telemetry_.max_controller_reference_alignment_error_rad,
        publication_match.reference_error_rad);
      telemetry_.max_controller_output_alignment_error_rad = std::max(
        telemetry_.max_controller_output_alignment_error_rad, publication_match.output_error_rad);
      telemetry_.command_alignment_check_count += 2U;
      following_window_active = true;

      if (publication_match.kind == control::PublicationMatchKind::kOlder) {
        commands_.active_publication_id = publication_match.publication_id;
        ++telemetry_.older_publication_recovery_count;
        failure_reason =
          "Trajectory controller is executing an older checked publication";
        return std::nullopt;
      }
      commands_.active_publication_id = publication_match.publication_id;
      if (require_hardware_io_status_) {
        const rclcpp::Time read_time(
          hardware_status->sample.read_stamp_ns, current_time.get_clock_type());
        expected_hardware_sample = commands_.publications.expectedAt(
          publication_match.publication_id, read_time);
        if (!expected_hardware_sample.has_value()) {
          failure_reason =
            "Hardware read timestamp is outside the acknowledged checked trajectory";
          ++telemetry_.command_alignment_failure_count;
          requires_latched_halt = true;
          return std::nullopt;
        }
      }
      if (commands_.bootstrap_ack_pending) {
        if (commands_.bootstrap_trajectory_origin.has_value() &&
          *commands_.bootstrap_trajectory_origin ==
          publication_match.sample_times.trajectory_origin)
        {
          ++telemetry_.bootstrap_acknowledgement_count;
          commands_.clearBootstrap();
          commands_.clearRearmDeadlines();
        }
      }
    }
  }

  if (following_window_active) {
    ++telemetry_.following_check_count;
    const auto maximum_absolute_difference = [](
      const std::vector<double> & reference, const std::vector<double> & feedback) {
        double maximum = 0.0;
        std::size_t maximum_index = 0U;
        for (std::size_t index = 0; index < reference.size(); ++index) {
          const double error = std::abs(reference[index] - feedback[index]);
          if (error > maximum) {
            maximum = error;
            maximum_index = index;
          }
        }
        return std::pair<double, std::size_t>{maximum, maximum_index};
      };
    double position_error = 0.0;
    std::size_t position_error_index = 0U;
    double velocity_error = 0.0;
    std::size_t velocity_error_index = 0U;
    if (require_hardware_io_status_) {
      if (!expected_hardware_sample.has_value() ||
        expected_hardware_sample->positions.size() !=
        static_cast<Eigen::Index>(hardware_status->sample.position.size()) ||
        expected_hardware_sample->velocities.size() !=
        static_cast<Eigen::Index>(hardware_status->sample.velocity.size()))
      {
        failure_reason = "No acknowledged trajectory sample covers the hardware read time";
        ++telemetry_.command_alignment_failure_count;
        requires_latched_halt = true;
        return std::nullopt;
      }
      for (std::size_t index = 0; index < hardware_status->sample.position.size(); ++index) {
        const Eigen::Index axis = static_cast<Eigen::Index>(index);
        const double q_error = std::abs(
          expected_hardware_sample->positions[axis] - hardware_status->sample.position[index]);
        const double dq_error = std::abs(
          expected_hardware_sample->velocities[axis] - hardware_status->sample.velocity[index]);
        if (q_error > position_error) {
          position_error = q_error;
          position_error_index = index;
        }
        if (dq_error > velocity_error) {
          velocity_error = dq_error;
          velocity_error_index = index;
        }
      }
    } else {
      const auto errors = maximum_absolute_difference(
        controller_state->reference.positions, controller_state->feedback.positions);
      position_error = errors.first;
      position_error_index = errors.second;
      const auto velocity_errors = maximum_absolute_difference(
        controller_state->reference.velocities, controller_state->feedback.velocities);
      velocity_error = velocity_errors.first;
      velocity_error_index = velocity_errors.second;
    }
    telemetry_.last_following_position_error_rad = position_error;
    telemetry_.last_following_velocity_error_rad_s = velocity_error;
    telemetry_.last_following_position_error_joint = parameters_.joint_names[position_error_index];
    telemetry_.last_following_velocity_error_joint = parameters_.joint_names[velocity_error_index];
    if (position_error > telemetry_.max_following_position_error_rad) {
      telemetry_.max_following_position_error_rad = position_error;
      telemetry_.max_following_position_error_joint =
        telemetry_.last_following_position_error_joint;
    }
    if (velocity_error > telemetry_.max_following_velocity_error_rad_s) {
      telemetry_.max_following_velocity_error_rad_s = velocity_error;
      telemetry_.max_following_velocity_error_joint =
        telemetry_.last_following_velocity_error_joint;
    }
    const control::FollowingErrorPolicy following_policy{
      reaction_threshold_rad(),
      parameters_.following_velocity_tolerance_rad_s,
    };
    const control::FollowingErrorAction following_action =
      control::classifyFollowingError(position_error, velocity_error, following_policy);
    if (following_action != control::FollowingErrorAction::kContinue) {
      ++telemetry_.following_error_count;
      const bool exceeded_profile_growth = hardware_execution_budget_.has_value() &&
        control::executionResponseGrowthExceeded(
        position_error, *hardware_execution_budget_);
      failure_reason =
        "Trajectory following error: position=" + std::to_string(position_error) +
        " rad (" + telemetry_.last_following_position_error_joint + "), velocity=" +
        std::to_string(velocity_error) + " rad/s (" +
        telemetry_.last_following_velocity_error_joint + ")";
      if (exceeded_profile_growth) {
        failure_reason += "; measured growth exceeded the validated T + R response bound";
      }
      // Validate the exact JTC feedback sample that triggered recovery. The
      // independently updated PlanningSceneMonitor state can be one callback
      // behind and is not sufficient for this transition.
      std::string collision_failure;
      const ActualFeedbackSafety feedback_safety =
        validate_controller_feedback_collision_state(
        *controller_state, "Following-error recovery", collision_failure);
      if (feedback_safety != ActualFeedbackSafety::kSafe) {
        failure_reason += "; " + collision_failure;
        requires_latched_halt = true;
        return std::nullopt;
      }

      // Stop extending the command and let its checked braking suffix run.
      // Following accuracy alone never creates a permanent halt: the
      // controlled re-arm path checks physical bounds and collision state on
      // every sample, then waits for a settled hold.
      requires_latched_halt = exceeded_profile_growth;
      return std::nullopt;
    }
  }

  while (!commands_.queue.empty() &&
    commands_.queue.front().time_stamp <
    current_time - rclcpp::Duration::from_seconds(servo_parameters_.max_expected_latency))
  {
    commands_.queue.pop_front();
  }
  if (!commands_.queue.empty() && commands_.queue.back().time_stamp > current_time) {
    initial_state_received_ = true;
    return commands_.queue.back();
  }
  if (commands_.epoch_active) {
    failure_reason = "Buffered command timeline was exhausted";
    return std::nullopt;
  }
  commands_.queue.clear();
  const auto retained_hold = applyProvenHardwareHold(
    measured, controller_state, failure_reason, requires_latched_halt);
  if (!retained_hold.has_value()) {
    return std::nullopt;
  }
  const bool retained_stationary_hold_used = *retained_hold;
  if (require_hardware_io_status_ && !retained_stationary_hold_used) {
    for (std::size_t index = 0; index < position_bound_sets_.size(); ++index) {
      if (!control::isWireCommand(
          measured.positions[static_cast<Eigen::Index>(index)],
          position_bound_sets_[index].wire))
      {
        failure_reason = "Hardware state is outside the wire interval without a proven "
          "stationary command hold for " + parameters_.joint_names[index];
        requires_latched_halt = true;
        return std::nullopt;
      }
    }
  }
  measured.time_stamp = current_time;
  initial_state_received_ = true;
  return measured;
}

std::optional<bool> CollisionAwareServoComponent::applyProvenHardwareHold(
  moveit_servo::KinematicState & measured,
  const std::optional<control_msgs::msg::JointTrajectoryControllerState> & controller_state,
  std::string & failure_reason, bool & requires_latched_halt)
{
  bool retained_stationary_hold_used = false;
  if (require_hardware_io_status_ && proven_q_hold_.has_value() &&
    cycle_hardware_io_status_.has_value() &&
    proven_q_hold_epoch_ == cycle_hardware_io_status_->sample.epoch &&
    controller_state.has_value())
  {
    if (proven_q_hold_->size() != measured.positions.size()) {
      failure_reason = "Retained stationary hardware command has the wrong joint count";
      requires_latched_halt = true;
      return std::nullopt;
    }
    const auto controller_matches_hold = [&](const std::vector<double> & values) {
        if (values.size() != static_cast<std::size_t>(proven_q_hold_->size())) {
          return false;
        }
        for (std::size_t index = 0; index < values.size(); ++index) {
          if (std::abs(values[index] - (*proven_q_hold_)[static_cast<Eigen::Index>(index)]) >
            kControllerCommandAlignmentToleranceRad)
          {
            return false;
          }
        }
        return true;
      };
    const bool controller_hold_owned = controller_matches_hold(
      controller_state->reference.positions) &&
      controller_matches_hold(controller_state->output.positions) &&
      std::all_of(
      controller_state->reference.velocities.begin(),
      controller_state->reference.velocities.end(), [](const double velocity) {
        return std::abs(velocity) <= kStoppedVelocityRadps;
      }) &&
      std::all_of(
      controller_state->output.velocities.begin(),
      controller_state->output.velocities.end(), [](const double velocity) {
        return std::abs(velocity) <= kStoppedVelocityRadps;
      });
    bool actual_within_execution_budget = controller_hold_owned &&
      hardware_execution_error_rad_.size() == position_bound_sets_.size() &&
      motion_limits_.numerical_policy.has_value();
    if (actual_within_execution_budget) {
      const double arithmetic_error =
        motion_limits_.numerical_policy->position_error.value;
      for (std::size_t index = 0; index < position_bound_sets_.size(); ++index) {
        const Eigen::Index axis = static_cast<Eigen::Index>(index);
        const double sample = measured.positions[axis];
        const auto sample_uncertainty = control::floatMeasurementHalfUlp(sample);
        const double execution_error = hardware_execution_error_rad_[index];
        if (!sample_uncertainty || !std::isfinite(execution_error) || execution_error < 0.0 ||
          std::abs(sample - (*proven_q_hold_)[axis]) >
          execution_error + sample_uncertainty->value + arithmetic_error ||
          std::abs(measured.velocities[axis]) > kStoppedVelocityRadps)
        {
          actual_within_execution_budget = false;
          break;
        }
      }
    }
    if (actual_within_execution_budget) {
      measured.positions = *proven_q_hold_;
      measured.velocities.setZero();
      measured.accelerations.setZero();
      retained_stationary_hold_used = true;
    } else {
      proven_q_hold_.reset();
      proven_q_hold_epoch_.reset();
      for (std::size_t index = 0; index < position_bound_sets_.size(); ++index) {
        if (!control::isWireCommand(
            measured.positions[static_cast<Eigen::Index>(index)],
            position_bound_sets_[index].wire))
        {
          failure_reason = "Hardware state is outside the wire interval without a matching "
            "stationary command ownership and execution-error proof for " +
            parameters_.joint_names[index];
          requires_latched_halt = true;
          return std::nullopt;
        }
      }
    }
  } else {
    if (proven_q_hold_.has_value() &&
      (!cycle_hardware_io_status_.has_value() ||
      proven_q_hold_epoch_ != cycle_hardware_io_status_->sample.epoch))
    {
      proven_q_hold_.reset();
      proven_q_hold_epoch_.reset();
    }
  }
  return retained_stationary_hold_used;
}

void CollisionAwareServoComponent::observe_controller_feedback(
  const control_msgs::msg::JointTrajectoryControllerState & state,
  const rcl_clock_type_t clock_type)
{
  const rclcpp::Time stamp(state.header.stamp, clock_type);
  if (last_feedback_sample_time_.has_value() && stamp <= *last_feedback_sample_time_) {
    return;
  }
  const Eigen::Map<const Eigen::VectorXd> velocity(
    state.feedback.velocities.data(),
    static_cast<Eigen::Index>(state.feedback.velocities.size()));
  Eigen::Index velocity_index = 0;
  const double maximum_velocity = velocity.cwiseAbs().maxCoeff(&velocity_index);
  if (maximum_velocity > telemetry_.max_feedback_joint_velocity_rad_s) {
    telemetry_.max_feedback_joint_velocity_rad_s = maximum_velocity;
    telemetry_.max_feedback_joint_velocity_name =
      parameters_.joint_names[static_cast<std::size_t>(velocity_index)];
  }

  if (last_feedback_sample_time_.has_value() && last_feedback_velocity_.has_value()) {
    const double elapsed_sec = (stamp - *last_feedback_sample_time_).seconds();
    const Eigen::VectorXd acceleration =
      (velocity - *last_feedback_velocity_) / elapsed_sec;
    if (acceleration.allFinite()) {
      Eigen::Index acceleration_index = 0;
      const double maximum_acceleration =
        acceleration.cwiseAbs().maxCoeff(&acceleration_index);
      if (maximum_acceleration > telemetry_.max_feedback_joint_acceleration_rad_s2) {
        telemetry_.max_feedback_joint_acceleration_rad_s2 = maximum_acceleration;
        telemetry_.max_feedback_joint_acceleration_name =
          parameters_.joint_names[static_cast<std::size_t>(acceleration_index)];
      }
      if (last_feedback_acceleration_.has_value()) {
        const Eigen::VectorXd jerk =
          (acceleration - *last_feedback_acceleration_) / elapsed_sec;
        if (jerk.allFinite()) {
          Eigen::Index jerk_index = 0;
          const double maximum_jerk = jerk.cwiseAbs().maxCoeff(&jerk_index);
          if (maximum_jerk > telemetry_.max_feedback_joint_jerk_rad_s3) {
            telemetry_.max_feedback_joint_jerk_rad_s3 = maximum_jerk;
            telemetry_.max_feedback_joint_jerk_name =
              parameters_.joint_names[static_cast<std::size_t>(jerk_index)];
          }
        }
      }
      last_feedback_acceleration_ = acceleration;
    }
  }
  last_feedback_sample_time_ = stamp;
  last_feedback_velocity_ = velocity;
}

bool CollisionAwareServoComponent::feedback_motion_within_limits(
  const control_msgs::msg::JointTrajectoryControllerState & state,
  std::string & failure_reason) const
{
  const Eigen::Map<const Eigen::VectorXd> position(
    state.feedback.positions.data(),
    static_cast<Eigen::Index>(state.feedback.positions.size()));
  const Eigen::Map<const Eigen::VectorXd> velocity(
    state.feedback.velocities.data(),
    static_cast<Eigen::Index>(state.feedback.velocities.size()));
  const Eigen::VectorXd & safe_lower = motion_limits_.lower_position;
  const Eigen::VectorXd & safe_upper = motion_limits_.upper_position;

  if ((position.array() < safe_lower.array() -
    control::kControllerPositionFeedbackTolerance.value).any() ||
    (position.array() > safe_upper.array() +
    control::kControllerPositionFeedbackTolerance.value).any())
  {
    failure_reason = "Measured joint position exceeds the robot joint limits";
    return false;
  }
  if (physical_joint_velocity_limits_.size() != velocity.size() ||
    (velocity.cwiseAbs().array() >
    physical_joint_velocity_limits_.array() + kFeedbackLimitTolerance).any())
  {
    failure_reason = "Measured joint velocity exceeds the robot's physical URDF limit";
    return false;
  }
  // The controller reports position and velocity, but not acceleration or
  // jerk. Their finite-difference estimates remain useful telemetry, yet are
  // too sensitive to sample jitter to serve as a hard stop. The published
  // command itself is still constrained by the exact acceleration/jerk
  // limits in every QP step and in the braking-tail generator.
  return true;
}

bool CollisionAwareServoComponent::hardware_feedback_motion_within_limits(
  const hardware::HardwareIoSample & sample, std::string & failure_reason) const
{
  if (sample.position.size() != parameters_.joint_names.size() ||
    sample.velocity.size() != parameters_.joint_names.size() ||
    physical_joint_velocity_limits_.size() !=
    static_cast<Eigen::Index>(sample.velocity.size()))
  {
    failure_reason = "Coherent hardware feedback joint-count mismatch: positions=" +
      std::to_string(sample.position.size()) + ", velocities=" +
      std::to_string(sample.velocity.size()) + ", joints=" +
      std::to_string(parameters_.joint_names.size()) + ", velocity_limits=" +
      std::to_string(physical_joint_velocity_limits_.size());
    return false;
  }
  for (std::size_t index = 0; index < sample.velocity.size(); ++index) {
    if (!std::isfinite(sample.velocity[index]) ||
      std::abs(sample.velocity[index]) >
      physical_joint_velocity_limits_[static_cast<Eigen::Index>(index)] +
      kFeedbackLimitTolerance)
    {
      failure_reason = "Measured hardware velocity exceeds physical URDF limit for " +
        parameters_.joint_names[index];
      return false;
    }
  }
  return true;
}

}  // namespace face_tracking_arm
