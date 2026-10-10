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
  measured_robot_state->setJointGroupPositions(
    joint_model_group_, controller_state.feedback.positions);
  measured_robot_state->update(true);

  planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
  if (!planning_scene) {
    reason = context + " lost the PlanningScene";
    return ActualFeedbackSafety::kUnsafe;
  }
  const control::CollisionActualStateValidationResult validation =
    collision_constraint_builder_->validateActualState(
    *planning_scene, *measured_robot_state, *joint_model_group_);
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
  const rclcpp::Time sample_time(state->header.stamp, current_time.get_clock_type());
  if (commands_.last_rearm_sample_time.has_value() &&
    sample_time <= *commands_.last_rearm_sample_time)
  {
    reason = "Safe re-arm is waiting for the next controller feedback sample";
    return RearmReadiness::kWaiting;
  }
  commands_.last_rearm_sample_time = sample_time;
  observe_controller_feedback(*state, current_time.get_clock_type());

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
    state->reference.positions, state->feedback.positions);
  const double reference_velocity = maximum_absolute(state->reference.velocities);
  const double feedback_velocity = maximum_absolute(state->feedback.velocities);

  if (std::abs(state->speed_scaling_factor - 1.0) >
    kControllerSpeedScalingTolerance)
  {
    ++telemetry_.controller_speed_scaling_failure_count;
    reason = "Safe re-arm detected trajectory-controller speed scaling";
    return RearmReadiness::kUnsafe;
  }

  std::string feedback_failure;
  if (!feedback_motion_within_limits(*state, feedback_failure)) {
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
    kControllerCommandAlignmentToleranceRad, parameters_.collision_tracking_error_bound_rad,
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
  measured.accelerations.setZero();
  if (!finite_state(measured, parameters_.joint_names.size())) {
    return std::nullopt;
  }
  measured.time_stamp = state_time;

  const auto controller_state = current_controller_state();
  if (!controller_state.has_value()) {
    failure_reason = "Fresh finite trajectory controller state is unavailable";
    return std::nullopt;
  }
  observe_controller_feedback(*controller_state, current_time.get_clock_type());
  if (!feedback_motion_within_limits(*controller_state, failure_reason)) {
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
    feedback_motion_state.position = Eigen::Map<const Eigen::VectorXd>(
      controller_state->feedback.positions.data(),
      static_cast<Eigen::Index>(controller_state->feedback.positions.size()));
    feedback_motion_state.velocity = Eigen::Map<const Eigen::VectorXd>(
      controller_state->feedback.velocities.data(),
      static_cast<Eigen::Index>(controller_state->feedback.velocities.size()));
    // JTC state has no measured acceleration field.
    feedback_motion_state.acceleration.setZero(feedback_motion_state.position.size());
    *actual_feedback = std::move(feedback_motion_state);
    if (actual_feedback_stamp != nullptr) {
      *actual_feedback_stamp = rclcpp::Time(
        controller_state->header.stamp, current_time.get_clock_type());
    }
  }

  bool following_window_active = false;
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
        parameters_.collision_tracking_error_bound_rad &&
        maximum_absolute(controller_state->reference.velocities) <=
        kStoppedVelocityRadps &&
        maximum_absolute(controller_state->feedback.velocities) <=
        kStoppedVelocityRadps;
      if (preserving_validated_bootstrap_hold) {
        following_window_active = false;
      } else {
        failure_reason =
          "Trajectory controller reference/output do not match any retained checked "
          "publication";
        ++telemetry_.command_alignment_failure_count;
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
    const auto [position_error, position_error_index] = maximum_absolute_difference(
      controller_state->reference.positions, controller_state->feedback.positions);
    const auto [velocity_error, velocity_error_index] = maximum_absolute_difference(
      controller_state->reference.velocities, controller_state->feedback.velocities);
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
      parameters_.following_position_tolerance_rad,
      parameters_.following_velocity_tolerance_rad_s,
    };
    const control::FollowingErrorAction following_action =
      control::classifyFollowingError(position_error, velocity_error, following_policy);
    if (following_action != control::FollowingErrorAction::kContinue) {
      ++telemetry_.following_error_count;
      failure_reason =
        "Trajectory following error: position=" + std::to_string(position_error) +
        " rad (" + telemetry_.last_following_position_error_joint + "), velocity=" +
        std::to_string(velocity_error) + " rad/s (" +
        telemetry_.last_following_velocity_error_joint + ")";
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
      requires_latched_halt = false;
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
  measured.time_stamp = current_time;
  initial_state_received_ = true;
  return measured;
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

  if ((position.array() < safe_lower.array() - kFeedbackLimitTolerance).any() ||
    (position.array() > safe_upper.array() + kFeedbackLimitTolerance).any())
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

}  // namespace face_tracking_arm
