// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/motion_reference.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include "face_tracking_arm/pointing_geometry.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/compact_maneuver.hpp"
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"
#include "face_tracking_arm/face_target_age_policy.hpp"
#include "face_tracking_arm/snapshot_timing.hpp"
#include "servo_types.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kRestArrivalRad{0.025};
constexpr double kRestReleaseRad{0.060};
constexpr double kSettledVelocityRadps{0.01};
constexpr double kSettledAccelerationRadps2{0.03};
constexpr double kCompactRollLimitRad{0.0890117918517108};
const double kCompactEdgeSineLimit{std::sin(kCompactRollLimitRad)};
// The queue-tail stopping envelope already contains the published remainder.
constexpr double kCompactQueueTailResidualLatencySec{0.0};

std::string jsonNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string j1BoundaryAssessmentJson(const J1BoundaryAssessment & assessment)
{
  return "{\"actual_position_rad\":" + jsonNumber(assessment.actual_position_rad) +
         ",\"brake_min_position_rad\":" + jsonNumber(assessment.brake_min_position_rad) +
         ",\"brake_max_position_rad\":" + jsonNumber(assessment.brake_max_position_rad) +
         ",\"working_lower_rad\":" + jsonNumber(assessment.working_lower_rad) +
         ",\"working_upper_rad\":" + jsonNumber(assessment.working_upper_rad) +
         ",\"edge_direction\":" +
         std::to_string(static_cast<int>(assessment.edge_direction)) + "}";
}

template<typename T>
T readOrDeclareParameter(
  const rclcpp::Node::SharedPtr & node, const std::string & name, const T & fallback,
  const rcl_interfaces::msg::ParameterDescriptor & descriptor)
{
  if (node->has_parameter(name)) {
    T value{};
    node->get_parameter(name, value);
    return value;
  }
  return node->declare_parameter<T>(name, fallback, descriptor);
}

const moveit::core::LinkModel * rigidRoot(const moveit::core::LinkModel * link)
{
  while (link->getParentJointModel() &&
    link->getParentJointModel()->getType() == moveit::core::JointModel::FIXED &&
    link->getParentLinkModel())
  {
    link = link->getParentLinkModel();
  }
  return link;
}

std::int64_t toNanoseconds(const double time_sec)
{
  if (!std::isfinite(time_sec) || time_sec < 0.0 ||
    time_sec > static_cast<double>(std::numeric_limits<std::int64_t>::max()) * 1.0e-9)
  {
    return 0;
  }
  return static_cast<std::int64_t>(std::llround(time_sec * 1.0e9));
}

bool isFreshActualFeedback(
  const rclcpp::Clock::SharedPtr & clock, const double tick_time_sec,
  const std::optional<double> & feedback_time_sec, const double timeout_sec)
{
  if (!feedback_time_sec || !std::isfinite(tick_time_sec) ||
    !std::isfinite(*feedback_time_sec) || !std::isfinite(timeout_sec) || timeout_sec < 0.0)
  {
    return false;
  }
  if (clock && clock->get_clock_type() == RCL_ROS_TIME && clock->ros_time_is_active()) {
    const std::int64_t tick_ns = toNanoseconds(tick_time_sec);
    const std::int64_t feedback_ns = toNanoseconds(*feedback_time_sec);
    const std::int64_t timeout_ns = toNanoseconds(timeout_sec);
    return isFreshSnapshotTimestamp(
      feedback_ns, tick_ns, timeout_ns, kSimTimeFeedbackFutureToleranceNs);
  }
  return tick_time_sec >= *feedback_time_sec &&
         tick_time_sec - *feedback_time_sec <= timeout_sec;
}

std::int64_t sourceFaceStampNs(const msg::TrackingTarget & target)
{
  return static_cast<std::int64_t>(target.face_stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(target.face_stamp.nanosec);
}

std::int64_t commandStampNs(const msg::TrackingTarget & target)
{
  return static_cast<std::int64_t>(target.header.stamp.sec) * 1'000'000'000LL +
         static_cast<std::int64_t>(target.header.stamp.nanosec);
}

bool isFreshMeasuredFace(
  const msg::TrackingTarget & target, const double time_sec,
  const double freshness_timeout_sec)
{
  if (target.mode != msg::TrackingTarget::FACE || target.face_is_prediction ||
    !std::isfinite(time_sec) || !std::isfinite(freshness_timeout_sec) ||
    freshness_timeout_sec <= 0.0 ||
    time_sec > static_cast<double>(std::numeric_limits<std::int64_t>::max()) * 1.0e-9)
  {
    return false;
  }
  return validFaceTargetAge(
    sourceFaceStampNs(target), commandStampNs(target), toNanoseconds(time_sec),
    static_cast<std::int64_t>(std::llround(freshness_timeout_sec * 1.0e9)),
    false, 0, 0);
}

std::optional<LinearVelocityConstraint> makeOutwardStoppingConstraint(
  const Eigen::VectorXd & outward_gradient, const double remaining_distance,
  const JointMotionState & motion_state, const JointMotionLimits & limits,
  const double period_sec, const double residual_latency_sec)
{
  if (outward_gradient.size() != motion_state.position.size() ||
    !outward_gradient.allFinite() || !std::isfinite(remaining_distance))
  {
    return std::nullopt;
  }
  double maximum_outward_speed = 0.0;
  try {
    maximum_outward_speed = maximumSafeCollisionApproachSpeed(
      std::max(0.0, remaining_distance), -outward_gradient, motion_state, limits,
      period_sec, residual_latency_sec);
    if (motion_state.velocity.lpNorm<Eigen::Infinity>() <= 1.0e-12 &&
      motion_state.acceleration.lpNorm<Eigen::Infinity>() <= 1.0e-12)
    {
      // A stationary command can hold exactly on a virtual envelope boundary.
      maximum_outward_speed = std::max(0.0, maximum_outward_speed);
    }
  } catch (const std::exception &) {
    return std::nullopt;
  }
  if (!std::isfinite(maximum_outward_speed)) {
    return std::nullopt;
  }
  LinearVelocityConstraint row;
  row.coefficients = outward_gradient;
  row.upper_bound = maximum_outward_speed;
  return row;
}

}  // namespace

double preferredBaseJointAngle(
  const Eigen::Vector3d & face_in_planning_frame,
  const Eigen::Isometry3d & base_pose_in_planning_frame)
{
  const Eigen::Vector3d ray_in_base = base_pose_in_planning_frame.linear().transpose() *
    (face_in_planning_frame - base_pose_in_planning_frame.translation());
  return std::atan2(ray_in_base.y(), ray_in_base.x());
}

MotionReference::MotionReference(
  const rclcpp::Node::SharedPtr & node,
  const planning_scene_monitor::PlanningSceneMonitorPtr & scene_monitor,
  ControllerParameters parameters, JointMotionLimits limits,
  CollisionConstraintConfig collision_config)
: clock_(node->has_parameter("require_hardware_io_status") &&
    node->get_parameter("require_hardware_io_status").as_bool() ?
    rclcpp::Clock::SharedPtr{} : node->get_clock()),
  parameters_(std::move(parameters)), limits_(std::move(limits)),
  collision_validator_(collision_config), robot_model_(scene_monitor->getRobotModel()),
  group_(robot_model_->getJointModelGroup(parameters_.planning_group_name)),
  command_link_(robot_model_->getLinkModel(parameters_.command_frame)),
  base_link_(robot_model_->getLinkModel("link_base"))
{
  if (parameters_.recovery_mode == "compact") {
    recovery_mode_ = RecoveryMode::kCompact;
  } else if (parameters_.recovery_mode == "disabled") {
    recovery_mode_ = RecoveryMode::kDisabled;
  } else {
    throw std::invalid_argument("recovery.mode must be either 'compact' or 'disabled'");
  }
  diagnostics_.recovery_mode = parameters_.recovery_mode;

  if (!base_link_) {
    base_link_ = robot_model_->getRootLink();
  }
  if (!group_ || !command_link_ || !base_link_) {
    throw std::invalid_argument("Motion reference requires the planning group and screen link");
  }
  if (!parameters_.gaze_frame.empty()) {
    const auto * gaze_link = robot_model_->getLinkModel(parameters_.gaze_frame);
    if (!gaze_link || rigidRoot(gaze_link) != rigidRoot(command_link_)) {
      throw std::invalid_argument(
              "gaze_frame must be a camera optical link rigidly fixed to the screen");
    }
  }

  TargetMotionEstimatorConfig target_motion_config;
  target_motion_config.stale_timeout_sec = parameters_.source_timeout_sec;
  target_motion_config.maximum_sample_interval_sec = parameters_.source_timeout_sec;
  target_motion_config.maximum_source_gap_sec = parameters_.source_timeout_sec;
  target_motion_config.timing_mode = parameters_.timing_mode == "auto" ?
    MotionTimingMode::kAuto : MotionTimingMode::kManual;
  target_motion_estimator_ = TargetMotionEstimator(target_motion_config);

  moveit::core::RobotState named_state(robot_model_);
  named_state.setToDefaultValues();
  if (!named_state.setToDefaultValues(group_, "rest")) {
    throw std::invalid_argument("Motion reference requires a named rest posture");
  }
  named_state.copyJointGroupPositions(group_, rest_positions_);

  posture_task_builder_.emplace(robot_model_, group_, limits_, PostureTaskBuilderParameters{
      parameters_.position_gain, parameters_.joint_position_margin_rad,
      parameters_.maximum_linear_reference_mps, parameters_.maximum_angular_reference_radps,
      parameters_.control_period_sec, parameters_.residual_command_latency_sec});

  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  search_enabled_ = readOrDeclareParameter(node, "search_enabled", true, descriptor);
  if (search_enabled_) {
    if (!named_state.setToDefaultValues(group_, "search")) {
      throw std::invalid_argument("Idle search requires a named search posture");
    }
    named_state.copyJointGroupPositions(group_, search_positions_);
    const auto & names = group_->getVariableNames();
    const auto base = std::find(names.begin(), names.end(), "joint1");
    if (base == names.end()) {
      throw std::invalid_argument("Idle search requires joint1");
    }
    search_base_index_ = std::distance(names.begin(), base);
    search_config_.speed_rad_s = readOrDeclareParameter(
      node, "search_speed_rad_s", 0.30, descriptor);
    search_config_.sweep_half_range_rad = readOrDeclareParameter(
      node, "search_sweep_half_range_rad", 2.80, descriptor);
    search_config_.local_half_range_rad = readOrDeclareParameter(
      node, "search_local_half_range_rad", 0.35, descriptor);
    search_config_.local_duration_sec = readOrDeclareParameter(
      node, "search_local_duration_sec", 6.0, descriptor);
    search_.emplace();
    search_->start(rest_positions_, search_positions_, limits_.lower_position +
      limits_.position_margin, limits_.upper_position - limits_.position_margin,
      search_base_index_, search_config_, false, 0.0);
  }

  BackgroundPathPlannerConfig planner_config;
  planner_config.move_group_name = parameters_.planning_group_name;
  planner_ = std::make_unique<BackgroundPathPlanner>(
    node, robot_model_, scene_monitor, planner_config,
    [validator = collision_validator_, limits = limits_, group = group_] (
      const planning_scene::PlanningScene & scene, const moveit::core::RobotState & state)
    {
      Eigen::VectorXd position;
      state.copyJointGroupPositions(group, position);
      if ((position.array() < limits.lower_position.array()).any() ||
      (position.array() > limits.upper_position.array()).any())
      {
        return false;
      }
      JointMotionState stationary{position, Eigen::VectorXd::Zero(position.size()),
        Eigen::VectorXd::Zero(position.size())};
      const auto validation = validator.validatePose(scene, state, *group, stationary);
      return validation.input_valid && !validation.unsafe;
    });
  compact_maneuver_ = std::make_unique<CompactManeuver>(
    node, recovery_mode_ == RecoveryMode::kCompact, parameters_, limits_,
    collision_validator_, robot_model_, group_, command_link_, base_link_,
    posture_task_builder_ ? &*posture_task_builder_ : nullptr, clock_,
    *planner_, follower_, diagnostics_);
}

MotionReference::~MotionReference() = default;

std::vector<CompactEvent> MotionReference::drainCompactEvents()
{
  return compact_maneuver_->drainEvents();
}

bool MotionReference::compactManeuverActive() const noexcept
{
  return compact_maneuver_->active();
}

bool MotionReference::hasActiveCompactProfile() const noexcept
{
  return compact_maneuver_->hasActiveProfile();
}

const JointMotionLimits & MotionReference::activeCommandLimits() const noexcept
{
  return compact_maneuver_->activeCommandLimits();
}

void MotionReference::recordCompactProfileCommandPublished(const bool published) noexcept
{
  compact_maneuver_->recordProfileCommandPublished(published);
}

EmergencyBrakeTail MotionReference::compactProfileBrakingTail(
  const JointMotionState & candidate_state, const EmergencyBrakeTailConfig & config) const
{
  return compact_maneuver_->profileBrakingTail(candidate_state, config);
}

bool MotionReference::validateCompactPathPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  return compact_maneuver_->validateCompactPathPosition(state, position, violation);
}

bool MotionReference::maybeArmFaceUprightRollGuard(
  const moveit::core::RobotState & state, const double time_sec,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  if (upright_roll_guard_armed_) {
    return true;
  }
  if (!actual_feedback || !command_queue_state ||
    !isFreshActualFeedback(
      clock_, time_sec, actual_feedback_time_sec, parameters_.state_feedback_timeout_sec))
  {
    return false;
  }
  const auto joint_count = static_cast<Eigen::Index>(group_->getVariableCount());
  if (actual_feedback->position.size() != joint_count ||
    actual_feedback->velocity.size() != joint_count ||
    command_queue_state->position.size() != joint_count ||
    command_queue_state->velocity.size() != joint_count ||
    command_queue_state->acceleration.size() != joint_count ||
    !actual_feedback->position.allFinite() || !actual_feedback->velocity.allFinite() ||
    !command_queue_state->position.allFinite() || !command_queue_state->velocity.allFinite() ||
    !command_queue_state->acceleration.allFinite())
  {
    return false;
  }

  const auto edge_tilt_at = [this, &state](const Eigen::VectorXd & position)
    -> std::optional<double> {
      moveit::core::RobotState candidate(state);
      candidate.setJointGroupPositions(group_, position);
      candidate.update(true);
      const auto edge = evaluateHorizontalEdgeUprightWithSineCap(
        candidate.getGlobalLinkTransform(command_link_).linear(), kCompactEdgeSineLimit);
      if (!edge) {
        return std::nullopt;
      }
      return std::abs(std::asin(std::clamp(edge->signed_edge_sine, -1.0, 1.0)));
    };
  const auto actual_roll = edge_tilt_at(actual_feedback->position);
  const auto command_roll = edge_tilt_at(command_queue_state->position);
  if (!actual_roll || !command_roll) {
    return false;
  }

  const bool active_available = active_publication != nullptr;
  const bool latest_available = latest_publication != nullptr;
  const bool no_publication_history = !active_available && !latest_available;
  if (active_available != latest_available) {
    return false;
  }
  const auto has_stopped_record = [joint_count](
    const PublishedTrajectoryRecord * publication) {
      if (publication == nullptr || publication->execution_queue.empty() ||
        !std::isfinite(publication->stationary_time.seconds()))
      {
        return false;
      }
      const auto stationary = std::find_if(
        publication->execution_queue.begin(), publication->execution_queue.end(),
        [publication](const auto & sample) {
          return sample.time_stamp == publication->stationary_time;
        });
      if (stationary == publication->execution_queue.end() ||
        stationary->positions.size() != joint_count ||
        stationary->velocities.size() != joint_count ||
        stationary->accelerations.size() != joint_count ||
        !stationary->positions.allFinite() || !stationary->velocities.allFinite() ||
        !stationary->accelerations.allFinite())
      {
        return false;
      }
      return stationary->velocities.lpNorm<Eigen::Infinity>() <= 1.0e-6 &&
             stationary->accelerations.lpNorm<Eigen::Infinity>() <= 1.0e-6;
    };
  if (!no_publication_history &&
    (!has_stopped_record(active_publication) ||
    !has_stopped_record(latest_publication)))
  {
    return false;
  }
  if (no_publication_history) {
    const bool positions_match =
      (actual_feedback->position - command_queue_state->position).
      lpNorm<Eigen::Infinity>() <= kRestArrivalRad;
    if (!positions_match || !motionSettled(*actual_feedback, command_queue_state)) {
      return false;
    }
  }

  double maximum_abs_roll_rad = std::max(*actual_roll, *command_roll);
  const auto include_edge_publication = [time_sec, &edge_tilt_at,
      &maximum_abs_roll_rad](const PublishedTrajectoryRecord * publication) {
      if (publication == nullptr || publication->execution_queue.empty()) {
        return true;
      }
      const double stop_time = publication->stationary_time.seconds();
      if (!std::isfinite(stop_time)) {
        return false;
      }
      if (time_sec >= stop_time) {
        return true;
      }
      const moveit_servo::KinematicState * previous = nullptr;
      std::size_t next_index = publication->execution_queue.size();
      for (std::size_t index = 0; index < publication->execution_queue.size(); ++index) {
        const auto & sample = publication->execution_queue[index];
        const double sample_time = sample.time_stamp.seconds();
        if (!std::isfinite(sample_time)) {
          return false;
        }
        if (sample_time < time_sec) {
          previous = &sample;
        } else {
          next_index = index;
          break;
        }
      }
      const auto include_position = [&edge_tilt_at, &maximum_abs_roll_rad](
        const Eigen::VectorXd & position) {
          const auto tilt = edge_tilt_at(position);
          if (!tilt) {
            return false;
          }
          maximum_abs_roll_rad = std::max(maximum_abs_roll_rad, *tilt);
          return true;
        };
      if (previous != nullptr && next_index < publication->execution_queue.size()) {
        const auto & next = publication->execution_queue[next_index];
        const double before_time = previous->time_stamp.seconds();
        const double after_time = next.time_stamp.seconds();
        if (!std::isfinite(before_time) || !std::isfinite(after_time) ||
          after_time <= before_time || previous->positions.size() != next.positions.size() ||
          !previous->positions.allFinite() || !next.positions.allFinite() ||
          !include_position(previous->positions))
        {
          return false;
        }
        const auto include_interpolated = [&](const double at_time) {
            const double fraction = std::clamp(
              (at_time - before_time) / (after_time - before_time), 0.0, 1.0);
            return include_position(
              previous->positions + fraction * (next.positions - previous->positions));
          };
        if (!include_interpolated(time_sec) ||
          (after_time > stop_time && before_time < stop_time &&
          !include_interpolated(stop_time)))
        {
          return false;
        }
      } else if (previous != nullptr && !include_position(previous->positions)) {
        return false;
      }
      for (std::size_t index = next_index;
        index < publication->execution_queue.size(); ++index)
      {
        const auto & sample = publication->execution_queue[index];
        const double sample_time = sample.time_stamp.seconds();
        if (sample_time > stop_time) {
          break;
        }
        if (sample_time >= time_sec && !include_position(sample.positions)) {
          return false;
        }
      }
      return true;
    };
  if (!include_edge_publication(active_publication) ||
    (latest_publication != active_publication &&
    !include_edge_publication(latest_publication)))
  {
    return false;
  }

  upright_roll_guard_armed_ = true;
  diagnostics_.upright_roll_guard_armed = true;
  const std::string details =
    "{\"actual_abs_roll_error_rad\":" + jsonNumber(*actual_roll) +
    ",\"command_abs_roll_error_rad\":" + jsonNumber(*command_roll) +
    ",\"fullstop_max_abs_roll_error_rad\":" + jsonNumber(maximum_abs_roll_rad) +
    ",\"maximum_abs_roll_cap_rad\":" + jsonNumber(kCompactRollLimitRad) + "}";
  compact_maneuver_->recordFaceRollGuardArmed(
    time_sec, "actual_command_and_published_stop_envelope_within_cap", details);
  return true;
}

bool MotionReference::validateUprightRollPose(
  const moveit::core::RobotState & state, std::string * violation) const
{
  if (!compactManeuverActive()) {
    const auto edge = evaluateHorizontalEdgeUprightWithSineCap(
      state.getGlobalLinkTransform(command_link_).linear(), kCompactEdgeSineLimit);
    if (!edge) {
      if (violation) {
        *violation = "screen roll";
      }
      return false;
    }
    return true;
  }
  return true;
}

bool MotionReference::validateUprightRollPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  if (violation) {
    violation->clear();
  }
  if (!group_ || !command_link_ ||
    position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    !position.allFinite())
  {
    if (violation) {
      *violation = "screen roll state is unavailable";
    }
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, position);
  candidate.update(true);
  return validateUprightRollPose(candidate, violation);
}

bool MotionReference::validateCommandPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  Eigen::VectorXd previous_position;
  if (group_) {
    state.copyJointGroupPositions(group_, previous_position);
  }
  return validateCommandTransition(state, previous_position, position, violation);
}

bool MotionReference::validateCommandTransition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & previous_position,
  const Eigen::VectorXd & position, std::string * violation) const
{
  if (violation) {
    violation->clear();
  }
  if (compactManeuverActive()) {
    if (!group_ || previous_position.size() != static_cast<Eigen::Index>(
        group_->getVariableCount()) || position.size() != previous_position.size() ||
      !previous_position.allFinite() || !position.allFinite())
    {
      if (violation) {
        *violation = "state is unavailable";
      }
      return false;
    }
    constexpr double kMaximumGeometrySampleStepRad{0.02};
    constexpr std::size_t kMaximumGeometrySamples{4096U};
    const double span = (position - previous_position).lpNorm<Eigen::Infinity>();
    if (!std::isfinite(span)) {
      if (violation) {
        *violation = "nonfinite geometry interpolation span";
      }
      return false;
    }
    const double sample_count = std::max(
      1.0, std::ceil(span / kMaximumGeometrySampleStepRad));
    if (!std::isfinite(sample_count) ||
      sample_count > static_cast<double>(kMaximumGeometrySamples))
    {
      if (violation) {
        *violation = "geometry interpolation sample bound";
      }
      return false;
    }
    const std::size_t steps = static_cast<std::size_t>(sample_count);
    for (std::size_t step = 0U; step <= steps; ++step) {
      const double fraction = static_cast<double>(step) / static_cast<double>(steps);
      if (!validateCompactPathPosition(
          state, previous_position + fraction * (position - previous_position), violation))
      {
        return false;
      }
    }
    return true;
  }
  if (upright_roll_guard_armed_) {
    return validateUprightRollPosition(state, position, violation);
  }
  if (rest_mode_ != RestMode::kRest) {
    return true;
  }
  if (!group_ || !command_link_) {
    if (violation) {
      *violation = "screen roll state is unavailable";
    }
    return false;
  }
  const Eigen::Index joint_count = static_cast<Eigen::Index>(group_->getVariableCount());
  if (previous_position.size() != joint_count ||
    position.size() != joint_count || !previous_position.allFinite() || !position.allFinite())
  {
    if (violation) {
      *violation = "screen roll state is unavailable";
    }
    return false;
  }
  const auto edge_measure = [this, &state](const Eigen::VectorXd & candidate_position)
    -> std::optional<std::pair<double, double>>
    {
      moveit::core::RobotState candidate(state);
      candidate.setJointGroupPositions(group_, candidate_position);
      candidate.update(true);
      const Eigen::Matrix3d rotation =
        candidate.getGlobalLinkTransform(command_link_).linear();
      if (!validEdgeRotation(rotation)) {
        return std::nullopt;
      }
      return std::make_pair(std::abs(rotation(2, 1)), rotation(2, 2));
    };
  const auto previous_edge = edge_measure(previous_position);
  const auto next_edge = edge_measure(position);
  if (!previous_edge || !next_edge || previous_edge->second < 0.0 ||
    next_edge->second < 0.0)
  {
    if (violation) {
      *violation = "screen roll";
    }
    return false;
  }
  const bool previous_is_valid =
    previous_edge->first <= kCompactEdgeSineLimit && previous_edge->second >= 0.0;
  if (previous_is_valid) {
    if (next_edge->first <= kCompactEdgeSineLimit) {
      return true;
    }
    if (violation) {
      *violation = "screen roll";
    }
    return false;
  }
  if (next_edge->first > previous_edge->first) {
    if (violation) {
      *violation = "screen roll acquisition must not worsen";
    }
    return false;
  }
  return true;
}

void MotionReference::reset(const bool preserve_compact_maneuver)
{
  planner_->cancel();
  compact_maneuver_->cancelProfileWork();
  follower_.reset();
  target_motion_estimator_.reset();
  pending_generation_.reset();
  rest_mode_ = RestMode::kNone;
  rest_upright_start_active_ = false;
  search_pattern_.clear();
  search_paused_ = false;
  rest_completed_ = false;
  rest_settling_ = false;
  const bool disarm_face_roll_guard = compact_maneuver_->reset(
    preserve_compact_maneuver, disabled_limit_hold_latched_);
  if (disarm_face_roll_guard) {
    upright_roll_guard_armed_ = false;
    diagnostics_.upright_roll_guard_armed = false;
  }
}

void MotionReference::rejectPath()
{
  compact_maneuver_->rejectPath();
}

const MotionReferenceDiagnostics & MotionReference::diagnostics() const noexcept
{
  return diagnostics_;
}

Eigen::VectorXd MotionReference::groupPosition(const moveit::core::RobotState & state) const
{
  Eigen::VectorXd position;
  state.copyJointGroupPositions(group_, position);
  return position;
}

CompactManeuverContext MotionReference::compactContext(
  const moveit::core::RobotState & state,
  const planning_scene::PlanningScene & scene,
  const std::optional<msg::TrackingTarget> & target,
  const TargetMotionEstimate & target_motion, const double time_sec,
  const std::uint64_t scene_revision, const Eigen::VectorXd & position,
  const bool settled, const bool fresh_actual_state,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication) const
{
  return CompactManeuverContext{
    state, scene, target, target_motion, j1_boundary_assessment_, time_sec,
    scene_revision, position,
    settled, fresh_actual_state, actual_feedback, actual_feedback_time_sec,
    command_queue_state, active_publication, latest_publication,
    upright_roll_guard_armed_};
}


std::optional<HierarchicalVelocityTask> MotionReference::faceTask(
  const moveit::core::RobotState & state, const msg::TrackingTarget & target,
  const TargetMotionEstimate & target_motion, MotionReferenceResult & result,
  const bool face_tracking)
{
  Eigen::MatrixXd jacobian;
  if (!state.getJacobian(group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false)) {
    return std::nullopt;
  }
  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  if (root) {
    const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
    jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
    jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
  }
  std::optional<GazeKinematics> gaze;
  if (!parameters_.gaze_frame.empty()) {
    const auto * gaze_link = robot_model_->getLinkModel(parameters_.gaze_frame);
    GazeKinematics kinematics;
    kinematics.pose = state.getGlobalLinkTransform(gaze_link);
    kinematics.pose.linear() *= face_tracking_arm::opticalToPointingRotation();
    if (state.getJacobian(group_, gaze_link, Eigen::Vector3d::Zero(),
      kinematics.jacobian, false))
    {
      if (root) {
        const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
        kinematics.jacobian.topRows(3) = (rotation * kinematics.jacobian.topRows(3)).eval();
        kinematics.jacobian.bottomRows(3) =
          (rotation * kinematics.jacobian.bottomRows(3)).eval();
      }
      gaze = std::move(kinematics);
    }
  }
  TrackingVelocityTaskConfig config;
  config.position_gain = parameters_.position_gain;
  config.orientation_gain = parameters_.orientation_gain;
  config.position_deadband_m = parameters_.position_deadband_m;
  config.pointing_deadband_rad = parameters_.pointing_deadband_rad;
  config.maximum_linear_reference_mps = parameters_.maximum_linear_reference_mps;
  config.control_period_sec = parameters_.control_period_sec;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  config.roll_weight = parameters_.secondary_roll_weight;
  const bool posture_enabled = face_tracking && !compactManeuverActive();
  if (posture_enabled) {
    config.face_distance_min_m = parameters_.face_distance_min_m;
    config.face_distance_max_m = parameters_.face_distance_max_m;
  }
  const Eigen::Quaterniond orientation(
    target.pose.orientation.w, target.pose.orientation.x,
    target.pose.orientation.y, target.pose.orientation.z);
  if (!orientation.coeffs().allFinite() || orientation.norm() < 1.0e-9) {
    return std::nullopt;
  }
  auto task = makeTrackingVelocityTask(
    state.getGlobalLinkTransform(command_link_), jacobian,
    Eigen::Vector3d(target.pose.position.x, target.pose.position.y, target.pose.position.z),
    face_tracking ? std::optional<Eigen::Vector3d>{Eigen::Vector3d(
      target.face.x, target.face.y, target.face.z)} : std::nullopt,
    orientation.normalized().toRotationMatrix(), config, target_motion.face_velocity,
    target_motion.goal_velocity, gaze);
  if (!task) {
    return std::nullopt;
  }
  if (posture_enabled && face_tracking) {
    const auto posture_result = posture_task_builder_->apply(state, task->task);
    diagnostics_.posture_mode = postureTaskModeName(posture_result.mode);
    diagnostics_.posture_unavailable_reason = postureUnavailableReasonName(
      posture_result.unavailable_reason);
  }
  result.tracking = *task;
  return task->task;
}

bool MotionReference::appendUprightConstraints(
  const moveit::core::RobotState & state, const JointMotionState & motion_state,
  MotionReferenceResult & result)
{
  const auto fail = [this](std::string reason) {
      diagnostics_.last_plan_message = "FACE roll support unavailable: " + reason;
      return false;
    };
  if (!group_ || !command_link_ || !motion_state.position.allFinite() ||
    motion_state.position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    motion_state.velocity.size() != motion_state.position.size() ||
    motion_state.acceleration.size() != motion_state.position.size() ||
    !motion_state.velocity.allFinite() || !motion_state.acceleration.allFinite())
  {
    return fail("invalid group, command link, or queue position");
  }
  moveit::core::RobotState task_state(state);
  task_state.setJointGroupPositions(group_, motion_state.position);
  task_state.update(true);
  const Eigen::Matrix3d screen_rotation =
    task_state.getGlobalLinkTransform(command_link_).linear();
  Eigen::MatrixXd jacobian;
  if (!screen_rotation.allFinite() || !task_state.getJacobian(
      group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false))
  {
    return fail("screen rotation or Jacobian unavailable");
  }
  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  if (root) {
    const Eigen::Matrix3d rotation = task_state.getGlobalLinkTransform(root).linear();
    jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
    jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig config;
  config.orientation_gain = parameters_.orientation_gain;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  const auto edge_task = makeHorizontalEdgePathTask(
    task_state.getGlobalLinkTransform(command_link_), jacobian,
    Eigen::VectorXd::Zero(motion_state.position.size()), config);
  if (!edge_task || !std::isfinite(edge_task->edge_tilt_error_rad)) {
    return fail("horizontal edge task unavailable");
  }
  const Eigen::RowVectorXd edge_gradient = edge_task->task.primary_matrix.row(0);
  std::vector<LinearVelocityConstraint> roll_constraints;
  roll_constraints.reserve(3U);
  for (const double direction : {1.0, -1.0}) {
    const double signed_edge_sine = direction * edge_task->signed_edge_sine;
    const Eigen::VectorXd outward_gradient = direction * edge_gradient.transpose();
    if (outward_gradient.norm() <= 1.0e-10) {
      if (signed_edge_sine > kCompactEdgeSineLimit) {
        return fail("edge tilt cap exceeded: value=" + jsonNumber(signed_edge_sine) +
          ", cap=" + jsonNumber(kCompactEdgeSineLimit));
      }
      continue;
    }
    const auto row = makeOutwardStoppingConstraint(
      outward_gradient, std::max(0.0, kCompactEdgeSineLimit - signed_edge_sine),
      motion_state, activeCommandLimits(), parameters_.control_period_sec,
      kCompactQueueTailResidualLatencySec);
    if (!row) {
      return fail("edge stopping row infeasible: value=" + jsonNumber(signed_edge_sine) +
        ", cap=" + jsonNumber(kCompactEdgeSineLimit));
    }
    roll_constraints.push_back(*row);
  }
  const Eigen::Vector3d vertical_edge = screen_rotation.col(2);
  const Eigen::VectorXd vertical_gradient = -
    (vertical_edge.cross(Eigen::Vector3d::UnitZ()).transpose() *
    jacobian.bottomRows(3)).transpose();
  if (vertical_edge.z() < 0.0 || !vertical_gradient.allFinite()) {
    return fail("vertical edge infeasible: z=" + jsonNumber(vertical_edge.z()));
  }
  if (vertical_gradient.norm() > 1.0e-10) {
    const auto row = makeOutwardStoppingConstraint(
      vertical_gradient, vertical_edge.z(), motion_state, activeCommandLimits(),
      parameters_.control_period_sec, kCompactQueueTailResidualLatencySec);
    if (!row) {
      return fail("vertical edge stopping row infeasible: z=" + jsonNumber(vertical_edge.z()));
    }
    roll_constraints.push_back(*row);
  }
  result.compact_roll_constraints = std::move(roll_constraints);
  result.compact_pitch_constraints.clear();
  return true;
}

bool MotionReference::appendFaceRollGuardConstraints(
  const moveit::core::RobotState & state,
  const std::optional<JointMotionState> & command_queue_state,
  MotionReferenceResult & result)
{
  if (!upright_roll_guard_armed_ || compactManeuverActive()) {
    return true;
  }
  if (!command_queue_state || !appendUprightConstraints(
      state, *command_queue_state, result))
  {
    if (!command_queue_state) {
      diagnostics_.last_plan_message = "FACE roll support unavailable: queue state missing";
    }
    result.compact_roll_constraints.clear();
    result.motion_reference_support_unavailable = true;
    return false;
  }
  return true;
}

bool MotionReference::motionSettled(
  const JointMotionState & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state) const
{
  if (!command_queue_state || actual_feedback.position.size() == 0 ||
    actual_feedback.velocity.size() != actual_feedback.position.size() ||
    command_queue_state->position.size() != actual_feedback.position.size() ||
    command_queue_state->velocity.size() != actual_feedback.position.size() ||
    command_queue_state->acceleration.size() != actual_feedback.position.size() ||
    !actual_feedback.position.allFinite() || !actual_feedback.velocity.allFinite() ||
    !command_queue_state->position.allFinite() || !command_queue_state->velocity.allFinite() ||
    !command_queue_state->acceleration.allFinite())
  {
    return false;
  }
  return actual_feedback.velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
         command_queue_state->velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
         command_queue_state->acceleration.lpNorm<Eigen::Infinity>() <
         kSettledAccelerationRadps2;
}

std::optional<HierarchicalVelocityTask> MotionReference::pathTask(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  const Eigen::VectorXd & waypoint, const double gain, const double speed,
  const bool upright_roll_primary, double * measured_roll_error,
  const bool horizontal_edge_primary)
{
  const auto joint_task = makeJointPathVelocityTask(position, waypoint, gain, speed);
  if (!joint_task) {
    return std::nullopt;
  }
  if (!upright_roll_primary) {
    return joint_task;
  }
  Eigen::MatrixXd jacobian;
  if (!state.getJacobian(group_, command_link_, Eigen::Vector3d::Zero(), jacobian, false)) {
    return std::nullopt;
  }
  const auto * root = group_->getJointModels().front()->getParentLinkModel();
  if (root) {
    const Eigen::Matrix3d rotation = state.getGlobalLinkTransform(root).linear();
    jacobian.topRows(3) = (rotation * jacobian.topRows(3)).eval();
    jacobian.bottomRows(3) = (rotation * jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig config;
  config.orientation_gain = parameters_.orientation_gain;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  const Eigen::Isometry3d screen_pose = state.getGlobalLinkTransform(command_link_);
  if (horizontal_edge_primary) {
    const auto edge_task = makeHorizontalEdgePathTask(
      screen_pose, jacobian, joint_task->primary_reference, config);
    if (!edge_task) {
      return std::nullopt;
    }
    if (measured_roll_error) {
      *measured_roll_error = edge_task->edge_tilt_error_rad;
    }
    return edge_task->task;
  }
  const auto roll_task = makeUprightRollPathTask(
    screen_pose, jacobian, joint_task->primary_reference, config);
  if (!roll_task) {
    return std::nullopt;
  }
  if (measured_roll_error) {
    *measured_roll_error = roll_task->roll_error_rad;
  }
  return roll_task->task;
}

void MotionReference::updateJ1BoundaryAndLimitHold(
  const moveit::core::RobotState & state,
  const std::optional<msg::TrackingTarget> & target,
  const TargetMotionEstimate & target_motion, const double time_sec,
  const bool fresh_actual_state,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state)
{
  j1_boundary_assessment_ = J1BoundaryAssessment{};
  const bool has_face = target && target->mode == msg::TrackingTarget::FACE;
  const auto & names = group_->getVariableNames();
  const auto base = std::find(names.begin(), names.end(), "joint1");
  if (fresh_actual_state && actual_feedback && command_queue_state && base != names.end()) {
    const auto index = static_cast<Eigen::Index>(std::distance(names.begin(), base));
    const auto expected = static_cast<Eigen::Index>(names.size());
    const auto valid_motion_state = [expected](const JointMotionState & motion) {
        return motion.position.size() == expected && motion.velocity.size() == expected &&
               motion.acceleration.size() == expected && motion.position.allFinite() &&
               motion.velocity.allFinite() && motion.acceleration.allFinite();
      };
    if (index < limits_.lower_position.size() && index < limits_.upper_position.size() &&
      index < limits_.position_margin.size() && index < limits_.max_velocity.size() &&
      index < limits_.max_acceleration.size() && index < limits_.max_jerk.size() &&
      limits_.numerical_policy && limits_.numerical_policy->valid() &&
      valid_motion_state(*actual_feedback) && valid_motion_state(*command_queue_state))
    {
      detail::ScalarBrakingOptions options;
      options.period_sec = parameters_.control_period_sec;
      options.numerical_policy = *limits_.numerical_policy;
      const auto rollout = detail::rolloutScalarBraking(
        detail::ScalarBrakingState{command_queue_state->position[index],
          command_queue_state->velocity[index], command_queue_state->acceleration[index]},
        detail::ScalarBrakingLimits{limits_.max_velocity[index],
          limits_.max_acceleration[index], limits_.max_jerk[index]}, options, false);
      if (rollout.commandAvailable() && std::isfinite(rollout.minimum_position) &&
        std::isfinite(rollout.maximum_position))
      {
        j1_boundary_assessment_ = assessJ1Boundary(J1BoundaryInput{
            actual_feedback->position[index], rollout.minimum_position,
            rollout.maximum_position, limits_.lower_position[index],
            limits_.upper_position[index], limits_.position_margin[index],
            limits_.numerical_policy->position_error.value});
      }
    }
  }

  if (recovery_mode_ == RecoveryMode::kDisabled && has_face &&
    isFreshMeasuredFace(*target, time_sec, parameters_.face_target_freshness_timeout_sec))
  {
    last_real_face_target_ = *target;
    last_real_face_motion_ = target_motion;
  }
  const auto requested_j1_direction = [&](
    const msg::TrackingTarget & face, const TargetMotionEstimate & face_motion)
    -> std::optional<BaseJointDirection> {
      if (!fresh_actual_state || !actual_feedback ||
        !j1_boundary_assessment_.valid)
      {
        return std::nullopt;
      }
      moveit::core::RobotState actual_state(state);
      actual_state.setJointGroupPositions(group_, actual_feedback->position);
      actual_state.update(true);
      const Eigen::Isometry3d base_pose = actual_state.getGlobalLinkTransform(base_link_);
      const auto * tracking_link = parameters_.gaze_frame.empty() ? command_link_ :
        robot_model_->getLinkModel(parameters_.gaze_frame);
      if (!tracking_link) {
        return std::nullopt;
      }
      Eigen::Isometry3d tracking_pose = actual_state.getGlobalLinkTransform(tracking_link);
      if (!parameters_.gaze_frame.empty()) {
        tracking_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
      }
      const Eigen::Vector3d face_position(face.face.x, face.face.y, face.face.z);
      const Eigen::Vector3d ray = base_pose.linear().transpose() *
        (face_position - tracking_pose.translation());
      const Eigen::Vector3d normal =
        base_pose.linear().transpose() * tracking_pose.linear().col(0);
      const double ray_squared = ray.head<2>().squaredNorm();
      if (!ray.allFinite() || !normal.allFinite() || ray_squared <= 1.0e-9) {
        return std::nullopt;
      }
      const double bearing = std::atan2(ray.y(), ray.x());
      const double normal_heading = std::atan2(normal.y(), normal.x());
      const double yaw_error = std::atan2(
        std::sin(bearing - normal_heading), std::cos(bearing - normal_heading));
      const Eigen::Vector3d target_velocity_base =
        base_pose.linear().transpose() * face_motion.face_velocity;
      const double bearing_rate =
        (ray.x() * target_velocity_base.y() - ray.y() * target_velocity_base.x()) /
        ray_squared;
      if (!std::isfinite(yaw_error) || !std::isfinite(bearing_rate)) {
        return std::nullopt;
      }
      const double error_request = std::abs(yaw_error) > parameters_.pointing_deadband_rad ?
        parameters_.orientation_gain * yaw_error : 0.0;
      const double target_rate_request = std::abs(bearing_rate) > 0.02 ? bearing_rate : 0.0;
      const double request = error_request + target_rate_request;
      if (!std::isfinite(request)) {
        return std::nullopt;
      }
      return request == 0.0 ? BaseJointDirection::kNone :
             request > 0.0 ? BaseJointDirection::kPositive : BaseJointDirection::kNegative;
    };

  if (recovery_mode_ == RecoveryMode::kDisabled && fresh_actual_state &&
    j1_boundary_assessment_.valid)
  {
    const msg::TrackingTarget * face_target = nullptr;
    const TargetMotionEstimate * face_motion = nullptr;
    const bool cached_face_fresh = !has_face && last_real_face_target_ &&
      last_real_face_motion_ && isFreshMeasuredFace(*last_real_face_target_, time_sec,
      parameters_.face_target_freshness_timeout_sec);
    if (has_face) {
      face_target = &*target;
      face_motion = &target_motion;
    } else if (cached_face_fresh) {
      face_target = &*last_real_face_target_;
      face_motion = &*last_real_face_motion_;
    }
    std::optional<BaseJointDirection> yaw_direction;
    if (face_target && face_motion) {
      yaw_direction = requested_j1_direction(*face_target, *face_motion);
    }
    const bool outward_request =
      !disabled_limit_hold_latched_ && face_target && yaw_direction &&
      j1_boundary_assessment_.edge_direction != BaseJointDirection::kNone &&
      *yaw_direction == j1_boundary_assessment_.edge_direction;
    const bool release_observation_fresh = !outward_request &&
      disabled_limit_hold_latched_ && has_face && !target->face_is_prediction &&
      isFreshMeasuredFace(*target, time_sec, parameters_.face_target_freshness_timeout_sec);
    if (outward_request) {
      disabled_limit_hold_latched_ = true;
      disabled_limit_hold_source_stamp_ns_ = sourceFaceStampNs(*face_target);
    } else if (release_observation_fresh) {
      const std::int64_t source_stamp_ns = sourceFaceStampNs(*target);
      const bool newer_than_hold = !disabled_limit_hold_source_stamp_ns_ ||
        source_stamp_ns > *disabled_limit_hold_source_stamp_ns_;
      const bool request_moves_inward = yaw_direction &&
        (*yaw_direction == BaseJointDirection::kNone ||
        (j1_boundary_assessment_.edge_direction != BaseJointDirection::kNone &&
        *yaw_direction != j1_boundary_assessment_.edge_direction));
      const bool boundary_interior =
        j1_boundary_assessment_.edge_direction == BaseJointDirection::kNone;
      if (newer_than_hold && (boundary_interior || request_moves_inward)) {
        disabled_limit_hold_latched_ = false;
        disabled_limit_hold_source_stamp_ns_.reset();
      }
    }
  }
  diagnostics_.disabled_limit_hold_latched = disabled_limit_hold_latched_;
}

bool MotionReference::handleRestSearchMode(
  const moveit::core::RobotState & state,
  const std::optional<msg::TrackingTarget> & target,
  const double time_sec, const std::uint64_t scene_revision,
  const Eigen::VectorXd & position,
  const bool settled, const bool fresh_actual_state,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state,
  MotionReferenceResult & result)
{
  if (target && (target->mode == msg::TrackingTarget::REST ||
    target->mode == msg::TrackingTarget::SEARCH))
  {
    const RestMode requested = target->mode == msg::TrackingTarget::REST ? RestMode::kRest :
      RestMode::kSearch;
    const bool resume_search = requested == RestMode::kSearch && search_paused_ &&
      search_ && search_pattern_ == target->search_pattern;
    if (rest_mode_ != requested ||
      (requested == RestMode::kSearch && search_pattern_ != target->search_pattern))
    {
      planner_->cancel();
      compact_maneuver_->cancelProfileWork();
      pending_generation_.reset();
      follower_.reset();
      rest_mode_ = requested;
      rest_upright_start_active_ = false;
      rest_completed_ = false;
      rest_settling_ = false;
      if (requested == RestMode::kSearch) {
        search_pattern_ = target->search_pattern;
        if (resume_search) {
          search_->resume(time_sec);
        } else {
          search_->start(position, search_positions_, limits_.lower_position +
            limits_.position_margin, limits_.upper_position - limits_.position_margin,
            search_base_index_, search_config_, search_pattern_ == "local_then_sweep", time_sec);
        }
        search_paused_ = false;
      } else {
        search_pattern_.clear();
        search_paused_ = false;
      }
    }
    if (search_ && requested == RestMode::kSearch) {
      search_->update(position, time_sec);
      if (!search_->preparing()) {
        result.task = pathTask(
          state, position, search_->goal(), 2.0, search_config_.speed_rad_s, true);
        result.follows_path = result.task.has_value();
        diagnostics_.state = search_->phase();
        return true;
      }
    }
    if (requested == RestMode::kRest) {
      if (!group_ || !command_link_) {
        result.motion_reference_support_unavailable = true;
        diagnostics_.state = "REST_UPRIGHT_START_UNAVAILABLE";
        diagnostics_.tracking_phase = "REST";
        return true;
      }
      moveit::core::RobotState acquisition_state(state);
      acquisition_state.setJointGroupPositions(group_, position);
      acquisition_state.update(true);
      const Eigen::Matrix3d rotation =
        acquisition_state.getGlobalLinkTransform(command_link_).linear();
      if (!evaluateHorizontalEdgeUprightWithSineCap(rotation, kCompactEdgeSineLimit)) {
        if (!rest_upright_start_active_) {
          if (!fresh_actual_state || !actual_feedback || !command_queue_state ||
            !motionSettled(*actual_feedback, command_queue_state))
          {
            result.motion_reference_support_unavailable = true;
            diagnostics_.state = "REST_UPRIGHT_START_REQUIRES_SETTLED_STATE";
            diagnostics_.tracking_phase = "REST";
            return true;
          }
          rest_upright_start_active_ = true;
        }
        if (!group_ || !command_link_) {
          result.motion_reference_support_unavailable = true;
          diagnostics_.state = "REST_UPRIGHT_START_UNAVAILABLE";
          diagnostics_.tracking_phase = "REST";
          return true;
        }
        result.task = pathTask(
          acquisition_state, position, position, 1.0, 0.01, true, nullptr, true);
        if (!result.task) {
          result.motion_reference_support_unavailable = true;
          diagnostics_.state = "REST_UPRIGHT_START_UNAVAILABLE";
          diagnostics_.tracking_phase = "REST";
          return true;
        }
        diagnostics_.state = "REST_UPRIGHT_START";
        diagnostics_.tracking_phase = "REST";
        return true;
      }
      rest_upright_start_active_ = false;
    }
    const Eigen::VectorXd & goal = requested == RestMode::kSearch ?
      search_->goal() : rest_positions_;
    const auto append_rest_edge_guard = [&]() {
        if (requested != RestMode::kRest) {
          return true;
        }
        if (!command_queue_state) {
          return false;
        }
        return appendUprightConstraints(state, *command_queue_state, result);
      };
    const auto valid_rest_edge_path = [&](const BackgroundPathResult & plan) {
        if (plan.path.empty()) {
          return false;
        }
        moveit::core::RobotState path_state(state);
        for (const Eigen::VectorXd & waypoint : plan.path) {
          if (waypoint.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
            !waypoint.allFinite())
          {
            return false;
          }
          path_state.setJointGroupPositions(group_, waypoint);
          path_state.update(true);
          const Eigen::Matrix3d rotation =
            path_state.getGlobalLinkTransform(command_link_).linear();
          if (!evaluateHorizontalEdgeUprightWithSineCap(rotation, kCompactEdgeSineLimit)) {
            return false;
          }
        }
        return true;
      };
    if (follower_.active()) {
      const auto path = follower_.update(position, time_sec);
      diagnostics_.path_progress_rad = path.progress_rad;
      diagnostics_.path_distance_rad = path.distance_from_path_rad;
      if (path.status == JointPathStatus::kTracking) {
        result.task = pathTask(
          state, position, path.reference_position, 3.0, 0.35, true, nullptr,
          requested == RestMode::kRest);
        if (requested == RestMode::kRest &&
          (!result.task || !append_rest_edge_guard()))
        {
          result.task.reset();
          result.motion_reference_support_unavailable = true;
          diagnostics_.state = "REST_EDGE_GUARD_UNAVAILABLE";
          return true;
        }
        result.follows_path = true;
        diagnostics_.state = requested == RestMode::kSearch ? "SEARCH_PATH" : "REST_PATH";
        return true;
      }
      follower_.reset();
      if (path.status == JointPathStatus::kArrived) {
        ++diagnostics_.paths_completed;
        rest_settling_ = true;
      } else if (requested == RestMode::kSearch) {
        ++diagnostics_.paths_abandoned;
        search_->retryFrom(position, time_sec);
      }
    }
    const Eigen::VectorXd & arrival_position = fresh_actual_state ?
      actual_feedback->position : position;
    const double error = (arrival_position - goal).lpNorm<Eigen::Infinity>();
    const bool actual_settled = fresh_actual_state ?
      (actual_feedback->velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
      actual_feedback->acceleration.lpNorm<Eigen::Infinity>() < kSettledAccelerationRadps2) :
      settled;
    moveit::core::RobotState actual_edge_state(state);
    if (fresh_actual_state) {
      actual_edge_state.setJointGroupPositions(group_, actual_feedback->position);
      actual_edge_state.update(true);
    }
    const auto current_edge_error = measureHorizontalEdgeTiltError(
      actual_edge_state.getGlobalLinkTransform(command_link_));
    const bool current_edge_inside = current_edge_error &&
      std::abs(*current_edge_error) <= kCompactRollLimitRad &&
      actual_edge_state.getGlobalLinkTransform(command_link_).linear()(2, 2) >= 0.0;
    if (error < kRestArrivalRad && actual_settled &&
      (requested != RestMode::kRest || (fresh_actual_state && current_edge_inside)))
    {
      if (requested == RestMode::kRest) {
        rest_completed_ = true;
        rest_settling_ = false;
        upright_roll_guard_armed_ = false;
        diagnostics_.upright_roll_guard_armed = false;
      } else {
        search_->prepared(position, time_sec);
        rest_settling_ = false;
        result.task = pathTask(state, position, search_->goal(), 2.0,
          search_config_.speed_rad_s, true);
        result.follows_path = result.task.has_value();
      }
      diagnostics_.state = requested == RestMode::kSearch ? search_->phase() : "REST_HOLD";
      return true;
    }
    if (requested == RestMode::kRest && rest_completed_ && error > kRestReleaseRad) {
      rest_completed_ = false;
    }
    if (requested == RestMode::kRest && rest_completed_ && !actual_settled) {
      rest_settling_ = true;
    }
    if (requested == RestMode::kRest && rest_settling_) {
      diagnostics_.state = "REST_SETTLING";
      if (error >= kRestArrivalRad) {
        result.task = pathTask(state, position, goal, 3.0, 0.10, true, nullptr, true);
        result.follows_path = result.task.has_value();
        if (!result.task || !append_rest_edge_guard()) {
          result.task.reset();
          result.motion_reference_support_unavailable = true;
          diagnostics_.state = "REST_EDGE_GUARD_UNAVAILABLE";
        }
      }
      return true;
    }
    if (requested == RestMode::kSearch && search_->preparing()) {
      if (rest_settling_) {
        if (error < kRestReleaseRad) {
          diagnostics_.state = "SEARCH_SETTLING";
          if (error >= kRestArrivalRad) {
            result.task = pathTask(state, position, goal, 1.0, 0.10, true);
            result.follows_path = result.task.has_value();
          }
          return true;
        }
        rest_settling_ = false;
      }
      if (!pending_generation_ && !planner_->isBusy()) {
        pending_generation_ = planner_->submit(state, goal, scene_revision);
        ++diagnostics_.plans_requested;
      }
      if (auto plan = planner_->takeResult()) {
        if (pending_generation_ && plan->generation == *pending_generation_) {
          pending_generation_.reset();
          if (plan->success && plan->scene_revision == scene_revision &&
            follower_.setPath(plan->path, position, time_sec))
          {
            ++diagnostics_.plans_accepted;
          } else {
            ++diagnostics_.plans_rejected;
            search_->retryFrom(position, time_sec);
          }
        }
      }
      diagnostics_.state = "SEARCH_PLANNING";
      return true;
    }
    if (requested == RestMode::kRest && pending_generation_) {
      if (auto plan = planner_->takeResult()) {
        if (plan->generation == *pending_generation_) {
          pending_generation_.reset();
          diagnostics_.last_plan_message = plan->message;
          if (plan->success && plan->scene_revision == scene_revision &&
            valid_rest_edge_path(*plan) &&
            follower_.setPath(plan->path, position, time_sec))
          {
            ++diagnostics_.plans_accepted;
          } else {
            ++diagnostics_.plans_rejected;
            diagnostics_.last_plan_message = "REST path rejected: " + plan->message;
          }
        }
      }
    }
    if (requested == RestMode::kRest && !rest_completed_ && !follower_.active() &&
      !pending_generation_ && !planner_->isBusy())
    {
      pending_generation_ = planner_->submit(
        state, goal, HorizontalEdgePathRequirement{
          command_link_->getName(), kCompactEdgeSineLimit}, scene_revision);
      ++diagnostics_.plans_requested;
      diagnostics_.state = "REST_PLANNING";
      return true;
    }
    if (requested == RestMode::kRest && !rest_completed_ && !follower_.active()) {
      diagnostics_.state = "REST_PLANNING";
      return true;
    }
    diagnostics_.state = "SEARCH_SETTLING";
    return true;
  }

  return false;
}

void MotionReference::updateControlClock(const double time_sec)
{
  if (!std::isfinite(time_sec)) {
    return;
  }
  if (last_control_time_sec_ && time_sec < *last_control_time_sec_) {
    // Keep the limit hold across a ROS clock epoch reset, but allow the new
    // epoch's first independently fresh observation to prove access again.
    disabled_limit_hold_source_stamp_ns_.reset();
    last_real_face_target_.reset();
    last_real_face_motion_.reset();
    if (rest_mode_ != RestMode::kNone || search_paused_) {
      planner_->cancel();
      compact_maneuver_->cancelProfileWork();
      pending_generation_.reset();
      follower_.reset();
      rest_mode_ = RestMode::kNone;
      rest_settling_ = false;
      search_pattern_.clear();
      search_paused_ = false;
    }
  }
  last_control_time_sec_ = time_sec;
}

MotionReferenceResult MotionReference::update(
  const moveit::core::RobotState & state,
  const std::optional<msg::TrackingTarget> & target,
  const planning_scene::PlanningScene & scene,
  const double time_sec, const std::uint64_t scene_revision,
  const std::optional<JointMotionState> & actual_feedback,
  const std::optional<double> & actual_feedback_time_sec,
  const std::optional<JointMotionState> & command_queue_state,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication)
{
  MotionReferenceResult result;
  diagnostics_.posture_mode = "unavailable";
  diagnostics_.posture_unavailable_reason = "none";
  updateControlClock(time_sec);
  const bool has_face = target && target->mode == msg::TrackingTarget::FACE;
  const auto mode = target ? target->mode : msg::TrackingTarget::HOLD;
  const bool idle = mode == msg::TrackingTarget::REST || mode == msg::TrackingTarget::SEARCH;
  const bool healthy_face_loss = target && target->allow_recovery && !has_face &&
    (mode == msg::TrackingTarget::HOLD || mode == msg::TrackingTarget::REST ||
    mode == msg::TrackingTarget::SEARCH);
  if (!disabled_limit_hold_latched_ &&
    ((idle && compact_maneuver_->phase() != FaceTrackingPhase::kCompactManeuver) ||
    ((!has_face && !healthy_face_loss) &&
    compact_maneuver_->phase() != FaceTrackingPhase::kCompactManeuver)))
  {
    upright_roll_guard_armed_ = false;
    diagnostics_.upright_roll_guard_armed = false;
  }
  if (search_paused_ && mode != msg::TrackingTarget::HOLD &&
    mode != msg::TrackingTarget::SEARCH)
  {
    search_pattern_.clear();
    search_paused_ = false;
  }
  if (rest_mode_ != RestMode::kNone && !idle) {
    const bool preserve_search_on_hold = mode == msg::TrackingTarget::HOLD &&
      rest_mode_ == RestMode::kSearch && search_.has_value();
    if (preserve_search_on_hold) {
      search_->pause(time_sec);
      search_paused_ = true;
    } else {
      search_pattern_.clear();
      search_paused_ = false;
    }
    planner_->cancel();
    compact_maneuver_->cancelProfileWork();
    pending_generation_.reset();
    follower_.reset();
    rest_mode_ = RestMode::kNone;
    rest_upright_start_active_ = false;
    rest_settling_ = false;
  }
  if (compact_maneuver_->updateTargetIdentity(target, time_sec)) {
    target_motion_estimator_.reset();
  }
  compact_maneuver_->consumeCompactProfileWorkResult(time_sec, scene_revision);
  TargetMotionEstimate target_motion;
  if (has_face) {
    const double measurement_time = static_cast<double>(target->face_stamp.sec) +
      1.0e-9 * static_cast<double>(target->face_stamp.nanosec);
    target_motion = target_motion_estimator_.update(
      Eigen::Vector3d(target->face.x, target->face.y, target->face.z),
      Eigen::Vector3d(target->pose.position.x, target->pose.position.y,
      target->pose.position.z), measurement_time, time_sec);
  }

  Eigen::VectorXd position = groupPosition(state);
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
  state.copyJointGroupVelocities(group_, velocity);
  state.copyJointGroupAccelerations(group_, acceleration);
  const bool settled = velocity.size() == position.size() &&
    acceleration.size() == position.size() &&
    velocity.lpNorm<Eigen::Infinity>() < kSettledVelocityRadps &&
    acceleration.lpNorm<Eigen::Infinity>() < kSettledAccelerationRadps2;
  const bool fresh_actual_state = actual_feedback && isFreshActualFeedback(
    clock_, time_sec, actual_feedback_time_sec, parameters_.state_feedback_timeout_sec) &&
    actual_feedback->position.size() == position.size() &&
    actual_feedback->velocity.size() == position.size() &&
    actual_feedback->position.allFinite() && actual_feedback->velocity.allFinite();
  const auto begin_tick_context = compactContext(
    state, scene, target, target_motion, time_sec, scene_revision, position, settled,
    fresh_actual_state, actual_feedback, actual_feedback_time_sec, command_queue_state,
    active_publication, latest_publication);
  compact_maneuver_->beginTick(begin_tick_context);

  updateJ1BoundaryAndLimitHold(
    state, target, target_motion, time_sec, fresh_actual_state,
    actual_feedback, command_queue_state);
  if (!upright_roll_guard_armed_ && !compactManeuverActive() &&
    (has_face || (healthy_face_loss &&
    compact_maneuver_->phase() == FaceTrackingPhase::kCompactManeuver)))
  {
    static_cast<void>(maybeArmFaceUprightRollGuard(
        state, time_sec, actual_feedback, actual_feedback_time_sec, command_queue_state,
        active_publication, latest_publication));
  }
  diagnostics_.upright_roll_guard_armed = upright_roll_guard_armed_;
  diagnostics_.disabled_limit_hold_latched = disabled_limit_hold_latched_;
  if (recovery_mode_ == RecoveryMode::kDisabled && disabled_limit_hold_latched_ &&
    mode != msg::TrackingTarget::POSE)
  {
    diagnostics_.state = "DISABLED_LIMIT_HOLD";
    diagnostics_.tracking_phase = "FACE_LIMIT_HOLD";
    if (upright_roll_guard_armed_) {
      static_cast<void>(appendFaceRollGuardConstraints(
          state, command_queue_state, result));
    }
    return result;
  }

  const auto context = compactContext(
    state, scene, target, target_motion, time_sec, scene_revision, position, settled,
    fresh_actual_state, actual_feedback, actual_feedback_time_sec, command_queue_state,
    active_publication, latest_publication);
  CompactManeuverOutcome outcome = compact_maneuver_->update(context);
  if (outcome.disarm_face_roll_guard) {
    upright_roll_guard_armed_ = false;
    diagnostics_.upright_roll_guard_armed = false;
  }
  if (outcome.handled) {
    result = std::move(outcome.result);
    const bool build_face = outcome.face_directive == CompactFaceDirective::kBuildFace ||
      outcome.face_directive == CompactFaceDirective::kBuildFaceAndGuard;
    if (build_face && has_face) {
      result.task = faceTask(state, *target, target_motion, result, has_face);
    }
    if (outcome.face_directive == CompactFaceDirective::kBuildFaceAndGuard ||
      outcome.face_directive == CompactFaceDirective::kGuardOnly)
    {
      static_cast<void>(appendFaceRollGuardConstraints(
          state, command_queue_state, result));
    }
    return result;
  }

  if (mode == msg::TrackingTarget::HOLD) {
    diagnostics_.state = "HOLD";
    diagnostics_.tracking_phase = "FACE";
    return result;
  }

  if (mode == msg::TrackingTarget::SEARCH && !search_enabled_) {
    diagnostics_.state = "SEARCH_UNAVAILABLE";
    diagnostics_.last_plan_message = "SEARCH rejected: search is disabled";
    return result;
  }
  if (handleRestSearchMode(
      state, target, time_sec, scene_revision, position, settled,
      fresh_actual_state, actual_feedback, command_queue_state, result))
  {
    return result;
  }

  rest_mode_ = RestMode::kNone;
  rest_completed_ = false;
  rest_settling_ = false;
  if (has_face || mode == msg::TrackingTarget::POSE) {
    result.task = faceTask(
      state, *target, target_motion, result, has_face);
    if (!has_face) {
      diagnostics_.state = result.task ? "POSE" : "INVALID_TASK";
      diagnostics_.tracking_phase = "POSE";
      return result;
    }
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = result.task ? "FACE" : "INVALID_TASK";
    static_cast<void>(appendFaceRollGuardConstraints(
        state, command_queue_state, result));
    if (recovery_mode_ == RecoveryMode::kCompact &&
      compact_maneuver_->phase() == FaceTrackingPhase::kTracking &&
      j1_boundary_assessment_.valid &&
      j1_boundary_assessment_.edge_direction != BaseJointDirection::kNone)
    {
      const std::string boundary_details =
        j1BoundaryAssessmentJson(j1_boundary_assessment_);
      compact_maneuver_->onJ1Boundary(
        context, "j1_working_boundary", boundary_details);
    }
    return result;
  }

  diagnostics_.state = "NO_FACE";
  diagnostics_.tracking_phase = "FACE";
  return result;
}

}  // namespace face_tracking_arm::control
