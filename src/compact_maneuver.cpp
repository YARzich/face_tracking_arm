// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_maneuver.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <moveit_msgs/msg/planning_scene.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

#include "face_tracking_arm/compact_upright.hpp"
#include "compact_maneuver_validation.hpp"
#include "compact_path_attachment.hpp"
#include "compact_plan_admission.hpp"
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"
#include "face_tracking_arm/snapshot_timing.hpp"
#include "servo_types.hpp"

namespace face_tracking_arm::control
{
namespace
{

using detail::kCompactRadiusFloorM;

constexpr double kSettledVelocityRadps{0.01};
constexpr double kSettledAccelerationRadps2{0.03};
constexpr double kCompactGoalToleranceRad{0.025};
constexpr double kCompactReserveRad{0.35};
constexpr double kCompactReturnRadiusM{0.10};
constexpr double kCompactReturnDirectionToleranceRad{0.2617993877991494};
constexpr double kCompactReturnHorizontalEdgeToleranceRad{0.2617993877991494};
// Compact constraints act at the predicted queue tail. Its published remainder
// through stop is included in the immutable entry envelope, so it must not be
// charged a second command-latency interval. Collision constraints still use
// the configured residual latency.
constexpr double kCompactQueueTailResidualLatencySec{0.0};

std::string jsonNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string jsonFiniteNumber(const double value)
{
  return std::isfinite(value) ? jsonNumber(value) : std::string{"null"};
}

std::string jsonString(const std::string & value)
{
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size() + 2U);
  escaped.push_back('"');
  for (const unsigned char character : value) {
    switch (character) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (character < 0x20U) {
          escaped += "\\u00";
          escaped.push_back(kHexDigits[character >> 4U]);
          escaped.push_back(kHexDigits[character & 0x0fU]);
        } else {
          escaped.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  escaped.push_back('"');
  return escaped;
}

std::string jsonVector(const Eigen::VectorXd & value)
{
  std::string result{"["};
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (index > 0) {
      result += ',';
    }
    result += jsonNumber(value[index]);
  }
  result += ']';
  return result;
}

std::string jsonStringArray(const std::vector<std::string> & values)
{
  std::string result{"["};
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index > 0U) {
      result += ',';
    }
    result += jsonString(values[index]);
  }
  result += ']';
  return result;
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

CompactManeuver::CompactManeuver(
  const rclcpp::Node::SharedPtr & node, const bool compact_recovery_enabled,
  const ControllerParameters & parameters, const JointMotionLimits & limits,
  const CollisionConstraintBuilder & collision_validator,
  moveit::core::RobotModelConstPtr robot_model,
  const moveit::core::JointModelGroup * group,
  const moveit::core::LinkModel * command_link,
  const moveit::core::LinkModel * base_link,
  const PostureTaskBuilder * posture_task_builder,
  const rclcpp::Clock::SharedPtr & clock,
  BackgroundPathPlanner & planner, JointPathFollower & follower,
  MotionReferenceDiagnostics & diagnostics)
: parameters_(parameters), limits_(limits), collision_validator_(collision_validator),
  robot_model_(std::move(robot_model)), group_(group), command_link_(command_link),
  base_link_(base_link), posture_task_builder_(posture_task_builder), clock_(clock),
  planner_(planner), follower_(follower), diagnostics_(diagnostics)
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.read_only = true;
  compact_speed_scale_ = readOrDeclareParameter(
    node, "compact_maneuver.speed_scale", 1.0, descriptor);
  compact_fold_speed_scale_ = readOrDeclareParameter(
    node, "compact_maneuver.fold_speed_scale", 1.0, descriptor);
  compact_turn_speed_scale_ = readOrDeclareParameter(
    node, "compact_maneuver.turn_speed_scale", 1.0, descriptor);
  compact_return_speed_scale_ = readOrDeclareParameter(
    node, "compact_maneuver.return_speed_scale", 1.0, descriptor);
  for (const double scale : {compact_speed_scale_, compact_fold_speed_scale_,
      compact_turn_speed_scale_, compact_return_speed_scale_})
  {
    if (!std::isfinite(scale) || scale <= 0.0) {
      throw std::invalid_argument(
        "compact_maneuver phase speed scales must be finite and positive");
    }
  }
  const auto & variable_names = group_->getVariableNames();
  const std::size_t joint_count = variable_names.size();
  compact_limits_ = limits_;
  compact_limits_.max_velocity = Eigen::VectorXd::Zero(
    static_cast<Eigen::Index>(joint_count));
  compact_limits_.max_acceleration = Eigen::VectorXd::Zero(
    static_cast<Eigen::Index>(joint_count));
  compact_limits_.max_jerk = Eigen::VectorXd::Zero(
    static_cast<Eigen::Index>(joint_count));
  const auto default_limit = [joint_count](const double value) {
      return std::vector<double>(joint_count, value);
    };
  const auto compact_velocity = readOrDeclareParameter(
    node, "compact_maneuver.max_velocity_rad_s", default_limit(3.1), descriptor);
  const auto compact_acceleration = readOrDeclareParameter(
    node, "compact_maneuver.max_acceleration_rad_s2", default_limit(19.0), descriptor);
  const auto compact_jerk = readOrDeclareParameter(
    node, "compact_maneuver.max_jerk_rad_s3", default_limit(400.0), descriptor);
  if (compact_velocity.size() != joint_count || compact_acceleration.size() != joint_count ||
    compact_jerk.size() != joint_count)
  {
    throw std::invalid_argument("compact_maneuver v/a/j vectors must match the planning group");
  }
  for (std::size_t index = 0U; index < joint_count; ++index) {
    const Eigen::Index axis = static_cast<Eigen::Index>(index);
    const auto urdf_joint = robot_model_->getURDF()->getJoint(variable_names[index]);
    if (!std::isfinite(compact_velocity[index]) || compact_velocity[index] <= 0.0 ||
      !std::isfinite(compact_acceleration[index]) || compact_acceleration[index] <= 0.0 ||
      !std::isfinite(compact_jerk[index]) || compact_jerk[index] <= 0.0 ||
      !urdf_joint || !urdf_joint->limits ||
      !std::isfinite(urdf_joint->limits->velocity) || urdf_joint->limits->velocity <= 0.0)
    {
      throw std::invalid_argument("compact_maneuver v/a/j limits must be finite and positive");
    }
    compact_limits_.max_velocity[axis] = std::min(
      compact_velocity[index], urdf_joint->limits->velocity);
    compact_limits_.max_acceleration[axis] = compact_acceleration[index];
    compact_limits_.max_jerk[axis] = compact_jerk[index];
  }
  const PositionInterval compact_interval{
    compact_limits_.lower_position.minCoeff(), compact_limits_.upper_position.maxCoeff()};
  const auto compact_policy = NumericalPolicy::fromDiscreteMotionLimits(
    compact_interval, compact_limits_.max_velocity.maxCoeff(),
    compact_limits_.max_acceleration.maxCoeff(), compact_limits_.max_jerk.maxCoeff(),
    parameters_.control_period_sec,
    limits_.numerical_policy ? limits_.numerical_policy->solver_residual : SolverResidual{1.0e-5});
  if (!compact_policy) {
    throw std::invalid_argument("Unable to derive compact motion numerical policy");
  }
  compact_limits_.numerical_policy = compact_policy;
  compact_return_limits_ = limits_;
  const double return_scale = std::min(compact_speed_scale_ * compact_return_speed_scale_, 1.0);
  compact_return_limits_.max_velocity *= return_scale;
  compact_return_limits_.max_acceleration *= return_scale * return_scale;
  compact_return_limits_.max_jerk *= return_scale * return_scale * return_scale;
  const auto return_policy = NumericalPolicy::fromDiscreteMotionLimits(
    compact_interval, compact_return_limits_.max_velocity.maxCoeff(),
    compact_return_limits_.max_acceleration.maxCoeff(), compact_return_limits_.max_jerk.maxCoeff(),
    parameters_.control_period_sec,
    limits_.numerical_policy ? limits_.numerical_policy->solver_residual : SolverResidual{1.0e-5});
  if (!return_policy) {
    throw std::invalid_argument("Unable to derive compact return motion numerical policy");
  }
  compact_return_limits_.numerical_policy = return_policy;
  compact_goal_settings_ = {3.0, 0.10 * compact_speed_scale_};

  moveit::core::RobotState named_state(robot_model_);
  named_state.setToDefaultValues();
  const std::vector<double> configured_goal = readOrDeclareParameter(
    node, "compact_maneuver.goal_joints_rad", std::vector<double>{}, descriptor);
  if (configured_goal.empty()) {
    select_default_compact_goal_ = true;
    for (const auto * profile : {"compact_turn_negative_wrist", "compact_turn_positive_wrist"}) {
      if (!named_state.setToDefaultValues(group_, profile)) {
        continue;
      }
      CompactGoal candidate;
      candidate.profile_id = profile;
      named_state.copyJointGroupPositions(group_, candidate.bounded_positions);
      if (compact_goal_.profile_id.empty() ||
        candidate.profile_id == "compact_turn_negative_wrist")
      {
        compact_goal_ = candidate;
      }
      default_compact_goals_.push_back(std::move(candidate));
    }
  } else {
    if (configured_goal.size() != static_cast<std::size_t>(group_->getVariableCount())) {
      throw std::invalid_argument(
        "compact_maneuver.goal_joints_rad must match the planning group variable count");
    }
    compact_goal_.profile_id = "configured_goal_override";
    compact_goal_.bounded_positions = Eigen::Map<const Eigen::VectorXd>(
      configured_goal.data(), static_cast<Eigen::Index>(configured_goal.size()));
    if (!compact_goal_.bounded_positions.allFinite() ||
      (compact_goal_.bounded_positions.array() < limits_.lower_position.array()).any() ||
      (compact_goal_.bounded_positions.array() > limits_.upper_position.array()).any())
    {
      throw std::invalid_argument(
        "compact_maneuver.goal_joints_rad must be finite and inside joint bounds");
    }
  }
  compact_goal_.path_requirements.base_link_name = "link_base";
  compact_goal_.path_requirements.screen_link_name = parameters_.command_frame;
  compact_goal_.path_requirements.maximum_assembly_radius_m = kCompactRadiusFloorM;
  for (auto & candidate : default_compact_goals_) {
    candidate.path_requirements.base_link_name = "link_base";
    candidate.path_requirements.screen_link_name = parameters_.command_frame;
    candidate.path_requirements.maximum_assembly_radius_m = kCompactRadiusFloorM;
  }
  if (!compact_goal_.profile_id.empty()) {
    compact_goal_.path_requirements.base_link_name = base_link_->getName();
    compact_geometry_validator_ = std::make_shared<CompactGeometryValidator>(
      robot_model_, base_link_->getName());
    if (compact_recovery_enabled) {
      auto dynamics = std::make_shared<CompactDynamicsModel>(
        robot_model_, parameters_.planning_group_name, base_link_->getName());
      compact_profile_worker_ = std::make_unique<CompactProfileWorker>(std::move(dynamics));
    }
  }
}

FaceTrackingPhase CompactManeuver::phase() const noexcept
{
  return face_policy_.phase();
}

bool CompactManeuver::active() const noexcept
{
  return face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
         compact_commit_initialized_;
}

Eigen::VectorXd CompactManeuver::groupPosition(
  const moveit::core::RobotState & state) const
{
  Eigen::VectorXd position;
  state.copyJointGroupPositions(group_, position);
  return position;
}

std::vector<CompactEvent> CompactManeuver::drainEvents()
{
  std::vector<CompactEvent> drained;
  drained.reserve(compact_events_.size());
  while (!compact_events_.empty()) {
    drained.push_back(std::move(compact_events_.front()));
    compact_events_.pop_front();
  }
  return drained;
}

void CompactManeuver::recordFaceRollGuardArmed(
  const double time_sec, std::string reason, std::string details_json)
{
  appendCompactEvent("upright_roll_guard_armed", time_sec,
    std::move(reason), std::move(details_json));
}

void CompactManeuver::cancelProfileWork()
{
  cancelCompactProfileWork();
}

bool CompactManeuver::takeFaceRollGuardDisarmRequest() noexcept
{
  const bool requested = face_roll_guard_disarm_requested_;
  face_roll_guard_disarm_requested_ = false;
  return requested;
}

bool CompactManeuver::reset(
  const bool preserve_compact_maneuver, const bool disabled_limit_hold_latched)
{
  const bool preserve_committed_maneuver = preserve_compact_maneuver &&
    compact_commit_initialized_;
  const bool preserve_return_stage = preserve_committed_maneuver &&
    compact_return_path_pending_ && compact_goal_reached_;
  const bool preserve_return_profile = preserve_return_stage &&
    compact_return_profile_active_;
  pending_generation_.reset();
  face_roll_guard_disarm_requested_ = false;
  compact_preparation_.reset();
  active_scene_revision_.reset();
  target_id_.reset();
  target_bearing_motion_sign_ = 0;
  target_bearing_moving_ = false;
  compact_path_adopted_ = false;
  compact_path_execution_.clear();
  compact_prepared_return_profile_.reset();
  compact_return_profile_active_ = false;
  if (!preserve_return_profile) {
    compact_return_endpoint_.resize(0);
  }
  compact_goal_reached_ = preserve_return_stage;
  compact_path_endpoint_reached_ = preserve_return_stage && !preserve_return_profile;
  compact_goal_event_emitted_ = preserve_return_stage && compact_goal_event_emitted_;
  compact_return_path_pending_ = preserve_return_stage;
  compact_plan_pending_ = false;
  diagnostics_.compact_actual_settled = false;
  diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_roll_error_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
  if (!preserve_committed_maneuver) {
    face_policy_.clearForStop();
    compact_commit_initialized_ = false;
    compact_radius_floor_latched_ = false;
    diagnostics_.compact_radius_floor_latched = false;
    diagnostics_.compact_effective_radius_cap_m =
      std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_pitch_cap_rad = std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_actual_settled = false;
    compact_path_started_event_emitted_ = false;
    if (!preserve_compact_maneuver) {
      compact_goal_.path_requirements.maximum_assembly_radius_m = kCompactRadiusFloorM;
      compact_goal_.return_region.reset();
      diagnostics_.compact_entry_radius_cap_m = std::numeric_limits<double>::quiet_NaN();
      diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
      diagnostics_.planned_path_max_roll_error_rad = std::numeric_limits<double>::quiet_NaN();
      diagnostics_.compact_generation = 0U;
      diagnostics_.compact_reason.clear();
    }
  }
  diagnostics_.compact_goal_reached = preserve_return_stage ? 1.0 : 0.0;
  diagnostics_.compact_phase = preserve_return_stage ?
    "RETURN_PATH_REPLANNING" : face_policy_.phase() == FaceTrackingPhase::kCompactManeuver ?
    "PLANNING" : "INACTIVE";
  diagnostics_.state = disabled_limit_hold_latched ? "DISABLED_LIMIT_HOLD" :
    preserve_return_stage ? "COMPACT_RETURN_PATH_REPLANNING" : "HOLD";
  diagnostics_.tracking_phase = disabled_limit_hold_latched ? "FACE_LIMIT_HOLD" :
    preserve_return_stage ? "COMPACT" : "FACE";
  diagnostics_.disabled_limit_hold_latched = disabled_limit_hold_latched;
  return !preserve_committed_maneuver && !preserve_compact_maneuver &&
         !disabled_limit_hold_latched;
}

void CompactManeuver::rejectPath()
{
  if (compact_preparation_ && !compact_commit_initialized_) {
    return;
  }
  cancelCompactProfileWork();
  if (follower_.active()) {
    ++diagnostics_.paths_abandoned;
    follower_.reset();
    if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver) {
      compact_path_adopted_ = false;
    }
  }
  if (compact_path_execution_.active()) {
    compact_path_execution_.clear();
    compact_path_adopted_ = false;
    compact_plan_pending_ = false;
  }
}

void CompactManeuver::onJ1Boundary(
  const CompactManeuverContext & context, std::string reason,
  std::string boundary_details)
{
  if (compact_goal_.profile_id.empty() || !compact_geometry_validator_) {
    if (!compact_profile_unavailable_reported_) {
      diagnostics_.compact_reason = "no_verified_compact_profile_for_robot_model";
      diagnostics_.compact_phase = "UNAVAILABLE";
      appendCompactEvent("compact_unavailable", context.time_sec,
        diagnostics_.compact_reason, boundary_details);
      compact_profile_unavailable_reported_ = true;
    }
    return;
  }
  diagnostics_.compact_reason = std::move(reason);
  diagnostics_.compact_phase = "COMMIT_WAIT";
  diagnostics_.state = "FACE_J1_WORKING_BOUNDARY";
  requestPhase();
  requestCompactManeuver(context, diagnostics_.compact_reason);
  appendCompactEvent("j1_boundary_reached", context.time_sec,
    diagnostics_.compact_reason, std::move(boundary_details));
}

void CompactManeuver::requestPhase()
{
  static_cast<void>(face_policy_.update(FaceTrackingPolicyInput{true, false}));
}

bool CompactManeuver::updateTargetIdentity(
  const std::optional<msg::TrackingTarget> & target, const double time_sec)
{
  const bool has_identity = target && target->mode == msg::TrackingTarget::FACE &&
    target->target_id >= 0;
  if (!has_identity) {
    return false;
  }
  const std::int64_t current = target->target_id;
  const bool changed = target_id_ && *target_id_ != current;
  if (changed) {
    target_bearing_motion_sign_ = 0;
    target_bearing_moving_ = false;
    appendCompactEvent("target_id_changed", time_sec, "selector_track_generation_changed");
  }
  target_id_ = current;
  return changed;
}

void CompactManeuver::beginTick(const CompactManeuverContext & context)
{
  const auto & target = context.target;
  if (!(target && target->mode == msg::TrackingTarget::FACE)) {
    return;
  }
  const Eigen::Isometry3d base_pose = context.state.getGlobalLinkTransform(base_link_);
  const Eigen::Vector3d ray = base_pose.linear().transpose() *
    (Eigen::Vector3d(target->face.x, target->face.y, target->face.z) -
    base_pose.translation());
  const Eigen::Vector3d face_velocity_base =
    base_pose.linear().transpose() * context.target_motion.face_velocity;
  const double ray_squared = ray.head<2>().squaredNorm();
  int bearing_sign = 0;
  if (ray_squared > 1.0e-9 && ray.allFinite() && face_velocity_base.allFinite()) {
    const double rate = (ray.x() * face_velocity_base.y() -
      ray.y() * face_velocity_base.x()) / ray_squared;
    if (std::isfinite(rate) && std::abs(rate) >= 0.02) {
      bearing_sign = rate > 0.0 ? 1 : -1;
    }
  }
  if (bearing_sign != 0 && target_bearing_motion_sign_ != 0 &&
    bearing_sign != target_bearing_motion_sign_)
  {
    appendCompactEvent("target_reverse_intent", context.time_sec,
      "filtered_target_bearing_reversed", "{\"from_sign\":" +
      std::to_string(target_bearing_motion_sign_) + ",\"to_sign\":" +
      std::to_string(bearing_sign) + "}");
  } else if (bearing_sign == 0 && target_bearing_moving_) {
    appendCompactEvent("target_stopped", context.time_sec, "filtered_target_bearing_stationary");
  }
  if (bearing_sign != 0) {
    target_bearing_motion_sign_ = bearing_sign;
  }
  target_bearing_moving_ = bearing_sign != 0;
}

void CompactManeuver::appendCompactEvent(
  std::string event, const double time_sec, std::string reason, std::string details_json)
{
  if (compact_events_.size() == kCompactEventCapacity) {
    ++compact_event_overflow_count_;
    compact_events_.pop_front();
  }
  compact_events_.push_back(CompactEvent{
      std::move(event), toNanoseconds(time_sec), diagnostics_.compact_generation,
      std::move(reason), std::move(details_json)});
  diagnostics_.compact_event_overflow_count = compact_event_overflow_count_;
}

bool CompactManeuver::compactManeuverActive() const noexcept
{
  return face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
         compact_commit_initialized_;
}

bool CompactManeuver::hasActiveProfile() const noexcept
{
  return compact_path_execution_.active();
}

void CompactManeuver::cancelCompactProfileWork()
{
  if (compact_profile_worker_) {
    compact_profile_worker_->cancel();
  }
  pending_compact_profile_work_.reset();
}

CompactProfileFinalGeometryValidator CompactManeuver::makeCompactProfileFinalGeometryValidator(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & state,
  const std::uint64_t scene_revision, const double radius_cap_m,
  std::optional<CompactReturnRegion> return_region) const
{
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  return validation.makeCompactProfileFinalGeometryValidator(
    scene, state, scene_revision, radius_cap_m, std::move(return_region));
}

std::uint64_t CompactManeuver::submitCompactProfileWork(
  CompactProfileWorkRequest request,
  const moveit::core::RobotState & geometry_state,
  const planning_scene::PlanningScene & scene,
  const double radius_cap_m, const double return_radius_cap_m,
  std::optional<CompactReturnRegion> return_region,
  const double time_sec, const bool resync)
{
  if (!compact_profile_worker_ || pending_compact_profile_work_ ||
    request.source_route.empty() ||
    request.segment_speed_scales.size() + 1U != request.source_route.size() ||
    request.published_prefix.empty())
  {
    return 0U;
  }
  PendingCompactProfileWork pending;
  pending.generation = request.generation;
  pending.scene_revision = request.scene_revision;
  pending.prefix_endpoint_sample_index = request.published_prefix.size() - 1U;
  pending.fixed_prefix = request.published_prefix;
  pending.source_route_sample_count = request.source_route.size();
  const bool return_resync = resync && compact_return_profile_active_;
  request.final_geometry_validator = makeCompactProfileFinalGeometryValidator(
    scene, geometry_state, request.scene_revision,
    return_resync ? return_radius_cap_m : radius_cap_m,
    return_resync ? return_region : std::nullopt);
  request.return_final_geometry_validator = makeCompactProfileFinalGeometryValidator(
    scene, geometry_state, request.scene_revision, return_radius_cap_m,
    std::move(return_region));
  request.return_profile_required = !resync;
  request.limits = compact_limits_;
  request.control_period_sec = parameters_.control_period_sec;
  const std::uint64_t request_id = compact_profile_worker_->submit(std::move(request));
  pending.request_id = request_id;
  pending.kind = resync ? CompactProfileWorkKind::kResync :
    (compact_commit_initialized_ ? CompactProfileWorkKind::kReadoption :
    CompactProfileWorkKind::kInitial);
  pending_compact_profile_work_ = std::move(pending);
  (void)time_sec;
  return request_id;
}

void CompactManeuver::consumeCompactProfileWorkResult(
  const double time_sec, const std::uint64_t scene_revision)
{
  if (!compact_profile_worker_ || !pending_compact_profile_work_) {
    return;
  }
  if (compact_path_execution_.profile() && compact_path_execution_.cursor() <
    pending_compact_profile_work_->prefix_endpoint_sample_index)
  {
    return;
  }
  auto result = compact_profile_worker_->takeResult();
  if (!result) {
    return;
  }
  const auto pending = *pending_compact_profile_work_;
  pending_compact_profile_work_.reset();
  const auto same_state = [](const JointMotionState & lhs, const JointMotionState & rhs) {
      return lhs.position.size() == rhs.position.size() &&
             lhs.velocity.size() == rhs.velocity.size() &&
             lhs.acceleration.size() == rhs.acceleration.size() &&
             (lhs.position.array() == rhs.position.array()).all() &&
             (lhs.velocity.array() == rhs.velocity.array()).all() &&
             (lhs.acceleration.array() == rhs.acceleration.array()).all();
    };
  bool source_prefix_matches = compact_path_execution_.profile() &&
    compact_path_execution_.profile()->samples.size() >= pending.fixed_prefix.size();
  for (std::size_t index = 0U;
    source_prefix_matches && index < pending.fixed_prefix.size(); ++index)
  {
    source_prefix_matches = same_state(
      compact_path_execution_.profile()->samples[index], pending.fixed_prefix[index]);
  }
  const bool current = result->request_id == pending.request_id &&
    result->generation == pending.generation && pending.scene_revision == scene_revision &&
    result->scene_revision == scene_revision &&
    pending.generation == diagnostics_.compact_generation && source_prefix_matches &&
    compact_path_execution_.cursor() <= pending.prefix_endpoint_sample_index;
  bool returned_prefix_matches = true;
  if (result->profile.profile) {
    returned_prefix_matches =
      result->profile.profile->samples.size() > pending.prefix_endpoint_sample_index &&
      result->profile.profile->prefix_endpoint_sample_index ==
      pending.prefix_endpoint_sample_index;
    for (std::size_t index = 0U;
      returned_prefix_matches && index < pending.fixed_prefix.size(); ++index)
    {
      returned_prefix_matches = same_state(
        result->profile.profile->samples[index], pending.fixed_prefix[index]);
    }
  }
  const auto accepted = [](const CompactDynamicsResult & dynamics) {
      return dynamics.profile && (dynamics.status == CompactDynamicsStatus::kPass ||
             dynamics.status == CompactDynamicsStatus::kRetimed ||
             dynamics.status == CompactDynamicsStatus::kPrefixViolation);
    };
  const bool geometry_certified = result->profile.profile &&
    result->geometry_certificate.passedFor(scene_revision,
      result->profile.profile->samples.size(),
      result->profile.profile->samples.empty() ? 0U : result->profile.profile->samples.size() - 1U);
  const bool return_geometry_certified = result->return_profile &&
    result->return_profile->profile && result->return_geometry_certificate &&
    result->return_geometry_certificate->passedFor(scene_revision,
      result->return_profile->profile->samples.size(),
      result->return_profile->profile->samples.empty() ? 0U :
      result->return_profile->profile->samples.size() - 1U);
  const bool returned_profile_accepted = result->return_profile &&
    accepted(*result->return_profile) && return_geometry_certified;
  bool resync_return_endpoint_valid = false;
  if (pending.kind == CompactProfileWorkKind::kResync && compact_return_profile_active_ &&
    compact_goal_.return_region && result->profile.profile &&
    !result->profile.profile->samples.empty())
  {
    moveit::core::RobotState return_state(robot_model_);
    return_state.setToDefaultValues();
    return_state.update(true);
    const Eigen::VectorXd & endpoint = result->profile.profile->samples.back().position;
    resync_return_endpoint_valid = compact_return_endpoint_.size() == endpoint.size() &&
      (compact_return_endpoint_ - endpoint).lpNorm<Eigen::Infinity>() <=
      kCompactGoalToleranceRad &&
      validateCompactReturnRegion(return_state, endpoint, *compact_goal_.return_region);
  }
  const bool mandatory_return_ready = pending.kind == CompactProfileWorkKind::kResync &&
    compact_return_profile_active_ ? resync_return_endpoint_valid : returned_profile_accepted;
  if (!current || !returned_prefix_matches || !accepted(result->profile) ||
    !geometry_certified || !mandatory_return_ready)
  {
    ++diagnostics_.paths_abandoned;
    compact_path_execution_.clear();
    compact_path_adopted_ = false;
    compact_plan_pending_ = false;
    if (pending.kind != CompactProfileWorkKind::kResync) {
      compact_prepared_return_profile_.reset();
      compact_return_endpoint_.resize(0);
    } else if (compact_return_profile_active_) {
      compact_return_profile_active_ = false;
      compact_return_path_pending_ = true;
    }
    diagnostics_.compact_phase = "REPLANNING";
    diagnostics_.last_plan_message = !current || !returned_prefix_matches ?
      "discarded stale compact profile result" :
      (!accepted(result->profile) ?
      "compact profile build or dynamics validation failed: " + result->profile.failure_reason :
      (!geometry_certified ? "compact final geometry validation rejected: " +
      result->geometry_certificate.failure_reason :
      "compact mandatory return profile or geometry validation failed"));
    const auto * failed_profile = result->profile.profile ?
      &*result->profile.profile : nullptr;
    const std::string rejection_details = std::string{"{\"preparation_status\":\"rejected\","} +
    "\"worker_build_ms\":" + jsonFiniteNumber(result->build_elapsed_ms) +
    ",\"geometry_validation_ms\":" + jsonFiniteNumber(result->geometry_elapsed_ms) +
    ",\"worker_total_ms\":" + jsonFiniteNumber(result->total_elapsed_ms) +
    ",\"source_route_samples\":" + std::to_string(pending.source_route_sample_count) +
    ",\"final_profile_samples\":" + std::to_string(
      failed_profile ? failed_profile->samples.size() : 0U) +
    ",\"builder_mode\":\"" +
    (failed_profile ? failed_profile->builder_mode : std::string{"unknown"}) + "\"" +
    ",\"builder_fallback_reason\":" + jsonString(
      failed_profile ? failed_profile->builder_fallback_reason : std::string{}) +
    ",\"geometry_prepare_duration_sec\":" + jsonFiniteNumber(
      failed_profile ? failed_profile->geometry_prepare_duration_sec : 0.0) +
    ",\"profile_generation_duration_sec\":" + jsonFiniteNumber(
      failed_profile ? failed_profile->profile_generation_duration_sec : 0.0) + "}";
    appendCompactEvent("torque_profile_prepared", time_sec,
      diagnostics_.last_plan_message, rejection_details);
    return;
  }

  const std::size_t profile_cursor = compact_path_execution_.cursor();
  const double profile_progress = compact_path_execution_.progressRad();
  compact_path_execution_.install(
    std::move(*result->profile.profile), compact_path_execution_.prefixLimits(),
    profile_progress, profile_cursor);
  diagnostics_.compact_profile_duration_sec = compact_path_execution_.profile()->duration_sec;
  const Eigen::VectorXd & selected_return_goal = returned_profile_accepted ?
    result->return_profile->profile->samples.back().position :
    compact_path_execution_.profile()->samples.back().position;
  appendCompactEvent("return_goal_selected", time_sec, diagnostics_.compact_reason,
    "{\"joint_positions_rad\":" + jsonVector(selected_return_goal) + "}");
  if (returned_profile_accepted) {
    compact_prepared_return_profile_ = std::move(*result->return_profile->profile);
    compact_return_endpoint_ = compact_prepared_return_profile_->samples.back().position;
  }
  const auto & metrics = result->profile;
  const auto * prepared_profile = compact_path_execution_.profile();
  const std::string preparation_kind = pending.kind == CompactProfileWorkKind::kInitial ?
    "initial" : (pending.kind == CompactProfileWorkKind::kReadoption ?
    "readoption" : "resync");
  const std::size_t prefix_endpoint_index = pending.prefix_endpoint_sample_index;
  const Eigen::VectorXd & prefix_endpoint_position =
    prepared_profile->samples[prefix_endpoint_index].position;
  std::string details = "{\"request_id\":" + std::to_string(pending.request_id) +
    ",\"preparation_kind\":\"" + preparation_kind + "\"" +
    ",\"prefix_endpoint_sample_index\":" + std::to_string(prefix_endpoint_index) +
    ",\"prefix_endpoint_joint_names\":" +
    jsonStringArray(group_->getVariableNames()) +
    ",\"prefix_endpoint_position_rad\":" + jsonVector(prefix_endpoint_position) +
    ",\"elapsed_ms\":" + jsonNumber(metrics.elapsed_ms) +
    ",\"worker_build_ms\":" + jsonNumber(result->build_elapsed_ms) +
    ",\"geometry_validation_ms\":" + jsonNumber(result->geometry_elapsed_ms) +
    ",\"worker_total_ms\":" + jsonNumber(result->total_elapsed_ms) +
    ",\"source_route_samples\":" + std::to_string(pending.source_route_sample_count) +
    ",\"final_profile_samples\":" + std::to_string(
    prepared_profile ? prepared_profile->samples.size() : 0U) +
    ",\"builder_mode\":\"" +
    (prepared_profile ? prepared_profile->builder_mode : std::string{"unknown"}) + "\"" +
    ",\"builder_fallback_reason\":" + jsonString(
    prepared_profile ? prepared_profile->builder_fallback_reason : std::string{}) +
    ",\"geometry_prepare_duration_sec\":" + jsonNumber(
    prepared_profile ? prepared_profile->geometry_prepare_duration_sec : 0.0) +
    ",\"profile_generation_duration_sec\":" + jsonNumber(
    prepared_profile ? prepared_profile->profile_generation_duration_sec : 0.0) +
    ",\"speed_scale\":" + jsonNumber(metrics.applied_speed_scale) +
    ",\"maximum_effort_ratio_before\":" +
    jsonNumber(metrics.maximum_effort_ratio_before) +
    ",\"maximum_effort_ratio_after\":" +
    jsonNumber(metrics.maximum_effort_ratio_after) +
    ",\"prefix_violation\":" +
    (metrics.immutable_prefix_over_limit ? "true" : "false") + "}";
  appendCompactEvent("torque_profile_prepared", time_sec,
    metrics.failure_reason, details);
  if (metrics.immutable_prefix_over_limit) {
    appendCompactEvent("published_prefix_effort_violation", time_sec,
      "already_published_prefix_exceeds_effort_limit", details);
  }
  if (pending.kind == CompactProfileWorkKind::kInitial) {
    diagnostics_.compact_phase = "PATH";
    diagnostics_.last_plan_message = "compact path committed with torque-checked profile";
  }
}

const JointMotionLimits & CompactManeuver::activeCommandLimits() const noexcept
{
  if (!compactManeuverActive()) {
    return limits_;
  }
  if (compact_path_execution_.profile() && compact_path_execution_.cursor() <
    compact_path_execution_.prefixEndpoint())
  {
    return compact_path_execution_.prefixLimits();
  }
  if (compact_return_profile_active_) {
    return compact_limits_;
  }
  return compact_return_path_pending_ ? compact_return_limits_ : compact_limits_;
}

void CompactManeuver::recordProfileCommandPublished(const bool published) noexcept
{
  compact_path_execution_.recordCommandPublished(published);
}

EmergencyBrakeTail CompactManeuver::profileBrakingTail(
  const JointMotionState & candidate_state,
  const EmergencyBrakeTailConfig & config) const
{
  if (!compactManeuverActive() || !compact_path_adopted_ ||
    (compact_return_path_pending_ && !compact_return_profile_active_))
  {
    EmergencyBrakeTail failure;
    failure.failure_reason = "no active direct compact PATH sample is available";
    return failure;
  }
  return compact_path_execution_.brakingTail(
    candidate_state, activeCommandLimits(), config);
}

double CompactManeuver::effectiveCompactRadiusCap() const noexcept
{
  return compact_return_path_pending_ ?
         compact_goal_.path_requirements.maximum_assembly_radius_m :
         (compact_radius_floor_latched_ ? kCompactRadiusFloorM :
         compact_goal_.path_requirements.maximum_assembly_radius_m);
}

std::vector<CompactGoal> CompactManeuver::selectDefaultCompactGoals(
  const moveit::core::RobotState & state,
  const planning_scene::PlanningScene & scene,
  const double measured_joint6) const
{
  const auto & names = group_->getVariableNames();
  const auto joint6 = std::find(names.begin(), names.end(), "joint6");
  if (joint6 == names.end() || !std::isfinite(measured_joint6) ||
    limits_.lower_position.size() != static_cast<Eigen::Index>(names.size()) ||
    limits_.upper_position.size() != static_cast<Eigen::Index>(names.size()))
  {
    return {};
  }
  const auto joint6_index = static_cast<Eigen::Index>(std::distance(names.begin(), joint6));
  constexpr double kTwoPi{6.28318530717958647692};
  constexpr double kBoundTolerance{1.0e-10};
  constexpr double kBoundInsetRad{0.02};

  struct RankedCandidate
  {
    CompactGoal goal;
    double joint6_distance{0.0};
    std::size_t winding_rank{0U};
  };
  std::vector<std::vector<RankedCandidate>> candidates_by_profile;
  const auto valid_candidate = [this, &state, &scene](const CompactGoal & candidate) {
      if (candidate.bounded_positions.size() !=
        static_cast<Eigen::Index>(group_->getVariableCount()) ||
        !candidate.bounded_positions.allFinite() ||
        (candidate.bounded_positions.array() < limits_.lower_position.array()).any() ||
        (candidate.bounded_positions.array() > limits_.upper_position.array()).any())
      {
        return false;
      }
      moveit::core::RobotState goal_state(state);
      goal_state.setJointGroupPositions(group_, candidate.bounded_positions);
      goal_state.update(true);
      const double radius = compact_geometry_validator_->maximumAssemblyRadius(goal_state);
      if (!std::isfinite(radius) || radius > kCompactRadiusFloorM + 1.0e-6) {
        return false;
      }
      Eigen::VectorXd position;
      goal_state.copyJointGroupPositions(group_, position);
      const JointMotionState stationary{
        position, Eigen::VectorXd::Zero(position.size()),
        Eigen::VectorXd::Zero(position.size())};
      const auto clearance = collision_validator_.validatePose(
        scene, goal_state, *group_, stationary);
      return clearance.input_valid && !clearance.unsafe;
    };

  for (const auto & profile : default_compact_goals_) {
    if (profile.bounded_positions.size() != static_cast<Eigen::Index>(names.size())) {
      continue;
    }
    const double nominal_joint6 = profile.bounded_positions[joint6_index];
    std::vector<RankedCandidate> profile_candidates;
    for (int winding = -2; winding <= 2; ++winding) {
      CompactGoal candidate = profile;
      double equivalent_joint6 = nominal_joint6 + winding * kTwoPi;
      if (equivalent_joint6 < limits_.lower_position[joint6_index] - kBoundTolerance ||
        equivalent_joint6 > limits_.upper_position[joint6_index] + kBoundTolerance)
      {
        continue;
      }
      const bool at_joint6_lower_bound = std::abs(
        equivalent_joint6 - limits_.lower_position[joint6_index]) <= kBoundTolerance;
      const bool at_joint6_upper_bound = std::abs(
        equivalent_joint6 - limits_.upper_position[joint6_index]) <= kBoundTolerance;
      if (at_joint6_lower_bound) {
        equivalent_joint6 = limits_.lower_position[joint6_index] + kBoundInsetRad;
      }
      if (!at_joint6_lower_bound && at_joint6_upper_bound) {
        equivalent_joint6 = limits_.upper_position[joint6_index] - kBoundInsetRad;
      }
      candidate.bounded_positions[joint6_index] = equivalent_joint6;
      if (!valid_candidate(candidate)) {
        continue;
      }

      profile_candidates.push_back(RankedCandidate{
          std::move(candidate), std::abs(equivalent_joint6 - measured_joint6), 0U});
    }
    std::stable_sort(profile_candidates.begin(), profile_candidates.end(),
      [](const RankedCandidate & lhs, const RankedCandidate & rhs) {
        return lhs.joint6_distance < rhs.joint6_distance;
      });
    for (std::size_t index = 0; index < profile_candidates.size(); ++index) {
      profile_candidates[index].winding_rank = index;
    }
    candidates_by_profile.push_back(std::move(profile_candidates));
  }

  std::vector<CompactGoal> result;
  std::size_t maximum_winding_rank = 0U;
  for (const auto & profile_candidates : candidates_by_profile) {
    maximum_winding_rank = std::max(maximum_winding_rank, profile_candidates.size());
  }
  for (std::size_t rank = 0U; rank < maximum_winding_rank; ++rank) {
    std::vector<RankedCandidate> tier;
    for (const auto & profile_candidates : candidates_by_profile) {
      const auto candidate = std::find_if(
        profile_candidates.begin(), profile_candidates.end(),
        [rank](const RankedCandidate & value) {return value.winding_rank == rank;});
      if (candidate != profile_candidates.end()) {
        tier.push_back(*candidate);
      }
    }
    std::sort(tier.begin(), tier.end(),
      [](const RankedCandidate & lhs, const RankedCandidate & rhs) {
        if (lhs.joint6_distance != rhs.joint6_distance) {
          return lhs.joint6_distance < rhs.joint6_distance;
        }
        return lhs.goal.profile_id == "compact_turn_positive_wrist" &&
               rhs.goal.profile_id != "compact_turn_positive_wrist";
      });
    for (auto & candidate : tier) {
      result.push_back(std::move(candidate.goal));
    }
  }
  return result;
}

bool CompactManeuver::includeCompactEntryEnvelopePosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  double * maximum_radius, double * maximum_abs_pitch_rad,
  double * maximum_abs_roll_rad) const
{
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  return validation.includeCompactEntryEnvelopePosition(
    state, position, maximum_radius, maximum_abs_pitch_rad, maximum_abs_roll_rad);
}

bool CompactManeuver::includeCompactEntryEnvelope(
  const CompactManeuverContext & context,
  double * maximum_radius, double * maximum_abs_pitch_rad,
  double * maximum_abs_roll_rad) const
{
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  return validation.includeCompactEntryEnvelope(
    context.state, context.time_sec, context.active_publication,
    context.latest_publication, maximum_radius, maximum_abs_pitch_rad, maximum_abs_roll_rad);
}

void CompactManeuver::maybeLatchCompactRadiusCap(
  const CompactManeuverContext & context)
{
  const auto & state = context.state;
  const double time_sec = context.time_sec;
  const double publication_lookback_sec = context.actual_feedback_time_sec ?
    std::min(time_sec, *context.actual_feedback_time_sec) : time_sec;
  if (!context.fresh_actual_state || !context.actual_feedback || !context.command_queue_state) {
    return;
  }
  const auto & actual_feedback = *context.actual_feedback;
  const auto & command_queue_state = *context.command_queue_state;
  if (!compactManeuverActive() || compact_radius_floor_latched_ ||
    actual_feedback.position.size() != group_->getVariableCount() ||
    command_queue_state.position.size() != group_->getVariableCount() ||
    !actual_feedback.position.allFinite() || !command_queue_state.position.allFinite())
  {
    return;
  }
  double actual_radius = 0.0;
  double command_radius = 0.0;
  if (!includeCompactEntryEnvelopePosition(
      state, actual_feedback.position, &actual_radius) ||
    !includeCompactEntryEnvelopePosition(
      state, command_queue_state.position, &command_radius) ||
    std::max(actual_radius, command_radius) > kCompactRadiusFloorM + 1.0e-6)
  {
    return;
  }

  // Queue geometry is expensive; inspect it only after measured and queued q
  // have both entered the floor, then latch once for this commitment.
  double envelope_radius = 0.0;
  CompactManeuverContext envelope_context = context;
  envelope_context.time_sec = publication_lookback_sec;
  if (!includeCompactEntryEnvelope(envelope_context, &envelope_radius) ||
    envelope_radius > kCompactRadiusFloorM + 1.0e-6)
  {
    return;
  }

  compact_radius_floor_latched_ = true;
  const double total_envelope_radius = std::max(actual_radius, command_radius);
  const std::string details =
    "{\"entry_radius_cap_m\":" + jsonNumber(diagnostics_.compact_entry_radius_cap_m) +
    ",\"effective_radius_cap_m\":" + jsonNumber(kCompactRadiusFloorM) +
    ",\"actual_feedback_stamp_ns\":" +
    std::to_string(toNanoseconds(publication_lookback_sec)) +
    ",\"actual_position_rad\":" + jsonVector(actual_feedback.position) +
    ",\"command_position_rad\":" + jsonVector(command_queue_state.position) +
    ",\"actual_radius_m\":" + jsonNumber(actual_radius) +
    ",\"command_radius_m\":" + jsonNumber(command_radius) +
    ",\"published_remainder_radius_m\":" + jsonNumber(envelope_radius) +
    ",\"radius_envelope_max_m\":" +
    jsonNumber(std::max(total_envelope_radius, envelope_radius)) + "}";
  appendCompactEvent(
    "compact_radius_floor_latched", time_sec, "full_envelope_inside_floor", details);
}

bool CompactManeuver::validateCompactPathPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  std::string * violation) const
{
  if (violation) {
    violation->clear();
  }
  if (!compactManeuverActive()) {
    return true;
  }
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  return validation.validateCompactPathPosition(
    state, position, effectiveCompactRadiusCap(), violation);
}

bool CompactManeuver::validateCompactReturnRegion(
  const moveit::core::RobotState & state, const Eigen::VectorXd & positions,
  const CompactReturnRegion & region) const
{
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  return validation.validateCompactReturnRegion(state, positions, region);
}

void CompactManeuver::requestCompactManeuver(
  const CompactManeuverContext & context, std::string reason)
{
  const auto & state = context.state;
  const double time_sec = context.time_sec;
  const auto & scene = context.scene;
  const auto scene_revision = context.scene_revision;
  const std::optional<JointMotionState> actual_feedback = context.fresh_actual_state ?
    context.actual_feedback : std::nullopt;
  const auto & command_queue_state = context.command_queue_state;
  const bool upright_roll_guard_armed = context.upright_roll_guard_armed;
  if (face_policy_.phase() != FaceTrackingPhase::kCompactManeuver ||
    compact_commit_initialized_ || pending_generation_ ||
    (compact_preparation_ && !compact_preparation_->exhausted))
  {
    return;
  }
  if (!upright_roll_guard_armed) {
    diagnostics_.compact_phase = "COMMIT_WAIT";
    diagnostics_.state = "COMPACT_COMMIT_WAITING_FOR_ROLL_GUARD";
    return;
  }
  if (compact_goal_.profile_id.empty() || !compact_geometry_validator_) {
    ++diagnostics_.compact_generation;
    diagnostics_.compact_reason = "no_verified_compact_profile_for_robot_model";
    appendCompactEvent("cancelled", time_sec, diagnostics_.compact_reason);
    face_policy_.clearForStop();
    diagnostics_.compact_phase = "UNAVAILABLE";
    diagnostics_.state = "COMPACT_UNAVAILABLE";
    return;
  }
  if (!actual_feedback || !command_queue_state ||
    actual_feedback->position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    command_queue_state->position.size() != actual_feedback->position.size() ||
    !actual_feedback->position.allFinite() || !command_queue_state->position.allFinite())
  {
    diagnostics_.compact_phase = "COMMIT_WAIT";
    diagnostics_.state = "COMPACT_PREPARING_WAITING_FOR_STATE";
    return;
  }

  Eigen::VectorXd exhausted_key = actual_feedback->position;
  if (compact_preparation_ && compact_preparation_->exhausted) {
    constexpr double kRetryJointChangeRad{0.05};
    const bool meaningful_state_change =
      compact_preparation_->exhausted_start_positions.size() != exhausted_key.size() ||
      (compact_preparation_->exhausted_start_positions - exhausted_key).
      lpNorm<Eigen::Infinity>() >= kRetryJointChangeRad;
    if (!meaningful_state_change &&
      compact_preparation_->exhausted_scene_revision == scene_revision)
    {
      face_policy_.clearForStop();
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
      return;
    }
    compact_preparation_.reset();
  }

  double entry_radius_cap = kCompactRadiusFloorM;
  double entry_envelope_max_abs_pitch_rad = 0.0;
  double entry_envelope_max_abs_roll_rad = 0.0;
  bool entry_envelope_valid = includeCompactEntryEnvelopePosition(
    state, groupPosition(state), &entry_radius_cap, &entry_envelope_max_abs_pitch_rad,
    &entry_envelope_max_abs_roll_rad);
  if (actual_feedback) {
    entry_envelope_valid = includeCompactEntryEnvelopePosition(
      state, actual_feedback->position, &entry_radius_cap,
      &entry_envelope_max_abs_pitch_rad, &entry_envelope_max_abs_roll_rad) &&
      entry_envelope_valid;
  }
  if (command_queue_state) {
    entry_envelope_valid = includeCompactEntryEnvelopePosition(
      state, command_queue_state->position, &entry_radius_cap,
      &entry_envelope_max_abs_pitch_rad, &entry_envelope_max_abs_roll_rad) &&
      entry_envelope_valid;
  }
  entry_envelope_valid = includeCompactEntryEnvelope(
    context, &entry_radius_cap, &entry_envelope_max_abs_pitch_rad,
    &entry_envelope_max_abs_roll_rad) &&
    entry_envelope_valid;

  if (!entry_envelope_valid) {
    diagnostics_.compact_phase = "COMMIT_WAIT";
    diagnostics_.state = "COMPACT_PREPARING_WAITING_FOR_VALID_ENTRY_ENVELOPE";
    return;
  }

  std::vector<CompactGoal> candidates;
  if (select_default_compact_goal_) {
    const auto & names = group_->getVariableNames();
    const auto joint6 = std::find(names.begin(), names.end(), "joint6");
    if (joint6 == names.end() ||
      actual_feedback->position.size() != static_cast<Eigen::Index>(names.size()))
    {
      ++diagnostics_.compact_generation;
      diagnostics_.compact_reason = "compact_default_goal_unavailable";
      appendCompactEvent("cancelled", time_sec, diagnostics_.compact_reason);
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "UNAVAILABLE";
      diagnostics_.state = "COMPACT_UNAVAILABLE";
      return;
    }
    const auto joint6_index = static_cast<Eigen::Index>(
      std::distance(names.begin(), joint6));
    candidates = selectDefaultCompactGoals(
      state, scene, actual_feedback->position[joint6_index]);
    if (candidates.empty()) {
      ++diagnostics_.compact_generation;
      diagnostics_.compact_reason = "no_default_compact_goal_passed_pose_validation";
      CompactPreparation exhausted;
      exhausted.scene_revision = scene_revision;
      exhausted.exhausted_start_positions = actual_feedback->position;
      exhausted.exhausted_scene_revision = scene_revision;
      exhausted.exhausted = true;
      compact_preparation_ = std::move(exhausted);
      appendCompactEvent("cancelled", time_sec, diagnostics_.compact_reason,
        "{\"candidate_count\":0}");
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
      return;
    }
  } else {
    candidates.push_back(compact_goal_);
  }

  for (auto & candidate : candidates) {
    if (candidate.profile_id == "compact_turn_negative_wrist" ||
      candidate.profile_id == "compact_turn_positive_wrist")
    {
      candidate.path_requirements.staged_route = CompactStagedRouteRequirements{};
    }
  }

  CompactPreparation preparation;
  preparation.candidates = std::move(candidates);
  preparation.submitted_start_positions = actual_feedback->position;
  preparation.scene_revision = scene_revision;
  preparation.entry_radius_cap_m = entry_radius_cap;
  compact_preparation_ = std::move(preparation);
  ++diagnostics_.compact_generation;
  diagnostics_.compact_reason = std::move(reason);
  diagnostics_.compact_phase = "PREPARING";
  diagnostics_.tracking_phase = "FACE";
  diagnostics_.state = "FACE_COMPACT_PREPARING";
  diagnostics_.last_plan_message = "compact path preparation started";
  appendCompactEvent("preparing", time_sec, diagnostics_.compact_reason,
    "{\"candidate_count\":" +
    std::to_string(compact_preparation_->candidates.size()) + "}");
}

void CompactManeuver::submitCompactPlan(const CompactManeuverContext & context)
{
  const auto & state = context.state;
  const auto scene_revision = context.scene_revision;
  const double time_sec = context.time_sec;
  if (pending_generation_ || planner_.isBusy()) {
    return;
  }
  ++diagnostics_.plans_requested;
  CompactGoal request = compact_goal_;
  request.path_requirements.maximum_assembly_radius_m = effectiveCompactRadiusCap();
  pending_generation_ = planner_.submit(state, request, scene_revision);
  active_scene_revision_ = scene_revision;
  compact_plan_pending_ = true;
  diagnostics_.compact_phase = "PLANNING";
  if (diagnostics_.last_plan_message.empty() ||
    diagnostics_.last_plan_message == "not requested")
  {
    diagnostics_.last_plan_message = "compact path submitted";
  }
  (void)time_sec;
}

CompactManeuver::CompactPreparationSubmitOutcome
CompactManeuver::submitCompactPreparationPlan(const CompactManeuverContext & context)
{
  const auto & state = context.state;
  const double time_sec = context.time_sec;
  const auto scene_revision = context.scene_revision;
  const auto & actual_feedback = context.actual_feedback;
  const auto & command_queue_state = context.command_queue_state;
  const auto * active_publication = context.active_publication;
  const auto * latest_publication = context.latest_publication;
  if (!compact_preparation_ || compact_preparation_->exhausted || pending_generation_ ||
    planner_.isBusy() || !actual_feedback || !command_queue_state)
  {
    return CompactPreparationSubmitOutcome::kNotSubmitted;
  }
  const auto valid_motion_state = [this](const JointMotionState & motion) {
      const auto size = static_cast<Eigen::Index>(group_->getVariableCount());
      return motion.position.size() == size && motion.velocity.size() == size &&
             motion.acceleration.size() == size && motion.position.allFinite() &&
             motion.velocity.allFinite() && motion.acceleration.allFinite();
    };
  if (!valid_motion_state(*actual_feedback) || !valid_motion_state(*command_queue_state)) {
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_WAITING_FOR_VALID_MOTION_STATE";
    return CompactPreparationSubmitOutcome::kNotSubmitted;
  }
  auto & preparation = *compact_preparation_;
  if (scene_revision != preparation.scene_revision ||
    preparation.candidate_index >= preparation.candidates.size())
  {
    return CompactPreparationSubmitOutcome::kNotSubmitted;
  }

  double entry_radius_cap = kCompactRadiusFloorM;
  double maximum_abs_pitch = 0.0;
  double maximum_abs_roll = 0.0;
  bool envelope_valid = includeCompactEntryEnvelopePosition(
    state, groupPosition(state), &entry_radius_cap, &maximum_abs_pitch,
    &maximum_abs_roll);
  envelope_valid = includeCompactEntryEnvelopePosition(
    state, actual_feedback->position, &entry_radius_cap, &maximum_abs_pitch,
    &maximum_abs_roll) && envelope_valid;
  envelope_valid = includeCompactEntryEnvelopePosition(
    state, command_queue_state->position, &entry_radius_cap, &maximum_abs_pitch,
    &maximum_abs_roll) && envelope_valid;
  envelope_valid = includeCompactEntryEnvelope(
    context, &entry_radius_cap,
    &maximum_abs_pitch, &maximum_abs_roll) && envelope_valid;
  if (!envelope_valid) {
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_ENTRY_UNSAFE";
    diagnostics_.last_plan_message = "compact entry envelope unsafe: valid=" +
      std::to_string(envelope_valid) + ", max_abs_roll_rad=" +
      std::to_string(maximum_abs_roll);
    return CompactPreparationSubmitOutcome::kEntryEnvelopeUnsafe;
  }

  const JointMotionLimits prefix_limits = activeCommandLimits();
  std::string prefix_failure;
  const auto published_prefix = detail::findCompactPublishedPrefix(
    *command_queue_state, time_sec, latest_publication, active_publication,
    parameters_.control_period_sec, prefix_limits, &prefix_failure);
  if (!published_prefix) {
    diagnostics_.last_plan_message =
      "compact candidate not submitted: no validated stationary prefix: " + prefix_failure;
    return CompactPreparationSubmitOutcome::kNotSubmitted;
  }
  const JointMotionState & future_start = published_prefix->samples.back();

  CompactGoal request = preparation.candidates[preparation.candidate_index];
  const auto & joint_names = group_->getVariableNames();
  const auto joint1_it = std::find(joint_names.begin(), joint_names.end(), "joint1");
  if (joint1_it == joint_names.end() || !posture_task_builder_) {
    diagnostics_.last_plan_message = "compact return region calibration is unavailable";
    return CompactPreparationSubmitOutcome::kCandidateRejected;
  }
  const auto joint1_index = static_cast<Eigen::Index>(
    std::distance(joint_names.begin(), joint1_it));
  if (future_start.position.size() != static_cast<Eigen::Index>(joint_names.size()) ||
    request.bounded_positions.size() != future_start.position.size())
  {
    diagnostics_.last_plan_message = "compact return region joint dimensions are invalid";
    return CompactPreparationSubmitOutcome::kCandidateRejected;
  }
  moveit::core::RobotState reference_state(state);
  reference_state.setJointGroupPositions(group_, future_start.position);
  reference_state.update(true);
  CompactReturnRegion return_region;
  return_region.reference_positions = future_start.position;
  return_region.center_world = reference_state.getGlobalLinkTransform(command_link_).translation();
  return_region.outward_x_world = return_region.center_world -
    reference_state.getGlobalLinkTransform(base_link_).translation();
  return_region.outward_x_world.z() = 0.0;
  if (!return_region.center_world.allFinite() ||
    !return_region.outward_x_world.allFinite() ||
    return_region.outward_x_world.head<2>().norm() <= 1.0e-6)
  {
    diagnostics_.last_plan_message = "compact return outward direction is unavailable";
    return CompactPreparationSubmitOutcome::kCandidateRejected;
  }
  return_region.outward_x_world.normalize();
  const double turn_delta = request.bounded_positions[joint1_index] -
    future_start.position[joint1_index];
  const double turn_direction = std::abs(turn_delta) > 1.0e-3 ? turn_delta :
    (context.j1_boundary_assessment.edge_direction == BaseJointDirection::kNegative ? -1.0 : 1.0);
  return_region.radius_m = kCompactReturnRadiusM;
  return_region.direction_tolerance_rad = kCompactReturnDirectionToleranceRad;
  return_region.horizontal_edge_tolerance_rad =
    kCompactReturnHorizontalEdgeToleranceRad;
  return_region.target_joint1_rad = future_start.position[joint1_index] +
    std::copysign(2.0 * std::acos(-1.0), turn_direction);
  return_region.joint1_tolerance_rad = kCompactGoalToleranceRad;
  return_region.comfortable_joint3_lower = posture_task_builder_->comfortableJoint3Lower();
  return_region.comfortable_joint3_upper = posture_task_builder_->comfortableJoint3Upper();
  return_region.minimum_elbow_height_m = parameters_.maximum_linear_reference_mps *
    parameters_.control_period_sec;
  if (!std::isfinite(return_region.radius_m) || return_region.radius_m <= 0.0 ||
    !std::isfinite(return_region.comfortable_joint3_lower) ||
    !std::isfinite(return_region.comfortable_joint3_upper) ||
    return_region.comfortable_joint3_lower > return_region.comfortable_joint3_upper ||
    !std::isfinite(return_region.minimum_elbow_height_m))
  {
    diagnostics_.last_plan_message = "compact return posture bounds are unavailable";
    return CompactPreparationSubmitOutcome::kCandidateRejected;
  }
  preparation.return_region = return_region;
  request.return_region = return_region;
  preparation.candidates[preparation.candidate_index].return_region = return_region;
  preparation.exhausted_start_positions = actual_feedback->position;
  preparation.submitted_start_positions = future_start.position;
  request.path_requirements.maximum_assembly_radius_m = entry_radius_cap;
  request.path_requirements.base_link_name = base_link_->getName();
  request.path_requirements.screen_link_name = command_link_->getName();

  moveit::core::RobotState start(state);
  start.setJointGroupPositions(group_, future_start.position);
  start.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(future_start.position.size()));
  start.setJointGroupAccelerations(
    group_, Eigen::VectorXd::Zero(future_start.position.size()));
  start.update(true);
  preparation.entry_radius_cap_m = entry_radius_cap;
  preparation.scene_revision = scene_revision;
  const std::string frozen_region_details =
    "{\"reference_joint_positions_rad\":" + jsonVector(return_region.reference_positions) +
    ",\"center_position_m\":" + jsonVector(return_region.center_world) +
    ",\"outward_direction\":" + jsonVector(return_region.outward_x_world) +
    ",\"position_tolerance_m\":" + jsonNumber(return_region.radius_m) +
    ",\"direction_tolerance_rad\":" + jsonNumber(return_region.direction_tolerance_rad) +
    ",\"horizontal_edge_tolerance_rad\":" +
    jsonNumber(return_region.horizontal_edge_tolerance_rad) +
    ",\"joint3_lower_rad\":" + jsonNumber(return_region.comfortable_joint3_lower) +
    ",\"joint3_upper_rad\":" + jsonNumber(return_region.comfortable_joint3_upper) +
    ",\"minimum_elbow_height_m\":" + jsonNumber(return_region.minimum_elbow_height_m) +
    ",\"joint1_target_rad\":" + jsonNumber(return_region.target_joint1_rad) +
    ",\"joint1_tolerance_rad\":" + jsonNumber(return_region.joint1_tolerance_rad) + "}";
  appendCompactEvent("return_workspace_frozen", time_sec, {}, frozen_region_details);
  ++diagnostics_.plans_requested;
  pending_generation_ = planner_.submit(start, request, scene_revision);
  preparation.generation = *pending_generation_;
  compact_plan_pending_ = true;
  diagnostics_.compact_phase = "PREPARING";
  diagnostics_.tracking_phase = "FACE";
  diagnostics_.state = "FACE_COMPACT_PREPARING";
  diagnostics_.last_plan_message = "compact candidate path submitted: " + request.profile_id;
  return CompactPreparationSubmitOutcome::kSubmitted;
}

bool CompactManeuver::compactSettled(
  const JointMotionState & actual_feedback,
  const std::optional<JointMotionState> & command_queue_state,
  const Eigen::VectorXd & goal) const
{
  // Actual feedback provides measured q/v only; its acceleration is zero-filled.
  // The command tail is the controller's source for the jerk-limited v/a state.
  return compactGoalPositionReached(actual_feedback, goal) &&
         motionSettled(actual_feedback, command_queue_state);
}

bool CompactManeuver::motionSettled(
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

void CompactManeuver::recordCompactSettledEvent(
  const CompactManeuverContext & context, const bool settled_now)
{
  const double time_sec = context.time_sec;
  const auto & actual_feedback = context.actual_feedback;
  const auto & actual_feedback_time_sec = context.actual_feedback_time_sec;
  const auto & command_queue_state = context.command_queue_state;
  if (settled_now && !diagnostics_.compact_actual_settled && actual_feedback &&
    actual_feedback_time_sec && command_queue_state)
  {
    const double actual_goal_error =
      (actual_feedback->position - compact_goal_.bounded_positions).
      lpNorm<Eigen::Infinity>();
    const double actual_speed = actual_feedback->velocity.lpNorm<Eigen::Infinity>();
    const double queue_goal_error =
      (command_queue_state->position - compact_goal_.bounded_positions).
      lpNorm<Eigen::Infinity>();
    const double queue_speed = command_queue_state->velocity.lpNorm<Eigen::Infinity>();
    const double queue_acceleration =
      command_queue_state->acceleration.lpNorm<Eigen::Infinity>();
    const std::string details =
      "{\"actual_feedback_stamp_ns\":" +
      std::to_string(toNanoseconds(*actual_feedback_time_sec)) +
      ",\"actual_goal_error_max_rad\":" + jsonNumber(actual_goal_error) +
      ",\"actual_speed_max_rad_s\":" + jsonNumber(actual_speed) +
      ",\"actual_position_rad\":" + jsonVector(actual_feedback->position) +
      ",\"actual_velocity_rad_s\":" + jsonVector(actual_feedback->velocity) +
      ",\"command_queue_goal_error_max_rad\":" + jsonNumber(queue_goal_error) +
      ",\"command_queue_speed_max_rad_s\":" + jsonNumber(queue_speed) +
      ",\"command_queue_acceleration_max_rad_s2\":" +
      jsonNumber(queue_acceleration) +
      ",\"command_queue_position_rad\":" + jsonVector(command_queue_state->position) +
      ",\"command_queue_velocity_rad_s\":" + jsonVector(command_queue_state->velocity) +
      ",\"command_queue_acceleration_rad_s2\":" +
      jsonVector(command_queue_state->acceleration) + "}";
    appendCompactEvent("compact_goal_settled", time_sec, "actual_and_queue_stopped", details);
  }
  diagnostics_.compact_actual_settled = settled_now;
}

bool CompactManeuver::appendCompactPathConstraints(
  const moveit::core::RobotState & state, const JointMotionState & motion_state,
  MotionReferenceResult & result)
{
  const auto fail = [this](std::string reason) {
      diagnostics_.last_plan_message = "compact radius support unavailable: " + reason;
      return false;
    };
  if (!compact_geometry_validator_ || !group_ || !motion_state.position.allFinite() ||
    motion_state.position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    motion_state.velocity.size() != motion_state.position.size() ||
    motion_state.acceleration.size() != motion_state.position.size() ||
    !motion_state.velocity.allFinite() || !motion_state.acceleration.allFinite())
  {
    return fail("invalid validator, group, or queue position");
  }
  moveit::core::RobotState task_state(state);
  task_state.setJointGroupPositions(group_, motion_state.position);
  task_state.update(true);
  const auto support_set = compact_geometry_validator_->perLinkRadiusSupports(task_state, *group_);
  const double radius_cap = effectiveCompactRadiusCap();
  if (!support_set || !std::isfinite(support_set->maximum_radius_m) ||
    !std::isfinite(radius_cap) ||
    support_set->maximum_radius_m > radius_cap + 1.0e-6)
  {
    return fail("maximum radius=" + jsonNumber(support_set ?
        support_set->maximum_radius_m : std::numeric_limits<double>::quiet_NaN()) +
      ", cap=" + jsonNumber(radius_cap));
  }

  std::vector<LinearVelocityConstraint> constraints;
  constraints.reserve(support_set->supports.size());
  for (const auto & support : support_set->supports) {
    if (!std::isfinite(support.radius_m) ||
      support.outward_gradient.size() != motion_state.position.size() ||
      !support.outward_gradient.allFinite() || support.radius_m > radius_cap + 1.0e-6)
    {
      return fail("invalid per-link support radius=" + jsonNumber(support.radius_m) +
        ", cap=" + jsonNumber(radius_cap) + ", gradient_size=" +
        std::to_string(support.outward_gradient.size()));
    }
    const double gradient_norm = support.outward_gradient.norm();
    if (!std::isfinite(gradient_norm)) {
      return fail("nonfinite outward radius gradient");
    }
    if (gradient_norm <= 1.0e-10) {
      if (support.radius_m > radius_cap) {
        return fail("zero radius gradient outside cap: radius=" +
          jsonNumber(support.radius_m) + ", cap=" + jsonNumber(radius_cap));
      }
      continue;
    }

    // The entry cap can be initialized from this exact pose. Reserve only
    // consumes positive slack; a virtual negative gap must not force motion
    // inward when the measured assembly is already within its committed cap.
    const double remaining_radius_m = std::max(
      0.0, radius_cap - support.radius_m -
      collision_validator_.config().numerical_distance_reserve_m);
    const auto row = makeOutwardStoppingConstraint(
      support.outward_gradient, remaining_radius_m, motion_state, activeCommandLimits(),
      parameters_.control_period_sec, kCompactQueueTailResidualLatencySec);
    if (!row) {
      return fail("stopping row infeasible: radius=" + jsonNumber(support.radius_m) +
        ", cap=" + jsonNumber(radius_cap) + ", gradient_norm=" +
        jsonNumber(gradient_norm) + ", remaining=" + jsonNumber(remaining_radius_m) +
        ", velocity_max=" + jsonNumber(motion_state.velocity.cwiseAbs().maxCoeff()) +
        ", acceleration_max=" +
        jsonNumber(motion_state.acceleration.cwiseAbs().maxCoeff()));
    }
    constraints.push_back(*row);
  }
  result.compact_radius_constraints = std::move(constraints);
  result.compact_roll_constraints.clear();
  result.compact_pitch_constraints.clear();
  return true;
}

bool CompactManeuver::compactGoalPositionReached(
  const JointMotionState & feedback, const Eigen::VectorXd & goal) const
{
  if (feedback.position.size() != goal.size() || !feedback.position.allFinite() ||
    (feedback.position - goal).lpNorm<Eigen::Infinity>() > kCompactGoalToleranceRad ||
    feedback.position.size() != limits_.lower_position.size())
  {
    return false;
  }
  // The committed profile is selected to restore both bounded winding axes.
  // Never wrap their actual coordinates to manufacture reserve.
  for (const char * name : {"joint1", "joint4"}) {
    const auto found = std::find(
      group_->getVariableNames().begin(), group_->getVariableNames().end(), name);
    if (found == group_->getVariableNames().end()) {
      return false;
    }
    const auto index = std::distance(group_->getVariableNames().begin(), found);
    const auto reserve = measureBoundedJointReserve(
      feedback.position[index], limits_.lower_position[index], limits_.upper_position[index]);
    if (!reserve.inside_bounds || reserve.nearest_bound_margin_rad < kCompactReserveRad) {
      return false;
    }
  }
  return true;
}

std::optional<HierarchicalVelocityTask> CompactManeuver::compactTask(
  const CompactManeuverContext & context, MotionReferenceResult & result)
{
  const auto & state = context.state;
  const auto & position = context.position;
  const auto & scene = context.scene;
  const auto & command_queue_state = context.command_queue_state;
  const double time_sec = context.time_sec;
  const auto scene_revision = context.scene_revision;
  const auto * active_publication = context.active_publication;
  const auto * latest_publication = context.latest_publication;
  if (compact_path_execution_.profile()) {
    if (!command_queue_state || compact_path_execution_.cursor() >=
      compact_path_execution_.profile()->samples.size())
    {
      diagnostics_.compact_phase = "PATH_WAITING_FOR_QUEUE_STATE";
      return std::nullopt;
    }
    const auto & expected =
      compact_path_execution_.profile()->samples[compact_path_execution_.cursor()];
    const auto queue_match = compact_path_execution_.matchQueueState(
      *command_queue_state, kCompactProfilePositionToleranceRad, 1.0e-6, 1.0e-5);
    bool queue_matches = queue_match.status == CompactQueueMatchStatus::kMatchedCurrent;
    std::string resync_failure_reason;
    if (!queue_matches && queue_match.status == CompactQueueMatchStatus::kMatchedFuture &&
      compact_path_adopted_ && active_scene_revision_ &&
      *active_scene_revision_ == scene_revision)
    {
      queue_matches = compact_path_execution_.commitPublishedQueueMatch(queue_match.sample_index);
    }
    if (!queue_matches && !pending_compact_profile_work_ && compact_path_adopted_ &&
      active_scene_revision_ && *active_scene_revision_ == scene_revision &&
      command_queue_state->position.size() == expected.position.size() &&
      command_queue_state->velocity.size() == expected.velocity.size() &&
      command_queue_state->acceleration.size() == expected.acceleration.size() &&
      command_queue_state->position.allFinite() && command_queue_state->velocity.allFinite() &&
      command_queue_state->acceleration.allFinite() &&
      compact_path_execution_.cursor() >= compact_path_execution_.prefixEndpoint() &&
      compact_path_execution_.profile()->path_progress_rad_per_sample.size() ==
      compact_path_execution_.profile()->samples.size() &&
      compact_path_execution_.profile()->speed_scale_per_interval.size() + 1U ==
      compact_path_execution_.profile()->samples.size())
    {
      const auto match_status = [&queue_match]() {
          switch (queue_match.status) {
            case CompactQueueMatchStatus::kMatchedCurrent: return "matched_current";
            case CompactQueueMatchStatus::kMatchedFuture: return "matched_future";
            case CompactQueueMatchStatus::kNoMatch: return "no_match";
          }
          return "unknown";
        }();
      const auto publication_times = [](const PublishedTrajectoryRecord * publication) {
          const auto time_at = [publication](const std::size_t index) {
              return publication && index < publication->execution_queue.size() ?
                     jsonFiniteNumber(
                publication->execution_queue[index].time_stamp.seconds()) : std::string{"null"};
            };
          const std::string id = publication ? std::to_string(publication->id) : "null";
          const std::string stationary = publication ?
            jsonFiniteNumber(publication->stationary_time.seconds()) : "null";
          const std::string start = publication && !publication->execution_queue.empty() ?
            time_at(0U) : "null";
          const std::string end = publication && !publication->execution_queue.empty() ?
            time_at(publication->execution_queue.size() - 1U) : "null";
          return std::string{"\"id\":"} + id +
                 ",\"start_time_sec\":" + start +
                 ",\"stationary_time_sec\":" + stationary +
                 ",\"end_time_sec\":" + end;
        };
      const auto & group_joint_names = group_->getVariableNames();
      const std::string details =
        "{\"match_status\":" + jsonString(match_status) +
        ",\"match_sample_index\":" + std::to_string(queue_match.sample_index) +
        ",\"cursor\":" + std::to_string(compact_path_execution_.cursor()) +
        ",\"profile_size\":" + std::to_string(
          compact_path_execution_.profile()->samples.size()) +
        ",\"prefix_endpoint_index\":" + std::to_string(
          compact_path_execution_.prefixEndpoint()) +
        ",\"expected_current_position_rad\":" + jsonVector(expected.position) +
        ",\"expected_current_velocity_radps\":" + jsonVector(expected.velocity) +
        ",\"expected_current_acceleration_radps2\":" + jsonVector(expected.acceleration) +
        ",\"command_queue_position_rad\":" + jsonVector(command_queue_state->position) +
        ",\"command_queue_velocity_radps\":" + jsonVector(command_queue_state->velocity) +
        ",\"command_queue_acceleration_radps2\":" +
        jsonVector(command_queue_state->acceleration) +
        ",\"joint_names\":" + jsonStringArray(group_joint_names) +
        ",\"time_sec\":" + jsonFiniteNumber(time_sec) +
        ",\"scene_revision\":" + std::to_string(scene_revision) +
        ",\"active_scene_revision\":" + (active_scene_revision_ ?
        std::to_string(*active_scene_revision_) : std::string{"null"}) +
        ",\"active_publication\":{" + publication_times(active_publication) + "}" +
        ",\"latest_publication\":{" + publication_times(latest_publication) + "}" + "}";
      appendCompactEvent("queue_mismatch_resync", time_sec,
        match_status, details);

      const auto projected_progress = compact_path_execution_.projectQueueStateToProgress(
        *command_queue_state, kCompactProfilePositionToleranceRad);
      std::string prefix_failure;
      const auto published_prefix = detail::findCompactPublishedPrefix(
        *command_queue_state, time_sec, latest_publication, active_publication,
        parameters_.control_period_sec, activeCommandLimits(), &prefix_failure);
      std::optional<double> stopped_progress;
      if (projected_progress && published_prefix) {
        stopped_progress = compact_path_execution_.projectQueueStateToProgress(
          published_prefix->samples.back(), kCompactProfilePositionToleranceRad,
          *projected_progress);
      }
      const auto suffix = stopped_progress ?
        compact_path_execution_.remainingRoute(*stopped_progress) : std::nullopt;
      if (projected_progress && published_prefix && stopped_progress && suffix &&
        suffix->path.size() >= 2U &&
        suffix->speed_scales.size() + 1U == suffix->path.size())
      {
        std::vector<Eigen::VectorXd> remaining_route = suffix->path;
        std::vector<double> remaining_scales = suffix->speed_scales;
        const JointMotionState & stopped_start = published_prefix->samples.back();
        if ((remaining_route.front() - stopped_start.position).norm() > 1.0e-12) {
          remaining_route.insert(remaining_route.begin(), stopped_start.position);
          remaining_scales.insert(remaining_scales.begin(), remaining_scales.front());
        }
        const double progress_at_retime =
          compact_path_execution_.absoluteProgressRad(*stopped_progress);
        const double path_scale = compact_speed_scale_ * (compact_return_profile_active_ ?
          compact_return_speed_scale_ : 1.0);
        CompactProfileWorkRequest request;
        request.generation = diagnostics_.compact_generation;
        request.scene_revision = scene_revision;
        request.source_route = remaining_route;
        request.segment_speed_scales = remaining_scales;
        request.start_state = stopped_start;
        request.published_prefix = published_prefix->samples;
        request.prefix_limits = activeCommandLimits();
        request.return_profile = compact_return_profile_active_ ? std::nullopt :
          compact_prepared_return_profile_;
        request.requested_speed_scale = path_scale;
        request.return_speed_scale = 1.0;
        if (submitCompactProfileWork(
            std::move(request), state, scene, effectiveCompactRadiusCap(),
            compact_goal_.path_requirements.maximum_assembly_radius_m,
            compact_goal_.return_region, time_sec, true) != 0U)
        {
          compact_path_execution_.install(
            detail::compactProfileForPublishedPrefix(
              published_prefix->samples, parameters_.control_period_sec),
            compact_limits_, progress_at_retime);
          diagnostics_.compact_profile_duration_sec =
            std::numeric_limits<double>::quiet_NaN();
          queue_matches = true;
        } else {
          resync_failure_reason = "could not submit smooth compact continuation";
        }
      } else {
        resync_failure_reason = !projected_progress ?
          "queue state cannot be projected uniquely onto the forward smooth profile" :
          (!published_prefix ? prefix_failure :
          (!stopped_progress ?
          "published brake endpoint cannot be projected forward onto the smooth profile" :
          "forward smooth profile remainder is unavailable"));
      }
    }
    if (!queue_matches) {
      cancelCompactProfileWork();
      ++diagnostics_.paths_abandoned;
      compact_path_execution_.clear();
      compact_path_adopted_ = false;
      follower_.reset();
      diagnostics_.compact_phase = "REPLANNING";
      diagnostics_.last_plan_message =
        "compact profile queue state changed; preparing from the current queue";
      if (!resync_failure_reason.empty()) {
        diagnostics_.last_plan_message += ": " + resync_failure_reason;
      }
      return std::nullopt;
    }
    if (compact_path_execution_.cursor() + 1U < compact_path_execution_.profile()->samples.size()) {
      result.compact_profile_velocity =
        compact_path_execution_.profile()->samples[compact_path_execution_.cursor() + 1U].velocity;
      result.compact_profile_command = true;
      diagnostics_.compact_phase = compact_return_profile_active_ ? "RETURN_PATH" : "PATH";
      diagnostics_.path_progress_rad = compact_path_execution_.progressRad();
      diagnostics_.path_distance_rad = std::max(
        0.0, compact_path_execution_.totalPathLengthRad() -
        compact_path_execution_.progressRad());
      if (!compact_return_profile_active_ && !compact_path_started_event_emitted_) {
        appendCompactEvent("path_started", time_sec, diagnostics_.compact_reason);
        compact_path_started_event_emitted_ = true;
      }
      return std::nullopt;
    }
    if (pending_compact_profile_work_) {
      diagnostics_.compact_phase = compact_return_profile_active_ ?
        "RETURN_PATH_PREPARING" : "PATH_PREPARING_TORQUE_PROFILE";
      return std::nullopt;
    }
    compact_path_execution_.clear();
    ++diagnostics_.paths_completed;
    compact_path_endpoint_reached_ = true;
    compact_path_adopted_ = false;
    follower_.reset();
    diagnostics_.compact_phase = compact_return_profile_active_ ?
      "RETURN_SETTLING" : "SETTLING";
    appendCompactEvent(compact_return_profile_active_ ?
      "return_profile_completed" : "profile_completed", time_sec,
      "timed_path_endpoint_reached", "{\"profile_duration_sec\":" +
      jsonNumber(diagnostics_.compact_profile_duration_sec) + "}");
    return std::nullopt;
  }

  if (compact_return_profile_active_) {
    return std::nullopt;
  }

  if (compact_path_adopted_ && !compact_path_endpoint_reached_) {
    compact_path_adopted_ = false;
    compact_plan_pending_ = false;
    follower_.reset();
    diagnostics_.compact_phase = "REPLANNING";
    diagnostics_.last_plan_message = "compact PATH has no accepted timed profile";
    return std::nullopt;
  }
  if (!compact_path_endpoint_reached_) {
    if (compact_plan_pending_) {
      return std::nullopt;
    }
    return std::nullopt;
  }
  if ((position - compact_goal_.bounded_positions).lpNorm<Eigen::Infinity>() <=
    kCompactGoalToleranceRad)
  {
    return std::nullopt;
  }
  return makeJointPathVelocityTask(position, compact_goal_.bounded_positions,
    compact_goal_settings_.gain, compact_goal_settings_.maximum_velocity_radps);
}

void CompactManeuver::consumeCompactPreparationResult(
  const BackgroundPathResult & plan, const CompactManeuverContext & context)
{
  const auto & state = context.state;
  const double time_sec = context.time_sec;
  const auto & scene = context.scene;
  const auto scene_revision = context.scene_revision;
  const auto & actual_feedback = context.actual_feedback;
  const auto & actual_feedback_time_sec = context.actual_feedback_time_sec;
  const auto & command_queue_state = context.command_queue_state;
  const auto * active_publication = context.active_publication;
  const auto * latest_publication = context.latest_publication;
  if (!compact_preparation_ || !pending_generation_ ||
    plan.generation != *pending_generation_ ||
    plan.generation != compact_preparation_->generation)
  {
    return;
  }
  auto & preparation = *compact_preparation_;
  pending_generation_.reset();
  compact_plan_pending_ = false;
  diagnostics_.last_plan_wall_ms = 1000.0 * plan.elapsed_time_s;
  diagnostics_.last_plan_message = plan.message;

  if (plan.scene_revision != scene_revision ||
    preparation.scene_revision != scene_revision)
  {
    // Scene tokens are opaque revisions. A result validated against an older
    // scene cannot be advanced or adopted in this one.
    planner_.cancel();
    cancelCompactProfileWork();
    compact_preparation_.reset();
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_SCENE_CHANGED";
    diagnostics_.last_plan_message = "discarded compact result from stale scene";
    return;
  }
  if (preparation.candidate_index >= preparation.candidates.size()) {
    return;
  }

  const auto reject_candidate = [this, &preparation,
      scene_revision, time_sec, plan_generation = plan.generation](
    const std::string & reason, const std::string & additional_details = {}) {
      ++diagnostics_.plans_rejected;
      diagnostics_.last_plan_message = reason;
      const auto & rejected_candidate = preparation.candidates[preparation.candidate_index];
      appendCompactEvent("candidate_rejected", time_sec, reason,
        "{\"candidate_index\":" + std::to_string(preparation.candidate_index) +
        ",\"profile_id\":\"" + rejected_candidate.profile_id +
        "\",\"goal_joints_rad\":" + jsonVector(rejected_candidate.bounded_positions) +
        ",\"plan_generation\":" + std::to_string(plan_generation) +
        (additional_details.empty() ? std::string{} : "," + additional_details) + "}");
      ++preparation.candidate_index;
      if (preparation.candidate_index < preparation.candidates.size()) {
        diagnostics_.compact_phase = "PREPARING";
        diagnostics_.tracking_phase = "FACE";
        diagnostics_.state = "FACE_COMPACT_PREPARING";
        return;
      }
      preparation.exhausted = true;
      preparation.exhausted_scene_revision = scene_revision;
      face_policy_.clearForStop();
      diagnostics_.compact_phase = "PREPARATION_EXHAUSTED";
      diagnostics_.tracking_phase = "FACE";
      diagnostics_.state = "FACE_COMPACT_PREPARATION_EXHAUSTED";
    };

  const CompactGoal candidate = preparation.candidates[preparation.candidate_index];
  const std::size_t accepted_candidate_index = preparation.candidate_index;
  const JointMotionLimits prefix_limits = activeCommandLimits();
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  const detail::CompactPlanAdmissionEnvironment environment{
    state, scene, *group_, compact_geometry_validator_.get(), collision_validator_, validation,
    compact_limits_, prefix_limits,
    actual_feedback ? &*actual_feedback : nullptr,
    command_queue_state ? &*command_queue_state : nullptr,
    active_publication, latest_publication, time_sec, parameters_.control_period_sec,
    compact_speed_scale_, compact_return_speed_scale_, compact_fold_speed_scale_,
    compact_turn_speed_scale_, compactManeuverActive()};
  const detail::CompactPreparationAdmissionInput input{
    plan, candidate, preparation.submitted_start_positions, preparation.return_region,
    preparation.entry_radius_cap_m, scene_revision, planner_goal_tolerance_rad_,
    actual_feedback && isFreshActualFeedback(
      clock_, time_sec, actual_feedback_time_sec, parameters_.state_feedback_timeout_sec),
    kCompactRadiusFloorM};
  const auto admission = detail::admitCompactPreparationPlan(input, environment);
  if (admission.metadata.phase_scales_computed) {
    diagnostics_.compact_profile_duration_sec = std::numeric_limits<double>::quiet_NaN();
    diagnostics_.compact_profile_tempo_limited = admission.metadata.profile_tempo_limited;
  }
  if (admission.disposition == detail::CompactAdmissionDisposition::kWait) {
    diagnostics_.compact_phase = "PREPARING";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARING_WAITING_FOR_FRESH_STATE";
    diagnostics_.last_plan_message = admission.reason;
    return;
  }
  if (admission.disposition != detail::CompactAdmissionDisposition::kReady ||
    !admission.admitted)
  {
    reject_candidate(admission.reason, admission.additional_details);
    return;
  }

  const auto & ready = *admission.admitted;
  const double entry_radius_cap = ready.entry_radius_cap_m;
  const double entry_max_pitch = ready.entry_max_pitch_rad;
  const double entry_max_roll = ready.entry_max_roll_rad;
  const auto & published_prefix = ready.published_prefix;
  const auto & compact_start = ready.compact_start;
  const auto & attachment = ready.attachment;
  const auto & bridge_validation = ready.bridge_validation;
  const auto & attached_path = ready.path;
  const auto & phase_scales = ready.phase_scales;
  const auto & return_profile = ready.return_profile;
  const auto & metrics = *plan.validation_metrics;
  const Eigen::VectorXd & attachment_position = attachment.position;

  CompactProfileWorkRequest request;
  request.generation = diagnostics_.compact_generation;
  request.scene_revision = scene_revision;
  request.source_route = attached_path;
  request.segment_speed_scales = phase_scales;
  request.start_state = compact_start;
  request.published_prefix = published_prefix.samples;
  request.prefix_limits = prefix_limits;
  request.return_profile = return_profile;
  request.requested_speed_scale = compact_speed_scale_;
  request.return_speed_scale = compact_speed_scale_ * compact_return_speed_scale_;
  if (submitCompactProfileWork(
      std::move(request), state, scene, entry_radius_cap, entry_radius_cap,
      preparation.return_region, time_sec) == 0U)
  {
    reject_candidate("compact candidate could not submit torque validation");
    return;
  }
  compact_path_execution_.install(
    detail::compactProfileForPublishedPrefix(
      published_prefix.samples, parameters_.control_period_sec), prefix_limits);

  CompactGoal committed_goal = candidate;
  committed_goal.path_requirements.maximum_assembly_radius_m = entry_radius_cap;
  committed_goal.path_requirements.base_link_name = base_link_->getName();
  committed_goal.path_requirements.screen_link_name = command_link_->getName();
  compact_goal_ = std::move(committed_goal);
  compact_preparation_.reset();
  compact_commit_initialized_ = true;
  face_roll_guard_disarm_requested_ = true;
  compact_radius_floor_latched_ = false;
  compact_plan_pending_ = false;
  compact_path_adopted_ = true;
  compact_goal_reached_ = false;
  compact_path_endpoint_reached_ = false;
  compact_return_path_pending_ = false;
  compact_goal_event_emitted_ = false;
  compact_path_started_event_emitted_ = false;
  diagnostics_.compact_goal_reached = 0.0;
  diagnostics_.compact_entry_radius_cap_m = entry_radius_cap;
  diagnostics_.compact_effective_radius_cap_m = entry_radius_cap;
  diagnostics_.compact_pitch_cap_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.compact_radius_floor_latched = false;
  diagnostics_.compact_actual_settled = false;
  diagnostics_.planned_path_max_radius_m = std::max(
    attachment.maximum_radius_m, bridge_validation.maximum_radius);
  diagnostics_.planned_path_max_roll_error_rad = std::max(
    attachment.maximum_roll_error_rad, bridge_validation.maximum_abs_roll);
  diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.compact_reason = diagnostics_.compact_reason.empty() ?
    "j1_working_boundary" : diagnostics_.compact_reason;
  diagnostics_.compact_phase = "PATH";
  diagnostics_.tracking_phase = "COMPACT";
  diagnostics_.state = "COMPACT_PATH";
  diagnostics_.last_plan_message = "compact path committed: " + plan.message;
  active_scene_revision_ = scene_revision;
  ++diagnostics_.plans_accepted;

  const std::string details =
    "{\"goal_profile\":\"" + compact_goal_.profile_id +
    "\",\"goal_joints_rad\":" + jsonVector(compact_goal_.bounded_positions) +
    ",\"staged_route_enabled\":" +
    (compact_goal_.path_requirements.staged_route ? "true" : "false") +
    ",\"entry_radius_cap_m\":" + jsonNumber(entry_radius_cap) +
    ",\"entry_envelope_max_abs_pitch_rad\":" + jsonNumber(entry_max_pitch) +
    ",\"entry_envelope_max_abs_roll_rad\":" + jsonNumber(entry_max_roll) +
    ",\"candidate_index\":" + std::to_string(accepted_candidate_index) +
    ",\"attachment\":{" +
    "\"used_certificate\":" + (attachment.used_certificate ? "true" : "false") +
    ",\"sample_index\":" + (attachment.used_certificate ?
    std::to_string(attachment.sample_index) : "null") +
    ",\"segment_index\":" + (attachment.used_certificate ?
    std::to_string(attachment.segment_index) : "null") +
    ",\"segment_fraction\":" + (attachment.used_certificate ?
    jsonNumber(attachment.segment_fraction) : "null") +
    ",\"positions_rad\":" + jsonVector(attachment_position) +
    ",\"actual_bridge_distance_rad\":" + jsonNumber(ready.actual_to_attachment_rad) +
    ",\"queue_bridge_distance_rad\":" + jsonNumber(ready.queue_to_attachment_rad) +
    ",\"suffix_maximum_radius_m\":" + jsonNumber(attachment.maximum_radius_m) +
    ",\"suffix_maximum_abs_pitch_rad\":" +
    jsonNumber(attachment.maximum_abs_pitch_rad) +
    ",\"suffix_maximum_roll_error_rad\":" +
    jsonNumber(attachment.maximum_roll_error_rad) +
    "},\"worker_full_path_metrics\":{" +
    "\"maximum_assembly_radius_m\":" + jsonNumber(metrics.maximum_assembly_radius_m) +
    ",\"maximum_abs_screen_pitch_rad\":" +
    jsonNumber(metrics.maximum_abs_screen_pitch_rad) +
    ",\"maximum_roll_error_rad\":" + jsonNumber(metrics.maximum_roll_error_rad) +
    "},\"follower_seed\":\"command_queue_state\",\"follower_seed_positions_rad\":" +
    jsonVector(command_queue_state->position) + "}";
  appendCompactEvent("triggered", time_sec, diagnostics_.compact_reason, details);
}

void CompactManeuver::adoptCommittedCompactPlan(
  const BackgroundPathResult & plan, const CompactManeuverContext & context)
{
  const auto & state = context.state;
  const auto & command_queue_state = context.command_queue_state;
  const auto & scene = context.scene;
  const double time_sec = context.time_sec;
  const auto scene_revision = context.scene_revision;
  const auto * active_publication = context.active_publication;
  const auto * latest_publication = context.latest_publication;
  if (!pending_generation_ || plan.generation != *pending_generation_) {
    return;
  }
  pending_generation_.reset();
  compact_prepared_return_profile_.reset();
  compact_return_endpoint_.resize(0);
  diagnostics_.last_plan_wall_ms = 1000.0 * plan.elapsed_time_s;
  diagnostics_.last_plan_message = plan.message;

  const JointMotionLimits prefix_limits = activeCommandLimits();
  const detail::CompactManeuverValidation validation(
    robot_model_, group_, command_link_, limits_, collision_validator_,
    compact_geometry_validator_, static_cast<std::size_t>(parameters_.segment_validation_substeps));
  const detail::CompactPlanAdmissionEnvironment environment{
    state, scene, *group_, compact_geometry_validator_.get(), collision_validator_, validation,
    compact_limits_, prefix_limits, nullptr,
    command_queue_state ? &*command_queue_state : nullptr,
    active_publication, latest_publication, time_sec, parameters_.control_period_sec,
    compact_speed_scale_, compact_return_speed_scale_, compact_fold_speed_scale_,
    compact_turn_speed_scale_, compactManeuverActive()};
  const detail::CompactCommittedAdmissionInput input{
    plan, compact_goal_, scene_revision, effectiveCompactRadiusCap(),
    compact_goal_.path_requirements.maximum_assembly_radius_m};
  const auto admission = detail::admitCommittedCompactPlan(input, environment);
  const auto & prefix_failure = admission.prefix_failure;
  bool accepted = admission.disposition == detail::CompactAdmissionDisposition::kReady &&
    admission.admitted.has_value();
  if (accepted) {
    const auto & ready = *admission.admitted;
    CompactProfileWorkRequest request;
    request.generation = diagnostics_.compact_generation;
    request.scene_revision = scene_revision;
    request.source_route = ready.path;
    request.segment_speed_scales = ready.phase_scales;
    request.start_state = ready.compact_start;
    request.published_prefix = ready.published_prefix.samples;
    request.prefix_limits = prefix_limits;
    request.return_profile = ready.return_profile;
    request.requested_speed_scale = compact_speed_scale_;
    request.return_speed_scale = compact_speed_scale_ * compact_return_speed_scale_;
    accepted = submitCompactProfileWork(
      std::move(request), state, scene, effectiveCompactRadiusCap(),
      compact_goal_.path_requirements.maximum_assembly_radius_m,
      compact_goal_.return_region, time_sec) != 0U;
    if (accepted) {
      ++diagnostics_.plans_accepted;
      compact_plan_pending_ = false;
      compact_path_adopted_ = true;
      diagnostics_.compact_profile_duration_sec = std::numeric_limits<double>::quiet_NaN();
      diagnostics_.compact_profile_tempo_limited = std::any_of(
        ready.phase_scales.begin(), ready.phase_scales.end(), [this](const double phase_scale) {
          return compact_speed_scale_ * phase_scale > 1.0;
        });
      compact_path_execution_.install(
        detail::compactProfileForPublishedPrefix(
          ready.published_prefix.samples, parameters_.control_period_sec), prefix_limits);
      face_roll_guard_disarm_requested_ = true;
      active_scene_revision_ = scene_revision;
      diagnostics_.compact_phase = "PATH";
      diagnostics_.planned_path_max_radius_m =
        plan.validation_metrics->maximum_assembly_radius_m;
      diagnostics_.planned_path_max_roll_error_rad =
        plan.validation_metrics->maximum_roll_error_rad;
      diagnostics_.planned_path_max_abs_pitch_rad =
        plan.validation_metrics->maximum_abs_screen_pitch_rad;
      diagnostics_.last_plan_message = "compact path accepted: " + plan.message;
      return;
    }
  }
  ++diagnostics_.plans_rejected;
  compact_plan_pending_ = false;
  compact_path_adopted_ = false;
  compact_path_execution_.clear();
  compact_prepared_return_profile_.reset();
  compact_return_endpoint_.resize(0);
  compact_return_profile_active_ = false;
  diagnostics_.planned_path_max_radius_m = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_roll_error_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.planned_path_max_abs_pitch_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.compact_phase = "REPLANNING";
  diagnostics_.last_plan_message = "compact path rejected: " +
    (prefix_failure.empty() ? plan.message : prefix_failure);
}

bool CompactManeuver::handleCompactPreparation(
  const CompactManeuverContext & context, CompactManeuverOutcome & outcome)
{
  const auto & state = context.state;
  const auto & target = context.target;
  const auto scene_revision = context.scene_revision;
  const bool fresh_actual_state = context.fresh_actual_state;
  const auto & actual_feedback = context.actual_feedback;
  const auto & command_queue_state = context.command_queue_state;
  auto & result = outcome.result;
  const bool has_face = target && target->mode == msg::TrackingTarget::FACE;
  const bool explicit_hold = target && target->mode == msg::TrackingTarget::HOLD &&
    !target->allow_recovery;
  const bool recoverable_idle = target && target->allow_recovery &&
    (target->mode == msg::TrackingTarget::HOLD ||
    target->mode == msg::TrackingTarget::SEARCH || target->mode == msg::TrackingTarget::REST);
  const bool continue_without_face = recoverable_idle && !has_face;
  const auto planner_status = [this]() {
      return "busy=" + std::to_string(planner_.isBusy()) + ", pending_generation=" +
             (pending_generation_ ? std::to_string(*pending_generation_) : "none");
    };
  if ((!has_face && !continue_without_face) || explicit_hold) {
    planner_.cancel();
    cancelCompactProfileWork();
    pending_generation_.reset();
    compact_preparation_.reset();
    compact_plan_pending_ = false;
    face_policy_.clearForStop();
    diagnostics_.compact_phase = "PREPARATION_CANCELLED";
    diagnostics_.tracking_phase = "FACE";
    diagnostics_.state = "FACE_COMPACT_PREPARATION_CANCELLED_TARGET_LOST";
    return true;
  }

  if (compact_preparation_ && compact_preparation_->scene_revision != scene_revision) {
    planner_.cancel();
    cancelCompactProfileWork();
    pending_generation_.reset();
    compact_plan_pending_ = false;
    compact_preparation_.reset();
    diagnostics_.last_plan_message =
      "restarting compact preparation for changed scene revision";
  }
  // Keep the fresh actual/queue inputs required by FACE guard recovery before
  // submitting or adopting a compact route.
  if (context.upright_roll_guard_armed) {
    const auto valid_joint_state = [this](const JointMotionState & joint_state) {
        const auto expected = static_cast<Eigen::Index>(group_->getVariableCount());
        return joint_state.position.size() == expected &&
               joint_state.velocity.size() == expected &&
               joint_state.acceleration.size() == expected &&
               joint_state.position.allFinite() && joint_state.velocity.allFinite() &&
               joint_state.acceleration.allFinite();
      };
    if (!fresh_actual_state || !actual_feedback || !command_queue_state ||
      !valid_joint_state(*actual_feedback) || !valid_joint_state(*command_queue_state))
    {
      diagnostics_.compact_phase = "PREPARING_WAIT";
      diagnostics_.state = "COMPACT_PREPARATION_MOTION_STATE_UNAVAILABLE";
      diagnostics_.last_plan_message =
        "compact preparation requires fresh actual/queue motion state";
      result.motion_reference_support_unavailable = true;
      outcome.face_directive = CompactFaceDirective::kGuardOnly;
      return true;
    }
  }

  requestCompactManeuver(context, diagnostics_.compact_reason.empty() ?
    "j1_working_boundary" : diagnostics_.compact_reason);
  if (compact_preparation_ && pending_generation_ && fresh_actual_state &&
    actual_feedback && command_queue_state)
  {
    if (auto plan = planner_.takeResult()) {
      consumeCompactPreparationResult(*plan, context);
    }
  }
  if (!compact_commit_initialized_ && compact_preparation_ &&
    !compact_preparation_->exhausted)
  {
    const auto & message = diagnostics_.last_plan_message;
    const bool generic_message = message == "compact path preparation started" ||
      message.rfind("compact preparation waiting:", 0U) == 0U;
    const bool planner_waiting = pending_generation_ || planner_.isBusy();
    if (generic_message && planner_waiting) {
      diagnostics_.last_plan_message = "compact preparation waiting: " + planner_status();
    }
    if (generic_message && !planner_waiting &&
      (!fresh_actual_state || !actual_feedback || !command_queue_state))
    {
      diagnostics_.last_plan_message =
        "compact preparation waiting: fresh actual/queue unavailable; " +
        planner_status();
    }
  }
  if (!compact_commit_initialized_) {
    if (compact_preparation_ && !compact_preparation_->exhausted &&
      !pending_generation_ && !planner_.isBusy() && fresh_actual_state &&
      command_queue_state)
    {
      moveit::core::RobotState start(state);
      start.setJointGroupPositions(group_, actual_feedback->position);
      start.update(true);
      const CompactManeuverContext preparation_context{
        start, context.scene, context.target, context.target_motion,
        context.j1_boundary_assessment,
        context.time_sec, context.scene_revision, context.position, context.settled,
        context.fresh_actual_state, context.actual_feedback,
        context.actual_feedback_time_sec, context.command_queue_state,
        context.active_publication, context.latest_publication,
        context.upright_roll_guard_armed};
      const auto submit_outcome = submitCompactPreparationPlan(preparation_context);
      if (submit_outcome == CompactPreparationSubmitOutcome::kEntryEnvelopeUnsafe) {
        diagnostics_.state = "COMPACT_PREPARING_WAITING_FOR_VALID_ENTRY_ENVELOPE";
        outcome.face_directive = CompactFaceDirective::kGuardOnly;
        return true;
      }
    }
    outcome.face_directive = has_face ?
      (context.upright_roll_guard_armed ? CompactFaceDirective::kBuildFaceAndGuard :
      CompactFaceDirective::kBuildFace) : (context.upright_roll_guard_armed ?
      CompactFaceDirective::kGuardOnly : CompactFaceDirective::kNone);
    diagnostics_.tracking_phase = "FACE";
    if (!has_face) {
      diagnostics_.state = "FACE_COMPACT_PREPARING_WAITING_FOR_FACE";
    } else if (!compact_preparation_ || compact_preparation_->exhausted) {
      diagnostics_.compact_phase = compact_preparation_ ?
        "PREPARATION_EXHAUSTED" : "COMMIT_WAIT";
    } else {
      diagnostics_.compact_phase = "PREPARING";
    }
    if (has_face) {
      if (!compact_preparation_) {
        diagnostics_.state = "FACE_COMPACT_PREPARING_WAIT";
      } else if (!compact_preparation_->exhausted) {
        diagnostics_.state = pending_generation_ || planner_.isBusy() ?
          "FACE_COMPACT_PREPARING" : "FACE_COMPACT_PREPARING_WAIT";
      }
    }
    return true;
  }
  return false;
}

CompactManeuverOutcome CompactManeuver::update(const CompactManeuverContext & context)
{
  CompactManeuverOutcome outcome;
  const auto & state = context.state;
  const auto & target = context.target;
  const double time_sec = context.time_sec;
  const std::uint64_t scene_revision = context.scene_revision;
  Eigen::VectorXd position = context.position;
  const bool fresh_actual_state = context.fresh_actual_state;
  const auto & actual_feedback = context.actual_feedback;
  const auto & command_queue_state = context.command_queue_state;
  const bool has_face = target && target->mode == msg::TrackingTarget::FACE;
  const auto mode = target ? target->mode : msg::TrackingTarget::HOLD;
  const bool healthy_face_loss = target && target->allow_recovery && !has_face &&
    (mode == msg::TrackingTarget::HOLD || mode == msg::TrackingTarget::REST ||
    mode == msg::TrackingTarget::SEARCH);
  auto & result = outcome.result;
  const auto finish = [this, &outcome]() {
      outcome.disarm_face_roll_guard = takeFaceRollGuardDisarmRequest();
      return std::move(outcome);
    };
  recordCompactSettledEvent(context, compact_goal_reached_ && fresh_actual_state &&
    compactSettled(*actual_feedback, command_queue_state, compact_goal_.bounded_positions));
  if (face_policy_.phase() == FaceTrackingPhase::kCompactManeuver &&
    compact_commit_initialized_ && compact_goal_.bounded_positions.size() == position.size())
  {
    if (fresh_actual_state) {
      diagnostics_.compact_actual_goal_error_rad =
        (actual_feedback->position - compact_goal_.bounded_positions).lpNorm<Eigen::Infinity>();
      diagnostics_.compact_actual_speed_max_rad_s =
        actual_feedback->velocity.lpNorm<Eigen::Infinity>();
    } else {
      diagnostics_.compact_actual_goal_error_rad.reset();
      diagnostics_.compact_actual_speed_max_rad_s.reset();
    }
    if (command_queue_state && command_queue_state->position.size() == position.size() &&
      command_queue_state->velocity.size() == position.size() &&
      command_queue_state->acceleration.size() == position.size() &&
      command_queue_state->position.allFinite() && command_queue_state->velocity.allFinite() &&
      command_queue_state->acceleration.allFinite())
    {
      diagnostics_.compact_queue_tail_goal_error_rad =
        (command_queue_state->position - compact_goal_.bounded_positions).
        lpNorm<Eigen::Infinity>();
      diagnostics_.compact_queue_tail_speed_max_rad_s =
        command_queue_state->velocity.lpNorm<Eigen::Infinity>();
      diagnostics_.compact_queue_tail_acceleration_max_rad_s2 =
        command_queue_state->acceleration.lpNorm<Eigen::Infinity>();
    } else {
      diagnostics_.compact_queue_tail_goal_error_rad.reset();
      diagnostics_.compact_queue_tail_speed_max_rad_s.reset();
      diagnostics_.compact_queue_tail_acceleration_max_rad_s2.reset();
    }
  }
  if (fresh_actual_state && actual_feedback && command_queue_state) {
    maybeLatchCompactRadiusCap(context);
  }
  diagnostics_.compact_radius_floor_latched = compact_radius_floor_latched_;
  diagnostics_.compact_effective_radius_cap_m = compact_commit_initialized_ ?
    effectiveCompactRadiusCap() : std::numeric_limits<double>::quiet_NaN();
  if (face_policy_.phase() != FaceTrackingPhase::kCompactManeuver) {
    outcome.disarm_face_roll_guard = takeFaceRollGuardDisarmRequest();
    return outcome;
  }
  outcome.handled = true;
  const bool explicit_hold = mode == msg::TrackingTarget::HOLD &&
    (!target || !target->allow_recovery);
  if (!compact_commit_initialized_ && handleCompactPreparation(context, outcome)) {
    outcome.handled = true;
    return finish();
  }
  diagnostics_.tracking_phase = "COMPACT";
  bool compact_radius_constraints_computed = false;
  const auto compute_compact_radius_constraints = [&](const bool refresh = false) {
      if ((!refresh && compact_radius_constraints_computed) || !compact_commit_initialized_) {
        return;
      }
      compact_radius_constraints_computed = true;
      if (!command_queue_state || !appendCompactPathConstraints(
          state, *command_queue_state, result))
      {
        result.motion_reference_support_unavailable = true;
      }
    };
  if (explicit_hold || !target ||
    (!has_face && !healthy_face_loss))
  {
    compute_compact_radius_constraints();
    diagnostics_.state = "COMPACT_COMMITTED_WAITING_FOR_HEALTHY_COMMAND";
    return finish();
  }
  if (active_scene_revision_ && *active_scene_revision_ != scene_revision &&
    !pending_generation_)
  {
    cancelCompactProfileWork();
    if (compact_return_profile_active_) {
      compact_path_execution_.clear();
      compact_prepared_return_profile_.reset();
      compact_path_endpoint_reached_ = false;
      compact_return_profile_active_ = false;
      compact_return_path_pending_ = true;
    }
    follower_.reset();
    compact_path_adopted_ = false;
    compact_plan_pending_ = false;
    compact_prepared_return_profile_.reset();
    if (!compact_return_profile_active_) {
      compact_return_endpoint_.resize(0);
    }
  }
  if (!compact_return_profile_active_) {
    if (auto plan = planner_.takeResult()) {
      adoptCommittedCompactPlan(*plan, context);
    }
  }
  // compactTask owns any goal-position correction after a timed path endpoint;
  // do not submit a replacement plan while that endpoint is settling.
  if (!compact_return_profile_active_ && !compact_path_adopted_ &&
    !compact_path_endpoint_reached_ &&
    !pending_generation_ && !planner_.isBusy() &&
    fresh_actual_state && motionSettled(*actual_feedback, command_queue_state))
  {
    moveit::core::RobotState start(state);
    start.setJointGroupPositions(group_, actual_feedback->position);
    start.update(true);
    position = actual_feedback->position;
    const CompactManeuverContext start_context{
      start, context.scene, context.target, context.target_motion,
      context.j1_boundary_assessment,
      context.time_sec, context.scene_revision, position, context.settled,
      context.fresh_actual_state, context.actual_feedback,
      context.actual_feedback_time_sec, context.command_queue_state,
      context.active_publication, context.latest_publication,
      context.upright_roll_guard_armed};
    submitCompactPlan(start_context);
  }
  const CompactManeuverContext task_context{
    state, context.scene, context.target, context.target_motion,
    context.j1_boundary_assessment,
    context.time_sec, context.scene_revision, position, context.settled,
    context.fresh_actual_state, context.actual_feedback,
    context.actual_feedback_time_sec, context.command_queue_state,
    context.active_publication, context.latest_publication,
    context.upright_roll_guard_armed};
  result.task = compactTask(task_context, result);
  // A timed PATH sample is already paired with its published queue prefix.
  // Its exact candidate and generated braking suffix are checked against the
  // compact geometry caps; worst-case outward-stop rows would reject that
  // known trajectory without adding coverage. FACE/RETURN and QP phases keep
  // the rows before the solver is called.
  if (!result.compact_profile_command) {
    compute_compact_radius_constraints();
  }
  if (serviceCompactReturnProfileStage(context, result)) {
    return finish();
  }
  if (compact_path_endpoint_reached_ && fresh_actual_state &&
    compactGoalPositionReached(*actual_feedback, compact_goal_.bounded_positions))
  {
    compact_goal_reached_ = true;
    diagnostics_.compact_goal_reached = 1.0;
    if (!compact_goal_event_emitted_) {
      appendCompactEvent("goal_reached", time_sec, diagnostics_.compact_reason);
      compact_goal_event_emitted_ = true;
    }
    const bool compact_settled = compactSettled(
      *actual_feedback, command_queue_state, compact_goal_.bounded_positions);
    recordCompactSettledEvent(context, compact_settled);
    if (compact_settled) {
      const bool start_queue_matches_return = compact_prepared_return_profile_ &&
        compact_prepared_return_profile_->samples.size() > 1U && command_queue_state &&
        command_queue_state->position.size() ==
        compact_prepared_return_profile_->samples.front().position.size() &&
        command_queue_state->velocity.size() ==
        compact_prepared_return_profile_->samples.front().velocity.size() &&
        command_queue_state->acceleration.size() ==
        compact_prepared_return_profile_->samples.front().acceleration.size() &&
        (command_queue_state->position -
        compact_prepared_return_profile_->samples.front().position).
        lpNorm<Eigen::Infinity>() <= kCompactProfilePositionToleranceRad &&
        (command_queue_state->velocity -
        compact_prepared_return_profile_->samples.front().velocity).
        lpNorm<Eigen::Infinity>() <= 1.0e-6 &&
        (command_queue_state->acceleration -
        compact_prepared_return_profile_->samples.front().acceleration).
        lpNorm<Eigen::Infinity>() <= 1.0e-5 && active_scene_revision_ &&
        *active_scene_revision_ == scene_revision;
      if (start_queue_matches_return) {
        compact_path_execution_.install(
            std::move(*compact_prepared_return_profile_), compact_limits_);
        compact_prepared_return_profile_.reset();
        compact_return_endpoint_ = compact_path_execution_.profile()->samples.back().position;
        compact_return_profile_active_ = true;
        compact_return_path_pending_ = true;
        compact_path_endpoint_reached_ = false;
        compact_path_adopted_ = true;
        diagnostics_.compact_actual_settled = false;
        diagnostics_.compact_profile_duration_sec =
          compact_path_execution_.profile()->duration_sec;
        diagnostics_.compact_profile_tempo_limited =
          compact_speed_scale_ * compact_return_speed_scale_ > 1.0;
        diagnostics_.compact_phase = "RETURN_PATH";
        appendCompactEvent(
            "return_profile_started", time_sec, "checked_staged_return",
            "{\"goal_positions_rad\":" + jsonVector(compact_return_endpoint_) +
            ",\"profile_duration_sec\":" +
            jsonNumber(compact_path_execution_.profile()->duration_sec) +
            ",\"speed_scale\":" + jsonNumber(
              compact_speed_scale_ * compact_return_speed_scale_) + "}");
        result.task.reset();
        result.compact_radius_constraints.clear();
        result.compact_roll_constraints.clear();
        result.compact_pitch_constraints.clear();
        const CompactManeuverContext return_context{
          state, context.scene, context.target, context.target_motion,
          context.j1_boundary_assessment,
          context.time_sec, context.scene_revision, position, context.settled,
          context.fresh_actual_state, context.actual_feedback,
          context.actual_feedback_time_sec, context.command_queue_state,
          context.active_publication, context.latest_publication,
          context.upright_roll_guard_armed};
        result.task = compactTask(return_context, result);
        result.follows_path = result.compact_profile_command;
        if (!result.compact_profile_command) {
          diagnostics_.state = "COMPACT_RETURN_BRAKING";
        } else {
          diagnostics_.tracking_phase = "COMPACT";
          diagnostics_.state = "COMPACT_RETURN_PATH";
        }
        return finish();
      }
      if (!start_queue_matches_return) {
        compact_return_path_pending_ = true;
        compact_path_endpoint_reached_ = false;
        compact_path_adopted_ = false;
        result.task.reset();
        result.follows_path = false;
        diagnostics_.compact_phase = "RETURN_PATH_REPLANNING";
        diagnostics_.state = "COMPACT_RETURN_PATH_REPLANNING";
        diagnostics_.last_plan_message =
          "mandatory return profile is unavailable or its queue start changed";
        return finish();
      }
    }
  }
  result.follows_path = compact_path_adopted_ &&
    (!compact_return_path_pending_ || compact_return_profile_active_) &&
    (result.task.has_value() || result.compact_profile_command);
  if (result.compact_profile_command) {
    diagnostics_.state = compact_return_profile_active_ ?
      "COMPACT_RETURN_PATH" : "COMPACT_PATH";
    return finish();
  }
  if (result.task) {
    if (compact_return_path_pending_) {
      return finish();
    }
    diagnostics_.state = !compact_path_adopted_ ? "COMPACT_PLANNING" :
      (compact_path_endpoint_reached_ ? "COMPACT_SETTLING" : "COMPACT_PATH");
    return finish();
  }
  if (compact_path_endpoint_reached_) {
    diagnostics_.state = "COMPACT_SETTLING";
    return finish();
  }
  diagnostics_.state = "COMPACT_PLANNING";
  return finish();
}

void CompactManeuver::finishCompactReturnProfileStage(
  const CompactManeuverContext & context, MotionReferenceResult & result)
{
  const auto & state = context.state;
  const double time_sec = context.time_sec;
  const auto & actual_feedback = context.actual_feedback;
  const auto & actual_feedback_time_sec = context.actual_feedback_time_sec;
  const auto & command_queue_state = context.command_queue_state;
  const bool fresh_actual_state = context.fresh_actual_state;
  const bool queue_at_return_endpoint = compact_path_endpoint_reached_ &&
    command_queue_state &&
    compact_goal_.return_region && validateCompactReturnRegion(
      state, command_queue_state->position, *compact_goal_.return_region);
  const bool actual_at_return_endpoint = compact_path_endpoint_reached_ &&
    fresh_actual_state && actual_feedback &&
    compact_goal_.return_region && validateCompactReturnRegion(
      state, actual_feedback->position, *compact_goal_.return_region);
  const bool feedback_matches_queue = fresh_actual_state && actual_feedback &&
    command_queue_state && actual_feedback->position.size() ==
    command_queue_state->position.size() &&
    (actual_feedback->position - command_queue_state->position).
    lpNorm<Eigen::Infinity>() <= kCompactGoalToleranceRad;
  const bool return_settled = fresh_actual_state && actual_feedback &&
    compact_path_endpoint_reached_ && motionSettled(*actual_feedback, command_queue_state) &&
    feedback_matches_queue && queue_at_return_endpoint && actual_at_return_endpoint;
  const bool stopped_without_reaching_region = compact_path_endpoint_reached_ &&
    fresh_actual_state && actual_feedback && motionSettled(*actual_feedback, command_queue_state) &&
    feedback_matches_queue && (!queue_at_return_endpoint || !actual_at_return_endpoint);
  if (stopped_without_reaching_region) {
    compact_return_profile_active_ = false;
    compact_path_endpoint_reached_ = false;
    compact_path_adopted_ = false;
    compact_path_execution_.clear();
    compact_return_endpoint_.resize(0);
    diagnostics_.compact_phase = "RETURN_PATH_REPLANNING";
    diagnostics_.state = "COMPACT_RETURN_PATH_REPLANNING";
    diagnostics_.last_plan_message =
      "return profile stopped outside the frozen workspace; replanning return";
    result.follows_path = false;
    return;
  }
  if (!return_settled) {
    result.follows_path = false;
    diagnostics_.compact_phase = compact_path_endpoint_reached_ ?
      "RETURN_SETTLING" : "RETURN_BRAKING";
    diagnostics_.state = compact_path_endpoint_reached_ ?
      "COMPACT_RETURN_SETTLING" : "COMPACT_RETURN_BRAKING";
    return;
  }

  const std::int64_t stamp_ns = actual_feedback_time_sec ?
    toNanoseconds(*actual_feedback_time_sec) : 0;
  const std::string return_details =
    "{\"actual_feedback_stamp_ns\":" + std::to_string(stamp_ns) +
    ",\"actual_joint_positions_rad\":" + jsonVector(actual_feedback->position) + "}";
  appendCompactEvent("return_workspace_reached", time_sec,
    "actual_and_queue_inside_frozen_region", return_details);
  appendCompactEvent("completed", time_sec, diagnostics_.compact_reason, return_details);
  compact_return_profile_active_ = false;
  compact_prepared_return_profile_.reset();
  compact_return_endpoint_.resize(0);
  compact_path_execution_.clear();
  compact_path_endpoint_reached_ = false;
  compact_path_adopted_ = false;
  static_cast<void>(face_policy_.update(FaceTrackingPolicyInput{false, true}));
  compact_commit_initialized_ = false;
  compact_radius_floor_latched_ = false;
  compact_goal_.return_region.reset();
  compact_goal_reached_ = false;
  compact_goal_event_emitted_ = false;
  compact_plan_pending_ = false;
  compact_return_path_pending_ = false;
  diagnostics_.compact_radius_floor_latched = false;
  diagnostics_.compact_effective_radius_cap_m = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.compact_pitch_cap_rad = std::numeric_limits<double>::quiet_NaN();
  diagnostics_.compact_actual_settled = false;
  diagnostics_.compact_phase = "COMPLETE";
  diagnostics_.tracking_phase = "FACE";
  diagnostics_.state = "COMPACT_COMPLETE";
  diagnostics_.compact_goal_reached = 0.0;
  result.task.reset();
  result.compact_profile_command = false;
  result.compact_profile_velocity.reset();
  result.follows_path = false;
  result.compact_radius_constraints.clear();
  result.compact_roll_constraints.clear();
  result.compact_pitch_constraints.clear();
  active_scene_revision_.reset();
}

bool CompactManeuver::serviceCompactReturnProfileStage(
  const CompactManeuverContext & context, MotionReferenceResult & result)
{
  if (!compact_return_profile_active_ || result.compact_profile_command) {
    return false;
  }
  if (pending_compact_profile_work_) {
    diagnostics_.compact_phase = "RETURN_PATH_PREPARING_TORQUE_PROFILE";
    diagnostics_.state = "COMPACT_RETURN_PATH_PREPARING";
    return true;
  }
  finishCompactReturnProfileStage(context, result);
  return true;
}

}  // namespace face_tracking_arm::control
