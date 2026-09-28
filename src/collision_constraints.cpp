// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/collision_constraints.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/attached_body.hpp>
#include <moveit/robot_state/robot_state.hpp>

namespace face_tracking_arm::control
{
namespace
{

using collision_detection::BodyType;
using collision_detection::DistanceResult;
using collision_detection::DistanceResultsData;
namespace BodyTypes = collision_detection::BodyTypes;

constexpr double kComparisonTolerance = 1.0e-8;

[[nodiscard]] bool isFinite(const Eigen::VectorXd & vector)
{
  return vector.array().isFinite().all();
}

[[nodiscard]] bool isFinite(const Eigen::Vector3d & vector)
{
  return vector.array().isFinite().all();
}

[[nodiscard]] int bodyTypeRank(const BodyType type)
{
  switch (type) {
    case BodyTypes::ROBOT_LINK:
      return 0;
    case BodyTypes::ROBOT_ATTACHED:
      return 1;
    case BodyTypes::WORLD_OBJECT:
      return 2;
  }
  return 3;
}

[[nodiscard]] std::string bodyToken(const BodyType type, const std::string & name)
{
  switch (type) {
    case BodyTypes::ROBOT_LINK:
      return "robot:" + name;
    case BodyTypes::ROBOT_ATTACHED:
      return "attached:" + name;
    case BodyTypes::WORLD_OBJECT:
      return "world:" + name;
  }
  return "unknown:" + name;
}

struct DistanceCandidate
{
  CollisionPairKind kind{CollisionPairKind::kSelf};
  BodyType body_types[2]{BodyTypes::WORLD_OBJECT, BodyTypes::WORLD_OBJECT};
  std::string body_names[2];
  Eigen::Vector3d nearest_points[2]{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::Zero()};
  double distance_m{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] bool canonicalLess(
  const BodyType left_type, const std::string & left_name,
  const BodyType right_type, const std::string & right_name)
{
  return std::make_tuple(bodyTypeRank(left_type), left_name) <
         std::make_tuple(bodyTypeRank(right_type), right_name);
}

[[nodiscard]] DistanceCandidate canonicalize(
  const DistanceResultsData & data, const CollisionPairKind kind)
{
  DistanceCandidate candidate;
  candidate.kind = kind;
  candidate.body_types[0] = data.body_types[0];
  candidate.body_types[1] = data.body_types[1];
  candidate.body_names[0] = data.link_names[0];
  candidate.body_names[1] = data.link_names[1];
  candidate.nearest_points[0] = data.nearest_points[0];
  candidate.nearest_points[1] = data.nearest_points[1];
  candidate.normal = data.normal;
  candidate.distance_m = data.distance;

  if (!canonicalLess(
      candidate.body_types[0], candidate.body_names[0],
      candidate.body_types[1], candidate.body_names[1]))
  {
    std::swap(candidate.body_types[0], candidate.body_types[1]);
    std::swap(candidate.body_names[0], candidate.body_names[1]);
    std::swap(candidate.nearest_points[0], candidate.nearest_points[1]);
    candidate.normal = -candidate.normal;
  }
  return candidate;
}

[[nodiscard]] std::string candidateKey(const DistanceCandidate & candidate)
{
  return bodyToken(candidate.body_types[0], candidate.body_names[0]) + "|" +
         bodyToken(candidate.body_types[1], candidate.body_names[1]);
}

[[nodiscard]] std::string readablePair(const DistanceCandidate & candidate)
{
  return candidate.body_names[0] + " <-> " + candidate.body_names[1];
}

[[nodiscard]] bool candidateLess(
  const DistanceCandidate & left, const DistanceCandidate & right)
{
  if (left.distance_m != right.distance_m) {
    return left.distance_m < right.distance_m;
  }
  const auto left_key = candidateKey(left);
  const auto right_key = candidateKey(right);
  if (left_key != right_key) {
    return left_key < right_key;
  }
  for (int body = 0; body < 2; ++body) {
    for (int axis = 0; axis < 3; ++axis) {
      if (left.nearest_points[body][axis] != right.nearest_points[body][axis]) {
        return left.nearest_points[body][axis] < right.nearest_points[body][axis];
      }
    }
  }
  return false;
}

void appendDistanceCandidates(
  const DistanceResult & distance_result,
  const CollisionPairKind kind,
  std::vector<DistanceCandidate> & candidates)
{
  for (const auto & [unused_pair, contacts] : distance_result.distances) {
    (void)unused_pair;
    for (const auto & contact : contacts) {
      candidates.push_back(canonicalize(contact, kind));
    }
  }
}

[[nodiscard]] bool isNamedPair(
  const DistanceCandidate & candidate,
  const std::string & first,
  const std::string & second)
{
  return
    (candidate.body_names[0] == first && candidate.body_names[1] == second) ||
    (candidate.body_names[0] == second && candidate.body_names[1] == first);
}

[[nodiscard]] bool isInvariantPair(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config)
{
  return candidate.kind == CollisionPairKind::kSelf &&
         isNamedPair(
    candidate, config.monitor_link_name,
    config.invariant_mount_neighbor_name);
}

[[nodiscard]] bool isMonitorNearPair(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config)
{
  return candidate.kind == CollisionPairKind::kSelf &&
         isNamedPair(candidate, config.monitor_link_name, config.monitor_near_link_name);
}

[[nodiscard]] double pairDistanceLipschitz(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config)
{
  if (!config.distance_bounds.empty()) {
    std::pair<std::string, std::string> key;
    if (candidate.kind == CollisionPairKind::kSelf) {
      key = std::minmax(candidate.body_names[0], candidate.body_names[1]);
    } else {
      key.first = candidate.body_types[0] == BodyTypes::WORLD_OBJECT ?
        candidate.body_names[1] : candidate.body_names[0];
    }
    const auto bound = config.distance_bounds.find(key);
    return bound == config.distance_bounds.end() ?
           config.default_distance_lipschitz_m_per_rad : bound->second;
  }
  if (isInvariantPair(candidate, config)) {
    return 0.0;
  }
  if (isMonitorNearPair(candidate, config)) {
    return config.monitor_near_distance_lipschitz_m_per_rad;
  }
  return config.default_distance_lipschitz_m_per_rad;
}

[[nodiscard]] double requiredClearance(
  const DistanceCandidate & candidate,
  const CollisionConstraintConfig & config,
  const double additional_joint_position_uncertainty_rad = 0.0)
{
  const double effective_joint_position_uncertainty =
    config.tracking_position_error_bound_rad + additional_joint_position_uncertainty_rad;
  return config.hard_clearance_m + config.numerical_distance_reserve_m +
         pairDistanceLipschitz(candidate, config) * effective_joint_position_uncertainty;
}

[[nodiscard]] double maximumRequiredClearance(
  const CollisionConstraintConfig & config,
  const double additional_joint_position_uncertainty_rad = 0.0)
{
  const double effective_joint_position_uncertainty =
    config.tracking_position_error_bound_rad + additional_joint_position_uncertainty_rad;
  return config.hard_clearance_m + config.numerical_distance_reserve_m +
         std::max(
    config.default_distance_lipschitz_m_per_rad,
    config.monitor_near_distance_lipschitz_m_per_rad) *
         effective_joint_position_uncertainty;
}

[[nodiscard]] const moveit::core::LinkModel * collisionBodyLink(
  const moveit::core::RobotState & state, const BodyType type, const std::string & name)
{
  if (type == BodyTypes::ROBOT_LINK) {
    return state.getRobotModel()->getLinkModel(name);
  }
  if (type == BodyTypes::ROBOT_ATTACHED) {
    const auto * body = state.getAttachedBody(name);
    return body == nullptr ? nullptr : body->getAttachedLink();
  }
  return nullptr;
}

[[nodiscard]] bool jointAffectsLink(
  const moveit::core::JointModel * joint, const moveit::core::LinkModel * link)
{
  for (auto * ancestor = link; ancestor != nullptr; ancestor = ancestor->getParentLinkModel()) {
    if (ancestor->getParentJointModel() == joint) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool relativeBodyLinks(
  const DistanceCandidate & candidate,
  const moveit::core::RobotState & state,
  const moveit::core::LinkModel * & first,
  const moveit::core::LinkModel * & second)
{
  // A mimic driver may affect a different branch without being its ancestor.
  // Keep the original full reserve for such models rather than infer a mask.
  if (!state.getRobotModel()->getMimicJointModels().empty()) {
    return false;
  }
  const bool first_world = candidate.body_types[0] == BodyTypes::WORLD_OBJECT;
  const bool second_world = candidate.body_types[1] == BodyTypes::WORLD_OBJECT;
  if ((candidate.kind == CollisionPairKind::kSelf && (first_world || second_world)) ||
    (candidate.kind == CollisionPairKind::kWorld && first_world == second_world))
  {
    return false;
  }
  first = collisionBodyLink(state, candidate.body_types[0], candidate.body_names[0]);
  second = collisionBodyLink(state, candidate.body_types[1], candidate.body_names[1]);
  return (first_world || first != nullptr) && (second_world || second != nullptr);
}

[[nodiscard]] std::vector<bool> relativeMotionVariables(
  const DistanceCandidate & candidate,
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group)
{
  std::vector<bool> variables(group.getVariableCount(), true);
  const moveit::core::LinkModel * first = nullptr;
  const moveit::core::LinkModel * second = nullptr;
  if (!relativeBodyLinks(candidate, state, first, second)) {
    return variables;
  }
  const auto & names = group.getVariableNames();
  for (std::size_t index = 0; index < names.size(); ++index) {
    const auto * joint = state.getRobotModel()->getJointOfVariable(names[index]);
    if (joint != nullptr) {
      variables[index] = jointAffectsLink(joint, first) != jointAffectsLink(joint, second);
    }
  }
  return variables;
}

[[nodiscard]] double pairInterSampleUncertainty(
  const DistanceCandidate & candidate,
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group,
  const Eigen::VectorXd & joint_uncertainty)
{
  const double fallback = joint_uncertainty.maxCoeff();
  if (fallback == 0.0) {
    return 0.0;
  }
  const moveit::core::LinkModel * first = nullptr;
  const moveit::core::LinkModel * second = nullptr;
  if (!relativeBodyLinks(candidate, state, first, second)) {
    return fallback;
  }
  double uncertainty = 0.0;
  const auto & names = group.getVariableNames();
  for (Eigen::Index index = 0; index < joint_uncertainty.size(); ++index) {
    if (joint_uncertainty[index] == 0.0) {
      continue;
    }
    const auto * joint = state.getRobotModel()->getJointOfVariable(names[index]);
    if (joint == nullptr) {
      return fallback;
    }
    // Common ancestors transform both self bodies rigidly together. For world
    // pairs the world side has no ancestors, retaining all robot-side motion.
    if (jointAffectsLink(joint, first) != jointAffectsLink(joint, second)) {
      uncertainty = std::max(uncertainty, joint_uncertainty[index]);
    }
  }
  return uncertainty;
}

void recordClearanceFailure(
  const DistanceCandidate & candidate, const double required_clearance,
  std::string & failure_reason)
{
  if (failure_reason.empty() && candidate.distance_m < required_clearance) {
    std::ostringstream message;
    message << (candidate.kind == CollisionPairKind::kSelf ? "self" : "world") <<
      " clearance violation: " << readablePair(candidate) << ", distance=" <<
      candidate.distance_m << " m, required=" << required_clearance << " m";
    failure_reason = message.str();
  }
}

[[nodiscard]] Eigen::MatrixXd pointJacobian(
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group,
  const BodyType body_type,
  const std::string & body_name,
  const Eigen::Vector3d & point_in_model_frame,
  std::string & error)
{
  const auto variable_count = static_cast<Eigen::Index>(group.getVariableCount());
  if (body_type == BodyTypes::WORLD_OBJECT) {
    return Eigen::MatrixXd::Zero(3, variable_count);
  }

  const auto * link = collisionBodyLink(state, body_type, body_name);

  if (link == nullptr) {
    error = "collision body '" + body_name + "' has no kinematic link";
    return {};
  }

  // A fixed link outside the group's updated subtree has exactly zero velocity.
  if (group.getUpdatedLinkModelsSet().count(link) == 0U) {
    return Eigen::MatrixXd::Zero(3, variable_count);
  }

  const Eigen::Vector3d point_in_link_frame =
    state.getGlobalLinkTransform(link).inverse() * point_in_model_frame;
  Eigen::MatrixXd full_jacobian;
  if (!state.getJacobian(&group, link, point_in_link_frame, full_jacobian, false) ||
    full_jacobian.rows() != 6 || full_jacobian.cols() != variable_count)
  {
    error = "failed to compute nearest-point Jacobian for '" + body_name + "'";
    return {};
  }
  if (!full_jacobian.array().isFinite().all()) {
    error = "nearest-point Jacobian for '" + body_name + "' is non-finite";
    return {};
  }
  return full_jacobian.topRows(3);
}

[[nodiscard]] Eigen::Matrix3d modelToGroupRootRotation(
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group)
{
  if (group.getJointModels().empty()) {
    return Eigen::Matrix3d::Identity();
  }
  const auto * root_link = group.getJointModels().front()->getParentLinkModel();
  if (root_link == nullptr) {
    return Eigen::Matrix3d::Identity();
  }
  return state.getGlobalLinkTransform(root_link).linear().transpose();
}

struct ReachableVelocityInterval
{
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
};

struct ReachableApproachInterval
{
  Eigen::VectorXd lower_contribution;
  Eigen::VectorXd upper_contribution;
  double minimum_mps{0.0};
  double maximum_mps{0.0};
};

[[nodiscard]] bool computeReachableVelocityInterval(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const double period_sec,
  ReachableVelocityInterval & interval,
  std::string & error)
{
  const Eigen::Index joint_count = state.position.size();
  interval.lower.resize(joint_count);
  interval.upper.resize(joint_count);
  for (Eigen::Index index = 0; index < joint_count; ++index) {
    const double minimum_acceleration = std::max(
      -limits.max_acceleration[index],
      state.acceleration[index] - limits.max_jerk[index] * period_sec);
    const double maximum_acceleration = std::min(
      limits.max_acceleration[index],
      state.acceleration[index] + limits.max_jerk[index] * period_sec);

    const double safe_lower_position =
      limits.lower_position[index];
    const double safe_upper_position =
      limits.upper_position[index];
    interval.lower[index] = std::max({
          -limits.max_velocity[index],
          state.velocity[index] + minimum_acceleration * period_sec,
          (safe_lower_position - state.position[index]) / period_sec});
    interval.upper[index] = std::min({
          limits.max_velocity[index],
          state.velocity[index] + maximum_acceleration * period_sec,
          (safe_upper_position - state.position[index]) / period_sec});
    if (!std::isfinite(interval.lower[index]) || !std::isfinite(interval.upper[index])) {
      error = "reachable joint-velocity interval is non-finite";
      return false;
    }
    if (interval.lower[index] > interval.upper[index]) {
      error = "reachable joint-velocity interval is empty";
      return false;
    }
  }
  return true;
}

[[nodiscard]] ReachableApproachInterval projectApproachInterval(
  const Eigen::VectorXd & distance_gradient,
  const ReachableVelocityInterval & velocity_interval)
{
  ReachableApproachInterval result;
  result.lower_contribution.resize(distance_gradient.size());
  result.upper_contribution.resize(distance_gradient.size());
  for (Eigen::Index index = 0; index < distance_gradient.size(); ++index) {
    if (distance_gradient[index] >= 0.0) {
      result.lower_contribution[index] =
        -distance_gradient[index] * velocity_interval.upper[index];
      result.upper_contribution[index] =
        -distance_gradient[index] * velocity_interval.lower[index];
    } else {
      result.lower_contribution[index] =
        -distance_gradient[index] * velocity_interval.lower[index];
      result.upper_contribution[index] =
        -distance_gradient[index] * velocity_interval.upper[index];
    }
    result.minimum_mps += result.lower_contribution[index];
    result.maximum_mps += result.upper_contribution[index];
  }
  return result;
}

/// Return a reachable velocity for each joint whose approach contribution is an upper bound for
/// that joint over every velocity in the reachable box with total approach no greater than cap.
/// The assembled vector need not itself satisfy the total cap: duplicating the shared slack is
/// intentional and makes the later per-joint stopping simulation conservative for the whole row.
[[nodiscard]] Eigen::VectorXd conservativeCandidateVelocityForApproachCap(
  const double cap,
  const Eigen::VectorXd & distance_gradient,
  const ReachableVelocityInterval & velocity_interval,
  const ReachableApproachInterval & approach_interval)
{
  if (!std::isfinite(cap) ||
    cap < approach_interval.minimum_mps - kComparisonTolerance ||
    cap > approach_interval.maximum_mps + kComparisonTolerance)
  {
    throw std::invalid_argument("approach cap lies outside the reachable interval");
  }

  Eigen::VectorXd candidate_velocity(distance_gradient.size());
  for (Eigen::Index index = 0; index < distance_gradient.size(); ++index) {
    if (distance_gradient[index] == 0.0) {
      candidate_velocity[index] = velocity_interval.lower[index];
      continue;
    }
    const double contribution_cap = std::clamp(
      cap - (approach_interval.minimum_mps -
      approach_interval.lower_contribution[index]),
      approach_interval.lower_contribution[index],
      approach_interval.upper_contribution[index]);
    candidate_velocity[index] = std::clamp(
      -contribution_cap / distance_gradient[index],
      velocity_interval.lower[index], velocity_interval.upper[index]);
  }
  return candidate_velocity;
}

[[nodiscard]] double requiredCollisionDistanceForApproachCap(
  const double cap,
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const ReachableVelocityInterval & velocity_interval,
  const ReachableApproachInterval & approach_interval,
  const double period_sec,
  const double residual_latency_sec)
{
  const Eigen::VectorXd candidate_velocity =
    conservativeCandidateVelocityForApproachCap(
    cap, distance_gradient, velocity_interval, approach_interval);
  const double stopping_distance = jerkLimitedCollisionStoppingDistance(
    distance_gradient, motion_state, candidate_velocity, motion_limits,
    period_sec, residual_latency_sec);
  return std::max(0.0, cap * period_sec + stopping_distance);
}

void validateConfig(const CollisionConstraintConfig & config)
{
  const auto finite = [](const double value) {return std::isfinite(value);};
  if (!finite(config.hard_clearance_m) || config.hard_clearance_m <= 0.0) {
    throw std::invalid_argument("hard_clearance_m must be finite and positive");
  }
  if (!finite(config.query_distance_m) ||
    config.query_distance_m <= config.hard_clearance_m)
  {
    throw std::invalid_argument("query_distance_m must exceed hard_clearance_m");
  }
  if (!finite(config.tracking_position_error_bound_rad) ||
    config.tracking_position_error_bound_rad < 0.0)
  {
    throw std::invalid_argument(
            "tracking_position_error_bound_rad must be finite and non-negative");
  }
  if (!finite(config.numerical_distance_reserve_m) ||
    config.numerical_distance_reserve_m < 0.0)
  {
    throw std::invalid_argument(
            "numerical_distance_reserve_m must be finite and non-negative");
  }
  if (!finite(config.default_distance_lipschitz_m_per_rad) ||
    config.default_distance_lipschitz_m_per_rad < 0.0 ||
    !finite(config.monitor_near_distance_lipschitz_m_per_rad) ||
    config.monitor_near_distance_lipschitz_m_per_rad < 0.0)
  {
    throw std::invalid_argument(
            "collision distance Lipschitz bounds must be finite and non-negative");
  }
  if (!finite(maximumRequiredClearance(config)) ||
    maximumRequiredClearance(config) >= config.query_distance_m)
  {
    throw std::invalid_argument(
            "query_distance_m must exceed the maximum robust clearance");
  }
  if (config.maximum_constraint_rows == 0U || config.maximum_contacts_per_pair == 0U) {
    throw std::invalid_argument("collision constraint/contact limits must be non-zero");
  }
  if (!finite(config.minimum_gradient_norm) || config.minimum_gradient_norm <= 0.0) {
    throw std::invalid_argument("minimum_gradient_norm must be finite and positive");
  }
  if (!finite(config.separation_buffer_m) || config.separation_buffer_m < 0.0 ||
    !finite(config.separation_max_velocity_mps) || config.separation_max_velocity_mps <= 0.0 ||
    !finite(config.separation_lookahead_sec) || config.separation_lookahead_sec < 0.0 ||
    maximumRequiredClearance(config) + config.separation_buffer_m >= config.query_distance_m)
  {
    throw std::invalid_argument("invalid separation guidance or insufficient query distance");
  }
  if (config.monitor_link_name.empty() || config.monitor_near_link_name.empty() ||
    config.invariant_mount_neighbor_name.empty() || config.protected_joint_name.empty())
  {
    throw std::invalid_argument("safety-profile link and joint names must be non-empty");
  }
  if (config.monitor_link_name == config.monitor_near_link_name ||
    config.monitor_link_name == config.invariant_mount_neighbor_name ||
    config.monitor_near_link_name == config.invariant_mount_neighbor_name)
  {
    throw std::invalid_argument("safety-profile monitor links must be distinct");
  }
  if (!finite(config.protected_joint_lower_rad) ||
    !finite(config.protected_joint_upper_rad) ||
    config.protected_joint_lower_rad >= config.protected_joint_upper_rad)
  {
    throw std::invalid_argument("protected joint corridor is invalid");
  }
  if (!finite(config.residual_latency_sec) || config.residual_latency_sec < 0.0) {
    throw std::invalid_argument("residual_latency_sec must be finite and non-negative");
  }
}

[[nodiscard]] bool validateMotionInput(
  const JointMotionState & state,
  const JointMotionLimits & limits,
  const Eigen::Index expected_size,
  std::string & error)
{
  const auto correct_size = [expected_size](const Eigen::VectorXd & value) {
      return value.size() == expected_size;
    };
  if (!correct_size(state.position) || !correct_size(state.velocity) ||
    !correct_size(state.acceleration) || !correct_size(limits.lower_position) ||
    !correct_size(limits.upper_position) || !correct_size(limits.position_margin) ||
    !correct_size(limits.max_velocity) ||
    !correct_size(limits.max_acceleration) || !correct_size(limits.max_jerk))
  {
    error = "motion state/limit dimensions do not match the JointModelGroup";
    return false;
  }
  if (!isFinite(state.position) || !isFinite(state.velocity) ||
    !isFinite(state.acceleration) || !isFinite(limits.lower_position) ||
    !isFinite(limits.upper_position) || !isFinite(limits.position_margin) ||
    !isFinite(limits.max_velocity) ||
    !isFinite(limits.max_acceleration) || !isFinite(limits.max_jerk))
  {
    error = "motion state/limits contain non-finite values";
    return false;
  }
  if ((limits.lower_position.array() >= limits.upper_position.array()).any() ||
    (limits.position_margin.array() < 0.0).any() ||
    ((limits.lower_position + limits.position_margin).array() >
    (limits.upper_position - limits.position_margin).array()).any() ||
    (limits.max_velocity.array() <= 0.0).any() ||
    (limits.max_acceleration.array() <= 0.0).any() ||
    (limits.max_jerk.array() <= 0.0).any())
  {
    error = "motion magnitude limits must be positive";
    return false;
  }
  return true;
}

[[nodiscard]] bool validateSegmentMotionState(
  const JointMotionState & state,
  const Eigen::Index expected_size,
  std::string & error)
{
  if (state.position.size() != expected_size || state.velocity.size() != expected_size ||
    state.acceleration.size() != expected_size)
  {
    error = "segment motion-state dimensions do not match the JointModelGroup";
    return false;
  }
  if (!isFinite(state.position) || !isFinite(state.velocity) ||
    !isFinite(state.acceleration))
  {
    error = "segment motion state contains non-finite values";
    return false;
  }
  return true;
}

[[nodiscard]] bool validateSafetyProfileModel(
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const CollisionConstraintConfig & config,
  std::string & error)
{
  for (const auto & required_link_name : {
        config.monitor_link_name, config.monitor_near_link_name,
        config.invariant_mount_neighbor_name
      })
  {
    const auto * required_link = robot_state.getRobotModel()->getLinkModel(required_link_name);
    if (required_link == nullptr) {
      error = "safety-profile link '" + required_link_name + "' is absent from RobotModel";
      return false;
    }
    // A disabled display remains as a fixed control/camera mount frame.
    if (required_link_name != config.monitor_link_name && required_link->getShapes().empty()) {
      error = "safety-profile link '" + required_link_name + "' has no collision geometry";
      return false;
    }
    if (joint_model_group.getUpdatedLinkModelsSet().count(required_link) == 0U) {
      error = "safety-profile link '" + required_link_name +
        "' is not controlled by the JointModelGroup";
      return false;
    }
  }
  return true;
}

void setFailure(CollisionConstraintDiagnostics & diagnostics, const std::string & message)
{
  diagnostics.valid = false;
  if (diagnostics.failure_reason.empty()) {
    diagnostics.failure_reason = message;
  }
}

}  // namespace

double jerkLimitedStoppingDistance(
  const double toward_velocity,
  const double toward_acceleration,
  const double max_acceleration,
  const double max_jerk,
  const double period_sec,
  const double residual_latency_sec)
{
  constexpr std::size_t kMaximumSimulationTicks = 10000;
  if (!std::isfinite(toward_velocity) || !std::isfinite(toward_acceleration) ||
    !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
    !std::isfinite(max_jerk) || max_jerk <= 0.0 ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0 ||
    std::abs(toward_acceleration) > max_acceleration + kComparisonTolerance)
  {
    throw std::invalid_argument("invalid jerk-limited stopping-distance input");
  }

  const double residual_ticks_value = std::ceil(residual_latency_sec / period_sec);
  const double acceleration_step = max_jerk * period_sec;
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks) ||
    !std::isfinite(acceleration_step) || acceleration_step <= 0.0)
  {
    throw std::invalid_argument("discrete stopping horizon is invalid");
  }
  const auto residual_ticks = static_cast<std::size_t>(residual_ticks_value);

  double position = 0.0;
  double velocity = toward_velocity;
  double acceleration = toward_acceleration;
  double maximum_position = 0.0;
  std::size_t simulated_ticks = 0;
  const auto integrate_tick = [&]() {
      velocity += acceleration * period_sec;
      position += velocity * period_sec;
      if (!std::isfinite(position) || !std::isfinite(velocity) ||
        !std::isfinite(acceleration))
      {
        throw std::invalid_argument("discrete stopping trajectory is non-finite");
      }
      maximum_position = std::max(maximum_position, position);
      ++simulated_ticks;
    };

  for (std::size_t tick = 0; tick < residual_ticks; ++tick) {
    acceleration = std::min(max_acceleration, acceleration + acceleration_step);
    integrate_tick();
  }
  while ((velocity > 0.0 || acceleration > 0.0) &&
    simulated_ticks < kMaximumSimulationTicks)
  {
    acceleration = std::max(-max_acceleration, acceleration - acceleration_step);
    integrate_tick();
  }
  if (velocity > 0.0 || acceleration > 0.0) {
    throw std::invalid_argument("discrete stopping trajectory exceeds bounded horizon");
  }
  return std::max(0.0, maximum_position);
}

double maximumSafeVelocityTowardBoundary(
  const double remaining_distance,
  const double current_velocity_toward_boundary,
  const double minimum_reachable_velocity,
  const double maximum_reachable_velocity,
  const double max_acceleration,
  const double max_jerk,
  const double period_sec,
  const double residual_latency_sec)
{
  if (!std::isfinite(remaining_distance) ||
    !std::isfinite(current_velocity_toward_boundary) ||
    !std::isfinite(minimum_reachable_velocity) ||
    !std::isfinite(maximum_reachable_velocity) ||
    minimum_reachable_velocity > maximum_reachable_velocity ||
    !std::isfinite(max_acceleration) || max_acceleration <= 0.0 ||
    !std::isfinite(max_jerk) || max_jerk <= 0.0 ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0)
  {
    throw std::invalid_argument("invalid safe-boundary-velocity input");
  }

  const double minimum_acceleration =
    (minimum_reachable_velocity - current_velocity_toward_boundary) / period_sec;
  const double maximum_acceleration_candidate =
    (maximum_reachable_velocity - current_velocity_toward_boundary) / period_sec;
  if (std::abs(minimum_acceleration) > max_acceleration + kComparisonTolerance ||
    std::abs(maximum_acceleration_candidate) > max_acceleration + kComparisonTolerance)
  {
    throw std::invalid_argument(
            "reachable boundary-velocity interval violates the acceleration limit");
  }

  const auto required_distance = [&](const double candidate_velocity) {
      const double candidate_acceleration =
        (candidate_velocity - current_velocity_toward_boundary) / period_sec;
      return candidate_velocity * period_sec + jerkLimitedStoppingDistance(
        candidate_velocity, candidate_acceleration, max_acceleration, max_jerk,
        period_sec, residual_latency_sec);
    };

  if (required_distance(minimum_reachable_velocity) > remaining_distance) {
    return minimum_reachable_velocity;
  }
  if (required_distance(maximum_reachable_velocity) <= remaining_distance) {
    return maximum_reachable_velocity;
  }

  double lower = minimum_reachable_velocity;
  double upper = maximum_reachable_velocity;
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    if (required_distance(middle) <= remaining_distance) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

double jerkLimitedCollisionStoppingDistance(
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const Eigen::VectorXd & candidate_joint_velocity,
  const JointMotionLimits & motion_limits,
  const double period_sec,
  const double residual_latency_sec)
{
  constexpr std::size_t kMaximumSimulationTicks = 10000;
  const Eigen::Index joint_count = distance_gradient.size();
  std::string input_error;
  if (joint_count <= 0 || candidate_joint_velocity.size() != joint_count ||
    !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !std::isfinite(residual_latency_sec) || residual_latency_sec < 0.0 ||
    !isFinite(distance_gradient) || !isFinite(candidate_joint_velocity) ||
    !validateMotionInput(motion_state, motion_limits, joint_count, input_error))
  {
    throw std::invalid_argument(
            input_error.empty() ?
            "invalid vector collision stopping-distance input" : input_error);
  }

  const Eigen::VectorXd candidate_acceleration =
    (candidate_joint_velocity - motion_state.velocity) / period_sec;
  const Eigen::VectorXd candidate_position =
    motion_state.position + candidate_joint_velocity * period_sec;
  const Eigen::VectorXd minimum_position =
    motion_limits.lower_position;
  const Eigen::VectorXd maximum_joint_position =
    motion_limits.upper_position;
  if (!isFinite(candidate_acceleration) || !isFinite(candidate_position) ||
    (candidate_joint_velocity.cwiseAbs().array() >
    motion_limits.max_velocity.array() + kComparisonTolerance).any() ||
    (candidate_acceleration.cwiseAbs().array() >
    motion_limits.max_acceleration.array() + kComparisonTolerance).any() ||
    ((candidate_acceleration - motion_state.acceleration).cwiseAbs().array() >
    motion_limits.max_jerk.array() * period_sec + kComparisonTolerance).any() ||
    (candidate_position.array() < minimum_position.array() - kComparisonTolerance).any() ||
    (candidate_position.array() > maximum_joint_position.array() + kComparisonTolerance).any())
  {
    throw std::invalid_argument("candidate joint velocity is not reachable in one period");
  }

  const double residual_ticks_value = std::ceil(residual_latency_sec / period_sec);
  if (!std::isfinite(residual_ticks_value) || residual_ticks_value < 0.0 ||
    residual_ticks_value > static_cast<double>(kMaximumSimulationTicks))
  {
    throw std::invalid_argument("vector collision stopping horizon is invalid");
  }
  const auto residual_ticks = static_cast<std::size_t>(residual_ticks_value);

  // Work in per-joint approach coordinates. This preserves every individual
  // acceleration/jerk saturation while making both gradient signs symmetric.
  Eigen::VectorXd velocity_contribution =
    -distance_gradient.array() * candidate_joint_velocity.array();
  Eigen::VectorXd acceleration_contribution =
    -distance_gradient.array() * candidate_acceleration.array();
  const Eigen::VectorXd maximum_acceleration_contribution =
    distance_gradient.cwiseAbs().array() * motion_limits.max_acceleration.array();
  const Eigen::VectorXd jerk_step =
    distance_gradient.cwiseAbs().array() * motion_limits.max_jerk.array() * period_sec;
  if (!isFinite(velocity_contribution) || !isFinite(acceleration_contribution) ||
    !isFinite(maximum_acceleration_contribution) || !isFinite(jerk_step) ||
    maximum_acceleration_contribution.maxCoeff() <= 0.0)
  {
    throw std::invalid_argument("distance gradient has no controllable projected limits");
  }

  double position = 0.0;
  double maximum_position = 0.0;
  std::size_t simulated_ticks = 0;
  const auto integrate_tick = [&]() {
      velocity_contribution += acceleration_contribution * period_sec;
      const double approach_velocity = velocity_contribution.sum();
      position += approach_velocity * period_sec;
      if (!isFinite(velocity_contribution) || !isFinite(acceleration_contribution) ||
        !std::isfinite(approach_velocity) || !std::isfinite(position))
      {
        throw std::invalid_argument("vector collision stopping trajectory is non-finite");
      }
      maximum_position = std::max(maximum_position, position);
      ++simulated_ticks;
    };

  for (std::size_t tick = 0; tick < residual_ticks; ++tick) {
    acceleration_contribution =
      (acceleration_contribution + jerk_step).cwiseMin(
      maximum_acceleration_contribution);
    integrate_tick();
  }

  auto approach_velocity = velocity_contribution.sum();
  auto approach_acceleration = acceleration_contribution.sum();
  while ((approach_velocity > 0.0 || approach_acceleration > 0.0) &&
    simulated_ticks < kMaximumSimulationTicks)
  {
    acceleration_contribution =
      (acceleration_contribution - jerk_step).cwiseMax(
      -maximum_acceleration_contribution);
    integrate_tick();
    approach_velocity = velocity_contribution.sum();
    approach_acceleration = acceleration_contribution.sum();
  }
  if (approach_velocity > 0.0 || approach_acceleration > 0.0) {
    throw std::invalid_argument("vector collision stopping trajectory exceeds bounded horizon");
  }
  return std::max(0.0, maximum_position);
}

double maximumSafeCollisionApproachSpeed(
  const double remaining_distance,
  const Eigen::VectorXd & distance_gradient,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const double period_sec,
  const double residual_latency_sec)
{
  const Eigen::Index joint_count = distance_gradient.size();
  if (joint_count <= 0 || motion_state.position.size() != joint_count ||
    motion_state.velocity.size() != joint_count ||
    motion_state.acceleration.size() != joint_count ||
    motion_limits.lower_position.size() != joint_count ||
    motion_limits.upper_position.size() != joint_count ||
    motion_limits.position_margin.size() != joint_count ||
    motion_limits.max_velocity.size() != joint_count ||
    motion_limits.max_acceleration.size() != joint_count ||
    motion_limits.max_jerk.size() != joint_count ||
    !std::isfinite(remaining_distance) || !isFinite(distance_gradient) ||
    !isFinite(motion_state.position) || !isFinite(motion_state.velocity) ||
    !isFinite(motion_state.acceleration) ||
    !isFinite(motion_limits.lower_position) ||
    !isFinite(motion_limits.upper_position) ||
    !isFinite(motion_limits.position_margin) ||
    !isFinite(motion_limits.max_velocity) ||
    !isFinite(motion_limits.max_acceleration) || !isFinite(motion_limits.max_jerk) ||
    (motion_limits.lower_position.array() >= motion_limits.upper_position.array()).any() ||
    (motion_limits.position_margin.array() < 0.0).any() ||
    ((motion_limits.lower_position + motion_limits.position_margin).array() >
    (motion_limits.upper_position - motion_limits.position_margin).array()).any() ||
    (motion_limits.max_velocity.array() <= 0.0).any() ||
    (motion_limits.max_acceleration.array() <= 0.0).any() ||
    (motion_limits.max_jerk.array() <= 0.0).any())
  {
    throw std::invalid_argument("invalid collision approach-envelope input");
  }

  ReachableVelocityInterval reachable_velocity;
  std::string interval_error;
  if (!computeReachableVelocityInterval(
      motion_state, motion_limits, period_sec, reachable_velocity, interval_error))
  {
    throw std::invalid_argument(interval_error);
  }
  const ReachableApproachInterval reachable_approach =
    projectApproachInterval(distance_gradient, reachable_velocity);
  const auto required_distance = [&](const double candidate_approach_speed) {
      return requiredCollisionDistanceForApproachCap(
        candidate_approach_speed, distance_gradient, motion_state, motion_limits,
        reachable_velocity, reachable_approach, period_sec, residual_latency_sec);
    };

  if (required_distance(reachable_approach.minimum_mps) > remaining_distance) {
    return reachable_approach.minimum_mps;
  }
  if (required_distance(reachable_approach.maximum_mps) <= remaining_distance) {
    return reachable_approach.maximum_mps;
  }

  double lower = reachable_approach.minimum_mps;
  double upper = reachable_approach.maximum_mps;
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    if (required_distance(middle) <= remaining_distance) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

CollisionConstraintBuilder::CollisionConstraintBuilder(CollisionConstraintConfig config)
: config_(std::move(config))
{
  validateConfig(config_);
}

CollisionConstraintResult CollisionConstraintBuilder::build(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const JointMotionState & motion_state,
  const JointMotionLimits & motion_limits,
  const double period_sec) const
{
  CollisionConstraintResult result;
  auto & diagnostics = result.diagnostics;
  diagnostics.valid = false;

  const auto variable_count = static_cast<Eigen::Index>(joint_model_group.getVariableCount());
  if (!std::isfinite(period_sec) || period_sec <= 0.0) {
    setFailure(diagnostics, "control period must be finite and positive");
    return result;
  }
  if (variable_count <= 0) {
    setFailure(diagnostics, "JointModelGroup has no variables");
    return result;
  }
  if (robot_state.getRobotModel().get() != scene.getRobotModel().get()) {
    setFailure(diagnostics, "RobotState and PlanningScene use different RobotModels");
    return result;
  }
  if (!validateSafetyProfileModel(
      robot_state, joint_model_group, config_, diagnostics.failure_reason))
  {
    return result;
  }
  if (!validateMotionInput(motion_state, motion_limits, variable_count,
      diagnostics.failure_reason))
  {
    return result;
  }

  const auto & variable_names = joint_model_group.getVariableNames();
  const auto protected_joint = std::find(
    variable_names.begin(), variable_names.end(), config_.protected_joint_name);
  if (protected_joint == variable_names.end()) {
    setFailure(
      diagnostics,
      "protected joint '" + config_.protected_joint_name + "' is absent from the group");
    return result;
  }
  const auto protected_index = static_cast<Eigen::Index>(
    std::distance(variable_names.begin(), protected_joint));

  moveit::core::RobotState state(robot_state);
  state.update(true);
  Eigen::VectorXd state_positions(variable_count);
  state.copyJointGroupPositions(&joint_model_group, state_positions);
  if (!isFinite(state_positions) ||
    (state_positions - motion_state.position).cwiseAbs().maxCoeff() > kComparisonTolerance)
  {
    setFailure(diagnostics, "RobotState and queued motion-state positions disagree");
    return result;
  }

  ReachableVelocityInterval reachable_velocity;
  if (!computeReachableVelocityInterval(
      motion_state, motion_limits, period_sec, reachable_velocity,
      diagnostics.failure_reason))
  {
    return result;
  }

  const double q = motion_state.position[protected_index];
  const double qdot = motion_state.velocity[protected_index];
  const double max_velocity = motion_limits.max_velocity[protected_index];
  const double max_acceleration = motion_limits.max_acceleration[protected_index];
  const double max_jerk = motion_limits.max_jerk[protected_index];

  const double upper_remaining = config_.protected_joint_upper_rad - q;
  const double lower_remaining = q - config_.protected_joint_lower_rad;
  double safe_upper_velocity = maximumSafeVelocityTowardBoundary(
    upper_remaining, qdot,
    reachable_velocity.lower[protected_index], reachable_velocity.upper[protected_index],
    max_acceleration, max_jerk, period_sec, config_.residual_latency_sec);
  double safe_lower_velocity = maximumSafeVelocityTowardBoundary(
    lower_remaining, -qdot,
    -reachable_velocity.upper[protected_index], -reachable_velocity.lower[protected_index],
    max_acceleration, max_jerk, period_sec, config_.residual_latency_sec);

  if (upper_remaining < 0.0) {
    diagnostics.protected_joint_corridor_violated = true;
    safe_upper_velocity = std::max(-max_velocity, upper_remaining / period_sec);
  }
  if (lower_remaining < 0.0) {
    diagnostics.protected_joint_corridor_violated = true;
    safe_lower_velocity = std::max(-max_velocity, lower_remaining / period_sec);
  }

  LinearVelocityConstraint protected_joint_row;
  protected_joint_row.coefficients = Eigen::VectorXd::Zero(variable_count);
  protected_joint_row.coefficients[protected_index] = 1.0;
  protected_joint_row.lower_bound = -safe_lower_velocity;
  protected_joint_row.upper_bound = safe_upper_velocity;
  result.constraints.push_back(std::move(protected_joint_row));

  collision_detection::AllowedCollisionMatrix query_acm(
    scene.getAllowedCollisionMatrix());
  collision_detection::DistanceRequest distance_request;
  distance_request.enable_nearest_points = true;
  distance_request.enable_signed_distance = true;
  distance_request.compute_gradient = true;
  // SINGLE returns the nearest result for every body pair. Re-querying after
  // allowing each minimum pair would traverse the same mesh pairs repeatedly
  // and is substantially slower than consuming this bounded result once.
  distance_request.type = collision_detection::DistanceRequestType::SINGLE;
  distance_request.max_contacts_per_body = 1U;
  distance_request.group_name = joint_model_group.getName();
  distance_request.distance_threshold = config_.query_distance_m;
  distance_request.acm = &query_acm;
  distance_request.enableGroup(scene.getRobotModel());

  const auto & collision_environment = scene.getCollisionEnvUnpadded();
  if (!collision_environment) {
    setFailure(diagnostics, "PlanningScene has no unpadded collision environment");
    return result;
  }

  DistanceResult self_distances;
  DistanceResult world_distances;
  try {
    const auto query_start = std::chrono::steady_clock::now();
    collision_environment->distanceSelf(distance_request, self_distances, state);
    const auto self_query_end = std::chrono::steady_clock::now();
    collision_environment->distanceRobot(distance_request, world_distances, state);
    diagnostics.self_query_ms = std::chrono::duration<double, std::milli>(
      self_query_end - query_start).count();
    diagnostics.world_query_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - self_query_end).count();
  } catch (const std::exception & exception) {
    setFailure(
      diagnostics, std::string("collision constraint query failed: ") + exception.what());
    return result;
  }
  diagnostics.collision_detected = self_distances.collision || world_distances.collision;
  diagnostics.query_joint_position = state_positions;
  diagnostics.query_distance_m = distance_request.distance_threshold;
  diagnostics.protected_joint_index = static_cast<std::size_t>(protected_index);

  std::vector<DistanceCandidate> candidates;
  appendDistanceCandidates(self_distances, CollisionPairKind::kSelf, candidates);
  appendDistanceCandidates(world_distances, CollisionPairKind::kWorld, candidates);
  std::sort(candidates.begin(), candidates.end(), candidateLess);

  // Keep a deterministic bounded number defensively. SINGLE is configured for one nearest
  // result per body pair, but this also protects the result if a collision backend returns more.
  std::map<std::string, std::size_t> retained_per_pair;
  std::vector<DistanceCandidate> retained_candidates;
  retained_candidates.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    auto & retained_count = retained_per_pair[candidateKey(candidate)];
    if (retained_count < config_.maximum_contacts_per_pair) {
      retained_candidates.push_back(candidate);
      ++retained_count;
    }
  }

  const Eigen::Matrix3d model_to_root = modelToGroupRootRotation(state, joint_model_group);
  for (const auto & candidate : retained_candidates) {
    ++diagnostics.candidate_contacts;
    CollisionPairDiagnostic pair_diagnostic;
    pair_diagnostic.kind = candidate.kind;
    pair_diagnostic.first_body = candidate.body_names[0];
    pair_diagnostic.second_body = candidate.body_names[1];
    pair_diagnostic.distance_m = candidate.distance_m;
    pair_diagnostic.invariant_pair = isInvariantPair(candidate, config_);
    pair_diagnostic.distance_lipschitz_m_per_rad = pairDistanceLipschitz(candidate, config_);
    pair_diagnostic.required_clearance_m = requiredClearance(candidate, config_);
    pair_diagnostic.relative_motion_variables = relativeMotionVariables(
      candidate, state, joint_model_group);

    if (!std::isfinite(candidate.distance_m) ||
      !isFinite(candidate.nearest_points[0]) || !isFinite(candidate.nearest_points[1]) ||
      !isFinite(candidate.normal))
    {
      setFailure(diagnostics,
          "FCL returned non-finite distance data for " + readablePair(candidate));
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }

    if (candidate.kind == CollisionPairKind::kSelf) {
      diagnostics.minimum_self_distance_m = std::min(
        diagnostics.minimum_self_distance_m, candidate.distance_m);
    } else {
      diagnostics.minimum_world_distance_m = std::min(
        diagnostics.minimum_world_distance_m, candidate.distance_m);
    }
    if (candidate.distance_m < diagnostics.minimum_distance_m) {
      diagnostics.minimum_distance_m = candidate.distance_m;
      diagnostics.closest_pair = readablePair(candidate);
    }
    diagnostics.hard_clearance_violated =
      diagnostics.hard_clearance_violated || candidate.distance_m < config_.hard_clearance_m;
    pair_diagnostic.robust_clearance_violated =
      candidate.distance_m < pair_diagnostic.required_clearance_m;
    diagnostics.robust_clearance_violated =
      diagnostics.robust_clearance_violated || pair_diagnostic.robust_clearance_violated;

    std::string jacobian_error;
    const Eigen::MatrixXd first_jacobian = pointJacobian(
      state, joint_model_group, candidate.body_types[0], candidate.body_names[0],
      candidate.nearest_points[0], jacobian_error);
    if (!jacobian_error.empty()) {
      setFailure(diagnostics, jacobian_error);
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }
    const Eigen::MatrixXd second_jacobian = pointJacobian(
      state, joint_model_group, candidate.body_types[1], candidate.body_names[1],
      candidate.nearest_points[1], jacobian_error);
    if (!jacobian_error.empty()) {
      setFailure(diagnostics, jacobian_error);
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }

    Eigen::Vector3d normal_in_root = model_to_root * candidate.normal;
    const double normal_norm = normal_in_root.norm();
    if (!std::isfinite(normal_norm) || normal_norm < config_.minimum_gradient_norm) {
      setFailure(diagnostics, "FCL returned an unusable normal for " + readablePair(candidate));
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }
    normal_in_root /= normal_norm;
    const Eigen::VectorXd gradient =
      (normal_in_root.transpose() * (second_jacobian - first_jacobian)).transpose();
    pair_diagnostic.gradient_norm = gradient.norm();

    if (pair_diagnostic.invariant_pair &&
      pair_diagnostic.gradient_norm < config_.minimum_gradient_norm)
    {
      if (pair_diagnostic.robust_clearance_violated) {
        setFailure(
          diagnostics,
          "fixed monitor attachment invariant violates robust clearance: " +
          readablePair(candidate));
      } else {
        ++diagnostics.skipped_invariant_contacts;
      }
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }

    if (!isFinite(gradient) || pair_diagnostic.gradient_norm < config_.minimum_gradient_norm) {
      ++diagnostics.uncontrollable_contacts;
      if (pair_diagnostic.robust_clearance_violated) {
        setFailure(
          diagnostics,
          "robust-clearance violation for " + readablePair(candidate) +
          " has a zero joint-space gradient");
      }
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }

    if (config_.separation_buffer_m > 0.0) {
      const double approach_speed = std::max(0.0, -gradient.dot(motion_state.velocity));
      const double predicted_headroom = candidate.distance_m -
        pair_diagnostic.required_clearance_m - approach_speed * config_.separation_lookahead_sec;
      const double progress = std::clamp(
        1.0 - predicted_headroom / config_.separation_buffer_m, 0.0, 1.0);
      const double activation = progress * progress * (3.0 - 2.0 * progress);
      if (activation > 0.0) {
        // Reuse the already evaluated distance Jacobian; no extra mesh queries.
        // This preference never replaces or relaxes the hard rows below.
        result.separation_guidance.push_back({
            gradient, activation, config_.separation_max_velocity_mps * activation});
      }
    }

    if (candidate.distance_m < diagnostics.minimum_controllable_distance_m) {
      diagnostics.minimum_controllable_distance_m = candidate.distance_m;
      diagnostics.closest_controllable_pair = readablePair(candidate);
    }

    const ReachableApproachInterval reachable_approach =
      projectApproachInterval(gradient, reachable_velocity);
    pair_diagnostic.minimum_reachable_approach_speed_mps =
      reachable_approach.minimum_mps;
    pair_diagnostic.maximum_reachable_approach_speed_mps =
      reachable_approach.maximum_mps;
    try {
      pair_diagnostic.jerk_aware_safe_approach_speed_mps =
        maximumSafeCollisionApproachSpeed(
        candidate.distance_m - pair_diagnostic.required_clearance_m, gradient, motion_state,
        motion_limits, period_sec, config_.residual_latency_sec);
      const double minimum_required_distance =
        requiredCollisionDistanceForApproachCap(
        reachable_approach.minimum_mps, gradient, motion_state, motion_limits,
        reachable_velocity, reachable_approach, period_sec, config_.residual_latency_sec);
      pair_diagnostic.outside_viability =
        minimum_required_distance >
        candidate.distance_m - pair_diagnostic.required_clearance_m + kComparisonTolerance;
    } catch (const std::invalid_argument & exception) {
      setFailure(
        diagnostics, "failed to build approach envelope for " + readablePair(candidate) +
        ": " + exception.what());
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }
    pair_diagnostic.applied_approach_speed_cap_mps = std::max(
      pair_diagnostic.jerk_aware_safe_approach_speed_mps,
      reachable_approach.minimum_mps);
    if (pair_diagnostic.outside_viability) {
      ++diagnostics.outside_viability_contacts;
    }

    if (pair_diagnostic.applied_approach_speed_cap_mps + kComparisonTolerance >=
      reachable_approach.maximum_mps)
    {
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }

    if (result.constraints.size() - 1U >= config_.maximum_constraint_rows) {
      ++diagnostics.truncated_constraints;
      diagnostics.pairs.push_back(std::move(pair_diagnostic));
      continue;
    }

    LinearVelocityConstraint constraint;
    constraint.coefficients = gradient;
    constraint.lower_bound = -pair_diagnostic.applied_approach_speed_cap_mps;
    result.constraints.push_back(std::move(constraint));
    pair_diagnostic.constraint_added = true;
    ++diagnostics.active_distance_constraints;
    diagnostics.pairs.push_back(std::move(pair_diagnostic));
  }

  if (diagnostics.truncated_constraints != 0U) {
    setFailure(
      diagnostics,
      "collision row budget exhausted before all query-distance contacts were constrained");
  }
  if (diagnostics.failure_reason.empty()) {
    diagnostics.valid = true;
    diagnostics.distance_query_complete = true;
  }
  return result;
}

CollisionActualStateValidationResult CollisionConstraintBuilder::validateActualState(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group) const
{
  CollisionActualStateValidationResult result;
  const auto fail = [&result](const std::string & reason) {
      result.input_valid = false;
      result.unsafe = true;
      if (result.failure_reason.empty()) {
        result.failure_reason = reason;
      }
    };

  const Eigen::Index variable_count = static_cast<Eigen::Index>(
    joint_model_group.getVariableCount());
  if (variable_count <= 0) {
    fail("JointModelGroup has no variables");
    return result;
  }
  if (robot_state.getRobotModel().get() != scene.getRobotModel().get()) {
    fail("RobotState and PlanningScene use different RobotModels");
    return result;
  }
  if (!validateSafetyProfileModel(
      robot_state, joint_model_group, config_, result.failure_reason))
  {
    result.unsafe = true;
    return result;
  }

  const auto & variable_names = joint_model_group.getVariableNames();
  const auto protected_joint = std::find(
    variable_names.begin(), variable_names.end(), config_.protected_joint_name);
  if (protected_joint == variable_names.end()) {
    fail("protected joint '" + config_.protected_joint_name + "' is absent from the group");
    return result;
  }
  const Eigen::Index protected_index = static_cast<Eigen::Index>(
    std::distance(variable_names.begin(), protected_joint));

  Eigen::VectorXd positions(variable_count);
  robot_state.copyJointGroupPositions(&joint_model_group, positions);
  if (!isFinite(positions)) {
    fail("RobotState contains non-finite joint positions");
    return result;
  }
  result.protected_joint_position_rad = positions[protected_index];
  result.protected_joint_corridor_violated =
    positions[protected_index] < config_.protected_joint_lower_rad ||
    positions[protected_index] > config_.protected_joint_upper_rad;

  const auto & collision_environment = scene.getCollisionEnvUnpadded();
  if (!collision_environment) {
    fail("PlanningScene has no unpadded collision environment");
    return result;
  }

  moveit::core::RobotState state(robot_state);
  state.update(true);

  collision_detection::DistanceRequest distance_request;
  distance_request.enable_nearest_points = false;
  distance_request.enable_signed_distance = true;
  distance_request.compute_gradient = false;
  distance_request.type = collision_detection::DistanceRequestType::SINGLE;
  distance_request.max_contacts_per_body = 1U;
  distance_request.group_name = joint_model_group.getName();
  distance_request.distance_threshold = config_.hard_clearance_m;
  distance_request.acm = &scene.getAllowedCollisionMatrix();
  distance_request.enableGroup(scene.getRobotModel());

  DistanceResult self_distance;
  DistanceResult world_distance;
  try {
    collision_environment->distanceSelf(distance_request, self_distance, state);
    collision_environment->distanceRobot(distance_request, world_distance, state);
  } catch (const std::exception & exception) {
    fail(std::string("actual-state collision query failed: ") + exception.what());
    return result;
  }

  const auto record_minimum =
    [this, &result, &fail](
    const DistanceResult & distance_result,
    const CollisionPairKind kind,
    double & minimum_distance) {
      const auto & reported = distance_result.minimum_distance;
      const bool has_first_body = !reported.link_names[0].empty();
      const bool has_second_body = !reported.link_names[1].empty();
      if (!has_first_body && !has_second_body) {
        const bool is_empty_sentinel =
          reported.distance == std::numeric_limits<double>::max() ||
          reported.distance == std::numeric_limits<double>::infinity();
        if (distance_result.collision || !distance_result.distances.empty() ||
          !is_empty_sentinel)
        {
          fail("collision backend returned an invalid unnamed minimum-distance result");
          return false;
        }
        return true;
      }
      if (!has_first_body || !has_second_body || !std::isfinite(reported.distance)) {
        fail("collision backend returned invalid minimum-distance data");
        return false;
      }

      const DistanceCandidate candidate = canonicalize(reported, kind);
      minimum_distance = candidate.distance_m;
      if (candidate.distance_m < result.minimum_distance_m) {
        result.minimum_distance_m = candidate.distance_m;
        result.closest_pair = readablePair(candidate);
      }
      result.collision_detected =
        result.collision_detected || distance_result.collision || candidate.distance_m <= 0.0;
      result.hard_clearance_violated =
        result.hard_clearance_violated || candidate.distance_m < config_.hard_clearance_m;
      return true;
    };

  if (!record_minimum(
      self_distance, CollisionPairKind::kSelf, result.minimum_self_distance_m) ||
    !record_minimum(
      world_distance, CollisionPairKind::kWorld, result.minimum_world_distance_m))
  {
    return result;
  }

  result.input_valid = true;
  result.unsafe = result.collision_detected || result.hard_clearance_violated ||
    result.protected_joint_corridor_violated;
  return result;
}

bool CollisionConstraintBuilder::shouldValidateSegment(
  const CollisionConstraintDiagnostics & diagnostics,
  const double activation_distance_m) const
{
  if (!diagnostics.valid) {
    throw std::invalid_argument("cannot select segment validation from invalid diagnostics");
  }
  if (!std::isfinite(activation_distance_m) ||
    activation_distance_m < config_.hard_clearance_m ||
    activation_distance_m > config_.query_distance_m)
  {
    throw std::invalid_argument(
            "segment validation distance must lie inside hard-clearance/query interval");
  }
  return std::isfinite(diagnostics.minimum_controllable_distance_m) &&
         diagnostics.minimum_controllable_distance_m <= activation_distance_m;
}

bool CollisionConstraintBuilder::canCertifySegment(
  const CollisionConstraintDiagnostics & diagnostics,
  const Eigen::VectorXd & start_position,
  const Eigen::VectorXd & end_position) const
{
  const Eigen::Index count = diagnostics.query_joint_position.size();
  if (!diagnostics.valid || !diagnostics.distance_query_complete ||
    diagnostics.collision_detected || diagnostics.robust_clearance_violated ||
    diagnostics.hard_clearance_violated || diagnostics.protected_joint_corridor_violated ||
    count <= 0 || start_position.size() != count || end_position.size() != count ||
    !isFinite(diagnostics.query_joint_position) || !isFinite(start_position) ||
    !isFinite(end_position) || !std::isfinite(diagnostics.query_distance_m) ||
    diagnostics.query_distance_m <= 0.0 ||
    diagnostics.protected_joint_index >= static_cast<std::size_t>(count) ||
    diagnostics.candidate_contacts != diagnostics.pairs.size())
  {
    return false;
  }
  const auto protected_index = static_cast<Eigen::Index>(diagnostics.protected_joint_index);
  if (std::min(start_position[protected_index], end_position[protected_index]) <
    config_.protected_joint_lower_rad ||
    std::max(start_position[protected_index], end_position[protected_index]) >
    config_.protected_joint_upper_rad)
  {
    return false;
  }

  const Eigen::VectorXd excursion =
    (start_position - diagnostics.query_joint_position).cwiseAbs().cwiseMax(
    (end_position - diagnostics.query_joint_position).cwiseAbs());
  // SINGLE reports each queried body's nearest pair distance. Any unreported,
  // non-ACM-exempt pair starts at least query_distance away. Bounding its full
  // possible decrease also covers a previously distant pair entering the query.
  if (!isFinite(excursion) ||
    maximumRequiredClearance(config_, excursion.maxCoeff()) > diagnostics.query_distance_m)
  {
    return false;
  }
  for (const auto & pair : diagnostics.pairs) {
    if (!std::isfinite(pair.distance_m) ||
      pair.relative_motion_variables.size() != static_cast<std::size_t>(count))
    {
      return false;
    }
    double relative_excursion = 0.0;
    for (Eigen::Index index = 0; index < count; ++index) {
      if (pair.relative_motion_variables[static_cast<std::size_t>(index)]) {
        relative_excursion = std::max(relative_excursion, excursion[index]);
      }
    }
    DistanceCandidate candidate;
    candidate.kind = pair.kind;
    candidate.body_names[0] = pair.first_body;
    candidate.body_names[1] = pair.second_body;
    if (pair.distance_m < requiredClearance(candidate, config_, relative_excursion)) {
      return false;
    }
  }
  return true;
}

CollisionSegmentValidationResult CollisionConstraintBuilder::validateSegment(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & current_robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const JointMotionState & current_motion_state,
  const JointMotionState & candidate_motion_state,
  const std::size_t substeps) const
{
  CollisionSegmentValidationResult result;
  result.requested_substeps = substeps;
  const Eigen::Index variable_count = static_cast<Eigen::Index>(
    joint_model_group.getVariableCount());

  const auto fail = [&result](const std::string & reason) {
      result.input_valid = false;
      result.unsafe = true;
      result.failure_reason = reason;
    };
  if (substeps < 2U) {
    fail("segment validation requires at least two equal substeps");
    return result;
  }
  if (variable_count <= 0) {
    fail("JointModelGroup has no variables");
    return result;
  }
  if (current_robot_state.getRobotModel().get() != scene.getRobotModel().get()) {
    fail("RobotState and PlanningScene use different RobotModels");
    return result;
  }
  if (!validateSafetyProfileModel(
      current_robot_state, joint_model_group, config_, result.failure_reason))
  {
    result.unsafe = true;
    return result;
  }
  if (!validateSegmentMotionState(
      current_motion_state, variable_count, result.failure_reason) ||
    !validateSegmentMotionState(candidate_motion_state, variable_count, result.failure_reason))
  {
    result.unsafe = true;
    return result;
  }

  const auto & variable_names = joint_model_group.getVariableNames();
  const auto protected_joint = std::find(
    variable_names.begin(), variable_names.end(), config_.protected_joint_name);
  if (protected_joint == variable_names.end()) {
    fail("protected joint '" + config_.protected_joint_name + "' is absent from the group");
    return result;
  }
  const Eigen::Index protected_index = static_cast<Eigen::Index>(
    std::distance(variable_names.begin(), protected_joint));

  moveit::core::RobotState sample_state(current_robot_state);
  sample_state.update(true);
  Eigen::VectorXd robot_positions(variable_count);
  sample_state.copyJointGroupPositions(&joint_model_group, robot_positions);
  if (!isFinite(robot_positions) ||
    (robot_positions - current_motion_state.position).cwiseAbs().maxCoeff() >
    kComparisonTolerance)
  {
    fail("RobotState and segment start positions disagree");
    return result;
  }

  const auto & collision_environment = scene.getCollisionEnvUnpadded();
  if (!collision_environment) {
    fail("PlanningScene has no unpadded collision environment");
    return result;
  }

  collision_detection::DistanceRequest distance_request;
  // Query one pair at a time. Safe pairs are temporarily allowed and the query is repeated,
  // because the globally closest invariant pair can have a much smaller robust threshold than
  // another pair and must not hide that pair.
  distance_request.enable_nearest_points = false;
  distance_request.enable_signed_distance = true;
  distance_request.compute_gradient = false;
  distance_request.type = collision_detection::DistanceRequestType::SINGLE;
  distance_request.max_contacts_per_body = 1U;
  distance_request.group_name = joint_model_group.getName();
  const Eigen::VectorXd inter_sample_joint_uncertainty =
    (candidate_motion_state.position - current_motion_state.position).cwiseAbs() /
    (2.0 * static_cast<double>(substeps));
  distance_request.distance_threshold = maximumRequiredClearance(
    config_, inter_sample_joint_uncertainty.maxCoeff());
  if (!isFinite(inter_sample_joint_uncertainty) ||
    !std::isfinite(distance_request.distance_threshold))
  {
    fail("segment robust-clearance threshold is non-finite");
    return result;
  }
  distance_request.enableGroup(scene.getRobotModel());

  result.input_valid = true;
  result.unsafe = false;
  result.samples.reserve(substeps + 1U);
  try {
    for (std::size_t sample_index = 0; sample_index <= substeps; ++sample_index) {
      const double fraction = static_cast<double>(sample_index) /
        static_cast<double>(substeps);
      const double start_weight = 1.0 - fraction;
      const Eigen::VectorXd positions =
        start_weight * current_motion_state.position +
        fraction * candidate_motion_state.position;
      const Eigen::VectorXd velocities =
        start_weight * current_motion_state.velocity +
        fraction * candidate_motion_state.velocity;
      const Eigen::VectorXd accelerations =
        start_weight * current_motion_state.acceleration +
        fraction * candidate_motion_state.acceleration;
      sample_state.setJointGroupPositions(&joint_model_group, positions);
      sample_state.setJointGroupVelocities(&joint_model_group, velocities);
      sample_state.setJointGroupAccelerations(&joint_model_group, accelerations);
      sample_state.update(true);

      CollisionSegmentSampleDiagnostic sample;
      sample.sample_index = sample_index;
      sample.interpolation_fraction = fraction;
      sample.protected_joint_position_rad = positions[protected_index];
      sample.protected_joint_corridor_violated =
        positions[protected_index] < config_.protected_joint_lower_rad ||
        positions[protected_index] > config_.protected_joint_upper_rad;

      collision_detection::AllowedCollisionMatrix sample_acm(
        scene.getAllowedCollisionMatrix());
      distance_request.acm = &sample_acm;
      std::set<std::string> temporarily_allowed_pairs;
      const auto query_and_record =
        [this, &sample, &sample_state, &sample_acm, &temporarily_allowed_pairs,
          &distance_request, &collision_environment, &inter_sample_joint_uncertainty,
          &joint_model_group, &result, &fail](
        const CollisionPairKind kind, double & minimum_distance) {
          constexpr std::size_t kMaximumPairQueries = 10000U;
          for (std::size_t query_index = 0; query_index < kMaximumPairQueries; ++query_index) {
            DistanceResult distance_result;
            if (kind == CollisionPairKind::kSelf) {
              collision_environment->distanceSelf(
                distance_request, distance_result, sample_state);
            } else {
              collision_environment->distanceRobot(
                distance_request, distance_result, sample_state);
            }

            const auto & reported = distance_result.minimum_distance;
            const bool has_first_body = !reported.link_names[0].empty();
            const bool has_second_body = !reported.link_names[1].empty();
            if (!has_first_body && !has_second_body) {
              const bool is_empty_sentinel =
                reported.distance == std::numeric_limits<double>::max() ||
                reported.distance == std::numeric_limits<double>::infinity();
              if (distance_result.collision || !distance_result.distances.empty() ||
                !is_empty_sentinel)
              {
                fail("collision backend returned an invalid unnamed segment result");
                return false;
              }
              return true;
            }
            if (!has_first_body || !has_second_body || !std::isfinite(reported.distance)) {
              fail("collision backend returned invalid segment minimum-distance data");
              return false;
            }

            const DistanceCandidate candidate = canonicalize(reported, kind);
            minimum_distance = std::min(minimum_distance, candidate.distance_m);
            if (candidate.distance_m < sample.minimum_distance_m) {
              sample.minimum_distance_m = candidate.distance_m;
              sample.closest_pair = readablePair(candidate);
            }
            sample.collision_detected = sample.collision_detected ||
              distance_result.collision || candidate.distance_m <= 0.0;
            sample.hard_clearance_violated = sample.hard_clearance_violated ||
              candidate.distance_m < config_.hard_clearance_m;
            const double pair_required_clearance = requiredClearance(
              candidate, config_, pairInterSampleUncertainty(
                candidate, sample_state, joint_model_group, inter_sample_joint_uncertainty));
            sample.robust_clearance_violated = sample.robust_clearance_violated ||
              candidate.distance_m < pair_required_clearance;
            recordClearanceFailure(candidate, pair_required_clearance, result.failure_reason);
            if (sample.collision_detected || sample.robust_clearance_violated) {
              return true;
            }
          // FCL SINGLE can still populate minimum_distance with the globally closest pair when
          // it lies outside distance_threshold. Every remaining pair is then farther away and,
          // because this threshold is the maximum pair-specific requirement, also robustly safe.
            if (candidate.distance_m >= distance_request.distance_threshold) {
              return true;
            }

            const std::string key = candidateKey(candidate);
            if (!temporarily_allowed_pairs.insert(key).second) {
              fail("collision backend repeated a temporarily allowed segment pair");
              return false;
            }
            sample_acm.setEntry(candidate.body_names[0], candidate.body_names[1], true);
          }
          fail("collision segment query exceeded the bounded pair count");
          return false;
        };

      if (!query_and_record(CollisionPairKind::kSelf, sample.minimum_self_distance_m) ||
        !query_and_record(CollisionPairKind::kWorld, sample.minimum_world_distance_m))
      {
        return result;
      }

      result.minimum_self_distance_m = std::min(
        result.minimum_self_distance_m, sample.minimum_self_distance_m);
      result.minimum_world_distance_m = std::min(
        result.minimum_world_distance_m, sample.minimum_world_distance_m);
      if (sample.minimum_distance_m < result.minimum_distance_m) {
        result.minimum_distance_m = sample.minimum_distance_m;
        result.closest_pair = sample.closest_pair;
      }
      result.robust_clearance_violated = result.robust_clearance_violated ||
        sample.robust_clearance_violated;
      const bool sample_unsafe = sample.collision_detected ||
        sample.robust_clearance_violated || sample.protected_joint_corridor_violated;
      if (sample_unsafe && !result.unsafe) {
        result.unsafe = true;
        result.first_unsafe_sample = sample_index;
        result.first_unsafe_fraction = fraction;
      }
      result.samples.push_back(std::move(sample));
      ++result.evaluated_samples;
    }
  } catch (const std::exception & exception) {
    fail(std::string("collision segment query failed: ") + exception.what());
  }
  return result;
}

CollisionSegmentValidationResult CollisionConstraintBuilder::validatePath(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & start_robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const std::vector<JointMotionState> & waypoints,
  const std::size_t substeps_per_segment) const
{
  CollisionSegmentValidationResult result;
  const Eigen::Index variable_count = static_cast<Eigen::Index>(
    joint_model_group.getVariableCount());
  const auto fail = [&result](const std::string & reason) {
      result.input_valid = false;
      result.unsafe = true;
      if (result.failure_reason.empty()) {
        result.failure_reason = reason;
      }
    };

  if (waypoints.size() < 2U) {
    fail("path validation requires at least two waypoints");
    return result;
  }
  if (substeps_per_segment < 2U) {
    fail("path validation requires at least two equal substeps per segment");
    return result;
  }
  const std::size_t segment_count = waypoints.size() - 1U;
  constexpr std::size_t kMaximumSize = std::numeric_limits<std::size_t>::max();
  if (segment_count > (kMaximumSize - 1U) / substeps_per_segment) {
    fail("path validation sample count overflows size_t");
    return result;
  }
  result.requested_substeps = segment_count * substeps_per_segment;
  if (result.requested_substeps >= result.samples.max_size()) {
    fail("path validation sample count exceeds vector capacity");
    return result;
  }
  if (variable_count <= 0) {
    fail("JointModelGroup has no variables");
    return result;
  }
  if (start_robot_state.getRobotModel().get() != scene.getRobotModel().get()) {
    fail("RobotState and PlanningScene use different RobotModels");
    return result;
  }
  if (!validateSafetyProfileModel(
      start_robot_state, joint_model_group, config_, result.failure_reason))
  {
    result.unsafe = true;
    return result;
  }
  for (const JointMotionState & waypoint : waypoints) {
    if (!validateSegmentMotionState(waypoint, variable_count, result.failure_reason)) {
      result.unsafe = true;
      return result;
    }
  }

  const auto & variable_names = joint_model_group.getVariableNames();
  const auto protected_joint = std::find(
    variable_names.begin(), variable_names.end(), config_.protected_joint_name);
  if (protected_joint == variable_names.end()) {
    fail("protected joint '" + config_.protected_joint_name + "' is absent from the group");
    return result;
  }
  const Eigen::Index protected_index = static_cast<Eigen::Index>(
    std::distance(variable_names.begin(), protected_joint));

  moveit::core::RobotState sample_state(start_robot_state);
  sample_state.update(true);
  Eigen::VectorXd robot_positions(variable_count);
  sample_state.copyJointGroupPositions(&joint_model_group, robot_positions);
  if (!isFinite(robot_positions) ||
    (robot_positions - waypoints.front().position).cwiseAbs().maxCoeff() >
    kComparisonTolerance)
  {
    fail("RobotState and path start positions disagree");
    return result;
  }

  const auto & collision_environment = scene.getCollisionEnvUnpadded();
  if (!collision_environment) {
    fail("PlanningScene has no unpadded collision environment");
    return result;
  }

  std::vector<Eigen::VectorXd> inter_sample_joint_uncertainties;
  try {
    inter_sample_joint_uncertainties.reserve(segment_count);
    for (std::size_t segment_index = 0; segment_index < segment_count; ++segment_index) {
      const Eigen::VectorXd uncertainty =
        (waypoints[segment_index + 1U].position - waypoints[segment_index].position)
        .cwiseAbs() /
        (2.0 * static_cast<double>(substeps_per_segment));
      if (!isFinite(uncertainty)) {
        fail("path robust-clearance uncertainty is non-finite");
        return result;
      }
      inter_sample_joint_uncertainties.push_back(uncertainty);
    }
    result.samples.reserve(result.requested_substeps + 1U);
  } catch (const std::exception & exception) {
    fail(std::string("failed to allocate path validation storage: ") + exception.what());
    return result;
  }

  collision_detection::DistanceRequest distance_request;
  distance_request.enable_nearest_points = false;
  distance_request.enable_signed_distance = true;
  distance_request.compute_gradient = false;
  distance_request.type = collision_detection::DistanceRequestType::SINGLE;
  distance_request.max_contacts_per_body = 1U;
  distance_request.group_name = joint_model_group.getName();
  distance_request.enableGroup(scene.getRobotModel());

  DistanceResult distance_result;
  const auto evaluate_sample =
    [this, &scene, &joint_model_group, &sample_state, &collision_environment,
      &distance_request, &distance_result, protected_index, &result, &fail](
    const Eigen::VectorXd & positions,
    const Eigen::VectorXd & inter_sample_joint_uncertainty,
    const std::size_t sample_index) {
      distance_request.distance_threshold = maximumRequiredClearance(
        config_, inter_sample_joint_uncertainty.maxCoeff());
      if (!std::isfinite(distance_request.distance_threshold)) {
        fail("path robust-clearance threshold is non-finite");
        return false;
      }

      sample_state.setJointGroupPositions(&joint_model_group, positions);
      sample_state.updateCollisionBodyTransforms();

      CollisionSegmentSampleDiagnostic sample;
      sample.sample_index = sample_index;
      sample.interpolation_fraction = static_cast<double>(sample_index) /
        static_cast<double>(result.requested_substeps);
      sample.protected_joint_position_rad = positions[protected_index];
      sample.protected_joint_corridor_violated =
        positions[protected_index] < config_.protected_joint_lower_rad ||
        positions[protected_index] > config_.protected_joint_upper_rad;

      collision_detection::AllowedCollisionMatrix sample_acm(
        scene.getAllowedCollisionMatrix());
      distance_request.acm = &sample_acm;
      std::set<std::string> temporarily_allowed_pairs;
      const auto query_and_record =
        [this, &sample, &sample_state, &sample_acm, &temporarily_allowed_pairs,
          &distance_request, &distance_result, &collision_environment,
          &inter_sample_joint_uncertainty, &joint_model_group, &result, &fail](
        const CollisionPairKind kind, double & minimum_distance) {
          constexpr std::size_t kMaximumPairQueries = 10000U;
          for (std::size_t query_index = 0; query_index < kMaximumPairQueries; ++query_index) {
            distance_result.clear();
            if (kind == CollisionPairKind::kSelf) {
              collision_environment->distanceSelf(
                distance_request, distance_result, sample_state);
            } else {
              collision_environment->distanceRobot(
                distance_request, distance_result, sample_state);
            }

            const auto & reported = distance_result.minimum_distance;
            const bool has_first_body = !reported.link_names[0].empty();
            const bool has_second_body = !reported.link_names[1].empty();
            if (!has_first_body && !has_second_body) {
              const bool is_empty_sentinel =
                reported.distance == std::numeric_limits<double>::max() ||
                reported.distance == std::numeric_limits<double>::infinity();
              if (distance_result.collision || !distance_result.distances.empty() ||
                !is_empty_sentinel)
              {
                fail("collision backend returned an invalid unnamed path result");
                return false;
              }
              return true;
            }
            if (!has_first_body || !has_second_body || !std::isfinite(reported.distance)) {
              fail("collision backend returned invalid path minimum-distance data");
              return false;
            }

            const DistanceCandidate candidate = canonicalize(reported, kind);
            minimum_distance = std::min(minimum_distance, candidate.distance_m);
            if (candidate.distance_m < sample.minimum_distance_m) {
              sample.minimum_distance_m = candidate.distance_m;
              sample.closest_pair = readablePair(candidate);
            }
            sample.collision_detected = sample.collision_detected ||
              distance_result.collision || candidate.distance_m <= 0.0;
            sample.hard_clearance_violated = sample.hard_clearance_violated ||
              candidate.distance_m < config_.hard_clearance_m;
            const double pair_required_clearance = requiredClearance(
              candidate, config_, pairInterSampleUncertainty(
                candidate, sample_state, joint_model_group, inter_sample_joint_uncertainty));
            sample.robust_clearance_violated = sample.robust_clearance_violated ||
              candidate.distance_m < pair_required_clearance;
            recordClearanceFailure(candidate, pair_required_clearance, result.failure_reason);
            if (sample.collision_detected || sample.robust_clearance_violated) {
              return true;
            }
            // FCL SINGLE can return the globally closest pair even when it lies outside the
            // requested threshold. All remaining pairs are then farther away and robustly safe.
            if (candidate.distance_m >= distance_request.distance_threshold) {
              return true;
            }

            const std::string key = candidateKey(candidate);
            if (!temporarily_allowed_pairs.insert(key).second) {
              fail("collision backend repeated a temporarily allowed path pair");
              return false;
            }
            sample_acm.setEntry(candidate.body_names[0], candidate.body_names[1], true);
          }
          fail("collision path query exceeded the bounded pair count");
          return false;
        };

      if (!query_and_record(CollisionPairKind::kSelf, sample.minimum_self_distance_m) ||
        !query_and_record(CollisionPairKind::kWorld, sample.minimum_world_distance_m))
      {
        return false;
      }

      result.minimum_self_distance_m = std::min(
        result.minimum_self_distance_m, sample.minimum_self_distance_m);
      result.minimum_world_distance_m = std::min(
        result.minimum_world_distance_m, sample.minimum_world_distance_m);
      if (sample.minimum_distance_m < result.minimum_distance_m) {
        result.minimum_distance_m = sample.minimum_distance_m;
        result.closest_pair = sample.closest_pair;
      }
      result.robust_clearance_violated = result.robust_clearance_violated ||
        sample.robust_clearance_violated;
      const bool sample_unsafe = sample.collision_detected ||
        sample.robust_clearance_violated || sample.protected_joint_corridor_violated;
      if (sample_unsafe && !result.unsafe) {
        result.unsafe = true;
        result.first_unsafe_sample = sample_index;
        result.first_unsafe_fraction = sample.interpolation_fraction;
      }
      result.samples.push_back(std::move(sample));
      ++result.evaluated_samples;
      return true;
    };

  result.input_valid = true;
  result.unsafe = false;
  try {
    if (!evaluate_sample(
        waypoints.front().position, inter_sample_joint_uncertainties.front(), 0U))
    {
      return result;
    }

    std::size_t sample_index = 0U;
    for (std::size_t segment_index = 0; segment_index < segment_count; ++segment_index) {
      const JointMotionState & start = waypoints[segment_index];
      const JointMotionState & end = waypoints[segment_index + 1U];
      for (std::size_t local_index = 1U; local_index <= substeps_per_segment; ++local_index) {
        ++sample_index;
        const double fraction = static_cast<double>(local_index) /
          static_cast<double>(substeps_per_segment);
        const Eigen::VectorXd positions =
          (1.0 - fraction) * start.position + fraction * end.position;
        Eigen::VectorXd uncertainty = inter_sample_joint_uncertainties[segment_index];
        if (local_index == substeps_per_segment && segment_index + 1U < segment_count) {
          uncertainty = uncertainty.cwiseMax(inter_sample_joint_uncertainties[segment_index + 1U]);
        }
        if (!evaluate_sample(positions, uncertainty, sample_index)) {
          return result;
        }
      }
    }
  } catch (const std::exception & exception) {
    fail(std::string("collision path query failed: ") + exception.what());
  }
  return result;
}

const CollisionConstraintConfig & CollisionConstraintBuilder::config() const noexcept
{
  return config_;
}

}  // namespace face_tracking_arm::control
