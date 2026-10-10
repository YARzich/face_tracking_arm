// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_distance_query.hpp"

#include <algorithm>
#include <tuple>
#include <utility>

#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/attached_body.hpp>
#include <moveit/robot_state/robot_state.hpp>

namespace face_tracking_arm::control::collision_detail
{

using collision_detection::BodyType;
using collision_detection::DistanceResult;
using collision_detection::DistanceResultsData;
namespace BodyTypes = collision_detection::BodyTypes;

namespace
{

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

[[nodiscard]] bool canonicalLess(
  const BodyType left_type, const std::string & left_name,
  const BodyType right_type, const std::string & right_name)
{
  return std::make_tuple(bodyTypeRank(left_type), left_name) <
         std::make_tuple(bodyTypeRank(right_type), right_name);
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

[[nodiscard]] bool isMonitorNearPair(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config)
{
  return candidate.kind == CollisionPairKind::kSelf &&
         isNamedPair(candidate, config.monitor_link_name, config.monitor_near_link_name);
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

}  // namespace

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

[[nodiscard]] bool isInvariantPair(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config)
{
  return candidate.kind == CollisionPairKind::kSelf &&
         isNamedPair(
    candidate, config.monitor_link_name,
    config.invariant_mount_neighbor_name);
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
  const double additional_joint_position_uncertainty_rad)
{
  const double effective_joint_position_uncertainty =
    config.tracking_position_error_bound_rad + additional_joint_position_uncertainty_rad;
  return config.hard_clearance_m + config.numerical_distance_reserve_m +
         pairDistanceLipschitz(candidate, config) * effective_joint_position_uncertainty;
}

[[nodiscard]] double maximumRequiredClearance(
  const CollisionConstraintConfig & config,
  const double additional_joint_position_uncertainty_rad)
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

[[nodiscard]] bool validateCollisionModel(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const CollisionConstraintConfig & config,
  std::string & error)
{
  if (joint_model_group.getVariableCount() == 0U) {
    error = "JointModelGroup has no variables";
    return false;
  }
  if (robot_state.getRobotModel().get() != scene.getRobotModel().get()) {
    error = "RobotState and PlanningScene use different RobotModels";
    return false;
  }
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

}  // namespace face_tracking_arm::control::collision_detail
