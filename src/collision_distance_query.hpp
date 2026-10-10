// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef COLLISION_DISTANCE_QUERY_HPP_
#define COLLISION_DISTANCE_QUERY_HPP_

#include <Eigen/Core>

#include <limits>
#include <string>
#include <vector>

#include <moveit/collision_detection/collision_common.hpp>
#include <moveit/robot_model/link_model.hpp>

#include "face_tracking_arm/collision_constraints.hpp"

namespace face_tracking_arm::control::collision_detail
{

/// Canonical body order keeps pair keys, gradients and diagnostics deterministic.
struct DistanceCandidate
{
  CollisionPairKind kind{CollisionPairKind::kSelf};
  collision_detection::BodyType body_types[2]{
    collision_detection::BodyTypes::WORLD_OBJECT, collision_detection::BodyTypes::WORLD_OBJECT};
  std::string body_names[2];
  Eigen::Vector3d nearest_points[2]{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::Zero()};
  double distance_m{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] DistanceCandidate canonicalize(
  const collision_detection::DistanceResultsData & data, const CollisionPairKind kind);

[[nodiscard]] std::string candidateKey(const DistanceCandidate & candidate);

[[nodiscard]] std::string readablePair(const DistanceCandidate & candidate);

[[nodiscard]] bool candidateLess(
  const DistanceCandidate & left, const DistanceCandidate & right);

void appendDistanceCandidates(
  const collision_detection::DistanceResult & distance_result,
  const CollisionPairKind kind,
  std::vector<DistanceCandidate> & candidates);

[[nodiscard]] bool isInvariantPair(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config);

[[nodiscard]] double pairDistanceLipschitz(
  const DistanceCandidate & candidate, const CollisionConstraintConfig & config);

[[nodiscard]] double requiredClearance(
  const DistanceCandidate & candidate,
  const CollisionConstraintConfig & config,
  const double additional_joint_position_uncertainty_rad = 0.0);

[[nodiscard]] double maximumRequiredClearance(
  const CollisionConstraintConfig & config,
  const double additional_joint_position_uncertainty_rad = 0.0);

[[nodiscard]] const moveit::core::LinkModel * collisionBodyLink(
  const moveit::core::RobotState & state, const collision_detection::BodyType type,
  const std::string & name);

[[nodiscard]] std::vector<bool> relativeMotionVariables(
  const DistanceCandidate & candidate,
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group);

[[nodiscard]] double pairInterSampleUncertainty(
  const DistanceCandidate & candidate,
  const moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group,
  const Eigen::VectorXd & joint_uncertainty);

[[nodiscard]] bool validateCollisionModel(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const CollisionConstraintConfig & config,
  std::string & error);

}  // namespace face_tracking_arm::control::collision_detail

#endif  // COLLISION_DISTANCE_QUERY_HPP_
