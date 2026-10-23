// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/collision_constraints.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <moveit/collision_detection/collision_env.hpp>
#include <moveit/collision_detection/collision_matrix.hpp>
#include <moveit/planning_scene/planning_scene.hpp>
#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_state/robot_state.hpp>

#include "collision_distance_query.hpp"
#include "collision_stopping.hpp"

namespace face_tracking_arm::control
{

using collision_detail::DistanceCandidate;
using collision_detail::appendDistanceCandidates;
using collision_detail::candidateKey;
using collision_detail::candidateLess;
using collision_detail::collisionBodyLink;
using collision_detail::isInvariantPair;
using collision_detail::maximumRequiredClearance;
using collision_detail::pairDistanceLipschitz;
using collision_detail::pairInterSampleUncertainty;
using collision_detail::readablePair;
using collision_detail::relativeMotionVariables;
using collision_detail::requiredClearance;
using collision_detail::validateCollisionModel;
using collision_detail::ReachableVelocityInterval;
using collision_detail::ReachableApproachInterval;
using collision_detail::computeReachableVelocityInterval;
using collision_detail::projectApproachInterval;
using collision_detail::requiredCollisionDistanceForApproachCap;
using collision_detail::validateMotionInput;
using collision_detail::isFinite;
using collision_detail::kComparisonTolerance;
using collision_detection::BodyType;
using collision_detection::DistanceResult;
namespace BodyTypes = collision_detection::BodyTypes;

namespace
{

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
  if (config.segment_validation_substeps < 2U) {
    throw std::invalid_argument("segment_validation_substeps must be at least two");
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
  if (!config.numerical_policy.has_value() || !config.numerical_policy->valid()) {
    throw std::invalid_argument("collision numerical policy is missing or invalid");
  }
}

bool sameMotionNumericalPolicy(
  const NumericalPolicy & left, const NumericalPolicy & right) noexcept
{
  return left.position_error.value == right.position_error.value &&
         left.velocity_error.value == right.velocity_error.value &&
         left.acceleration_error.value == right.acceleration_error.value &&
         left.jerk_error.value == right.jerk_error.value;
}

void setFailure(CollisionConstraintDiagnostics & diagnostics, const std::string & message)
{
  diagnostics.valid = false;
  if (diagnostics.failure_reason.empty()) {
    diagnostics.failure_reason = message;
  }
}

}  // namespace

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
  if (!validateCollisionModel(
      scene, robot_state, joint_model_group, config_, diagnostics.failure_reason))
  {
    return result;
  }
  if (!validateMotionInput(motion_state, motion_limits, variable_count,
      diagnostics.failure_reason))
  {
    return result;
  }
  if (!motion_limits.numerical_policy.has_value() ||
    !sameMotionNumericalPolicy(*motion_limits.numerical_policy, *config_.numerical_policy))
  {
    setFailure(diagnostics,
      "collision and motion limits must share one q/v/a/jerk numerical policy");
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

  // The queued next position is q + v_next * dt. Bound its segment validator's
  // half-substep uncertainty for every reachable next velocity; pair filtering below
  // removes joints that move both self-collision bodies together.
  Eigen::VectorXd next_segment_joint_uncertainty(variable_count);
  const double inter_sample_divisor =
    2.0 * static_cast<double>(config_.segment_validation_substeps);
  for (Eigen::Index index = 0; index < variable_count; ++index) {
    const double maximum_reachable_speed = std::max(
      std::abs(reachable_velocity.lower[index]),
      std::abs(reachable_velocity.upper[index]));
    next_segment_joint_uncertainty[index] =
      maximum_reachable_speed * period_sec / inter_sample_divisor;
  }
  if (!isFinite(next_segment_joint_uncertainty)) {
    setFailure(diagnostics, "next-segment collision uncertainty is non-finite");
    return result;
  }

  const double q = motion_state.position[protected_index];
  if (q > config_.protected_joint_upper_rad || q < config_.protected_joint_lower_rad) {
    diagnostics.protected_joint_corridor_violated = true;
  }

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
    const double next_segment_pair_uncertainty = pairInterSampleUncertainty(
      candidate, state, joint_model_group, next_segment_joint_uncertainty);
    pair_diagnostic.required_clearance_m = requiredClearance(
      candidate, config_, next_segment_pair_uncertainty);
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

    if (result.constraints.size() >= config_.maximum_constraint_rows) {
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

const CollisionConstraintConfig & CollisionConstraintBuilder::config() const noexcept
{
  return config_;
}

}  // namespace face_tracking_arm::control
