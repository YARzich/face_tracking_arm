// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/collision_constraints.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
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
using collision_detail::canonicalize;
using collision_detail::candidateKey;
using collision_detail::maximumRequiredClearance;
using collision_detail::readablePair;
using collision_detail::requiredClearance;
using collision_detail::pairInterSampleUncertainty;
using collision_detail::validateCollisionModel;
using collision_detail::validateSegmentMotionState;
using collision_detail::isFinite;
using collision_detail::kComparisonTolerance;
using collision_detection::DistanceResult;

namespace
{

namespace BodyTypes = collision_detection::BodyTypes;

enum class SampledMotionKind { kSegment, kPath };

const char * motionName(const SampledMotionKind kind)
{
  return kind == SampledMotionKind::kSegment ? "segment" : "path";
}

void failValidation(
  CollisionSegmentValidationResult & result, const std::string & reason,
  const SampledMotionKind kind)
{
  result.input_valid = false;
  result.unsafe = true;
  if (kind == SampledMotionKind::kSegment || result.failure_reason.empty()) {
    result.failure_reason = reason;
  }
}

collision_detection::DistanceRequest signedDistanceRequest(
  const planning_scene::PlanningScene & scene,
  const moveit::core::JointModelGroup & group)
{
  collision_detection::DistanceRequest request;
  request.enable_nearest_points = false;
  request.enable_signed_distance = true;
  request.compute_gradient = false;
  request.type = collision_detection::DistanceRequestType::ALL;
  request.max_contacts_per_body = 1U;
  request.group_name = group.getName();
  request.enableGroup(scene.getRobotModel());
  return request;
}

bool prepareSampleState(
  const planning_scene::PlanningScene & scene, moveit::core::RobotState & state,
  const moveit::core::JointModelGroup & group, const Eigen::VectorXd & start_position,
  const CollisionConstraintConfig & config, const SampledMotionKind kind,
  Eigen::Index & protected_index, std::string & error)
{
  const auto & names = group.getVariableNames();
  const auto protected_joint = std::find(names.begin(), names.end(), config.protected_joint_name);
  if (protected_joint == names.end()) {
    error = "protected joint '" + config.protected_joint_name + "' is absent from the group";
    return false;
  }
  protected_index = static_cast<Eigen::Index>(std::distance(names.begin(), protected_joint));
  state.update(true);
  Eigen::VectorXd positions(group.getVariableCount());
  state.copyJointGroupPositions(&group, positions);
  if (!isFinite(positions) ||
    (positions - start_position).cwiseAbs().maxCoeff() > kComparisonTolerance)
  {
    error = std::string("RobotState and ") + motionName(kind) + " start positions disagree";
    return false;
  }
  if (!scene.getCollisionEnvUnpadded()) {
    error = "PlanningScene has no unpadded collision environment";
    return false;
  }
  return true;
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

bool sampledCandidateLess(const DistanceCandidate & left, const DistanceCandidate & right)
{
  if (left.distance_m != right.distance_m) {
    return left.distance_m < right.distance_m;
  }
  return candidateKey(left) < candidateKey(right);
}

/// Queries all pairs in the robust-clearance envelope once per sample.
/// Segment/path callers retain ownership of interpolation and RobotState updates.
class SampledCollisionQuery
{
public:
  SampledCollisionQuery(
    const CollisionConstraintConfig & config, const planning_scene::PlanningScene & scene,
    const moveit::core::JointModelGroup & group, const moveit::core::RobotState & state,
    const Eigen::Index protected_index, const SampledMotionKind kind,
    CollisionSegmentValidationResult & result)
  : config_(config), scene_(scene), group_(group), state_(state),
    environment_(*scene.getCollisionEnvUnpadded()), protected_index_(protected_index),
    kind_(kind), result_(result), request_(signedDistanceRequest(scene, group))
  {
    request_.acm = &scene_.getAllowedCollisionMatrix();
  }

  SampledCollisionQuery(const SampledCollisionQuery &) = delete;
  SampledCollisionQuery & operator=(const SampledCollisionQuery &) = delete;

  bool evaluate(
    const Eigen::VectorXd & positions, const Eigen::VectorXd & joint_uncertainty,
    const std::size_t sample_index)
  {
    request_.distance_threshold = maximumRequiredClearance(config_, joint_uncertainty.maxCoeff());
    if (!std::isfinite(request_.distance_threshold)) {
      fail(std::string(motionName(kind_)) + " robust-clearance threshold is non-finite");
      return false;
    }
    CollisionSegmentSampleDiagnostic sample;
    sample.sample_index = sample_index;
    sample.interpolation_fraction = static_cast<double>(sample_index) /
      static_cast<double>(result_.requested_substeps);
    sample.protected_joint_position_rad = positions[protected_index_];
    const BoundaryClass protected_position_class = classifyPosition(
      positions[protected_index_],
      PositionInterval{config_.protected_joint_lower_rad, config_.protected_joint_upper_rad},
      config_.numerical_policy->position_error);
    sample.protected_joint_corridor_violated =
      protected_position_class != BoundaryClass::kInterior;
    if (!queryAndRecord(
        CollisionPairKind::kSelf, joint_uncertainty, sample, sample.minimum_self_distance_m) ||
      !queryAndRecord(
        CollisionPairKind::kWorld, joint_uncertainty, sample, sample.minimum_world_distance_m))
    {
      return false;
    }
    appendSample(std::move(sample));
    return true;
  }

private:
  void fail(const std::string & reason)
  {
    failValidation(result_, reason, kind_);
  }

  bool queryAndRecord(
    const CollisionPairKind kind, const Eigen::VectorXd & joint_uncertainty,
    CollisionSegmentSampleDiagnostic & sample, double & minimum_distance)
  {
    distance_result_.clear();
    if (kind == CollisionPairKind::kSelf) {
      ++result_.distance_query_count;
      environment_.distanceSelf(request_, distance_result_, state_);
    } else {
      ++result_.distance_query_count;
      environment_.distanceRobot(request_, distance_result_, state_);
    }

    std::map<std::string, DistanceCandidate> candidates_by_body_pair;
    for (const auto & [body_pair, records] : distance_result_.distances) {
      if (records.empty()) {
        fail(std::string("collision backend returned an empty ") + motionName(kind_) +
          " pair record");
        return false;
      }
      for (const auto & record : records) {
        const bool first_named = !record.link_names[0].empty();
        const bool second_named = !record.link_names[1].empty();
        if (!first_named || !second_named || !std::isfinite(record.distance) ||
          !(record.distance < request_.distance_threshold))
        {
          fail(std::string("collision backend returned invalid ") +
            motionName(kind_) + " pair-distance data");
          return false;
        }
        const bool first_world = record.body_types[0] == BodyTypes::WORLD_OBJECT;
        const bool second_world = record.body_types[1] == BodyTypes::WORLD_OBJECT;
        if ((kind == CollisionPairKind::kSelf && (first_world || second_world)) ||
          (kind == CollisionPairKind::kWorld && first_world == second_world))
        {
          fail(std::string("collision backend returned invalid ") +
            motionName(kind_) + " body types");
          return false;
        }
        const auto expected_pair = std::minmax(record.link_names[0], record.link_names[1]);
        if (body_pair.first != expected_pair.first || body_pair.second != expected_pair.second) {
          fail(std::string("collision backend returned inconsistent ") +
            motionName(kind_) + " pair names");
          return false;
        }
        DistanceCandidate candidate = canonicalize(record, kind);
        const std::string key = candidateKey(candidate);
        const auto [it, inserted] = candidates_by_body_pair.emplace(key, candidate);
        if (!inserted && candidate.distance_m < it->second.distance_m) {
          it->second = std::move(candidate);
        }
      }
    }

    const auto & reported = distance_result_.minimum_distance;
    const bool has_first_body = !reported.link_names[0].empty();
    const bool has_second_body = !reported.link_names[1].empty();
    if (candidates_by_body_pair.empty()) {
      const bool is_empty_sentinel =
        reported.distance == std::numeric_limits<double>::max() ||
        reported.distance == std::numeric_limits<double>::infinity();
      if (distance_result_.collision || has_first_body || has_second_body ||
        !is_empty_sentinel)
      {
        fail(std::string("collision backend returned an invalid unnamed ") +
          motionName(kind_) + " result");
        return false;
      }
      return true;
    }
    if (!has_first_body || !has_second_body || !std::isfinite(reported.distance)) {
      fail(std::string("collision backend returned invalid ") +
        motionName(kind_) + " minimum-distance data");
      return false;
    }

    std::vector<DistanceCandidate> candidates;
    candidates.reserve(candidates_by_body_pair.size());
    for (auto & [unused_key, candidate] : candidates_by_body_pair) {
      (void)unused_key;
      candidates.push_back(std::move(candidate));
    }
    std::sort(candidates.begin(), candidates.end(), sampledCandidateLess);
    const DistanceCandidate reported_minimum = canonicalize(reported, kind);
    const auto matches_reported_minimum = std::find_if(
      candidates.begin(), candidates.end(),
      [&reported_minimum](const DistanceCandidate & candidate) {
        return std::abs(candidate.distance_m - reported_minimum.distance_m) <= 1.0e-9 &&
               collision_detail::candidateKey(candidate) ==
               collision_detail::candidateKey(reported_minimum);
      });
    if (matches_reported_minimum == candidates.end()) {
      fail(std::string("collision backend returned inconsistent ") +
        motionName(kind_) + " minimum-distance data");
      return false;
    }

    bool has_negative_distance = false;
    for (const DistanceCandidate & candidate : candidates) {
      minimum_distance = std::min(minimum_distance, candidate.distance_m);
      if (candidate.distance_m < sample.minimum_distance_m) {
        sample.minimum_distance_m = candidate.distance_m;
        sample.closest_pair = readablePair(candidate);
      }
      has_negative_distance = has_negative_distance || candidate.distance_m <= 0.0;
      sample.collision_detected = sample.collision_detected || candidate.distance_m <= 0.0;
      sample.hard_clearance_violated = sample.hard_clearance_violated ||
        candidate.distance_m < config_.hard_clearance_m;
      const double required_clearance = requiredClearance(
        candidate, config_,
        pairInterSampleUncertainty(candidate, state_, group_, joint_uncertainty));
      sample.robust_clearance_violated = sample.robust_clearance_violated ||
        candidate.distance_m < required_clearance;
      recordClearanceFailure(candidate, required_clearance, result_.failure_reason);
    }
    if (distance_result_.collision != has_negative_distance) {
      fail(std::string("collision backend returned inconsistent ") +
        motionName(kind_) + " collision flag");
      return false;
    }
    sample.collision_detected = sample.collision_detected || distance_result_.collision;
    return true;
  }

  void appendSample(CollisionSegmentSampleDiagnostic sample)
  {
    result_.minimum_self_distance_m = std::min(
      result_.minimum_self_distance_m, sample.minimum_self_distance_m);
    result_.minimum_world_distance_m = std::min(
      result_.minimum_world_distance_m, sample.minimum_world_distance_m);
    if (sample.minimum_distance_m < result_.minimum_distance_m) {
      result_.minimum_distance_m = sample.minimum_distance_m;
      result_.closest_pair = sample.closest_pair;
    }
    result_.robust_clearance_violated = result_.robust_clearance_violated ||
      sample.robust_clearance_violated;
    const bool unsafe = sample.collision_detected || sample.robust_clearance_violated ||
      sample.protected_joint_corridor_violated;
    if (unsafe && !result_.unsafe) {
      result_.unsafe = true;
      result_.first_unsafe_sample = sample.sample_index;
      result_.first_unsafe_fraction = sample.interpolation_fraction;
    }
    result_.samples.push_back(std::move(sample));
    ++result_.evaluated_samples;
  }

  const CollisionConstraintConfig & config_;
  const planning_scene::PlanningScene & scene_;
  const moveit::core::JointModelGroup & group_;
  const moveit::core::RobotState & state_;
  const collision_detection::CollisionEnv & environment_;
  Eigen::Index protected_index_;
  SampledMotionKind kind_;
  CollisionSegmentValidationResult & result_;
  collision_detection::DistanceRequest request_;
  DistanceResult distance_result_;
};

}  // namespace

CollisionActualStateValidationResult CollisionConstraintBuilder::validateActualState(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const ActualPositionEncoding position_encoding) const
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
  if (!validateCollisionModel(
      scene, robot_state, joint_model_group, config_, result.failure_reason))
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
  const PositionInterval protected_interval{
    config_.protected_joint_lower_rad, config_.protected_joint_upper_rad};
  if (position_encoding == ActualPositionEncoding::kSdkFloatPromotedToDouble) {
    result.protected_joint_boundary_class = classifyMeasuredPosition(
      positions[protected_index], protected_interval);
    result.protected_joint_corridor_violated =
      result.protected_joint_boundary_class != BoundaryClass::kInterior;
  } else {
    result.protected_joint_boundary_class = classifyPosition(
      positions[protected_index], protected_interval,
      kControllerPositionFeedbackTolerance);
    result.protected_joint_corridor_violated =
      result.protected_joint_boundary_class == BoundaryClass::kOutside ||
      result.protected_joint_boundary_class == BoundaryClass::kInvalid;
  }

  const auto & collision_environment = scene.getCollisionEnvUnpadded();
  if (!collision_environment) {
    fail("PlanningScene has no unpadded collision environment");
    return result;
  }

  moveit::core::RobotState state(robot_state);
  state.update(true);

  auto distance_request = signedDistanceRequest(scene, joint_model_group);
  // This single-state check only needs its nearest pair; the ALL map is reserved
  // for segment samples where non-nearest unsafe pairs must also be evaluated.
  distance_request.type = collision_detection::DistanceRequestType::SINGLE;
  distance_request.distance_threshold = config_.hard_clearance_m;
  distance_request.acm = &scene.getAllowedCollisionMatrix();

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
  const PositionInterval protected_interval{
    config_.protected_joint_lower_rad, config_.protected_joint_upper_rad};
  const BoundaryClass lower_class = classifyPosition(
    std::min(start_position[protected_index], end_position[protected_index]),
    protected_interval, config_.numerical_policy->position_error);
  const BoundaryClass upper_class = classifyPosition(
    std::max(start_position[protected_index], end_position[protected_index]),
    protected_interval, config_.numerical_policy->position_error);
  if (lower_class != BoundaryClass::kInterior || upper_class != BoundaryClass::kInterior) {
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

bool CollisionConstraintBuilder::canCertifyPath(
  const CollisionConstraintDiagnostics & diagnostics,
  const std::vector<JointMotionState> & waypoints) const
{
  const Eigen::Index count = diagnostics.query_joint_position.size();
  if (waypoints.size() < 2U || count <= 0) {
    return false;
  }
  Eigen::VectorXd lower = waypoints.front().position;
  Eigen::VectorXd upper = waypoints.front().position;
  if (lower.size() != count || !isFinite(lower)) {
    return false;
  }
  for (const auto & waypoint : waypoints) {
    if (waypoint.position.size() != count || !isFinite(waypoint.position)) {
      return false;
    }
    lower = lower.cwiseMin(waypoint.position);
    upper = upper.cwiseMax(waypoint.position);
  }
  // Every interpolated segment lies inside this axis-aligned envelope. Certifying
  // the envelope is conservative and avoids one collision query per tail waypoint.
  return canCertifySegment(diagnostics, lower, upper);
}

CollisionSegmentValidationResult CollisionConstraintBuilder::validatePose(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & robot_state,
  const moveit::core::JointModelGroup & joint_model_group,
  const JointMotionState & pose) const
{
  CollisionSegmentValidationResult result;
  result.requested_substeps = 1U;
  const Eigen::Index variable_count = static_cast<Eigen::Index>(
    joint_model_group.getVariableCount());

  const auto fail = [&result](const std::string & reason) {
      failValidation(result, reason, SampledMotionKind::kSegment);
    };
  if (!validateCollisionModel(
      scene, robot_state, joint_model_group, config_, result.failure_reason))
  {
    result.unsafe = true;
    return result;
  }
  if (!validateSegmentMotionState(pose, variable_count, result.failure_reason)) {
    result.unsafe = true;
    return result;
  }

  moveit::core::RobotState sample_state(robot_state);
  Eigen::Index protected_index = 0;
  if (!prepareSampleState(
      scene, sample_state, joint_model_group, pose.position, config_,
      SampledMotionKind::kSegment, protected_index, result.failure_reason))
  {
    return result;
  }

  const Eigen::VectorXd no_inter_sample_uncertainty = Eigen::VectorXd::Zero(variable_count);
  SampledCollisionQuery query(
    config_, scene, joint_model_group, sample_state, protected_index,
    SampledMotionKind::kSegment, result);
  try {
    sample_state.setJointGroupPositions(&joint_model_group, pose.position);
    sample_state.setJointGroupVelocities(&joint_model_group, pose.velocity);
    sample_state.setJointGroupAccelerations(&joint_model_group, pose.acceleration);
    sample_state.update(true);
    result.input_valid = true;
    result.unsafe = false;
    if (!query.evaluate(pose.position, no_inter_sample_uncertainty, 0U)) {
      return result;
    }
  } catch (const std::exception & exception) {
    fail(std::string("collision pose query failed: ") + exception.what());
  }
  return result;
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
      failValidation(result, reason, SampledMotionKind::kSegment);
    };
  if (substeps < 2U) {
    fail("segment validation requires at least two equal substeps");
    return result;
  }
  if (substeps == std::numeric_limits<std::size_t>::max()) {
    fail("segment validation sample count overflows size_t");
    return result;
  }
  if (substeps >= result.samples.max_size()) {
    fail("segment validation sample count exceeds vector capacity");
    return result;
  }
  if (!validateCollisionModel(
      scene, current_robot_state, joint_model_group, config_, result.failure_reason))
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

  moveit::core::RobotState sample_state(current_robot_state);
  Eigen::Index protected_index = 0;
  if (!prepareSampleState(
      scene, sample_state, joint_model_group, current_motion_state.position,
      config_, SampledMotionKind::kSegment, protected_index, result.failure_reason))
  {
    return result;
  }

  const Eigen::VectorXd inter_sample_joint_uncertainty =
    (candidate_motion_state.position - current_motion_state.position).cwiseAbs() /
    (2.0 * static_cast<double>(substeps));
  const double distance_threshold = maximumRequiredClearance(
    config_, inter_sample_joint_uncertainty.maxCoeff());
  if (!isFinite(inter_sample_joint_uncertainty) ||
    !std::isfinite(distance_threshold))
  {
    fail("segment robust-clearance threshold is non-finite");
    return result;
  }
  SampledCollisionQuery query(
    config_, scene, joint_model_group, sample_state, protected_index,
    SampledMotionKind::kSegment, result);

  try {
    result.samples.reserve(substeps + 1U);
  } catch (const std::exception & exception) {
    fail(std::string("failed to allocate segment validation storage: ") + exception.what());
    return result;
  }
  result.input_valid = true;
  result.unsafe = false;
  Eigen::VectorXd positions(variable_count);
  try {
    for (std::size_t sample_index = 0; sample_index <= substeps; ++sample_index) {
      const double fraction = static_cast<double>(sample_index) /
        static_cast<double>(substeps);
      const double start_weight = 1.0 - fraction;
      positions.noalias() = start_weight * current_motion_state.position +
        fraction * candidate_motion_state.position;
      sample_state.setJointGroupPositions(&joint_model_group, positions);
      sample_state.update(true);

      if (!query.evaluate(positions, inter_sample_joint_uncertainty, sample_index)) {
        return result;
      }
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
      failValidation(result, reason, SampledMotionKind::kPath);
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
  if (!validateCollisionModel(
      scene, start_robot_state, joint_model_group, config_, result.failure_reason))
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

  moveit::core::RobotState sample_state(start_robot_state);
  Eigen::Index protected_index = 0;
  if (!prepareSampleState(
      scene, sample_state, joint_model_group, waypoints.front().position,
      config_, SampledMotionKind::kPath, protected_index, result.failure_reason))
  {
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

  result.input_valid = true;
  result.unsafe = false;
  std::vector<bool> proven_dense_samples;
  try {
    proven_dense_samples.assign(result.requested_substeps + 1U, false);
  } catch (const std::exception & exception) {
    fail(std::string("failed to allocate path certification storage: ") + exception.what());
    return result;
  }

  const auto merge_observed_query = [&result](
    CollisionSegmentValidationResult & query_result) {
      result.distance_query_count += query_result.distance_query_count;
      result.evaluated_samples += query_result.evaluated_samples;
      for (auto & sample : query_result.samples) {
        result.minimum_self_distance_m = std::min(
          result.minimum_self_distance_m, sample.minimum_self_distance_m);
        result.minimum_world_distance_m = std::min(
          result.minimum_world_distance_m, sample.minimum_world_distance_m);
        if (sample.minimum_distance_m < result.minimum_distance_m) {
          result.minimum_distance_m = sample.minimum_distance_m;
          result.closest_pair = sample.closest_pair;
        }
        result.samples.push_back(std::move(sample));
      }
    };

  const auto validate_dense_segment = [
    this, &result, &scene, &sample_state, &joint_model_group, protected_index,
    &waypoints, segment_count, substeps_per_segment, &inter_sample_joint_uncertainties,
    &proven_dense_samples](const std::size_t segment_index) {
      SampledCollisionQuery query(
        config_, scene, joint_model_group, sample_state, protected_index,
        SampledMotionKind::kPath, result);
      const JointMotionState & start = waypoints[segment_index];
      const JointMotionState & end = waypoints[segment_index + 1U];
      for (std::size_t local_index = 0U; local_index <= substeps_per_segment; ++local_index) {
        const std::size_t sample_index = segment_index * substeps_per_segment + local_index;
        if (proven_dense_samples[sample_index]) {
          continue;
        }
        const double fraction = static_cast<double>(local_index) /
          static_cast<double>(substeps_per_segment);
        const Eigen::VectorXd positions =
          (1.0 - fraction) * start.position + fraction * end.position;
        Eigen::VectorXd uncertainty = inter_sample_joint_uncertainties[segment_index];
        if (local_index == 0U && segment_index > 0U) {
          uncertainty = uncertainty.cwiseMax(
            inter_sample_joint_uncertainties[segment_index - 1U]);
        }
        if (local_index == substeps_per_segment && segment_index + 1U < segment_count) {
          uncertainty = uncertainty.cwiseMax(
            inter_sample_joint_uncertainties[segment_index + 1U]);
        }
        sample_state.setJointGroupPositions(&joint_model_group, positions);
        sample_state.updateCollisionBodyTransforms();
        if (!query.evaluate(positions, uncertainty, sample_index)) {
          return false;
        }
        proven_dense_samples[sample_index] = true;
        if (result.unsafe) {
          return false;
        }
      }
      return true;
    };

  const auto certify_or_split = [
    this, &result, &scene, &sample_state, &joint_model_group, protected_index,
    variable_count, &fail,
    &waypoints, segment_count, substeps_per_segment, &inter_sample_joint_uncertainties,
    &proven_dense_samples, &merge_observed_query, &validate_dense_segment](
    auto && self, const std::size_t first_waypoint, const std::size_t last_waypoint) -> bool {
      const std::size_t middle_waypoint = first_waypoint +
        (last_waypoint - first_waypoint) / 2U;
      const std::size_t anchor_sample_index = middle_waypoint * substeps_per_segment;
      Eigen::VectorXd block_uncertainty = Eigen::VectorXd::Zero(variable_count);
      const Eigen::VectorXd & anchor_position = waypoints[middle_waypoint].position;
      for (std::size_t waypoint_index = first_waypoint;
        waypoint_index <= last_waypoint; ++waypoint_index)
      {
        block_uncertainty = block_uncertainty.cwiseMax(
          (waypoints[waypoint_index].position - anchor_position).cwiseAbs());
      }
      // Preserve the old dense oracle's half-cell reserve, including both neighbors
      // at shared endpoints. Add it to the path envelope so the block proves at least
      // as much as every prior dense sample could prove.
      const std::size_t first_reserved_segment = first_waypoint > 0U ?
        first_waypoint - 1U : first_waypoint;
      const std::size_t last_reserved_segment = std::min(last_waypoint, segment_count - 1U);
      Eigen::VectorXd dense_inter_sample_reserve = Eigen::VectorXd::Zero(variable_count);
      for (std::size_t segment_index = first_reserved_segment;
        segment_index <= last_reserved_segment; ++segment_index)
      {
        dense_inter_sample_reserve = dense_inter_sample_reserve.cwiseMax(
          inter_sample_joint_uncertainties[segment_index]);
      }
      block_uncertainty += dense_inter_sample_reserve;

      const Eigen::Index protected_index_eigen = static_cast<Eigen::Index>(protected_index);
      bool protected_corridor_safe = true;
      for (std::size_t waypoint_index = first_waypoint;
        waypoint_index <= last_waypoint; ++waypoint_index)
      {
        const double protected_position = waypoints[waypoint_index].position[protected_index_eigen];
        const BoundaryClass protected_class = classifyPosition(
          protected_position,
          PositionInterval{config_.protected_joint_lower_rad,
            config_.protected_joint_upper_rad},
          config_.numerical_policy->position_error);
        protected_corridor_safe = protected_corridor_safe &&
          protected_class == BoundaryClass::kInterior;
      }

      bool block_certified = false;
      if (protected_corridor_safe) {
        CollisionSegmentValidationResult attempt_result;
        attempt_result.requested_substeps = result.requested_substeps;
        attempt_result.input_valid = true;
        attempt_result.unsafe = false;
        SampledCollisionQuery query(
          config_, scene, joint_model_group, sample_state, protected_index,
          SampledMotionKind::kPath, attempt_result);
        sample_state.setJointGroupPositions(&joint_model_group, anchor_position);
        sample_state.updateCollisionBodyTransforms();
        (void)query.evaluate(anchor_position, block_uncertainty, anchor_sample_index);
        block_certified = attempt_result.input_valid && !attempt_result.unsafe;
        const std::size_t block_end_sample_index = last_waypoint * substeps_per_segment;
        for (auto & sample : attempt_result.samples) {
          sample.path_block_certificate_attempt = true;
          sample.path_block_certified = block_certified;
          sample.path_block_end_sample_index = block_end_sample_index;
        }
        merge_observed_query(attempt_result);
        if (!attempt_result.input_valid) {
          fail(attempt_result.failure_reason.empty() ?
            "path block collision query failed" : attempt_result.failure_reason);
          return false;
        }
        if (block_certified) {
          ++result.certified_path_block_count;
          for (std::size_t sample_index = first_waypoint * substeps_per_segment;
            sample_index <= block_end_sample_index; ++sample_index)
          {
            proven_dense_samples[sample_index] = true;
          }
          return true;
        }
      }

      ++result.inconclusive_path_block_count;
      if (last_waypoint - first_waypoint == 1U) {
        // This is the irreducible leaf: retain the former dense sampling and the
        // globally computed shared-endpoint uncertainty from both adjacent segments.
        return validate_dense_segment(first_waypoint);
      }
      // Search left first so any reported unsafe sample remains the first one on the
      // original dense grid. The speculative block result never poisons global safety.
      if (!self(self, first_waypoint, middle_waypoint)) {
        return false;
      }
      return self(self, middle_waypoint, last_waypoint);
    };

  try {
    if (!certify_or_split(certify_or_split, 0U, segment_count)) {
      return result;
    }
  } catch (const std::exception & exception) {
    fail(std::string("collision path query failed: ") + exception.what());
  }
  return result;
}

}  // namespace face_tracking_arm::control
