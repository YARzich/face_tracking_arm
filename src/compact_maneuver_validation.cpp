// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "compact_maneuver_validation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include <moveit_msgs/msg/planning_scene.hpp>

#include "face_tracking_arm/compact_upright.hpp"

namespace face_tracking_arm::control::detail
{
namespace
{

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

}  // namespace

CompactManeuverValidation::CompactManeuverValidation(
  moveit::core::RobotModelConstPtr robot_model,
  const moveit::core::JointModelGroup * group,
  const moveit::core::LinkModel * command_link,
  const JointMotionLimits & limits,
  const CollisionConstraintBuilder & collision_validator,
  std::shared_ptr<const CompactGeometryValidator> compact_geometry_validator,
  const std::size_t segment_validation_substeps)
: robot_model_(std::move(robot_model)), group_(group), command_link_(command_link),
  limits_(limits), collision_validator_(collision_validator),
  compact_geometry_validator_(std::move(compact_geometry_validator)),
  segment_validation_substeps_(segment_validation_substeps)
{}

CompactProfileFinalGeometryValidator
CompactManeuverValidation::makeCompactProfileFinalGeometryValidator(
  const planning_scene::PlanningScene & scene,
  const moveit::core::RobotState & state,
  const std::uint64_t scene_revision, const double radius_cap_m,
  std::optional<CompactReturnRegion> return_region) const
{
  moveit_msgs::msg::PlanningScene scene_message;
  scene.getPlanningSceneMsg(scene_message);
  auto scene_snapshot = std::make_shared<planning_scene::PlanningScene>(robot_model_);
  if (!scene_snapshot->setPlanningSceneMsg(scene_message)) {
    return [scene_revision](const CompactJointProfile &) {
             CompactProfileGeometryCertificate certificate;
             certificate.status = CompactProfileGeometryStatus::kInvalidInput;
             certificate.scene_revision = scene_revision;
             certificate.failure_reason =
               "could not serialize an immutable planning-scene snapshot";
             return certificate;
           };
  }

  const auto geometry = compact_geometry_validator_;
  const auto collision_validator = collision_validator_;
  const auto model = robot_model_;
  const auto * group = group_;
  const auto * command_link = command_link_;
  const auto * shoulder_link = robot_model_->getLinkModel("link2");
  const auto * elbow_link = robot_model_->getLinkModel("link3");
  const auto * wrist_link = robot_model_->getLinkModel("link4");
  const auto & joint_names = group_->getVariableNames();
  const auto joint1_it = std::find(joint_names.begin(), joint_names.end(), "joint1");
  const auto joint3_it = std::find(joint_names.begin(), joint_names.end(), "joint3");
  const Eigen::Index joint1_index = joint1_it == joint_names.end() ? -1 :
    std::distance(joint_names.begin(), joint1_it);
  const Eigen::Index joint3_index = joint3_it == joint_names.end() ? -1 :
    std::distance(joint_names.begin(), joint3_it);
  moveit::core::RobotState reference_state(state);
  const double radius_limit = radius_cap_m + 1.0e-6;
  const std::size_t collision_substeps = segment_validation_substeps_;
  const double maximum_geometry_step_rad = 0.02;

  return [scene_snapshot = std::move(scene_snapshot), geometry, collision_validator,
           model, group, command_link, shoulder_link, elbow_link, wrist_link,
           joint1_index, joint3_index, return_region = std::move(return_region),
           reference_state = std::move(reference_state), scene_revision,
           radius_limit, collision_substeps, maximum_geometry_step_rad](
    const CompactJointProfile & profile) {
           CompactProfileGeometryCertificate certificate;
           certificate.scene_revision = scene_revision;
           certificate.checked_samples = profile.samples.size();
           certificate.checked_intervals = profile.samples.empty() ? 0U : profile.samples.size() -
             1U;
           const auto fail = [&certificate](
             const CompactProfileGeometryStatus status, const std::string & reason) {
               certificate.status = status;
               certificate.failure_reason = reason;
               return certificate;
             };
           if (!scene_snapshot || !geometry || !model || group == nullptr ||
             profile.samples.empty() || collision_substeps == 0U ||
             !std::isfinite(radius_limit) || radius_limit < 0.0)
           {
             return fail(CompactProfileGeometryStatus::kInvalidInput,
          "final compact geometry validator input is incomplete");
           }

           const auto validate_position = [&](const Eigen::VectorXd & position,
             std::string * failure_reason) {
               if (position.size() != static_cast<Eigen::Index>(group->getVariableCount()) ||
                 !position.allFinite())
               {
                 if (failure_reason != nullptr) {
                   *failure_reason = "joint position dimensions or values are invalid";
                 }
                 return false;
               }
               moveit::core::RobotState candidate(reference_state);
               candidate.setJointGroupPositions(group, position);
               candidate.update(true);
               const double radius = geometry->maximumAssemblyRadius(candidate);
               if (!std::isfinite(radius) || radius > radius_limit) {
                 if (failure_reason != nullptr) {
                   *failure_reason = "radius=" + jsonFiniteNumber(radius) +
                     " exceeds limit=" + jsonFiniteNumber(radius_limit);
                 }
                 return false;
               }
               return true;
             };

           std::vector<JointMotionState> waypoints;
           waypoints.reserve(profile.samples.size());
           for (std::size_t index = 0U; index < profile.samples.size(); ++index) {
             const auto & sample = profile.samples[index];
             std::string violation;
             if (!validate_position(sample.position, &violation)) {
               return fail(CompactProfileGeometryStatus::kUnsafe,
            "final compact profile sample violates radius limit at index " +
            std::to_string(index) + ": " + violation);
             }
             if (index > 0U) {
               const auto & previous = profile.samples[index - 1U];
               const double span = (sample.position - previous.position).
                 lpNorm<Eigen::Infinity>();
               const double count = std::max(1.0, std::ceil(span / maximum_geometry_step_rad));
               if (!std::isfinite(count) || count > 4096.0) {
                 return fail(CompactProfileGeometryStatus::kInvalidInput,
              "final compact profile geometry interval exceeds validation bound");
               }
               const std::size_t steps = static_cast<std::size_t>(count);
               for (std::size_t step = 1U; step < steps; ++step) {
                 const double fraction = static_cast<double>(step) / static_cast<double>(steps);
                 violation.clear();
                 if (!validate_position(previous.position + fraction *
                   (sample.position - previous.position), &violation))
                 {
                   return fail(CompactProfileGeometryStatus::kUnsafe,
                "final compact profile interval violates radius limit: " +
                violation);
                 }
               }
             }
             waypoints.push_back(sample);
           }

           if (return_region) {
             if (command_link == nullptr || shoulder_link == nullptr || elbow_link == nullptr ||
               wrist_link == nullptr || joint1_index < 0 || joint3_index < 0)
             {
               return fail(CompactProfileGeometryStatus::kInvalidInput,
                 "compact return posture links or joints are unavailable");
             }
             moveit::core::RobotState endpoint(reference_state);
             endpoint.setJointGroupPositions(group, waypoints.back().position);
             endpoint.update(true);
             const auto & screen_pose = endpoint.getGlobalLinkTransform(command_link);
             const bool inside_region = compactReturnRegionContains(
               *return_region, screen_pose.translation(), screen_pose.linear(),
               endpoint.getGlobalLinkTransform(shoulder_link).translation(),
               endpoint.getGlobalLinkTransform(elbow_link).translation(),
               endpoint.getGlobalLinkTransform(wrist_link).translation(),
               waypoints.back().position[joint1_index],
               waypoints.back().position[joint3_index]);
             if (!inside_region) {
               return fail(CompactProfileGeometryStatus::kUnsafe,
                 "final compact return profile endpoint is outside the frozen region");
             }
           }

           moveit::core::RobotState collision_start(reference_state);
           collision_start.setJointGroupPositions(group, waypoints.front().position);
           collision_start.update(true);
           const auto collision = collision_validator.validatePathWithRefinement(
        *scene_snapshot, collision_start, *group, waypoints, collision_substeps, 3U);
           if (!collision.input_valid || collision.unsafe) {
             return fail(CompactProfileGeometryStatus::kUnsafe,
          "final compact profile collision validation failed: " + collision.failure_reason);
           }
           certificate.status = CompactProfileGeometryStatus::kPass;
           return certificate;
         };
}

bool CompactManeuverValidation::validateCompactPathPosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  const double radius_cap_m, std::string * violation) const
{
  if (violation) {
    violation->clear();
  }
  if (!compact_geometry_validator_ || position.size() != group_->getVariableCount() ||
    !position.allFinite())
  {
    if (violation) {
      *violation = "state is unavailable";
    }
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, position);
  candidate.update(true);
  const double radius = compact_geometry_validator_->maximumAssemblyRadius(candidate);
  if (!std::isfinite(radius) || radius > radius_cap_m + 1.0e-6) {
    if (violation) {
      *violation = "assembly radius";
    }
    return false;
  }
  return true;
}

bool CompactManeuverValidation::validateCompactReturnRegion(
  const moveit::core::RobotState & state, const Eigen::VectorXd & positions,
  const CompactReturnRegion & region) const
{
  const auto * shoulder_link = robot_model_->getLinkModel("link2");
  const auto * elbow_link = robot_model_->getLinkModel("link3");
  const auto * wrist_link = robot_model_->getLinkModel("link4");
  const auto & joint_names = group_->getVariableNames();
  const auto joint1_it = std::find(joint_names.begin(), joint_names.end(), "joint1");
  const auto joint3_it = std::find(joint_names.begin(), joint_names.end(), "joint3");
  if (!shoulder_link || !elbow_link || !wrist_link ||
    joint1_it == joint_names.end() || joint3_it == joint_names.end() ||
    positions.size() != static_cast<Eigen::Index>(joint_names.size()) ||
    !positions.allFinite())
  {
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, positions);
  candidate.update(true);
  const auto & screen_pose = candidate.getGlobalLinkTransform(command_link_);
  return compactReturnRegionContains(
    region, screen_pose.translation(), screen_pose.linear(),
    candidate.getGlobalLinkTransform(shoulder_link).translation(),
    candidate.getGlobalLinkTransform(elbow_link).translation(),
    candidate.getGlobalLinkTransform(wrist_link).translation(),
    positions[std::distance(joint_names.begin(), joint1_it)],
    positions[std::distance(joint_names.begin(), joint3_it)]);
}

bool CompactManeuverValidation::validateCompactBridgeSegment(
  const moveit::core::RobotState & state, const Eigen::VectorXd & from,
  const Eigen::VectorXd & to, const std::size_t profile_sample_index,
  CompactBridgeValidation & validation) const
{
  const auto record_failure = [&validation, profile_sample_index](
    const std::string & predicate, const Eigen::VectorXd & position,
    const std::size_t interpolation_step, const Eigen::Index axis,
    const double value, const double limit) {
      validation.failure_reason = predicate + " at bridge sample " +
        std::to_string(profile_sample_index) + ", interpolation " +
        std::to_string(interpolation_step) + ", axis " + std::to_string(axis) +
        ": value=" + jsonNumber(value) + ", limit=" + jsonNumber(limit);
      validation.failure_details = "\"bridge_sample_index\":" +
        std::to_string(profile_sample_index) +
        ",\"bridge_interpolation_step\":" + std::to_string(interpolation_step) +
        ",\"bridge_axis\":" + std::to_string(axis) +
        ",\"bridge_value\":" + jsonFiniteNumber(value) +
        ",\"bridge_limit\":" + jsonFiniteNumber(limit) +
        ",\"bridge_position_rad\":" + jsonVector(position);
    };
  if (from.size() != to.size() || from.size() != limits_.lower_position.size() ||
    !from.allFinite() || !to.allFinite())
  {
    validation.failure_reason = "invalid bridge segment dimensions or nonfinite positions";
    validation.failure_details = "\"bridge_sample_index\":" +
      std::to_string(profile_sample_index);
    return false;
  }
  constexpr double kMaximumJointStepRad{0.02};
  const double span = (to - from).lpNorm<Eigen::Infinity>();
  const double sample_count = std::max(
    1.0, std::ceil(span / kMaximumJointStepRad));
  if (!std::isfinite(sample_count) || sample_count > 4096.0) {
    validation.failure_reason = "bridge interpolation sample bound exceeded: span=" +
      jsonNumber(span) + ", steps=" + jsonNumber(sample_count);
    validation.failure_details = "\"bridge_sample_index\":" +
      std::to_string(profile_sample_index) + ",\"bridge_span_rad\":" +
      jsonNumber(span) + ",\"bridge_interpolation_steps\":" +
      jsonNumber(sample_count);
    return false;
  }
  const auto steps = static_cast<std::size_t>(sample_count);
  for (std::size_t step = 0U; step <= steps; ++step) {
    const double fraction = static_cast<double>(step) / static_cast<double>(steps);
    const Eigen::VectorXd position = from + fraction * (to - from);
    for (Eigen::Index axis = 0; axis < position.size(); ++axis) {
      if (position[axis] < limits_.lower_position[axis]) {
        record_failure("joint lower bound", position, step, axis, position[axis],
          limits_.lower_position[axis]);
        return false;
      }
      if (position[axis] > limits_.upper_position[axis]) {
        record_failure("joint upper bound", position, step, axis, position[axis],
          limits_.upper_position[axis]);
        return false;
      }
    }
    moveit::core::RobotState bridge_state(state);
    bridge_state.setJointGroupPositions(group_, position);
    bridge_state.update(true);
    const double radius = compact_geometry_validator_->maximumAssemblyRadius(bridge_state);
    if (!std::isfinite(radius)) {
      record_failure("nonfinite assembly radius", position, step, -1, radius,
        validation.radius_cap);
      return false;
    }
    if (radius > validation.radius_cap + 1.0e-6) {
      record_failure("assembly radius cap", position, step, -1, radius,
        validation.radius_cap);
      return false;
    }
    validation.maximum_radius = std::max(validation.maximum_radius, radius);
  }
  return true;
}

bool CompactManeuverValidation::includeCompactEntryEnvelopePosition(
  const moveit::core::RobotState & state, const Eigen::VectorXd & position,
  double * maximum_radius, double * maximum_abs_pitch_rad,
  double * maximum_abs_roll_rad) const
{
  if ((maximum_radius != nullptr && !compact_geometry_validator_) ||
    position.size() != static_cast<Eigen::Index>(group_->getVariableCount()) ||
    !position.allFinite())
  {
    return false;
  }
  moveit::core::RobotState candidate(state);
  candidate.setJointGroupPositions(group_, position);
  candidate.update(true);
  if (maximum_radius != nullptr) {
    const double radius = compact_geometry_validator_->maximumAssemblyRadius(candidate);
    if (!std::isfinite(radius)) {
      return false;
    }
    *maximum_radius = std::max(*maximum_radius, radius);
  }
  if (maximum_abs_pitch_rad != nullptr || maximum_abs_roll_rad != nullptr) {
    const auto pose = evaluateCompactUprightPose(
      candidate.getGlobalLinkTransform(command_link_).linear());
    if (pose && maximum_abs_pitch_rad != nullptr &&
      std::isfinite(pose->signed_pitch_rad))
    {
      *maximum_abs_pitch_rad = std::max(
        *maximum_abs_pitch_rad, std::abs(pose->signed_pitch_rad));
    }
    if (pose && maximum_abs_roll_rad != nullptr &&
      std::isfinite(pose->signed_roll_error_rad))
    {
      *maximum_abs_roll_rad = std::max(
        *maximum_abs_roll_rad, std::abs(pose->signed_roll_error_rad));
    }
  }
  return true;
}

bool CompactManeuverValidation::includeCompactEntryEnvelope(
  const moveit::core::RobotState & state, const double time_sec,
  const PublishedTrajectoryRecord * active_publication,
  const PublishedTrajectoryRecord * latest_publication,
  double * maximum_radius, double * maximum_abs_pitch_rad,
  double * maximum_abs_roll_rad) const
{
  Eigen::VectorXd position;
  state.copyJointGroupPositions(group_, position);
  if (!includeCompactEntryEnvelopePosition(
      state, position, maximum_radius, maximum_abs_pitch_rad,
      maximum_abs_roll_rad))
  {
    return false;
  }
  const auto include_publication = [this, &state, time_sec, &maximum_radius,
      maximum_abs_pitch_rad, maximum_abs_roll_rad](
    const PublishedTrajectoryRecord * publication) {
      if (publication == nullptr || publication->execution_queue.empty()) {
        return true;
      }
      const double stop_time_sec = publication->stationary_time.seconds();
      if (!std::isfinite(stop_time_sec)) {
        return false;
      }
      if (time_sec >= stop_time_sec) {
        return true;
      }
      const moveit_servo::KinematicState * previous = nullptr;
      std::size_t next_index = publication->execution_queue.size();
      for (std::size_t index = 0; index < publication->execution_queue.size(); ++index) {
        const auto & sample = publication->execution_queue[index];
        const double sample_time_sec = sample.time_stamp.seconds();
        if (!std::isfinite(sample_time_sec)) {
          return false;
        }
        if (sample_time_sec < time_sec) {
          previous = &sample;
        } else {
          next_index = index;
          break;
        }
      }
      const auto include_interpolated = [this, &state, &maximum_radius,
          maximum_abs_pitch_rad, maximum_abs_roll_rad](
        const moveit_servo::KinematicState & before,
        const moveit_servo::KinematicState & after, const double at_time_sec) {
          const double before_time_sec = before.time_stamp.seconds();
          const double after_time_sec = after.time_stamp.seconds();
          if (after_time_sec <= before_time_sec ||
            before.positions.size() != after.positions.size() ||
            !before.positions.allFinite() || !after.positions.allFinite())
          {
            return false;
          }
          const double fraction = std::clamp(
            (at_time_sec - before_time_sec) / (after_time_sec - before_time_sec), 0.0, 1.0);
          const Eigen::VectorXd interpolated_position =
            before.positions + fraction * (after.positions - before.positions);
          return includeCompactEntryEnvelopePosition(state, interpolated_position,
            maximum_radius, maximum_abs_pitch_rad, maximum_abs_roll_rad);
        };
      if (previous != nullptr && next_index < publication->execution_queue.size()) {
        const auto & next = publication->execution_queue[next_index];
        const double next_time_sec = next.time_stamp.seconds();
        if (!includeCompactEntryEnvelopePosition(
            state, previous->positions, maximum_radius, maximum_abs_pitch_rad,
            maximum_abs_roll_rad))
        {
          return false;
        }
        if (!include_interpolated(*previous, next, time_sec)) {
          return false;
        }
        if (next_time_sec > stop_time_sec && previous->time_stamp.seconds() < stop_time_sec &&
          !include_interpolated(*previous, next, stop_time_sec))
        {
          return false;
        }
      } else if (previous != nullptr && time_sec < stop_time_sec) {
        // A publication with no future sample holds its final point until stop.
        if (!includeCompactEntryEnvelopePosition(
            state, previous->positions, maximum_radius, maximum_abs_pitch_rad,
            maximum_abs_roll_rad))
        {
          return false;
        }
      }
      for (std::size_t index = next_index;
        index < publication->execution_queue.size(); ++index)
      {
        const auto & sample = publication->execution_queue[index];
        if (sample.time_stamp.seconds() > stop_time_sec) {
          break;
        }
        if (sample.time_stamp.seconds() >= time_sec &&
          !includeCompactEntryEnvelopePosition(
            state, sample.positions, maximum_radius, maximum_abs_pitch_rad,
            maximum_abs_roll_rad))
        {
          return false;
        }
      }
      return true;
    };
  if (!include_publication(active_publication)) {
    return false;
  }
  return latest_publication == active_publication || include_publication(latest_publication);
}

}  // namespace face_tracking_arm::control::detail
