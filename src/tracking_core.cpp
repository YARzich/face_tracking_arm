// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/tracking_core.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace face_tracking_arm::tracking
{
namespace
{

constexpr double kQuaternionNormEpsilon = 1.0e-12;

bool is_finite(const Eigen::Vector3d & vector)
{
  return vector.array().isFinite().all();
}

bool is_finite(const Eigen::Quaterniond & quaternion)
{
  return quaternion.coeffs().array().isFinite().all();
}

bool valid_geometry_config(const TrackingGeometryConfig & config)
{
  return std::isfinite(config.minimum_face_distance_m) &&
         config.minimum_face_distance_m > 0.0 &&
         std::isfinite(config.safe_reach_radius_m) &&
         config.safe_reach_radius_m > 0.0 &&
         std::isfinite(config.direction_epsilon_m) &&
         config.direction_epsilon_m > 0.0 &&
         std::isfinite(config.up_projection_epsilon) &&
         config.up_projection_epsilon > 0.0;
}

std::optional<Eigen::Quaterniond> look_at_orientation(
  const Eigen::Vector3d & origin,
  const Eigen::Vector3d & target,
  const TrackingGeometryConfig & config)
{
  const Eigen::Vector3d target_from_origin = target - origin;
  const double distance = target_from_origin.norm();
  if (!std::isfinite(distance) || distance <= config.direction_epsilon_m) {
    return std::nullopt;
  }

  const Eigen::Vector3d direction = target_from_origin / distance;
  Eigen::Vector3d screen_up = Eigen::Vector3d::UnitZ() -
    direction.dot(Eigen::Vector3d::UnitZ()) * direction;
  const double up_projection_norm = screen_up.norm();
  if (!std::isfinite(up_projection_norm) ||
    up_projection_norm <= config.up_projection_epsilon)
  {
    return std::nullopt;
  }
  screen_up /= up_projection_norm;
  const Eigen::Vector3d screen_left = screen_up.cross(direction).normalized();

  Eigen::Matrix3d rotation;
  rotation.col(0) = direction;
  rotation.col(1) = screen_left;
  rotation.col(2) = screen_up;
  Eigen::Quaterniond orientation{rotation};
  orientation.normalize();
  return orientation;
}

}  // namespace

GeometryResult compute_face_geometry(
  const Eigen::Vector3d & face_in_planning_frame,
  const Eigen::Vector3d & base_in_planning_frame,
  const Eigen::Vector3d & reach_center_in_planning_frame,
  const TrackingGeometryConfig & config)
{
  if (!is_finite(face_in_planning_frame) || !is_finite(base_in_planning_frame) ||
    !is_finite(reach_center_in_planning_frame))
  {
    return RejectReason::kNonFinite;
  }
  if (!valid_geometry_config(config)) {
    return RejectReason::kInvalidConfiguration;
  }

  const Eigen::Vector3d face_from_base =
    face_in_planning_frame - base_in_planning_frame;
  const double base_to_face_m = face_from_base.norm();
  if (!std::isfinite(base_to_face_m)) {
    return RejectReason::kNonFinite;
  }
  if (base_to_face_m <= config.direction_epsilon_m) {
    return RejectReason::kDegenerateDirection;
  }
  if (base_to_face_m <= config.minimum_face_distance_m) {
    return RejectReason::kFaceInsideMinimumDistance;
  }

  const Eigen::Vector3d face_from_reach_center =
    face_in_planning_frame - reach_center_in_planning_frame;
  const double reach_center_to_face_m = face_from_reach_center.norm();
  if (!std::isfinite(reach_center_to_face_m)) {
    return RejectReason::kNonFinite;
  }
  if (reach_center_to_face_m <= config.direction_epsilon_m) {
    return RejectReason::kDegenerateDirection;
  }

  const Eigen::Vector3d direction = face_from_reach_center / reach_center_to_face_m;
  const auto orientation = look_at_orientation(
    reach_center_in_planning_frame, face_in_planning_frame, config);
  if (!orientation.has_value()) {
    return RejectReason::kDegenerateUpProjection;
  }

  const double desired_range_m =
    reach_center_to_face_m - config.minimum_face_distance_m;
  // If the whole reach sphere lies inside the face exclusion radius, no target
  // can satisfy the minimum distance. Otherwise the center-to-face line gives
  // the closest permissible point. A negative range retreats past the center
  // when a close face requires it, while remaining inside the reach sphere.
  if (desired_range_m < -config.safe_reach_radius_m) {
    return RejectReason::kReachEnvelopeUnavailable;
  }
  const bool reach_limited = desired_range_m > config.safe_reach_radius_m;
  const double selected_range_m = std::min(desired_range_m, config.safe_reach_radius_m);

  FaceGeometry geometry;
  geometry.monitor_pose.position =
    reach_center_in_planning_frame + selected_range_m * direction;
  geometry.monitor_pose.orientation = *orientation;
  geometry.face_in_planning_frame = face_in_planning_frame;
  geometry.face_distance_m =
    (face_in_planning_frame - geometry.monitor_pose.position).norm();
  geometry.reach_limited = reach_limited;
  if (!is_finite(geometry.monitor_pose.position) ||
    !is_finite(geometry.monitor_pose.orientation) ||
    !std::isfinite(geometry.face_distance_m))
  {
    return RejectReason::kNonFinite;
  }
  return geometry;
}

TrackingCore::TrackingCore(TrackingConfig config)
: config_(std::move(config))
{
  if (!valid_geometry_config(config_.geometry)) {
    throw std::invalid_argument("tracking geometry configuration must be finite and positive");
  }
  if (config_.freshness_ns <= 0) {
    throw std::invalid_argument("freshness_ns must be positive");
  }
  if (config_.return_delay_ns <= config_.freshness_ns) {
    throw std::invalid_argument("return_delay_ns must be greater than freshness_ns");
  }
  if (!is_finite(config_.rest_position)) {
    throw std::invalid_argument("rest_position must be finite");
  }
  if (config_.planning_frame.empty()) {
    throw std::invalid_argument("planning_frame must not be empty");
  }
  rest_target_.position = config_.rest_position;
}

IngestResult TrackingCore::ingest_face(
  const FaceSample & sample, const RosTimeNanoseconds now_ns)
{
  (void)observe_clock(now_ns);

  if (sample.frame_id.empty()) {
    return {RejectReason::kEmptyFrame, std::nullopt};
  }
  if (sample.frame_id != config_.planning_frame) {
    return {RejectReason::kUnexpectedFrame, std::nullopt};
  }
  if (!is_finite(sample.face_in_planning_frame) ||
    !is_finite(sample.base_in_planning_frame) ||
    !is_finite(sample.reach_center_in_planning_frame))
  {
    return {RejectReason::kNonFinite, std::nullopt};
  }
  if (sample.measurement_stamp_ns < 0) {
    return {RejectReason::kNegativeStamp, std::nullopt};
  }
  if (sample.measurement_stamp_ns > now_ns) {
    return {RejectReason::kFutureStamp, std::nullopt};
  }
  if (last_accepted_stamp_ns_.has_value() &&
    sample.measurement_stamp_ns <= *last_accepted_stamp_ns_)
  {
    return {RejectReason::kNotNewer, std::nullopt};
  }

  const RosTimeNanoseconds age_ns = now_ns - sample.measurement_stamp_ns;
  if (age_ns >= config_.freshness_ns) {
    return {RejectReason::kStale, std::nullopt};
  }

  GeometryResult geometry_result = compute_face_geometry(
    sample.face_in_planning_frame,
    sample.base_in_planning_frame,
    sample.reach_center_in_planning_frame,
    config_.geometry);
  if (const auto * reason = std::get_if<RejectReason>(&geometry_result)) {
    return {*reason, std::nullopt};
  }

  const FaceGeometry geometry = std::get<FaceGeometry>(geometry_result);
  latest_face_ = AcceptedFace{geometry, sample.measurement_stamp_ns};
  last_accepted_stamp_ns_ = sample.measurement_stamp_ns;
  hold_pose_.reset();
  return {RejectReason::kNone, geometry};
}

TickResult TrackingCore::tick(
  const RosTimeNanoseconds now_ns,
  const std::optional<ActualTcpPose> & actual_tcp_pose)
{
  const bool clock_was_reset = observe_clock(now_ns);
  const TickStatus regular_status =
    clock_was_reset ? TickStatus::kClockReset : TickStatus::kCommand;

  if (!latest_face_.has_value()) {
    mode_ = config_.search_when_idle ? TargetMode::kSearch : TargetMode::kRest;
    hold_pose_.reset();
    return {
      regular_status,
      make_pose_command(mode_, rest_target_, now_ns)};
  }

  const RosTimeNanoseconds age_ns = now_ns - latest_face_->measurement_stamp_ns;
  if (age_ns < config_.freshness_ns) {
    mode_ = TargetMode::kFace;
    hold_pose_.reset();
    Pose3d target_pose = latest_face_->geometry.monitor_pose;
    if (actual_tcp_pose.has_value()) {
      if (actual_tcp_pose->measurement_stamp_ns < 0 ||
        actual_tcp_pose->measurement_stamp_ns > now_ns ||
        !is_finite(actual_tcp_pose->pose.position) ||
        !is_finite(actual_tcp_pose->pose.orientation))
      {
        return {TickStatus::kInvalidActualPose, std::nullopt};
      }
      const auto refined_orientation = look_at_orientation(
        actual_tcp_pose->pose.position,
        latest_face_->geometry.face_in_planning_frame,
        config_.geometry);
      if (refined_orientation.has_value()) {
        target_pose.orientation = *refined_orientation;
      }
    }
    return {
      regular_status,
      make_pose_command(
        TargetMode::kFace, target_pose, now_ns)};
  }

  if (age_ns < config_.return_delay_ns) {
    mode_ = TargetMode::kHold;
    if (!hold_pose_.has_value()) {
      if (!actual_tcp_pose.has_value()) {
        return {TickStatus::kHoldPoseUnavailable, std::nullopt};
      }
      if (!latch_hold_pose(*actual_tcp_pose, now_ns)) {
        return {TickStatus::kInvalidActualPose, std::nullopt};
      }
    }
    return {regular_status, make_hold_command(now_ns)};
  }

  mode_ = config_.search_when_idle ? TargetMode::kSearch : TargetMode::kRest;
  hold_pose_.reset();
  return {
    regular_status,
    make_pose_command(mode_, rest_target_, now_ns)};
}

bool TrackingCore::update_rest_target(const Pose3d & pose) noexcept
{
  const double quaternion_norm = pose.orientation.norm();
  if (!is_finite(pose.position) || !is_finite(pose.orientation) ||
    !std::isfinite(quaternion_norm) || quaternion_norm <= kQuaternionNormEpsilon)
  {
    return false;
  }
  rest_target_ = pose;
  rest_target_.orientation.normalize();
  return true;
}

void TrackingCore::reset() noexcept
{
  latest_face_.reset();
  last_accepted_stamp_ns_.reset();
  last_observed_time_ns_.reset();
  hold_pose_.reset();
  previous_orientation_.reset();
  mode_ = TargetMode::kRest;
}

TargetMode TrackingCore::mode() const noexcept
{
  return mode_;
}

bool TrackingCore::observe_clock(const RosTimeNanoseconds now_ns) noexcept
{
  if (last_observed_time_ns_.has_value() && now_ns < *last_observed_time_ns_) {
    reset();
    last_observed_time_ns_ = now_ns;
    return true;
  }
  last_observed_time_ns_ = now_ns;
  return false;
}

TargetCommand TrackingCore::make_pose_command(
  const TargetMode mode,
  const Pose3d & pose,
  const RosTimeNanoseconds now_ns)
{
  const Eigen::Quaterniond orientation =
    align_quaternion(pose.orientation.normalized());

  previous_orientation_ = orientation;

  TargetCommand command;
  command.mode = mode;
  command.pose = Pose3d{pose.position, orientation};
  command.command_stamp_ns = now_ns;
  if (latest_face_.has_value()) {
    command.source_face_stamp_ns = latest_face_->measurement_stamp_ns;
    if (mode == TargetMode::kFace) {
      command.face_in_planning_frame = latest_face_->geometry.face_in_planning_frame;
    }
  }
  return command;
}

TargetCommand TrackingCore::make_hold_command(const RosTimeNanoseconds now_ns) const
{
  TargetCommand command;
  command.mode = TargetMode::kHold;
  command.pose = *hold_pose_;
  command.command_stamp_ns = now_ns;
  command.source_face_stamp_ns = latest_face_->measurement_stamp_ns;
  return command;
}

bool TrackingCore::latch_hold_pose(
  const ActualTcpPose & actual_tcp_pose, const RosTimeNanoseconds now_ns)
{
  if (actual_tcp_pose.measurement_stamp_ns < 0 ||
    actual_tcp_pose.measurement_stamp_ns > now_ns ||
    !is_finite(actual_tcp_pose.pose.position) ||
    !is_finite(actual_tcp_pose.pose.orientation))
  {
    return false;
  }

  const double quaternion_norm = actual_tcp_pose.pose.orientation.norm();
  if (!std::isfinite(quaternion_norm) || quaternion_norm <= kQuaternionNormEpsilon) {
    return false;
  }

  Eigen::Quaterniond orientation = actual_tcp_pose.pose.orientation.normalized();
  orientation = align_quaternion(orientation);

  hold_pose_ = Pose3d{actual_tcp_pose.pose.position, orientation};
  previous_orientation_ = orientation;
  return true;
}

Eigen::Quaterniond TrackingCore::align_quaternion(
  const Eigen::Quaterniond & quaternion) const
{
  Eigen::Quaterniond aligned = quaternion;
  if (previous_orientation_.has_value() && previous_orientation_->dot(aligned) < 0.0) {
    aligned.coeffs() *= -1.0;
  }
  return aligned;
}

}  // namespace face_tracking_arm::tracking
