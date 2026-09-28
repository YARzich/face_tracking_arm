// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__TRACKING_CORE_HPP_
#define FACE_TRACKING_ARM__TRACKING_CORE_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

namespace face_tracking_arm::tracking
{

using RosTimeNanoseconds = std::int64_t;

struct Pose3d
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

struct FaceSample
{
  Eigen::Vector3d face_in_planning_frame{Eigen::Vector3d::Zero()};
  Eigen::Vector3d base_in_planning_frame{Eigen::Vector3d::Zero()};
  Eigen::Vector3d reach_center_in_planning_frame{Eigen::Vector3d::Zero()};
  RosTimeNanoseconds measurement_stamp_ns{0};
  std::string frame_id;
};

struct TrackingGeometryConfig
{
  /// Preferred screen-to-face distance; never a reason to discard a detected face.
  double minimum_face_distance_m{0.40};
  double safe_reach_radius_m{0.42};
  double direction_epsilon_m{1.0e-6};
  double up_projection_epsilon{1.0e-6};
};

struct TrackingConfig
{
  TrackingGeometryConfig geometry;
  RosTimeNanoseconds freshness_ns{200'000'000};
  RosTimeNanoseconds return_delay_ns{2'000'000'000};
  Eigen::Vector3d rest_position{0.20, 0.00, 1.05};
  std::string planning_frame{"world"};
  bool search_when_idle{false};
};

struct FaceGeometry
{
  Pose3d monitor_pose;
  Eigen::Vector3d face_in_planning_frame{Eigen::Vector3d::Zero()};
  double face_distance_m{0.0};
  bool reach_limited{false};
};

enum class RejectReason
{
  kNone,
  kEmptyFrame,
  kUnexpectedFrame,
  kNonFinite,
  kInvalidConfiguration,
  kNegativeStamp,
  kFutureStamp,
  kStale,
  kNotNewer,
  kDegenerateDirection,
  kFaceInsideMinimumDistance,
  kDegenerateUpProjection,
  kReachEnvelopeUnavailable,
};

using GeometryResult = std::variant<FaceGeometry, RejectReason>;

struct IngestResult
{
  RejectReason reason{RejectReason::kNone};
  std::optional<FaceGeometry> accepted_geometry;

  [[nodiscard]] bool accepted() const noexcept
  {
    return reason == RejectReason::kNone && accepted_geometry.has_value();
  }
};

enum class TargetMode
{
  kFace,
  kHold,
  kRest,
  kSearch,
};

enum class TickStatus
{
  kCommand,
  kHoldPoseUnavailable,
  kInvalidActualPose,
  kClockReset,
};

struct ActualTcpPose
{
  Pose3d pose;
  RosTimeNanoseconds measurement_stamp_ns{0};
};

struct TargetCommand
{
  TargetMode mode{TargetMode::kRest};
  Pose3d pose;
  RosTimeNanoseconds command_stamp_ns{0};
  std::optional<RosTimeNanoseconds> source_face_stamp_ns;
  std::optional<Eigen::Vector3d> face_in_planning_frame;
};

struct TickResult
{
  TickStatus status{TickStatus::kCommand};
  std::optional<TargetCommand> command;
};

/// Compute a full 3D look-at pose in one common frame.
///
/// Local monitor +X initially points from the reach center to the face. While tracking,
/// TrackingCore refines this direction from the measured TCP to the face on every
/// tick. Local +Z is the projection of world +Z onto the screen plane. The target
/// is the closest point to the face inside the sphere centered at reach_center,
/// preferring the configured screen-to-face distance. Unattainable distance is
/// clamped to the envelope without discarding the face. IK/collisions are checked later.
[[nodiscard]] GeometryResult compute_face_geometry(
  const Eigen::Vector3d & face_in_planning_frame,
  const Eigen::Vector3d & base_in_planning_frame,
  const Eigen::Vector3d & reach_center_in_planning_frame,
  const TrackingGeometryConfig & config);

/// Pure, thread-confined policy for accepting face samples and selecting a target.
///
/// All times are absolute readings of the same ROS clock expressed in nanoseconds.
/// The caller must invoke all methods from one execution context. In particular,
/// ActualTcpPose must be a fresh, complete FK snapshot validated by the ROS adapter.
class TrackingCore final
{
public:
  explicit TrackingCore(TrackingConfig config);

  [[nodiscard]] IngestResult ingest_face(
    const FaceSample & sample, RosTimeNanoseconds now_ns);

  [[nodiscard]] TickResult tick(
    RosTimeNanoseconds now_ns,
    const std::optional<ActualTcpPose> & actual_tcp_pose = std::nullopt);

  /// Replace the REST supplier output without changing arbitration state.
  [[nodiscard]] bool update_rest_target(const Pose3d & pose) noexcept;

  void reset() noexcept;

  [[nodiscard]] TargetMode mode() const noexcept;

private:
  struct AcceptedFace
  {
    FaceGeometry geometry;
    RosTimeNanoseconds measurement_stamp_ns{0};
  };

  [[nodiscard]] bool observe_clock(RosTimeNanoseconds now_ns) noexcept;
  [[nodiscard]] TargetCommand make_pose_command(
    TargetMode mode,
    const Pose3d & pose,
    RosTimeNanoseconds now_ns);
  [[nodiscard]] TargetCommand make_hold_command(RosTimeNanoseconds now_ns) const;
  [[nodiscard]] bool latch_hold_pose(
    const ActualTcpPose & actual_tcp_pose, RosTimeNanoseconds now_ns);
  [[nodiscard]] Eigen::Quaterniond align_quaternion(
    const Eigen::Quaterniond & quaternion) const;

  TrackingConfig config_;
  Pose3d rest_target_;
  std::optional<AcceptedFace> latest_face_;
  std::optional<RosTimeNanoseconds> last_accepted_stamp_ns_;
  std::optional<RosTimeNanoseconds> last_observed_time_ns_;
  std::optional<Pose3d> hold_pose_;
  std::optional<Eigen::Quaterniond> previous_orientation_;
  TargetMode mode_{TargetMode::kRest};
};

}  // namespace face_tracking_arm::tracking

#endif  // FACE_TRACKING_ARM__TRACKING_CORE_HPP_
