// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__JOINT_PATH_FOLLOWER_HPP_
#define FACE_TRACKING_ARM__JOINT_PATH_FOLLOWER_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace face_tracking_arm::control
{

struct JointPathFollowerConfig
{
  double lookahead_distance_rad{0.10};
  double maximum_attach_distance_rad{0.20};
  double maximum_corridor_distance_rad{0.12};
  double arrival_tolerance_rad{0.02};
  double minimum_progress_rad{0.005};
  double stall_timeout_sec{1.5};
  std::size_t projection_search_segments{4};
};

enum class JointPathStatus
{
  kTracking,
  kArrived,
  kInvalid,
  kStalled,
};

struct JointPathUpdate
{
  JointPathStatus status{JointPathStatus::kInvalid};
  Eigen::VectorXd reference_position;
  double progress_rad{0.0};
  double total_length_rad{0.0};
  double distance_from_path_rad{0.0};
};

/// Follow a geometric joint-space polyline; distances use the Euclidean norm.
/// Angles are real bounded coordinates and are never wrapped by this class.
/// This class neither generates velocities nor checks collisions.
class JointPathFollower final
{
public:
  explicit JointPathFollower(JointPathFollowerConfig config = {});

  /// Attach within the first configured segment window, discard the passed prefix,
  /// and prepend current_position as a connection to the remaining path.
  /// A failed attachment preserves any previously accepted path.
  /// The caller must collision-check the connection and subsequent commands.
  [[nodiscard]] bool setPath(
    const std::vector<Eigen::VectorXd> & path,
    const Eigen::VectorXd & current_position, double time_sec);

  /// Advance only forward using the current and following local segments.
  /// Invalid/stalled/arrived are terminal until a new path is accepted or reset().
  [[nodiscard]] JointPathUpdate update(
    const Eigen::VectorXd & current_position, double time_sec);

  void reset() noexcept;
  [[nodiscard]] bool active() const noexcept;

private:
  [[nodiscard]] Eigen::VectorXd positionAt(double distance_rad) const;

  JointPathFollowerConfig config_;
  std::vector<Eigen::VectorXd> path_;
  std::vector<double> cumulative_length_;
  std::size_t segment_{0};
  double progress_anchor_rad_{0.0};
  double last_progress_time_sec_{0.0};
  double last_update_time_sec_{0.0};
  JointPathUpdate last_update_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__JOINT_PATH_FOLLOWER_HPP_
