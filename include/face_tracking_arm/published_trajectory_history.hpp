// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__PUBLISHED_TRAJECTORY_HISTORY_HPP_
#define FACE_TRACKING_ARM__PUBLISHED_TRAJECTORY_HISTORY_HPP_

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <moveit_servo/utils/datatypes.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/time.hpp>

namespace face_tracking_arm::control
{

using PublicationId = std::uint64_t;

struct PublishedTrajectoryRecord final
{
  PublicationId id{0};
  std::deque<moveit_servo::KinematicState> execution_queue;
  rclcpp::Time stationary_time;
};

struct ExpectedPositionState final
{
  Eigen::VectorXd positions;
  Eigen::VectorXd velocities;
  bool terminal_hold{false};
};

struct ControllerSampleTimes final
{
  rclcpp::Time trajectory_origin;
  rclcpp::Time reference;
  rclcpp::Time output;
};

enum class PublicationMatchKind : std::uint8_t
{
  kNone,
  kLatest,
  kOlder,
};

struct PublicationMatch final
{
  PublicationMatchKind kind{PublicationMatchKind::kNone};
  PublicationId publication_id{0};
  ControllerSampleTimes sample_times;
  ExpectedPositionState expected_reference;
  ExpectedPositionState expected_output;
  double reference_error_rad{0.0};
  double output_error_rad{0.0};

  [[nodiscard]] bool matched() const noexcept
  {
    return kind != PublicationMatchKind::kNone;
  }
};

/// Bounded record of position-only trajectories that were accepted and sent.
///
/// MoveIt Servo's trajectory composer deliberately omits the final state in an
/// execution queue. Consequently the last transmitted state is the penultimate
/// queue entry, while the final entry is retained as a local brake sentinel.
class PublishedTrajectoryHistory final
{
public:
  explicit PublishedTrajectoryHistory(std::size_t capacity = 8U);

  /// Append a structurally valid, safely published execution queue.
  ///
  /// The stationary timestamp must identify an entry that was actually
  /// transmitted. All entries from that point through the local sentinel must
  /// describe the same stopped position. Invalid records are rejected without
  /// consuming an identity.
  [[nodiscard]] std::optional<PublicationId> append(
    std::deque<moveit_servo::KinematicState> execution_queue,
    const rclcpp::Time & stationary_time);

  void clear() noexcept;

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  /// Returned pointers remain valid until the next non-const member call.
  [[nodiscard]] const PublishedTrajectoryRecord * latest() const noexcept;
  [[nodiscard]] const PublishedTrajectoryRecord * find(PublicationId id) const noexcept;

  /// Recover the trajectory-time samples represented by one JTC state message.
  ///
  /// JTC's header uses controller wall/sim time, while reference.time_from_start
  /// follows its possibly shifted trajectory clock. feedback.time_from_start
  /// relates both clocks. The output position command is the setpoint prepared
  /// one controller update after the reference sample.
  [[nodiscard]] static std::optional<ControllerSampleTimes> correctedSampleTimes(
    const control_msgs::msg::JointTrajectoryControllerState & state,
    rcl_clock_type_t clock_type, const rclcpp::Duration & controller_period) noexcept;

  /// Match both reference and output against the same publication.
  ///
  /// The newest publication is tried first. Samples inside an explicitly
  /// transmitted horizon may match by value because successive replacements
  /// deliberately share a checked prefix. Extrapolating a terminal hold also
  /// requires an exact JTC trajectory-origin match.
  [[nodiscard]] PublicationMatch match(
    const control_msgs::msg::JointTrajectoryControllerState & state,
    rcl_clock_type_t clock_type, const rclcpp::Duration & controller_period,
    double position_tolerance_rad) const;

private:
  [[nodiscard]] static std::optional<ExpectedPositionState> expectedStateAt(
    const PublishedTrajectoryRecord & record, const rclcpp::Time & sample_time,
    bool allow_terminal_extrapolation);

  std::size_t capacity_;
  PublicationId next_id_{1};
  std::deque<PublishedTrajectoryRecord> records_;
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__PUBLISHED_TRAJECTORY_HISTORY_HPP_
