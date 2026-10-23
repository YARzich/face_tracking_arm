// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__HARDWARE_IO_STATUS_POLICY_HPP_
#define FACE_TRACKING_ARM__HARDWARE_IO_STATUS_POLICY_HPP_

#include <array>
#include <chrono>
#include <cstdint>
#include "face_tracking_arm/msg/hardware_io_status.hpp"

namespace face_tracking_arm::hardware
{

struct HardwareIoSample
{
  std::array<double, 6> position{};
  std::array<double, 6> velocity{};
  std::uint64_t epoch{0};
  std::uint64_t read_sequence{0};
  std::uint64_t read_epoch{0};
  std::uint64_t robot_msg_sequence{0};
  std::uint64_t write_sequence{0};
  std::uint64_t driver_command_sequence{0};
  std::uint32_t write_in_flight{0};
  std::uint64_t fault_sequence{0};
  std::uint64_t fault_read_attempt_sequence{0};
  std::uint64_t fault_epoch{0};
  std::uint8_t fault_kind{0};
  std::int32_t fault_code{0};
  std::int64_t fault_stamp_ns{0};
  std::int64_t read_stamp_ns{0};
  std::uint32_t read_age_us{0};
  std::uint32_t robot_msg_age_us{0};
};

enum class HardwareIoStatusFailure
{
  kNone,
  kWrongSource,
  kRuntimeActionsDisabled,
  kHardwareInactive,
  kInhibitLatched,
  kInvalidEpochOrSequence,
  kReadFailed,
  kWriteFailed,
  kRobotMessageUnavailable,
  kInvalidTimestamp,
  kFutureOrRewoundTime,
  kTransportTooOld,
  kSampleTooOld,
  kSequenceRegressed,
  kNonFiniteValue,
  kJointBoundaryUnresolved,
  kJointOutsideLimit,
  kDriverFaultLatched,
};

[[nodiscard]] inline const char * hardwareIoStatusFailureName(
  const HardwareIoStatusFailure failure) noexcept
{
  switch (failure) {
    case HardwareIoStatusFailure::kNone: return "none";
    case HardwareIoStatusFailure::kWrongSource: return "wrong_source";
    case HardwareIoStatusFailure::kRuntimeActionsDisabled: return "runtime_actions_disabled";
    case HardwareIoStatusFailure::kHardwareInactive: return "hardware_inactive";
    case HardwareIoStatusFailure::kInhibitLatched: return "inhibit_latched";
    case HardwareIoStatusFailure::kInvalidEpochOrSequence: return "invalid_epoch_or_sequence";
    case HardwareIoStatusFailure::kReadFailed: return "read_failed";
    case HardwareIoStatusFailure::kWriteFailed: return "write_failed";
    case HardwareIoStatusFailure::kRobotMessageUnavailable: return "robot_message_unavailable";
    case HardwareIoStatusFailure::kInvalidTimestamp: return "invalid_timestamp";
    case HardwareIoStatusFailure::kFutureOrRewoundTime: return "future_or_rewound_time";
    case HardwareIoStatusFailure::kTransportTooOld: return "transport_too_old";
    case HardwareIoStatusFailure::kSampleTooOld: return "sample_too_old";
    case HardwareIoStatusFailure::kSequenceRegressed: return "sequence_regressed";
    case HardwareIoStatusFailure::kNonFiniteValue: return "non_finite_value";
    case HardwareIoStatusFailure::kJointBoundaryUnresolved: return "joint_boundary_unresolved";
    case HardwareIoStatusFailure::kJointOutsideLimit: return "joint_outside_limit";
    case HardwareIoStatusFailure::kDriverFaultLatched: return "driver_fault_latched";
  }
  return "unknown";
}

struct HardwareIoStatusAssessment
{
  bool valid{false};
  HardwareIoStatusFailure failure{HardwareIoStatusFailure::kInvalidEpochOrSequence};
  HardwareIoSample sample{};
  std::int64_t receive_ros_time_ns{0};
  std::int64_t header_stamp_ns{0};
  std::int64_t age_at_receive_ns{0};
  std::int64_t robot_age_at_receive_ns{0};
  std::int64_t maximum_sample_age_ns{0};
  std::int64_t maximum_robot_message_age_ns{0};
  std::int64_t maximum_transport_age_ns{0};
  std::chrono::steady_clock::time_point receive_steady_time{};
  std::uint32_t failure_joint_index{0};
};

// Validates one coherent driver status and keeps its provenance fresh through
// the bounded ROS transport and the executor's steady-clock residence time.
class HardwareIoStatusPolicy
{
public:
  using SteadyTime = std::chrono::steady_clock::time_point;

  HardwareIoStatusPolicy(
    std::int64_t maximum_sample_age_ns,
    std::int64_t maximum_transport_age_ns,
    std::int64_t maximum_robot_message_age_ns = 0);

  HardwareIoStatusAssessment observe(
    const msg::HardwareIoStatus & status,
    std::int64_t receive_ros_time_ns,
    SteadyTime receive_steady_time);

  HardwareIoStatusAssessment current(
    std::int64_t now_ros_time_ns,
    SteadyTime now_steady_time) const;

  void reset();

private:
  HardwareIoStatusAssessment reject(HardwareIoStatusFailure failure);

  std::int64_t maximum_sample_age_ns_;
  std::int64_t maximum_transport_age_ns_;
  std::int64_t maximum_robot_message_age_ns_;
  HardwareIoStatusAssessment latest_{};
  std::int64_t receive_ros_time_ns_{0};
  std::int64_t header_stamp_ns_{0};
  std::int64_t age_at_receive_ns_{0};
  std::int64_t robot_age_at_receive_ns_{0};
  SteadyTime receive_steady_time_{};
  std::int64_t last_header_stamp_ns_{0};
  std::int64_t last_read_stamp_ns_{0};
  std::int64_t last_robot_msg_stamp_ns_{0};
  std::int64_t last_write_stamp_ns_{0};
  std::uint64_t last_epoch_{0};
  std::uint64_t last_read_attempt_sequence_{0};
  std::uint64_t last_read_sequence_{0};
  std::uint64_t last_write_sequence_{0};
  std::uint64_t last_driver_command_sequence_{0};
  std::uint64_t last_robot_msg_sequence_{0};
  std::uint64_t last_inhibit_request_id_{0};
  std::array<double, 6> last_read_position_{};
  std::array<double, 6> last_read_velocity_{};
};

}  // namespace face_tracking_arm::hardware

#endif  // FACE_TRACKING_ARM__HARDWARE_IO_STATUS_POLICY_HPP_
