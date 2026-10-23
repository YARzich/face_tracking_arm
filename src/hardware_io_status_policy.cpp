// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/hardware_io_status_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace face_tracking_arm::hardware
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr std::int64_t kNanosecondsPerMicrosecond = 1000LL;

bool to_nanoseconds(
  const builtin_interfaces::msg::Time & stamp,
  std::int64_t & result)
{
  if (stamp.sec < 0 || stamp.nanosec >= static_cast<std::uint32_t>(kNanosecondsPerSecond)) {
    return false;
  }
  const auto sec = static_cast<std::int64_t>(stamp.sec);
  if (sec > (std::numeric_limits<std::int64_t>::max() - stamp.nanosec) /
    kNanosecondsPerSecond)
  {
    return false;
  }
  result = sec * kNanosecondsPerSecond + stamp.nanosec;
  return result > 0;
}

bool finite(const msg::HardwareIoStatus & status)
{
  for (std::size_t index = 0; index < status.q.size(); ++index) {
    if (!std::isfinite(status.q[index]) || !std::isfinite(status.dq[index]) ||
      !std::isfinite(status.command_double[index]) ||
      !std::isfinite(status.command_float[index]) ||
      !std::isfinite(status.candidate_double[index]) ||
      !std::isfinite(status.candidate_float[index]) ||
      !std::isfinite(status.fault_candidate_double[index]) ||
      !std::isfinite(status.fault_candidate_float[index]))
    {
      return false;
    }
  }
  return true;
}

}  // namespace

HardwareIoStatusPolicy::HardwareIoStatusPolicy(
  const std::int64_t maximum_sample_age_ns,
  const std::int64_t maximum_transport_age_ns,
  const std::int64_t maximum_robot_message_age_ns)
: maximum_sample_age_ns_(maximum_sample_age_ns),
  maximum_transport_age_ns_(maximum_transport_age_ns),
  maximum_robot_message_age_ns_(maximum_robot_message_age_ns > 0 ?
    maximum_robot_message_age_ns : maximum_sample_age_ns)
{
  if (maximum_sample_age_ns_ <= 0 || maximum_transport_age_ns_ < 0 ||
    maximum_transport_age_ns_ > maximum_sample_age_ns_ ||
    maximum_robot_message_age_ns_ <= 0 ||
    maximum_transport_age_ns_ > maximum_robot_message_age_ns_)
  {
    throw std::invalid_argument("Hardware I/O age limits are invalid");
  }
}

HardwareIoStatusAssessment HardwareIoStatusPolicy::reject(
  const HardwareIoStatusFailure failure)
{
  latest_ = {};
  latest_.failure = failure;
  return latest_;
}

HardwareIoStatusAssessment HardwareIoStatusPolicy::observe(
  const msg::HardwareIoStatus & status,
  const std::int64_t receive_ros_time_ns,
  const SteadyTime receive_steady_time)
{
  if (status.source_kind != msg::HardwareIoStatus::SOURCE_HARDWARE) {
    return reject(HardwareIoStatusFailure::kWrongSource);
  }
  if (!status.runtime_actions_enabled) {
    return reject(HardwareIoStatusFailure::kRuntimeActionsDisabled);
  }
  if (!status.hardware_active) {
    return reject(HardwareIoStatusFailure::kHardwareInactive);
  }
  if (status.inhibit_latched) {
    return reject(HardwareIoStatusFailure::kInhibitLatched);
  }
  if (status.fault_latched || status.fault_kind != msg::HardwareIoStatus::FAULT_NONE) {
    (void)reject(HardwareIoStatusFailure::kDriverFaultLatched);
    latest_.sample.fault_sequence = status.fault_sequence;
    latest_.sample.fault_read_attempt_sequence = status.fault_read_attempt_sequence;
    latest_.sample.fault_epoch = status.fault_epoch;
    latest_.sample.fault_kind = status.fault_kind;
    latest_.sample.fault_code = status.fault_code;
    return latest_;
  }
  if (status.epoch == 0 || status.read_attempt_sequence == 0 ||
    status.read_sequence == 0 || status.read_sequence > status.read_attempt_sequence ||
    status.read_epoch != status.epoch)
  {
    return reject(HardwareIoStatusFailure::kInvalidEpochOrSequence);
  }
  if (!status.read_ok) {
    return reject(HardwareIoStatusFailure::kReadFailed);
  }
  if (status.read_code != 0) {
    return reject(HardwareIoStatusFailure::kReadFailed);
  }
  if (status.write_attempted && !status.write_ok) {
    return reject(HardwareIoStatusFailure::kWriteFailed);
  }
  if (status.write_attempted && status.write_code != 0) {
    return reject(HardwareIoStatusFailure::kWriteFailed);
  }
  if (status.write_attempted && status.write_sequence == 0) {
    return reject(HardwareIoStatusFailure::kWriteFailed);
  }
  if (status.robot_msg_sequence == 0 || status.robot_msg_age_us ==
    std::numeric_limits<std::uint32_t>::max())
  {
    return reject(HardwareIoStatusFailure::kRobotMessageUnavailable);
  }
  if (status.robot_mode != 1U || status.robot_state > 2U ||
    status.robot_error_code != 0U ||
    (status.robot_mt_able & 0x3FU) != 0x3FU ||
    (status.robot_mt_brake & 0x3FU) != 0x3FU)
  {
    return reject(HardwareIoStatusFailure::kRobotMessageUnavailable);
  }
  if (!finite(status)) {
    return reject(HardwareIoStatusFailure::kNonFiniteValue);
  }

  std::int64_t header_stamp_ns = 0;
  std::int64_t read_stamp_ns = 0;
  std::int64_t robot_msg_stamp_ns = 0;
  std::int64_t write_stamp_ns = 0;
  if (receive_ros_time_ns <= 0 || !to_nanoseconds(status.header.stamp, header_stamp_ns) ||
    !to_nanoseconds(status.read_stamp, read_stamp_ns) ||
    !to_nanoseconds(status.robot_msg_stamp, robot_msg_stamp_ns) ||
    (status.write_sequence != 0 && !to_nanoseconds(status.write_stamp, write_stamp_ns)))
  {
    return reject(HardwareIoStatusFailure::kInvalidTimestamp);
  }
  if (header_stamp_ns > receive_ros_time_ns || read_stamp_ns > header_stamp_ns ||
    robot_msg_stamp_ns > header_stamp_ns || write_stamp_ns > header_stamp_ns)
  {
    return reject(HardwareIoStatusFailure::kFutureOrRewoundTime);
  }
  const std::int64_t transport_age_ns = receive_ros_time_ns - header_stamp_ns;
  if (transport_age_ns > maximum_transport_age_ns_) {
    return reject(HardwareIoStatusFailure::kTransportTooOld);
  }
  if (status.read_age_us == std::numeric_limits<std::uint32_t>::max()) {
    return reject(HardwareIoStatusFailure::kSampleTooOld);
  }
  const auto read_age_ns = static_cast<std::int64_t>(status.read_age_us) *
    kNanosecondsPerMicrosecond;
  const auto ros_read_age_ns = header_stamp_ns - read_stamp_ns;
  if (std::llabs(ros_read_age_ns - read_age_ns) > maximum_transport_age_ns_) {
    return reject(HardwareIoStatusFailure::kInvalidTimestamp);
  }
  const auto robot_age_ns = static_cast<std::int64_t>(status.robot_msg_age_us) *
    kNanosecondsPerMicrosecond;
  const auto ros_robot_age_ns = header_stamp_ns - robot_msg_stamp_ns;
  if (std::llabs(ros_robot_age_ns - robot_age_ns) > maximum_transport_age_ns_) {
    return reject(HardwareIoStatusFailure::kInvalidTimestamp);
  }
  const auto sample_age_at_receive_ns = read_age_ns + transport_age_ns;
  const auto robot_age_at_receive_ns = robot_age_ns + transport_age_ns;
  if (sample_age_at_receive_ns > maximum_sample_age_ns_) {
    return reject(HardwareIoStatusFailure::kSampleTooOld);
  }
  if (robot_age_at_receive_ns > maximum_robot_message_age_ns_) {
    return reject(HardwareIoStatusFailure::kRobotMessageUnavailable);
  }

  if (last_epoch_ != 0 && status.epoch != last_epoch_) {
    return reject(HardwareIoStatusFailure::kSequenceRegressed);
  }
  if (header_stamp_ns < last_header_stamp_ns_ ||
    status.read_attempt_sequence < last_read_attempt_sequence_ ||
    status.read_sequence < last_read_sequence_ ||
    status.write_sequence < last_write_sequence_ ||
    status.driver_command_sequence < last_driver_command_sequence_ ||
    status.robot_msg_sequence < last_robot_msg_sequence_ ||
    status.inhibit_request_id < last_inhibit_request_id_ ||
    (status.read_sequence > last_read_sequence_ && read_stamp_ns < last_read_stamp_ns_) ||
    (status.robot_msg_sequence > last_robot_msg_sequence_ &&
    robot_msg_stamp_ns < last_robot_msg_stamp_ns_) ||
    (status.write_sequence > last_write_sequence_ && write_stamp_ns < last_write_stamp_ns_) ||
    (status.read_sequence == last_read_sequence_ && read_stamp_ns != last_read_stamp_ns_) ||
    (status.robot_msg_sequence == last_robot_msg_sequence_ &&
    robot_msg_stamp_ns != last_robot_msg_stamp_ns_) ||
    (status.read_sequence == last_read_sequence_ &&
    (status.q != last_read_position_ || status.dq != last_read_velocity_)))
  {
    return reject(HardwareIoStatusFailure::kSequenceRegressed);
  }
  if (status.read_sequence != status.read_attempt_sequence) {
    return reject(HardwareIoStatusFailure::kInvalidEpochOrSequence);
  }

  last_epoch_ = status.epoch;
  last_header_stamp_ns_ = header_stamp_ns;
  last_read_attempt_sequence_ = status.read_attempt_sequence;
  last_read_sequence_ = status.read_sequence;
  last_write_sequence_ = status.write_sequence;
  last_driver_command_sequence_ = status.driver_command_sequence;
  last_robot_msg_sequence_ = status.robot_msg_sequence;
  last_inhibit_request_id_ = status.inhibit_request_id;
  last_read_stamp_ns_ = read_stamp_ns;
  last_robot_msg_stamp_ns_ = robot_msg_stamp_ns;
  last_write_stamp_ns_ = write_stamp_ns;
  receive_ros_time_ns_ = receive_ros_time_ns;
  header_stamp_ns_ = header_stamp_ns;
  age_at_receive_ns_ = sample_age_at_receive_ns;
  robot_age_at_receive_ns_ = robot_age_at_receive_ns;
  receive_steady_time_ = receive_steady_time;
  latest_.valid = true;
  latest_.failure = HardwareIoStatusFailure::kNone;
  latest_.receive_ros_time_ns = receive_ros_time_ns;
  latest_.header_stamp_ns = header_stamp_ns;
  latest_.age_at_receive_ns = age_at_receive_ns_;
  latest_.robot_age_at_receive_ns = robot_age_at_receive_ns_;
  latest_.maximum_sample_age_ns = maximum_sample_age_ns_;
  latest_.maximum_robot_message_age_ns = maximum_robot_message_age_ns_;
  latest_.maximum_transport_age_ns = maximum_transport_age_ns_;
  latest_.receive_steady_time = receive_steady_time;
  latest_.sample.epoch = status.epoch;
  latest_.sample.read_sequence = status.read_sequence;
  latest_.sample.read_epoch = status.read_epoch;
  latest_.sample.robot_msg_sequence = status.robot_msg_sequence;
  latest_.sample.write_sequence = status.write_sequence;
  latest_.sample.driver_command_sequence = status.driver_command_sequence;
  latest_.sample.write_in_flight = status.write_in_flight;
  latest_.sample.fault_sequence = status.fault_sequence;
  latest_.sample.fault_read_attempt_sequence = status.fault_read_attempt_sequence;
  latest_.sample.fault_kind = status.fault_kind;
  latest_.sample.fault_code = status.fault_code;
  latest_.sample.read_stamp_ns = read_stamp_ns;
  latest_.sample.read_age_us = status.read_age_us;
  latest_.sample.robot_msg_age_us = status.robot_msg_age_us;
  std::copy(status.q.begin(), status.q.end(), latest_.sample.position.begin());
  std::copy(status.dq.begin(), status.dq.end(), latest_.sample.velocity.begin());
  last_read_position_ = latest_.sample.position;
  last_read_velocity_ = latest_.sample.velocity;
  return latest_;
}

HardwareIoStatusAssessment HardwareIoStatusPolicy::current(
  const std::int64_t now_ros_time_ns,
  const SteadyTime now_steady_time) const
{
  auto result = latest_;
  if (!result.valid) {
    return result;
  }
  if (now_ros_time_ns < receive_ros_time_ns_ || now_ros_time_ns < header_stamp_ns_) {
    result.valid = false;
    result.failure = HardwareIoStatusFailure::kFutureOrRewoundTime;
    return result;
  }
  if (now_steady_time < receive_steady_time_) {
    result.valid = false;
    result.failure = HardwareIoStatusFailure::kFutureOrRewoundTime;
    return result;
  }
  const auto elapsed_steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
    now_steady_time - receive_steady_time_).count();
  const auto elapsed_ros_ns = now_ros_time_ns - receive_ros_time_ns_;
  if (std::llabs(elapsed_ros_ns - elapsed_steady_ns) > maximum_transport_age_ns_) {
    result.valid = false;
    result.failure = HardwareIoStatusFailure::kInvalidTimestamp;
    return result;
  }
  if (age_at_receive_ns_ + elapsed_steady_ns > maximum_sample_age_ns_) {
    result.valid = false;
    result.failure = HardwareIoStatusFailure::kSampleTooOld;
  }
  if (result.valid && robot_age_at_receive_ns_ + elapsed_steady_ns >
    maximum_robot_message_age_ns_)
  {
    result.valid = false;
    result.failure = HardwareIoStatusFailure::kRobotMessageUnavailable;
  }
  return result;
}

void HardwareIoStatusPolicy::reset()
{
  latest_ = {};
  receive_ros_time_ns_ = 0;
  header_stamp_ns_ = 0;
  age_at_receive_ns_ = 0;
  robot_age_at_receive_ns_ = 0;
  receive_steady_time_ = {};
  last_header_stamp_ns_ = 0;
  last_read_stamp_ns_ = 0;
  last_robot_msg_stamp_ns_ = 0;
  last_write_stamp_ns_ = 0;
  last_epoch_ = 0;
  last_read_attempt_sequence_ = 0;
  last_read_sequence_ = 0;
  last_write_sequence_ = 0;
  last_driver_command_sequence_ = 0;
  last_robot_msg_sequence_ = 0;
  last_inhibit_request_id_ = 0;
  last_read_position_ = {};
  last_read_velocity_ = {};
}

}  // namespace face_tracking_arm::hardware
