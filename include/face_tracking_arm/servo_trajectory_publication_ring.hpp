// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__SERVO_TRAJECTORY_PUBLICATION_RING_HPP_
#define FACE_TRACKING_ARM__SERVO_TRAJECTORY_PUBLICATION_RING_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace face_tracking_arm
{
namespace diagnostics
{

constexpr std::size_t kTrajectoryJointCount = 6U;
constexpr std::size_t kTrajectoryPointCapacity = 2048U;
constexpr std::size_t kTrajectoryPublicationQueueCapacity = 64U;
constexpr std::size_t kTrajectoryNameCapacity = 64U;
constexpr std::size_t kTrajectoryFrameCapacity = 128U;

struct ServoTrajectoryPointRecord
{
  std::array<double, kTrajectoryJointCount> positions{};
  bool has_velocities{false};
  bool has_accelerations{false};
  std::int32_t time_sec{0};
  std::uint32_t time_nanosec{0U};
};

struct ServoTrajectoryPublicationRecord
{
  std::int32_t stamp_sec{0};
  std::uint32_t stamp_nanosec{0U};
  std::int32_t stationary_sec{0};
  std::uint32_t stationary_nanosec{0U};
  std::int32_t trajectory_stamp_sec{0};
  std::uint32_t trajectory_stamp_nanosec{0U};
  std::uint64_t publication_id{0U};
  std::uint8_t publication_kind{0U};
  std::uint64_t hardware_epoch{0U};
  std::uint64_t hardware_read_sequence{0U};
  std::uint64_t control_tick_sequence{0U};
  std::uint32_t tail_start_index{0U};
  std::uint8_t joint_count{0U};
  std::uint16_t point_count{0U};
  std::array<std::uint8_t, kTrajectoryJointCount> joint_name_sizes{};
  std::array<std::array<char, kTrajectoryNameCapacity>, kTrajectoryJointCount> joint_names{};
  std::uint8_t event_frame_id_size{0U};
  std::array<char, kTrajectoryFrameCapacity> event_frame_id{};
  std::uint8_t trajectory_frame_id_size{0U};
  std::array<char, kTrajectoryFrameCapacity> trajectory_frame_id{};
  std::array<ServoTrajectoryPointRecord, kTrajectoryPointCapacity> points{};
};

inline void copy_servo_trajectory_record(
  const ServoTrajectoryPublicationRecord & source,
  ServoTrajectoryPublicationRecord & destination) noexcept
{
  destination.stamp_sec = source.stamp_sec;
  destination.stamp_nanosec = source.stamp_nanosec;
  destination.stationary_sec = source.stationary_sec;
  destination.stationary_nanosec = source.stationary_nanosec;
  destination.trajectory_stamp_sec = source.trajectory_stamp_sec;
  destination.trajectory_stamp_nanosec = source.trajectory_stamp_nanosec;
  destination.publication_id = source.publication_id;
  destination.publication_kind = source.publication_kind;
  destination.hardware_epoch = source.hardware_epoch;
  destination.hardware_read_sequence = source.hardware_read_sequence;
  destination.control_tick_sequence = source.control_tick_sequence;
  destination.tail_start_index = source.tail_start_index;
  destination.joint_count = source.joint_count;
  destination.point_count = source.point_count;
  destination.joint_name_sizes = source.joint_name_sizes;
  destination.joint_names = source.joint_names;
  destination.event_frame_id_size = source.event_frame_id_size;
  destination.event_frame_id = source.event_frame_id;
  destination.trajectory_frame_id_size = source.trajectory_frame_id_size;
  destination.trajectory_frame_id = source.trajectory_frame_id;
  const auto count = source.point_count <= kTrajectoryPointCapacity ?
    source.point_count : kTrajectoryPointCapacity;
  for (std::size_t index = 0; index < count; ++index) {
    destination.points[index] = source.points[index];
  }
}

inline bool copy_bounded_string(
  const std::string & source, char * destination, std::size_t capacity,
  std::uint8_t & size) noexcept
{
  if (source.size() >= capacity || source.size() > UINT8_MAX) {
    return false;
  }
  size = static_cast<std::uint8_t>(source.size());
  if (!source.empty()) {
    std::memcpy(destination, source.data(), source.size());
  }
  return true;
}

// Accepts the ROS JointTrajectory shape without introducing ROS into this header.
template<typename PublicationMessage>
bool copy_servo_trajectory_publication(
  const PublicationMessage & source, ServoTrajectoryPublicationRecord & destination) noexcept
{
  const auto & trajectory = source.trajectory;
  if (source.publication_id == 0U || trajectory.joint_names.size() != kTrajectoryJointCount ||
    trajectory.points.empty() || trajectory.points.size() > kTrajectoryPointCapacity)
  {
    return false;
  }
  destination.stamp_sec = source.header.stamp.sec;
  destination.stamp_nanosec = source.header.stamp.nanosec;
  destination.stationary_sec = source.stationary_time.sec;
  destination.stationary_nanosec = source.stationary_time.nanosec;
  destination.trajectory_stamp_sec = trajectory.header.stamp.sec;
  destination.trajectory_stamp_nanosec = trajectory.header.stamp.nanosec;
  destination.publication_id = source.publication_id;
  destination.publication_kind = source.publication_kind;
  destination.hardware_epoch = source.hardware_epoch;
  destination.hardware_read_sequence = source.hardware_read_sequence;
  destination.control_tick_sequence = source.control_tick_sequence;
  destination.tail_start_index = source.tail_start_index;
  destination.joint_count = static_cast<std::uint8_t>(trajectory.joint_names.size());
  destination.point_count = static_cast<std::uint16_t>(trajectory.points.size());
  if (!copy_bounded_string(
      source.header.frame_id, destination.event_frame_id.data(),
      destination.event_frame_id.size(), destination.event_frame_id_size) ||
    !copy_bounded_string(
      trajectory.header.frame_id, destination.trajectory_frame_id.data(),
      destination.trajectory_frame_id.size(), destination.trajectory_frame_id_size))
  {
    return false;
  }
  for (std::size_t joint = 0; joint < kTrajectoryJointCount; ++joint) {
    if (trajectory.joint_names[joint].empty() ||
      !copy_bounded_string(
        trajectory.joint_names[joint], destination.joint_names[joint].data(),
        destination.joint_names[joint].size(), destination.joint_name_sizes[joint]))
    {
      return false;
    }
  }
  for (std::size_t index = 0; index < trajectory.points.size(); ++index) {
    const auto & point = trajectory.points[index];
    auto & record = destination.points[index];
    if (point.positions.size() != kTrajectoryJointCount ||
      (!point.velocities.empty() && point.velocities.size() != kTrajectoryJointCount) ||
      (!point.accelerations.empty() && point.accelerations.size() != kTrajectoryJointCount) ||
      !point.effort.empty())
    {
      return false;
    }
    record.has_velocities = !point.velocities.empty();
    record.has_accelerations = !point.accelerations.empty();
    for (const auto value : point.velocities) {
      if (value != 0.0) {return false;}
    }
    for (const auto value : point.accelerations) {
      if (value != 0.0) {return false;}
    }
    for (std::size_t joint = 0; joint < kTrajectoryJointCount; ++joint) {
      record.positions[joint] = point.positions[joint];
    }
    record.time_sec = point.time_from_start.sec;
    record.time_nanosec = point.time_from_start.nanosec;
  }
  return true;
}

template<std::size_t Capacity>
class ServoTrajectoryPublicationRing final
{
  static_assert(Capacity > 0U, "publication ring must have non-zero capacity");

public:
  ServoTrajectoryPublicationRing()
  : records_(new ServoTrajectoryPublicationRecord[Capacity]())
  {
  }

  ServoTrajectoryPublicationRing(const ServoTrajectoryPublicationRing &) = delete;
  ServoTrajectoryPublicationRing & operator=(const ServoTrajectoryPublicationRing &) = delete;

  [[nodiscard]] bool tryPush(const ServoTrajectoryPublicationRecord & record) noexcept
  {
    const auto head = head_.load(std::memory_order_relaxed);
    const auto tail = tail_.load(std::memory_order_acquire);
    if (head - tail >= Capacity) {
      dropped_.fetch_add(1U, std::memory_order_relaxed);
      return false;
    }
    copy_servo_trajectory_record(record, records_[head % Capacity]);
    head_.store(head + 1U, std::memory_order_release);
    return true;
  }

  template<typename PublicationMessage>
  [[nodiscard]] bool tryEncodeAndPush(const PublicationMessage & source) noexcept
  {
    const auto head = head_.load(std::memory_order_relaxed);
    const auto tail = tail_.load(std::memory_order_acquire);
    if (head - tail >= Capacity) {
      dropped_.fetch_add(1U, std::memory_order_relaxed);
      return false;
    }
    auto & destination = records_[head % Capacity];
    if (!copy_servo_trajectory_publication(source, destination)) {
      dropped_.fetch_add(1U, std::memory_order_relaxed);
      return false;
    }
    head_.store(head + 1U, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool tryPop(ServoTrajectoryPublicationRecord & record) noexcept
  {
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto head = head_.load(std::memory_order_acquire);
    if (tail == head) {return false;}
    copy_servo_trajectory_record(records_[tail % Capacity], record);
    tail_.store(tail + 1U, std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::uint64_t droppedCount() const noexcept
  {
    return dropped_.load(std::memory_order_relaxed);
  }

  template<typename Consumer>
  bool tryConsume(Consumer && consumer)
  {
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto head = head_.load(std::memory_order_acquire);
    if (tail == head) {return false;}
    consumer(records_[tail % Capacity]);
    tail_.store(tail + 1U, std::memory_order_release);
    return true;
  }

private:
  std::unique_ptr<ServoTrajectoryPublicationRecord[]> records_;
  std::atomic<std::uint64_t> head_{0U};
  std::atomic<std::uint64_t> tail_{0U};
  std::atomic<std::uint64_t> dropped_{0U};
};

}  // namespace diagnostics
}  // namespace face_tracking_arm

#endif  // FACE_TRACKING_ARM__SERVO_TRAJECTORY_PUBLICATION_RING_HPP_
