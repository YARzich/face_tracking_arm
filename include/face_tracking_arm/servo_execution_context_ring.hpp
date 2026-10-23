// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__SERVO_EXECUTION_CONTEXT_RING_HPP_
#define FACE_TRACKING_ARM__SERVO_EXECUTION_CONTEXT_RING_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace face_tracking_arm::diagnostics
{

struct ServoExecutionContextRecord
{
  std::int64_t control_stamp_ns{0};
  std::int64_t target_source_stamp_ns{0};
  std::int64_t face_stamp_ns{0};
  std::int64_t tail_stationary_stamp_ns{0};
  std::uint64_t tick_sequence{0U};
  std::uint64_t hardware_epoch{0U};
  std::uint64_t hardware_read_sequence{0U};
  std::uint64_t active_publication_id{0U};
  std::uint64_t latest_publication_id{0U};
  std::uint64_t tail_publication_id{0U};
  std::uint64_t cycle_duration_ns{0U};
  std::uint64_t dropped_publication_events{0U};
  std::uint32_t active_collision_rows{0U};
  std::uint8_t target_mode{0U};
  std::uint8_t controller_mode{0U};
  std::int64_t target_id{-1};
  bool target_available{false};
  bool target_used_for_reference{false};
  bool face_tracking_published{false};
  bool collision_distance_available{false};
  double minimum_world_distance_m{std::numeric_limits<double>::quiet_NaN()};
  bool active_world_pair_available{false};
  double active_world_pair_distance_m{std::numeric_limits<double>::quiet_NaN()};
  double target_position[3]{};
  double face_position[3]{};
  char target_frame[128]{};
  char closest_pair[128]{};
  char active_world_pair[128]{};
};

template<std::size_t Capacity>
class ServoExecutionContextRing final
{
  static_assert(Capacity > 0U, "execution context ring must have capacity");

public:
  [[nodiscard]] bool tryPush(const ServoExecutionContextRecord & record) noexcept
  {
    const std::uint64_t head = head_.load(std::memory_order_relaxed);
    const std::uint64_t tail = tail_.load(std::memory_order_acquire);
    if (head - tail >= Capacity) {
      dropped_.fetch_add(1U, std::memory_order_relaxed);
      return false;
    }
    records_[head % Capacity] = record;
    head_.store(head + 1U, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool tryPop(ServoExecutionContextRecord & record) noexcept
  {
    const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
    const std::uint64_t head = head_.load(std::memory_order_acquire);
    if (tail == head) {
      return false;
    }
    record = records_[tail % Capacity];
    tail_.store(tail + 1U, std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::uint64_t droppedCount() const noexcept
  {
    return dropped_.load(std::memory_order_relaxed);
  }

private:
  std::array<ServoExecutionContextRecord, Capacity> records_{};
  std::atomic<std::uint64_t> head_{0U};
  std::atomic<std::uint64_t> tail_{0U};
  std::atomic<std::uint64_t> dropped_{0U};
};

}  // namespace face_tracking_arm::diagnostics

#endif  // FACE_TRACKING_ARM__SERVO_EXECUTION_CONTEXT_RING_HPP_
