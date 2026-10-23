// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_MAILBOX_HPP_
#define FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_MAILBOX_HPP_

#include <atomic>
#include <cstdint>
#include <optional>
#include <utility>

#include "face_tracking_arm/hierarchical_velocity_qp.hpp"

namespace face_tracking_arm::control
{

struct QpFailureSnapshotEvent
{
  std::uint64_t event_id{0U};
  std::int64_t ros_stamp_ns{0};
  std::int64_t steady_stamp_ns{0};
  std::uint64_t hardware_epoch{0U};
  std::uint64_t hardware_read_sequence{0U};
  std::uint64_t active_publication_id{0U};
  std::uint64_t latest_publication_id{0U};
  QpFailureSnapshotDisposition primary_disposition{QpFailureSnapshotDisposition::kNotNeeded};
  QpFailureSnapshotDisposition secondary_disposition{QpFailureSnapshotDisposition::kNotNeeded};
  std::optional<QpFailureSnapshot> primary;
  std::optional<QpFailureSnapshot> secondary;
};

struct QpFailureSnapshotContext
{
  std::int64_t ros_stamp_ns{0};
  std::int64_t steady_stamp_ns{0};
  std::uint64_t hardware_epoch{0U};
  std::uint64_t hardware_read_sequence{0U};
  std::uint64_t active_publication_id{0U};
  std::uint64_t latest_publication_id{0U};
};

/// Single-slot, nonblocking handoff from the control thread to a non-RT consumer.
/// Full/contended submissions are counted and dropped; no caller waits for I/O.
class QpFailureSnapshotMailbox
{
public:
  [[nodiscard]] bool trySubmit(
    QpResult && result,
    const QpFailureSnapshotContext & context = {}) noexcept
  {
    if (result.primary_snapshot_disposition == QpFailureSnapshotDisposition::kNotNeeded &&
      result.secondary_snapshot_disposition == QpFailureSnapshotDisposition::kNotNeeded)
    {
      return false;
    }
    std::uint8_t expected = 0U;
    if (!state_.compare_exchange_strong(
        expected, 1U, std::memory_order_acquire, std::memory_order_relaxed))
    {
      dropped_.fetch_add(1U, std::memory_order_relaxed);
      return false;
    }
    pending_.event_id = next_event_id_.fetch_add(1U, std::memory_order_relaxed);
    pending_.ros_stamp_ns = context.ros_stamp_ns;
    pending_.steady_stamp_ns = context.steady_stamp_ns;
    pending_.hardware_epoch = context.hardware_epoch;
    pending_.hardware_read_sequence = context.hardware_read_sequence;
    pending_.active_publication_id = context.active_publication_id;
    pending_.latest_publication_id = context.latest_publication_id;
    pending_.primary_disposition = result.primary_snapshot_disposition;
    pending_.secondary_disposition = result.secondary_snapshot_disposition;
    pending_.primary = std::move(result.primary_failure_snapshot);
    pending_.secondary = std::move(result.secondary_failure_snapshot);
    state_.store(2U, std::memory_order_release);
    return true;
  }

  [[nodiscard]] std::optional<QpFailureSnapshotEvent> tryTake() noexcept
  {
    std::uint8_t expected = 2U;
    if (!state_.compare_exchange_strong(
        expected, 3U, std::memory_order_acquire, std::memory_order_relaxed))
    {
      return std::nullopt;
    }
    std::optional<QpFailureSnapshotEvent> event{std::move(pending_)};
    pending_ = QpFailureSnapshotEvent{};
    state_.store(0U, std::memory_order_release);
    return event;
  }

  [[nodiscard]] std::uint64_t droppedCount() const noexcept
  {
    return dropped_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<std::uint8_t> state_{0U};  // empty, producer-owned, full, consumer-owned
  QpFailureSnapshotEvent pending_;
  std::atomic<std::uint64_t> next_event_id_{1U};
  std::atomic<std::uint64_t> dropped_{0U};
};

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__QP_FAILURE_SNAPSHOT_MAILBOX_HPP_
