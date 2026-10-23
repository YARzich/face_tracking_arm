// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>

namespace face_tracking_arm
{
namespace hardware
{

// With the runtime policy armed, STOP owns the transition before a controller
// reset/deactivation can start another SDK-side action.
inline bool defer_controller_deactivation_until_stop(
  bool runtime_actions_enabled, bool hardware_active) noexcept
{
  return runtime_actions_enabled && hardware_active;
}

inline bool refuse_activation_after_latched_fault(
  bool runtime_actions_enabled, bool inhibit_latched, bool fault_latched) noexcept
{
  return runtime_actions_enabled && (inhibit_latched || fault_latched);
}

// Keep the status publisher alive for an active gated disconnect so the
// monitor can record the latched failure; legacy mode retains process exit.
inline bool should_shutdown_on_disconnected_read(
  bool runtime_actions_enabled, bool read_still_in_active_epoch) noexcept
{
  return !runtime_actions_enabled || !read_still_in_active_epoch;
}

// One-shot admission gate: close is atomic, and try_enter never retries.
class HardwareIoWriteGate
{
public:
  bool try_enter() noexcept
  {
    auto state = state_.load(std::memory_order_acquire);
    if ((state & kClosed) != 0 || (state & kCountMask) != 0) {return false;}
    return state_.compare_exchange_strong(
            state, state + 1, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  void leave() noexcept
  {
    state_.fetch_sub(1, std::memory_order_release);
  }

  void close() noexcept
  {
    state_.fetch_or(kClosed, std::memory_order_acq_rel);
  }

  std::uint32_t in_flight() const noexcept
  {
    return state_.load(std::memory_order_acquire) & kCountMask;
  }

  bool closed() const noexcept
  {
    return (state_.load(std::memory_order_acquire) & kClosed) != 0;
  }

private:
  static constexpr std::uint32_t kClosed = std::uint32_t{1} << 31;
  static constexpr std::uint32_t kCountMask = kClosed - 1;
  std::atomic<std::uint32_t> state_{0};
};

}  // namespace hardware
}  // namespace face_tracking_arm
