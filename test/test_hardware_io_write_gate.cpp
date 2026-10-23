// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "hardware_io_write_gate.hpp"

namespace
{
bool check(bool condition, const char * message)
{
  if (!condition) {std::cerr << "FAILED: " << message << '\n';}
  return condition;
}
}

int main(const int argc, char ** argv)
{
  using face_tracking_arm::hardware::HardwareIoWriteGate;
  using face_tracking_arm::hardware::defer_controller_deactivation_until_stop;
  using face_tracking_arm::hardware::refuse_activation_after_latched_fault;
  using face_tracking_arm::hardware::should_shutdown_on_disconnected_read;
  if (argc == 2 && std::strcmp(argv[1], "--emit-cross-layer-trace") == 0) {
    HardwareIoWriteGate trace_gate;
    constexpr std::uint64_t trace_epoch = 7;
    if (!trace_gate.try_enter()) {
      return 2;
    }
    std::cout << "{\"event\":\"write_admitted\",\"epoch\":" << trace_epoch
              << ",\"write_in_flight\":" << trace_gate.in_flight() << "}\n";
    // This is the same gate close used by the driver's first read-fault latch.
    trace_gate.close();
    std::cout << "{\"event\":\"read_fault_latched\",\"epoch\":" << trace_epoch
              << ",\"fault_latched\":true,\"inhibit_latched\":true"
              << ",\"inhibit_epoch\":" << trace_epoch
              << ",\"inhibit_request_id\":11,\"runtime_actions_enabled\":true"
              << ",\"hardware_active\":true,\"write_in_flight\":"
              << trace_gate.in_flight() << "}\n";
    if (trace_gate.try_enter()) {
      return 3;
    }
    trace_gate.leave();
    std::cout << "{\"event\":\"admitted_write_drained\",\"epoch\":" << trace_epoch
              << ",\"fault_latched\":true,\"inhibit_latched\":true"
              << ",\"inhibit_epoch\":" << trace_epoch
              << ",\"inhibit_request_id\":11,\"runtime_actions_enabled\":true"
              << ",\"hardware_active\":true,\"write_in_flight\":"
              << trace_gate.in_flight() << "}\n";
    return trace_gate.closed() && trace_gate.in_flight() == 0 ? 0 : 4;
  }
  bool passed = check(defer_controller_deactivation_until_stop(true, true),
                        "runtime fault defers lifecycle deactivation until STOP");
  passed &= check(!defer_controller_deactivation_until_stop(false, true),
                    "stage 3 gate off keeps legacy lifecycle behavior");
  passed &= check(!defer_controller_deactivation_until_stop(true, false),
                    "inactive hardware does not create a runtime stop transition");
  passed &= check(refuse_activation_after_latched_fault(true, true, false),
                    "latched inhibit requires operator recovery before reactivation");
  passed &= check(refuse_activation_after_latched_fault(true, false, true),
                    "first fault remains latched through lifecycle transitions");
  passed &= check(!refuse_activation_after_latched_fault(false, true, true),
                    "stage 3 gate off preserves legacy activation policy");
  passed &= check(!should_shutdown_on_disconnected_read(true, true),
                    "active gated disconnect leaves status publisher alive");
  passed &= check(should_shutdown_on_disconnected_read(true, false),
                    "inactive disconnect keeps legacy process exit");
  passed &= check(should_shutdown_on_disconnected_read(false, true),
                    "gate-off disconnect keeps legacy process exit");
  HardwareIoWriteGate gate;
  std::mutex mutex;
  std::condition_variable condition;
  bool admitted = false;
  bool release_write = false;
  std::atomic<bool> failed{false};
  std::thread in_flight([&]() {
      if (!gate.try_enter()) {
        failed.store(true, std::memory_order_release);
        condition.notify_one();
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        admitted = true;
      }
      condition.notify_one();
      {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&]() {return release_write;});
      }
      gate.leave();
    });
  {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&]() {
        return admitted || failed.load(std::memory_order_acquire);
        });
  }
  passed &= check(!failed.load(std::memory_order_acquire), "first write is admitted");

    // This models a STOP/inhibit callback racing after SDK-write admission.
  gate.close();
  passed &= check(gate.closed(), "gate closes permanently");
  passed &= check(gate.in_flight() == 1, "admitted write remains in flight");
  passed &= check(!gate.try_enter(), "queued write cannot cross latch");
  {
    std::lock_guard<std::mutex> lock(mutex);
    release_write = true;
  }
  condition.notify_one();
  in_flight.join();
  passed &= check(gate.in_flight() == 0, "in-flight write leaves exactly once");
  passed &= check(!gate.try_enter(), "closed gate rejects later writes");

    // A bounded stress pass checks that close stays terminal under contention.
  HardwareIoWriteGate stressed;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  for (int index = 0; index < 8; ++index) {
    workers.emplace_back([&]() {
        while (!start.load(std::memory_order_acquire)) {}
        for (int attempt = 0; attempt < 10000; ++attempt) {
          if (stressed.try_enter()) {
            stressed.leave();
          }
        }
        });
  }
  start.store(true, std::memory_order_release);
  stressed.close();
  for (auto & worker : workers) {
    worker.join();
  }
  passed &= check(stressed.closed(), "stress gate remains closed");
  passed &= check(stressed.in_flight() == 0, "stress workers leave balanced count");
  passed &= check(!stressed.try_enter(), "stress gate rejects later writes");

    // Exercise the exact gate and lifecycle decision used by the patched
    // driver: a read fault closes ServoJ admission, the following reset path
    // retains its active epoch until STOP, and the monitor may proceed only
    // after the admitted write has left and the latch still matches.
  HardwareIoWriteGate driver_gate;
  constexpr std::uint64_t active_epoch = 23;
  std::uint64_t inhibit_epoch = 0;
  bool fault_latched = false;
  bool hardware_active = true;
  fault_latched = true;
  inhibit_epoch = active_epoch;
  driver_gate.close();
  passed &= check(!driver_gate.try_enter(), "read-fault latch blocks next write");
  const bool reset_path_must_defer = defer_controller_deactivation_until_stop(
        true, hardware_active);
  const std::uint64_t status_epoch = reset_path_must_defer ? active_epoch : active_epoch + 1;
  passed &= check(reset_path_must_defer, "active gated reset waits for STOP");
  passed &= check(status_epoch == inhibit_epoch && fault_latched,
                    "fault and inhibit provenance remain in active epoch");
  const bool stop_may_be_requested = fault_latched && driver_gate.closed() &&
    driver_gate.in_flight() == 0 && status_epoch == inhibit_epoch;
  passed &= check(stop_may_be_requested,
                    "monitor can progress to STOP with matching settled latch");
  return passed ? 0 : 1;
}
