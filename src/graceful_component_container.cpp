// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <ctime>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/component_manager.hpp>

namespace face_tracking_arm
{

class GracefulComponentManager final : public rclcpp_components::ComponentManager
{
public:
  using rclcpp_components::ComponentManager::ComponentManager;

  bool unload_all_components()
  {
    bool success = true;
    // Reverse load order stops Servo before unloading the tracking adapter.
    while (!node_wrappers_.empty()) {
      const auto component = std::prev(node_wrappers_.end());
      try {
        remove_node_from_executor(component->first);
      } catch (const std::exception & error) {
        RCLCPP_ERROR(
          get_logger(), "Failed to detach component %" PRIu64 " during shutdown: %s",
          static_cast<std::uint64_t>(component->first), error.what());
        success = false;
      }
      node_wrappers_.erase(component);
    }
    return success;
  }

  bool load_failed() const
  {
    return load_failed_.load(std::memory_order_acquire);
  }

protected:
  void on_load_node(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<LoadNode::Request> request,
    std::shared_ptr<LoadNode::Response> response) override
  {
    rclcpp_components::ComponentManager::on_load_node(request_header, request, response);
    if (!response->success) {
      load_failed_.store(true, std::memory_order_release);
      RCLCPP_ERROR(get_logger(), "Component load failed: %s", response->error_message.c_str());
      if (const auto executor = executor_.lock()) {
        executor->cancel();
      }
    }
  }

private:
  std::atomic<bool> load_failed_{false};
};

}  // namespace face_tracking_arm

int main(int argc, char * argv[])
{
  sigset_t shutdown_signals;
  sigemptyset(&shutdown_signals);
  sigaddset(&shutdown_signals, SIGINT);
  sigaddset(&shutdown_signals, SIGTERM);
  const int mask_result = pthread_sigmask(SIG_BLOCK, &shutdown_signals, nullptr);
  if (mask_result != 0) {
    std::cerr << "Unable to block shutdown signals: " << mask_result << '\n';
    return EXIT_FAILURE;
  }

  rclcpp::init(
    argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  auto component_manager =
    std::make_shared<face_tracking_arm::GracefulComponentManager>(executor);
  executor->add_node(component_manager);

  std::exception_ptr spin_error;
  std::atomic<bool> spin_finished{false};
  std::thread executor_thread(
    [&executor, &spin_error, &spin_finished]() {
      try {
        executor->spin();
      } catch (...) {
        spin_error = std::current_exception();
      }
      spin_finished.store(true, std::memory_order_release);
    });

  while (!executor->is_spinning() &&
    !spin_finished.load(std::memory_order_acquire))
  {
    std::this_thread::yield();
  }

  int received_signal = 0;
  bool signal_wait_failed = false;
  if (!spin_finished.load(std::memory_order_acquire)) {
    while (!spin_finished.load(std::memory_order_acquire)) {
      const timespec timeout{0, 100'000'000};
      const int wait_result = sigtimedwait(&shutdown_signals, nullptr, &timeout);
      if (wait_result == SIGINT || wait_result == SIGTERM) {
        received_signal = wait_result;
        break;
      }
      if (wait_result == -1 && (errno == EAGAIN || errno == EINTR)) {
        continue;
      }
      std::cerr << "Unable to wait for a shutdown signal: " << errno << '\n';
      signal_wait_failed = true;
      break;
    }
  }

  executor->cancel();
  executor_thread.join();
  const bool load_failed = component_manager->load_failed();
  // Pre-shutdown callbacks stop component-owned workers and release MoveIt's
  // references to their parent nodes before their plugin libraries are unloaded.
  rclcpp::shutdown();
  const bool cleanup_succeeded = component_manager->unload_all_components();
  executor->remove_node(component_manager);
  component_manager.reset();
  executor.reset();

  if (spin_error) {
    try {
      std::rethrow_exception(spin_error);
    } catch (const std::exception & error) {
      std::cerr << "Component executor failed: " << error.what() << '\n';
    }
    return EXIT_FAILURE;
  }
  const bool expected_signal = received_signal == SIGINT || received_signal == SIGTERM;
  return expected_signal && !signal_wait_failed && !load_failed && cleanup_succeeded ?
         EXIT_SUCCESS : EXIT_FAILURE;
}
