// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include "test_face_scenarios.hpp"

namespace face_tracking_arm
{
namespace
{
bool reports_ready(const diagnostic_msgs::msg::DiagnosticStatus & status)
{
  if (status.name != "face_tracking_arm/tracking" ||
    status.level != diagnostic_msgs::msg::DiagnosticStatus::OK)
  {
    return false;
  }
  for (const auto & value : status.values) {
    if (value.key == "ready" && value.value == "true") {
      return true;
    }
  }
  return false;
}

}  // namespace

class TestFaceTargetPublisherNode final : public rclcpp::Node
{
public:
  TestFaceTargetPublisherNode()
  : Node("test_face_target_publisher")
  {
    const auto scenario_name = declare_parameter<std::string>("scenario", "people");
    scenario_ = test_targets::parse_scenario(scenario_name);

    target_publisher_ = create_publisher<geometry_msgs::msg::PointStamped>(
      "/face/center", rclcpp::QoS(1).best_effort());
    diagnostics_subscription_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/tracking/diagnostics", rclcpp::QoS(1).reliable(),
      [this](const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr message) {
        if (started_at_) {
          return;
        }
        for (const auto & status : message->status) {
          if (reports_ready(status)) {
            started_at_ = now();
            RCLCPP_INFO(
              get_logger(), "Tracking is ready; starting test face scenario");
            return;
          }
        }
      });
    publish_timer_ = create_wall_timer(
      std::chrono::milliseconds(20), [this]() {publish_sample();});
  }

private:
  void publish_sample()
  {
    if (!started_at_) {
      return;
    }

    const rclcpp::Time sample_time = now();
    const double elapsed = (sample_time - *started_at_).seconds();
    if (elapsed < 0.0) {
      started_at_ = sample_time;
      last_phase_.reset();
      return;
    }
    const test_targets::Sample sample = test_targets::sample(scenario_, elapsed);
    if (!last_phase_ || *last_phase_ != sample.phase) {
      last_phase_ = sample.phase;
      RCLCPP_INFO(
        get_logger(), "%.*s", static_cast<int>(sample.phase.size()), sample.phase.data());
    }
    if (!sample.target) {
      return;
    }

    geometry_msgs::msg::PointStamped target;
    target.header.stamp = sample_time;
    target.header.frame_id = "world";
    target.point.x = sample.target->x;
    target.point.y = sample.target->y;
    target.point.z = sample.target->z;
    target_publisher_->publish(target);
  }

  test_targets::Scenario scenario_;
  std::optional<rclcpp::Time> started_at_;
  std::optional<std::string_view> last_phase_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_publisher_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_subscription_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};
}  // namespace face_tracking_arm

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<face_tracking_arm::TestFaceTargetPublisherNode>());
  rclcpp::shutdown();
  return 0;
}
