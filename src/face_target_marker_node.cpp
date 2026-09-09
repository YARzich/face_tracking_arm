// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ros_gz_interfaces/msg/entity.hpp>
#include <ros_gz_interfaces/srv/set_entity_pose.hpp>
#include <tf2/exceptions.hpp>
#include <tf2/time.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

namespace face_tracking_arm
{
namespace
{
constexpr char kTargetTopic[] = "/face/center";
constexpr double kHiddenMarkerZ = -10.0;
constexpr double kPositionToleranceSquared = 1.0e-12;
// Gazebo's blocking set-pose endpoint may wait up to five seconds internally.
// Keep our request pending longer so a retry cannot overlap that in-flight command.
constexpr auto kServiceRequestTimeout = std::chrono::seconds(6);

bool is_finite(const geometry_msgs::msg::Point & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

double squared_distance(
  const geometry_msgs::msg::Point & left,
  const geometry_msgs::msg::Point & right)
{
  const double dx = left.x - right.x;
  const double dy = left.y - right.y;
  const double dz = left.z - right.z;
  return dx * dx + dy * dy + dz * dz;
}
}  // namespace

class FaceTargetMarkerNode final : public rclcpp::Node
{
public:
  FaceTargetMarkerNode()
  : Node("face_target_marker"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_, this, false)
  {
    planning_frame_ = declare_parameter<std::string>("planning_frame", "world");
    marker_entity_name_ = declare_parameter<std::string>(
      "marker_entity_name", "face_target_marker");
    set_pose_service_ = declare_parameter<std::string>(
      "set_pose_service", "/world/lite6_table/set_pose/blocking");
    const double update_rate_hz = declare_parameter<double>("update_rate_hz", 20.0);
    freshness_timeout_sec_ = declare_parameter<double>(
      "face_target_freshness_timeout_sec", 0.50);

    if (planning_frame_.empty()) {
      throw std::invalid_argument("planning_frame must not be empty");
    }
    if (marker_entity_name_.empty()) {
      throw std::invalid_argument("marker_entity_name must not be empty");
    }
    if (set_pose_service_.empty()) {
      throw std::invalid_argument("set_pose_service must not be empty");
    }
    if (!std::isfinite(update_rate_hz) || update_rate_hz <= 0.0) {
      throw std::invalid_argument("update_rate_hz must be finite and positive");
    }
    if (!std::isfinite(freshness_timeout_sec_) || freshness_timeout_sec_ <= 0.0) {
      throw std::invalid_argument(
              "face_target_freshness_timeout_sec must be finite and positive");
    }

    pose_client_ = create_client<ros_gz_interfaces::srv::SetEntityPose>(set_pose_service_);
    target_subscription_ = create_subscription<geometry_msgs::msg::PointStamped>(
      kTargetTopic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      [this](const geometry_msgs::msg::PointStamped::ConstSharedPtr message) {
        receive_target(*message);
      });

    const auto update_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / update_rate_hz));
    update_timer_ = create_wall_timer(update_period, [this]() {update_marker();});

    RCLCPP_INFO(
      get_logger(), "Gazebo target marker initialized: %s -> %s at %.1f Hz",
      kTargetTopic, marker_entity_name_.c_str(), update_rate_hz);
  }

private:
  void receive_target(const geometry_msgs::msg::PointStamped & message)
  {
    const rclcpp::Time current_time = now();
    observe_clock(current_time);
    if (message.header.frame_id.empty() || !is_finite(message.point)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Ignoring target marker sample with an empty frame or non-finite point");
      return;
    }

    const rclcpp::Time stamp(message.header.stamp, get_clock()->get_clock_type());
    const double age_sec = (current_time - stamp).seconds();
    if (
      stamp.nanoseconds() <= 0 || !std::isfinite(age_sec) || age_sec < 0.0 ||
      age_sec >= freshness_timeout_sec_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Ignoring target marker sample with an invalid, future, or stale stamp");
      return;
    }
    if (latest_target_stamp_ && stamp <= *latest_target_stamp_) {
      return;
    }
    latest_target_ = message;
    latest_target_stamp_ = stamp;
  }

  void observe_clock(const rclcpp::Time & current_time)
  {
    if (last_observed_time_ && current_time < *last_observed_time_) {
      latest_target_.reset();
      latest_target_stamp_.reset();
      resolved_target_.reset();
      resolved_target_stamp_.reset();
      last_commanded_position_.reset();
      RCLCPP_WARN(get_logger(), "ROS clock moved backwards; clearing target marker state");
    }
    last_observed_time_ = current_time;
  }

  std::optional<geometry_msgs::msg::Point> current_target_in_planning_frame()
  {
    if (!latest_target_ || !latest_target_stamp_) {
      return std::nullopt;
    }

    if (!resolved_target_stamp_ || *latest_target_stamp_ > *resolved_target_stamp_) {
      try {
        geometry_msgs::msg::Point resolved_point;
        if (latest_target_->header.frame_id == planning_frame_) {
          resolved_point = latest_target_->point;
        } else {
          const auto transformed = tf_buffer_.transform(
            *latest_target_, planning_frame_, tf2::durationFromSec(0.0));
          resolved_point = transformed.point;
        }
        if (is_finite(resolved_point)) {
          resolved_target_ = resolved_point;
          resolved_target_stamp_ = latest_target_stamp_;
        }
      } catch (const tf2::TransformException & error) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Cannot transform target marker from '%s' to '%s': %s",
          latest_target_->header.frame_id.c_str(), planning_frame_.c_str(), error.what());
      }
    }

    if (!resolved_target_ || !resolved_target_stamp_) {
      return std::nullopt;
    }

    const double age_sec = (now() - *resolved_target_stamp_).seconds();
    if (!std::isfinite(age_sec) || age_sec < 0.0 || age_sec >= freshness_timeout_sec_) {
      return std::nullopt;
    }
    return resolved_target_;
  }

  void update_marker()
  {
    observe_clock(now());
    if (request_pending_) {
      if (
        pending_request_id_ &&
        std::chrono::steady_clock::now() - pending_request_started_at_ >
        kServiceRequestTimeout)
      {
        pose_client_->remove_pending_request(*pending_request_id_);
        pending_request_id_.reset();
        request_pending_ = false;
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Gazebo marker pose request timed out; retrying the newest state");
      }
      return;
    }
    if (!pose_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for Gazebo set-pose service '%s'", set_pose_service_.c_str());
      return;
    }

    const auto target = current_target_in_planning_frame();
    if (target) {
      if (
        marker_visible_ && last_commanded_position_ &&
        squared_distance(*target, *last_commanded_position_) <= kPositionToleranceSquared)
      {
        return;
      }
      send_pose(*target, true);
      return;
    }

    if (marker_visible_) {
      geometry_msgs::msg::Point hidden_position;
      hidden_position.z = kHiddenMarkerZ;
      send_pose(hidden_position, false);
    }
  }

  void send_pose(const geometry_msgs::msg::Point & position, const bool visible)
  {
    auto request = std::make_shared<ros_gz_interfaces::srv::SetEntityPose::Request>();
    request->entity.name = marker_entity_name_;
    request->entity.type = ros_gz_interfaces::msg::Entity::MODEL;
    request->pose.position = position;
    request->pose.orientation.w = 1.0;

    request_pending_ = true;
    pending_request_started_at_ = std::chrono::steady_clock::now();
    const auto request_handle = pose_client_->async_send_request(
      request,
      [this, position, visible](
        const rclcpp::Client<ros_gz_interfaces::srv::SetEntityPose>::SharedFuture future)
      {
        request_pending_ = false;
        pending_request_id_.reset();
        const auto response = future.get();
        if (!response->success) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000,
            "Gazebo rejected a pose update for marker '%s'",
            marker_entity_name_.c_str());
          return;
        }
        marker_visible_ = visible;
        if (visible) {
          last_commanded_position_ = position;
        } else {
          last_commanded_position_.reset();
        }
      });
    pending_request_id_ = request_handle.request_id;
  }

  std::string planning_frame_;
  std::string marker_entity_name_;
  std::string set_pose_service_;
  double freshness_timeout_sec_{0.50};
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Client<ros_gz_interfaces::srv::SetEntityPose>::SharedPtr pose_client_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_subscription_;
  rclcpp::TimerBase::SharedPtr update_timer_;
  std::optional<geometry_msgs::msg::PointStamped> latest_target_;
  std::optional<rclcpp::Time> latest_target_stamp_;
  std::optional<geometry_msgs::msg::Point> resolved_target_;
  std::optional<rclcpp::Time> resolved_target_stamp_;
  std::optional<rclcpp::Time> last_observed_time_;
  std::optional<geometry_msgs::msg::Point> last_commanded_position_;
  std::optional<std::int64_t> pending_request_id_;
  std::chrono::steady_clock::time_point pending_request_started_at_{};
  bool request_pending_{false};
  bool marker_visible_{false};
};
}  // namespace face_tracking_arm

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<face_tracking_arm::FaceTargetMarkerNode>());
  rclcpp::shutdown();
  return 0;
}
