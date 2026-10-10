// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "face_tracking_arm/table_geometry.hpp"
#include "face_tracking_arm/tracking_core.hpp"
#include "face_tracking_arm/msg/tracking_target.hpp"
#include "face_tracking_arm/msg/perception_status.hpp"

#include <controller_manager_msgs/srv/list_controllers.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/allowed_collision_matrix.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/msg/servo_status.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/servo_command_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <tf2/exceptions.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

namespace face_tracking_arm
{
namespace
{
using namespace std::chrono_literals;

constexpr char kFaceTopic[] = "/face/center";
constexpr char kJointStateTopic[] = "/joint_states";
constexpr char kPoseCommandTopic[] = "/servo_node/pose_target_cmds";
constexpr char kServoStatusTopic[] = "/servo_node/status";
constexpr char kDiagnosticsTopic[] = "/tracking/diagnostics";
constexpr char kPlanningSceneTopic[] = "/planning_scene";
constexpr char kGetPlanningSceneService[] = "/get_planning_scene";
constexpr char kSwitchCommandTypeService[] = "/servo_node/switch_command_type";
constexpr char kListControllersService[] = "/controller_manager/list_controllers";
constexpr char kTableId[] = "round_table";
constexpr std::int64_t kJointStateTimeoutNs = 100'000'000;
constexpr std::int64_t kTcpPoseTimeoutNs = 30'000'000;
constexpr std::int64_t kServoStatusTimeoutNs = 200'000'000;
constexpr std::int64_t kMaximumClockSkewNs = 10'000'000;
constexpr auto kReadinessRequestTimeout = 2s;
constexpr std::array<const char *, 6> kArmJoints = {
  "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

std::int64_t seconds_to_nanoseconds(const double seconds, const char * parameter_name)
{
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    throw std::invalid_argument(std::string(parameter_name) + " must be finite and positive");
  }
  return static_cast<std::int64_t>(std::llround(seconds * 1.0e9));
}

bool finite_point(const geometry_msgs::msg::Point & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

std::string bool_string(const bool value)
{
  return value ? "true" : "false";
}

diagnostic_msgs::msg::KeyValue key_value(std::string key, std::string value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = std::move(key);
  result.value = std::move(value);
  return result;
}

const char * mode_name(const tracking::TargetMode mode)
{
  switch (mode) {
    case tracking::TargetMode::kFace:
      return "FACE";
    case tracking::TargetMode::kHold:
      return "HOLD";
    case tracking::TargetMode::kRest:
      return "REST";
    case tracking::TargetMode::kSearch:
      return "SEARCH";
  }
  return "UNKNOWN";
}

const char * reject_reason_name(const tracking::RejectReason reason)
{
  switch (reason) {
    case tracking::RejectReason::kNone:
      return "none";
    case tracking::RejectReason::kEmptyFrame:
      return "empty_frame";
    case tracking::RejectReason::kUnexpectedFrame:
      return "unexpected_frame";
    case tracking::RejectReason::kNonFinite:
      return "non_finite";
    case tracking::RejectReason::kInvalidConfiguration:
      return "invalid_configuration";
    case tracking::RejectReason::kNegativeStamp:
      return "negative_stamp";
    case tracking::RejectReason::kFutureStamp:
      return "future_stamp";
    case tracking::RejectReason::kStale:
      return "stale";
    case tracking::RejectReason::kNotNewer:
      return "not_newer";
    case tracking::RejectReason::kDegenerateDirection:
      return "degenerate_direction";
    case tracking::RejectReason::kFaceInsideMinimumDistance:
      return "face_inside_minimum_distance";
    case tracking::RejectReason::kDegenerateUpProjection:
      return "degenerate_up_projection";
    case tracking::RejectReason::kReachEnvelopeUnavailable:
      return "reach_envelope_unavailable";
  }
  return "unknown";
}

bool is_dangerous_servo_status(const std::int8_t code)
{
  return code == moveit_msgs::msg::ServoStatus::HALT_FOR_SINGULARITY ||
         code == moveit_msgs::msg::ServoStatus::HALT_FOR_COLLISION ||
         code == moveit_msgs::msg::ServoStatus::JOINT_BOUND;
}

bool is_known_servo_status(const std::int8_t code)
{
  return code >= moveit_msgs::msg::ServoStatus::NO_WARNING &&
         code <= moveit_msgs::msg::ServoStatus::JOINT_BOUND;
}

bool collision_pair_allowed(
  const moveit_msgs::msg::AllowedCollisionMatrix & matrix,
  const std::string & first,
  const std::string & second)
{
  const auto first_iterator = std::find(matrix.entry_names.begin(), matrix.entry_names.end(),
        first);
  const auto second_iterator = std::find(matrix.entry_names.begin(), matrix.entry_names.end(),
        second);
  if (first_iterator == matrix.entry_names.end() || second_iterator == matrix.entry_names.end()) {
    return false;
  }
  const auto first_index = static_cast<std::size_t>(
    std::distance(matrix.entry_names.begin(), first_iterator));
  const auto second_index = static_cast<std::size_t>(
    std::distance(matrix.entry_names.begin(), second_iterator));
  return first_index < matrix.entry_values.size() &&
         second_index < matrix.entry_values[first_index].enabled.size() &&
         matrix.entry_values[first_index].enabled[second_index];
}

void allow_collision_pair(
  moveit_msgs::msg::AllowedCollisionMatrix & matrix,
  const std::string & first,
  const std::string & second)
{
  auto ensure_square = [&matrix]() {
      matrix.entry_values.resize(matrix.entry_names.size());
      for (auto & row : matrix.entry_values) {
        row.enabled.resize(matrix.entry_names.size(), false);
      }
    };
  ensure_square();

  auto ensure_name = [&matrix, &ensure_square](const std::string & name) {
      auto iterator = std::find(matrix.entry_names.begin(), matrix.entry_names.end(), name);
      if (iterator == matrix.entry_names.end()) {
        matrix.entry_names.push_back(name);
        ensure_square();
        return matrix.entry_names.size() - 1;
      }
      return static_cast<std::size_t>(std::distance(matrix.entry_names.begin(), iterator));
    };

  const std::size_t first_index = ensure_name(first);
  const std::size_t second_index = ensure_name(second);
  matrix.entry_values[first_index].enabled[second_index] = true;
  matrix.entry_values[second_index].enabled[first_index] = true;
}

}  // namespace

class FaceTrackingComponent final : public rclcpp::Node
{
public:
  explicit FaceTrackingComponent(const rclcpp::NodeOptions & options)
  : Node("face_tracking_controller", options), tf_buffer_(get_clock())
  {
    const double minimum_face_distance_m =
      declare_parameter<double>("minimum_face_distance_m", 0.40);
    const double safe_reach_radius_m =
      declare_parameter<double>("safe_reach_radius_m", 0.42);
    const double freshness_sec =
      declare_parameter<double>("face_target_freshness_timeout_sec", 0.50);
    const double return_delay_sec = declare_parameter<double>("return_to_rest_delay_sec", 2.0);
    const double prediction_timeout_sec = declare_parameter<double>(
      "face_prediction_timeout_sec", 0.0);
    require_perception_status_ = declare_parameter<bool>("require_perception_status", false);
    rcl_interfaces::msg::ParameterDescriptor idle_descriptor;
    idle_descriptor.read_only = true;
    table_ = tracking::make_table(
      declare_parameter<std::string>("table_shape", "cylinder", idle_descriptor),
      declare_parameter<std::vector<double>>("table_dimensions_m", {0.05, 0.50}, idle_descriptor),
      declare_parameter<std::vector<double>>("table_center_m", {0.0, 0.0, 0.725}, idle_descriptor),
      declare_parameter<double>("table_yaw_rad", 0.0, idle_descriptor));
    idle_behavior_ = declare_parameter<std::string>("idle_behavior", "rest", idle_descriptor);
    if (idle_behavior_ != "rest" && idle_behavior_ != "search_sweep" &&
      idle_behavior_ != "search_local_then_sweep")
    {
      throw std::invalid_argument("Unknown idle_behavior");
    }
    const std::vector<double> rest_position =
      declare_parameter<std::vector<double>>("rest_position_m", {0.20, 0.00, 1.05});
    planning_frame_ = declare_parameter<std::string>("planning_frame", "world");
    base_frame_ = declare_parameter<std::string>("base_frame", "link_base");
    reach_center_frame_ = declare_parameter<std::string>("reach_center_frame", "link1");
    control_frame_ = declare_parameter<std::string>("control_frame", "monitor_control_frame");

    if (rest_position.size() != 3) {
      throw std::invalid_argument("rest_position_m must contain exactly three values");
    }
    if (planning_frame_.empty() || base_frame_.empty() || reach_center_frame_.empty() ||
      control_frame_.empty())
    {
      throw std::invalid_argument(
              "planning_frame, base_frame, reach_center_frame, and control_frame "
              "must not be empty");
    }

    tracking::TrackingConfig config;
    config.search_when_idle = idle_behavior_ != "rest";
    config.geometry.minimum_face_distance_m = minimum_face_distance_m;
    config.geometry.safe_reach_radius_m = safe_reach_radius_m;
    freshness_timeout_ns_ = seconds_to_nanoseconds(
      freshness_sec, "face_target_freshness_timeout_sec");
    config.freshness_ns = freshness_timeout_ns_;
    config.return_delay_ns = seconds_to_nanoseconds(
      return_delay_sec, "return_to_rest_delay_sec");
    if (!std::isfinite(prediction_timeout_sec) || prediction_timeout_sec < 0.0) {
      throw std::invalid_argument("face_prediction_timeout_sec must be finite and non-negative");
    }
    config.prediction_timeout_ns = prediction_timeout_sec == 0.0 ? 0 :
      seconds_to_nanoseconds(prediction_timeout_sec, "face_prediction_timeout_sec");
    config_prediction_enabled_ = prediction_timeout_sec > 0.0;
    config.rest_position = {rest_position[0], rest_position[1], rest_position[2]};
    config.planning_frame = planning_frame_;
    core_ = std::make_unique<tracking::TrackingCore>(std::move(config));

    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_, this, false);

    pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      kPoseCommandTopic, rclcpp::QoS(1).reliable());
    tracking_target_publisher_ = create_publisher<msg::TrackingTarget>(
      "/servo_node/tracking_target_cmds", rclcpp::QoS(1).reliable());
    diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      kDiagnosticsTopic, rclcpp::QoS(1).reliable());
    planning_scene_publisher_ = create_publisher<moveit_msgs::msg::PlanningScene>(
      kPlanningSceneTopic, rclcpp::QoS(1).reliable());

    face_subscription_ = create_subscription<geometry_msgs::msg::PointStamped>(
      kFaceTopic, rclcpp::QoS(1).best_effort(),
      [this](geometry_msgs::msg::PointStamped::ConstSharedPtr message) {
        receive_face(std::move(message));
      });
    if (require_perception_status_) {
      perception_status_subscription_ = create_subscription<msg::PerceptionStatus>(
        "/face/perception_status", rclcpp::QoS(10).best_effort(),
        [this](msg::PerceptionStatus::ConstSharedPtr message) {
          std::lock_guard<std::mutex> lock(perception_mutex_);
          latest_perception_status_ = *message;
          perception_status_received_at_ = std::chrono::steady_clock::now();
        });
    }
    joint_state_subscription_ = create_subscription<sensor_msgs::msg::JointState>(
      kJointStateTopic, rclcpp::SensorDataQoS().keep_last(1),
      [this](const sensor_msgs::msg::JointState::ConstSharedPtr message) {
        receive_joint_state(*message);
      });
    servo_status_subscription_ = create_subscription<moveit_msgs::msg::ServoStatus>(
      kServoStatusTopic, rclcpp::QoS(10).reliable(),
      [this](const moveit_msgs::msg::ServoStatus::ConstSharedPtr message) {
        receive_servo_status(*message);
      });

    controller_client_ = create_client<controller_manager_msgs::srv::ListControllers>(
      kListControllersService);
    planning_scene_client_ = create_client<moveit_msgs::srv::GetPlanningScene>(
      kGetPlanningSceneService);
    command_type_client_ = create_client<moveit_msgs::srv::ServoCommandType>(
      kSwitchCommandTypeService);

    control_timer_ = create_wall_timer(10ms, [this]() {control_tick();});
    readiness_timer_ = create_wall_timer(250ms, [this]() {readiness_tick();});
    diagnostics_timer_ = create_wall_timer(500ms, [this]() {publish_diagnostics();});

    RCLCPP_INFO(
      get_logger(),
      "Reactive tracking initialized: %s -> %s, reach center=%s, TCP=%s, period=10 ms",
      base_frame_.c_str(), planning_frame_.c_str(), reach_center_frame_.c_str(),
      control_frame_.c_str());
  }

private:
  struct PendingFace
  {
    geometry_msgs::msg::PointStamped message;
    std::chrono::steady_clock::time_point received_at;
  };

  void receive_face(geometry_msgs::msg::PointStamped::ConstSharedPtr message)
  {
    const std::int64_t now_ns = now().nanoseconds();
    const std::int64_t stamp_ns = rclcpp::Time(
      message->header.stamp, get_clock()->get_clock_type()).nanoseconds();
    if (message->header.frame_id.empty()) {
      reject_face("empty_frame");
      return;
    }
    if (!finite_point(message->point)) {
      reject_face("non_finite");
      return;
    }
    if (stamp_ns <= 0) {
      reject_face("unset_stamp");
      return;
    }
    if (stamp_ns - now_ns > kMaximumClockSkewNs) {
      reject_face("future_stamp");
      return;
    }
    if (stamp_ns <= now_ns && now_ns - stamp_ns >= freshness_timeout_ns_) {
      reject_face("stale");
      return;
    }
    if (last_accepted_face_stamp_ns_.has_value() &&
      stamp_ns <= *last_accepted_face_stamp_ns_)
    {
      reject_face("not_newer");
      return;
    }
    if (pending_face_.has_value()) {
      const std::int64_t pending_stamp_ns = rclcpp::Time(
        pending_face_->message.header.stamp, get_clock()->get_clock_type()).nanoseconds();
      if (stamp_ns <= pending_stamp_ns) {
        reject_face("not_newer");
        return;
      }
    }

    pending_face_ = PendingFace{*message, std::chrono::steady_clock::now()};
    ++received_face_count_;
  }

  void receive_joint_state(const sensor_msgs::msg::JointState & message)
  {
    if (message.name.size() != message.position.size()) {
      joint_state_complete_ = false;
      return;
    }
    for (const char * required_joint : kArmJoints) {
      const auto iterator = std::find(message.name.begin(), message.name.end(), required_joint);
      if (iterator == message.name.end()) {
        joint_state_complete_ = false;
        return;
      }
      const auto index = static_cast<std::size_t>(std::distance(message.name.begin(), iterator));
      if (!std::isfinite(message.position[index])) {
        joint_state_complete_ = false;
        return;
      }
    }

    const std::int64_t stamp_ns = rclcpp::Time(
      message.header.stamp, get_clock()->get_clock_type()).nanoseconds();
    if (stamp_ns < 0) {
      joint_state_complete_ = false;
      return;
    }
    last_joint_state_stamp_ns_ = stamp_ns;
    joint_state_complete_ = true;
  }

  void receive_servo_status(const moveit_msgs::msg::ServoStatus & message)
  {
    last_servo_code_ = message.code;
    last_servo_message_ = message.message;
    last_servo_status_stamp_ns_ = now().nanoseconds();
    if (is_dangerous_servo_status(message.code)) {
      if (!safety_hold_requested_ && !safety_hold_pose_.has_value()) {
        safety_hold_requested_ = true;
        safety_face_sequence_ = accepted_face_count_;
      }
      if (message.code != last_logged_dangerous_code_) {
        RCLCPP_WARN(
          get_logger(), "Servo requested safety hold: code=%d, %s",
          static_cast<int>(message.code), message.message.c_str());
        last_logged_dangerous_code_ = message.code;
      }
    } else if (message.code == moveit_msgs::msg::ServoStatus::NO_WARNING) {
      last_logged_dangerous_code_ = moveit_msgs::msg::ServoStatus::INVALID;
    }
  }

  void control_tick()
  {
    const std::int64_t now_ns = now().nanoseconds();
    if (last_control_time_ns_.has_value() && now_ns < *last_control_time_ns_) {
      handle_clock_reset();
    }
    last_control_time_ns_ = now_ns;

    if (require_perception_status_ && !perception_status_healthy(now_ns)) {
      pending_face_.reset();
      core_->reset();
      last_adapter_status_ = perception_status_failure(now_ns);
      const auto actual_pose = current_tcp_pose(now_ns);
      if (actual_pose) {
        last_known_tcp_pose_ = actual_pose->pose;
      }
      // HOLD carries no motion objective. A cached measured pose is used when
      // available; the identity pose is only a valid stamped envelope when TF
      // is temporarily missing and does not command an endpoint.
      publish_pose(last_known_tcp_pose_.value_or(tracking::Pose3d{}), now_ns,
        "PERCEPTION_HOLD");
      return;
    }

    const bool infrastructure_ready_now = infrastructure_ready(now_ns);
    ready_ = motion_ready(now_ns);
    if (!infrastructure_ready_now) {
      return;
    }

    // FACE orientation must always be computed from the measured screen center.
    // Passing an absent pose to the geometry core would instead aim from the
    // robot base, causing a different orientation whenever TF becomes stale.
    const auto actual_pose = current_tcp_pose(now_ns);
    if (!actual_pose.has_value()) {
      last_adapter_status_ = "tcp_pose_unavailable";
      return;
    }
    last_known_tcp_pose_ = actual_pose->pose;

    update_perception_identity();

    process_pending_face(now_ns);

    if (safety_hold_requested_ && !safety_hold_pose_.has_value()) {
      safety_hold_pose_ = actual_pose->pose;
      safety_hold_requested_ = false;
      publish_pose(*safety_hold_pose_, now_ns, "SAFETY_HOLD");
      return;
    }

    if (safety_hold_pose_.has_value()) {
      if (last_servo_code_ == moveit_msgs::msg::ServoStatus::NO_WARNING &&
        has_safe_retarget(now_ns))
      {
        RCLCPP_INFO(get_logger(), "Fresh safe controller status released the safety hold");
        safety_hold_pose_.reset();
      } else {
        publish_pose(*safety_hold_pose_, now_ns, "SAFETY_HOLD");
        return;
      }
    }

    const tracking::TickResult result = core_->tick(
      now_ns, actual_pose,
      require_perception_status_ && config_prediction_enabled_ && perception_no_face());
    if (!result.command.has_value()) {
      if (result.status == tracking::TickStatus::kHoldPoseUnavailable) {
        last_adapter_status_ = "hold_pose_unavailable";
      } else if (result.status == tracking::TickStatus::kInvalidActualPose) {
        last_adapter_status_ = "invalid_actual_pose";
      }
      return;
    }

    publish_command(*result.command);
    last_adapter_status_ = result.status == tracking::TickStatus::kClockReset ?
      "clock_reset" : "command";
  }

  void process_pending_face(const std::int64_t now_ns)
  {
    if (!pending_face_.has_value()) {
      return;
    }
    const std::int64_t stamp_ns = rclcpp::Time(
      pending_face_->message.header.stamp, get_clock()->get_clock_type()).nanoseconds();
    if (stamp_ns > now_ns) {
      if (stamp_ns - now_ns > kMaximumClockSkewNs) {
        reject_face("future_stamp");
        pending_face_.reset();
      } else {
        last_adapter_status_ = "waiting_measurement_time";
      }
      return;
    }
    if (now_ns - stamp_ns >= freshness_timeout_ns_) {
      reject_face("tf_unavailable_before_deadline");
      pending_face_.reset();
      return;
    }

    if (require_perception_status_) {
      std::lock_guard<std::mutex> lock(perception_mutex_);
      if (!latest_perception_status_) {
        last_adapter_status_ = "waiting_perception_status";
        return;
      }
      const std::int64_t source_stamp_ns = rclcpp::Time(
        latest_perception_status_->source_stamp, get_clock()->get_clock_type()).nanoseconds();
      if (source_stamp_ns < stamp_ns) {
        last_adapter_status_ = "waiting_matching_perception_status";
        return;
      }
      if (source_stamp_ns != stamp_ns ||
        latest_perception_status_->state != msg::PerceptionStatus::OK)
      {
        reject_face("perception_status_mismatch");
        pending_face_.reset();
        return;
      }
    }

    const auto sample = transform_face(pending_face_->message);
    if (!sample.has_value()) {
      last_adapter_status_ = "waiting_exact_tf";
      return;
    }

    const auto received_at = pending_face_->received_at;
    pending_face_.reset();
    const tracking::IngestResult ingest_result = core_->ingest_face(*sample, now_ns);
    if (!ingest_result.accepted()) {
      reject_face(reject_reason_name(ingest_result.reason));
      return;
    }

    ++accepted_face_count_;
    last_accepted_face_stamp_ns_ = sample->measurement_stamp_ns;
    last_accepted_face_target_ = *ingest_result.accepted_geometry;
    last_face_received_at_ = received_at;
    last_adapter_status_ = "face_accepted";
  }

  std::optional<tracking::FaceSample> transform_face(
    const geometry_msgs::msg::PointStamped & message)
  {
    try {
      const rclcpp::Time measurement_time(
        message.header.stamp, get_clock()->get_clock_type());
      geometry_msgs::msg::PointStamped transformed_face;
      if (message.header.frame_id == planning_frame_) {
        transformed_face = message;
      } else {
        const auto transform = tf_buffer_.lookupTransform(
          planning_frame_, message.header.frame_id, measurement_time,
          rclcpp::Duration::from_nanoseconds(0));
        tf2::doTransform(message, transformed_face, transform);
      }

      Eigen::Vector3d base_position = Eigen::Vector3d::Zero();
      if (base_frame_ != planning_frame_) {
        const auto base_transform = tf_buffer_.lookupTransform(
          planning_frame_, base_frame_, measurement_time,
          rclcpp::Duration::from_nanoseconds(0));
        base_position = {
          base_transform.transform.translation.x,
          base_transform.transform.translation.y,
          base_transform.transform.translation.z};
      }

      Eigen::Vector3d reach_center_position = Eigen::Vector3d::Zero();
      if (reach_center_frame_ != planning_frame_) {
        const auto reach_center_transform = tf_buffer_.lookupTransform(
          planning_frame_, reach_center_frame_, measurement_time,
          rclcpp::Duration::from_nanoseconds(0));
        reach_center_position = {
          reach_center_transform.transform.translation.x,
          reach_center_transform.transform.translation.y,
          reach_center_transform.transform.translation.z};
      }

      tracking::FaceSample sample;
      sample.face_in_planning_frame = {
        transformed_face.point.x, transformed_face.point.y, transformed_face.point.z};
      sample.base_in_planning_frame = base_position;
      sample.reach_center_in_planning_frame = reach_center_position;
      sample.measurement_stamp_ns = measurement_time.nanoseconds();
      sample.frame_id = planning_frame_;
      return sample;
    } catch (const tf2::TransformException &) {
      return std::nullopt;
    }
  }

  std::optional<tracking::ActualTcpPose> current_tcp_pose(const std::int64_t now_ns)
  {
    if (!joint_state_fresh(now_ns)) {
      return std::nullopt;
    }
    try {
      const rclcpp::Time latest(0, 0, get_clock()->get_clock_type());
      const auto transform = tf_buffer_.lookupTransform(
        planning_frame_, control_frame_, latest, rclcpp::Duration::from_nanoseconds(0));
      const std::int64_t transform_stamp_ns = rclcpp::Time(
        transform.header.stamp, get_clock()->get_clock_type()).nanoseconds();
      if (transform_stamp_ns < 0 || transform_stamp_ns > now_ns ||
        now_ns - transform_stamp_ns >= kTcpPoseTimeoutNs)
      {
        return std::nullopt;
      }
      tracking::ActualTcpPose result;
      result.pose.position = {
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z};
      result.pose.orientation = Eigen::Quaterniond{
        transform.transform.rotation.w,
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z};
      result.measurement_stamp_ns = transform_stamp_ns;
      return result;
    } catch (const tf2::TransformException &) {
      return std::nullopt;
    }
  }

  void publish_command(const tracking::TargetCommand & command)
  {
    geometry_msgs::msg::PoseStamped message;
    message.header.frame_id = planning_frame_;
    message.header.stamp = rclcpp::Time(
      command.command_stamp_ns, get_clock()->get_clock_type());
    message.pose.position.x = command.pose.position.x();
    message.pose.position.y = command.pose.position.y();
    message.pose.position.z = command.pose.position.z();
    message.pose.orientation.x = command.pose.orientation.x();
    message.pose.orientation.y = command.pose.orientation.y();
    message.pose.orientation.z = command.pose.orientation.z();
    message.pose.orientation.w = command.pose.orientation.w();
    pose_publisher_->publish(message);

    msg::TrackingTarget detailed;
    detailed.header = message.header;
    detailed.pose = message.pose;
    detailed.mode = command.mode == tracking::TargetMode::kFace ? msg::TrackingTarget::FACE :
      (command.mode == tracking::TargetMode::kHold ? msg::TrackingTarget::HOLD :
      (command.mode == tracking::TargetMode::kSearch ? msg::TrackingTarget::SEARCH :
      msg::TrackingTarget::REST));
    detailed.allow_recovery = command.mode != tracking::TargetMode::kFace;
    detailed.target_id = active_target_id_.value_or(-1);
    if (command.mode == tracking::TargetMode::kSearch) {
      detailed.search_pattern = idle_behavior_ == "search_local_then_sweep" &&
        last_accepted_face_stamp_ns_ ? "local_then_sweep" : "sweep";
    }
    if (command.face_in_planning_frame && command.source_face_stamp_ns) {
      detailed.face.x = command.face_in_planning_frame->x();
      detailed.face.y = command.face_in_planning_frame->y();
      detailed.face.z = command.face_in_planning_frame->z();
      detailed.face_stamp = rclcpp::Time(
        *command.source_face_stamp_ns, get_clock()->get_clock_type());
      detailed.face_is_prediction = command.face_prediction_expiry_ns.has_value();
      if (command.face_prediction_expiry_ns) {
        detailed.face_prediction_expiry = rclcpp::Time(
          *command.face_prediction_expiry_ns, get_clock()->get_clock_type());
      }
    }
    tracking_target_publisher_->publish(detailed);

    const std::string new_mode = mode_name(command.mode);
    note_published_mode(new_mode);
    last_command_stamp_ns_ = command.command_stamp_ns;
    if (command.mode == tracking::TargetMode::kFace && last_face_received_at_.has_value() &&
      command.source_face_stamp_ns == last_accepted_face_stamp_ns_)
    {
      last_logical_latency_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - *last_face_received_at_).count();
      last_face_received_at_.reset();
    }
  }

  void publish_pose(
    const tracking::Pose3d & pose, const std::int64_t stamp_ns, const std::string & mode)
  {
    geometry_msgs::msg::PoseStamped message;
    message.header.frame_id = planning_frame_;
    message.header.stamp = rclcpp::Time(stamp_ns, get_clock()->get_clock_type());
    message.pose.position.x = pose.position.x();
    message.pose.position.y = pose.position.y();
    message.pose.position.z = pose.position.z();
    message.pose.orientation.x = pose.orientation.x();
    message.pose.orientation.y = pose.orientation.y();
    message.pose.orientation.z = pose.orientation.z();
    message.pose.orientation.w = pose.orientation.w();
    pose_publisher_->publish(message);
    msg::TrackingTarget detailed;
    detailed.header = message.header;
    detailed.pose = message.pose;
    detailed.mode = msg::TrackingTarget::HOLD;
    tracking_target_publisher_->publish(detailed);
    note_published_mode(mode);
    last_command_stamp_ns_ = stamp_ns;
  }

  void note_published_mode(const std::string & mode)
  {
    if (mode != published_mode_) {
      RCLCPP_INFO(get_logger(), "Target source: %s", mode.c_str());
      published_mode_ = mode;
    }
  }

  void readiness_tick()
  {
    expire_readiness_requests();
    request_controller_state();
    request_planning_scene();
    request_pose_mode();
  }

  void expire_readiness_requests()
  {
    const auto now_steady = std::chrono::steady_clock::now();
    if (controller_request_pending_ && controller_request_started_at_ &&
      now_steady - *controller_request_started_at_ >= kReadinessRequestTimeout)
    {
      if (controller_request_id_) {
        controller_client_->remove_pending_request(*controller_request_id_);
      }
      controller_request_pending_ = false;
      controller_request_id_.reset();
      ++controller_request_generation_;
      RCLCPP_WARN(get_logger(), "Controller readiness request timed out; retrying");
    }
    if (scene_request_pending_ && scene_request_started_at_ &&
      now_steady - *scene_request_started_at_ >= kReadinessRequestTimeout)
    {
      if (scene_request_id_) {
        planning_scene_client_->remove_pending_request(*scene_request_id_);
      }
      scene_request_pending_ = false;
      scene_request_id_.reset();
      ++scene_request_generation_;
      RCLCPP_WARN(get_logger(), "Planning scene readiness request timed out; retrying");
    }
    if (pose_mode_request_pending_ && pose_mode_request_started_at_ &&
      now_steady - *pose_mode_request_started_at_ >= kReadinessRequestTimeout)
    {
      if (pose_mode_request_id_) {
        command_type_client_->remove_pending_request(*pose_mode_request_id_);
      }
      pose_mode_request_pending_ = false;
      pose_mode_request_id_.reset();
      ++pose_mode_request_generation_;
      RCLCPP_WARN(get_logger(), "Servo POSE mode request timed out; retrying");
    }
  }

  void request_controller_state()
  {
    if (controller_request_pending_ || !controller_client_->service_is_ready()) {
      return;
    }
    controller_request_pending_ = true;
    controller_request_started_at_ = std::chrono::steady_clock::now();
    const auto generation = ++controller_request_generation_;
    const auto weak_self = std::weak_ptr<FaceTrackingComponent>(
      std::static_pointer_cast<FaceTrackingComponent>(shared_from_this()));
    auto request = std::make_shared<controller_manager_msgs::srv::ListControllers::Request>();
    const auto handle = controller_client_->async_send_request(
      request,
      [weak_self, generation](
        rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedFuture future) {
        const auto self = weak_self.lock();
        if (!self || generation != self->controller_request_generation_ ||
        !self->controller_request_pending_)
        {
          return;
        }
        self->controller_request_pending_ = false;
        self->controller_request_id_.reset();
        self->controller_request_started_at_.reset();
        bool joint_state_broadcaster_active = false;
        bool arm_controller_active = false;
        try {
          for (const auto & controller : future.get()->controller) {
            if (controller.name == "joint_state_broadcaster" && controller.state == "active") {
              joint_state_broadcaster_active = true;
            }
            if (controller.name == "lite6_arm_controller" && controller.state == "active") {
              arm_controller_active = true;
            }
          }
        } catch (const std::exception & error) {
          RCLCPP_WARN(self->get_logger(), "Controller readiness request failed: %s", error.what());
        }
        self->controllers_ready_ = joint_state_broadcaster_active && arm_controller_active;
      });
    controller_request_id_ = handle.request_id;
  }

  void request_planning_scene()
  {
    if (scene_ready_ || scene_request_pending_ || !planning_scene_client_->service_is_ready()) {
      return;
    }
    scene_request_pending_ = true;
    scene_request_started_at_ = std::chrono::steady_clock::now();
    const auto generation = ++scene_request_generation_;
    const auto weak_self = std::weak_ptr<FaceTrackingComponent>(
      std::static_pointer_cast<FaceTrackingComponent>(shared_from_this()));
    auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
    request->components.components =
      moveit_msgs::msg::PlanningSceneComponents::WORLD_OBJECT_GEOMETRY |
      moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
    const auto handle = planning_scene_client_->async_send_request(
      request,
      [weak_self, generation](
        rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedFuture future) {
        const auto self = weak_self.lock();
        if (!self || generation != self->scene_request_generation_ ||
        !self->scene_request_pending_)
        {
          return;
        }
        self->scene_request_pending_ = false;
        self->scene_request_id_.reset();
        self->scene_request_started_at_.reset();
        try {
          const auto & scene = future.get()->scene;
          const bool table_present = tracking::has_table(scene, self->table_);
          const bool mounting_contact_allowed =
          self->table_.operation == moveit_msgs::msg::CollisionObject::REMOVE ||
          collision_pair_allowed(scene.allowed_collision_matrix, self->base_frame_, kTableId);
          self->scene_ready_ = table_present && mounting_contact_allowed;
          if (!self->scene_ready_) {
            self->publish_table_scene(scene.allowed_collision_matrix);
          }
        } catch (const std::exception & error) {
          RCLCPP_WARN(self->get_logger(), "Planning scene readiness request failed: %s",
            error.what());
        }
      });
    scene_request_id_ = handle.request_id;
  }

  void publish_table_scene(moveit_msgs::msg::AllowedCollisionMatrix collision_matrix)
  {
    if (planning_scene_publisher_->get_subscription_count() == 0) {
      return;
    }
    if (table_.operation == moveit_msgs::msg::CollisionObject::ADD) {
      allow_collision_pair(collision_matrix, base_frame_, kTableId);
    }

    moveit_msgs::msg::PlanningScene update;
    update.is_diff = true;
    update.robot_state.is_diff = true;
    update.world.collision_objects.push_back(table_);
    update.allowed_collision_matrix = std::move(collision_matrix);
    planning_scene_publisher_->publish(update);
    ++scene_update_count_;
  }

  void request_pose_mode()
  {
    if (pose_mode_ready_ || pose_mode_request_pending_ || !controllers_ready_ ||
      !joint_state_fresh(now().nanoseconds()) || !scene_ready_ ||
      !command_type_client_->service_is_ready())
    {
      return;
    }
    pose_mode_request_pending_ = true;
    pose_mode_request_started_at_ = std::chrono::steady_clock::now();
    const auto generation = ++pose_mode_request_generation_;
    const auto weak_self = std::weak_ptr<FaceTrackingComponent>(
      std::static_pointer_cast<FaceTrackingComponent>(shared_from_this()));
    auto request = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
    request->command_type = moveit_msgs::srv::ServoCommandType::Request::POSE;
    const auto handle = command_type_client_->async_send_request(
      request,
      [weak_self, generation](
        rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedFuture future) {
        const auto self = weak_self.lock();
        if (!self || generation != self->pose_mode_request_generation_ ||
        !self->pose_mode_request_pending_)
        {
          return;
        }
        self->pose_mode_request_pending_ = false;
        self->pose_mode_request_id_.reset();
        self->pose_mode_request_started_at_.reset();
        try {
          self->pose_mode_ready_ = future.get()->success;
          if (self->pose_mode_ready_) {
            RCLCPP_INFO(self->get_logger(), "MoveIt Servo POSE mode is ready");
          }
        } catch (const std::exception & error) {
          RCLCPP_WARN(self->get_logger(), "Servo command-type request failed: %s", error.what());
        }
      });
    pose_mode_request_id_ = handle.request_id;
  }

  bool joint_state_fresh(const std::int64_t now_ns) const
  {
    return joint_state_complete_ && last_joint_state_stamp_ns_.has_value() &&
           *last_joint_state_stamp_ns_ <= now_ns &&
           now_ns - *last_joint_state_stamp_ns_ < kJointStateTimeoutNs;
  }

  bool servo_status_fresh(const std::int64_t now_ns) const
  {
    return last_servo_status_stamp_ns_.has_value() &&
           *last_servo_status_stamp_ns_ <= now_ns &&
           now_ns - *last_servo_status_stamp_ns_ < kServoStatusTimeoutNs &&
           is_known_servo_status(last_servo_code_);
  }

  bool perception_status_healthy(const std::int64_t now_ns) const
  {
    std::lock_guard<std::mutex> lock(perception_mutex_);
    if (!latest_perception_status_ || !perception_status_received_at_) {
      return false;
    }
    const auto wall_age = std::chrono::steady_clock::now() - *perception_status_received_at_;
    if (wall_age >= std::chrono::nanoseconds(freshness_timeout_ns_)) {
      return false;
    }
    const auto & status = *latest_perception_status_;
    const std::int64_t publication_stamp_ns = rclcpp::Time(
      status.header.stamp, get_clock()->get_clock_type()).nanoseconds();
    const std::int64_t source_stamp_ns = rclcpp::Time(
      status.source_stamp, get_clock()->get_clock_type()).nanoseconds();
    if (publication_stamp_ns <= 0 || publication_stamp_ns > now_ns + kMaximumClockSkewNs ||
      now_ns - std::min(now_ns, publication_stamp_ns) >= freshness_timeout_ns_ ||
      source_stamp_ns <= 0 || source_stamp_ns > now_ns + kMaximumClockSkewNs ||
      now_ns - std::min(now_ns, source_stamp_ns) >= freshness_timeout_ns_)
    {
      return false;
    }
    return status.state == msg::PerceptionStatus::OK ||
           status.state == msg::PerceptionStatus::NO_FACE;
  }

  bool perception_no_face() const
  {
    std::lock_guard<std::mutex> lock(perception_mutex_);
    return latest_perception_status_ &&
           latest_perception_status_->state == msg::PerceptionStatus::NO_FACE;
  }

  void update_perception_identity()
  {
    std::optional<std::int64_t> target_id;
    std::optional<std::int64_t> status_source_stamp_ns;
    {
      std::lock_guard<std::mutex> lock(perception_mutex_);
      if (latest_perception_status_ &&
        latest_perception_status_->state == msg::PerceptionStatus::OK &&
        latest_perception_status_->target_id >= 0)
      {
        target_id = latest_perception_status_->target_id;
        status_source_stamp_ns = rclcpp::Time(
          latest_perception_status_->source_stamp, get_clock()->get_clock_type()).nanoseconds();
      }
    }
    if (!target_id) {
      return;
    }
    if (active_target_id_ && *active_target_id_ != *target_id) {
      core_->reset();
      if (pending_face_) {
        const std::int64_t pending_stamp_ns = rclcpp::Time(
          pending_face_->message.header.stamp, get_clock()->get_clock_type()).nanoseconds();
        if (!status_source_stamp_ns || pending_stamp_ns != *status_source_stamp_ns) {
          pending_face_.reset();
        }
      }
      last_accepted_face_stamp_ns_.reset();
      last_accepted_face_target_.reset();
      last_face_received_at_.reset();
      last_adapter_status_ = "target_identity_changed";
    }
    active_target_id_ = target_id;
  }

  std::string perception_status_failure(const std::int64_t now_ns) const
  {
    std::lock_guard<std::mutex> lock(perception_mutex_);
    if (!latest_perception_status_ || !perception_status_received_at_) {
      return "perception_status_waiting";
    }
    const auto & status = *latest_perception_status_;
    const std::int64_t publication_stamp_ns = rclcpp::Time(
      status.header.stamp, get_clock()->get_clock_type()).nanoseconds();
    const std::int64_t source_stamp_ns = rclcpp::Time(
      status.source_stamp, get_clock()->get_clock_type()).nanoseconds();
    if (publication_stamp_ns <= 0 || publication_stamp_ns > now_ns + kMaximumClockSkewNs ||
      now_ns - std::min(now_ns, publication_stamp_ns) >= freshness_timeout_ns_)
    {
      return "perception_status_stale";
    }
    if (source_stamp_ns <= 0 || source_stamp_ns > now_ns + kMaximumClockSkewNs ||
      now_ns - std::min(now_ns, source_stamp_ns) >= freshness_timeout_ns_)
    {
      return "perception_source_stale";
    }
    if (std::chrono::steady_clock::now() - *perception_status_received_at_ >=
      std::chrono::nanoseconds(freshness_timeout_ns_))
    {
      return "perception_heartbeat_lost";
    }
    return "perception_fault_" + std::to_string(status.state) + ":" + status.reason;
  }

  bool has_safe_retarget(const std::int64_t now_ns) const
  {
    // Safety is decided by the controller, which must report a fresh NO_WARNING
    // before this predicate is used. A new observation of the same person is
    // enough; requiring changed geometry would latch a static face forever.
    if (!last_accepted_face_stamp_ns_) {
      return true;  // Resume the configured idle behavior when no face exists.
    }
    if (*last_accepted_face_stamp_ns_ > now_ns) {
      return false;
    }
    const bool face_fresh = now_ns - *last_accepted_face_stamp_ns_ < freshness_timeout_ns_;
    return !face_fresh || accepted_face_count_ > safety_face_sequence_;
  }

  bool infrastructure_ready(const std::int64_t now_ns) const
  {
    return controllers_ready_ && joint_state_fresh(now_ns) && servo_status_fresh(now_ns) &&
           scene_ready_ && pose_mode_ready_;
  }

  bool motion_ready(const std::int64_t now_ns) const
  {
    return infrastructure_ready(now_ns) && !is_dangerous_servo_status(last_servo_code_);
  }

  void publish_diagnostics()
  {
    const std::int64_t now_ns = now().nanoseconds();
    const bool infrastructure_ready_now = infrastructure_ready(now_ns);
    ready_ = motion_ready(now_ns);

    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "face_tracking_arm/tracking";
    status.hardware_id = "lite6_gazebo";
    if (safety_hold_pose_.has_value() || safety_hold_requested_) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      status.message = "SAFETY_HOLD";
    } else if (ready_) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      status.message = published_mode_;
    } else {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
      status.message = "STARTING";
    }

    status.values.push_back(key_value("ready", bool_string(ready_)));
    status.values.push_back(key_value(
      "infrastructure_ready", bool_string(infrastructure_ready_now)));
    status.values.push_back(key_value("mode", published_mode_));
    status.values.push_back(key_value("idle_behavior", idle_behavior_));
    status.values.push_back(key_value("controller_ready", bool_string(controllers_ready_)));
    status.values.push_back(key_value("joint_state_ready", bool_string(joint_state_fresh(now_ns))));
    status.values.push_back(key_value("servo_status_ready",
        bool_string(servo_status_fresh(now_ns))));
    status.values.push_back(key_value("scene_ready", bool_string(scene_ready_)));
    status.values.push_back(key_value("pose_mode_ready", bool_string(pose_mode_ready_)));
    if (require_perception_status_) {
      status.values.push_back(key_value(
        "perception_status_ready", bool_string(perception_status_healthy(now_ns))));
      status.values.push_back(key_value(
        "perception_status", perception_status_failure(now_ns)));
    }
    status.values.push_back(key_value(
      "safety_hold", bool_string(safety_hold_pose_.has_value() || safety_hold_requested_)));
    status.values.push_back(key_value("safety_retarget_ready", bool_string(
        has_safe_retarget(now_ns))));
    status.values.push_back(key_value("servo_status_code", std::to_string(last_servo_code_)));
    status.values.push_back(key_value("servo_status_message", last_servo_message_));
    status.values.push_back(key_value("adapter_status", last_adapter_status_));
    status.values.push_back(key_value("last_reject_reason", last_reject_reason_));
    status.values.push_back(key_value("face_received", std::to_string(received_face_count_)));
    status.values.push_back(key_value("face_accepted", std::to_string(accepted_face_count_)));
    status.values.push_back(key_value("face_rejected", std::to_string(rejected_face_count_)));
    status.values.push_back(key_value("scene_updates", std::to_string(scene_update_count_)));
    status.values.push_back(key_value(
      "logical_latency_ms", std::to_string(last_logical_latency_ms_)));
    if (last_joint_state_stamp_ns_.has_value() && *last_joint_state_stamp_ns_ <= now_ns) {
      status.values.push_back(key_value(
        "joint_state_age_ms",
        std::to_string(static_cast<double>(now_ns - *last_joint_state_stamp_ns_) / 1.0e6)));
    }
    if (last_servo_status_stamp_ns_.has_value() && *last_servo_status_stamp_ns_ <= now_ns) {
      status.values.push_back(key_value(
        "servo_status_age_ms",
        std::to_string(static_cast<double>(now_ns - *last_servo_status_stamp_ns_) / 1.0e6)));
    }
    if (last_accepted_face_stamp_ns_.has_value()) {
      status.values.push_back(key_value(
        "source_face_stamp_ns", std::to_string(*last_accepted_face_stamp_ns_)));
    }
    if (last_accepted_face_target_.has_value()) {
      status.values.push_back(key_value(
        "effective_face_distance_m",
        std::to_string(last_accepted_face_target_->face_distance_m)));
      status.values.push_back(key_value(
        "reach_limited", bool_string(last_accepted_face_target_->reach_limited)));
    }
    if (last_command_stamp_ns_.has_value()) {
      status.values.push_back(key_value(
        "command_stamp_ns", std::to_string(*last_command_stamp_ns_)));
    }

    array.status.push_back(std::move(status));
    diagnostics_publisher_->publish(array);
  }

  void reject_face(const std::string & reason)
  {
    ++rejected_face_count_;
    last_reject_reason_ = reason;
  }

  void handle_clock_reset()
  {
    core_->reset();
    pending_face_.reset();
    last_accepted_face_stamp_ns_.reset();
    last_joint_state_stamp_ns_.reset();
    last_servo_status_stamp_ns_.reset();
    joint_state_complete_ = false;
    safety_hold_requested_ = false;
    safety_hold_pose_.reset();
    last_accepted_face_target_.reset();
    last_adapter_status_ = "clock_reset";
    RCLCPP_WARN(get_logger(), "ROS clock moved backwards; tracking state was reset");
  }

  std::string planning_frame_;
  std::string base_frame_;
  std::string reach_center_frame_;
  std::string control_frame_;
  std::unique_ptr<tracking::TrackingCore> core_;

  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<msg::TrackingTarget>::SharedPtr tracking_target_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
  rclcpp::Publisher<moveit_msgs::msg::PlanningScene>::SharedPtr planning_scene_publisher_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr face_subscription_;
  rclcpp::Subscription<msg::PerceptionStatus>::SharedPtr perception_status_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
  rclcpp::Subscription<moveit_msgs::msg::ServoStatus>::SharedPtr servo_status_subscription_;
  rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedPtr controller_client_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr planning_scene_client_;
  rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr command_type_client_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::TimerBase::SharedPtr readiness_timer_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  std::optional<PendingFace> pending_face_;
  std::optional<std::int64_t> active_target_id_;
  mutable std::mutex perception_mutex_;
  std::optional<msg::PerceptionStatus> latest_perception_status_;
  std::optional<std::chrono::steady_clock::time_point> perception_status_received_at_;
  std::optional<std::int64_t> last_accepted_face_stamp_ns_;
  std::optional<std::int64_t> last_joint_state_stamp_ns_;
  std::optional<std::int64_t> last_control_time_ns_;
  std::optional<std::int64_t> last_command_stamp_ns_;
  std::optional<std::chrono::steady_clock::time_point> last_face_received_at_;
  std::string idle_behavior_{"rest"};
  moveit_msgs::msg::CollisionObject table_;
  std::optional<tracking::Pose3d> safety_hold_pose_;
  std::optional<tracking::FaceGeometry> last_accepted_face_target_;
  std::optional<tracking::Pose3d> last_known_tcp_pose_;

  bool joint_state_complete_{false};
  bool controllers_ready_{false};
  bool scene_ready_{false};
  bool pose_mode_ready_{false};
  bool ready_{false};
  bool require_perception_status_{false};
  bool config_prediction_enabled_{false};
  bool controller_request_pending_{false};
  bool scene_request_pending_{false};
  bool pose_mode_request_pending_{false};
  std::optional<std::int64_t> controller_request_id_;
  std::optional<std::int64_t> scene_request_id_;
  std::optional<std::int64_t> pose_mode_request_id_;
  std::optional<std::chrono::steady_clock::time_point> controller_request_started_at_;
  std::optional<std::chrono::steady_clock::time_point> scene_request_started_at_;
  std::optional<std::chrono::steady_clock::time_point> pose_mode_request_started_at_;
  std::uint64_t controller_request_generation_{0};
  std::uint64_t scene_request_generation_{0};
  std::uint64_t pose_mode_request_generation_{0};
  bool safety_hold_requested_{false};
  std::uint64_t received_face_count_{0};
  std::uint64_t accepted_face_count_{0};
  std::uint64_t rejected_face_count_{0};
  std::uint64_t safety_face_sequence_{0};
  std::uint64_t scene_update_count_{0};
  std::int64_t freshness_timeout_ns_{0};
  std::int8_t last_servo_code_{moveit_msgs::msg::ServoStatus::INVALID};
  std::int8_t last_logged_dangerous_code_{moveit_msgs::msg::ServoStatus::INVALID};
  std::optional<std::int64_t> last_servo_status_stamp_ns_;
  double last_logical_latency_ms_{0.0};
  std::string published_mode_{"NONE"};
  std::string last_servo_message_{"not_received"};
  std::string last_adapter_status_{"starting"};
  std::string last_reject_reason_{"none"};
};

}  // namespace face_tracking_arm

RCLCPP_COMPONENTS_REGISTER_NODE(face_tracking_arm::FaceTrackingComponent)
