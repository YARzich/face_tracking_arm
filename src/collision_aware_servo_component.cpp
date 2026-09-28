// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "collision_aware_servo_component.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace face_tracking_arm
{
namespace
{

using namespace std::chrono_literals;

constexpr double kMaximumExpectedCommandLatencySec = 0.070;

constexpr char kTableCollisionObjectName[] = "round_table";

bool finite_pose(const geometry_msgs::msg::Pose & pose)
{
  return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) &&
         std::isfinite(pose.position.z) && std::isfinite(pose.orientation.x) &&
         std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
         std::isfinite(pose.orientation.w);
}

}  // namespace

CollisionAwareServoComponent::CollisionAwareServoComponent(const rclcpp::NodeOptions & options)
: Node("collision_aware_servo", options)
{
  parameters_ = declareControllerParameters(*this);
  validateControllerParameters(parameters_);
  control_time_grid_ = std::make_unique<control::ControlTimeGrid>(
    seconds_to_nanoseconds(parameters_.control_period_sec));

  use_tracking_target_ = declare_parameter<bool>("use_tracking_target", false);
  tracking_target_subscription_ = create_subscription<msg::TrackingTarget>(
    "~/tracking_target_cmds", rclcpp::QoS(1).reliable(),
    [this](msg::TrackingTarget::ConstSharedPtr message) {
      if (use_tracking_target_) {
        receive_target(*message);
      }
    });
  pose_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
    "~/pose_target_cmds", rclcpp::QoS(1).reliable(),
    [this](geometry_msgs::msg::PoseStamped::ConstSharedPtr message) {
      receive_pose(std::move(message));
    });
  controller_activity_subscription_ =
    create_subscription<controller_manager_msgs::msg::ControllerManagerActivity>(
    "/controller_manager/activity",
    rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
    [this](controller_manager_msgs::msg::ControllerManagerActivity::ConstSharedPtr message) {
      receive_controller_activity(*message);
    });
  controller_state_subscription_ =
    create_subscription<control_msgs::msg::JointTrajectoryControllerState>(
    "/lite6_arm_controller/controller_state", rclcpp::QoS(1).reliable(),
    [this](control_msgs::msg::JointTrajectoryControllerState::ConstSharedPtr message) {
      std::lock_guard<std::mutex> lock(controller_state_mutex_);
      latest_controller_state_ = std::move(*message);
    });
  status_publisher_ = create_publisher<moveit_msgs::msg::ServoStatus>(
    "~/status", rclcpp::QoS(10).reliable());
  diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "~/diagnostics", rclcpp::QoS(1).reliable());
  switch_command_type_service_ = create_service<moveit_msgs::srv::ServoCommandType>(
    "~/switch_command_type",
    [this](
      const moveit_msgs::srv::ServoCommandType::Request::SharedPtr request,
      moveit_msgs::srv::ServoCommandType::Response::SharedPtr response)
    {
      switch_command_type(*request, *response);
    });
  pause_service_ = create_service<std_srvs::srv::SetBool>(
    "~/pause_servo",
    [this](
      const std_srvs::srv::SetBool::Request::SharedPtr request,
      std_srvs::srv::SetBool::Response::SharedPtr response)
    {
      set_paused(*request, *response);
    });

  // Servo and PlanningSceneMonitor require shared_from_this(). A one-shot timer
  // defers their construction until the composable node is owned by the container.
  initialization_timer_ = create_wall_timer(1ms, [this]() {initialize();});
}

CollisionAwareServoComponent::~CollisionAwareServoComponent()
{
  shutdown_resources();
  if (shutdown_callback_) {
    get_node_base_interface()->get_context()->remove_pre_shutdown_callback(*shutdown_callback_);
  }
}

void CollisionAwareServoComponent::shutdown_resources()
{
  const std::lock_guard<std::mutex> lock(initialization_mutex_);
  stop_control_.store(true, std::memory_order_release);
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  if (initialization_timer_) {
    initialization_timer_->cancel();
  }
  motion_reference_.reset();
  planning_scene_monitor_.reset();
  servo_parameter_listener_.reset();
}

void CollisionAwareServoComponent::initialize()
{
  const std::lock_guard<std::mutex> lock(initialization_mutex_);
  if (initialization_started_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  initialization_timer_->cancel();

  // PlanningSceneMonitor retains its parent node. Break that ownership cycle
  // while the ROS context is still usable, before the container unloads plugins.
  const std::weak_ptr<CollisionAwareServoComponent> weak_self =
    std::static_pointer_cast<CollisionAwareServoComponent>(shared_from_this());
  shutdown_callback_ = get_node_base_interface()->get_context()->add_pre_shutdown_callback(
    [weak_self]() {
      if (const auto self = weak_self.lock()) {
        self->shutdown_resources();
      }
    });

  servo_parameter_listener_ = std::make_shared<servo::ParamListener>(
    shared_from_this(), "moveit_servo");
  servo_parameters_ = servo_parameter_listener_->get_params();
  validate_servo_parameters();

  planning_scene_monitor_ = moveit_servo::createPlanningSceneMonitor(
    shared_from_this(), servo_parameters_);
  if (!planning_scene_monitor_ || !planning_scene_monitor_->getRobotModel() ||
    !planning_scene_monitor_->getStateMonitor())
  {
    throw std::runtime_error("MoveIt PlanningSceneMonitor initialization failed");
  }
  planning_scene_monitor_->getStateMonitor()->enableCopyDynamics(true);
  planning_scene_monitor_->addUpdateCallback(
    [this](const planning_scene_monitor::PlanningSceneMonitor::SceneUpdateType update_type) {
      const int collision_scene_bits =
      planning_scene_monitor::PlanningSceneMonitor::UPDATE_TRANSFORMS |
      planning_scene_monitor::PlanningSceneMonitor::UPDATE_GEOMETRY;
      if ((static_cast<int>(update_type) & collision_scene_bits) != 0) {
        planning_scene_collision_revision_.fetch_add(1U, std::memory_order_acq_rel);
      }
    });

  const auto robot_model = planning_scene_monitor_->getRobotModel();
  joint_model_group_ = robot_model->getJointModelGroup(parameters_.planning_group_name);
  command_link_ = robot_model->getLinkModel(parameters_.command_frame);
  if (!joint_model_group_ || !command_link_) {
    throw std::runtime_error("configured planning group or command frame is absent");
  }
  if (robot_model->getModelFrame() != parameters_.planning_frame) {
    throw std::runtime_error(
            "planning_frame must equal the RobotModel frame for the QP controller");
  }
  initialize_motion_limits(*robot_model);

  control::HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  // Every published branch already contains a jerk-limited braking suffix.
  // While a replacement is in flight, JTC keeps executing that suffix rather
  // than an adversarial accelerating command. Charging transport latency a
  // second time here destroys recursive feasibility near a velocity limit.
  // The real transport lead and collision reserve remain enforced below.
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance =
    parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  velocity_qp_ = std::make_unique<control::HierarchicalVelocityQp>(qp_config);

  control::EmergencyBrakeTailConfig braking_tail_config;
  braking_tail_config.period_sec = parameters_.control_period_sec;
  // Keep enough already-transmitted stationary suffix to move the next
  // branch point forward after an occasional delayed control callback. One
  // additional point is needed because composeTrajectoryMessage withholds
  // the final state as its interpolation sentinel.
  braking_tail_config.terminal_hold_points = static_cast<std::size_t>(std::ceil(
    2.0 * servo_parameters_.max_expected_latency / parameters_.control_period_sec)) + 1U;
  // Use the same numerical boundary as the QP. Otherwise a sub-microradian
  // discretization residual can make the QP's valid brake suffix fail its
  // second, stricter reconstruction check and latch the controller.
  braking_tail_config.comparison_tolerance =
    2.0 * qp_config.solution_feasibility_tolerance;
  emergency_brake_tail_generator_ =
    std::make_unique<control::EmergencyBrakeTailGenerator>(braking_tail_config);

  control::CollisionConstraintConfig collision_config;
  collision_config.hard_clearance_m = parameters_.hard_clearance_m;
  collision_config.query_distance_m = parameters_.collision_query_distance_m;
  collision_config.maximum_constraint_rows = static_cast<std::size_t>(
    parameters_.maximum_collision_constraints);
  collision_config.minimum_gradient_norm = parameters_.collision_gradient_epsilon;
  collision_config.tracking_position_error_bound_rad =
    parameters_.collision_tracking_error_bound_rad;
  collision_config.numerical_distance_reserve_m =
    parameters_.numerical_distance_reserve_m;
  collision_config.default_distance_lipschitz_m_per_rad =
    parameters_.default_distance_lipschitz_m_per_rad;
  collision_config.monitor_near_distance_lipschitz_m_per_rad =
    parameters_.monitor_near_distance_lipschitz_m_per_rad;
  // The generic joint braking envelope covers every axis. Keep the collision
  // builder's selected-axis row consistent with that same physical interval.
  const std::string guarded_joint = parameters_.monitor_guard_joint_name.empty() ?
    parameters_.joint_names.at(4) : parameters_.monitor_guard_joint_name;
  const auto guarded_index = static_cast<Eigen::Index>(std::distance(
    parameters_.joint_names.begin(), std::find(
      parameters_.joint_names.begin(), parameters_.joint_names.end(), guarded_joint)));
  collision_config.protected_joint_name = guarded_joint;
  collision_config.protected_joint_lower_rad = motion_limits_.lower_position[guarded_index];
  collision_config.protected_joint_upper_rad = motion_limits_.upper_position[guarded_index];
  collision_config.invariant_mount_neighbor_name = "link6";
  collision_config.residual_latency_sec = parameters_.residual_command_latency_sec;
  control::configureGeometryBounds(collision_config, *robot_model, *joint_model_group_);
  collision_constraint_builder_ =
    std::make_unique<control::CollisionConstraintBuilder>(collision_config);

  motion_reference_ = std::make_unique<control::MotionReference>(
    shared_from_this(), planning_scene_monitor_, parameters_, motion_limits_, collision_config);

  trajectory_publisher_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(
    servo_parameters_.command_out_topic, rclcpp::QoS(1).reliable());
  const double queue_span_periods = std::ceil(
    2.0 * servo_parameters_.max_expected_latency / parameters_.control_period_sec);
  max_command_queue_points_ = static_cast<std::size_t>(queue_span_periods) + 3U;

  publish_status(
    moveit_msgs::msg::ServoStatus::NO_WARNING,
    "Waiting for fresh robot state and active trajectory controller");
  initialization_timer_ = create_wall_timer(
    10ms, [this]() {complete_initialization_when_ready();});
}

void CollisionAwareServoComponent::complete_initialization_when_ready()
{
  if (initialized_.load(std::memory_order_acquire)) {
    initialization_timer_->cancel();
    return;
  }

  const rclcpp::Time current_time = now();
  const auto state_monitor = planning_scene_monitor_->getStateMonitor();
  if (current_time.nanoseconds() <= 0 || trajectory_publisher_->get_subscription_count() == 0U ||
    !controller_activity_received_.load(std::memory_order_acquire) ||
    !trajectory_controller_active_.load(std::memory_order_acquire))
  {
    return;
  }
  const rclcpp::Time oldest_allowed_state = current_time -
    rclcpp::Duration::from_seconds(parameters_.state_feedback_timeout_sec);
  const rclcpp::Time state_time = state_monitor->getCurrentStateTime();
  if (state_time.nanoseconds() <= 0 || state_time > current_time ||
    state_time < oldest_allowed_state || !state_monitor->haveCompleteState(oldest_allowed_state))
  {
    return;
  }
  if (!current_controller_state().has_value()) {
    return;
  }
  {
    planning_scene_monitor::LockedPlanningSceneRO locked_scene(planning_scene_monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = locked_scene;
    if (!planning_scene || !planning_scene->getWorld() ||
      planning_scene->getWorld()->hasObject(kTableCollisionObjectName) !=
      parameters_.table_collision_enabled)
    {
      return;
    }
  }

  initialization_timer_->cancel();

  last_validated_scene_revision_ = planning_scene_collision_revision_.load(
    std::memory_order_acquire);
  // The first command epoch is opened only after several fresh controller
  // samples prove that its existing desired state is a stationary hold.
  rearm_pending_ = true;
  initialized_.store(true, std::memory_order_release);
  stop_control_.store(false, std::memory_order_release);
  control_thread_ = std::thread([this]() {control_loop();});
  RCLCPP_INFO(
    get_logger(),
    "Collision-aware Servo initialized: period=%.3f s, group=%s, frame=%s",
    parameters_.control_period_sec, parameters_.planning_group_name.c_str(),
    parameters_.command_frame.c_str());
}

void CollisionAwareServoComponent::validate_servo_parameters() const
{
  if (servo_parameters_.command_out_type != "trajectory_msgs/JointTrajectory" ||
    servo_parameters_.command_out_topic != "/lite6_arm_controller/joint_trajectory" ||
    servo_parameters_.joint_topic != "/joint_states" ||
    servo_parameters_.command_in_type != "speed_units")
  {
    throw std::invalid_argument(
            "collision-aware Servo requires the calibrated speed-unit JTC topic contract");
  }
  if (servo_parameters_.move_group_name != parameters_.planning_group_name) {
    throw std::invalid_argument(
            "moveit_servo.move_group_name and planning_group_name must match");
  }
  if (std::abs(servo_parameters_.publish_period - parameters_.control_period_sec) > 1.0e-9) {
    throw std::invalid_argument(
            "moveit_servo.publish_period and control_period_sec must match");
  }
  if (!std::isfinite(servo_parameters_.max_expected_latency) ||
    servo_parameters_.max_expected_latency > kMaximumExpectedCommandLatencySec ||
    servo_parameters_.max_expected_latency <
    parameters_.residual_command_latency_sec + 2.0 * parameters_.control_period_sec)
  {
    throw std::invalid_argument(
            "max_expected_latency must preserve the calibrated fixed-grid buffer");
  }
  if (parameters_.state_feedback_timeout_sec >
    servo_parameters_.max_expected_latency - parameters_.control_period_sec)
  {
    throw std::invalid_argument(
            "state_feedback_timeout_sec must fit inside retained command history");
  }
  if (!servo_parameters_.publish_joint_positions ||
    servo_parameters_.publish_joint_velocities ||
    servo_parameters_.publish_joint_accelerations)
  {
    throw std::invalid_argument(
            "collision-aware Servo requires position-only trajectory output");
  }
  if (servo_parameters_.check_collisions || servo_parameters_.use_smoothing) {
    throw std::runtime_error(
            "downstream Servo collision scaling and smoothing must remain disabled");
  }
}

void CollisionAwareServoComponent::initialize_motion_limits(
  const moveit::core::RobotModel & robot_model)
{
  const std::vector<std::string> group_variables = joint_model_group_->getVariableNames();
  if (parameters_.joint_names != group_variables) {
    throw std::invalid_argument(
            "joint_names must exactly match the planning group's variable order");
  }
  const std::size_t size = group_variables.size();
  if (parameters_.max_joint_velocity_rad_s.size() != size ||
    parameters_.max_joint_acceleration_rad_s2.size() != size ||
    parameters_.max_joint_jerk_rad_s3.size() != size)
  {
    throw std::invalid_argument("every joint limit array must match joint_names");
  }

  motion_limits_.lower_position.resize(size);
  motion_limits_.upper_position.resize(size);
  motion_limits_.position_margin = Eigen::VectorXd::Constant(
    size, parameters_.joint_position_margin_rad);
  motion_limits_.max_velocity.resize(size);
  motion_limits_.max_acceleration.resize(size);
  motion_limits_.max_jerk.resize(size);
  physical_joint_velocity_limits_.resize(size);
  for (std::size_t index = 0; index < size; ++index) {
    const moveit::core::VariableBounds & bounds = robot_model.getVariableBounds(
      group_variables[index]);
    if (!bounds.position_bounded_ ||
      !std::isfinite(bounds.min_position_) || !std::isfinite(bounds.max_position_))
    {
      throw std::invalid_argument("all controlled joints require finite position bounds");
    }
    const double maximum_velocity = parameters_.max_joint_velocity_rad_s[index];
    const double maximum_acceleration = parameters_.max_joint_acceleration_rad_s2[index];
    const double maximum_jerk = parameters_.max_joint_jerk_rad_s3[index];
    const auto urdf_joint = robot_model.getURDF()->getJoint(group_variables[index]);
    if (!bounds.velocity_bounded_ || !std::isfinite(bounds.max_velocity_) ||
      bounds.max_velocity_ <= 0.0 || !urdf_joint || !urdf_joint->limits ||
      !std::isfinite(urdf_joint->limits->velocity) ||
      urdf_joint->limits->velocity <= 0.0)
    {
      throw std::invalid_argument(
              "all controlled joints require finite positive planning and URDF velocity limits");
    }
    if (!std::isfinite(maximum_velocity) || maximum_velocity <= 0.0 ||
      maximum_velocity > bounds.max_velocity_ ||
      !std::isfinite(maximum_acceleration) || maximum_acceleration <= 0.0 ||
      !std::isfinite(maximum_jerk) || maximum_jerk <= 0.0 ||
      maximum_velocity > urdf_joint->limits->velocity)
    {
      throw std::invalid_argument(
          "joint limits must be positive; velocity may not exceed the robot model");
    }
    motion_limits_.lower_position[static_cast<Eigen::Index>(index)] = bounds.min_position_;
    motion_limits_.upper_position[static_cast<Eigen::Index>(index)] = bounds.max_position_;
    motion_limits_.max_velocity[static_cast<Eigen::Index>(index)] = maximum_velocity;
    motion_limits_.max_acceleration[static_cast<Eigen::Index>(index)] = maximum_acceleration;
    motion_limits_.max_jerk[static_cast<Eigen::Index>(index)] = maximum_jerk;
    physical_joint_velocity_limits_[static_cast<Eigen::Index>(index)] =
      urdf_joint->limits->velocity;
  }

  if (parameters_.monitor_guard_joint_name.empty()) {
    return;
  }
  const auto guard_iterator = std::find(
    group_variables.begin(), group_variables.end(), parameters_.monitor_guard_joint_name);
  if (guard_iterator == group_variables.end()) {
    throw std::invalid_argument("monitor_guard_joint_name is absent from joint_names");
  }
  const Eigen::Index guard_index = static_cast<Eigen::Index>(
    std::distance(group_variables.begin(), guard_iterator));
  if (parameters_.monitor_guard_min_position_rad <
    motion_limits_.lower_position[guard_index] ||
    parameters_.monitor_guard_max_position_rad >
    motion_limits_.upper_position[guard_index])
  {
    throw std::invalid_argument("monitor guard corridor exceeds the URDF joint bounds");
  }
  motion_limits_.lower_position[guard_index] =
    parameters_.monitor_guard_min_position_rad;
  motion_limits_.upper_position[guard_index] =
    parameters_.monitor_guard_max_position_rad;
  // The corridor is already a calibrated safety margin inside the URDF range.
  // Applying the generic joint margin again would unnecessarily shrink it.
  motion_limits_.position_margin[guard_index] = 0.0;
}

void CollisionAwareServoComponent::receive_pose(
  geometry_msgs::msg::PoseStamped::ConstSharedPtr message)
{
  if (use_tracking_target_) {
    return;
  }
  msg::TrackingTarget target;
  target.header = message->header;
  target.pose = message->pose;
  target.mode = msg::TrackingTarget::POSE;
  receive_target(target);
}

void CollisionAwareServoComponent::receive_target(const msg::TrackingTarget & target)
{
  const auto * message = &target;
  pose_message_count_.fetch_add(1U, std::memory_order_relaxed);
  if (!finite_pose(message->pose) || message->header.frame_id != parameters_.planning_frame) {
    pose_invalid_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  if (message->mode > msg::TrackingTarget::SEARCH ||
    (message->mode == msg::TrackingTarget::SEARCH &&
    message->search_pattern != "sweep" && message->search_pattern != "local_then_sweep"))
  {
    pose_invalid_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  if (message->mode == msg::TrackingTarget::FACE) {
    const Eigen::Vector3d face(message->face.x, message->face.y, message->face.z);
    const rclcpp::Time face_stamp(message->face_stamp, get_clock()->get_clock_type());
    const rclcpp::Time command_stamp(message->header.stamp, get_clock()->get_clock_type());
    if (!face.allFinite() || face_stamp.nanoseconds() <= 0 || face_stamp > command_stamp ||
      (command_stamp - face_stamp).seconds() > 0.5)
    {
      pose_invalid_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
  }
  const Eigen::Quaterniond orientation(
    message->pose.orientation.w, message->pose.orientation.x,
    message->pose.orientation.y, message->pose.orientation.z);
  if (!std::isfinite(orientation.norm()) || orientation.norm() <= kFiniteEpsilon) {
    pose_invalid_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  const rclcpp::Time stamp(message->header.stamp, get_clock()->get_clock_type());
  const rclcpp::Time current_time = now();
  const rclcpp::Duration maximum_clock_skew = rclcpp::Duration::from_seconds(
    parameters_.control_period_sec);
  if (stamp.nanoseconds() <= 0 || stamp > current_time + maximum_clock_skew) {
    pose_future_count_.fetch_add(1U, std::memory_order_relaxed);
    return;
  }
  // Components receive /clock independently. A pose stamped with the
  // publisher's current simulation time can therefore be one simulation
  // update ahead of this node. Clamp that bounded skew instead of repeatedly
  // rejecting an otherwise fresh 100 Hz target.
  msg::TrackingTarget accepted_message = *message;
  const rclcpp::Time accepted_stamp = std::min(stamp, current_time);
  accepted_message.header.stamp = accepted_stamp;
  std::lock_guard<std::mutex> lock(target_mutex_);
  if (latest_pose_.has_value()) {
    const rclcpp::Time latest_stamp(
      latest_pose_->header.stamp, get_clock()->get_clock_type());
    if (accepted_stamp <= latest_stamp) {
      pose_not_newer_count_.fetch_add(1U, std::memory_order_relaxed);
      return;
    }
  }
  latest_pose_ = std::move(accepted_message);
  pose_accepted_count_.fetch_add(1U, std::memory_order_relaxed);
}

void CollisionAwareServoComponent::receive_controller_activity(
  const controller_manager_msgs::msg::ControllerManagerActivity & message) noexcept
{
  bool controller_active = false;
  for (const auto & controller : message.controllers) {
    if (controller.name == "lite6_arm_controller") {
      controller_active =
        controller.state.id == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
      break;
    }
  }
  trajectory_controller_active_.store(controller_active, std::memory_order_release);
  controller_activity_received_.store(true, std::memory_order_release);
}

void CollisionAwareServoComponent::switch_command_type(
  const moveit_msgs::srv::ServoCommandType::Request & request,
  moveit_msgs::srv::ServoCommandType::Response & response)
{
  const bool pose_requested =
    request.command_type == moveit_msgs::srv::ServoCommandType::Request::POSE;
  response.success = initialized_.load(std::memory_order_acquire) && pose_requested;
  if (response.success) {
    accept_pose_commands_.store(true, std::memory_order_release);
  } else if (!pose_requested) {
    RCLCPP_WARN(
      get_logger(), "Collision-aware backend accepts only the external POSE contract");
  }
}

void CollisionAwareServoComponent::set_paused(
  const std_srvs::srv::SetBool::Request & request,
  std_srvs::srv::SetBool::Response & response)
{
  if (request.data) {
    pause_requested_.store(true, std::memory_order_release);
    response.success = true;
    response.message = "Constrained braking requested before pause";
  } else {
    pause_requested_.store(false, std::memory_order_release);
    pause_terminal_pending_.store(false, std::memory_order_release);
    const bool was_paused = paused_.exchange(false, std::memory_order_acq_rel);
    const bool rearm_latched_halt = halt_latched_.load(std::memory_order_acquire);
    if (was_paused || rearm_latched_halt) {
      reset_queue_requested_.store(true, std::memory_order_release);
    }
    response.success = true;
    response.message = was_paused || rearm_latched_halt ?
      "Safe re-arm requested" : "Pending pause cancelled without rebasing commands";
  }
}

std::optional<msg::TrackingTarget> CollisionAwareServoComponent::current_target(
  const rclcpp::Time & current_time) const
{
  std::lock_guard<std::mutex> lock(target_mutex_);
  if (!latest_pose_.has_value()) {
    return std::nullopt;
  }
  const rclcpp::Time stamp(
    latest_pose_->header.stamp, get_clock()->get_clock_type());
  if (stamp > current_time || current_time - stamp >=
    rclcpp::Duration::from_seconds(parameters_.incoming_command_timeout_sec))
  {
    return std::nullopt;
  }
  return latest_pose_;
}

}  // namespace face_tracking_arm

RCLCPP_COMPONENTS_REGISTER_NODE(face_tracking_arm::CollisionAwareServoComponent)
