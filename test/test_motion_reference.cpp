// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Geometry>
#include <geometric_shapes/shapes.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "face_tracking_arm/face_tracking_policy.hpp"
#include "face_tracking_arm/motion_reference.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"
#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/emergency_brake_tail.hpp"

namespace face_tracking_arm::control
{

/// Narrow friend adapter for deterministic preparation lifecycle tests.
/// Synthetic successful plans represent a worker-certified path only in tests
/// that exercise commit/admission state; the real planner integration test
/// still obtains and validates its path from BackgroundPathPlanner.
class MotionReferenceTestPeer final
{
public:
  [[nodiscard]] static bool hasPreparation(const MotionReference & reference)
  {
    return reference.compact_preparation_.has_value();
  }

  [[nodiscard]] static std::size_t candidateIndex(const MotionReference & reference)
  {
    return reference.compact_preparation_ ?
           reference.compact_preparation_->candidate_index : 0U;
  }

  [[nodiscard]] static std::size_t candidateCount(const MotionReference & reference)
  {
    return reference.compact_preparation_ ?
           reference.compact_preparation_->candidates.size() : 0U;
  }

  [[nodiscard]] static CompactGoal candidate(
    const MotionReference & reference, const std::size_t index)
  {
    if (!reference.compact_preparation_ ||
      index >= reference.compact_preparation_->candidates.size())
    {
      return {};
    }
    return reference.compact_preparation_->candidates[index];
  }

  [[nodiscard]] static bool preparationExhausted(const MotionReference & reference)
  {
    return reference.compact_preparation_ && reference.compact_preparation_->exhausted;
  }

  static void clearDefaultCandidatesForTest(MotionReference & reference)
  {
    reference.default_compact_goals_.clear();
  }

  static void activateResultGeneration(
    MotionReference & reference, const std::uint64_t generation)
  {
    if (!reference.compact_preparation_ || reference.compact_preparation_->exhausted ||
      reference.compact_preparation_->candidate_index >=
      reference.compact_preparation_->candidates.size())
    {
      return;
    }
    reference.compact_preparation_->generation = generation;
    reference.pending_generation_ = generation;
    reference.active_scene_revision_ = reference.compact_preparation_->scene_revision;
  }

  static void cancelPlannerForTest(MotionReference & reference)
  {
    reference.planner_->cancel();
  }

  static void beginPreparation(
    MotionReference & reference, const moveit::core::RobotState & state,
    const double time_sec, const planning_scene::PlanningScene & scene,
    const std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication)
  {
    reference.requestCompactManeuver(
      state, time_sec, scene, scene_revision, "test_preparation",
      actual_feedback, command_queue_state, active_publication, latest_publication);
  }

  /// Exercise preparation admission after a synthetic qualified pair resource
  /// assessment; public update gate behavior has separate real-FK tests.
  static MotionReferenceResult handleQualifiedPreparationForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const msg::TrackingTarget & target, const planning_scene::PlanningScene & scene,
    const double time_sec, const std::uint64_t scene_revision,
    const JointMotionState & actual_feedback,
    const std::optional<JointMotionState> & command_queue_state,
    const bool preserve_existing_roll_guard = false)
  {
    const bool roll_guard_was_armed = reference.upright_roll_guard_armed_;
    qualifyPairYawForIsolatedAdmissionTest(reference);
    reference.upright_roll_guard_armed_ = preserve_existing_roll_guard && roll_guard_was_armed;
    reference.diagnostics_.upright_roll_guard_armed = reference.upright_roll_guard_armed_;
    Eigen::VectorXd position;
    state.copyJointGroupPositions(reference.group_, position);
    MotionReferenceResult result;
    static_cast<void>(reference.handleCompactPreparation(
        state, target, TargetMotionEstimate{}, scene, time_sec, scene_revision,
        position, true, actual_feedback, time_sec, command_queue_state,
        nullptr, nullptr, result));
    return result;
  }

  static bool armFaceUprightRollGuardFromFeedbackForTest(
    MotionReference & reference, const moveit::core::RobotState & state,
    const double time_sec, const JointMotionState & actual_feedback,
    const JointMotionState & command_queue_state)
  {
    return reference.maybeArmFaceUprightRollGuard(
      state, time_sec, actual_feedback, time_sec, command_queue_state, nullptr, nullptr);
  }

  /// Isolated planner/admission fixtures begin after the production pair gate.
  /// Real FK and public update gating are covered by dedicated tests below;
  /// this injected certificate keeps these cases focused on path admission.
  static void qualifyPairYawForIsolatedAdmissionTest(MotionReference & reference)
  {
    PairYawResourceInput input;
    input.target_bearing_rate_radps = 0.4;
    input.target_rate_deadband_radps = 0.01;
    input.j1 = PairYawJointResourceInput{
      -1.0, 1.9, 1.9, -2.0, 2.0, 0.10, BaseJointDirection::kNone};
    input.j4 = PairYawJointResourceInput{
      -1.0, 1.9, 1.9, -2.0, 2.0, 0.10, BaseJointDirection::kNone};
    reference.pair_yaw_assessment_ = assessPairYawResource(input);
    auto & assessment = reference.pair_yaw_assessment_;
    assessment.probe_valid = true;
    assessment.actual_probe_solved = true;
    assessment.relaxed_probe_solved = true;
    assessment.actual_available_yaw_rad = 0.0;
    assessment.pair_relaxed_available_yaw_rad = kPairYawProbeDisplacementRad;
    assessment.actual_roll_residual_rad = 0.0;
    assessment.relaxed_roll_residual_rad = 0.0;
    assessment.actual_elevation_residual_rad = 0.0;
    assessment.relaxed_elevation_residual_rad = 0.0;
    assessment.pair_bound_cause = pairYawProbeShowsPairBoundCause(assessment);
    assessment.exhausted = assessment.pair_bound_cause;
    reference.upright_roll_guard_armed_ = true;
    static_cast<void>(reference.face_policy_.update({true, false}));
  }

  [[nodiscard]] static BackgroundPathResult resultForCurrentCandidate(
    const MotionReference & reference, const bool success,
    const double terminal_joint6_offset_rad = 0.0)
  {
    BackgroundPathResult result;
    if (!reference.compact_preparation_ ||
      reference.compact_preparation_->candidate_index >=
      reference.compact_preparation_->candidates.size())
    {
      return result;
    }
    const auto & preparation = *reference.compact_preparation_;
    const auto & candidate = preparation.candidates[preparation.candidate_index];
    result.generation = preparation.generation;
    result.scene_revision = preparation.scene_revision;
    result.start_positions = preparation.submitted_start_positions;
    result.goal_positions = candidate.bounded_positions;
    result.path = {result.start_positions, result.goal_positions};
    result.path.back()[5] += terminal_joint6_offset_rad;
    result.success = success;
    result.error_code = success ? 0 : -1;
    result.message = success ? "test worker-certified path" : "injected preparation failure";
    result.profile_id = candidate.profile_id;
    if (success) {
      PathValidationMetrics metrics;
      // The handler contract receives worker-certified metrics. Lifecycle-only
      // fixtures model that certificate; they do not claim the two-point path
      // above was collision-checked by MoveIt.
      metrics.maximum_assembly_radius_m =
        candidate.path_requirements.maximum_assembly_radius_m;
      metrics.maximum_roll_error_rad = candidate.path_requirements.maximum_roll_error_rad;
      metrics.maximum_abs_screen_pitch_rad =
        candidate.path_requirements.maximum_abs_screen_pitch_rad.value_or(0.0);
      result.validation_metrics = metrics;
    }
    return result;
  }

  static void consume(
    MotionReference & reference, const BackgroundPathResult & plan,
    const moveit::core::RobotState & state, const double time_sec,
    const planning_scene::PlanningScene & scene, const std::uint64_t scene_revision,
    const std::optional<JointMotionState> & actual_feedback,
    const std::optional<double> & actual_feedback_time_sec,
    const std::optional<JointMotionState> & command_queue_state,
    const PublishedTrajectoryRecord * active_publication,
    const PublishedTrajectoryRecord * latest_publication)
  {
    reference.consumeCompactPreparationResult(
      plan, state, time_sec, scene, scene_revision, actual_feedback,
      actual_feedback_time_sec, command_queue_state, active_publication,
      latest_publication);
  }

  static CollisionSegmentValidationResult validatePose(
    const MotionReference & reference, const planning_scene::PlanningScene & scene,
    const moveit::core::RobotState & state, const Eigen::VectorXd & position)
  {
    const JointMotionState stationary{
      position, Eigen::VectorXd::Zero(position.size()),
      Eigen::VectorXd::Zero(position.size())};
    return reference.collision_validator_.validatePose(
      scene, state, *reference.group_, stationary);
  }

  [[nodiscard]] static PairYawResourceAssessment evaluatePairYawResource(
    const MotionReference & reference, const moveit::core::RobotState & state,
    const msg::TrackingTarget & target)
  {
    Eigen::VectorXd position;
    Eigen::VectorXd velocity;
    Eigen::VectorXd acceleration;
    state.copyJointGroupPositions(reference.group_, position);
    state.copyJointGroupVelocities(reference.group_, velocity);
    state.copyJointGroupAccelerations(reference.group_, acceleration);
    const JointMotionState feedback{position, velocity, acceleration};
    return reference.evaluatePairYawResource(
      state, target, TargetMotionEstimate{}, feedback, feedback);
  }

  static JointPathUpdate updateFollower(
    MotionReference & reference, const Eigen::VectorXd & position, const double time_sec)
  {
    return reference.follower_.update(position, time_sec);
  }
};

namespace
{

using namespace std::chrono_literals;

constexpr std::uint64_t kSceneRevision = 11;
constexpr double kCompactRollCapTestRad{5.1 * 3.14159265358979323846 / 180.0};

/// Synthetic worker metadata for MotionReference admission tests only. These
/// fixtures exercise certificate mapping/selection; they are not planner or
/// collision-validation evidence. BackgroundPathPlanner tests cover real FK facts.
void installSyntheticCompactCertificate(
  BackgroundPathResult & result,
  const std::function<std::array<double, 3>(std::size_t, double)> & metrics_at)
{
  ASSERT_TRUE(result.validation_metrics);
  ASSERT_GE(result.path.size(), 2U);
  result.compact_path_samples.clear();
  result.compact_goal_metrics = PathValidationMetrics{};
  result.compact_goal_metrics->maximum_assembly_radius_m = 0.39;
  result.compact_goal_metrics->maximum_roll_error_rad = 0.01;
  result.compact_goal_metrics->maximum_abs_screen_pitch_rad = 0.20;
  result.compact_goal_metrics->checked_samples = 1U;

  for (std::size_t segment = 0U; segment + 1U < result.path.size(); ++segment) {
    const auto & from = result.path[segment];
    const auto & to = result.path[segment + 1U];
    const auto subdivisions = std::max<std::size_t>(
      1U, static_cast<std::size_t>(std::ceil(
        (to - from).lpNorm<Eigen::Infinity>() / 0.01)));
    const std::size_t first_step = segment == 0U ? 0U : 1U;
    for (std::size_t step = first_step; step <= subdivisions; ++step) {
      const double fraction = static_cast<double>(step) /
        static_cast<double>(subdivisions);
      const auto values = metrics_at(segment, fraction);
      CompactPathSampleCertificate sample;
      sample.positions = from + fraction * (to - from);
      sample.original_segment_index = segment;
      sample.fraction = fraction;
      sample.maximum_assembly_radius_m = values[0];
      sample.maximum_roll_error_rad = values[1];
      sample.maximum_abs_screen_pitch_rad = values[2];
      result.compact_path_samples.push_back(std::move(sample));
    }
  }

  double suffix_radius = result.compact_goal_metrics->maximum_assembly_radius_m;
  double suffix_roll = result.compact_goal_metrics->maximum_roll_error_rad;
  double suffix_pitch = result.compact_goal_metrics->maximum_abs_screen_pitch_rad;
  double whole_radius = suffix_radius;
  double whole_roll = suffix_roll;
  double whole_pitch = suffix_pitch;
  for (auto sample = result.compact_path_samples.rbegin();
    sample != result.compact_path_samples.rend(); ++sample)
  {
    suffix_radius = std::max(suffix_radius, sample->maximum_assembly_radius_m);
    suffix_roll = std::max(suffix_roll, sample->maximum_roll_error_rad);
    suffix_pitch = std::max(suffix_pitch, sample->maximum_abs_screen_pitch_rad);
    sample->suffix_maximum_assembly_radius_m = suffix_radius;
    sample->suffix_maximum_roll_error_rad = suffix_roll;
    sample->suffix_maximum_abs_screen_pitch_rad = suffix_pitch;
    whole_radius = std::max(whole_radius, sample->maximum_assembly_radius_m);
    whole_roll = std::max(whole_roll, sample->maximum_roll_error_rad);
    whole_pitch = std::max(whole_pitch, sample->maximum_abs_screen_pitch_rad);
  }
  result.validation_metrics->maximum_assembly_radius_m = whole_radius;
  result.validation_metrics->maximum_roll_error_rad = whole_roll;
  result.validation_metrics->maximum_abs_screen_pitch_rad = whole_pitch;
  result.validation_metrics->checked_samples = result.compact_path_samples.size();
}

std::array<double, 3> safeSyntheticCompactMetrics()
{
  return {0.39, 0.01, 0.20};
}

std::optional<std::size_t> triggeredAttachmentSampleIndex(
  const std::vector<CompactEvent> & events)
{
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  if (triggered == events.end()) {
    return std::nullopt;
  }
  constexpr std::string_view field{"\"sample_index\":"};
  const auto start = triggered->details_json.find(field);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  const auto value_start = start + field.size();
  if (triggered->details_json.compare(value_start, 4U, "null") == 0) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(
      triggered->details_json.substr(value_start)));
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

TEST(CompactPathReferenceSpeed, ScalingChangesCompactPathVelocityReference)
{
  const Eigen::VectorXd current = Eigen::VectorXd::Zero(6);
  const Eigen::VectorXd distant_waypoint = Eigen::VectorXd::Constant(6, 1.0);
  for (const auto & [scale, expected_speed] : {
        std::pair<double, double>{0.5, 0.35}, {1.0, 0.70}, {1.5, 1.05}})
  {
    const auto settings = scaleCompactPathReference(6.0, 0.70, scale);
    const auto task = makeJointPathVelocityTask(
      current, distant_waypoint, settings.gain, settings.maximum_velocity_radps);
    ASSERT_TRUE(task);
    EXPECT_NEAR(
      task->primary_reference.lpNorm<Eigen::Infinity>(), expected_speed, 1.0e-12);
  }

  const Eigen::VectorXd nearby_waypoint = Eigen::VectorXd::Constant(6, 0.01);
  double previous_reference_norm = 0.0;
  for (const auto & [scale, expected_norm] : {
        std::pair<double, double>{0.5, 0.5}, {1.0, 1.0}, {1.5, 1.5}})
  {
    const auto settings = scaleCompactPathReference(6.0, 0.70, scale);
    const auto task = makeJointPathVelocityTask(
      current, nearby_waypoint, settings.gain, settings.maximum_velocity_radps);
    ASSERT_TRUE(task);
    const double reference_norm = task->primary_reference.norm();
    EXPECT_NEAR(reference_norm, expected_norm * 0.06 * std::sqrt(6.0), 1.0e-12);
    EXPECT_GT(reference_norm, previous_reference_norm);
    previous_reference_norm = reference_norm;
  }
}

TEST(CompactPathReferenceSpeed, RejectsNonPositiveOrNonFiniteScale)
{
  for (const double scale : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max()})
  {
    EXPECT_THROW((void)scaleCompactPathReference(6.0, 0.70, scale), std::invalid_argument);
  }
}

std::optional<Eigen::VectorXd> goalVectorFromDetails(const std::string & details)
{
  constexpr std::string_view kField{"\"goal_joints_rad\":["};
  const auto start = details.find(kField);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  std::size_t cursor = start + kField.size();
  Eigen::VectorXd result(6);
  for (Eigen::Index index = 0; index < result.size(); ++index) {
    const auto end = details.find_first_of(",]", cursor);
    if (end == std::string::npos) {
      return std::nullopt;
    }
    try {
      result[index] = std::stod(details.substr(cursor, end - cursor));
    } catch (const std::exception &) {
      return std::nullopt;
    }
    cursor = end + 1;
  }
  return result;
}

std::string readModelFile(const std::string & path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot read test robot model: " + path);
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/// The real xArm6 description is expanded by CMake, never by a running test.
/// No controller, sensor, simulator or robot command publisher is started here.
class MotionReferenceTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }

  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    std::string urdf = readModelFile(TEST_ROBOT_URDF_FILE);
    const auto closing_robot = urdf.rfind("</robot>");
    ASSERT_NE(closing_robot, std::string::npos);
    urdf.insert(closing_robot,
          R"(
      <link name="test_camera_optical"/>
      <joint name="test_camera_mount" type="fixed">
        <parent link="monitor_control_frame"/><child link="test_camera_optical"/>
        <origin xyz="0 0 0.125" rpy="-1.57079632679 0 -1.57079632679"/>
      </joint>)");
    std::string semantic = readModelFile(TEST_ROBOT_SRDF_FILE);
    if (!searchEnabled()) {
      const auto start = semantic.find("  <group_state name=\"search\"");
      ASSERT_NE(start, std::string::npos);
      const auto end = semantic.find("  </group_state>", start);
      ASSERT_NE(end, std::string::npos);
      semantic.erase(start, end + std::string("  </group_state>\n").size() - start);
    }
    if (!compactProfileEnabled()) {
      for (const auto * profile : {
            "compact_turn_negative_wrist", "compact_turn_positive_wrist"})
      {
        const auto start = semantic.find(
          "  <group_state name=\"" + std::string(profile) + "\"");
        if (start == std::string::npos) {
          continue;
        }
        const auto end = semantic.find("  </group_state>", start);
        ASSERT_NE(end, std::string::npos);
        semantic.erase(start, end + std::string("  </group_state>\n").size() - start);
      }
    }
    options.parameter_overrides({
          rclcpp::Parameter("robot_description", urdf),
          rclcpp::Parameter("robot_description_semantic", semantic),
          rclcpp::Parameter(
        "robot_description_kinematics.xarm6.kinematics_solver",
        "kdl_kinematics_plugin/KDLKinematicsPlugin"),
          rclcpp::Parameter(
        "robot_description_kinematics.xarm6.kinematics_solver_timeout", 0.01),
          rclcpp::Parameter(
        "recovery_planner.planner_configs.RRTConnectkConfigDefault.type", "geometric::RRTConnect"),
          rclcpp::Parameter(
        "recovery_planner.planner_configs.RRTConnectkConfigDefault.range", 0.10),
          rclcpp::Parameter(
        "recovery_planner.xarm6.planner_configs",
        std::vector<std::string>{"RRTConnectkConfigDefault"}),
          rclcpp::Parameter("recovery_planner.xarm6.longest_valid_segment_fraction", 0.00019),
          rclcpp::Parameter("search_enabled", searchEnabled()),
          rclcpp::Parameter("search_speed_rad_s", searchEnabled() ? 0.30 : -1.0),
          rclcpp::Parameter("compact_maneuver.goal_joints_rad", compactGoalOverride()),
    });
    node_ = std::make_shared<rclcpp::Node>("motion_reference_test", options);
    monitor_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node_, "robot_description");
    model_ = monitor_->getRobotModel();
    ASSERT_TRUE(model_);
    group_ = model_->getJointModelGroup("xarm6");
    ASSERT_NE(group_, nullptr);
    ASSERT_EQ(group_->getVariableCount(), 6U);
    parameters_.joint_names = group_->getVariableNames();
    parameters_.gaze_frame = gazeFrame();
    parameters_.command_frame = commandFrame();
    limits_.lower_position.resize(6);
    limits_.upper_position.resize(6);
    limits_.position_margin = Eigen::VectorXd::Constant(6, 0.10);
    limits_.max_velocity = Eigen::VectorXd::Constant(6, 1.0);
    limits_.max_acceleration = Eigen::VectorXd::Constant(6, 2.0);
    limits_.max_jerk = Eigen::VectorXd::Constant(6, 20.0);
    for (Eigen::Index joint = 0; joint < 6; ++joint) {
      const auto & bounds = model_->getVariableBounds(parameters_.joint_names[joint]);
      limits_.lower_position[joint] = bounds.min_position_;
      limits_.upper_position[joint] = bounds.max_position_;
    }
    {
      planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
      Eigen::Isometry3d tabletop = Eigen::Isometry3d::Identity();
      tabletop.translation().z() = 0.725;
      scene->getWorldNonConst()->addToObject(
        "round_table", std::make_shared<shapes::Cylinder>(0.5, 0.05), tabletop);
      scene->getAllowedCollisionMatrixNonConst().setEntry("link_base", "round_table", true);
      ASSERT_TRUE(scene->getCurrentStateNonConst().setToDefaultValues(group_, "rest"));
      scene->getCurrentStateNonConst().update();
    }
    collision_config_.protected_joint_lower_rad = limits_.lower_position[4];
    collision_config_.protected_joint_upper_rad = limits_.upper_position[4];
    collision_config_.invariant_mount_neighbor_name = "link6";
    configureGeometryBounds(collision_config_, *model_, *group_);
    reference_ = std::make_unique<MotionReference>(
      node_, monitor_, parameters_, limits_, collision_config_);
    clock_origin_ = std::chrono::steady_clock::now();
  }

  void TearDown() override
  {
    // Stop the planner worker before its scene monitor and node are released.
    reference_.reset();
    monitor_.reset();
    model_.reset();
    group_ = nullptr;
    node_.reset();
  }

  virtual std::string gazeFrame() const {return "";}
  virtual std::string commandFrame() const {return "monitor_control_frame";}
  virtual bool searchEnabled() const {return true;}
  virtual bool compactProfileEnabled() const {return true;}
  virtual std::vector<double> compactGoalOverride() const {return {};}

  void recordQualifyingFaceShortfall(const moveit::core::RobotState & robot_state)
  {
    auto selected = update(robot_state, target(msg::TrackingTarget::FACE));
    ASSERT_TRUE(selected.tracking);
    auto tracking = *selected.tracking;
    tracking.has_face_primary_roll = true;
    tracking.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
    tracking.face_primary_feedforward_radps.setZero();
    QpResult qp;
    qp.status = QpStatus::kSolved;
    qp.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
    for (int tick = 0; tick <= 8; ++tick) {
      reference_->recordFaceCommand(
        tracking, qp, 1.0 + 0.02 * static_cast<double>(tick), true);
    }
  }

  std::vector<CompactEvent> commitCompactWithLifecycleFixturePath(
    const moveit::core::RobotState & robot_state)
  {
    recordQualifyingFaceShortfall(robot_state);
    beginPreparationForResultTest(robot_state);
    if (MotionReferenceTestPeer::hasPreparation(*reference_)) {
      const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
      consumePreparationResult(robot_state, plan);
    }
    return reference_->drainCompactEvents();
  }

  void admitCurrentPreparationWithLifecycleFixturePath(
    const moveit::core::RobotState & robot_state,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::uint64_t scene_revision = kSceneRevision,
    const std::optional<JointMotionState> & actual_feedback_override = std::nullopt)
  {
    if (!MotionReferenceTestPeer::hasPreparation(*reference_)) {
      beginPreparationForResultTest(
        robot_state, command_queue_state, active_publication,
        latest_publication, scene_revision);
    }
    if (!MotionReferenceTestPeer::hasPreparation(*reference_)) {
      return;
    }
    MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
    MotionReferenceTestPeer::activateResultGeneration(
      *reference_, reference_->diagnostics().compact_generation + 100U);
    const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
    consumePreparationResult(
      robot_state, plan, command_queue_state, active_publication,
      latest_publication, scene_revision, actual_feedback_override);
  }

  moveit::core::RobotState state(const double base_angle = 0.0) const
  {
    moveit::core::RobotState result(model_);
    result.setToDefaultValues();
    if (!result.setToDefaultValues(group_, "rest")) {
      throw std::runtime_error("The real robot model must contain a named rest state");
    }
    result.setVariablePosition("joint1", base_angle);
    result.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    result.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    result.update();
    return result;
  }

  moveit::core::RobotState compactGoalState(const double joint1_offset = 0.0) const
  {
    moveit::core::RobotState result(model_);
    result.setToDefaultValues();
    if (!result.setToDefaultValues(group_, "compact_turn_negative_wrist")) {
      throw std::runtime_error("The xArm6 model must contain the compact goal state");
    }
    result.setVariablePosition("joint1", result.getVariablePosition("joint1") + joint1_offset);
    result.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    result.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    result.update();
    return result;
  }

  moveit::core::RobotState nearScreenPoleState() const
  {
    auto result = state();
    result.setVariablePosition("joint1", 0.2890392970728219);
    result.setVariablePosition("joint2", -0.3842942596277428);
    result.setVariablePosition("joint3", -3.025275867240606);
    result.setVariablePosition("joint4", 3.525770103294581);
    result.setVariablePosition("joint5", -0.26038283922245037);
    result.setVariablePosition("joint6", 4.5984044351267475);
    result.update();
    return result;
  }

  msg::TrackingTarget faceAlongLookDirection(const moveit::core::RobotState & robot_state) const
  {
    auto command = target(msg::TrackingTarget::FACE);
    const auto * look_link = parameters_.gaze_frame.empty() ? nullptr :
      model_->getLinkModel(parameters_.gaze_frame);
    Eigen::Isometry3d look_pose = look_link ?
      robot_state.getGlobalLinkTransform(look_link) :
      robot_state.getGlobalLinkTransform("monitor_control_frame");
    if (look_link) {
      look_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
    }
    const Eigen::Vector3d face = look_pose.translation() + 0.8 * look_pose.linear().col(0);
    command.face.x = face.x();
    command.face.y = face.y();
    command.face.z = face.z();
    return command;
  }

  msg::TrackingTarget target(const std::uint8_t mode) const
  {
    msg::TrackingTarget result;
    result.header.frame_id = "world";
    result.mode = mode;
    result.pose.orientation.w = 1.0;
    result.pose.position.x = 0.2;
    result.pose.position.z = 1.05;
    result.face.x = 0.85;
    result.face.z = 1.68;
    return result;
  }

  MotionReferenceResult update(
    const moveit::core::RobotState & robot_state,
    const std::optional<msg::TrackingTarget> & command,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const std::uint64_t scene_revision = kSceneRevision)
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count();
    Eigen::VectorXd actual_position;
    Eigen::VectorXd actual_velocity;
    robot_state.copyJointGroupPositions(group_, actual_position);
    robot_state.copyJointGroupVelocities(group_, actual_velocity);
    const JointMotionState actual_feedback{
      actual_position, actual_velocity, Eigen::VectorXd::Zero(actual_position.size())};
    const std::optional<JointMotionState> effective_command_queue_state = command_queue_state ?
      command_queue_state : std::optional<JointMotionState>{actual_feedback};
    return reference_->update(
      robot_state, command, *planning_scene, now, scene_revision, actual_feedback, now,
      effective_command_queue_state, active_publication, latest_publication);
  }

  PairYawResourceAssessment pairYawAssessment(
    const moveit::core::RobotState & robot_state,
    const msg::TrackingTarget & command) const
  {
    return MotionReferenceTestPeer::evaluatePairYawResource(
      *reference_, robot_state, command);
  }

  msg::TrackingTarget targetWithYawOffset(
    const moveit::core::RobotState & robot_state, const double yaw_offset_rad) const
  {
    auto command = faceAlongLookDirection(robot_state);
    const auto * tracking_link = parameters_.gaze_frame.empty() ?
      model_->getLinkModel(parameters_.command_frame) :
      model_->getLinkModel(parameters_.gaze_frame);
    if (!tracking_link) {
      throw std::runtime_error("The pair-yaw test needs a valid tracking link");
    }
    Eigen::Isometry3d tracking_pose = robot_state.getGlobalLinkTransform(tracking_link);
    if (!parameters_.gaze_frame.empty()) {
      tracking_pose.linear() *= face_tracking_arm::opticalToPointingRotation();
    }
    const Eigen::Isometry3d base_pose = robot_state.getGlobalLinkTransform("link_base");
    const Eigen::Vector3d face(command.face.x, command.face.y, command.face.z);
    Eigen::Vector3d ray_base = base_pose.linear().transpose() *
      (face - tracking_pose.translation());
    ray_base = Eigen::AngleAxisd(yaw_offset_rad, Eigen::Vector3d::UnitZ()) * ray_base;
    const Eigen::Vector3d offset_face = tracking_pose.translation() +
      base_pose.linear() * ray_base;
    command.face.x = offset_face.x();
    command.face.y = offset_face.y();
    command.face.z = offset_face.z();
    return command;
  }

  std::pair<moveit::core::RobotState, msg::TrackingTarget> criticalPairExhaustedCase() const
  {
    auto robot_state = state();
    constexpr std::array<double, 6> kJointAnglesRad{
      6.233, 0.261, -1.674, -0.056, -0.812, 6.283};
    for (std::size_t index = 0; index < kJointAnglesRad.size(); ++index) {
      robot_state.setVariablePosition(parameters_.joint_names[index], kJointAnglesRad[index]);
    }
    robot_state.update(true);
    auto command = target(msg::TrackingTarget::FACE);
    command.face.x = 0.850;
    command.face.y = 0.023;
    command.face.z = 1.686;
    return {std::move(robot_state), std::move(command)};
  }

  void consumePreparationResult(
    const moveit::core::RobotState & robot_state,
    const BackgroundPathResult & plan,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::uint64_t scene_revision = kSceneRevision,
    const std::optional<JointMotionState> & actual_feedback_override = std::nullopt)
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count();
    Eigen::VectorXd actual_position;
    Eigen::VectorXd actual_velocity;
    robot_state.copyJointGroupPositions(group_, actual_position);
    robot_state.copyJointGroupVelocities(group_, actual_velocity);
    const JointMotionState actual_feedback = actual_feedback_override.value_or(
      JointMotionState{
          actual_position, actual_velocity, Eigen::VectorXd::Zero(actual_position.size())});
    const std::optional<JointMotionState> effective_command_queue_state = command_queue_state ?
      command_queue_state : std::optional<JointMotionState>{actual_feedback};
    MotionReferenceTestPeer::consume(
      *reference_, plan, robot_state, now, *planning_scene, scene_revision,
      actual_feedback, now, effective_command_queue_state,
      active_publication, latest_publication);
  }

  void beginPreparationForResultTest(
    const moveit::core::RobotState & robot_state,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr,
    const std::uint64_t scene_revision = kSceneRevision)
  {
    MotionReferenceTestPeer::qualifyPairYawForIsolatedAdmissionTest(*reference_);
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const double now = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - clock_origin_).count();
    Eigen::VectorXd actual_position;
    Eigen::VectorXd actual_velocity;
    robot_state.copyJointGroupPositions(group_, actual_position);
    robot_state.copyJointGroupVelocities(group_, actual_velocity);
    const JointMotionState actual_feedback{
      actual_position, actual_velocity, Eigen::VectorXd::Zero(actual_position.size())};
    const std::optional<JointMotionState> effective_command_queue_state = command_queue_state ?
      command_queue_state : std::optional<JointMotionState>{actual_feedback};
    MotionReferenceTestPeer::beginPreparation(
      *reference_, robot_state, now, *planning_scene, scene_revision,
      actual_feedback, effective_command_queue_state,
      active_publication, latest_publication);
    MotionReferenceTestPeer::activateResultGeneration(
      *reference_, reference_->diagnostics().compact_generation + 100U);
  }

  void admitWorkerCertifiedLifecycleFixturePath(
    const moveit::core::RobotState & robot_state,
    const std::optional<JointMotionState> & command_queue_state = std::nullopt,
    const PublishedTrajectoryRecord * active_publication = nullptr,
    const PublishedTrajectoryRecord * latest_publication = nullptr)
  {
    const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
    consumePreparationResult(
      robot_state, plan, command_queue_state,
      active_publication, latest_publication);
  }

  std::optional<MotionReferenceResult> waitForRestPath(
    const moveit::core::RobotState & robot_state)
  {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto result = update(robot_state, target(msg::TrackingTarget::REST));
      if (result.follows_path && reference_->diagnostics().plans_accepted > 0U) {
        return result;
      }
      std::this_thread::sleep_for(1ms);
    }
    return std::nullopt;
  }

  void expectPendingRestPreempted(const std::uint8_t new_mode)
  {
    const auto start = state(0.4);
    const auto waiting = update(start, target(msg::TrackingTarget::REST));
    EXPECT_FALSE(waiting.task);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
    // The first update submits but cannot adopt its newly submitted result.
    // Change mode immediately, then keep polling past the planner's 250 ms budget
    // so a late completion from REST has an opportunity to reach the selector.
    const auto deadline = std::chrono::steady_clock::now() + 350ms;
    do {
      const auto selected = update(start, target(new_mode));
      EXPECT_FALSE(selected.follows_path);
      EXPECT_EQ(reference_->diagnostics().plans_accepted, 0U);
      if (new_mode == msg::TrackingTarget::HOLD) {
        EXPECT_FALSE(selected.task);
        EXPECT_EQ(reference_->diagnostics().state, "HOLD");
      } else {
        EXPECT_TRUE(selected.tracking);
        EXPECT_EQ(reference_->diagnostics().state, "FACE");
      }
      std::this_thread::sleep_for(2ms);
    } while (std::chrono::steady_clock::now() < deadline);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  }

  rclcpp::Node::SharedPtr node_;
  planning_scene_monitor::PlanningSceneMonitorPtr monitor_;
  moveit::core::RobotModelConstPtr model_;
  const moveit::core::JointModelGroup * group_{nullptr};
  ControllerParameters parameters_;
  JointMotionLimits limits_;
  CollisionConstraintConfig collision_config_;
  std::unique_ptr<MotionReference> reference_;
  std::chrono::steady_clock::time_point clock_origin_;
};

class MotionReferenceCameraTest : public MotionReferenceTest
{
protected:
  std::string gazeFrame() const override {return "test_camera_optical";}
};

class MotionReferenceSearchDisabledTest : public MotionReferenceTest
{
protected:
  bool searchEnabled() const override {return false;}
};

class MotionReferenceCompactDisabledTest : public MotionReferenceTest
{
protected:
  bool compactProfileEnabled() const override {return false;}
};

class MotionReferenceCustomCompactGoalTest : public MotionReferenceTest
{
protected:
  std::vector<double> compactGoalOverride() const override
  {
    return {0.0, -0.2715944, -2.8699983, 0.0, -1.151386, 0.7};
  }
};

class MotionReferenceInvalidCompactMountTest : public MotionReferenceTest
{
protected:
  std::string commandFrame() const override {return "test_camera_optical";}
};

TEST_F(MotionReferenceTest, PairYawGateHasValidJacobianAtAlignedFeasiblePose)
{
  const auto robot_state = state();
  const auto aligned = faceAlongLookDirection(robot_state);
  const auto assessment = pairYawAssessment(robot_state, aligned);

  ASSERT_TRUE(assessment.valid);
  EXPECT_FALSE(assessment.demand_present);

  auto pure_vertical_error = aligned;
  pure_vertical_error.face.z += 0.20;
  const auto vertical_assessment = pairYawAssessment(robot_state, pure_vertical_error);
  ASSERT_TRUE(vertical_assessment.valid);
  EXPECT_FALSE(vertical_assessment.demand_present);
  EXPECT_FALSE(vertical_assessment.exhausted);
}

TEST_F(MotionReferenceTest, JointThreeLimitExamplesDoNotExhaustJ1J4YawResource)
{
  constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
  const std::array<std::array<double, 6>, 2> joint_angles_deg{{
    {{15.0, -9.0, -70.0, -16.0, -56.0, 23.0}},
    {{183.0, 82.0, -222.9, 26.5, 9.0, -25.0}}}};

  for (const auto & angles_deg : joint_angles_deg) {
    auto robot_state = state();
    for (std::size_t index = 0; index < angles_deg.size(); ++index) {
      robot_state.setVariablePosition(
        parameters_.joint_names[index], angles_deg[index] * kDegreesToRadians);
    }
    robot_state.update(true);
    const auto yaw_demand = targetWithYawOffset(robot_state, 0.30);
    const auto assessment = pairYawAssessment(robot_state, yaw_demand);

    ASSERT_TRUE(assessment.valid);
    ASSERT_TRUE(assessment.demand_present);
    EXPECT_FALSE(assessment.pair_bound_candidate);
    EXPECT_FALSE(assessment.exhausted);
  }
}

TEST_F(MotionReferenceTest, PublicUpdateDoesNotCompactFromShortfallOrOtherJointLimits)
{
  constexpr double kDegreesToRadians = 3.14159265358979323846 / 180.0;
  const std::array<std::array<double, 6>, 2> joint_angles_deg{{
    {{15.0, -9.0, -70.0, -16.0, -56.0, 23.0}},
    {{183.0, 82.0, -222.9, 26.5, 9.0, -25.0}}}};

  for (const auto & angles_deg : joint_angles_deg) {
    auto robot_state = state();
    for (std::size_t index = 0; index < angles_deg.size(); ++index) {
      robot_state.setVariablePosition(
        parameters_.joint_names[index], angles_deg[index] * kDegreesToRadians);
    }
    robot_state.update(true);
    const auto yaw_demand = targetWithYawOffset(robot_state, 0.30);
    const auto selected = update(robot_state, yaw_demand);
    ASSERT_TRUE(selected.tracking);

    auto shortfall = *selected.tracking;
    shortfall.has_face_primary_roll = true;
    shortfall.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
    shortfall.face_primary_feedforward_radps.setZero();
    QpResult shortfall_qp;
    shortfall_qp.status = QpStatus::kSolved;
    shortfall_qp.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
    for (int tick = 0; tick < 9; ++tick) {
      reference_->recordFaceCommand(shortfall, shortfall_qp, 1.0 + 0.02 * tick, true);
    }

    EXPECT_FALSE(reference_->compactManeuverActive());
    EXPECT_EQ(reference_->diagnostics().compact_generation, 0U);
    EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
    const auto events = reference_->drainCompactEvents();
    EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
        return event.event == "pair_resource_exhausted";
    }));
  }
}

TEST_F(MotionReferenceTest, PublicUpdateTriggersForDirectionallyExhaustedJ1J4Pair)
{
  const auto [robot_state, yaw_demand] = criticalPairExhaustedCase();
  const auto assessment = pairYawAssessment(robot_state, yaw_demand);
  ASSERT_TRUE(assessment.valid);
  ASSERT_TRUE(assessment.demand_present);
  ASSERT_TRUE(assessment.pair_bound_candidate);
  ASSERT_TRUE(assessment.probe_valid);
  EXPECT_TRUE(assessment.actual_probe_solved);
  EXPECT_FALSE(assessment.actual_probe_infeasible);
  EXPECT_DOUBLE_EQ(assessment.probe_horizon_sec, kPairYawProbeHorizonSec);
  EXPECT_TRUE(assessment.pair_bound_cause);
  EXPECT_TRUE(assessment.exhausted);
  EXPECT_LT(assessment.actual_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
  EXPECT_GE(assessment.pair_relaxed_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
  EXPECT_LE(assessment.relaxed_roll_residual_rad, 1.0 * 3.14159265358979323846 / 180.0 + 1.0e-4);
  EXPECT_LE(assessment.relaxed_elevation_residual_rad,
        1.0 * 3.14159265358979323846 / 180.0 + 1.0e-4);
  const auto selected = update(robot_state, yaw_demand);

  ASSERT_TRUE(selected.tracking);
  EXPECT_EQ(reference_->diagnostics().state, "FACE_PAIR_YAW_RESOURCE_EXHAUSTED");
  EXPECT_EQ(reference_->diagnostics().compact_reason, "pair_yaw_resource_exhausted");
  const auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "pair_resource_exhausted";
  }));
}

TEST_F(MotionReferenceTest, PairYawProbeDoesNotBlameJ4WhenTheOtherSwivelCanContinue)
{
  auto robot_state = state();
  constexpr std::array<double, 6> kJointAnglesRad{
    6.233, -0.784, -1.010, 1.311, -1.097, 4.632};
  for (std::size_t index = 0; index < kJointAnglesRad.size(); ++index) {
    robot_state.setVariablePosition(parameters_.joint_names[index], kJointAnglesRad[index]);
  }
  robot_state.update(true);
  auto command = target(msg::TrackingTarget::FACE);
  command.face.x = 0.425;
  command.face.y = -0.736;
  command.face.z = 1.680;

  const auto assessment = pairYawAssessment(robot_state, command);

  ASSERT_TRUE(assessment.valid);
  ASSERT_TRUE(assessment.demand_present);
  EXPECT_TRUE(assessment.probe_valid);
  EXPECT_FALSE(assessment.exhausted);
  EXPECT_GE(assessment.actual_available_yaw_rad, 0.80 * kPairYawProbeDisplacementRad);
}

TEST_F(MotionReferenceTest, PublicYawReversalReleasesTheLatchedJ1Hold)
{
  auto robot_state = state();
  robot_state.setVariablePosition("joint1", 6.20);
  robot_state.setVariablePosition("joint2", 0.233);
  robot_state.setVariablePosition("joint3", -1.71);
  robot_state.setVariablePosition("joint4", 0.0168);
  robot_state.setVariablePosition("joint5", -0.8338);
  robot_state.setVariablePosition("joint6", 6.27);
  robot_state.update(true);
  Eigen::VectorXd position;
  robot_state.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};

  const auto edge_update = update(
    robot_state, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(edge_update.j1_hold_position_rad);
  const auto held_direction = static_cast<BaseJointDirection>(
    reference_->diagnostics().j1_hold_direction);
  ASSERT_NE(held_direction, BaseJointDirection::kNone);

  std::optional<msg::TrackingTarget> reversed_target;
  for (const double offset : {0.30, -0.30}) {
    auto candidate = targetWithYawOffset(robot_state, offset);
    const auto assessment = pairYawAssessment(robot_state, candidate);
    if (assessment.valid && assessment.demand_present &&
      assessment.j1.required_direction != BaseJointDirection::kNone &&
      assessment.j1.required_direction != held_direction)
    {
      reversed_target = std::move(candidate);
      break;
    }
  }
  ASSERT_TRUE(reversed_target);

  const auto reversed_update = update(
    robot_state, reversed_target, nullptr, nullptr, settled);

  EXPECT_FALSE(reversed_update.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().j1_hold_q, 0.0);
}

TEST_F(MotionReferenceTest, FreshPairAvailabilityCancelsPreparationBeforeCommit)
{
  const auto robot_state = state();
  const auto aligned_target = faceAlongLookDirection(robot_state);
  beginPreparationForResultTest(robot_state);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
  MotionReferenceTestPeer::activateResultGeneration(
    *reference_, reference_->diagnostics().compact_generation + 100U);
  const auto result = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);

  const auto refreshed = update(robot_state, aligned_target);

  EXPECT_TRUE(refreshed.tracking);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARATION_CANCELLED");
  consumePreparationResult(robot_state, result);
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "preparation_cancelled";
  }));
}

TEST_F(MotionReferenceTest, DefaultCompactGoalUsesNearestValidatedBoundedWristBranch)
{
  const auto joint6 = std::find(
    group_->getVariableNames().begin(), group_->getVariableNames().end(), "joint6");
  ASSERT_NE(joint6, group_->getVariableNames().end());
  const auto joint6_index = std::distance(group_->getVariableNames().begin(), joint6);
  const std::vector<std::pair<double, std::string>> cases{
    {limits_.upper_position[joint6_index] - 0.001, "positive_upper"},
    {limits_.lower_position[joint6_index] + 0.001, "positive_lower"},
    {3.14159265358979323846, "negative_pi"}};

  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (index > 0U) {
      reference_->reset(false);
    }
    auto start = compactGoalState();
    if (cases[index].second != "negative_pi") {
      ASSERT_TRUE(start.setToDefaultValues(group_, "compact_turn_positive_wrist"));
      start.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
      start.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    }
    start.setVariablePosition("joint6", cases[index].first);
    start.update(true);
    const auto events = commitCompactWithLifecycleFixturePath(start);
    const auto triggered = std::find_if(
      events.begin(), events.end(), [](const CompactEvent & event) {
        return event.event == "triggered";
      });
    ASSERT_NE(triggered, events.end()) << cases[index].second;
    ASSERT_TRUE(reference_->compactManeuverActive()) << cases[index].second;
    const auto goal = goalVectorFromDetails(triggered->details_json);
    ASSERT_TRUE(goal) << triggered->details_json;
    if (cases[index].second == "negative_pi") {
      EXPECT_NE(
        triggered->details_json.find("\"goal_profile\":\"compact_turn_negative_wrist\""),
        std::string::npos);
      EXPECT_NE(triggered->details_json.find("\"staged_route_enabled\":true"),
        std::string::npos);
      EXPECT_NEAR((*goal)[joint6_index], 3.14159265358979323846, 1.0e-12);
    } else {
      EXPECT_NE(
        triggered->details_json.find("\"goal_profile\":\"compact_turn_positive_wrist\""),
        std::string::npos);
      EXPECT_NE(triggered->details_json.find("\"staged_route_enabled\":true"),
        std::string::npos);
      const double expected = cases[index].second == "positive_upper" ?
        limits_.upper_position[joint6_index] - 0.02 :
        limits_.lower_position[joint6_index] + 0.02;
      EXPECT_NEAR((*goal)[joint6_index], expected, 1.0e-9);
    }
    if (index == 0U) {
      reference_->reset(true);
      auto different_winding_start = compactGoalState();
      different_winding_start.setVariablePosition(
        "joint6", limits_.lower_position[joint6_index] + 0.001);
      different_winding_start.update(true);
      JointMotionState moving_queue;
      different_winding_start.copyJointGroupPositions(group_, moving_queue.position);
      moving_queue.velocity = Eigen::VectorXd::Zero(moving_queue.position.size());
      moving_queue.acceleration = Eigen::VectorXd::Zero(moving_queue.position.size());
      moving_queue.velocity[0] = 0.02;
      (void)update(
        different_winding_start, target(msg::TrackingTarget::FACE),
        nullptr, nullptr, moving_queue);
      const auto rearmed = reference_->drainCompactEvents();
      EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
      EXPECT_TRUE(reference_->compactManeuverActive());
      EXPECT_FALSE(std::any_of(rearmed.begin(), rearmed.end(), [](const CompactEvent & event) {
          return event.event == "triggered";
      }));
    }
  }
}

TEST_F(MotionReferenceTest, PreparationFailureKeepsFaceActiveWithoutTriggeredEvent)
{
  const auto [start, initial_target] = criticalPairExhaustedCase();
  auto face_target = initial_target;
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};

  // First FACE update qualifies the pair-resource transition. Preparation is
  // entered on the following update, after the policy transition is visible.
  const auto face = update(start, face_target, nullptr, nullptr, settled);
  ASSERT_TRUE(face.task);
  ASSERT_TRUE(face.tracking);
  EXPECT_FALSE(face.actual_face_alignment);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  face_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  const auto preparation = update(start, face_target, nullptr, nullptr, settled);
  ASSERT_TRUE(preparation.task);
  ASSERT_TRUE(preparation.tracking);
  ASSERT_TRUE(preparation.actual_face_alignment);
  EXPECT_TRUE(std::isfinite(preparation.actual_face_alignment->pointing_error_rad));
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_FALSE(reference_->compactManeuverActive());

  const auto preparation_events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(preparation_events.begin(), preparation_events.end(),
    [](const CompactEvent & event) {return event.event == "preparing";}));
  EXPECT_FALSE(std::any_of(preparation_events.begin(), preparation_events.end(),
    [](const CompactEvent & event) {return event.event == "triggered";}));

  MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
  MotionReferenceTestPeer::activateResultGeneration(
    *reference_, reference_->diagnostics().compact_generation + 100U);
  const auto failure = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, false);
  consumePreparationResult(start, failure, settled);
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto failure_events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(failure_events.begin(), failure_events.end(),
    [](const CompactEvent & event) {return event.event == "triggered";}));
}

TEST_F(MotionReferenceTest, FailedDefaultCandidateAdvancesToDistinctBoundedGoal)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const std::size_t count = MotionReferenceTestPeer::candidateCount(*reference_);
  ASSERT_GE(count, 2U);
  const CompactGoal first = MotionReferenceTestPeer::candidate(*reference_, 0U);
  ASSERT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 0U);

  const auto first_failure = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, false);
  consumePreparationResult(start, first_failure);
  ASSERT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 1U);
  EXPECT_FALSE(reference_->compactManeuverActive());
  const CompactGoal second = MotionReferenceTestPeer::candidate(*reference_, 1U);
  EXPECT_NE(first.profile_id, second.profile_id);
  ASSERT_EQ(first.bounded_positions.size(), limits_.lower_position.size());
  ASSERT_EQ(second.bounded_positions.size(), limits_.lower_position.size());
  EXPECT_GT(
    (first.bounded_positions - second.bounded_positions).
    lpNorm<Eigen::Infinity>(), 1.0e-3);
  EXPECT_TRUE((first.bounded_positions.array() >= limits_.lower_position.array()).all());
  EXPECT_TRUE((first.bounded_positions.array() <= limits_.upper_position.array()).all());
  EXPECT_TRUE((second.bounded_positions.array() >= limits_.lower_position.array()).all());
  EXPECT_TRUE((second.bounded_positions.array() <= limits_.upper_position.array()).all());

  MotionReferenceTestPeer::activateResultGeneration(*reference_, 1002U);
  const auto second_success = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);
  consumePreparationResult(start, second_success);
  EXPECT_TRUE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  const auto triggered = std::find_if(events.begin(), events.end(),
      [](const CompactEvent & event) {return event.event == "triggered";});
  ASSERT_NE(triggered, events.end());
  EXPECT_NE(triggered->details_json.find("\"goal_profile\":\"" + second.profile_id + "\""),
    std::string::npos);
}

TEST_F(MotionReferenceTest, AllPreparationCandidatesExhaustOnceForUnchangedStateAndScene)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const std::size_t candidate_count = MotionReferenceTestPeer::candidateCount(*reference_);
  ASSERT_GT(candidate_count, 0U);
  for (std::size_t index = 0U; index < candidate_count; ++index) {
    ASSERT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), index);
    MotionReferenceTestPeer::activateResultGeneration(
      *reference_, 1100U + static_cast<std::uint64_t>(index));
    const auto failure = MotionReferenceTestPeer::resultForCurrentCandidate(
      *reference_, false);
    consumePreparationResult(start, failure);
    EXPECT_FALSE(reference_->compactManeuverActive());
  }
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  const auto generation = reference_->diagnostics().compact_generation;
  const auto exhausted_plan_requests = reference_->diagnostics().plans_requested;

  for (int attempt = 0; attempt < 3; ++attempt) {
    const auto face = update(start, target(msg::TrackingTarget::FACE));
    ASSERT_TRUE(face.task);
    EXPECT_FALSE(reference_->compactManeuverActive());
    EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  }
  EXPECT_EQ(reference_->diagnostics().compact_generation, generation);
  EXPECT_EQ(reference_->diagnostics().plans_requested, exhausted_plan_requests);
  const auto events = reference_->drainCompactEvents();
  EXPECT_EQ(std::count_if(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "preparing";
  }), 1);
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, EmptyDefaultCandidateSetExhaustsUntilStateOrSceneChanges)
{
  const auto start = compactGoalState();
  MotionReferenceTestPeer::clearDefaultCandidatesForTest(*reference_);
  beginPreparationForResultTest(start);

  ASSERT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::candidateCount(*reference_), 0U);
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARATION_EXHAUSTED");
  const auto first_generation = reference_->diagnostics().compact_generation;
  const auto first_plan_requests = reference_->diagnostics().plans_requested;

  beginPreparationForResultTest(start);
  EXPECT_EQ(reference_->diagnostics().compact_generation, first_generation);
  EXPECT_EQ(reference_->diagnostics().plans_requested, first_plan_requests);
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));

  constexpr std::uint64_t kChangedSceneRevision{kSceneRevision + 1U};
  beginPreparationForResultTest(
    start, std::nullopt, nullptr, nullptr, kChangedSceneRevision);
  EXPECT_EQ(reference_->diagnostics().compact_generation, first_generation + 1U);
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));

  const auto changed_state = compactGoalState(0.06);
  const auto scene_retry_generation = reference_->diagnostics().compact_generation;
  recordQualifyingFaceShortfall(changed_state);
  beginPreparationForResultTest(
    changed_state, std::nullopt, nullptr, nullptr, kChangedSceneRevision);
  EXPECT_EQ(reference_->diagnostics().compact_generation, scene_retry_generation + 1U);
  EXPECT_TRUE(MotionReferenceTestPeer::preparationExhausted(*reference_));
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
}

TEST_F(MotionReferenceTest, StalePreparationResultsCannotCommit)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));

  const auto stale_generation = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);
  MotionReferenceTestPeer::activateResultGeneration(*reference_, 2201U);
  consumePreparationResult(start, stale_generation);
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 0U);
  EXPECT_FALSE(reference_->compactManeuverActive());

  const auto stale_scene = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);
  consumePreparationResult(start, stale_scene, std::nullopt, nullptr, nullptr,
    kSceneRevision + 1U);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, PreparationResultWithStaleStartRejectsCandidate)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  ASSERT_GE(MotionReferenceTestPeer::candidateCount(*reference_), 2U);
  auto stale_start = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  stale_start.start_positions[0] += 0.01;
  stale_start.path.front() = stale_start.start_positions;

  consumePreparationResult(start, stale_start);
  EXPECT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 1U);
  EXPECT_FALSE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, AlignedFaceAndLegacyShortfallNeverAdmitCompact)
{
  const auto start = state();
  recordQualifyingFaceShortfall(start);
  const auto face = faceAlongLookDirection(start);
  for (int tick = 0; tick < 16; ++tick) {
    const auto measured = update(start, face);
    ASSERT_TRUE(measured.tracking);
    EXPECT_LE(measured.tracking->pointing_error_rad, kFaceTrackingErrorGuardRad);
    EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
    EXPECT_FALSE(reference_->compactManeuverActive());
  }

  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "pair_resource_exhausted" || event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, NearbyAdmissionRejectsCollisionWithinAttachDistance)
{
  const auto submitted_start = compactGoalState();
  recordQualifyingFaceShortfall(submitted_start);
  beginPreparationForResultTest(submitted_start);
  ASSERT_GE(MotionReferenceTestPeer::candidateCount(*reference_), 2U);
  const auto worker_result = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true);

  auto nearby_actual = submitted_start;
  const auto submitted_position = worker_result.start_positions;
  nearby_actual.setVariablePosition(
    "joint1", nearby_actual.getVariablePosition("joint1") + 0.12);
  nearby_actual.update(true);
  Eigen::VectorXd actual_position;
  nearby_actual.copyJointGroupPositions(group_, actual_position);
  ASSERT_LT((actual_position - submitted_position).norm(), 0.20);
  const Eigen::Isometry3d obstacle_pose =
    submitted_start.getGlobalLinkTransform("link6");
  {
    planning_scene_monitor::LockedPlanningSceneRW scene(monitor_);
    scene->getWorldNonConst()->addToObject(
      "preparation_bridge_obstacle", std::make_shared<shapes::Sphere>(0.002), obstacle_pose);
  }

  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto start_validation = MotionReferenceTestPeer::validatePose(
      *reference_, *planning_scene, submitted_start, submitted_position);
    const auto end_validation = MotionReferenceTestPeer::validatePose(
      *reference_, *planning_scene, nearby_actual, actual_position);
    ASSERT_TRUE(start_validation.input_valid) << start_validation.failure_reason;
    ASSERT_TRUE(end_validation.input_valid) << end_validation.failure_reason;
    EXPECT_TRUE(start_validation.unsafe) << start_validation.failure_reason;
    EXPECT_TRUE(end_validation.unsafe) << end_validation.failure_reason;
  }

  const JointMotionState nearby_queue{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(nearby_actual, worker_result, nearby_queue);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_GT(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("attachment bridge"),
    std::string::npos);
  EXPECT_NE(MotionReferenceTestPeer::candidateIndex(*reference_), 0U);
  const auto events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
  }));
}

TEST_F(MotionReferenceTest, CertifiedSafeSuffixRebasesPathAndFollowerDoesNotReplayPrefix)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd first_waypoint = plan.start_positions;
  first_waypoint[0] += 0.15;
  Eigen::VectorXd detour_waypoint = first_waypoint;
  detour_waypoint[1] += 0.50;
  plan.path = {
    plan.start_positions, first_waypoint, detour_waypoint, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](const std::size_t segment, const double fraction) {
      // A submitted-path prefix exceeds the current cap; the selected suffix is safe.
      if (segment == 0U && fraction < 0.5) {
        return std::array<double, 3>{0.90, 0.01, 0.20};
      }
      return safeSyntheticCompactMetrics();
    });

  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.10;
  const JointMotionState actual_feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  Eigen::VectorXd queue_position = actual_position;
  queue_position[0] += 0.20;
  const JointMotionState queue_feedback{
    queue_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto selected_index = [&]() {
      double nearest_distance = std::numeric_limits<double>::infinity();
      std::size_t nearest = 0U;
      for (std::size_t index = 0U; index < plan.compact_path_samples.size(); ++index) {
        const double distance =
          (actual_position - plan.compact_path_samples[index].positions).norm();
        if (distance < nearest_distance) {
          nearest_distance = distance;
          nearest = index;
        }
      }
      return nearest;
    }();
  ASSERT_GT(selected_index, 0U);
  ASSERT_EQ(plan.compact_path_samples[selected_index].original_segment_index, 0U);
  ASSERT_LE(
    (actual_position - plan.compact_path_samples[selected_index].positions).norm(), 0.20);
  EXPECT_NEAR(
    (queue_position - plan.compact_path_samples[selected_index].positions).norm(), 0.20,
    1.0e-9);
  consumePreparationResult(start, plan, queue_feedback, nullptr, nullptr,
    kSceneRevision, actual_feedback);

  ASSERT_TRUE(reference_->compactManeuverActive());
  const auto events = reference_->drainCompactEvents();
  const auto emitted_sample_index = triggeredAttachmentSampleIndex(events);
  ASSERT_TRUE(emitted_sample_index);
  EXPECT_EQ(*emitted_sample_index, selected_index);
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(triggered, events.end());
  EXPECT_NE(triggered->details_json.find("\"used_certificate\":true"), std::string::npos);
  EXPECT_NE(triggered->details_json.find("\"worker_full_path_metrics\""), std::string::npos);
  EXPECT_NE(triggered->details_json.find("\"follower_seed\":\"command_queue_state\""),
    std::string::npos);
  EXPECT_NE(triggered->details_json.find("\"follower_seed_positions_rad\":"),
    std::string::npos);

  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto follower = MotionReferenceTestPeer::updateFollower(
    *reference_, queue_position, now + 0.01);
  ASSERT_EQ(follower.status, JointPathStatus::kTracking);
  EXPECT_NEAR(follower.distance_from_path_rad, 0.0, 1.0e-9);
  EXPECT_DOUBLE_EQ(follower.progress_rad, 0.0);
  const auto & attachment = plan.compact_path_samples[selected_index].positions;
  EXPECT_GE(follower.reference_position[0], attachment[0] - 1.0e-9);
  EXPECT_GT((follower.reference_position - plan.path.front()).norm(), 0.05);
}

TEST_F(MotionReferenceTest, CompactAdmissionRejectsCertificateGapOrMappingMismatch)
{
  const auto make_plan = [this](const moveit::core::RobotState & start) {
      recordQualifyingFaceShortfall(start);
      beginPreparationForResultTest(start);
      auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
      Eigen::VectorXd waypoint = plan.start_positions;
      waypoint[0] += 0.15;
      plan.path = {plan.start_positions, waypoint, plan.goal_positions};
      installSyntheticCompactCertificate(
        plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
      return plan;
    };

  auto start = compactGoalState();
  auto gap_plan = make_plan(start);
  const auto removed = std::find_if(
    gap_plan.compact_path_samples.begin() + 2,
    gap_plan.compact_path_samples.end() - 1,
    [](const CompactPathSampleCertificate & sample) {
      return sample.original_segment_index == 0U && sample.fraction > 0.25 &&
             sample.fraction < 0.75;
    });
  ASSERT_NE(removed, gap_plan.compact_path_samples.end() - 1);
  gap_plan.compact_path_samples.erase(removed);
  consumePreparationResult(start, gap_plan);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);

  reference_->reset(false);
  start = compactGoalState();
  auto malformed_plan = make_plan(start);
  const auto mapped = std::find_if(
    malformed_plan.compact_path_samples.begin() + 1,
    malformed_plan.compact_path_samples.end(),
    [](const CompactPathSampleCertificate & sample) {
      return sample.original_segment_index == 0U && sample.fraction > 0.25;
    });
  ASSERT_NE(mapped, malformed_plan.compact_path_samples.end());
  mapped->positions[1] += 0.002;
  consumePreparationResult(start, malformed_plan);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, CompactAdmissionRejectsUnsafeRemainingSuffix)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd waypoint = plan.start_positions;
  waypoint[0] += 0.15;
  plan.path = {plan.start_positions, waypoint, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](const std::size_t segment, const double fraction) {
      if ((segment == 0U && fraction >= 0.75) || segment > 0U) {
        return std::array<double, 3>{0.90, 0.01, 0.20};
      }
      return safeSyntheticCompactMetrics();
    });
  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.05;
  const JointMotionState feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(start, plan, feedback, nullptr, nullptr, kSceneRevision, feedback);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_GT(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, CompactCertificateAdmissionEnforcesWindowAndFirstFourSegments)
{
  auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto queue_plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd queue_waypoint = queue_plan.start_positions;
  queue_waypoint[0] += 0.15;
  queue_plan.path = {queue_plan.start_positions, queue_waypoint, queue_plan.goal_positions};
  installSyntheticCompactCertificate(
    queue_plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
  Eigen::VectorXd queue_position = queue_plan.start_positions;
  queue_position[0] += 0.37;
  const JointMotionState queue_outside_window{
    queue_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(
    start, queue_plan, queue_outside_window, nullptr, nullptr, kSceneRevision,
    JointMotionState{queue_plan.start_positions, Eigen::VectorXd::Zero(6),
      Eigen::VectorXd::Zero(6)});
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);

  reference_->reset(false);
  start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.50;
  auto p1 = plan.start_positions;
  auto p2 = plan.start_positions;
  auto p3 = plan.start_positions;
  auto p4 = plan.start_positions;
  p1[3] += 0.05;
  p2[3] += 0.10;
  p3[3] += 0.15;
  p4[3] += 0.20;
  auto p5 = actual_position;
  plan.path = {plan.start_positions, p1, p2, p3, p4, p5, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
  const JointMotionState feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  consumePreparationResult(start, plan, feedback, nullptr, nullptr, kSceneRevision, feedback);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_GT(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("certified suffix attachment"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, CompactCertificateAdmissionChoosesEarliestTiedSample)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  Eigen::VectorXd first_waypoint = plan.start_positions;
  first_waypoint[0] += 0.08;
  plan.path = {plan.start_positions, first_waypoint, plan.goal_positions};
  installSyntheticCompactCertificate(
    plan, [](std::size_t, double) {return safeSyntheticCompactMetrics();});
  Eigen::VectorXd actual_position = plan.start_positions;
  actual_position[0] += 0.025;
  const JointMotionState feedback{
    actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto expected_earliest = static_cast<std::size_t>(2U);
  ASSERT_NEAR(
    (actual_position - plan.compact_path_samples[expected_earliest].positions).norm(),
    (actual_position - plan.compact_path_samples[expected_earliest + 1U].positions).norm(),
    1.0e-12);
  consumePreparationResult(start, plan, feedback, nullptr, nullptr, kSceneRevision, feedback);

  ASSERT_TRUE(reference_->compactManeuverActive());
  const auto emitted_sample_index =
    triggeredAttachmentSampleIndex(reference_->drainCompactEvents());
  ASSERT_TRUE(emitted_sample_index);
  EXPECT_EQ(*emitted_sample_index, expected_earliest);
}

TEST_F(MotionReferenceTest, WorkerEndpointWithinGoalToleranceIsAdmitted)
{
  const auto start = compactGoalState(0.05);
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true, 0.005);
  ASSERT_NEAR(
    (plan.path.back() - plan.goal_positions).lpNorm<Eigen::Infinity>(), 0.005, 1.0e-12);

  consumePreparationResult(start, plan);

  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
}

TEST_F(MotionReferenceTest, WorkerEndpointOutsideGoalToleranceIsRejected)
{
  const auto start = compactGoalState(0.05);
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(
    *reference_, true, 0.1);
  ASSERT_NEAR(
    (plan.path.back() - plan.goal_positions).lpNorm<Eigen::Infinity>(), 0.1, 1.0e-12);

  consumePreparationResult(start, plan);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(MotionReferenceTestPeer::candidateIndex(*reference_), 1U);
  EXPECT_EQ(reference_->diagnostics().plans_rejected, 1U);
  EXPECT_NE(reference_->diagnostics().last_plan_message.find("incomplete path"),
    std::string::npos);
}

TEST_F(MotionReferenceTest, TargetLossOrChangeCancelsOnlyUncommittedPreparation)
{
  auto [start, first_target] = criticalPairExhaustedCase();
  first_target.target_id = 17;
  (void)update(start, first_target);
  first_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  (void)update(start, first_target);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  auto changed_target = first_target;
  changed_target.target_id = 18;
  (void)update(start, changed_target);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());

  reference_->reset(false);
  auto healthy_case = criticalPairExhaustedCase();
  start = std::move(healthy_case.first);
  first_target = std::move(healthy_case.second);
  first_target.target_id = 17;
  (void)update(start, first_target);
  first_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  (void)update(start, first_target);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  auto healthy_loss = target(msg::TrackingTarget::HOLD);
  healthy_loss.allow_recovery = true;
  (void)update(start, healthy_loss);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());

  reference_->reset(false);
  start = compactGoalState(0.05);
  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  ASSERT_TRUE(reference_->compactManeuverActive());
  (void)update(start, healthy_loss);
  EXPECT_TRUE(reference_->compactManeuverActive());
  changed_target.target_id = 19;
  (void)update(start, changed_target);
  EXPECT_TRUE(reference_->compactManeuverActive());
}

TEST_F(MotionReferenceCustomCompactGoalTest, ExplicitCompactGoalIsNotBranchSelectedOrStaged)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  beginPreparationForResultTest(start);
  ASSERT_EQ(MotionReferenceTestPeer::candidateCount(*reference_), 1U);
  const auto configured_candidate = MotionReferenceTestPeer::candidate(*reference_, 0U);
  EXPECT_EQ(configured_candidate.profile_id, "configured_goal_override");
  EXPECT_DOUBLE_EQ(configured_candidate.bounded_positions[5], 0.7);
  EXPECT_FALSE(configured_candidate.path_requirements.staged_route.has_value());

  const auto plan = MotionReferenceTestPeer::resultForCurrentCandidate(*reference_, true);
  consumePreparationResult(start, plan);
  const auto events = reference_->drainCompactEvents();
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(triggered, events.end());
  EXPECT_NE(
    triggered->details_json.find("\"goal_profile\":\"configured_goal_override\""),
    std::string::npos);
  EXPECT_EQ(triggered->details_json.find("\"staged_route_enabled\":true"),
    std::string::npos);
  const auto goal = goalVectorFromDetails(triggered->details_json);
  ASSERT_TRUE(goal);
  EXPECT_DOUBLE_EQ((*goal)[5], 0.7);
}

TEST_F(MotionReferenceInvalidCompactMountTest, DefaultCandidatesRejectInvalidScreenMount)
{
  auto safe_mount_state = nearScreenPoleState();
  safe_mount_state.setVariablePosition(
    "joint5", safe_mount_state.getVariablePosition("joint5") + 0.15);
  safe_mount_state.update(true);
  const auto safe_mount_pose = evaluateCompactUprightPose(
    safe_mount_state.getGlobalLinkTransform("test_camera_optical").linear());
  ASSERT_TRUE(safe_mount_pose);
  ASSERT_LE(std::abs(safe_mount_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  const auto events = commitCompactWithLifecycleFixturePath(safe_mount_state);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "cancelled";
  }));
}

TEST(TrackingOrientationPriority, FaceTaskKeepsThreeExactOrientationRowsPrimary)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  TrackingVelocityTaskConfig config;
  config.position_deadband_m = 0.0;
  config.pointing_deadband_rad = 0.0;
  config.roll_deadband_rad = 0.0;
  const auto task = makeTrackingVelocityTask(
    current, jacobian, Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Vector3d(1.0, 0.3, 0.2), Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(task);
  EXPECT_EQ(task->task.primary_matrix.rows(), 3);
  EXPECT_EQ(task->task.primary_matrix.cols(), 6);
  EXPECT_TRUE(task->has_face_primary_roll);
}

TEST_F(MotionReferenceTest, HoldAndMissingTargetProduceNoTask)
{
  const auto robot_state = state();
  auto held = target(msg::TrackingTarget::HOLD);
  EXPECT_FALSE(update(robot_state, held).task);
  EXPECT_FALSE(update(robot_state, std::nullopt).task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, RestWithRecoveryDoesNotArmFaceRollGuard)
{
  const auto robot_state = state();
  auto rest_with_recovery = target(msg::TrackingTarget::REST);
  rest_with_recovery.allow_recovery = true;
  (void)update(robot_state, rest_with_recovery);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
}

TEST_F(MotionReferenceTest, FaceUsesExactUprightTrackingWithoutJointPathGuidance)
{
  auto face = target(msg::TrackingTarget::FACE);
  face.face_stamp = rclcpp::Time(1, 0, RCL_ROS_TIME);
  const auto result = update(state(), face);
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_EQ(result.tracking->task.primary_matrix.rows(), 3);
  EXPECT_EQ(result.tracking->task.primary_matrix.cols(), 6);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, FaceRollGuardArmsOnlyWithSettledOrCompleteStopEvidence)
{
  const auto robot_state = compactGoalState();
  auto face = target(msg::TrackingTarget::FACE);
  Eigen::VectorXd position;
  robot_state.copyJointGroupPositions(group_, position);
  const auto now_seconds = [this]() {
      return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_origin_).count();
    };
  const auto sample = [this](
    const Eigen::VectorXd & q, const double seconds) {
      moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
      point.joint_names = parameters_.joint_names;
      point.positions = q;
      point.velocities.setZero(q.size());
      point.accelerations.setZero(q.size());
      point.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return point;
    };

  JointMotionState moving_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  moving_queue.velocity[0] = 0.02;
  const auto moving_without_history = update(
    robot_state, face, nullptr, nullptr, moving_queue);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_TRUE(moving_without_history.compact_roll_constraints.empty());

  // A fresh, settled, matched actual/command state can arm without a retained
  // publication, because there is no in-flight suffix to certify.
  JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto settled_without_history = update(
    robot_state, face, nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  ASSERT_EQ(settled_without_history.compact_roll_constraints.size(), 2U);
  const auto arm_events = reference_->drainCompactEvents();
  const auto armed = std::find_if(
    arm_events.begin(), arm_events.end(), [](const CompactEvent & event) {
      return event.event == "upright_roll_guard_armed";
    });
  ASSERT_NE(armed, arm_events.end());
  EXPECT_NE(armed->details_json.find("\"fullstop_max_abs_roll_error_rad\":"),
    std::string::npos);

  reference_->reset(true);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  const auto after_controlled_rearm = update(robot_state, face);
  EXPECT_EQ(after_controlled_rearm.compact_roll_constraints.size(), 2U);

  reference_->reset(false);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  const auto after_full_reset = update(robot_state, face);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(after_full_reset.compact_roll_constraints.size(), 2U);

  const Eigen::VectorXd unsafe_published_stop = (Eigen::VectorXd(6) <<
    6.100528, -0.305181, -3.926985, 0.231062, 0.277314, 2.760624).finished();
  std::string violation;
  EXPECT_TRUE(reference_->validateCommandPosition(robot_state, position, &violation));
  EXPECT_TRUE(violation.empty());
  EXPECT_FALSE(reference_->validateCommandPosition(
      robot_state, unsafe_published_stop, &violation));
  EXPECT_EQ(violation, "screen roll");

  // A safe measured pose cannot arm when the newest publication omits its
  // certified stop, even though the active branch contains that endpoint.
  reference_->reset(false);
  const double now = now_seconds();
  PublishedTrajectoryRecord active;
  active.id = 10U;
  active.execution_queue.push_back(sample(position, now - 0.02));
  active.execution_queue.push_back(sample(position, now + 0.02));
  active.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.20) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord truncated_latest = active;
  truncated_latest.id = 11U;
  active.execution_queue.push_back(sample(position, now + 0.20));
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const auto incomplete_history = reference_->update(
    robot_state, face, *planning_scene, now, kSceneRevision, settled, now, settled,
    &active, &truncated_latest);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_TRUE(incomplete_history.compact_roll_constraints.empty());
}

TEST_F(MotionReferenceTest, FaceRollGuardRejectsTheV14PublishedStopBeforeCommit)
{
  const Eigen::VectorXd current_position = (Eigen::VectorXd(6) <<
    6.100528, -0.297505, -3.904281, 0.004713, 0.158596, 3.142188).finished();
  const Eigen::VectorXd unsafe_stop_position = (Eigen::VectorXd(6) <<
    6.100528, -0.305181, -3.926985, 0.231062, 0.277314, 2.760624).finished();
  auto current = state();
  current.setJointGroupPositions(group_, current_position);
  current.update(true);
  const auto current_pose = evaluateCompactUprightPose(
    current.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(current_pose);
  ASSERT_LE(std::abs(current_pose->signed_roll_error_rad), kCompactRollCapTestRad);

  auto stop = state();
  stop.setJointGroupPositions(group_, unsafe_stop_position);
  stop.update(true);
  const auto stop_pose = evaluateCompactUprightPose(
    stop.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(stop_pose);
  ASSERT_GT(std::abs(stop_pose->signed_roll_error_rad), kCompactRollCapTestRad);

  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto sample = [this](const Eigen::VectorXd & q, const double seconds) {
      moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
      point.joint_names = parameters_.joint_names;
      point.positions = q;
      point.velocities.setZero(q.size());
      point.accelerations.setZero(q.size());
      point.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return point;
    };
  PublishedTrajectoryRecord active;
  active.id = 21U;
  active.execution_queue.push_back(sample(current_position, now - 0.02));
  active.execution_queue.push_back(sample(current_position, now + 0.02));
  active.execution_queue.push_back(sample(unsafe_stop_position, now + 0.80));
  active.execution_queue.push_back(sample(unsafe_stop_position, now + 0.81));
  active.execution_queue.push_back(sample(unsafe_stop_position, now + 0.82));
  active.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.80) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord latest = active;
  latest.id = 22U;

  JointMotionState actual{
    current_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  JointMotionState command = actual;
  auto face = target(msg::TrackingTarget::FACE);
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const auto selected = reference_->update(
    current, face, *planning_scene, now, kSceneRevision, actual, now, command,
    &active, &latest);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_TRUE(selected.compact_roll_constraints.empty());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 0U);
}

TEST_F(MotionReferenceTest, FaceRemainsFiniteNearScreenUprightPole)
{
  const auto robot_state = nearScreenPoleState();
  const auto & screen = robot_state.getGlobalLinkTransform("monitor_control_frame");
  ASSERT_LT(screen.linear().col(0).head<2>().norm(), 0.11);

  const auto result = update(robot_state, faceAlongLookDirection(robot_state));
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_TRUE(result.task->primary_matrix.allFinite());
  EXPECT_TRUE(result.task->primary_reference.allFinite());
  EXPECT_GT(result.task->primary_matrix.row(2).norm(), 1.0e-6);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_TRUE(std::isfinite(result.tracking->roll_error_rad));
  EXPECT_GT(std::abs(result.tracking->roll_error_rad), 1.0e-3);
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_TRUE(result.compact_roll_constraints.empty());
}

TEST_F(MotionReferenceCameraTest, GazeFaceKeepsPhysicalScreenRollPrimary)
{
  const auto robot_state = state();
  auto face = target(msg::TrackingTarget::FACE);
  const auto * gaze_link = model_->getLinkModel(parameters_.gaze_frame);
  ASSERT_NE(gaze_link, nullptr);
  const Eigen::Isometry3d optical_pose = robot_state.getGlobalLinkTransform(gaze_link);
  const Eigen::Vector3d target_face =
    optical_pose.translation() + 0.8 * optical_pose.linear().col(2);
  face.face.x = target_face.x();
  face.face.y = target_face.y();
  face.face.z = target_face.z();
  const auto result = update(robot_state, face);
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_NEAR(result.tracking->pointing_error_rad, 0.0, 1.0e-8);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_EQ(result.task->secondary_matrix.rows(), 3);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(result.compact_roll_constraints.size(), 2U);

  Eigen::MatrixXd screen_jacobian;
  ASSERT_TRUE(robot_state.getJacobian(
      group_, model_->getLinkModel("monitor_control_frame"), Eigen::Vector3d::Zero(),
      screen_jacobian, false));
  const auto * root_link = group_->getJointModels().front()->getParentLinkModel();
  if (root_link) {
    const Eigen::Matrix3d root_rotation =
      robot_state.getGlobalLinkTransform(root_link).linear();
    screen_jacobian.topRows(3) = (root_rotation * screen_jacobian.topRows(3)).eval();
    screen_jacobian.bottomRows(3) =
      (root_rotation * screen_jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig config;
  config.orientation_gain = parameters_.orientation_gain;
  config.maximum_angular_reference_radps = parameters_.maximum_angular_reference_radps;
  const auto physical_roll = makeUprightRollPathTask(
    robot_state.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::VectorXd::Zero(6), config);
  ASSERT_TRUE(physical_roll);
  EXPECT_TRUE(result.task->primary_matrix.row(2).isApprox(
      physical_roll->task.primary_matrix.row(0), 1.0e-12));

  const auto pair_resource = pairYawAssessment(robot_state, face);
  ASSERT_TRUE(pair_resource.valid);
  EXPECT_FALSE(pair_resource.demand_present);
  EXPECT_FALSE(pair_resource.exhausted);
  EXPECT_FALSE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().tracking_phase, "FACE");
  EXPECT_EQ(reference_->diagnostics().compact_generation, 0U);
}

TEST_F(MotionReferenceCameraTest, GazeFaceRemainsFiniteNearScreenUprightPole)
{
  const auto robot_state = nearScreenPoleState();
  const auto & screen = robot_state.getGlobalLinkTransform("monitor_control_frame");
  ASSERT_LT(screen.linear().col(0).head<2>().norm(), 0.11);

  const auto result = update(robot_state, faceAlongLookDirection(robot_state));
  ASSERT_TRUE(result.tracking);
  ASSERT_TRUE(result.task);
  EXPECT_EQ(result.task->primary_matrix.rows(), 3);
  EXPECT_TRUE(result.task->primary_matrix.allFinite());
  EXPECT_TRUE(result.task->primary_reference.allFinite());
  EXPECT_GT(result.task->primary_matrix.row(2).norm(), 1.0e-6);
  EXPECT_TRUE(result.tracking->has_face_primary_roll);
  EXPECT_TRUE(std::isfinite(result.tracking->roll_error_rad));
  EXPECT_GT(std::abs(result.tracking->roll_error_rad), 1.0e-3);
}

TEST_F(MotionReferenceCompactDisabledTest, MissingCompactProfileReportsOnceWithoutBreakingFace)
{
  const auto [robot_state, face_target] = criticalPairExhaustedCase();
  const auto transition = update(robot_state, face_target);
  ASSERT_TRUE(transition.tracking);
  auto preparation_target = face_target;
  preparation_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  const auto result = update(robot_state, preparation_target);
  ASSERT_TRUE(result.tracking);
  EXPECT_TRUE(result.task);
  EXPECT_FALSE(reference_->compactManeuverActive());
  auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "compact_unavailable";
  }));
  recordQualifyingFaceShortfall(robot_state);
  events = reference_->drainCompactEvents();
  EXPECT_TRUE(events.empty());
}

TEST_F(MotionReferenceTest, ExplicitHoldPausesButHealthyHoldContinuesCommittedCompactGoal)
{
  const auto robot_state = state();
  const auto upright = evaluateCompactUprightPose(
    robot_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(upright);
  ASSERT_LE(std::abs(upright->signed_roll_error_rad), kCompactRollCapTestRad);
  JointMotionState moving_queue;
  robot_state.copyJointGroupPositions(group_, moving_queue.position);
  moving_queue.velocity = Eigen::VectorXd::Zero(6);
  moving_queue.acceleration = Eigen::VectorXd::Zero(6);
  moving_queue.velocity[0] = 0.02;
  recordQualifyingFaceShortfall(robot_state);
  admitCurrentPreparationWithLifecycleFixturePath(robot_state, moving_queue);
  const auto committed = update(
    robot_state, target(msg::TrackingTarget::FACE), nullptr, nullptr, moving_queue);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PATH");

  auto explicit_hold = target(msg::TrackingTarget::HOLD);
  const auto paused = update(
    robot_state, explicit_hold, nullptr, nullptr, moving_queue);
  EXPECT_FALSE(paused.task);
  EXPECT_EQ(reference_->diagnostics().state, "COMPACT_COMMITTED_WAITING_FOR_HEALTHY_COMMAND");
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);

  auto healthy_hold = explicit_hold;
  healthy_hold.allow_recovery = true;
  const auto continued = update(
    robot_state, healthy_hold, nullptr, nullptr, moving_queue);
  EXPECT_TRUE(continued.task);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PATH");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_NE(reference_->diagnostics().state, "COMPACT_COMMITTED_WAITING_FOR_HEALTHY_COMMAND");
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
}

TEST_F(MotionReferenceTest, CompactEntryCapCoversMeasuredCommandAndPublishedStopEnvelope)
{
  const auto start = compactGoalState(0.05);
  moveit::core::RobotState geometry_state(model_);
  CompactGeometryValidator geometry(model_, "link_base");
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  const double start_radius = geometry.maximumAssemblyRadius(start);

  Eigen::VectorXd published_stop_position = start_position;
  double published_stop_radius = start_radius;
  for (int second = 0; second <= 8; ++second) {
    for (int third = 0; third <= 8; ++third) {
      for (int fourth = 0; fourth <= 4; ++fourth) {
        Eigen::VectorXd candidate_position = start_position;
        const auto interpolate_limit = [this](const std::size_t joint, const int step,
          const int count) {
            const double fraction = static_cast<double>(step) / static_cast<double>(count);
            return limits_.lower_position[static_cast<Eigen::Index>(joint)] + fraction *
                   (limits_.upper_position[static_cast<Eigen::Index>(joint)] -
                   limits_.lower_position[static_cast<Eigen::Index>(joint)]);
          };
        candidate_position[1] = interpolate_limit(1U, second, 8);
        candidate_position[2] = interpolate_limit(2U, third, 8);
        candidate_position[3] = interpolate_limit(3U, fourth, 4);
        geometry_state.setJointGroupPositions(group_, candidate_position);
        geometry_state.update(true);
        const auto upright = evaluateCompactUprightPose(
          geometry_state.getGlobalLinkTransform("monitor_control_frame").linear());
        if (!upright ||
          std::abs(upright->signed_roll_error_rad) > kCompactRollCapTestRad)
        {
          continue;
        }
        const double radius = geometry.maximumAssemblyRadius(geometry_state);
        if (radius > published_stop_radius) {
          published_stop_position = std::move(candidate_position);
          published_stop_radius = radius;
        }
      }
    }
  }
  ASSERT_GT(published_stop_radius, start_radius + 1.0e-5);
  const auto absolute_pitch = [this, &geometry_state](const Eigen::VectorXd & position) {
      geometry_state.setJointGroupPositions(group_, position);
      geometry_state.update(true);
      const auto pose = evaluateCompactUprightPose(
        geometry_state.getGlobalLinkTransform("monitor_control_frame").linear());
      return pose ? std::optional<double>{std::abs(pose->signed_pitch_rad)} : std::nullopt;
    };
  const auto start_pitch = absolute_pitch(start_position);
  const auto published_stop_pitch = absolute_pitch(published_stop_position);
  Eigen::VectorXd goal_position;
  const auto compact_goal_state = compactGoalState();
  compact_goal_state.copyJointGroupPositions(group_, goal_position);
  const auto goal_pitch = absolute_pitch(goal_position);
  ASSERT_TRUE(start_pitch);
  ASSERT_TRUE(published_stop_pitch);
  ASSERT_TRUE(goal_pitch);
  const double expected_pitch_cap = std::max({
        0.5235987755982988, *start_pitch, *published_stop_pitch, *goal_pitch});

  recordQualifyingFaceShortfall(start);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  PublishedTrajectoryRecord publication;
  publication.id = 1U;
  const auto make_sample = [this](
    const Eigen::VectorXd & position, const double seconds) {
      moveit_servo::KinematicState sample(static_cast<int>(group_->getVariableCount()));
      sample.joint_names = parameters_.joint_names;
      sample.positions = position;
      sample.velocities.setZero(position.size());
      sample.accelerations.setZero(position.size());
      sample.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return sample;
    };
  publication.execution_queue.push_back(make_sample(start_position, now - 0.01));
  publication.execution_queue.push_back(make_sample(start_position, now + 0.01));
  publication.execution_queue.push_back(make_sample(published_stop_position, now + 0.5));
  publication.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.5) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord latest_publication = publication;
  latest_publication.id = 2U;
  PublishedTrajectoryRecord active_publication = publication;
  active_publication.id = 1U;
  active_publication.execution_queue.back() =
    make_sample(start_position, now + 0.5);

  const auto committed = update(
    start, target(msg::TrackingTarget::FACE), &active_publication, &latest_publication,
    JointMotionState{start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)});
  (void)committed;
  admitCurrentPreparationWithLifecycleFixturePath(
    start, JointMotionState{
        start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)},
    &active_publication, &latest_publication);
  ASSERT_TRUE(reference_->compactManeuverActive());
  const double expected_cap = std::max({0.40, start_radius,
        published_stop_radius});
  EXPECT_NEAR(reference_->diagnostics().compact_entry_radius_cap_m, expected_cap, 1.0e-9);
  EXPECT_NEAR(reference_->diagnostics().compact_pitch_cap_rad, expected_pitch_cap, 1.0e-9);
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().planned_path_max_radius_m));

  reference_->reset(true);
  (void)update(start, target(msg::TrackingTarget::FACE), &publication, &publication);
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_NEAR(reference_->diagnostics().compact_entry_radius_cap_m, expected_cap, 1.0e-9);
  EXPECT_NEAR(reference_->diagnostics().compact_pitch_cap_rad, expected_pitch_cap, 1.0e-9);
}

TEST_F(MotionReferenceTest, CompactPitchCapIncludesFreshActualAndSurvivesRearm)
{
  auto command_state = compactGoalState(0.05);
  command_state.setVariablePosition("joint5", -0.8);
  command_state.update(true);
  auto measured_state = command_state;
  measured_state.setVariablePosition("joint5", -0.7);
  measured_state.update(true);
  Eigen::VectorXd command_position;
  command_state.copyJointGroupPositions(group_, command_position);
  Eigen::VectorXd measured_position;
  measured_state.copyJointGroupPositions(group_, measured_position);

  const auto measured_pose = evaluateCompactUprightPose(
    measured_state.getGlobalLinkTransform("monitor_control_frame").linear());
  const auto command_pose = evaluateCompactUprightPose(
    command_state.getGlobalLinkTransform("monitor_control_frame").linear());
  const auto goal_state = compactGoalState();
  const auto goal_pose = evaluateCompactUprightPose(
    goal_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(measured_pose);
  ASSERT_TRUE(command_pose);
  ASSERT_TRUE(goal_pose);
  const double measured_abs_pitch = std::abs(measured_pose->signed_pitch_rad);
  ASSERT_GT(measured_abs_pitch, 0.5235987755982988);
  ASSERT_LT(measured_abs_pitch, 1.3962634015954636);
  ASSERT_GT(std::abs(measured_abs_pitch - std::abs(command_pose->signed_pitch_rad)), 0.05);

  recordQualifyingFaceShortfall(command_state);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto sample = [this, &command_position](const double seconds) {
      moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
      point.joint_names = parameters_.joint_names;
      point.positions = command_position;
      point.velocities.setZero(command_position.size());
      point.accelerations.setZero(command_position.size());
      point.time_stamp = rclcpp::Time(
        static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
      return point;
    };
  PublishedTrajectoryRecord publication;
  publication.id = 1U;
  publication.execution_queue.push_back(sample(now - 0.02));
  publication.execution_queue.push_back(sample(now + 0.02));
  publication.execution_queue.push_back(sample(now + 0.20));
  publication.stationary_time = rclcpp::Time(
    static_cast<std::int64_t>(std::llround((now + 0.20) * 1.0e9)), RCL_ROS_TIME);
  PublishedTrajectoryRecord latest_publication = publication;
  latest_publication.id = 2U;
  const JointMotionState actual{
    measured_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const JointMotionState queued{
    command_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  (void)reference_->update(
    command_state, target(msg::TrackingTarget::FACE), *planning_scene, now,
    kSceneRevision, actual, now, queued, &publication, &latest_publication);
  admitCurrentPreparationWithLifecycleFixturePath(
    command_state, queued, &publication, &latest_publication, kSceneRevision, actual);
  ASSERT_TRUE(reference_->compactManeuverActive());

  const double expected_cap = std::max({
        0.5235987755982988, measured_abs_pitch,
        std::abs(command_pose->signed_pitch_rad),
        std::abs(goal_pose->signed_pitch_rad)});
  EXPECT_NEAR(reference_->diagnostics().compact_pitch_cap_rad, expected_cap, 1.0e-9);

  reference_->reset(true);
  const double rearm_time = now + 0.02;
  (void)reference_->update(
    command_state, target(msg::TrackingTarget::FACE), *planning_scene, rearm_time,
    kSceneRevision, actual, rearm_time, queued, &publication, &latest_publication);
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_pitch_cap_rad, expected_cap);
}

TEST_F(MotionReferenceTest, CompactCandidateValidationRejectsPitchBeyondCommittedCap)
{
  // Keep the known folded wrist (J4=0, J6=pi) upright while varying only J5.
  // Interpolating toward nearScreenPoleState() also changes screen roll and makes
  // an unrelated hard-roll guard mask this pitch-envelope regression.
  auto entry = compactGoalState();
  entry.setVariablePosition("joint5", -0.8);
  entry.update(true);
  auto candidate = entry;
  candidate.setVariablePosition("joint5", -0.7);
  candidate.update(true);
  Eigen::VectorXd entry_position;
  Eigen::VectorXd candidate_position;
  entry.copyJointGroupPositions(group_, entry_position);
  candidate.copyJointGroupPositions(group_, candidate_position);
  const auto entry_pose = evaluateCompactUprightPose(
    entry.getGlobalLinkTransform("monitor_control_frame").linear());
  const auto candidate_pose = evaluateCompactUprightPose(
    candidate.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(entry_pose);
  ASSERT_TRUE(candidate_pose);
  ASSERT_LE(std::abs(entry_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_LE(std::abs(candidate_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_GT(
    std::abs(candidate_pose->signed_pitch_rad),
    std::abs(entry_pose->signed_pitch_rad) + kCompactPitchValidationToleranceRad);
  ASSERT_LT(std::abs(candidate_pose->signed_pitch_rad), 1.3962634015954636);

  recordQualifyingFaceShortfall(entry);
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const JointMotionState actual{
    entry_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const JointMotionState queued = actual;
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  (void)reference_->update(
    entry, target(msg::TrackingTarget::FACE), *planning_scene, now,
    kSceneRevision, actual, now, queued);
  admitCurrentPreparationWithLifecycleFixturePath(entry, queued);
  ASSERT_TRUE(reference_->compactManeuverActive());
  CompactGeometryValidator geometry(model_, "link_base");
  EXPECT_LE(
    geometry.maximumAssemblyRadius(candidate),
    reference_->diagnostics().compact_entry_radius_cap_m + 1.0e-6);
  ASSERT_GT(
    std::abs(candidate_pose->signed_pitch_rad),
    reference_->diagnostics().compact_pitch_cap_rad + kCompactPitchValidationToleranceRad);

  std::string violation;
  EXPECT_TRUE(reference_->validateCompactPathPosition(entry, entry_position, &violation));
  EXPECT_TRUE(violation.empty());
  EXPECT_FALSE(reference_->validateCompactPathPosition(
      entry, candidate_position, &violation));
  EXPECT_EQ(violation, "screen pitch");
}

TEST_F(MotionReferenceTest, CompactRadiusFloorLatchesOnlyAfterPublishedEnvelopeEnters)
{
  auto outside = state();
  outside.setVariablePosition("joint1", 6.20);
  outside.setVariablePosition("joint2", 0.233);
  outside.setVariablePosition("joint3", -1.71);
  outside.setVariablePosition("joint4", 0.0168);
  outside.setVariablePosition("joint5", -0.8338);
  outside.setVariablePosition("joint6", 6.27);
  outside.update();
  const auto inside = compactGoalState();
  CompactGeometryValidator geometry(model_, "link_base");
  Eigen::VectorXd outside_position;
  Eigen::VectorXd inside_position;
  outside.copyJointGroupPositions(group_, outside_position);
  inside.copyJointGroupPositions(group_, inside_position);
  ASSERT_GT(geometry.maximumAssemblyRadius(outside), 0.40);
  ASSERT_LE(geometry.maximumAssemblyRadius(inside), 0.40 + 1.0e-6);

  const auto now_seconds = [this]() {
      return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - clock_origin_).count();
    };
  const auto make_publication = [this](
    const std::uint64_t id, const Eigen::VectorXd & position, const double now) {
      PublishedTrajectoryRecord publication;
      publication.id = id;
      const auto sample = [this, &position](const double seconds) {
          moveit_servo::KinematicState point(static_cast<int>(group_->getVariableCount()));
          point.joint_names = parameters_.joint_names;
          point.positions = position;
          point.velocities.setZero(position.size());
          point.accelerations.setZero(position.size());
          point.time_stamp = rclcpp::Time(
            static_cast<std::int64_t>(std::llround(seconds * 1.0e9)), RCL_ROS_TIME);
          return point;
        };
      publication.execution_queue.push_back(sample(now - 0.02));
      publication.execution_queue.push_back(sample(now + 0.02));
      publication.execution_queue.push_back(sample(now + 0.20));
      publication.stationary_time = rclcpp::Time(
        static_cast<std::int64_t>(std::llround((now + 0.20) * 1.0e9)), RCL_ROS_TIME);
      return publication;
    };
  const auto update_with = [this](
    const moveit::core::RobotState & robot_state,
    const Eigen::VectorXd & actual_position,
    const Eigen::VectorXd & queued_position,
    const PublishedTrajectoryRecord & active,
    const PublishedTrajectoryRecord & latest,
    const double now) {
      const JointMotionState actual{
        actual_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
      const JointMotionState queued{
        queued_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(
        robot_state, target(msg::TrackingTarget::FACE), *planning_scene, now,
        kSceneRevision, actual, now, queued, &active, &latest);
    };

  recordQualifyingFaceShortfall(outside);
  double now = now_seconds();
  auto active = make_publication(1U, outside_position, now);
  auto latest = make_publication(2U, outside_position, now);
  (void)update_with(outside, outside_position, outside_position, active, latest, now);
  admitCurrentPreparationWithLifecycleFixturePath(
    outside, JointMotionState{
        outside_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)},
    &active, &latest);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_GT(reference_->diagnostics().compact_entry_radius_cap_m, 0.40);
  const double entry_cap = reference_->diagnostics().compact_entry_radius_cap_m;

  now = now_seconds();
  (void)update_with(inside, inside_position, inside_position, active, latest, now);
  EXPECT_TRUE(reference_->validateCompactPathPosition(inside, outside_position));
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_entry_radius_cap_m, entry_cap);

  now = now_seconds();
  active = make_publication(3U, inside_position, now);
  latest = make_publication(4U, inside_position, now);
  (void)update_with(inside, inside_position, inside_position, active, latest, now);
  EXPECT_FALSE(reference_->validateCompactPathPosition(inside, outside_position));
  EXPECT_TRUE(reference_->validateCompactPathPosition(inside, inside_position));
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_entry_radius_cap_m, entry_cap);
  reference_->reset(true);
  EXPECT_FALSE(reference_->validateCompactPathPosition(inside, outside_position));
}

TEST_F(MotionReferenceTest, CompactRadiusRowsDeflectOutwardCommandNearEntryCap)
{
  auto start = state();
  start.setVariablePosition("joint1", 6.20);
  start.setVariablePosition("joint2", 0.233);
  start.setVariablePosition("joint3", -1.71);
  start.setVariablePosition("joint4", 0.0168);
  start.setVariablePosition("joint5", -0.8338);
  start.setVariablePosition("joint6", 6.27);
  start.update();
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(position.size()),
    Eigen::VectorXd::Zero(position.size())};
  const auto edge_face = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(edge_face.tracking);
  ASSERT_TRUE(edge_face.j1_hold_position_rad);
  EXPECT_NE(reference_->diagnostics().j1_hold_direction, 0);

  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);

  const auto reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->compactManeuverActive());
  EXPECT_FALSE(reference.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().j1_hold_q, 0.0);
  ASSERT_GT(reference.compact_radius_constraints.size(), 0U);
  CompactGeometryValidator geometry(model_, "link_base");
  EXPECT_NEAR(
    reference_->diagnostics().compact_entry_radius_cap_m,
    geometry.maximumAssemblyRadius(start), 1.0e-6);

  const auto restrictive_row = std::max_element(
    reference.compact_radius_constraints.begin(),
    reference.compact_radius_constraints.end(), [](const auto & left, const auto & right) {
      const double left_rate = left.upper_bound / left.coefficients.norm();
      const double right_rate = right.upper_bound / right.coefficients.norm();
      return left_rate > right_rate;
    });
  ASSERT_NE(restrictive_row, reference.compact_radius_constraints.end());
  ASSERT_GT(restrictive_row->coefficients.norm(), 1.0e-8);

  HierarchicalVelocityTask hold_task;
  hold_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  hold_task.primary_reference = Eigen::VectorXd::Zero(6);
  hold_task.primary_weights = Eigen::VectorXd::Ones(6);
  hold_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  hold_task.secondary_reference = Eigen::VectorXd(0);
  hold_task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult stationary_hold = solver.solve(
    hold_task, settled, limits_, reference.compact_radius_constraints);
  ASSERT_TRUE(stationary_hold.command_available());
  EXPECT_NEAR(stationary_hold.joint_velocity.norm(), 0.0, 1.0e-5);
  for (const auto & row : reference.compact_radius_constraints) {
    EXPECT_GE(row.upper_bound, -1.0e-8);
  }

  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  task.primary_reference = 0.2 * restrictive_row->coefficients.normalized();
  task.primary_weights = Eigen::VectorXd::Ones(6);
  task.secondary_matrix = Eigen::MatrixXd(0, 6);
  task.secondary_reference = Eigen::VectorXd(0);
  task.secondary_weights = Eigen::VectorXd(0);
  const QpResult unguarded = solver.solve(task, settled, limits_);
  const QpResult guarded = solver.solve(
    task, settled, limits_, reference.compact_radius_constraints);
  ASSERT_TRUE(unguarded.command_available());
  ASSERT_TRUE(guarded.command_available());
  EXPECT_GT(
    restrictive_row->coefficients.dot(unguarded.joint_velocity),
    restrictive_row->upper_bound + 1.0e-6);
  EXPECT_LE(
    restrictive_row->coefficients.dot(guarded.joint_velocity),
    restrictive_row->upper_bound + 2.0e-5);
}

TEST_F(MotionReferenceTest, CompactRadiusRowsDoNotBindFarInsideFloorCap)
{
  const auto start = compactGoalState();
  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState settled{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_GT(reference.compact_radius_constraints.size(), 0U);

  const auto strongest_row = std::max_element(
    reference.compact_radius_constraints.begin(),
    reference.compact_radius_constraints.end(), [](const auto & left, const auto & right) {
      return left.coefficients.norm() < right.coefficients.norm();
    });
  ASSERT_NE(strongest_row, reference.compact_radius_constraints.end());
  ASSERT_GT(strongest_row->coefficients.norm(), 1.0e-8);

  HierarchicalVelocityTask task;
  task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  task.primary_reference = 0.001 * strongest_row->coefficients.normalized();
  task.primary_weights = Eigen::VectorXd::Ones(6);
  task.secondary_matrix = Eigen::MatrixXd(0, 6);
  task.secondary_reference = Eigen::VectorXd(0);
  task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult unguarded = solver.solve(task, settled, limits_);
  const QpResult guarded = solver.solve(
    task, settled, limits_, reference.compact_radius_constraints);
  ASSERT_TRUE(unguarded.command_available());
  ASSERT_TRUE(guarded.command_available());
  EXPECT_NEAR((guarded.joint_velocity - unguarded.joint_velocity).norm(), 0.0, 1.0e-5);
}

TEST_F(MotionReferenceTest, CompactPitchRowsHoldAtCapDeflectOutwardAndAllowInwardMotion)
{
  auto start = compactGoalState();
  start.setVariablePosition("joint5", -0.8);
  start.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  start.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  start.update(true);
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  const auto start_pose = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_pose);
  ASSERT_LE(std::abs(start_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  ASSERT_GT(std::abs(start_pose->signed_pitch_rad), 0.5235987755982988);
  ASSERT_LT(std::abs(start_pose->signed_pitch_rad), 1.3962634015954636);

  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  const JointMotionState settled{
    start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, settled);
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_EQ(reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(reference.compact_roll_constraints.size(), 2U);
  const auto boundary_row = std::min_element(
    reference.compact_pitch_constraints.begin(), reference.compact_pitch_constraints.end(),
    [](const auto & left, const auto & right) {return left.upper_bound < right.upper_bound;});
  ASSERT_NE(boundary_row, reference.compact_pitch_constraints.end());
  EXPECT_NEAR(boundary_row->upper_bound, 0.0, 1.0e-8);
  auto compact_constraints = reference.compact_radius_constraints;
  compact_constraints.insert(
    compact_constraints.end(), reference.compact_roll_constraints.begin(),
    reference.compact_roll_constraints.end());
  compact_constraints.insert(
    compact_constraints.end(), reference.compact_pitch_constraints.begin(),
    reference.compact_pitch_constraints.end());

  HierarchicalVelocityTask hold_task;
  hold_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  hold_task.primary_reference = Eigen::VectorXd::Zero(6);
  hold_task.primary_weights = Eigen::VectorXd::Ones(6);
  hold_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  hold_task.secondary_reference = Eigen::VectorXd(0);
  hold_task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  const QpResult stationary_hold = solver.solve(
    hold_task, settled, limits_, compact_constraints);
  ASSERT_TRUE(stationary_hold.command_available());
  EXPECT_NEAR(stationary_hold.joint_velocity.norm(), 0.0, 1.0e-5);

  HierarchicalVelocityTask outward_task = hold_task;
  outward_task.primary_reference = 0.2 * boundary_row->coefficients.normalized();
  const QpResult unguarded_outward = solver.solve(outward_task, settled, limits_);
  const QpResult guarded_outward = solver.solve(
    outward_task, settled, limits_, compact_constraints);
  ASSERT_TRUE(unguarded_outward.command_available());
  ASSERT_TRUE(guarded_outward.command_available());
  EXPECT_GT(
    boundary_row->coefficients.dot(unguarded_outward.joint_velocity),
    boundary_row->upper_bound + 1.0e-6);
  EXPECT_LE(
    boundary_row->coefficients.dot(guarded_outward.joint_velocity),
    boundary_row->upper_bound + 2.0e-5);

  HierarchicalVelocityTask inward_task = hold_task;
  inward_task.primary_reference = -0.1 * boundary_row->coefficients.normalized();
  const QpResult guarded_inward = solver.solve(
    inward_task, settled, limits_, compact_constraints);
  ASSERT_TRUE(guarded_inward.command_available());
  EXPECT_LT(boundary_row->coefficients.dot(guarded_inward.joint_velocity), -1.0e-5);

  const Eigen::VectorXd outward_velocity =
    0.02 * boundary_row->coefficients.normalized();
  const JointMotionState moving_queue{
    start_position, outward_velocity, Eigen::VectorXd::Zero(6)};
  const auto moving_reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, moving_queue);
  ASSERT_EQ(moving_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(moving_reference.compact_roll_constraints.size(), 2U);
  const auto moving_boundary_row = std::find_if(
    moving_reference.compact_pitch_constraints.begin(),
    moving_reference.compact_pitch_constraints.end(),
    [&boundary_row](const auto & row) {
      return row.coefficients.isApprox(boundary_row->coefficients, 1.0e-12);
    });
  ASSERT_NE(moving_boundary_row, moving_reference.compact_pitch_constraints.end());
  const double current_outward_speed =
    boundary_row->coefficients.dot(moving_queue.velocity);
  EXPECT_LE(moving_boundary_row->upper_bound, current_outward_speed + 1.0e-6);

  const QpResult unguarded_moving = solver.solve(outward_task, moving_queue, limits_);
  auto moving_constraints = moving_reference.compact_radius_constraints;
  moving_constraints.insert(
    moving_constraints.end(), moving_reference.compact_roll_constraints.begin(),
    moving_reference.compact_roll_constraints.end());
  moving_constraints.insert(
    moving_constraints.end(), moving_reference.compact_pitch_constraints.begin(),
    moving_reference.compact_pitch_constraints.end());
  const QpResult guarded_moving = solver.solve(
    outward_task, moving_queue, limits_, moving_constraints);
  ASSERT_TRUE(unguarded_moving.command_available());
  ASSERT_TRUE(guarded_moving.command_available());
  EXPECT_GT(
    moving_boundary_row->coefficients.dot(unguarded_moving.joint_velocity),
    moving_boundary_row->upper_bound + 1.0e-6);
  EXPECT_LE(
    moving_boundary_row->coefficients.dot(guarded_moving.joint_velocity),
    moving_boundary_row->upper_bound + 2.0e-5);
}

TEST_F(MotionReferenceTest, CompactOrientationViableMotionHasCertifiedBrakingTail)
{
  const auto start = compactGoalState(0.05);
  const auto start_pose = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_pose);
  recordQualifyingFaceShortfall(start);
  admitCurrentPreparationWithLifecycleFixturePath(start);
  const auto stationary_reference = update(
    start, target(msg::TrackingTarget::FACE));
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_TRUE(std::isfinite(reference_->diagnostics().compact_pitch_cap_rad));
  ASSERT_GT(
    reference_->diagnostics().compact_pitch_cap_rad -
    std::abs(start_pose->signed_pitch_rad), 0.02);
  ASSERT_GT(kCompactRollCapTestRad - std::abs(start_pose->signed_roll_error_rad), 0.02);
  CompactGeometryValidator geometry(model_, "link_base");
  EXPECT_GT(
    reference_->diagnostics().compact_entry_radius_cap_m -
    geometry.maximumAssemblyRadius(start), 0.01);
  ASSERT_EQ(stationary_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(stationary_reference.compact_roll_constraints.size(), 2U);
  const auto outward_row = std::min_element(
    stationary_reference.compact_pitch_constraints.begin(),
    stationary_reference.compact_pitch_constraints.end(),
    [](const auto & left, const auto & right) {return left.upper_bound < right.upper_bound;});
  ASSERT_NE(outward_row, stationary_reference.compact_pitch_constraints.end());

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const auto roll_outward_row = std::min_element(
    stationary_reference.compact_roll_constraints.begin(),
    stationary_reference.compact_roll_constraints.end(),
    [](const auto & left, const auto & right) {return left.upper_bound < right.upper_bound;});
  ASSERT_NE(roll_outward_row, stationary_reference.compact_roll_constraints.end());
  const Eigen::VectorXd velocity = 0.03 * outward_row->coefficients.normalized() +
    0.03 * roll_outward_row->coefficients.normalized();
  const JointMotionState moving{position, velocity, Eigen::VectorXd::Zero(6)};
  const auto moving_reference = update(
    start, target(msg::TrackingTarget::FACE), nullptr, nullptr, moving);
  ASSERT_EQ(moving_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(moving_reference.compact_roll_constraints.size(), 2U);

  HierarchicalVelocityTask outward_task;
  outward_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  outward_task.primary_reference = 0.2 * outward_row->coefficients.normalized() +
    0.2 * roll_outward_row->coefficients.normalized();
  outward_task.primary_weights = Eigen::VectorXd::Ones(6);
  outward_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  outward_task.secondary_reference = Eigen::VectorXd(0);
  outward_task.secondary_weights = Eigen::VectorXd(0);
  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = parameters_.residual_command_latency_sec;
  const HierarchicalVelocityQp solver(qp_config);
  auto constraints = moving_reference.compact_radius_constraints;
  constraints.insert(
    constraints.end(), moving_reference.compact_roll_constraints.begin(),
    moving_reference.compact_roll_constraints.end());
  constraints.insert(
    constraints.end(), moving_reference.compact_pitch_constraints.begin(),
    moving_reference.compact_pitch_constraints.end());
  const QpResult command = solver.solve(outward_task, moving, limits_, constraints);
  ASSERT_TRUE(command.command_available());

  const Eigen::VectorXd next_position = position +
    parameters_.control_period_sec * command.joint_velocity;
  const Eigen::VectorXd next_acceleration =
    (command.joint_velocity - velocity) / parameters_.control_period_sec;
  const JointMotionState branch{next_position, command.joint_velocity, next_acceleration};
  EmergencyBrakeTailConfig tail_config;
  tail_config.period_sec = parameters_.control_period_sec;
  const EmergencyBrakeTail tail = EmergencyBrakeTailGenerator(tail_config).generate(
    branch, limits_);
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  ASSERT_GT(tail.first_stationary_point, 0U);

  for (std::size_t index = 0U; index < tail.points.size(); ++index) {
    std::string violation;
    EXPECT_TRUE(reference_->validateCompactPathPosition(
        start, tail.points[index].position, &violation)) <<
      "braking tail point " << index << " violates " << violation;
  }
}

TEST_F(MotionReferenceTest, CompactQueueTailRowsAllowDumpedStateAndCertifyEmergencyBrake)
{
  auto start = state();
  const Eigen::VectorXd position = (Eigen::VectorXd(6) <<
    6.1149018406659783, 0.26252163795647276, -1.6730647452367475,
    -0.051544811767068766, -0.80646278725974319, 6.2830336538974318).finished();
  const Eigen::VectorXd velocity = (Eigen::VectorXd(6) <<
    -0.002, -0.0000944063747657, -0.00199843840694, -0.002, 0.002, -0.002).finished();
  const Eigen::VectorXd acceleration = (Eigen::VectorXd(6) <<
    -0.2, -0.0094406375, -0.1998438407, -0.2, 0.2, -0.2).finished();
  start.setJointGroupPositions(group_, position);
  start.setJointGroupVelocities(group_, velocity);
  start.setJointGroupAccelerations(group_, acceleration);
  start.update(true);
  const JointMotionState queue_tail{position, velocity, acceleration};

  EmergencyBrakeTailConfig prior_tail_config;
  prior_tail_config.period_sec = parameters_.control_period_sec;
  const EmergencyBrakeTail prior_tail = EmergencyBrakeTailGenerator(
    prior_tail_config).generate(queue_tail, limits_);
  ASSERT_TRUE(prior_tail.command_available()) << prior_tail.failure_reason;
  const double evidence_start = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto selected = update(start, target(msg::TrackingTarget::FACE));
  ASSERT_TRUE(selected.tracking);
  auto shortfall = *selected.tracking;
  shortfall.has_face_primary_roll = true;
  shortfall.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
  shortfall.face_primary_feedforward_radps.setZero();
  QpResult accepted_shortfall;
  accepted_shortfall.status = QpStatus::kSolved;
  accepted_shortfall.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
  for (int tick = 0; tick <= 8; ++tick) {
    reference_->recordFaceCommand(
      shortfall, accepted_shortfall,
      evidence_start + 0.02 * static_cast<double>(tick + 1), true);
  }
  const double update_time = evidence_start + 0.18;
  PublishedTrajectoryRecord publication;
  publication.id = 1U;
  for (std::size_t index = 0U; index < prior_tail.points.size(); ++index) {
    const auto & point = prior_tail.points[index];
    moveit_servo::KinematicState sample(static_cast<int>(group_->getVariableCount()));
    sample.joint_names = parameters_.joint_names;
    sample.positions = point.position;
    sample.velocities = point.velocity;
    sample.accelerations = point.acceleration;
    sample.time_stamp = rclcpp::Time(
      static_cast<std::int64_t>(std::llround(
        (update_time + static_cast<double>(index) * parameters_.control_period_sec) * 1.0e9)),
      RCL_ROS_TIME);
    publication.execution_queue.push_back(std::move(sample));
  }
  publication.stationary_time = publication.execution_queue.back().time_stamp;

  MotionReferenceResult compact_reference;
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    compact_reference = reference_->update(
      start, target(msg::TrackingTarget::FACE), *planning_scene, update_time,
      kSceneRevision, queue_tail, update_time, queue_tail, &publication, &publication);
  }
  admitCurrentPreparationWithLifecycleFixturePath(
    start, queue_tail, &publication, &publication, kSceneRevision, queue_tail);
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    compact_reference = reference_->update(
      start, target(msg::TrackingTarget::FACE), *planning_scene, update_time,
      kSceneRevision, queue_tail, update_time, queue_tail, &publication, &publication);
  }
  ASSERT_TRUE(reference_->compactManeuverActive());
  ASSERT_EQ(compact_reference.compact_radius_constraints.size(), 6U);
  ASSERT_EQ(compact_reference.compact_pitch_constraints.size(), 2U);
  ASSERT_EQ(compact_reference.compact_roll_constraints.size(), 2U);
  CompactGeometryValidator geometry(model_, "link_base");
  for (const auto & point : prior_tail.points) {
    auto queued_state = start;
    queued_state.setJointGroupPositions(group_, point.position);
    queued_state.update(true);
    EXPECT_LE(
      geometry.maximumAssemblyRadius(queued_state),
      reference_->diagnostics().compact_entry_radius_cap_m + 1.0e-6);
  }

  HierarchicalVelocityTask outward_task;
  outward_task.primary_matrix = Eigen::MatrixXd::Identity(6, 6);
  outward_task.primary_reference = Eigen::VectorXd::Zero(6);
  const auto & radius_boundary_row = compact_reference.compact_radius_constraints[5];
  const auto & pitch_boundary_row = compact_reference.compact_pitch_constraints[0];
  outward_task.primary_reference +=
    0.2 * radius_boundary_row.coefficients.normalized() +
    0.2 * pitch_boundary_row.coefficients.normalized();
  outward_task.primary_weights = Eigen::VectorXd::Ones(6);
  outward_task.secondary_matrix = Eigen::MatrixXd(0, 6);
  outward_task.secondary_reference = Eigen::VectorXd(0);
  outward_task.secondary_weights = Eigen::VectorXd(0);

  HierarchicalVelocityQpConfig qp_config;
  qp_config.period_sec = parameters_.control_period_sec;
  qp_config.residual_command_latency_sec = 0.0;
  qp_config.velocity_regularization = parameters_.velocity_regularization_weight;
  qp_config.primary_preservation_tolerance =
    parameters_.primary_orientation_tolerance_radps;
  qp_config.solver_absolute_tolerance = parameters_.solver_absolute_tolerance;
  qp_config.solver_relative_tolerance = parameters_.solver_relative_tolerance;
  qp_config.solver_time_limit_sec = parameters_.solver_time_limit_sec;
  qp_config.solver_max_iterations = parameters_.solver_max_iterations;
  const HierarchicalVelocityQp solver(qp_config);
  auto constraints = compact_reference.compact_radius_constraints;
  constraints.insert(
    constraints.end(), compact_reference.compact_roll_constraints.begin(),
    compact_reference.compact_roll_constraints.end());
  constraints.insert(
    constraints.end(), compact_reference.compact_pitch_constraints.begin(),
    compact_reference.compact_pitch_constraints.end());
  const QpResult command = solver.solve(outward_task, queue_tail, limits_, constraints);
  ASSERT_TRUE(command.command_available()) << static_cast<int>(command.status);
  EXPECT_LE(
    radius_boundary_row.coefficients.dot(command.joint_velocity),
    radius_boundary_row.upper_bound + 2.0e-5);
  EXPECT_LE(
    pitch_boundary_row.coefficients.dot(command.joint_velocity),
    pitch_boundary_row.upper_bound + 2.0e-5);

  const Eigen::VectorXd branch_position =
    position + parameters_.control_period_sec * command.joint_velocity;
  const Eigen::VectorXd branch_acceleration =
    (command.joint_velocity - velocity) / parameters_.control_period_sec;
  const EmergencyBrakeTailGenerator tail_generator(
    EmergencyBrakeTailConfig{parameters_.control_period_sec});
  const EmergencyBrakeTail tail = tail_generator.generate(
    JointMotionState{branch_position, command.joint_velocity, branch_acceleration}, limits_);
  ASSERT_TRUE(tail.command_available()) << tail.failure_reason;
  bool outward_tail_rejected_by_radius = false;
  for (std::size_t index = 0U; index < tail.points.size(); ++index) {
    std::string violation;
    if (!reference_->validateCompactPathPosition(
        start, tail.points[index].position, &violation))
    {
      EXPECT_EQ(violation, "assembly radius") << "tail point " << index;
      outward_tail_rejected_by_radius = true;
      break;
    }
  }
  EXPECT_TRUE(outward_tail_rejected_by_radius)
    << "the outward probe should remain covered by the hard compact radius guard";

  const Eigen::VectorXd goal = (Eigen::VectorXd(6) <<
    0.0, -0.2715944, -2.8699983, 0.0, 1.5707963267948966,
    limits_.upper_position[5] - 0.02).finished();
  Eigen::VectorXd prefix = goal;
  prefix[0] = position[0] + std::clamp(goal[0] - position[0], -0.35, 0.35);
  prefix[3] = position[3];
  const auto planned_fold_task = makeJointPathVelocityTask(position, prefix, 6.0, 0.70);
  ASSERT_TRUE(planned_fold_task);
  const QpResult planned_fold_command = solver.solve(
    *planned_fold_task, queue_tail, limits_, constraints);
  ASSERT_TRUE(planned_fold_command.command_available()) <<
    static_cast<int>(planned_fold_command.status);
  const Eigen::VectorXd fold_branch_position = position +
    parameters_.control_period_sec * planned_fold_command.joint_velocity;
  const Eigen::VectorXd fold_branch_acceleration =
    (planned_fold_command.joint_velocity - velocity) / parameters_.control_period_sec;
  const EmergencyBrakeTail fold_tail = tail_generator.generate(
    JointMotionState{
        fold_branch_position, planned_fold_command.joint_velocity, fold_branch_acceleration},
    limits_);
  ASSERT_TRUE(fold_tail.command_available()) << fold_tail.failure_reason;
  for (std::size_t index = 0U; index < fold_tail.points.size(); ++index) {
    std::string violation;
    EXPECT_TRUE(reference_->validateCompactPathPosition(
      start, fold_tail.points[index].position, &violation)) <<
      "planned fold tail point " << index << " violates " << violation;
  }
}

TEST_F(MotionReferenceTest, CompactCommitAdoptsCheckedPathAndPreservesGoalAcrossRearm)
{
  const auto [start, face_target] = criticalPairExhaustedCase();
  const auto start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  ASSERT_LE(std::abs(start_upright->signed_roll_error_rad), kCompactRollCapTestRad);
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  const JointMotionState settled_queue{
    start_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const auto initial_face = update(
    start, face_target, nullptr, nullptr, settled_queue);
  ASSERT_TRUE(initial_face.task);
  auto preparation_target = face_target;
  preparation_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  const auto waiting = update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  EXPECT_TRUE(waiting.task);
  EXPECT_TRUE(waiting.tracking);
  EXPECT_FALSE(waiting.follows_path);
  // Entry-envelope caps are committed only after a worker result is admitted.
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_effective_radius_cap_m));
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_entry_radius_cap_m));
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_pitch_cap_rad));

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  MotionReferenceResult planned;
  do {
    planned = update(
      start, preparation_target, nullptr, nullptr, settled_queue);
    if (reference_->diagnostics().plans_accepted > 0U) {
      break;
    }
    std::this_thread::sleep_for(2ms);
  } while (std::chrono::steady_clock::now() < deadline);
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U)
    << reference_->diagnostics().last_plan_message;
  EXPECT_TRUE(planned.follows_path);
  ASSERT_TRUE(planned.task);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PATH");
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_entry_radius_cap_m));
  EXPECT_DOUBLE_EQ(
    reference_->diagnostics().compact_effective_radius_cap_m,
    reference_->diagnostics().compact_entry_radius_cap_m);
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_pitch_cap_rad));

  const auto events = reference_->drainCompactEvents();
  const auto triggered = std::find_if(
    events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(triggered, events.end());
  EXPECT_EQ(triggered->generation_id, 1U);
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "path_started";
  }));

  // A controlled executor re-arm drops the follower but keeps the immutable
  // compact goal. The next healthy FACE update must request a fresh path from
  // current q and must not expose the goal as an unplanned direct joint task.
  reference_->reset(true);
  ASSERT_TRUE(reference_->compactManeuverActive());
  const auto rearmed = update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
  if (rearmed.task) {
    EXPECT_EQ(rearmed.task->primary_matrix.rows(), 1);
    EXPECT_EQ(rearmed.task->secondary_reference.norm(), 0.0);
  }
  const auto replan_deadline = std::chrono::steady_clock::now() + 3s;
  while (reference_->diagnostics().plans_requested < 2U &&
    std::chrono::steady_clock::now() < replan_deadline)
  {
    (void)update(start, preparation_target, nullptr, nullptr, settled_queue);
    std::this_thread::sleep_for(2ms);
  }
  EXPECT_GE(reference_->diagnostics().plans_requested, 2U);
  EXPECT_LE(reference_->diagnostics().plans_accepted,
    reference_->diagnostics().plans_requested);
  const auto rearm_events = reference_->drainCompactEvents();
  EXPECT_FALSE(std::any_of(
      rearm_events.begin(), rearm_events.end(), [](const CompactEvent & event) {
        return event.event == "path_started" && event.generation_id == 1U;
      }));
}

TEST_F(MotionReferenceTest, CompactPlanningKeepsUprightCorrectionOutsideRollAdmission)
{
  auto start = compactGoalState(0.05);
  auto start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  auto probe = start;
  probe.setVariablePosition("joint6", probe.getVariablePosition("joint6") + 1.0e-4);
  probe.update(true);
  const auto probe_upright = evaluateCompactUprightPose(
    probe.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(probe_upright);
  const double joint6_direction =
    std::abs(probe_upright->signed_roll_error_rad) >
    std::abs(start_upright->signed_roll_error_rad) ? 1.0 : -1.0;
  start.setVariablePosition("joint6", start.getVariablePosition("joint6") +
    joint6_direction * 0.15);
  start.update(true);
  start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  ASSERT_GT(std::abs(start_upright->signed_roll_error_rad), kCompactRollCapTestRad);

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  JointMotionState moving_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  moving_queue.velocity[0] = 0.02;
  const JointMotionState actual_feedback{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const double now = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - clock_origin_).count();
  const auto correction = MotionReferenceTestPeer::handleQualifiedPreparationForTest(
    *reference_, start, target(msg::TrackingTarget::FACE), *planning_scene,
    now, kSceneRevision, actual_feedback, moving_queue);

  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_phase, "UPRIGHT_START");
  EXPECT_EQ(reference_->diagnostics().state, "COMPACT_UPRIGHT_START");
  EXPECT_FALSE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  ASSERT_TRUE(correction.task);
  EXPECT_EQ(correction.task->primary_matrix.rows(), 1);
  EXPECT_EQ(correction.task->secondary_matrix.rows(), 6);
  EXPECT_GT(correction.task->primary_reference.norm(), 1.0e-6);

  // Once feedback and the command queue have recovered inside the roll
  // admission envelope and are settled, the same pending policy may commit.
  const auto recovered = compactGoalState(0.05);
  Eigen::VectorXd recovered_position;
  recovered.copyJointGroupPositions(group_, recovered_position);
  const JointMotionState recovered_queue{
    recovered_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  const JointMotionState recovered_feedback{
    recovered_position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  ASSERT_TRUE(MotionReferenceTestPeer::armFaceUprightRollGuardFromFeedbackForTest(
      *reference_, recovered, now + parameters_.control_period_sec,
      recovered_feedback, recovered_queue));
  const auto recovered_update = MotionReferenceTestPeer::handleQualifiedPreparationForTest(
    *reference_, recovered, target(msg::TrackingTarget::FACE), *planning_scene,
    now + parameters_.control_period_sec, kSceneRevision,
    recovered_feedback, recovered_queue, true);
  EXPECT_TRUE(reference_->diagnostics().upright_roll_guard_armed);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_GT(reference_->diagnostics().compact_generation, 0U);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  EXPECT_FALSE(recovered_update.follows_path);
}

TEST_F(MotionReferenceTest, CompactPreparationKeepsFaceActiveWhilePlanRuns)
{
  const auto [start, face_target] = criticalPairExhaustedCase();

  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  Eigen::VectorXd moving_velocity = Eigen::VectorXd::Zero(6);
  moving_velocity[0] = 0.02;
  const JointMotionState settled_queue{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  // Let a stationary real-FK update arm the roll guard before requesting the
  // plan. Motion is introduced only after preparation has started.
  (void)update(
    start, face_target, nullptr, nullptr, settled_queue);
  auto preparation_target = face_target;
  preparation_target.face_stamp = rclcpp::Time(1, 20'000'000, RCL_ROS_TIME);
  (void)update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  ASSERT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_FALSE(reference_->compactManeuverActive());
  ASSERT_GE(reference_->diagnostics().plans_requested, 1U);
  const auto requests_while_preparing = reference_->diagnostics().plans_requested;
  // Hold the worker result pending deterministically so this test checks the
  // FACE behavior while preparation is running, independent of planner speed.
  MotionReferenceTestPeer::cancelPlannerForTest(*reference_);
  MotionReferenceTestPeer::activateResultGeneration(
    *reference_, reference_->diagnostics().compact_generation + 100U);

  const JointMotionState moving_queue{
    position, moving_velocity, Eigen::VectorXd::Zero(6)};
  preparation_target.face_stamp = rclcpp::Time(1, 40'000'000, RCL_ROS_TIME);
  const auto moving_face = update(
    start, preparation_target, nullptr, nullptr, moving_queue);
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_while_preparing);
  EXPECT_TRUE(moving_face.tracking);
  EXPECT_TRUE(moving_face.task);
  EXPECT_FALSE(moving_face.follows_path);

  Eigen::VectorXd moving_acceleration = Eigen::VectorXd::Zero(6);
  moving_acceleration[0] = 0.04;
  const JointMotionState accelerating_queue{
    position, Eigen::VectorXd::Zero(6), moving_acceleration};
  preparation_target.face_stamp = rclcpp::Time(1, 60'000'000, RCL_ROS_TIME);
  (void)update(
    start, preparation_target, nullptr, nullptr, accelerating_queue);
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_while_preparing);

  preparation_target.face_stamp = rclcpp::Time(1, 80'000'000, RCL_ROS_TIME);
  const auto settled = update(
    start, preparation_target, nullptr, nullptr, settled_queue);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_TRUE(MotionReferenceTestPeer::hasPreparation(*reference_));
  EXPECT_EQ(reference_->diagnostics().plans_requested, requests_while_preparing);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "PREPARING");
  EXPECT_FALSE(settled.follows_path);
}

TEST_F(MotionReferenceCameraTest, CompactLifecycleCompletesAndShortfallAloneDoesNotRestart)
{
  auto start = compactGoalState(0.05);
  start.setVariablePosition("joint6", start.getVariablePosition("joint6") + 0.04);
  start.update(true);
  const auto start_upright = evaluateCompactUprightPose(
    start.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(start_upright);
  ASSERT_GT(std::abs(start_upright->signed_roll_error_rad), 0.03);
  ASSERT_LT(std::abs(start_upright->signed_roll_error_rad), kCompactRollCapTestRad);
  // This lifecycle test starts after pair-gate qualification; the real FK gate
  // is exercised by dedicated public-update cases above.
  admitCurrentPreparationWithLifecycleFixturePath(start);
  const auto plan_requests_before_lifecycle = reference_->diagnostics().plans_requested;
  const auto committed_generation = reference_->diagnostics().compact_generation;

  double simulated_time = 1.18;
  const auto motion_state = [this](const moveit::core::RobotState & state) {
      JointMotionState result;
      state.copyJointGroupPositions(group_, result.position);
      state.copyJointGroupVelocities(group_, result.velocity);
      state.copyJointGroupAccelerations(group_, result.acceleration);
      return result;
    };
  const auto aligned_face = [this](const moveit::core::RobotState & state) {
      auto command = target(msg::TrackingTarget::FACE);
      const auto & screen = state.getGlobalLinkTransform("monitor_control_frame");
      const Eigen::Quaterniond orientation(screen.linear());
      command.pose.position.x = screen.translation().x();
      command.pose.position.y = screen.translation().y();
      command.pose.position.z = screen.translation().z();
      command.pose.orientation.x = orientation.x();
      command.pose.orientation.y = orientation.y();
      command.pose.orientation.z = orientation.z();
      command.pose.orientation.w = orientation.w();
      const auto * gaze_link = parameters_.gaze_frame.empty() ? nullptr :
        model_->getLinkModel(parameters_.gaze_frame);
      Eigen::Isometry3d look_frame = gaze_link ?
        state.getGlobalLinkTransform(gaze_link) : screen;
      if (gaze_link) {
        look_frame.linear() *= face_tracking_arm::opticalToPointingRotation();
      }
      const Eigen::Vector3d face = look_frame.translation() +
        0.8 * look_frame.linear().col(0);
      command.face.x = face.x();
      command.face.y = face.y();
      command.face.z = face.z();
      return command;
    };
  const auto misaligned_face = [&aligned_face](const moveit::core::RobotState & state) {
      auto command = aligned_face(state);
      const auto & screen = state.getGlobalLinkTransform("monitor_control_frame");
      const Eigen::Vector3d face(
        command.face.x, command.face.y, command.face.z);
      const Eigen::Vector3d offset = 0.4 * screen.linear().col(1);
      command.face.x = face.x() + offset.x();
      command.face.y = face.y() + offset.y();
      command.face.z = face.z() + offset.z();
      return command;
    };
  const auto update_at = [this, &simulated_time, &motion_state](
    const moveit::core::RobotState & state, msg::TrackingTarget command,
    const moveit::core::RobotState * actual_state_override = nullptr)
    {
      simulated_time += parameters_.control_period_sec;
      command.face_stamp.sec = static_cast<std::int32_t>(simulated_time);
      command.face_stamp.nanosec = static_cast<std::uint32_t>(
        (simulated_time - static_cast<double>(command.face_stamp.sec)) * 1.0e9);
      const JointMotionState current = motion_state(state);
      const JointMotionState actual = actual_state_override ?
        motion_state(*actual_state_override) : current;
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(
        state, command, *planning_scene, simulated_time, kSceneRevision,
        actual, simulated_time, current);
    };
  const auto solved_qp = []() {
      QpResult result;
      result.status = QpStatus::kSolved;
      result.achieved_primary_velocity = Eigen::VectorXd::Zero(1);
      return result;
    };
  const auto solved_full_face_qp = []() {
      QpResult result;
      result.status = QpStatus::kSolved;
      result.achieved_primary_velocity = Eigen::VectorXd::Zero(3);
      return result;
    };
  const auto record_aligned_full_face = [this, &solved_full_face_qp](
    const MotionReferenceResult & measured, const double time_sec) {
      ASSERT_TRUE(measured.tracking);
      auto stable = *measured.tracking;
      stable.signed_face_primary_error_rad.setZero();
      stable.face_primary_feedforward_radps.setZero();
      reference_->recordFaceCommand(
        stable, solved_full_face_qp(), time_sec, true, measured.actual_face_alignment);
    };

  MotionReferenceResult first_path;
  const auto first_plan_deadline = std::chrono::steady_clock::now() + 3s;
  do {
    first_path = update_at(start, aligned_face(start));
    if (reference_->diagnostics().plans_accepted >= 1U) {
      break;
    }
    std::this_thread::sleep_for(2ms);
  } while (std::chrono::steady_clock::now() < first_plan_deadline);
  ASSERT_EQ(reference_->diagnostics().compact_generation, 1U);
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U)
    << reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(first_path.follows_path);
  ASSERT_TRUE(first_path.task);
  EXPECT_EQ(first_path.task->primary_matrix.rows(), 6);
  EXPECT_TRUE(first_path.task->primary_matrix.isApprox(Eigen::MatrixXd::Identity(6, 6)));
  ASSERT_GE(
    reference_->diagnostics().planned_path_max_roll_error_rad,
    std::abs(start_upright->signed_roll_error_rad));
  EXPECT_LE(
    reference_->diagnostics().planned_path_max_roll_error_rad,
    kCompactRollCapTestRad + kCompactPitchValidationToleranceRad);
  ASSERT_EQ(first_path.compact_roll_constraints.size(), 2U);
  ASSERT_EQ(first_path.compact_pitch_constraints.size(), 2U);
  const auto joint6 = std::find(
    group_->getVariableNames().begin(), group_->getVariableNames().end(), "joint6");
  ASSERT_NE(joint6, group_->getVariableNames().end());
  const auto joint6_index = std::distance(group_->getVariableNames().begin(), joint6);
  Eigen::VectorXd start_position;
  start.copyJointGroupPositions(group_, start_position);
  auto roll_probe = start;
  roll_probe.setVariablePosition("joint6", roll_probe.getVariablePosition("joint6") + 1.0e-4);
  roll_probe.update(true);
  const auto probe_upright = evaluateCompactUprightPose(
    roll_probe.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(probe_upright);
  const double q6_direction =
    std::abs(probe_upright->signed_roll_error_rad) >
    std::abs(start_upright->signed_roll_error_rad) ? 1.0 : -1.0;
  Eigen::VectorXd beyond_roll_cap = start_position;
  beyond_roll_cap[joint6_index] += q6_direction * 0.06;
  auto beyond_roll_state = start;
  beyond_roll_state.setJointGroupPositions(group_, beyond_roll_cap);
  beyond_roll_state.update(true);
  const auto beyond_roll_pose = evaluateCompactUprightPose(
    beyond_roll_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(beyond_roll_pose);
  ASSERT_GT(std::abs(beyond_roll_pose->signed_roll_error_rad), kCompactRollCapTestRad);
  std::string compact_violation;
  EXPECT_FALSE(reference_->validateCompactPathPosition(
      start, beyond_roll_cap, &compact_violation));
  EXPECT_EQ(compact_violation, "screen roll");

  auto commit_events = reference_->drainCompactEvents();
  const auto trigger = std::find_if(
    commit_events.begin(), commit_events.end(), [](const auto & event) {
      return event.event == "triggered";
    });
  ASSERT_NE(trigger, commit_events.end());
  const auto committed_goal = goalVectorFromDetails(trigger->details_json);
  ASSERT_TRUE(committed_goal);
  std::string endpoint_violation;
  EXPECT_TRUE(reference_->validateCompactPathPosition(start, *committed_goal, &endpoint_violation))
    << endpoint_violation;

  // Advance synthetic feedback in small steps along the follower's checked-path
  // reference. This exercises lifecycle policy without simulating the QP or a
  // physical controller, and respects the follower's bounded projection window.
  moveit::core::RobotState current(start);
  MotionReferenceResult return_step;
  constexpr double kSyntheticFeedbackStepRad{0.005};
  for (int tick = 0; tick < 100; ++tick) {
    return_step = update_at(current, misaligned_face(current));
    if (reference_->diagnostics().compact_phase == "RETURN_FACE") {
      break;
    }
    EXPECT_EQ(reference_->diagnostics().paths_abandoned, 0U)
      << "checked path lost during incremental feedback, tick=" << tick
      << ", state=" << reference_->diagnostics().state
      << ", progress=" << reference_->diagnostics().path_progress_rad
      << ", distance=" << reference_->diagnostics().path_distance_rad;
    ASSERT_TRUE(return_step.task);
    ASSERT_EQ(return_step.task->primary_matrix.rows(), 6);
    ASSERT_TRUE(return_step.task->primary_matrix.isApprox(
        Eigen::MatrixXd::Identity(6, 6)));

    Eigen::VectorXd feedback_step = return_step.task->primary_reference;
    ASSERT_EQ(feedback_step.size(), 6);
    ASSERT_TRUE(feedback_step.allFinite());
    const double largest_step = feedback_step.cwiseAbs().maxCoeff();
    ASSERT_GT(largest_step, 1.0e-9)
      << "follower produced no reference step before lifecycle completion";
    feedback_step *= std::min(1.0, kSyntheticFeedbackStepRad / largest_step);
    Eigen::VectorXd current_position;
    current.copyJointGroupPositions(group_, current_position);
    Eigen::VectorXd next_position = current_position + feedback_step;
    std::string feedback_violation;
    ASSERT_TRUE(reference_->validateCompactPathPosition(
        start, next_position, &feedback_violation))
      << feedback_violation;
    current.setJointGroupPositions(group_, next_position);
    current.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
    current.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
    current.update(true);
    if (reference_->diagnostics().compact_phase == "SETTLING" &&
      (next_position - *committed_goal).lpNorm<Eigen::Infinity>() <= kSyntheticFeedbackStepRad)
    {
      current.setJointGroupPositions(group_, *committed_goal);
      current.update(true);
    }
  }
  ASSERT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE")
    << "state=" << reference_->diagnostics().state
    << ", plan='" << reference_->diagnostics().last_plan_message
    << "', paths_abandoned=" << reference_->diagnostics().paths_abandoned
    << ", path_progress=" << reference_->diagnostics().path_progress_rad
    << ", path_distance=" << reference_->diagnostics().path_distance_rad;
  EXPECT_EQ(reference_->diagnostics().paths_completed, 1U);
  EXPECT_EQ(reference_->diagnostics().paths_abandoned, 0U);
  EXPECT_TRUE(return_step.compact_pitch_constraints.empty());
  EXPECT_EQ(return_step.compact_roll_constraints.size(), 2U);
  EXPECT_FALSE(return_step.compact_radius_constraints.empty());
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_pitch_cap_rad));
  auto actual_misaligned = current;
  actual_misaligned.setVariablePosition(
    "joint1", actual_misaligned.getVariablePosition("joint1") + 0.25);
  actual_misaligned.update(true);
  const auto command_aligned_actual_misaligned = update_at(
    current, aligned_face(current), &actual_misaligned);
  ASSERT_TRUE(command_aligned_actual_misaligned.tracking);
  ASSERT_TRUE(command_aligned_actual_misaligned.task);
  EXPECT_EQ(command_aligned_actual_misaligned.task->primary_matrix.rows(), 1);
  EXPECT_LE(command_aligned_actual_misaligned.tracking->pointing_error_rad,
    kFaceTrackingErrorGuardRad);
  ASSERT_TRUE(command_aligned_actual_misaligned.actual_face_alignment);
  EXPECT_GT(command_aligned_actual_misaligned.actual_face_alignment->pointing_error_rad,
    kFaceTrackingErrorGuardRad);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);
  reference_->recordFaceCommand(
    *command_aligned_actual_misaligned.tracking, solved_full_face_qp(), simulated_time,
    true, command_aligned_actual_misaligned.actual_face_alignment);
  EXPECT_FALSE(reference_->diagnostics().compact_return_face_qualified);

  const auto first_goal = update_at(current, misaligned_face(current));
  ASSERT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  ASSERT_TRUE(reference_->compactManeuverActive());
  EXPECT_TRUE(reference_->diagnostics().compact_actual_settled);
  EXPECT_TRUE(std::isfinite(reference_->diagnostics().compact_effective_radius_cap_m));
  ASSERT_TRUE(first_goal.tracking);
  ASSERT_TRUE(first_goal.task);
  EXPECT_EQ(first_goal.task->primary_matrix.rows(), 1);
  EXPECT_EQ(first_goal.task->secondary_matrix.rows(), 2);
  EXPECT_TRUE(first_goal.task->primary_matrix.isApprox(
      first_goal.tracking->task.primary_matrix.row(2), 1.0e-12));
  EXPECT_GT(first_goal.tracking->pointing_error_rad, kFaceTrackingErrorGuardRad);
  EXPECT_LE(first_goal.tracking->roll_error_rad, kFaceTrackingErrorGuardRad);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);

  Eigen::MatrixXd screen_jacobian;
  ASSERT_TRUE(current.getJacobian(
      group_, model_->getLinkModel("monitor_control_frame"), Eigen::Vector3d::Zero(),
      screen_jacobian, false));
  const auto * root_link = group_->getJointModels().front()->getParentLinkModel();
  if (root_link) {
    const Eigen::Matrix3d root_rotation =
      current.getGlobalLinkTransform(root_link).linear();
    screen_jacobian.topRows(3) = (root_rotation * screen_jacobian.topRows(3)).eval();
    screen_jacobian.bottomRows(3) =
      (root_rotation * screen_jacobian.bottomRows(3)).eval();
  }
  TrackingVelocityTaskConfig upright_config;
  upright_config.orientation_gain = parameters_.orientation_gain;
  upright_config.maximum_angular_reference_radps =
    parameters_.maximum_angular_reference_radps;
  const auto gravity_upright = makeUprightRollPathTask(
    current.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::VectorXd::Zero(6), upright_config);
  ASSERT_TRUE(gravity_upright);
  const auto * gaze_link = model_->getLinkModel(parameters_.gaze_frame);
  ASSERT_NE(gaze_link, nullptr);
  GazeKinematics gaze;
  gaze.pose = current.getGlobalLinkTransform(gaze_link);
  gaze.pose.linear() *= face_tracking_arm::opticalToPointingRotation();
  ASSERT_TRUE(current.getJacobian(
      group_, gaze_link, Eigen::Vector3d::Zero(), gaze.jacobian, false));
  if (root_link) {
    const Eigen::Matrix3d root_rotation =
      current.getGlobalLinkTransform(root_link).linear();
    gaze.jacobian.topRows(3) = (root_rotation * gaze.jacobian.topRows(3)).eval();
    gaze.jacobian.bottomRows(3) =
      (root_rotation * gaze.jacobian.bottomRows(3)).eval();
  }
  const auto first_goal_command = misaligned_face(current);
  const auto expected_return_pointing = makeFaceReturnPointingTask(
    current.getGlobalLinkTransform("monitor_control_frame"), screen_jacobian,
    Eigen::Vector3d(first_goal_command.face.x, first_goal_command.face.y,
      first_goal_command.face.z), Eigen::Vector3d::Zero(), upright_config,
    first_goal.tracking->task.primary_matrix.topRows(2),
    first_goal.tracking->task.primary_reference.head(2), gaze);
  ASSERT_TRUE(expected_return_pointing);
  EXPECT_FALSE(expected_return_pointing->used_normal_fallback);
  EXPECT_TRUE(first_goal.task->secondary_matrix.isApprox(
      expected_return_pointing->matrix, 1.0e-12));
  EXPECT_TRUE(first_goal.task->primary_matrix.row(0).isApprox(
      gravity_upright->task.primary_matrix.row(0), 1.0e-12));
  EXPECT_NEAR(first_goal.tracking->roll_error_rad,
    std::abs(gravity_upright->roll_error_rad), 1.0e-12);
  EXPECT_TRUE(first_goal.tracking->task.primary_matrix.row(2).isApprox(
      gravity_upright->task.primary_matrix.row(0), 1.0e-12));

  auto return_pitch_state = current;
  return_pitch_state.setVariablePosition(
    "joint5", return_pitch_state.getVariablePosition("joint5") + 0.70);
  return_pitch_state.setJointGroupVelocities(group_, Eigen::VectorXd::Zero(6));
  return_pitch_state.setJointGroupAccelerations(group_, Eigen::VectorXd::Zero(6));
  return_pitch_state.update(true);
  const auto return_pitch_pose = evaluateCompactUprightPose(
    return_pitch_state.getGlobalLinkTransform("monitor_control_frame").linear());
  ASSERT_TRUE(return_pitch_pose);
  const double return_abs_pitch = std::abs(return_pitch_pose->signed_pitch_rad);
  const double committed_pitch_cap = reference_->diagnostics().compact_pitch_cap_rad;
  const double committed_radius_cap = reference_->diagnostics().compact_entry_radius_cap_m;
  ASSERT_GT(return_abs_pitch, committed_pitch_cap + 0.05);
  Eigen::VectorXd return_position;
  return_pitch_state.copyJointGroupPositions(group_, return_position);
  std::string return_radius_violation;
  ASSERT_TRUE(reference_->validateCompactPathPosition(
      return_pitch_state, return_position, &return_radius_violation)) <<
    return_radius_violation;

  const auto return_before_rearm = update_at(
    return_pitch_state, misaligned_face(return_pitch_state));
  ASSERT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  ASSERT_TRUE(return_before_rearm.tracking);
  ASSERT_TRUE(return_before_rearm.task);
  EXPECT_TRUE(return_before_rearm.compact_pitch_constraints.empty());
  EXPECT_EQ(return_before_rearm.compact_roll_constraints.size(), 2U);
  EXPECT_FALSE(return_before_rearm.compact_radius_constraints.empty());
  const std::uint64_t plans_before_return_rearm =
    reference_->diagnostics().plans_requested;

  reference_->reset(true);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_goal_reached, 1.0);
  EXPECT_FALSE(reference_->diagnostics().compact_actual_settled);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_return_face_dwell_sec, 0.0);
  EXPECT_DOUBLE_EQ(
    reference_->diagnostics().compact_pitch_cap_rad, committed_pitch_cap);
  EXPECT_DOUBLE_EQ(
    reference_->diagnostics().compact_entry_radius_cap_m, committed_radius_cap);
  const auto return_after_rearm = update_at(
    return_pitch_state, misaligned_face(return_pitch_state));
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, plans_before_return_rearm);
  EXPECT_TRUE(reference_->compactManeuverActive());
  ASSERT_TRUE(return_after_rearm.tracking);
  ASSERT_TRUE(return_after_rearm.task);
  EXPECT_EQ(return_after_rearm.task->primary_matrix.rows(), 1);
  EXPECT_TRUE(return_after_rearm.compact_pitch_constraints.empty());
  EXPECT_EQ(return_after_rearm.compact_roll_constraints.size(), 2U);
  EXPECT_FALSE(return_after_rearm.compact_radius_constraints.empty());

  reference_->recordFaceCommand(
    *first_goal.tracking, solved_qp(), simulated_time, true,
    first_goal.actual_face_alignment);
  auto lost_reacquisition = *first_goal.tracking;
  lost_reacquisition.pointing_error_rad = kFaceTrackingErrorGuardRad + 0.01;
  reference_->recordFaceCommand(
    lost_reacquisition, solved_qp(), simulated_time, true,
    first_goal.actual_face_alignment);
  const auto first_aligned_after_gap = update_at(current, aligned_face(current));
  ASSERT_TRUE(first_aligned_after_gap.tracking);
  ASSERT_TRUE(first_aligned_after_gap.task);
  EXPECT_EQ(first_aligned_after_gap.task->primary_matrix.rows(), 3);
  EXPECT_TRUE(reference_->diagnostics().compact_return_full_face_active);
  record_aligned_full_face(first_aligned_after_gap, simulated_time);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  for (int tick = 0; tick < 5; ++tick) {
    const auto returned_face = update_at(current, aligned_face(current));
    ASSERT_TRUE(returned_face.tracking);
    ASSERT_TRUE(returned_face.task);
    EXPECT_EQ(returned_face.task->primary_matrix.rows(), 3);
    EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
    record_aligned_full_face(returned_face, simulated_time);
  }
  EXPECT_NEAR(
    reference_->diagnostics().compact_return_face_dwell_sec,
    5.0 * parameters_.control_period_sec, 1.0e-9);

  const auto failed_publication = update_at(current, aligned_face(current));
  ASSERT_TRUE(failed_publication.tracking);
  reference_->recordFaceCommand(
    *failed_publication.tracking, solved_full_face_qp(), simulated_time, false,
    failed_publication.actual_face_alignment);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_return_face_dwell_sec, 0.0);

  const auto forecast_crossing = update_at(current, aligned_face(current));
  ASSERT_TRUE(forecast_crossing.tracking);
  auto crossing_tracking = *forecast_crossing.tracking;
  crossing_tracking.pointing_error_rad = 0.06;
  crossing_tracking.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
  crossing_tracking.face_primary_feedforward_radps = Eigen::Vector3d(0.2, 0.0, 0.0);
  reference_->recordFaceCommand(
    crossing_tracking, solved_full_face_qp(), simulated_time, true,
    forecast_crossing.actual_face_alignment);
  EXPECT_FALSE(reference_->diagnostics().compact_return_face_qualified);
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_return_face_dwell_sec, 0.0);

  record_aligned_full_face(forecast_crossing, simulated_time);
  const int dwell_samples = static_cast<int>(std::ceil(
      0.30 / parameters_.control_period_sec));
  for (int tick = 0; tick < dwell_samples - 1; ++tick) {
    const auto returned_face = update_at(current, aligned_face(current));
    ASSERT_TRUE(returned_face.tracking);
    ASSERT_TRUE(returned_face.task);
    EXPECT_EQ(returned_face.task->primary_matrix.rows(), 3);
    EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
    record_aligned_full_face(returned_face, simulated_time);
  }
  EXPECT_TRUE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_phase, "RETURN_FACE");
  EXPECT_NEAR(
    reference_->diagnostics().compact_return_face_dwell_sec,
    0.30 - parameters_.control_period_sec, 1.0e-9);
  const auto settled_face = update_at(current, aligned_face(current));
  ASSERT_TRUE(settled_face.tracking);
  record_aligned_full_face(settled_face, simulated_time);
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_phase, "COMPLETE");
  EXPECT_DOUBLE_EQ(reference_->diagnostics().compact_goal_reached, 0.0);
  EXPECT_FALSE(reference_->diagnostics().compact_return_full_face_active);
  EXPECT_FALSE(reference_->diagnostics().compact_radius_floor_latched);
  EXPECT_TRUE(std::isnan(reference_->diagnostics().compact_effective_radius_cap_m));

  const auto face_again = update_at(current, misaligned_face(current));
  ASSERT_TRUE(face_again.tracking);
  EXPECT_FALSE(face_again.j1_hold_position_rad);
  EXPECT_EQ(reference_->diagnostics().j1_hold_direction, 0);
  auto shortfall = *face_again.tracking;
  shortfall.has_face_primary_roll = true;
  shortfall.signed_face_primary_error_rad = Eigen::Vector3d(0.06, 0.0, 0.0);
  shortfall.face_primary_feedforward_radps.setZero();
  QpResult shortfall_qp;
  shortfall_qp.status = QpStatus::kSolved;
  shortfall_qp.achieved_primary_velocity = Eigen::Vector3d(-0.20, 0.0, 0.0);
  reference_->recordFaceCommand(shortfall, shortfall_qp, simulated_time, true);
  for (int tick = 0; tick < 8; ++tick) {
    const auto shortfall_sample = update_at(current, misaligned_face(current));
    ASSERT_TRUE(shortfall_sample.tracking);
    reference_->recordFaceCommand(
      shortfall, shortfall_qp,
      simulated_time, true);
  }
  EXPECT_FALSE(reference_->compactManeuverActive());
  EXPECT_EQ(reference_->diagnostics().compact_generation, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_requested, plan_requests_before_lifecycle);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  EXPECT_EQ(reference_->diagnostics().compact_phase, "COMPLETE");
  EXPECT_EQ(reference_->diagnostics().paths_completed, 1U);
  EXPECT_EQ(reference_->diagnostics().compact_generation, committed_generation);

  const auto events = reference_->drainCompactEvents();
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "completed" && event.generation_id == 1U;
  }));
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "compact_goal_settled" && event.generation_id == 1U;
  }));
  EXPECT_TRUE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "face_return_full_face_started" && event.generation_id == 1U;
  }));
  EXPECT_FALSE(std::any_of(events.begin(), events.end(), [](const CompactEvent & event) {
      return event.event == "pair_resource_exhausted" && event.generation_id == 2U;
  }));
}

TEST_F(MotionReferenceTest, NamedRestPostureHoldsWithoutPlanning)
{
  const auto robot_state = state();
  const auto result = update(robot_state, target(msg::TrackingTarget::REST));
  EXPECT_FALSE(result.task);
  EXPECT_EQ(reference_->diagnostics().state, "REST_HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
}

TEST_F(MotionReferenceTest, SearchIsPreemptedByHold)
{
  const auto robot_state = state(0.4);
  const auto search = update(robot_state, target(msg::TrackingTarget::SEARCH));
  EXPECT_FALSE(search.task);
  auto held = update(robot_state, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(held.task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
}

TEST_F(MotionReferenceTest, PreferredBaseAngleUsesJointRootFrame)
{
  Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
  base_pose.linear() = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d face = base_pose.translation() +
    base_pose.linear() * Eigen::Vector3d(0.0, 1.0, 0.0);
  EXPECT_NEAR(preferredBaseJointAngle(face, base_pose), 0.5 * M_PI, 1.0e-12);
}

TEST_F(MotionReferenceSearchDisabledTest, ExternalSearchIsRejectedWithDiagnostic)
{
  const auto result = update(state(), target(msg::TrackingTarget::SEARCH));
  EXPECT_FALSE(result.task);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_UNAVAILABLE");
}

TEST_F(MotionReferenceTest, RealClearancePredicateAllowsRestPlanningAndPathAdoption)
{
  const auto start = state(0.4);
  Eigen::VectorXd position;
  start.copyJointGroupPositions(group_, position);
  const JointMotionState motion{
    position, Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6)};
  {
    planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
    const planning_scene::PlanningSceneConstPtr planning_scene = scene;
    const auto validity = CollisionConstraintBuilder(collision_config_).validateSegment(
      *planning_scene, start, *group_, motion, motion, 2);
    ASSERT_TRUE(validity.input_valid) << validity.failure_reason;
    ASSERT_FALSE(validity.unsafe) << validity.failure_reason;
  }
  const auto result = waitForRestPath(start);
  ASSERT_TRUE(result) << reference_->diagnostics().state << ": " <<
    reference_->diagnostics().last_plan_message;
  ASSERT_TRUE(result->task);
  EXPECT_TRUE(result->follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "REST_PATH");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_accepted, 1U);
  EXPECT_EQ(reference_->diagnostics().plans_rejected, 0U);
  EXPECT_EQ(result->task->primary_matrix.rows(), 1);
  ASSERT_EQ(result->task->secondary_matrix.rows(), 6);
  EXPECT_GT(result->task->secondary_reference.norm(), 0.0);
  EXPECT_LE(result->task->secondary_reference.cwiseAbs().maxCoeff(), 0.70 + 1.0e-9);
  EXPECT_DOUBLE_EQ(start.getVariablePosition("joint1"), 0.4);
}

TEST_F(MotionReferenceTest, HoldCancelsPendingRestWithoutAdoptingItsLateResult)
{
  expectPendingRestPreempted(msg::TrackingTarget::HOLD);
}

TEST_F(MotionReferenceTest, FaceCancelsPendingRestWithoutAdoptingItsLateResult)
{
  expectPendingRestPreempted(msg::TrackingTarget::FACE);
}

TEST_F(MotionReferenceTest, HoldImmediatelyPreemptsAnAdoptedRestPath)
{
  const auto start = state(0.4);
  ASSERT_TRUE(waitForRestPath(start)) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  const auto result = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(result.task);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}

TEST_F(MotionReferenceTest, FaceImmediatelyPreemptsAnAdoptedRestPath)
{
  const auto start = state(0.4);
  ASSERT_TRUE(waitForRestPath(start)) << reference_->diagnostics().last_plan_message;
  ASSERT_EQ(reference_->diagnostics().plans_accepted, 1U);
  const auto result = update(start, target(msg::TrackingTarget::FACE));
  EXPECT_TRUE(result.tracking);
  EXPECT_FALSE(result.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 1U);
}


TEST_F(MotionReferenceTest, SearchUsesTheSameExecutorAndIsPreemptedByHoldAndFace)
{
  auto start = state();
  ASSERT_TRUE(start.setToDefaultValues(group_, "search"));
  start.update();
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "sweep";
  const auto scanned = update(start, search);
  ASSERT_TRUE(scanned.task);
  EXPECT_TRUE(scanned.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_SWEEP");
  EXPECT_EQ(scanned.task->primary_matrix.rows(), 1);
  ASSERT_EQ(scanned.task->secondary_matrix.rows(), 6);
  EXPECT_LE(scanned.task->secondary_reference.norm(), .30001);
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  velocity[0] = .2;
  start.setJointGroupVelocities(group_, velocity);
  const auto held = update(start, target(msg::TrackingTarget::HOLD));
  EXPECT_FALSE(held.task);
  EXPECT_EQ(reference_->diagnostics().state, "HOLD");
  EXPECT_DOUBLE_EQ(start.getVariableVelocity("joint1"), .2);
  auto face = target(msg::TrackingTarget::FACE);
  face.face.x = 2.0;
  const auto tracked = update(start, face);
  EXPECT_FALSE(tracked.follows_path);
  EXPECT_EQ(reference_->diagnostics().state, "FACE");
  ASSERT_TRUE(tracked.tracking);
  EXPECT_EQ(tracked.tracking->task.primary_matrix.rows(), 3);
  EXPECT_DOUBLE_EQ(start.getVariableVelocity("joint1"), .2);
}

TEST_F(MotionReferenceTest, NearbySelectorIdentitySwitchClearsFaceVelocityHistory)
{
  const auto start = state();
  const auto & tcp = start.getGlobalLinkTransform("monitor_control_frame");
  auto command = target(msg::TrackingTarget::FACE);
  command.target_id = 17;
  command.pose.position.x = tcp.translation().x();
  command.pose.position.y = tcp.translation().y();
  command.pose.position.z = tcp.translation().z();
  command.pose.orientation.w = 1.0;
  command.face.x = tcp.translation().x() + 0.8;
  command.face.y = tcp.translation().y();
  command.face.z = tcp.translation().z();
  double simulated_time = 1.0;
  const auto update_at = [this, &simulated_time](
    const moveit::core::RobotState & state, const msg::TrackingTarget & target)
    {
      planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
      const planning_scene::PlanningSceneConstPtr planning_scene = scene;
      return reference_->update(state, target, *planning_scene, simulated_time, kSceneRevision);
    };
  command.face_stamp = rclcpp::Time(1'000'000'000);
  (void)update_at(start, command);
  simulated_time += 0.02;
  command.face_stamp = rclcpp::Time(1'020'000'000);
  command.face.y += 0.01;
  const auto moving_target = update_at(start, command);
  ASSERT_TRUE(moving_target.tracking);
  const auto inherited_feedforward = moving_target.tracking->task.primary_reference;
  ASSERT_GT(inherited_feedforward.norm(), 0.0);

  command.target_id = 18;
  simulated_time += 0.02;
  command.face_stamp = rclcpp::Time(1'040'000'000);
  const auto switched_target = update_at(start, command);
  ASSERT_TRUE(switched_target.tracking);

  MotionReference fresh_reference(
    node_, monitor_, parameters_, limits_, collision_config_);
  planning_scene_monitor::LockedPlanningSceneRO scene(monitor_);
  const planning_scene::PlanningSceneConstPtr planning_scene = scene;
  const auto fresh_target = fresh_reference.update(
    start, command, *planning_scene, simulated_time, kSceneRevision);
  ASSERT_TRUE(fresh_target.tracking);
  EXPECT_GT((inherited_feedforward - fresh_target.tracking->task.primary_reference).norm(), 1.0e-4);
  EXPECT_TRUE(switched_target.tracking->task.primary_reference.isApprox(
      fresh_target.tracking->task.primary_reference, 1.0e-12));
}

TEST_F(MotionReferenceTest, LocalSearchKeepsTheLastPostureAndStartsWithoutPlanning)
{
  auto start = state(.7);
  auto search = target(msg::TrackingTarget::SEARCH);
  search.search_pattern = "local_then_sweep";
  const auto scanned = update(start, search);
  ASSERT_TRUE(scanned.task);
  EXPECT_EQ(reference_->diagnostics().state, "SEARCH_LOCAL");
  EXPECT_EQ(reference_->diagnostics().plans_requested, 0U);
  EXPECT_EQ(scanned.task->primary_matrix.rows(), 1);
  ASSERT_EQ(scanned.task->secondary_matrix.rows(), 6);
  EXPECT_LT(scanned.task->secondary_reference.tail(5).norm(), 1e-9);
}

}  // namespace
}  // namespace face_tracking_arm::control
