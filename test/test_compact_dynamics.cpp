// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <srdfdom/srdfdom/model.h>
#include <urdf/urdf/model.h>

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/robot_model.hpp>

#include "face_tracking_arm/compact_dynamics.hpp"

namespace face_tracking_arm::control
{
namespace
{
constexpr double kPeriod = 0.01;
constexpr double kPi = 3.14159265358979323846;

std::string readFile(const std::string & path)
{
  std::ifstream input(path);
  if (!input) {throw std::runtime_error("cannot read model file: " + path);}
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

struct LoadedModel
{
  std::shared_ptr<urdf::Model> urdf;
  std::shared_ptr<srdf::Model> srdf;
  moveit::core::RobotModelConstPtr model;
};

LoadedModel parseModel(
  std::string urdf_text, const std::string & srdf_text,
  const double joint2_effort_override = -1.0)
{
  LoadedModel loaded;
  loaded.urdf = std::make_shared<urdf::Model>();
  if (!loaded.urdf->initString(urdf_text)) {throw std::runtime_error("URDF parse failed");}
  if (joint2_effort_override > 0.0) {
    const auto joint = loaded.urdf->getJoint("joint2");
    if (!joint || !joint->limits) {throw std::runtime_error("joint2 effort missing");}
    joint->limits->effort = joint2_effort_override;
  }
  loaded.srdf = std::make_shared<srdf::Model>();
  if (!loaded.srdf->initString(*loaded.urdf, srdf_text)) {
    throw std::runtime_error("SRDF parse failed");
  }
  loaded.model = std::make_shared<moveit::core::RobotModel>(loaded.urdf, loaded.srdf);
  return loaded;
}

LoadedModel loadXarm(const double joint2_effort_override = -1.0)
{
  return parseModel(readFile(TEST_ROBOT_URDF_FILE), readFile(TEST_ROBOT_SRDF_FILE),
    joint2_effort_override);
}

std::string handUrdf(
  const double joint2_effort = 30.0, const double damping = 0.1,
  const double friction = 0.5, const double world_pitch = 0.0, const bool add_payload = true)
{
  std::ostringstream xml;
  xml << "<robot name='hand_dynamics'>"
      << "<link name='world'/><link name='base_link'/>"
      << "<joint name='world_fixed' type='fixed'><parent link='world'/><child link='base_link'/>"
      << "<origin xyz='0 0 0' rpy='0 " << world_pitch << " 0'/></joint>"
      << "<link name='link1'/><link name='link2'/>"
      << "<joint name='z_joint' type='revolute'><parent link='base_link'/><child link='link1'/>"
      << "<origin xyz='0 0 0' rpy='0 0 0'/><axis xyz='0 0 1'/>"
      << "<limit lower='-2' upper='2' effort='100' velocity='3.1'/>"
      << "<dynamics damping='0' friction='0'/></joint>"
      << "<joint name='a_joint' type='revolute'><parent link='link1'/><child link='link2'/>"
      << "<origin xyz='0 0 0' rpy='0 0 0'/><axis xyz='0 1 0'/>"
      << "<limit lower='-2' upper='2' effort='" << joint2_effort << "' velocity='3.1'/>"
      << "<dynamics damping='" << damping << "' friction='" << friction << "'/></joint>";
  if (add_payload) {
    xml << "<link name='fixed_payload'><inertial><origin xyz='0 0 0' rpy='0 0 0'/>"
        << "<mass value='2'/><inertia ixx='0.001' ixy='0' ixz='0' iyy='0.001' "
        << "iyz='0' izz='0.001'/></inertial></link>"
        <<
      "<joint name='payload_mount' type='fixed'><parent link='link2'/><child link='fixed_payload'/>"
        << "<origin xyz='1 0 0' rpy='0 0 0'/></joint>";
  }
  xml << "</robot>";
  return xml.str();
}

const moveit::core::JointModelGroup & groupOf(const LoadedModel & loaded, const std::string & name)
{
  const auto * group = loaded.model->getJointModelGroup(name);
  if (!group) {throw std::runtime_error("missing test group: " + name);}
  return *group;
}

JointMotionLimits makeLimits(
  const moveit::core::RobotModelConstPtr & model,
  const std::string & group_name, const double max_velocity = 3.1,
  const double max_acceleration = 19.0, const double max_jerk = 400.0)
{
  const auto * group = model->getJointModelGroup(group_name);
  if (!group) {throw std::runtime_error("missing group while creating limits");}
  const auto count = static_cast<Eigen::Index>(group->getVariableCount());
  JointMotionLimits limits;
  limits.lower_position.resize(count); limits.upper_position.resize(count);
  for (Eigen::Index i = 0; i < count; ++i) {
    const auto & bounds =
      model->getVariableBounds(group->getVariableNames()[static_cast<size_t>(i)]);
    limits.lower_position[i] = bounds.min_position_;
    limits.upper_position[i] = bounds.max_position_;
  }
  limits.position_margin = Eigen::VectorXd::Zero(count);
  limits.max_velocity = Eigen::VectorXd::Constant(count, max_velocity);
  limits.max_acceleration = Eigen::VectorXd::Constant(count, max_acceleration);
  limits.max_jerk = Eigen::VectorXd::Constant(count, max_jerk);
  return limits;
}

std::optional<CompactJointProfile> makeLineProfile(
  const Eigen::VectorXd & start_position, const Eigen::VectorXd & direction,
  const JointMotionLimits & limits, const double length, const double speed_scale = 1.0)
{
  JointMotionState start;
  start.position = start_position;
  start.velocity = Eigen::VectorXd::Zero(start_position.size());
  start.acceleration = Eigen::VectorXd::Zero(start_position.size());
  return buildCompactJointProfile({start_position, start_position + length * direction},
    start, limits, kPeriod, speed_scale);
}

Eigen::VectorXd recordedQ()
{
  Eigen::VectorXd q(6);
  q << 6.155917197765239, -1.4147758378385782, -2.050348879011357,
    -2.4446028242311018, 0.8225835196011687, -0.262780043746944;
  return q;
}

Eigen::VectorXd recordedDirection()
{
  Eigen::VectorXd direction(6);
  direction << -0.52737626913677, 1.9237499988999707, -1.3793091109898903,
    0.0, 1.25909531013495, 0.4422072403702737;
  return direction / std::abs(direction[1]);
}

Eigen::VectorXd handPosition(
  const moveit::core::JointModelGroup & group,
  const double a_joint_value)
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(group.getVariableCount()));
  for (Eigen::Index i = 0; i < q.size(); ++i) {
    if (group.getVariableNames()[static_cast<size_t>(i)] == "a_joint") {q[i] = a_joint_value;}
  }
  return q;
}

Eigen::VectorXd handDirection(
  const moveit::core::JointModelGroup & group,
  const double a_joint_direction)
{
  Eigen::VectorXd d = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(group.getVariableCount()));
  for (Eigen::Index i = 0; i < d.size(); ++i) {
    if (group.getVariableNames()[static_cast<size_t>(i)] == "a_joint") {d[i] = a_joint_direction;}
  }
  return d;
}

std::size_t variableIndex(const moveit::core::JointModelGroup & group, const std::string & name)
{
  const auto & names = group.getVariableNames();
  const auto found = std::find(names.begin(), names.end(), name);
  if (found == names.end()) {throw std::runtime_error("missing group variable " + name);}
  return static_cast<std::size_t>(std::distance(names.begin(), found));
}

CompactJointProfile handConstantVelocity(
  const moveit::core::JointModelGroup & group, const double velocity, const double period)
{
  const auto q0 = handPosition(group, 0.0);
  const auto d = handDirection(group, 1.0);
  CompactJointProfile profile;
  for (const double delta : {0.0, velocity * period}) {
    JointMotionState state;
    state.position = q0 + delta * d;
    state.velocity = velocity * d;
    state.acceleration = Eigen::VectorXd::Zero(q0.size());
    profile.samples.push_back(std::move(state));
  }
  profile.checked_path_vertices = {q0, q0 + velocity * period * d};
  profile.checked_path_segment_scales = {1.0};
  profile.path_progress_rad_per_sample = {0.0, std::abs(velocity * period)};
  profile.speed_scale_per_interval = {1.0};
  profile.leg_end_sample_indices = {1U};
  profile.duration_sec = period;
  profile.path_length_rad = std::abs(velocity * period);
  return profile;
}

void expectSameState(const JointMotionState & lhs, const JointMotionState & rhs)
{
  ASSERT_EQ(lhs.position.size(), rhs.position.size());
  ASSERT_EQ(lhs.velocity.size(), rhs.velocity.size());
  ASSERT_EQ(lhs.acceleration.size(), rhs.acceleration.size());
  for (Eigen::Index i = 0; i < lhs.position.size(); ++i) {
    EXPECT_DOUBLE_EQ(lhs.position[i], rhs.position[i]);
    EXPECT_DOUBLE_EQ(lhs.velocity[i], rhs.velocity[i]);
    EXPECT_DOUBLE_EQ(lhs.acceleration[i], rhs.acceleration[i]);
  }
}

void expectFitsUrdfEffort(
  const CompactDynamicsResult & result, const LoadedModel & loaded,
  const moveit::core::JointModelGroup & group, const double tolerance = 1.0e-5)
{
  ASSERT_EQ(result.peak_required_effort_nm_after.size(), group.getVariableCount());
  for (size_t i = 0; i < group.getVariableCount(); ++i) {
    const std::string joint_name = group.getVariableNames()[i];
    const auto joint = loaded.urdf->getJoint(joint_name);
    ASSERT_TRUE(joint && joint->limits) << joint_name;
    EXPECT_LE(result.peak_required_effort_nm_after[i],
          joint->limits->effort + tolerance) << joint_name;
  }
}

void expectProfileContract(
  const CompactJointProfile & profile, const CompactJointProfile & input,
  const JointMotionLimits & limits, const double period)
{
  ASSERT_EQ(profile.checked_path_vertices.size(), input.checked_path_vertices.size());
  for (size_t vertex = 0; vertex < input.checked_path_vertices.size(); ++vertex) {
    EXPECT_TRUE(profile.checked_path_vertices[vertex].isApprox(input.checked_path_vertices[vertex],
          0.0));
  }
  EXPECT_EQ(profile.checked_path_segment_scales, input.checked_path_segment_scales);
  ASSERT_EQ(profile.path_progress_rad_per_sample.size(), profile.samples.size());
  ASSERT_EQ(profile.speed_scale_per_interval.size() + 1U, profile.samples.size());
  for (size_t i = 1; i < profile.samples.size(); ++i) {
    const auto & previous = profile.samples[i - 1];
    const auto & current = profile.samples[i];
    EXPECT_TRUE((current.position - previous.position - current.velocity * period).norm() < 1.0e-8);
    const Eigen::VectorXd expected_acceleration = (current.velocity - previous.velocity) / period;
    EXPECT_TRUE((current.acceleration - expected_acceleration).norm() < 1.0e-7);
    const Eigen::VectorXd jerk = (current.acceleration - previous.acceleration) / period;
    EXPECT_LE((current.velocity.cwiseAbs().array() - limits.max_velocity.array()).maxCoeff(),
          1.0e-5);
    EXPECT_LE((current.acceleration.cwiseAbs().array() -
      limits.max_acceleration.array()).maxCoeff(), 1.0e-4);
    EXPECT_LE((jerk.cwiseAbs().array() - limits.max_jerk.array()).maxCoeff(), 1.0e-2);
    const Eigen::VectorXd segment = input.checked_path_vertices.back() -
      input.checked_path_vertices.front();
    if (segment.squaredNorm() > 0.0) {
      const double projection = (current.position -
        input.checked_path_vertices.front()).dot(segment) /
        segment.squaredNorm();
      EXPECT_NEAR((current.position - input.checked_path_vertices.front() - projection *
        segment).norm(),
        0.0, 1.0e-7);
    }
  }
}

TEST(CompactDynamics, RecordedForwardFixtureRetimesAllJointsAndKeepsLinearPath)
{
  const auto loaded = loadXarm();
  const auto & group = groupOf(loaded, "xarm6");
  const auto limits = makeLimits(loaded.model, "xarm6");
  const auto old = makeLineProfile(recordedQ(), recordedDirection(), limits, 0.34);
  ASSERT_TRUE(old);
  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*old, limits, kPeriod, 1.0);
  ASSERT_EQ(result.status, CompactDynamicsStatus::kRetimed) << result.failure_reason;
  ASSERT_TRUE(result.profile);
  ASSERT_LT(result.applied_speed_scale, 1.0);
  ASSERT_GT(result.profile->duration_sec, old->duration_sec);
  ASSERT_EQ(result.peak_required_effort_nm_before.size(), group.getVariableCount());
  EXPECT_GT(result.peak_required_effort_nm_before[1], 50.0);
  EXPECT_GT(result.maximum_effort_ratio_before, 1.0);
  EXPECT_LE(result.maximum_effort_ratio_after, 1.0);
  EXPECT_GE(result.checked_interpolation_samples,
    old->samples.size() + result.profile->samples.size() - 2U);
  EXPECT_GE(result.checked_profile_samples, old->samples.size() + result.profile->samples.size());
  expectFitsUrdfEffort(result, loaded, group);
  expectProfileContract(*result.profile, *old, limits, kPeriod);
  EXPECT_EQ(result.profile->checked_path_vertices.size(), 2U);
}

TEST(CompactDynamics, SafeRecordedProfilePassesWithoutChangingAnySamples)
{
  const auto loaded = loadXarm();
  const auto & group = groupOf(loaded, "xarm6");
  const auto limits = makeLimits(loaded.model, "xarm6");
  const auto safe = makeLineProfile(recordedQ(), recordedDirection(), limits, 0.34, 0.5);
  ASSERT_TRUE(safe);
  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*safe, limits, kPeriod, 0.5);
  ASSERT_EQ(result.status, CompactDynamicsStatus::kPass) << result.failure_reason;
  ASSERT_TRUE(result.profile);
  ASSERT_EQ(result.profile->samples.size(), safe->samples.size());
  for (size_t i = 0; i < safe->samples.size(); ++i) {
    expectSameState(result.profile->samples[i], safe->samples[i]);
  }
  EXPECT_DOUBLE_EQ(result.profile->duration_sec, safe->duration_sec);
  EXPECT_DOUBLE_EQ(result.applied_speed_scale, 0.5);
  expectFitsUrdfEffort(result, loaded, group);
}

TEST(CompactDynamics, ReverseRecordedDirectionRemainsWithinAllJointEfforts)
{
  const auto loaded = loadXarm();
  const auto & group = groupOf(loaded, "xarm6");
  const auto limits = makeLimits(loaded.model, "xarm6");
  const auto reverse = makeLineProfile(recordedQ(), -recordedDirection(), limits, 0.34);
  ASSERT_TRUE(reverse);
  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*reverse, limits, kPeriod, 1.0);
  ASSERT_EQ(result.status, CompactDynamicsStatus::kPass) << result.failure_reason;
  ASSERT_TRUE(result.profile);
  expectFitsUrdfEffort(result, loaded, group);
  ASSERT_EQ(result.profile->samples.size(), reverse->samples.size());
  for (size_t i = 0; i < reverse->samples.size(); ++i) {
    expectSameState(result.profile->samples[i], reverse->samples[i]);
  }
}

TEST(CompactDynamics, PublishedPrefixIsPreservedDuringSuffixRetiming)
{
  const auto loaded = loadXarm();
  const auto & group = groupOf(loaded, "xarm6");
  const auto limits = makeLimits(loaded.model, "xarm6");
  auto input = makeLineProfile(recordedQ(), recordedDirection(), limits, 0.34);
  ASSERT_TRUE(input);
  JointMotionState first = input->samples.front();
  std::string failure;
  ASSERT_TRUE(prependValidatedCompactPrefix(*input, {first, first}, kPeriod, &failure)) << failure;
  const auto prefix_end = input->prefix_endpoint_sample_index;
  ASSERT_GT(prefix_end, 0U);
  const auto prefix = input->samples;
  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*input, limits, kPeriod, 1.0);
  ASSERT_EQ(result.status, CompactDynamicsStatus::kRetimed) << result.failure_reason;
  ASSERT_TRUE(result.profile);
  for (size_t i = 0; i <= prefix_end; ++i) {
    expectSameState(result.profile->samples[i], prefix[i]);
                                                                                                   }
  EXPECT_EQ(result.profile->prefix_endpoint_sample_index, prefix_end);
  expectFitsUrdfEffort(result, loaded, group);
}

TEST(CompactDynamics, EffectiveUrdfEffortOverrideControlsRetiming)
{
  const auto loaded = loadXarm(40.0);
  const auto & group = groupOf(loaded, "xarm6");
  const auto limits = makeLimits(loaded.model, "xarm6");
  const auto input = makeLineProfile(recordedQ(), recordedDirection(), limits, 0.34);
  ASSERT_TRUE(input);
  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*input, limits, kPeriod, 1.0);
  ASSERT_EQ(result.status, CompactDynamicsStatus::kRetimed) << result.failure_reason;
  ASSERT_TRUE(result.profile);
  EXPECT_LT(result.applied_speed_scale, 1.0);
  expectFitsUrdfEffort(result, loaded, group);
}

TEST(CompactDynamics, FixedBranchPayloadMatchesHandGravityAndVariableOrder)
{
  const auto loaded = parseModel(handUrdf(),
    "<robot name='hand_dynamics'><group name='hand'><chain base_link='base_link' tip_link='link2'/>"
    "</group></robot>");
  const auto & group = groupOf(loaded, "hand");
  ASSERT_EQ(group.getVariableNames(), (std::vector<std::string>{"z_joint", "a_joint"}));
  const auto limits = makeLimits(loaded.model, "hand");
  const auto q = handPosition(group, 0.0);
  const auto profile = makeLineProfile(q, Eigen::VectorXd::Zero(q.size()), limits, 0.0);
  ASSERT_TRUE(profile);
  CompactDynamicsModel dynamics(loaded.model, "hand", "base_link");
  const auto result = dynamics.validateAndRetime(*profile, limits, kPeriod, 1.0);
  ASSERT_EQ(result.status, CompactDynamicsStatus::kPass) << result.failure_reason;
  ASSERT_EQ(result.peak_required_effort_nm_after.size(), 2U);
  const auto j2 = variableIndex(group, "a_joint");
  EXPECT_NEAR(result.peak_required_effort_nm_after[j2], 2.0 * 9.81, 0.05);
}

TEST(CompactDynamics, WorldGravityIsRotatedIntoTheGroupBaseFrame)
{
  const std::string srdf =
    "<robot name='hand_dynamics'><group name='hand'><chain base_link='base_link' tip_link='link2'/>"
    "</group></robot>";
  const auto identity = parseModel(handUrdf(30.0, 0.0, 0.0, 0.0), srdf);
  const auto rotated = parseModel(handUrdf(30.0, 0.0, 0.0, kPi / 2.0), srdf);
  const auto & group = groupOf(identity, "hand");
  const auto q = handPosition(group, 0.0);
  const auto limits = makeLimits(identity.model, "hand");
  const auto profile = makeLineProfile(q, Eigen::VectorXd::Zero(q.size()), limits, 0.0);
  ASSERT_TRUE(profile);
  CompactDynamicsModel identity_dynamics(identity.model, "hand", "base_link");
  CompactDynamicsModel rotated_dynamics(rotated.model, "hand", "base_link");
  const auto identity_result = identity_dynamics.validateAndRetime(*profile, limits, kPeriod, 1.0);
  const auto rotated_result = rotated_dynamics.validateAndRetime(*profile, limits, kPeriod, 1.0);
  ASSERT_EQ(identity_result.status, CompactDynamicsStatus::kPass);
  ASSERT_EQ(rotated_result.status, CompactDynamicsStatus::kPass);
  const auto j2 = variableIndex(group, "a_joint");
  EXPECT_NEAR(identity_result.peak_required_effort_nm_after[j2], 19.62, 0.05);
  EXPECT_LT(rotated_result.peak_required_effort_nm_after[j2], 0.1);
}

TEST(CompactDynamics, UrdfDampingAndSignedFrictionAffectMovingEffort)
{
  const std::string srdf =
    "<robot name='hand_dynamics'><group name='hand'><chain base_link='base_link' tip_link='link2'/>"
    "</group></robot>";
  const auto loaded = parseModel(handUrdf(100.0, 0.1, 0.5), srdf);
  const auto no_dynamics = parseModel(handUrdf(100.0, 0.0, 0.0), srdf);
  const auto & group = groupOf(loaded, "hand");
  const auto positive = handConstantVelocity(group, 2.0, kPeriod);
  const auto negative = handConstantVelocity(group, -2.0, kPeriod);
  const auto limits = makeLimits(loaded.model, "hand");
  CompactDynamicsModel with_urdf_dynamics(loaded.model, "hand", "base_link");
  CompactDynamicsModel without_urdf_dynamics(no_dynamics.model, "hand", "base_link");
  const auto plus = with_urdf_dynamics.validateAndRetime(positive, limits, kPeriod, 1.0);
  const auto minus = with_urdf_dynamics.validateAndRetime(negative, limits, kPeriod, 1.0);
  const auto plain_plus = without_urdf_dynamics.validateAndRetime(positive, limits, kPeriod, 1.0);
  const auto plain_minus = without_urdf_dynamics.validateAndRetime(negative, limits, kPeriod, 1.0);
  ASSERT_EQ(plus.status, CompactDynamicsStatus::kPass);
  ASSERT_EQ(minus.status, CompactDynamicsStatus::kPass);
  ASSERT_EQ(plain_plus.status, CompactDynamicsStatus::kPass);
  ASSERT_EQ(plain_minus.status, CompactDynamicsStatus::kPass);
  const auto j2 = variableIndex(group, "a_joint");
  EXPECT_NEAR(plain_plus.peak_required_effort_nm_after[j2] - plus.peak_required_effort_nm_after[j2],
        0.7, 0.03);
  EXPECT_NEAR(minus.peak_required_effort_nm_after[j2] -
        plain_minus.peak_required_effort_nm_after[j2], 0.7, 0.03);
}

TEST(CompactDynamics, GravityHoldAboveEffectiveLimitFailsClosed)
{
  const auto loaded = parseModel(handUrdf(18.0),
    "<robot name='hand_dynamics'><group name='hand'><chain base_link='base_link' tip_link='link2'/>"
    "</group></robot>");
  const auto & group = groupOf(loaded, "hand");
  const auto limits = makeLimits(loaded.model, "hand");
  const auto q = handPosition(group, 0.0);
  const auto profile = makeLineProfile(q, Eigen::VectorXd::Zero(q.size()), limits, 0.0);
  ASSERT_TRUE(profile);
  CompactDynamicsModel dynamics(loaded.model, "hand", "base_link");
  const auto result = dynamics.validateAndRetime(*profile, limits, kPeriod, 1.0);
  EXPECT_EQ(result.status, CompactDynamicsStatus::kStaticInfeasible);
  EXPECT_FALSE(result.profile);
  EXPECT_FALSE(result.failure_reason.empty());
}

TEST(CompactDynamics, StaticFeasibleButNoBoundedTempoReturnsNoProfile)
{
  const auto loaded = parseModel(handUrdf(21.0),
    "<robot name='hand_dynamics'><group name='hand'><chain base_link='base_link' tip_link='link2'/>"
    "</group></robot>");
  const auto limits = makeLimits(loaded.model, "hand");
  const auto & group = groupOf(loaded, "hand");
  const auto input = makeLineProfile(handPosition(group, 0.0), handDirection(group, -1.0), limits,
        0.34);
  ASSERT_TRUE(input);
  CompactDynamicsModel dynamics(loaded.model, "hand", "base_link");
  const auto result = dynamics.validateAndRetime(*input, limits, kPeriod, 1.0);
  EXPECT_EQ(result.status, CompactDynamicsStatus::kNoFeasibleScale)
    << "effort ratio after=" << result.maximum_effort_ratio_after
    << " peak=" << (result.peak_required_effort_nm_after.empty() ? -1.0 :
  result.peak_required_effort_nm_after[variableIndex(group, "a_joint")])
    << " reason=" << result.failure_reason;
  EXPECT_FALSE(result.profile);
  EXPECT_LT(result.checked_profile_samples, 10000U);
  EXPECT_LT(result.checked_interpolation_samples, 10000U);
}

TEST(CompactDynamics, InvalidRequestedScaleFailsClosed)
{
  const auto loaded = loadXarm();
  const auto limits = makeLimits(loaded.model, "xarm6");
  const auto input = makeLineProfile(recordedQ(), recordedDirection(), limits, 0.34);
  ASSERT_TRUE(input);
  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*input, limits, kPeriod, 0.0);
  EXPECT_EQ(result.status, CompactDynamicsStatus::kInvalidInput);
  EXPECT_FALSE(result.profile);
}

TEST(CompactDynamics, InvalidSmoothPathProgressFailsClosed)
{
  const auto loaded = loadXarm();
  const auto limits = makeLimits(loaded.model, "xarm6");
  auto input = makeLineProfile(recordedQ(), recordedDirection(), limits, 0.34);
  ASSERT_TRUE(input);
  ASSERT_GE(input->path_progress_rad_per_sample.size(), 2U);
  input->path_progress_rad_per_sample.back() += 0.01;

  CompactDynamicsModel dynamics(loaded.model, "xarm6", "link_base");
  const auto result = dynamics.validateAndRetime(*input, limits, kPeriod, 1.0);
  EXPECT_EQ(result.status, CompactDynamicsStatus::kInvalidInput);
  EXPECT_FALSE(result.profile);
  EXPECT_NE(result.failure_reason.find("progress"), std::string::npos);
}

}  // namespace
}  // namespace face_tracking_arm::control
