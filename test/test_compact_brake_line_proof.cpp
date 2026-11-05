// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "compact_brake_line_proof.hpp"
#include "face_tracking_arm/compact_joint_profile.hpp"

namespace face_tracking_arm::control
{
namespace
{

JointMotionState state(const double q1, const double q2, const double q3)
{
  JointMotionState result;
  result.position = Eigen::Vector3d(q1, q2, q3);
  result.velocity = Eigen::Vector3d::Zero();
  result.acceleration = Eigen::Vector3d::Zero();
  return result;
}

JointMotionState capturedState(const std::array<double, 6> & positions)
{
  JointMotionState result;
  result.position.resize(6);
  for (std::size_t index = 0; index < positions.size(); ++index) {
    result.position[static_cast<Eigen::Index>(index)] = positions[index];
  }
  result.velocity = Eigen::VectorXd::Zero(6);
  result.acceleration = Eigen::VectorXd::Zero(6);
  return result;
}

std::vector<JointMotionState> lineTail(const double from, const double to)
{
  return {state(from, 0.0, 0.0), state((from + to) / 2.0, 0.0, 0.0),
    state(to, 0.0, 0.0), state(to, 0.0, 0.0)};
}

CompactBrakeLineProofContext context(
  const std::uint64_t scene, const std::uint64_t config_revision)
{
  CompactBrakeLineProofContext result;
  result.scene_revision = scene;
  result.config_revision = config_revision;
  result.model_identity = reinterpret_cast<const void *>(1U);
  result.group_identity = reinterpret_cast<const void *>(2U);
  result.fixed_positions = Eigen::VectorXd::Zero(1);
  return result;
}

constexpr std::size_t kProtectedAxis = 2U;
const PositionInterval kProtectedInterval{-1.0, 1.0};
constexpr Radians kPositionError{1.0e-6};
constexpr double kTubeError{1.0e-5};

TEST(CompactBrakeLineProofTest, ReusesOverlappingSameLineTail)
{
  CompactBrakeLineProof proof;
  const auto identity = context(4U, 1U);
  const auto first = proof.plan(
    lineTail(0.0, 1.0), identity, kTubeError, kProtectedAxis,
    kProtectedInterval, kPositionError);
  ASSERT_EQ(first.result, CompactBrakeLinePlan::Result::kValidate);
  proof.commit(first);

  const auto overlap = proof.plan(
    lineTail(0.25, 0.8), identity, kTubeError, kProtectedAxis,
    kProtectedInterval, kPositionError);
  EXPECT_EQ(overlap.result, CompactBrakeLinePlan::Result::kCovered);
}

TEST(CompactBrakeLineProofTest, ValidatesOnlyUncoveredExtension)
{
  CompactBrakeLineProof proof;
  const auto identity = context(4U, 1U);
  proof.commit(proof.plan(lineTail(0.0, 1.0), identity, kTubeError,
    kProtectedAxis, kProtectedInterval, kPositionError));

  const auto extension = proof.plan(
    lineTail(0.75, 1.5), identity, kTubeError, kProtectedAxis,
    kProtectedInterval, kPositionError);
  ASSERT_EQ(extension.result, CompactBrakeLinePlan::Result::kValidate);
  EXPECT_TRUE(extension.start_position.isApprox(Eigen::Vector3d(1.0, 0.0, 0.0)));
  EXPECT_TRUE(extension.end_position.isApprox(Eigen::Vector3d(1.5, 0.0, 0.0)));
}

TEST(CompactBrakeLineProofTest, RejectsOffLineTail)
{
  CompactBrakeLineProof proof;
  const auto identity = context(4U, 1U);
  proof.commit(proof.plan(lineTail(0.0, 1.0), identity, kTubeError,
    kProtectedAxis, kProtectedInterval, kPositionError));
  const std::vector<JointMotionState> off_line{
    state(0.5, 0.0, 0.0), state(0.75, 0.02, 0.0), state(1.0, 0.0, 0.0)};
  EXPECT_EQ(proof.plan(off_line, identity, kTubeError, kProtectedAxis,
    kProtectedInterval, kPositionError).result, CompactBrakeLinePlan::Result::kMiss);
}

TEST(CompactBrakeLineProofTest, SceneOrConfigChangeRequiresFullIntervalValidation)
{
  CompactBrakeLineProof proof;
  proof.commit(proof.plan(lineTail(0.0, 1.0), context(4U, 1U),
    kTubeError, kProtectedAxis, kProtectedInterval, kPositionError));
  const auto changed_scene = proof.plan(
    lineTail(0.25, 1.25), context(5U, 1U), kTubeError,
    kProtectedAxis, kProtectedInterval, kPositionError);
  ASSERT_EQ(changed_scene.result, CompactBrakeLinePlan::Result::kValidate);
  EXPECT_TRUE(changed_scene.start_position.isApprox(Eigen::Vector3d(0.25, 0.0, 0.0)));

  proof.commit(changed_scene);
  const auto changed_config = proof.plan(
    lineTail(0.5, 1.5), context(5U, 2U), kTubeError,
    kProtectedAxis, kProtectedInterval, kPositionError);
  ASSERT_EQ(changed_config.result, CompactBrakeLinePlan::Result::kValidate);
  EXPECT_TRUE(changed_config.start_position.isApprox(Eigen::Vector3d(0.5, 0.0, 0.0)));
}

TEST(CompactBrakeLineProofTest, FixedRobotStateChangeRequiresFullIntervalValidation)
{
  CompactBrakeLineProof proof;
  auto original = context(4U, 1U);
  proof.commit(proof.plan(lineTail(0.0, 1.0), original, kTubeError,
    kProtectedAxis, kProtectedInterval, kPositionError));
  auto changed = original;
  changed.fixed_positions[0] = 0.01;
  const auto plan = proof.plan(
    lineTail(0.25, 1.25), changed, kTubeError, kProtectedAxis,
    kProtectedInterval, kPositionError);
  ASSERT_EQ(plan.result, CompactBrakeLinePlan::Result::kValidate);
  EXPECT_TRUE(plan.start_position.isApprox(Eigen::Vector3d(0.25, 0.0, 0.0)));
}

TEST(CompactBrakeLineProofTest, ProtectedCorridorFailureCannotReuseProof)
{
  CompactBrakeLineProof proof;
  const auto identity = context(4U, 1U);
  proof.commit(proof.plan(lineTail(0.0, 0.5), identity, kTubeError,
    kProtectedAxis, kProtectedInterval, kPositionError));
  const auto outside = proof.plan(
    {state(0.25, 0.0, 0.0), state(0.4, 0.0, 1.0)}, identity,
    kTubeError, kProtectedAxis, kProtectedInterval, kPositionError);
  EXPECT_EQ(outside.result, CompactBrakeLinePlan::Result::kMiss);
}

TEST(CompactBrakeLineProofTest, CapturedSequentialTailsReuseProfileToleranceTube)
{
  // Positions captured from compact PATH publications 139, 149, and 153 in
  // gain1_actual_candidate. Tail endpoints and intermediate samples are kept
  // here to exercise the canonical-line projection seen in the runtime trace.
  const std::vector<JointMotionState> first_tail{
    capturedState({6.235680188398633, 0.24251008123218157, -1.6923025349119034,
        0.051236464306750031, -0.74976410849754649, 6.195002123492114}),
    capturedState({6.2347418647143673, 0.24110402014801133, -1.6955234992192241,
        0.051236464306750031, -0.74341744193336934, 6.1951886025536238}),
    capturedState({6.2245218247514007, 0.22578947495262242, -1.7306056190592076,
        0.051236464306750031, -0.67429077806975002, 6.1972196963998476})};
  const std::vector<JointMotionState> second_tail{
    capturedState({6.226202330621005, 0.22830768262934645, -1.7248369813119109,
        0.051236464306750031, -0.68565744246537941, 6.1968857187302238}),
    capturedState({6.1771059233060033, 0.15473760350232066, -1.8933692034342162,
        0.051236464306750031, -0.35357744496306714, 6.2066429614739533}),
    capturedState({6.1770586128709839, 0.15466670967040472, -1.8935316049861695,
        0.051236464306750031, -0.3532574449874667, 6.2066523637790105})};
  const std::vector<JointMotionState> third_tail{
    capturedState({6.2149247048763714, 0.2114083642520477, -1.7635494539947383,
        0.051236464306750031, -0.60937744289062556, 6.1991269933566402}),
    capturedState({6.1526763899481729, 0.11813038923834492, -1.9772279574116709,
        0.051236464306750031, -0.18834011253668398, 6.2114979987388743}),
    capturedState({6.1526030587762675, 0.11802050380243098, -1.977479679809053,
        0.051236464306750031, -0.18784411259055322, 6.2115125723112419})};
  const auto identity = context(4U, 1U);
  constexpr std::size_t kCapturedProtectedJoint = 3U;
  const PositionInterval captured_protected_interval{-1.0, 1.0};

  // This is NumericalPolicy::position_error for the captured 10 ms, 3 rad/s
  // policy (position scale 2*pi): projection drift exceeds it on tail 149.
  constexpr double kMachinePositionError{2.8036174747024086e-15};
  CompactBrakeLineProof machine_precision_proof;
  machine_precision_proof.commit(machine_precision_proof.plan(
    first_tail, identity, kMachinePositionError, kCapturedProtectedJoint,
    captured_protected_interval, Radians{kMachinePositionError}));
  EXPECT_EQ(machine_precision_proof.plan(
      second_tail, identity, kMachinePositionError, kCapturedProtectedJoint,
      captured_protected_interval, Radians{kMachinePositionError}).result,
    CompactBrakeLinePlan::Result::kMiss);

  CompactBrakeLineProof proof;
  const double tube_error = std::max(
    kMachinePositionError, kCompactProfilePositionToleranceRad);
  const auto first = proof.plan(
    first_tail, identity, tube_error, kCapturedProtectedJoint,
    captured_protected_interval, Radians{kMachinePositionError});
  ASSERT_EQ(first.result, CompactBrakeLinePlan::Result::kValidate);
  proof.commit(first);

  const auto second = proof.plan(
    second_tail, identity, tube_error, kCapturedProtectedJoint,
    captured_protected_interval, Radians{kMachinePositionError});
  ASSERT_EQ(second.result, CompactBrakeLinePlan::Result::kValidate);
  EXPECT_TRUE(second.origin.isApprox(first.origin, 0.0));
  EXPECT_TRUE(second.direction.isApprox(first.direction, 0.0));
  EXPECT_TRUE(second.start_position.isApprox(first.end_position, 1.0e-12));
  proof.commit(second);

  const auto third = proof.plan(
    third_tail, identity, tube_error, kCapturedProtectedJoint,
    captured_protected_interval, Radians{kMachinePositionError});
  ASSERT_EQ(third.result, CompactBrakeLinePlan::Result::kValidate);
  EXPECT_TRUE(third.origin.isApprox(first.origin, 0.0));
  EXPECT_TRUE(third.direction.isApprox(first.direction, 0.0));
  EXPECT_TRUE(third.start_position.isApprox(second.end_position, 1.0e-12));
  EXPECT_TRUE((third.end_position - third.start_position).norm() > 0.0);
}

}  // namespace
}  // namespace face_tracking_arm::control
