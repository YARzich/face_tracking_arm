// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "face_tracking_arm/posture_preference.hpp"

namespace face_tracking_arm::control
{
namespace
{
PosturePreferenceConfig configWithoutWristMargins()
{
  PosturePreferenceConfig config;
  config.comfortable_joint3_lower = -0.1;
  config.comfortable_joint3_upper = 0.1;
  config.position_margin_j5_rad = 0.0;
  config.position_margin_j6_rad = 0.0;
  return config;
}

PosturePreferenceConfig calibratedTestConfig()
{
  PosturePreferenceConfig config;
  config.comfortable_joint3_lower = -0.1;
  config.comfortable_joint3_upper = 0.1;
  return config;
}

constexpr Eigen::Index kJoints = 6;
const Eigen::VectorXd kJointPosition = Eigen::VectorXd::Zero(kJoints);
const Eigen::MatrixXd kZeroJacobian = Eigen::MatrixXd::Zero(3, kJoints);

// Signed vertical distance of the elbow from the infinite shoulder-wrist line.
double signedUpwardBend(
  const Eigen::Vector3d & shoulder, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist)
{
  const Eigen::Vector3d line = wrist - shoulder;
  const Eigen::Vector3d offset = elbow - shoulder;
  return (offset - line * (offset.dot(line) / line.squaredNorm())).z();
}

PosturePreferenceObjective objective(
  const Eigen::Vector3d & shoulder, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist,
  const Eigen::MatrixXd & js = kZeroJacobian,
  const Eigen::MatrixXd & je = kZeroJacobian,
  const Eigen::MatrixXd & jw = kZeroJacobian,
  const Eigen::VectorXd & q = kJointPosition,
  const PosturePreferenceConfig & config = configWithoutWristMargins())
{
  return makePosturePreferenceObjective(shoulder, elbow, wrist, js, je, jw, q, config);
}
}  // namespace

TEST(PosturePreference, NamedPosesCalibrateTheComfortableBranchAndWinding)
{
  constexpr double offset = 2.73318426911364;
  const auto calibration = calibratePostureBend(
    -0.3696309078954824, -0.3696309078954824 + offset,
    -0.4670725353145707, -0.4670725353145707 + offset,
    -3.927, 0.19198);
  ASSERT_TRUE(calibration.valid);
  EXPECT_NEAR(calibration.angle_offset_rad, offset, 1.0e-12);
  EXPECT_DOUBLE_EQ(calibration.preferred_sign, 1.0);
  EXPECT_NEAR(calibration.minimum_comfortable_sin,
    std::min(std::sin(-0.3696309078954824 + offset),
      std::sin(-0.4670725353145707 + offset)), 1.0e-12);
  EXPECT_LE(calibration.joint3_lower, -0.3696309078954824);
  EXPECT_GE(calibration.joint3_upper, -0.4670725353145707);
  EXPECT_FALSE(calibratePostureBend(
      -0.2, 0.2, 0.2, 0.2, -1.0, 1.0).valid)
    << "Named poses on opposite bend branches must not pick a false comfort range";
}

TEST(PosturePreference, OutsideComfortJointThreeReturnsTowardInterval)
{
  auto config = configWithoutWristMargins();
  config.comfortable_joint3_lower = -1.8;
  config.comfortable_joint3_upper = -0.9;
  Eigen::VectorXd q = kJointPosition;
  q[2] = -3.8;
  const auto result = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.4, 0.0, 0.1),
    Eigen::Vector3d(0.8, 0.0, 0.0), kZeroJacobian, kZeroJacobian,
    kZeroJacobian, q, config);
  ASSERT_TRUE(result.valid);
  ASSERT_FALSE(result.branch_comfortable);
  ASSERT_EQ(result.joint3_correction.matrix.rows(), 1);
  EXPECT_DOUBLE_EQ(result.joint3_correction.matrix(0, 2), 1.0);
  EXPECT_GT(result.joint3_correction.reference[0], 0.0);
  EXPECT_LE(result.joint3_correction.reference[0], config.maximum_angular_reference_radps);
  EXPECT_EQ(result.other_corrections.matrix.rows(), 0);
}

TEST(PosturePreference, UpwardBendAllowsElbowBelowWristAndRejectsNegativeH)
{
  const Eigen::Vector3d shoulder = Eigen::Vector3d::Zero();
  const Eigen::Vector3d wrist(1.0, 0.0, 1.0);
  const Eigen::Vector3d elbow_above_line(0.45, 0.0, 0.55);
  const Eigen::Vector3d elbow_below_line(0.55, 0.0, 0.45);
  ASSERT_LT(elbow_above_line.z(), wrist.z());
  EXPECT_NEAR(signedUpwardBend(shoulder, elbow_above_line, wrist), 0.05, 1.0e-12);
  EXPECT_NEAR(signedUpwardBend(shoulder, elbow_below_line, wrist), -0.05, 1.0e-12);

  const auto good = objective(shoulder, elbow_above_line, wrist);
  ASSERT_TRUE(good.valid);
  EXPECT_TRUE(good.branch_comfortable);
  EXPECT_TRUE(good.geometry_comfortable);
  EXPECT_TRUE(good.comfortable);
  EXPECT_EQ(good.joint3_correction.matrix.rows(), 0);
  EXPECT_EQ(good.other_corrections.matrix.rows(), 0);

  const auto bad = objective(shoulder, elbow_below_line, wrist);
  ASSERT_TRUE(bad.valid);
  EXPECT_TRUE(bad.branch_comfortable);
  EXPECT_FALSE(bad.geometry_comfortable);
  EXPECT_FALSE(bad.comfortable);
  EXPECT_EQ(bad.joint3_correction.matrix.rows(), 0);
  ASSERT_EQ(bad.other_corrections.matrix.rows(), 1);
  EXPECT_GT(bad.other_corrections.reference[0], 0.0);
}

TEST(PosturePreference, FaceDirectionIsNotAnInputToObjective)
{
  // The objective accepts only S/E/W and their Jacobians; target direction is
  // deliberately absent, so changing the face cannot change its posture row.
  using ObjectiveFunction = PosturePreferenceObjective (*)(
    const Eigen::Vector3d &, const Eigen::Vector3d &, const Eigen::Vector3d &,
    const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::MatrixXd &,
    const Eigen::VectorXd &, const PosturePreferenceConfig &);
  static_assert(std::is_same_v<decltype(&makePosturePreferenceObjective), ObjectiveFunction>);
  const Eigen::Vector3d shoulder = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.5, 0.0, -0.03);
  const Eigen::Vector3d wrist(1.0, 0.0, 0.0);
  const auto result = objective(shoulder, elbow, wrist);
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.other_corrections.matrix.rows(), 1);
  EXPECT_GT(result.other_corrections.reference[0], 0.0);
}

TEST(PosturePreference, VerticalShoulderWristLineFadesGeometry)
{
  const auto vertical = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0.0, 0.5),
    Eigen::Vector3d(0.0, 0.0, 1.0));
  ASSERT_TRUE(vertical.valid);
  EXPECT_TRUE(vertical.branch_comfortable);
  EXPECT_TRUE(vertical.geometry_comfortable)
    << "The upward side is indeterminate when the shoulder-wrist line is vertical";
  EXPECT_TRUE(vertical.comfortable);
  EXPECT_EQ(vertical.joint3_correction.matrix.rows(), 0);
  EXPECT_EQ(vertical.other_corrections.matrix.rows(), 0);
}

TEST(PosturePreference, AnalyticUpwardBendJacobianMatchesCenteredDifference)
{
  const Eigen::Vector3d shoulder(0.2, -0.4, 0.3);
  const Eigen::Vector3d elbow(0.7, -0.1, 0.6);
  const Eigen::Vector3d wrist(1.0, 0.2, 1.0);
  Eigen::MatrixXd js = Eigen::MatrixXd::Zero(3, kJoints);
  Eigen::MatrixXd je = Eigen::MatrixXd::Zero(3, kJoints);
  Eigen::MatrixXd jw = Eigen::MatrixXd::Zero(3, kJoints);
  for (Eigen::Index index = 0; index < kJoints; ++index) {
    const double scale = static_cast<double>(index + 1);
    js.col(index) = Eigen::Vector3d(0.01 * scale, -0.006 * scale, 0.003 * scale);
    je.col(index) = js.col(index) + Eigen::Vector3d(0.02 * scale, 0.01 * scale, -0.015 * scale);
    jw.col(index) = je.col(index) + Eigen::Vector3d(0.03, -0.02, 0.01) * scale;
  }
  const auto result = objective(shoulder, elbow, wrist, js, je, jw);
  ASSERT_EQ(result.other_corrections.matrix.rows(), 1);

  const Eigen::Vector3d line = wrist - shoulder;
  const double line_squared = line.squaredNorm();
  const double horizontal_squared = line.head<2>().squaredNorm();
  const double forearm_squared = (wrist - elbow).squaredNorm();
  const double activation = horizontal_squared / (line_squared + forearm_squared);
  constexpr double step = 1.0e-6;
  for (Eigen::Index index = 0; index < kJoints; ++index) {
    const double plus = signedUpwardBend(
      shoulder + step * js.col(index), elbow + step * je.col(index),
      wrist + step * jw.col(index));
    const double minus = signedUpwardBend(
      shoulder - step * js.col(index), elbow - step * je.col(index),
      wrist - step * jw.col(index));
    const double finite_difference = activation * (plus - minus) / (2.0 * step);
    EXPECT_NEAR(result.other_corrections.matrix(0, index), finite_difference, 2.0e-7)
      << "joint " << index;
  }
}

TEST(PosturePreference, ReserveStopsCorrectionAtOneLinearControlStep)
{
  auto config = configWithoutWristMargins();
  config.maximum_linear_reference_mps = 0.18;
  config.control_period_sec = 0.01;
  const Eigen::Vector3d shoulder = Eigen::Vector3d::Zero();
  const Eigen::Vector3d wrist(1.0, 0.0, 0.0);
  const auto beyond_reserve = objective(
    shoulder, Eigen::Vector3d(0.5, 0.0, 0.0021), wrist,
    kZeroJacobian, kZeroJacobian, kZeroJacobian, kJointPosition, config);
  EXPECT_TRUE(beyond_reserve.geometry_comfortable);
  EXPECT_TRUE(beyond_reserve.comfortable);
  EXPECT_EQ(beyond_reserve.joint3_correction.matrix.rows(), 0);
  EXPECT_EQ(beyond_reserve.other_corrections.matrix.rows(), 0);

  const auto violation = objective(
    shoulder, Eigen::Vector3d(0.5, 0.0, -0.01), wrist,
    kZeroJacobian, kZeroJacobian, kZeroJacobian, kJointPosition, config);
  EXPECT_FALSE(violation.geometry_comfortable);
  ASSERT_EQ(violation.other_corrections.matrix.rows(), 1);
  EXPECT_GT(violation.other_corrections.reference[0], 0.0);
  EXPECT_LE(violation.other_corrections.reference[0], config.maximum_linear_reference_mps);
}

TEST(PosturePreference, DegenerateLineSkipsGeometryAndPreservesJ5J6Margins)
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(kJoints);
  q[4] = -1.65;
  q[5] = 6.25;
  const auto degenerate = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.2),
    Eigen::Vector3d::Zero(), kZeroJacobian, kZeroJacobian, kZeroJacobian,
    q, calibratedTestConfig());
  ASSERT_TRUE(degenerate.valid);
  EXPECT_TRUE(degenerate.geometry_comfortable)
    << "A zero-length shoulder-wrist line has no defined upward side";
  EXPECT_EQ(degenerate.joint3_correction.matrix.rows(), 0);
  ASSERT_EQ(degenerate.other_corrections.matrix.rows(), 2);
  EXPECT_DOUBLE_EQ(degenerate.other_corrections.matrix(0, 4), 1.0);
  EXPECT_DOUBLE_EQ(degenerate.other_corrections.matrix(1, 5), 1.0);
  EXPECT_TRUE(degenerate.other_corrections.matrix.allFinite());
  EXPECT_TRUE(degenerate.other_corrections.reference.allFinite());
}

TEST(PosturePreference, UpwardGeometryAndBothWristMarginsCanBeActiveTogether)
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(kJoints);
  q[4] = -1.65;
  q[5] = 6.25;
  Eigen::MatrixXd je = Eigen::MatrixXd::Zero(3, kJoints);
  je(2, 1) = 0.1;
  const auto result = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.0, -0.03),
    Eigen::Vector3d(1.0, 0.0, 0.0), kZeroJacobian, je,
    kZeroJacobian, q, calibratedTestConfig());
  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.joint3_correction.matrix.rows(), 0);
  ASSERT_EQ(result.other_corrections.matrix.rows(), 3);
  EXPECT_GT(result.other_corrections.matrix.row(0).norm(), 0.0);
  EXPECT_NE(result.other_corrections.matrix(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(result.other_corrections.matrix(0, 4), 0.0);
  EXPECT_DOUBLE_EQ(result.other_corrections.matrix(0, 5), 0.0);
  EXPECT_DOUBLE_EQ(result.other_corrections.matrix(1, 4), 1.0);
  EXPECT_DOUBLE_EQ(result.other_corrections.matrix(2, 5), 1.0);
  EXPECT_GT(result.other_corrections.reference[0], 0.0);
  EXPECT_GT(result.other_corrections.reference[1], 0.0);
  EXPECT_LT(result.other_corrections.reference[2], 0.0);
}
}  // namespace face_tracking_arm::control
