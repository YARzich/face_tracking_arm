// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

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

// Signed distance along the target's perpendicular direction, positive when
// elbow and face project to opposite sides of the infinite base-wrist line.
double signedClearance(
  const Eigen::Vector3d & base, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist, const Eigen::Vector3d & face)
{
  const Eigen::Vector3d e = elbow - base;
  const Eigen::Vector3d w = wrist - base;
  const Eigen::Vector3d f = face - base;
  const Eigen::Vector3d f_perp = f - w * (f.dot(w) / w.squaredNorm());
  const double c = e.dot(f) - e.dot(w) * f.dot(w) / w.squaredNorm();
  return -c / f_perp.norm();
}

double smoothNormalizedClearance(
  const Eigen::Vector3d & base, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist, const Eigen::Vector3d & face, const double reserve)
{
  const Eigen::Vector3d e = elbow - base;
  const Eigen::Vector3d w = wrist - base;
  const Eigen::Vector3d f = face - base;
  const double d = w.squaredNorm();
  const Eigen::Vector3d f_perp = f - w * (f.dot(w) / d);
  const double c = e.dot(f) - e.dot(w) * f.dot(w) / d;
  const double forearm_length = (wrist - elbow).norm();
  const double r = f_perp.norm();
  return (-c - reserve * r) /
         (std::sqrt(r * r + forearm_length * forearm_length) * forearm_length);
}

PosturePreferenceObjective objective(
  const Eigen::Vector3d & base, const Eigen::Vector3d & elbow,
  const Eigen::Vector3d & wrist, const Eigen::Vector3d & face,
  const Eigen::MatrixXd & jb = kZeroJacobian,
  const Eigen::MatrixXd & je = kZeroJacobian,
  const Eigen::MatrixXd & jw = kZeroJacobian,
  const Eigen::VectorXd & q = kJointPosition,
  const PosturePreferenceConfig & config = configWithoutWristMargins())
{
  return makePosturePreferenceObjective(base, elbow, wrist, face, jb, je, jw, q, config);
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

TEST(PosturePreference, OutsideComfortFsideIsOffAndBranchPreferenceReturnsTowardInterval)
{
  auto config = configWithoutWristMargins();
  config.joint3_angle_offset_rad = 2.73318426911364;
  config.preferred_bend_sign = 1.0;
  config.comfortable_joint3_lower = -1.8;
  config.comfortable_joint3_upper = -0.9;
  Eigen::VectorXd q = kJointPosition;
  q[2] = -3.8;
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.4, -0.1, 0.0);
  const Eigen::Vector3d wrist(0.8, 0.0, 0.0);
  const auto toward = objective(
    base, elbow, wrist, Eigen::Vector3d(0.0, 0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, config);
  const auto opposite_face = objective(
    base, elbow, wrist, Eigen::Vector3d(0.0, -0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, config);
  ASSERT_TRUE(toward.valid);
  ASSERT_FALSE(toward.comfortable);
  ASSERT_EQ(toward.matrix.rows(), 1);
  EXPECT_TRUE(toward.matrix.isApprox(opposite_face.matrix));
  EXPECT_TRUE(toward.reference.isApprox(opposite_face.reference));
  EXPECT_DOUBLE_EQ(toward.matrix(0, 2), 1.0);
  EXPECT_GT(toward.reference[0], 0.0);
  EXPECT_FALSE(opposite_face.branch_comfortable);

  q[2] = -1.2;
  const auto inside = objective(
    base, elbow, wrist, Eigen::Vector3d(0.0, -0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, config);
  ASSERT_TRUE(inside.branch_comfortable);
  EXPECT_FALSE(inside.comfortable)
    << "The face-side row remains active after entering the normal branch";
  ASSERT_EQ(inside.matrix.rows(), 1);
  EXPECT_NE(inside.matrix(0, 2), 1.0)
    << "The interior face-side row remains independent of joint3";
}

TEST(PosturePreference, BranchPreferenceUsesDirectJointVelocityRow)
{
  auto config = configWithoutWristMargins();
  config.joint3_angle_offset_rad = 2.73318426911364;
  config.preferred_bend_sign = 1.0;
  config.comfortable_joint3_lower = -1.8;
  config.comfortable_joint3_upper = -0.9;
  Eigen::VectorXd q = kJointPosition;
  q[2] = -3.5;
  const auto result = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.2, 0.1, 0.0),
    Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(0.0, 0.2, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, config);
  ASSERT_EQ(result.matrix.rows(), 1);
  EXPECT_DOUBLE_EQ(result.matrix(0, 2), 1.0);
  EXPECT_GT(result.reference[0], 0.0);
  EXPECT_LE(result.reference[0], config.maximum_angular_reference_radps);
}

TEST(PosturePreference, ComfortableJointThreeIsNotPulledAwayFromIntervalBoundary)
{
  auto config = configWithoutWristMargins();
  config.comfortable_joint3_lower = -1.0;
  config.comfortable_joint3_upper = 1.0;
  config.maximum_angular_reference_radps = 0.7;
  config.control_period_sec = 0.01;
  config.position_gain = 1.0;
  Eigen::VectorXd q = kJointPosition;
  q[2] = -0.999;
  const auto at_boundary = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.4, -0.1, 0.0),
    Eigen::Vector3d(0.8, 0.0, 0.0), Eigen::Vector3d(0.0, 0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, config);
  ASSERT_TRUE(at_boundary.branch_comfortable);
  for (Eigen::Index row = 0; row < at_boundary.matrix.rows(); ++row) {
    EXPECT_DOUBLE_EQ(at_boundary.matrix(row, 2), 0.0)
      << "A GOOD pose has no J3 centering row; other soft geometry may remain";
    EXPECT_DOUBLE_EQ(at_boundary.reference[row], 0.0);
  }

  q[2] = -0.992;
  const auto inside_layer = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.4, -0.1, 0.0),
    Eigen::Vector3d(0.8, 0.0, 0.0), Eigen::Vector3d(0.0, 0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, config);
  ASSERT_TRUE(inside_layer.branch_comfortable);
  for (Eigen::Index row = 0; row < inside_layer.matrix.rows(); ++row) {
    EXPECT_DOUBLE_EQ(inside_layer.matrix(row, 2), 0.0);
    EXPECT_DOUBLE_EQ(inside_layer.reference[row], 0.0);
  }
}

TEST(PosturePreference, FaceSideMirroringChangesElbowClassification)
{
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.5, -0.02, 0.0);
  const Eigen::Vector3d wrist(1.0, 0.0, 0.0);
  const auto opposite = objective(base, elbow, wrist, Eigen::Vector3d(0.0, 0.4, 0.0));
  const auto same = objective(base, elbow, wrist, Eigen::Vector3d(0.0, -0.4, 0.0));
  EXPECT_GT(signedClearance(base, elbow, wrist, Eigen::Vector3d(0.0, 0.4, 0.0)), 0.0);
  EXPECT_LT(signedClearance(base, elbow, wrist, Eigen::Vector3d(0.0, -0.4, 0.0)), 0.0);
  EXPECT_TRUE(opposite.comfortable);
  EXPECT_FALSE(same.comfortable)
    << "The active face-side row makes the posture objective uncomfortable";
  EXPECT_TRUE(same.branch_comfortable);
  ASSERT_EQ(same.matrix.rows(), 1);
  EXPECT_GT(same.reference[0], 0.0);
}

TEST(PosturePreference, GeometryIsRigidTransformInvariantAndNotWorldZBased)
{
  const Eigen::Vector3d base(0.1, -0.2, 0.3);
  const Eigen::Vector3d wrist(0.1, -0.2, 1.0);  // vertical BW
  const Eigen::Vector3d elbow(0.4, -0.2, 0.1);  // same side despite its world height
  const Eigen::Vector3d face(0.5, -0.5, 0.8);
  const auto original = objective(base, elbow, wrist, face);
  EXPECT_FALSE(original.comfortable);
  EXPECT_TRUE(original.branch_comfortable);
  ASSERT_EQ(original.matrix.rows(), 1);

  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(
    0.9, Eigen::Vector3d(1.0, 2.0, -0.5).normalized()).toRotationMatrix();
  const Eigen::Vector3d translation(2.0, -3.0, 4.0);
  const auto transformed = objective(
    rotation * base + translation, rotation * elbow + translation,
    rotation * wrist + translation, rotation * face + translation,
    rotation * kZeroJacobian, rotation * kZeroJacobian, rotation * kZeroJacobian);
  EXPECT_FALSE(transformed.comfortable);
  EXPECT_TRUE(transformed.branch_comfortable);
  ASSERT_EQ(transformed.matrix.rows(), 1);
  EXPECT_TRUE(transformed.matrix.isApprox(original.matrix, 1.0e-11));
  EXPECT_TRUE(transformed.reference.isApprox(original.reference, 1.0e-11));
}

TEST(PosturePreference, ReferencePointChoiceCanChangeClassification)
{
  // Arbitrary anchors show why the reference point is part of the geometry.
  // This test does not prescribe which point represents the user's intent.
  const Eigen::Vector3d anchor_a(0.0, 0.0, 0.0);
  const Eigen::Vector3d anchor_b(0.0, 1.0, 0.0);
  const Eigen::Vector3d elbow(0.5, 0.1, 0.0);
  const Eigen::Vector3d wrist(1.0, 0.0, 0.0);
  const Eigen::Vector3d face(0.0, -0.2, 0.0);

  EXPECT_GT(signedClearance(anchor_a, elbow, wrist, face), 0.0);
  EXPECT_LT(signedClearance(anchor_b, elbow, wrist, face), 0.0);
  EXPECT_TRUE(objective(anchor_a, elbow, wrist, face).comfortable);
  const auto other_anchor = objective(anchor_b, elbow, wrist, face);
  EXPECT_FALSE(other_anchor.comfortable);
  EXPECT_TRUE(other_anchor.branch_comfortable);
  ASSERT_EQ(other_anchor.matrix.rows(), 1);
  EXPECT_GT(other_anchor.reference[0], 0.0);
}

TEST(PosturePreference, AnalyticJacobianMatchesCenteredDifferenceWithFixedFace)
{
  const Eigen::Vector3d base(0.2, -0.4, 0.3);
  const Eigen::Vector3d elbow(0.7, -0.1, 0.6);
  const Eigen::Vector3d wrist(1.0, 0.2, 1.0);
  const Eigen::Vector3d face(0.8, 0.8, 0.4);
  const Eigen::Vector3d forearm = wrist - elbow;
  Eigen::MatrixXd jb = Eigen::MatrixXd::Zero(3, kJoints);
  Eigen::MatrixXd je = Eigen::MatrixXd::Zero(3, kJoints);
  Eigen::MatrixXd jw = Eigen::MatrixXd::Zero(3, kJoints);
  for (Eigen::Index index = 0; index < kJoints; ++index) {
    const double scale = static_cast<double>(index + 1);
    const Eigen::Vector3d vb(0.01 * scale, -0.006 * scale, 0.003 * scale);
    const Eigen::Vector3d ve(0.02 * scale, 0.01 * scale, -0.015 * scale);
    const Eigen::Vector3d vw = forearm.cross(Eigen::Vector3d(0.03, -0.02, 0.01) * scale) +
      forearm.normalized() * (0.004 * scale);
    jb.col(index) = vb;
    je.col(index) = vb + ve;
    jw.col(index) = vb + ve + vw;
  }
  const auto result = objective(base, elbow, wrist, face, jb, je, jw);
  ASSERT_EQ(result.matrix.rows(), 1);
  const Eigen::Vector3d initial_w = wrist - base;
  const Eigen::Vector3d initial_f = face - base;
  const Eigen::Vector3d initial_f_perp = initial_f -
    initial_w * (initial_f.dot(initial_w) / initial_w.squaredNorm());
  const double forearm_length = (wrist - elbow).norm();
  const double activation = initial_w.squaredNorm() /
    (initial_w.squaredNorm() + forearm_length * forearm_length) *
    initial_f_perp.squaredNorm() /
    (initial_f_perp.squaredNorm() + forearm_length * forearm_length);

  constexpr double step = 1.0e-6;
  for (Eigen::Index index = 0; index < kJoints; ++index) {
    const double reserve = configWithoutWristMargins().maximum_linear_reference_mps *
      configWithoutWristMargins().control_period_sec;
    const double plus = smoothNormalizedClearance(
      base + step * jb.col(index), elbow + step * je.col(index),
      wrist + step * jw.col(index), face, reserve);
    const double minus = smoothNormalizedClearance(
      base - step * jb.col(index), elbow - step * je.col(index),
      wrist - step * jw.col(index), face, reserve);
    const double finite_difference = activation * (plus - minus) / (2.0 * step);
    EXPECT_NEAR(result.matrix(0, index), finite_difference, 2.0e-7) << "joint " << index;
  }
}

TEST(PosturePreference, FaceMovementChangesTheSideWithoutEnteringItsJacobian)
{
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.5, 0.03, 0.0);
  const Eigen::Vector3d wrist(1.0, 0.0, 0.0);
  const auto left = objective(base, elbow, wrist, Eigen::Vector3d(0.0, 0.4, 0.0));
  const auto right = objective(base, elbow, wrist, Eigen::Vector3d(0.0, -0.4, 0.0));
  ASSERT_EQ(left.matrix.rows(), 1);
  EXPECT_TRUE(right.comfortable);
}

TEST(PosturePreference, ReserveStopsCorrectionAndIsBasedOnOneControlStep)
{
  auto config = configWithoutWristMargins();
  config.maximum_linear_reference_mps = 0.18;
  config.control_period_sec = 0.01;
  const auto beyond_reserve = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, -0.01, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, kJointPosition, config);
  EXPECT_TRUE(beyond_reserve.comfortable);
  EXPECT_EQ(beyond_reserve.matrix.rows(), 0);

  const auto near_projection_beyond_reserve = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, -0.0021, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0e-4, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, kJointPosition, config);
  EXPECT_GT(signedClearance(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, -0.0021, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0e-4, 0.0)), 0.002);
  EXPECT_TRUE(near_projection_beyond_reserve.comfortable);

  const auto violation = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.01, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.5, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, kJointPosition, config);
  ASSERT_EQ(violation.matrix.rows(), 1);
  EXPECT_GT(violation.reference[0], 0.0);
  EXPECT_LE(violation.reference[0], config.maximum_linear_reference_mps);
}

TEST(PosturePreference, FaceOnLineAndApproachFadeMatrixAndReferenceSmoothly)
{
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.5, 0.05, 0.0);
  const Eigen::Vector3d wrist(1.0, 0.0, 0.0);
  Eigen::MatrixXd je = Eigen::MatrixXd::Zero(3, kJoints);
  je(1, 0) = 0.1;
  const auto on_line = objective(base, elbow, wrist, Eigen::Vector3d(0.5, 0.0, 0.0),
    kZeroJacobian, je);
  EXPECT_TRUE(on_line.comfortable);
  EXPECT_EQ(on_line.matrix.rows(), 0);

  const auto near = objective(base, elbow, wrist, Eigen::Vector3d(0.5, 1.0e-5, 0.0),
    kZeroJacobian, je);
  const auto ordinary = objective(base, elbow, wrist, Eigen::Vector3d(0.5, 0.2, 0.0),
    kZeroJacobian, je);
  ASSERT_EQ(near.matrix.rows(), 1);
  ASSERT_EQ(ordinary.matrix.rows(), 1);
  EXPECT_TRUE(near.matrix.allFinite());
  EXPECT_TRUE(near.reference.allFinite());
  EXPECT_LT(near.matrix.norm(), ordinary.matrix.norm() * 1.0e-5);
  EXPECT_LT(near.reference.norm(), ordinary.reference.norm() * 1.0e-5);
}

TEST(PosturePreference, RegularizedLeastSquaresCommandFadesAsFaceProjectionApproachesLine)
{
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.25, 0.03, 0.0);
  const Eigen::Vector3d wrist(0.5, 0.0, 0.0);
  Eigen::MatrixXd je = Eigen::MatrixXd::Zero(3, kJoints);
  je(1, 0) = 0.1;
  constexpr double production_regularization = 1.0e-4;
  const auto command_norm = [&](const double face_y) {
      const auto task = objective(base, elbow, wrist, Eigen::Vector3d(0.1, face_y, 0.0),
        kZeroJacobian, je);
      if (task.matrix.rows() == 0) {
        return 0.0;
      }
      EXPECT_EQ(task.matrix.rows(), 1);
      const Eigen::RowVectorXd row = task.matrix.row(0);
      const Eigen::VectorXd velocity = row.transpose() * task.reference[0] /
        (row.squaredNorm() + production_regularization);
      EXPECT_TRUE(velocity.allFinite());
      return velocity.norm();
    };

  double previous = command_norm(0.20);
  ASSERT_GT(previous, 0.0);
  for (const double face_y : {0.10, 0.05, 0.01, 0.001, 0.0001, 0.00001}) {
    const double current = command_norm(face_y);
    EXPECT_LT(current, previous) << "face projection y=" << face_y;
    previous = current;
  }
  EXPECT_LT(command_norm(-0.00001), 1.0e-12)
    << "after crossing, this elbow already has the preferred side";
}

TEST(PosturePreference, ShortBaseWristLineFadesGeometryAndKeepsFiniteRows)
{
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d elbow(0.0001, 0.01, 0.0);
  const Eigen::Vector3d wrist(0.0002, 0.0, 0.0);
  const Eigen::Vector3d face(0.0, 0.3, 0.0);
  Eigen::MatrixXd jw = Eigen::MatrixXd::Zero(3, kJoints);
  jw(1, 1) = 0.1;
  const auto short_line = objective(base, elbow, wrist, face, kZeroJacobian, kZeroJacobian, jw);
  const auto regular_line = objective(base, Eigen::Vector3d(0.5, 0.01, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), face, kZeroJacobian, kZeroJacobian, jw);
  ASSERT_EQ(short_line.matrix.rows(), 1);
  ASSERT_EQ(regular_line.matrix.rows(), 1);
  EXPECT_TRUE(short_line.matrix.allFinite());
  EXPECT_TRUE(short_line.reference.allFinite());
  EXPECT_LT(short_line.matrix.norm(), regular_line.matrix.norm());
  EXPECT_LT(short_line.reference.norm(), regular_line.reference.norm());
}

TEST(PosturePreference, DegenerateGeometrySkipsOnlyItsRowAndPreservesJ5J6Margins)
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(kJoints);
  q[4] = -1.65;
  q[5] = 6.25;
  const auto degenerate = objective(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.2),
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 1.0, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, calibratedTestConfig());
  ASSERT_TRUE(degenerate.valid);
  ASSERT_EQ(degenerate.matrix.rows(), 2);
  EXPECT_DOUBLE_EQ(degenerate.matrix(0, 4), 1.0);
  EXPECT_DOUBLE_EQ(degenerate.matrix(1, 5), 1.0);
  EXPECT_TRUE(degenerate.matrix.allFinite());
  EXPECT_TRUE(degenerate.reference.allFinite());

  const auto invalid = objective(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.01, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d::Constant(INFINITY));
  EXPECT_TRUE(invalid.valid);
  EXPECT_TRUE(invalid.comfortable);
  EXPECT_EQ(invalid.matrix.rows(), 0);
}

TEST(PosturePreference, WristMarginsRemainIndependentSoftRows)
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(kJoints);
  q[4] = -1.65;
  q[5] = 6.25;
  const auto result = objective(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.2),
    Eigen::Vector3d(1.0, 0.0, 1.0), Eigen::Vector3d(0.5, 0.0, 0.0),
    kZeroJacobian, kZeroJacobian, kZeroJacobian, q, calibratedTestConfig());
  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.matrix.rows(), 2);
  EXPECT_DOUBLE_EQ(result.matrix(0, 4), 1.0);
  EXPECT_DOUBLE_EQ(result.matrix(1, 5), 1.0);
  EXPECT_GT(result.reference[0], 0.0);
  EXPECT_LT(result.reference[1], 0.0);
}

TEST(PosturePreference, GeometryAndBothWristMarginsCanBeActiveTogether)
{
  Eigen::VectorXd q = Eigen::VectorXd::Zero(kJoints);
  q[4] = -1.65;
  q[5] = 6.25;
  Eigen::MatrixXd je = Eigen::MatrixXd::Zero(3, kJoints);
  je(1, 1) = 0.1;
  const auto result = objective(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.03, 0.0),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.5, 0.0),
    kZeroJacobian, je, kZeroJacobian, q, calibratedTestConfig());
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.matrix.rows(), 3);
  EXPECT_GT(result.matrix.row(0).norm(), 0.0);
  EXPECT_NE(result.matrix(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(0, 4), 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(0, 5), 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(1, 4), 1.0);
  EXPECT_DOUBLE_EQ(result.matrix(2, 5), 1.0);
  EXPECT_GT(result.reference[0], 0.0);
  EXPECT_GT(result.reference[1], 0.0);
  EXPECT_LT(result.reference[2], 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(1, 3), 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(2, 0), 0.0);
  EXPECT_DOUBLE_EQ(result.matrix(2, 3), 0.0);
}
}  // namespace face_tracking_arm::control
