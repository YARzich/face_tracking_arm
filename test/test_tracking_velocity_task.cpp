// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>

#include "face_tracking_arm/tracking_velocity_task.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kTolerance = 1.0e-8;

Eigen::Matrix3d upright_rotation(const Eigen::Vector3d & direction)
{
  Eigen::Matrix3d rotation;
  rotation.col(0) = direction.normalized();
  rotation.col(1) = Eigen::Vector3d::UnitZ().cross(rotation.col(0)).normalized();
  rotation.col(2) = rotation.col(0).cross(rotation.col(1));
  return rotation;
}

TrackingVelocityTaskConfig zero_deadbands()
{
  TrackingVelocityTaskConfig config;
  config.position_deadband_m = 0.0;
  config.pointing_deadband_rad = 0.0;
  config.roll_deadband_rad = 0.0;
  return config;
}

TEST(TrackingVelocityTask, AcceptedStaticPoseHasNoBackgroundMotionPreference)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(0.2, 0.1, 1.1);
  const Eigen::Vector3d face(0.8, -0.1, 1.7);
  current.linear() = upright_rotation(face - current.translation());
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(), face,
    Eigen::Matrix3d::Identity());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->accepted);
  EXPECT_EQ(result->task.primary_matrix.rows(), 2);
  EXPECT_EQ(result->task.secondary_matrix.rows(), 4);
  EXPECT_TRUE(result->task.primary_reference.isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_reference.isZero(kTolerance));
  EXPECT_NEAR(result->position_error_m, 0.0, kTolerance);
  EXPECT_NEAR(result->pointing_error_rad, 0.0, kTolerance);
  EXPECT_NEAR(result->roll_error_rad, 0.0, kTolerance);
}

TEST(TrackingVelocityTask, PointingAndUprightRollJacobiansMatchFiniteDifference)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(-0.1, 0.2, 1.0);
  const Eigen::Vector3d face(0.8, 0.6, 1.7);
  current.linear() = upright_rotation(face - current.translation());
  Eigen::MatrixXd jacobian(6, 4);
  jacobian << 0.1, -0.3, 0.4, 0.0,
    0.2, 0.4, -0.1, 0.3,
    -0.2, 0.0, 0.3, -0.4,
    0.3, 0.0, -0.2, 0.1,
    -0.1, 0.5, 0.0, -0.3,
    0.2, -0.4, 0.2, 0.0;
  const auto result = makeTrackingVelocityTask(
    current, jacobian, current.translation(), face, Eigen::Matrix3d::Identity());
  ASSERT_TRUE(result.has_value());
  Eigen::Matrix3d basis;
  basis.row(0) = current.rotation().col(1).transpose();
  basis.row(1) = current.rotation().col(2).transpose();
  basis.row(2) = current.rotation().col(0).transpose();
  constexpr double delta = 1.0e-6;
  for (Eigen::Index column = 0; column < jacobian.cols(); ++column) {
    const Eigen::Vector3d angular_velocity = jacobian.col(column).tail(3);
    const Eigen::Vector3d moved_position =
      current.translation() + delta * jacobian.col(column).head(3);
    const Eigen::Matrix3d moved_rotation = Eigen::AngleAxisd(
      delta * angular_velocity.norm(), angular_velocity.normalized()).toRotationMatrix() *
      current.rotation();
    const Eigen::Matrix3d moved_goal = upright_rotation(face - moved_position);
    // Residual is actual orientation relative to the independently recomputed
    // look-at frame. Positive task velocity must equal this residual derivative.
    const Eigen::AngleAxisd residual(moved_rotation * moved_goal.transpose());
    const Eigen::Vector3d derivative =
      basis * (residual.axis() * (residual.angle() / delta));
    EXPECT_TRUE(result->task.primary_matrix.col(column).isApprox(derivative.head<2>(), 2.0e-6)) <<
      "column " << column << ": expected " << derivative.transpose() <<
      ", actual " << result->task.primary_matrix.col(column).transpose();
    EXPECT_NEAR(result->task.secondary_matrix(3, column), derivative[2], 2.0e-6);
  }
}

TEST(TrackingVelocityTask, MovingFaceFeedForwardMatchesPointingAndRollFiniteDifference)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(-0.1, 0.2, 1.0);
  const Eigen::Vector3d face(0.8, 0.6, 1.7);
  const Eigen::Vector3d face_velocity(0.12, -0.20, 0.09);
  current.linear() = upright_rotation(face - current.translation());
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(), face,
    Eigen::Matrix3d::Identity(), TrackingVelocityTaskConfig{}, face_velocity);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->accepted);
  EXPECT_TRUE(result->task.secondary_reference.head<3>().isZero(kTolerance));

  constexpr double delta = 1.0e-6;
  const Eigen::Matrix3d earlier = upright_rotation(
    face - delta * face_velocity - current.translation());
  const Eigen::Matrix3d later = upright_rotation(
    face + delta * face_velocity - current.translation());
  const Eigen::AngleAxisd frame_change(later * earlier.transpose());
  const Eigen::Vector3d angular_velocity =
    frame_change.axis() * (frame_change.angle() / (2.0 * delta));
  const Eigen::Vector3d expected(
    current.rotation().col(1).dot(angular_velocity),
    current.rotation().col(2).dot(angular_velocity),
    current.rotation().col(0).dot(angular_velocity));
  EXPECT_TRUE(result->task.primary_reference.isApprox(expected.head<2>(), 1.0e-8)) <<
    "expected " << expected.transpose() <<
    ", actual " << result->task.primary_reference.transpose();
  EXPECT_NEAR(result->task.secondary_reference[3], expected[2], 1.0e-8);
  EXPECT_GT(std::abs(expected[2]), 0.01);

  // Equal face and TCP translations preserve their relative ray with no
  // angular TCP motion. This checks the feed-forward/Jacobian coupling sign.
  Eigen::VectorXd tcp_velocity = Eigen::VectorXd::Zero(6);
  tcp_velocity.head(3) = face_velocity;
  EXPECT_TRUE((result->task.primary_matrix * tcp_velocity).isApprox(
      result->task.primary_reference, kTolerance));
}

TEST(TrackingVelocityTask, PositionFeedForwardRemainsActiveInsideGeometricDeadbands)
{
  const Eigen::Vector3d goal_velocity(0.03, -0.02, 0.01);
  const auto result = makeTrackingVelocityTask(
    Eigen::Isometry3d::Identity(), Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(),
    Eigen::Vector3d::UnitX(), Eigen::Matrix3d::Identity(), TrackingVelocityTaskConfig{},
    Eigen::Vector3d::Zero(), goal_velocity);
  ASSERT_TRUE(result.has_value());
  EXPECT_FALSE(result->accepted);
  EXPECT_TRUE(result->task.primary_reference.isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_reference.head<3>().isApprox(goal_velocity, kTolerance));
  EXPECT_NEAR(result->position_error_m, 0.0, kTolerance);
}

TEST(TrackingVelocityTask, ZeroFeedForwardPreservesExistingTaskExactly)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(0.1, 0.2, 1.0);
  const Eigen::Vector3d face(0.8, 0.4, 1.7);
  const Eigen::Vector3d target_position(0.3, 0.1, 1.2);
  const Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  const auto original = makeTrackingVelocityTask(
    current, jacobian, target_position, face, Eigen::Matrix3d::Identity());
  const auto explicit_zero = makeTrackingVelocityTask(
    current, jacobian, target_position, face, Eigen::Matrix3d::Identity(),
    TrackingVelocityTaskConfig{}, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  ASSERT_TRUE(original.has_value());
  ASSERT_TRUE(explicit_zero.has_value());
  EXPECT_EQ(original->accepted, explicit_zero->accepted);
  EXPECT_TRUE(original->task.primary_matrix.isApprox(explicit_zero->task.primary_matrix, 0.0));
  EXPECT_TRUE(original->task.primary_reference.isApprox(
      explicit_zero->task.primary_reference, 0.0));
  EXPECT_TRUE(original->task.secondary_reference.isApprox(
      explicit_zero->task.secondary_reference, 0.0));
}

TEST(TrackingVelocityTask, CombinedFeedbackAndFeedForwardRespectSpeedCaps)
{
  const TrackingVelocityTaskConfig config;
  const auto result = makeTrackingVelocityTask(
    Eigen::Isometry3d::Identity(), Eigen::MatrixXd::Identity(6, 6),
    Eigen::Vector3d(2.0, 0.0, 0.0), Eigen::Vector3d(1.0, 1.0, 0.5),
    Eigen::Matrix3d::Identity(), config, Eigen::Vector3d(0.0, 5.0, 0.0),
    Eigen::Vector3d(0.0, 2.0, 0.0));
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->task.primary_reference.norm(), config.maximum_angular_reference_radps,
    kTolerance);
  EXPECT_NEAR(result->task.secondary_reference.head<3>().norm(),
        config.maximum_linear_reference_mps,
    kTolerance);
  const Eigen::Vector3d combined(config.position_gain * (2.0 - config.position_deadband_m),
    2.0, 0.0);
  EXPECT_TRUE(result->task.secondary_reference.head<3>().isApprox(
      combined.normalized() * config.maximum_linear_reference_mps, kTolerance));
}

TEST(TrackingVelocityTask, TranslationRequiresTheCorrectOppositePointingRotation)
{
  const Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(),
    Eigen::Vector3d(2.0, 0.0, 0.0), Eigen::Matrix3d::Identity());
  ASSERT_TRUE(result.has_value());
  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(6);
  velocity[1] = 0.2;
  // Moving TCP towards +Y makes the face move towards -Y relative to the TCP.
  velocity[5] = -0.1;
  EXPECT_TRUE((result->task.primary_matrix * velocity).isZero(kTolerance));
  velocity[5] = 0.1;
  EXPECT_GT((result->task.primary_matrix * velocity).norm(), 0.19);
}

TEST(TrackingVelocityTask, FixedPoseDoesNotCoupleTranslationIntoOrientation)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(),
    std::nullopt, Eigen::Matrix3d::Identity(), zero_deadbands());
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->task.primary_matrix.leftCols(3).isZero(kTolerance));
  EXPECT_NEAR(result->pointing_error_rad, 0.0, kTolerance);
  EXPECT_NEAR(result->roll_error_rad, 0.2, kTolerance);
  EXPECT_TRUE(result->task.primary_reference.isZero(kTolerance));
  EXPECT_LT(result->task.secondary_reference[3], 0.0);
  EXPECT_FALSE(result->accepted);
}

TEST(TrackingVelocityTask, PointingAtFaceDoesNotAcceptPositionOnOppositeSide)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(-0.3, 0.0, 1.0);
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d(0.3, 0.0, 1.0),
    Eigen::Vector3d(0.8, 0.0, 1.0), Eigen::Matrix3d::Identity());
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->pointing_error_rad, 0.0, kTolerance);
  EXPECT_NEAR(result->position_error_m, 0.6, kTolerance);
  EXPECT_GT(result->task.secondary_reference[0], 0.0);
  EXPECT_FALSE(result->accepted);
}

TEST(TrackingVelocityTask, ReferencesAreBoundedAndDeadbandBoundaryIsContinuous)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = (
    Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitX())).toRotationMatrix();
  TrackingVelocityTaskConfig config;
  const auto large = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d(2.0, 1.0, 1.0),
    std::nullopt, Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(large.has_value());
  EXPECT_LE(large->task.primary_reference.norm(), config.maximum_angular_reference_radps + 1.e-12);
  EXPECT_LE(large->task.secondary_reference.head<3>().norm(),
        config.maximum_linear_reference_mps + 1.e-12);

  current = Eigen::Isometry3d::Identity();
  for (const double displacement : {config.position_deadband_m - 1.e-8,
      config.position_deadband_m + 1.e-8})
  {
    const auto result = makeTrackingVelocityTask(
      current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d(displacement, 0.0, 0.0),
      std::nullopt, Eigen::Matrix3d::Identity(), config);
    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result->task.secondary_reference.norm(), 2.e-8);
    EXPECT_EQ(result->accepted, displacement <= config.position_deadband_m);
  }
}

TEST(TrackingVelocityTask, OppositeNormalProducesFiniteNonzeroCorrection)
{
  Eigen::Matrix3d target = Eigen::Matrix3d::Identity();
  target(0, 0) = -1.0;
  target(1, 1) = -1.0;
  const auto result = makeTrackingVelocityTask(
    Eigen::Isometry3d::Identity(), Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(),
    std::nullopt, target);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->task.primary_reference.allFinite());
  EXPECT_GT(result->pointing_error_rad, 3.14);
  EXPECT_GT(result->task.primary_reference.head(2).norm(), 0.1);
}

TEST(TrackingVelocityTask, RejectsInvalidAndUndefinedGeometry)
{
  const auto current = Eigen::Isometry3d::Identity();
  const Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  const auto invalid = [&](const Eigen::MatrixXd & matrix,
    const std::optional<Eigen::Vector3d> & face,
    const Eigen::Matrix3d & rotation,
    const TrackingVelocityTaskConfig & config = TrackingVelocityTaskConfig{}) {
      return !makeTrackingVelocityTask(
        current, matrix, Eigen::Vector3d::Zero(), face, rotation, config).has_value();
    };
  EXPECT_TRUE(invalid(Eigen::MatrixXd::Zero(5, 6), std::nullopt, Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(invalid(Eigen::MatrixXd::Zero(6, 0), std::nullopt, Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(invalid(jacobian, Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()));
  EXPECT_FALSE(invalid(jacobian, Eigen::Vector3d::UnitZ(), Eigen::Matrix3d::Identity()));
  EXPECT_TRUE(invalid(jacobian, std::nullopt, Eigen::Matrix3d::Zero()));
  Eigen::MatrixXd nonfinite = jacobian;
  nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(invalid(nonfinite, std::nullopt, Eigen::Matrix3d::Identity()));
  TrackingVelocityTaskConfig config;
  config.roll_weight = 0.0;
  EXPECT_TRUE(invalid(jacobian, std::nullopt, Eigen::Matrix3d::Identity(), config));
  const Eigen::Vector3d invalid_velocity = Eigen::Vector3d::Constant(
    std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(makeTrackingVelocityTask(
      current, jacobian, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
      Eigen::Matrix3d::Identity(), TrackingVelocityTaskConfig{}, invalid_velocity).has_value());
  EXPECT_FALSE(makeTrackingVelocityTask(
      current, jacobian, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitX(),
      Eigen::Matrix3d::Identity(), TrackingVelocityTaskConfig{}, Eigen::Vector3d::Zero(),
      invalid_velocity).has_value());
}

TEST(TrackingVelocityTask, VerticalFaceKeepsPointingAndDisablesUnobservableUprightRoll)
{
  for (const double x : {-1.0e-9, 0.0, 1.0e-9}) {
    const auto result = makeTrackingVelocityTask(
      Eigen::Isometry3d::Identity(), Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(x, 0.0, 1.0), Eigen::Matrix3d::Identity(),
      TrackingVelocityTaskConfig{}, Eigen::Vector3d(0.1, 0.1, 0.0));
    ASSERT_TRUE(result);
    EXPECT_EQ(result->task.primary_matrix.rows(), 2);
    EXPECT_TRUE(result->task.primary_matrix.allFinite());
    EXPECT_LE(result->task.primary_matrix.norm(), 3.0);
    EXPECT_GT(result->task.primary_reference.norm(), 0.1);
    EXPECT_TRUE(result->task.secondary_matrix.row(3).isZero(1.0e-6));
    EXPECT_NEAR(result->task.secondary_reference[3], 0.0, 1.0e-6);
  }
}

TEST(TrackingVelocityTask, OffsetAndRotatedCameraCentersFaceWithoutMovingMonitorPositionGoal)
{
  Eigen::Isometry3d monitor = Eigen::Isometry3d::Identity();
  monitor.translation() = Eigen::Vector3d(0.2, 0.0, 1.0);
  const Eigen::Vector3d offset(0.0, 0.03, 0.125);
  Eigen::Isometry3d optical = monitor;
  optical.translation() += offset;
  optical.linear() = Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitY()).toRotationMatrix() *
    opticalToPointingRotation().transpose();
  GazeKinematics gaze;
  gaze.pose = optical;
  gaze.pose.linear() *= opticalToPointingRotation();
  gaze.jacobian = Eigen::MatrixXd::Identity(6, 6);
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    gaze.jacobian.block<3, 1>(0, axis + 3) = Eigen::Vector3d::Unit(axis).cross(offset);
  }
  const Eigen::Vector3d face = optical.translation() + optical.linear().col(2);
  const auto result = makeTrackingVelocityTask(
    monitor, Eigen::MatrixXd::Identity(6, 6), monitor.translation(), face,
    Eigen::Matrix3d::Identity(), TrackingVelocityTaskConfig{},
    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), gaze);
  ASSERT_TRUE(result);
  EXPECT_NEAR(result->pointing_error_rad, 0.0, kTolerance);
  EXPECT_TRUE(result->task.primary_reference.isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_reference.head<3>().isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_matrix.topRows(3).isApprox(
      Eigen::MatrixXd::Identity(6, 6).topRows(3), kTolerance));
  EXPECT_TRUE(result->accepted);

  const auto monitor_only = makeTrackingVelocityTask(
    monitor, Eigen::MatrixXd::Identity(6, 6), monitor.translation(), face,
    Eigen::Matrix3d::Identity());
  ASSERT_TRUE(monitor_only);
  EXPECT_GT(monitor_only->pointing_error_rad, 0.02);
  EXPECT_FALSE(monitor_only->accepted);

  // Independent finite difference includes camera translation induced by a
  // rotation about the monitor origin; using the monitor Jacobian would fail.
  Eigen::Matrix<double, 2, 3> basis;
  basis.row(0) = gaze.pose.linear().col(1).transpose();
  basis.row(1) = gaze.pose.linear().col(2).transpose();
  constexpr double delta = 1.0e-6;
  for (Eigen::Index column = 0; column < 6; ++column) {
    const Eigen::Vector3d w = gaze.jacobian.col(column).tail<3>();
    const auto residual_at = [&](double step) {
        Eigen::Matrix3d moved_rotation = gaze.pose.linear();
        if (w.norm() > 0.0) {
          moved_rotation = Eigen::AngleAxisd(step * w.norm(), w.normalized()) * moved_rotation;
        }
        const Eigen::Vector3d moved_position = gaze.pose.translation() +
          step * gaze.jacobian.col(column).head<3>();
        const Eigen::AngleAxisd residual(
          moved_rotation * upright_rotation(face - moved_position).transpose());
        return Eigen::Vector3d(residual.axis() * residual.angle());
      };
    const Eigen::Vector2d derivative = basis *
      (residual_at(delta) - residual_at(-delta)) / (2.0 * delta);
    EXPECT_LT((result->task.primary_matrix.col(column) - derivative).norm(), 1.0e-8) << column;
  }
  gaze.jacobian.resize(5, 6);
  EXPECT_FALSE(makeTrackingVelocityTask(
      monitor, Eigen::MatrixXd::Identity(6, 6), monitor.translation(), face,
      Eigen::Matrix3d::Identity(), TrackingVelocityTaskConfig{},
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), gaze));
}

TEST(JointPathVelocityTask, PreservesPathDirectionAndRealBoundedAngles)
{
  const Eigen::Vector3d current(6.0, 0.0, 0.0);
  const Eigen::Vector3d waypoint(-6.0, 6.0, -3.0);
  const auto task = makeJointPathVelocityTask(current, waypoint, 2.0, 1.0);
  ASSERT_TRUE(task.has_value());
  EXPECT_TRUE(task->primary_matrix.isIdentity(kTolerance));
  EXPECT_TRUE(task->primary_reference.isApprox(Eigen::Vector3d(-1.0, 0.5, -0.25), kTolerance));
  EXPECT_EQ(task->secondary_matrix.rows(), 0);
  EXPECT_EQ(task->secondary_matrix.cols(), 3);
}

TEST(JointPathVelocityTask, DeadbandAndInvalidInput)
{
  const auto task = makeJointPathVelocityTask(
    Eigen::Vector3d::Zero(), Eigen::Vector3d(0.01, 0.005, 0.0), 2.0, 1.0, 0.02);
  ASSERT_TRUE(task.has_value());
  EXPECT_TRUE(task->primary_reference.isZero(kTolerance));
  EXPECT_FALSE(makeJointPathVelocityTask(
      Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(3), 2.0, 1.0).has_value());
  EXPECT_FALSE(makeJointPathVelocityTask(
      Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(), 2.0, 0.0).has_value());
}

}  // namespace
}  // namespace face_tracking_arm::control
