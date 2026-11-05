// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>

#include "face_tracking_arm/horizontal_edge_upright.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"
#include "face_tracking_arm/tracking_velocity_task.hpp"

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

Eigen::MatrixXd revolute_z_jacobian(
  const Eigen::Vector3d & point, const Eigen::Vector3d & axis_origin)
{
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  jacobian.block<3, 1>(0, 0) =
    Eigen::Vector3d::UnitZ().cross(point - axis_origin);
  jacobian.block<3, 1>(3, 0) = Eigen::Vector3d::UnitZ();
  return jacobian;
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
  EXPECT_EQ(result->task.primary_matrix.rows(), 3);
  EXPECT_EQ(result->task.secondary_matrix.rows(), 3);
  EXPECT_DOUBLE_EQ(result->task.primary_weights[2], 0.5);
  EXPECT_TRUE(result->task.primary_reference.isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_reference.isZero(kTolerance));
  EXPECT_NEAR(result->position_error_m, 0.0, kTolerance);
  EXPECT_NEAR(result->pointing_error_rad, 0.0, kTolerance);
  EXPECT_NEAR(result->roll_error_rad, 0.0, kTolerance);
  EXPECT_TRUE(result->has_face_primary_roll);
  EXPECT_TRUE(result->signed_face_primary_error_rad.isZero(kTolerance));
  EXPECT_TRUE(result->face_primary_feedforward_radps.isZero(kTolerance));
}

TEST(TrackingVelocityTask, FaceDistanceBandUsesControlFrameAndHasNoPointTargetInside)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(0.5, 0.0, 0.0);
  current.linear() = Eigen::Matrix3d::Identity();
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;

  const auto inside = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(inside.has_value());
  ASSERT_TRUE(inside->face_distance_interval.has_value());
  EXPECT_FALSE(inside->face_distance_outside_band);
  EXPECT_EQ(inside->task.secondary_matrix.rows(), 0);
  EXPECT_EQ(inside->task.distance_matrix.rows(), 0)
    << "A band membership must leave distance motion free";
  ASSERT_EQ(inside->face_distance_matrix.rows(), 1);
  EXPECT_NEAR(inside->face_distance_matrix(0, 0), -1.0, kTolerance);
  EXPECT_NEAR(inside->face_distance_reference[0], 0.0, kTolerance);
  EXPECT_NEAR(inside->face_distance_interval->lower_bound, -0.15, kTolerance);
  EXPECT_NEAR(inside->face_distance_interval->upper_bound, 0.15, kTolerance);
  EXPECT_DOUBLE_EQ(inside->position_error_m, 0.0);

  current.translation() = Eigen::Vector3d(0.55, 0.0, 0.0);
  const auto another_inside = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(another_inside.has_value());
  EXPECT_FALSE(another_inside->face_distance_outside_band);
  EXPECT_NEAR(another_inside->face_distance_reference[0], 0.0, kTolerance);
  EXPECT_EQ(another_inside->task.secondary_matrix.rows(), 0);

  current.translation() = Eigen::Vector3d(0.7, 0.1, -0.1);
  const auto outside = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(),
    Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.03, -0.01, 0.02));
  ASSERT_TRUE(outside.has_value());
  ASSERT_TRUE(outside->face_distance_interval.has_value());
  EXPECT_TRUE(outside->face_distance_outside_band);
  const Eigen::Vector3d radial =
    (current.translation() - Eigen::Vector3d(1.0, 0.0, 0.0)).normalized();
  EXPECT_NEAR(outside->position_error_m, 0.4 - std::sqrt(0.11), kTolerance);
  EXPECT_EQ(outside->task.secondary_matrix.rows(), 0);
  ASSERT_EQ(outside->task.distance_matrix.rows(), 3);
  EXPECT_TRUE(outside->task.distance_matrix.isApprox(
      Eigen::MatrixXd::Identity(6, 6).topRows(3), kTolerance));
  EXPECT_TRUE(outside->task.distance_reference.isApprox(
      radial * outside->face_distance_reference[0], kTolerance));
  EXPECT_TRUE(outside->task.distance_weights.isApprox(Eigen::VectorXd::Ones(3), kTolerance));

  config.face_distance_max_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(makeTrackingVelocityTask(
      current, Eigen::MatrixXd::Identity(6, 6), current.translation(),
      Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Matrix3d::Identity(), config));
}

TEST(TrackingVelocityTask, FarDistanceUsesFullCartesianObjectiveAlongFaceRayWhenStationary)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d::Zero();
  current.linear() = Eigen::Matrix3d::Identity();
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  constexpr double kFarDistance = 3.0;
  const Eigen::Vector3d face(kFarDistance, 0.0, 0.0);

  const auto far = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(), face,
    Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(far.has_value());
  ASSERT_TRUE(far->face_distance_outside_band);
  EXPECT_NEAR(far->position_error_m, kFarDistance - config.face_distance_max_m, kTolerance);
  ASSERT_EQ(far->face_distance_matrix.rows(), 1);
  EXPECT_NEAR(far->face_distance_reference[0], -config.maximum_linear_reference_mps,
    kTolerance)
    << "Unreachable distance should request only the maximum available approach rate";

  EXPECT_EQ(far->task.secondary_matrix.rows(), 0)
    << "Distance is a final soft objective and must not replace the gaze hierarchy level";
  EXPECT_TRUE(far->task.secondary_constraints.empty())
    << "An unreachable distance target must not become a hard velocity interval";
  ASSERT_EQ(far->task.distance_matrix.rows(), 3);
  EXPECT_TRUE(far->task.distance_matrix.isApprox(
      Eigen::MatrixXd::Identity(6, 6).topRows(3), kTolerance));
  EXPECT_TRUE(far->task.distance_reference.isApprox(
      (current.translation() - face).normalized() * far->face_distance_reference[0],
      kTolerance));
}

TEST(TrackingVelocityTask, DistanceObjectiveActivatesOnlyOutsideTheConfiguredBand)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  const Eigen::Vector3d face(1.0, 0.0, 0.0);
  const Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);

  for (const double distance : {0.3995, 0.6005}) {
    current.translation().x() = 1.0 - distance;
    const auto outside = makeTrackingVelocityTask(
      current, jacobian, current.translation(), face, Eigen::Matrix3d::Identity(), config);
    ASSERT_TRUE(outside.has_value());
    ASSERT_TRUE(outside->face_distance_outside_band);
    ASSERT_TRUE(outside->face_distance_interval);
    ASSERT_EQ(outside->task.distance_matrix.rows(), 3);
    EXPECT_TRUE(outside->task.secondary_constraints.empty());
    EXPECT_EQ(outside->task.secondary_matrix.rows(), 0);
    const Eigen::Vector3d expected = (current.translation() - face).normalized() *
      config.position_gain * (distance < 0.4 ? 0.4 - distance : 0.6 - distance);
    EXPECT_TRUE(outside->task.distance_reference.isApprox(expected, kTolerance));
  }

  for (const double distance : {0.4, 0.4005, 0.5, 0.5995, 0.6}) {
    current.translation().x() = 1.0 - distance;
    const auto inside = makeTrackingVelocityTask(
      current, jacobian, current.translation(), face, Eigen::Matrix3d::Identity(), config,
      Eigen::Vector3d(-0.025, 0.0, 0.0));
    ASSERT_TRUE(inside.has_value());
    ASSERT_FALSE(inside->face_distance_outside_band);
    ASSERT_TRUE(inside->face_distance_interval);
    EXPECT_EQ(inside->task.distance_matrix.rows(), 0)
      << "An empty distance level leaves motion within the band unconstrained";
    EXPECT_EQ(inside->task.distance_reference.size(), 0);
    EXPECT_TRUE(inside->task.secondary_constraints.empty());
  }
}

TEST(TrackingVelocityTask, ToleranceShellRetainsConfiguredIntervalAndAcceptance)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.position_deadband_m = 0.03;
  config.face_distance_min_m = 0.37;
  config.face_distance_max_m = 0.43;
  const Eigen::Vector3d face(1.0, 0.0, 0.0);
  const Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);

  for (const double distance : {0.3688, 0.4312}) {
    current.translation().x() = 1.0 - distance;
    const auto stationary = makeTrackingVelocityTask(
      current, jacobian, current.translation(), face, Eigen::Matrix3d::Identity(), config);
    ASSERT_TRUE(stationary.has_value());
    ASSERT_TRUE(stationary->face_distance_interval.has_value());
    EXPECT_TRUE(stationary->face_distance_outside_band);
    EXPECT_FALSE(stationary->accepted)
      << "The posture shell does not widen configured-band acceptance";
    EXPECT_NEAR(stationary->position_error_m, 0.0012, 1.0e-12);
    EXPECT_EQ(stationary->task.secondary_matrix.rows(), 0);
    EXPECT_EQ(stationary->task.distance_matrix.rows(), 3);
    EXPECT_NEAR(stationary->face_distance_interval->lower_bound,
      config.position_gain * (config.face_distance_min_m - distance), 1.0e-12);
    EXPECT_NEAR(stationary->face_distance_interval->upper_bound,
      config.position_gain * (config.face_distance_max_m - distance), 1.0e-12);

    const Eigen::Vector3d face_velocity(-0.025, 0.0, 0.0);
    const auto moving = makeTrackingVelocityTask(
      current, jacobian, current.translation(), face, Eigen::Matrix3d::Identity(), config,
      face_velocity);
    ASSERT_TRUE(moving.has_value());
    ASSERT_TRUE(moving->face_distance_interval.has_value());
    EXPECT_TRUE(moving->face_distance_outside_band);
    EXPECT_NEAR(moving->face_distance_reference[0],
      config.position_gain * (distance < config.face_distance_min_m ?
      config.face_distance_min_m - distance : config.face_distance_max_m - distance) + 0.025,
      1.0e-12);
  }

  for (const double distance : {config.face_distance_min_m, config.face_distance_max_m}) {
    current.translation().x() = 1.0 - distance;
    const auto at_configured_edge = makeTrackingVelocityTask(
      current, jacobian, current.translation(), face, Eigen::Matrix3d::Identity(), config);
    ASSERT_TRUE(at_configured_edge.has_value());
    EXPECT_TRUE(at_configured_edge->accepted);
    EXPECT_FALSE(at_configured_edge->face_distance_outside_band);
    EXPECT_EQ(at_configured_edge->task.distance_matrix.rows(), 0);
    EXPECT_EQ(at_configured_edge->task.secondary_matrix.rows(), 0);
  }
}

TEST(TrackingVelocityTask, J1CircleFollowsScreenTangentInEitherDirection)
{
  constexpr double omega = 0.13;
  const Eigen::Vector3d axis_origin(0.0, 0.0, 1.68);
  const Eigen::Vector3d screen = axis_origin + Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d face = axis_origin + Eigen::Vector3d(0.85, 0.0, 0.0);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  const Eigen::MatrixXd jacobian = revolute_z_jacobian(screen, axis_origin);

  for (const double direction : {-1.0, 1.0}) {
    const double signed_omega = direction * omega;
    const Eigen::Vector3d face_velocity(0.0, signed_omega * 0.85, 0.0);
    const auto result = makeTrackingVelocityTask(
      current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config, face_velocity);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(result->face_distance_outside_band);
    ASSERT_EQ(result->task.distance_matrix.rows(), 3);
    EXPECT_TRUE(result->task.distance_matrix.isApprox(jacobian.topRows(3), kTolerance));
    EXPECT_TRUE(result->task.distance_weights.isApprox(Eigen::Vector3d::Ones(), kTolerance));
    EXPECT_NEAR(result->task.distance_reference[0], -0.075, kTolerance);
    EXPECT_NEAR(result->task.distance_reference[1], signed_omega * 0.5, kTolerance);
    EXPECT_NEAR(result->task.distance_reference[2], 0.0, kTolerance);
    EXPECT_NE(result->task.distance_reference[1], face_velocity.y())
      << "The carry follows the screen's own orbit radius, not the face velocity";

    Eigen::VectorXd j1_only = Eigen::VectorXd::Zero(6);
    j1_only[0] = signed_omega;
    const Eigen::Vector3d screen_velocity = jacobian.topRows(3) * j1_only;
    EXPECT_NEAR(screen_velocity.y(), result->task.distance_reference[1], kTolerance);
    EXPECT_NEAR((screen_velocity - result->task.distance_reference).dot(
        (screen - face).normalized()), -0.075, kTolerance);

    GazeKinematics gaze;
    gaze.pose = current;
    gaze.pose.translation() += Eigen::Vector3d(0.1, 0.0, 0.0);
    gaze.jacobian = revolute_z_jacobian(gaze.pose.translation(), axis_origin);
    const auto gaze_primary = makeTrackingVelocityTask(
      current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config, face_velocity,
      Eigen::Vector3d::Zero(), gaze);
    ASSERT_TRUE(gaze_primary.has_value());
    EXPECT_NEAR(gaze_primary->task.distance_reference[1], signed_omega * 0.5, kTolerance);
  }
}

TEST(TrackingVelocityTask, StationaryAlignedFarDistanceWithPhysicalJ1IsRadialFullCartesian)
{
  const Eigen::Vector3d axis_origin(0.0, 0.0, 1.68);
  const Eigen::Vector3d screen = axis_origin + Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d face = axis_origin + Eigen::Vector3d(4.0, 0.0, 0.0);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  const Eigen::MatrixXd jacobian = revolute_z_jacobian(screen, axis_origin);
  const auto result = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->task.primary_reference.head<2>().isZero(0.0));
  ASSERT_EQ(result->task.distance_matrix.rows(), 3);
  EXPECT_TRUE(result->task.distance_matrix.isApprox(jacobian.topRows(3), kTolerance));
  EXPECT_TRUE(result->task.distance_weights.isApprox(Eigen::Vector3d::Ones(), kTolerance));
  EXPECT_NEAR(result->face_distance_reference[0], -config.maximum_linear_reference_mps, kTolerance);
  EXPECT_TRUE(result->task.distance_reference.isApprox(
      Eigen::Vector3d(config.maximum_linear_reference_mps, 0.0, 0.0), kTolerance));
}

TEST(TrackingVelocityTask, RadialRequestIsUnchangedWhenSaturatedAndTangentialCarryIsAdded)
{
  constexpr double omega = 0.13;
  const Eigen::Vector3d axis_origin(0.0, 0.0, 1.68);
  const Eigen::Vector3d screen = axis_origin + Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d face = axis_origin + Eigen::Vector3d(4.0, 0.0, 0.0);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  const auto result = makeTrackingVelocityTask(
    current, revolute_z_jacobian(screen, axis_origin), screen, face,
    Eigen::Matrix3d::Identity(), config, Eigen::Vector3d(0.0, omega * 4.0, 0.0));
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result->face_distance_outside_band);
  EXPECT_NEAR(result->face_distance_reference[0], -config.maximum_linear_reference_mps,
    kTolerance);
  EXPECT_NEAR(result->task.distance_reference[0], config.maximum_linear_reference_mps,
    kTolerance);
  EXPECT_NEAR(result->task.distance_reference[1], omega * 0.5, kTolerance);
  EXPECT_NEAR(result->task.distance_reference[2], 0.0, kTolerance);
}

TEST(TrackingVelocityTask, FollowVelocityUsesEffectivePointingReferenceAndIsContinuous)
{
  const Eigen::Vector3d axis_origin(0.0, 0.0, 1.68);
  const Eigen::Vector3d screen = axis_origin + Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d face = axis_origin + Eigen::Vector3d(0.85, 0.0, 0.0);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  const Eigen::MatrixXd jacobian = revolute_z_jacobian(screen, axis_origin);
  TrackingVelocityTaskConfig config;
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;

  const auto quiet = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.0, 0.001, 0.0));
  ASSERT_TRUE(quiet.has_value());
  EXPECT_TRUE(quiet->task.primary_reference.head<2>().isZero(0.0));
  EXPECT_NEAR(quiet->task.distance_reference[1], 0.0, 0.0);

  config.pointing_deadband_rad = 0.01;
  constexpr double boundary_face_speed = 0.014;
  const auto below = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.0, boundary_face_speed - 1.0e-9, 0.0));
  const auto above = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.0, boundary_face_speed + 1.0e-9, 0.0));
  ASSERT_TRUE(below.has_value());
  ASSERT_TRUE(above.has_value());
  EXPECT_NEAR(below->task.distance_reference[1], 0.0, 0.0);
  EXPECT_GT(above->task.distance_reference[1], 0.0);
  EXPECT_LT(above->task.distance_reference[1] - below->task.distance_reference[1], 1.0e-8);
}

TEST(TrackingVelocityTask, FollowVelocityUsesShiftedAndRotatedJ1Axis)
{
  constexpr double omega = 0.13;
  const Eigen::Vector3d axis_origin(0.3, -0.2, 1.68);
  const Eigen::Matrix3d base_rotation = Eigen::AngleAxisd(
    0.5 * std::acos(-1.0), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d screen = axis_origin + base_rotation * Eigen::Vector3d(0.5, 0.0, 0.0);
  const Eigen::Vector3d face = axis_origin + base_rotation * Eigen::Vector3d(0.85, 0.0, 0.0);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  current.linear() = base_rotation;
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  const Eigen::Vector3d face_velocity =
    omega * Eigen::Vector3d::UnitZ().cross(face - axis_origin);
  const auto result = makeTrackingVelocityTask(
    current, revolute_z_jacobian(screen, axis_origin), screen, face,
    Eigen::Matrix3d::Identity(), config, face_velocity);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->task.distance_reference[0], -omega * 0.5, kTolerance);
  EXPECT_NEAR(result->task.distance_reference[1], -0.075, kTolerance);
  EXPECT_NEAR(result->task.distance_reference[2], 0.0, kTolerance);
}

TEST(TrackingVelocityTask, FollowRateUsesExistingAngularSpeedCapAtLowPointingAuthority)
{
  const Eigen::Vector3d screen(0.5, 0.0, 1.68);
  const Eigen::Vector3d face(0.85, 0.0, 1.68);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  jacobian.block<3, 1>(0, 0) = Eigen::Vector3d(0.0, -0.315, 0.0);
  jacobian.block<3, 1>(3, 0) = Eigen::Vector3d::UnitZ();
  const auto result = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.0, 100.0, 0.0));
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->task.primary_reference.head<2>().norm(),
    config.maximum_angular_reference_radps, kTolerance);
  EXPECT_NEAR(result->task.distance_reference[0], -0.075, kTolerance);
  EXPECT_NEAR(result->task.distance_reference[1], -0.315 *
    config.maximum_angular_reference_radps, kTolerance);
  EXPECT_NEAR(result->task.distance_reference[2], 0.0, kTolerance);
}

TEST(TrackingVelocityTask, PrismaticOrPointingDegenerateJ1FallsBackToRadialReference)
{
  const Eigen::Vector3d screen(0.5, 0.0, 1.68);
  const Eigen::Vector3d face(0.85, 0.0, 1.68);
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = screen;
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.face_distance_min_m = 0.4;
  config.face_distance_max_m = 0.6;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  jacobian.block<3, 1>(0, 0) = Eigen::Vector3d(0.0, 0.5, 0.0);
  jacobian.block<3, 1>(3, 0).setZero();
  auto result = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.0, 0.85 * 0.13, 0.0));
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->task.distance_reference.isApprox(
      Eigen::Vector3d(-0.075, 0.0, 0.0), kTolerance));
  EXPECT_TRUE(result->task.distance_reference.allFinite());

  jacobian.block<3, 1>(0, 0) = Eigen::Vector3d(0.0, -0.35, 0.0);
  jacobian.block<3, 1>(3, 0) = Eigen::Vector3d::UnitZ();
  result = makeTrackingVelocityTask(
    current, jacobian, screen, face, Eigen::Matrix3d::Identity(), config,
    Eigen::Vector3d(0.0, 0.85 * 0.13, 0.0));
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->task.distance_reference.isApprox(
      Eigen::Vector3d(-0.075, 0.0, 0.0), kTolerance));
  EXPECT_TRUE(result->task.distance_reference.allFinite());
}

TEST(TrackingVelocityTask, FaceEdgeRecoveryKeepsOnlyPrimaryEdgeRowAndZeroSecondary)
{
  HierarchicalVelocityTask face_task;
  face_task.primary_matrix.resize(3, 4);
  face_task.primary_matrix <<
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    -0.5, 0.25, 1.5, -2.0;
  face_task.primary_reference = Eigen::Vector3d(0.3, -0.4, 0.17);
  face_task.primary_weights = Eigen::Vector3d(1.0, 1.0, 0.35);
  face_task.secondary_matrix = Eigen::MatrixXd::Zero(3, 4);
  face_task.secondary_reference = Eigen::Vector3d::Ones();
  face_task.secondary_weights = Eigen::Vector3d::Ones();

  const auto recovery = makeFaceEdgeOnlyRecoveryTask(face_task, true);
  ASSERT_TRUE(recovery.has_value());
  ASSERT_EQ(recovery->primary_matrix.rows(), 1);
  ASSERT_EQ(recovery->primary_matrix.cols(), 4);
  EXPECT_TRUE(recovery->primary_matrix.row(0).isApprox(face_task.primary_matrix.row(2), 0.0));
  ASSERT_EQ(recovery->primary_reference.size(), 1);
  EXPECT_DOUBLE_EQ(recovery->primary_reference[0], face_task.primary_reference[2]);
  ASSERT_EQ(recovery->primary_weights.size(), 1);
  EXPECT_DOUBLE_EQ(recovery->primary_weights[0], face_task.primary_weights[2]);
  EXPECT_TRUE(recovery->secondary_matrix.isApprox(Eigen::Matrix4d::Identity(), 0.0));
  EXPECT_TRUE(recovery->secondary_reference.isZero(0.0));
  EXPECT_TRUE(recovery->secondary_weights.isOnes(0.0));
}

TEST(TrackingVelocityTask, FaceEdgeRecoveryRejectsNonFaceOrMalformedRows)
{
  HierarchicalVelocityTask face_task;
  face_task.primary_matrix = Eigen::MatrixXd::Ones(3, 6);
  face_task.primary_reference = Eigen::Vector3d::Zero();
  face_task.primary_weights = Eigen::Vector3d::Ones();
  EXPECT_FALSE(makeFaceEdgeOnlyRecoveryTask(face_task, false).has_value());

  face_task.primary_matrix.resize(2, 6);
  EXPECT_FALSE(makeFaceEdgeOnlyRecoveryTask(face_task, true).has_value());
  face_task.primary_matrix = Eigen::MatrixXd::Ones(3, 6);
  face_task.primary_weights[2] = 0.0;
  EXPECT_FALSE(makeFaceEdgeOnlyRecoveryTask(face_task, true).has_value());
}

TEST(TrackingVelocityTask, SeparatesTargetFeedforwardFromGainCorrectedError)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(0.0, 0.0, 1.0);
  current.linear() = upright_rotation(Eigen::Vector3d::UnitX());
  const Eigen::Vector3d face(1.0, 0.0, 1.0);
  TrackingVelocityTaskConfig config = zero_deadbands();
  config.orientation_gain = 4.0;
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(), face,
    Eigen::Matrix3d::Identity(), config, Eigen::Vector3d(0.0, 0.13, 0.0));
  ASSERT_TRUE(result.has_value());
  ASSERT_TRUE(result->has_face_primary_roll);
  EXPECT_TRUE(result->signed_face_primary_error_rad.isZero(kTolerance));
  // For +X normal, +Y face motion induces +Z angular feed-forward, which is
  // the screen-up primary coordinate. It is distinct from proportional error.
  EXPECT_NEAR(result->face_primary_feedforward_radps[0], 0.0, kTolerance);
  EXPECT_NEAR(result->face_primary_feedforward_radps[1], 0.13, kTolerance);
  EXPECT_NEAR(result->face_primary_feedforward_radps[2], 0.0, kTolerance);
}

TEST(TrackingVelocityTask, GazeDeadbandAppliesToCombinedFeedbackAndFeedforward)
{
  const Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  const Eigen::Vector3d face(1.0, 0.0, 0.0);
  TrackingVelocityTaskConfig config;
  config.orientation_gain = 4.0;
  config.pointing_deadband_rad = 0.01;

  const auto slow_feedforward = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(), face,
    Eigen::Matrix3d::Identity(), config, Eigen::Vector3d(0.0, 0.02, 0.0));
  ASSERT_TRUE(slow_feedforward.has_value());
  EXPECT_NEAR(slow_feedforward->face_primary_feedforward_radps.head<2>().norm(),
    0.02, kTolerance);
  EXPECT_TRUE(slow_feedforward->task.primary_reference.head<2>().isZero(kTolerance))
    << "Target motion inside the gaze quiet zone must not bypass it";

  const auto fast_feedforward = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(), face,
    Eigen::Matrix3d::Identity(), config, Eigen::Vector3d(0.0, 0.15, 0.0));
  ASSERT_TRUE(fast_feedforward.has_value());
  EXPECT_NEAR(fast_feedforward->task.primary_reference.head<2>().norm(), 0.11, 1.0e-6)
    << "The continuous deadzone should retain feed-forward above gain*deadband";

  config.pointing_deadband_rad = 0.0;
  const auto no_deadband = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(), face,
    Eigen::Matrix3d::Identity(), config, Eigen::Vector3d(0.0, 0.02, 0.0));
  ASSERT_TRUE(no_deadband.has_value());
  EXPECT_TRUE(no_deadband->task.primary_reference.head<2>().isApprox(
      no_deadband->face_primary_feedforward_radps.head<2>(), kTolerance));
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
  Eigen::Matrix<double, 2, 3> basis;
  basis.row(0) = current.rotation().col(1).transpose();
  basis.row(1) = current.rotation().col(2).transpose();
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
    const Eigen::Vector2d derivative =
      basis * (residual.axis() * (residual.angle() / delta));
    EXPECT_TRUE(result->task.primary_matrix.topRows(2).col(column).isApprox(
        derivative, 2.0e-6)) <<
      "column " << column << ": expected " << derivative.transpose() <<
      ", actual " << result->task.primary_matrix.topRows(2).col(column).transpose();
    Eigen::Isometry3d earlier = current;
    Eigen::Isometry3d later = current;
    earlier.linear() = Eigen::AngleAxisd(
      -delta * angular_velocity.norm(), angular_velocity.normalized()).toRotationMatrix() *
      current.linear();
    later.linear() = Eigen::AngleAxisd(
      delta * angular_velocity.norm(), angular_velocity.normalized()).toRotationMatrix() *
      current.linear();
    const double earlier_edge_sine = earlier.linear()(2, 1);
    const double later_edge_sine = later.linear()(2, 1);
    const double edge_sine_rate = (later_edge_sine - earlier_edge_sine) / (2.0 * delta);
    EXPECT_NEAR(result->task.primary_matrix(2, column), edge_sine_rate, 2.0e-6);
    EXPECT_TRUE(result->task.secondary_matrix.col(column).isApprox(
        jacobian.col(column).head<3>(), 2.0e-6));
  }
}

TEST(TrackingVelocityTask, MovingFaceFeedForwardMatchesPointingFiniteDifference)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.translation() = Eigen::Vector3d(-0.1, 0.2, 1.0);
  const Eigen::Vector3d face(0.8, 0.6, 1.7);
  const Eigen::Vector3d face_velocity(0.12, -0.20, 0.09);
  current.linear() = upright_rotation(face - current.translation());
  const TrackingVelocityTaskConfig config = zero_deadbands();
  const auto result = makeTrackingVelocityTask(
    current, Eigen::MatrixXd::Identity(6, 6), current.translation(), face,
    Eigen::Matrix3d::Identity(), config, face_velocity);
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
    0.0);
  EXPECT_TRUE(result->task.primary_reference.isApprox(expected, 1.0e-8)) <<
    "expected " << expected.transpose() <<
    ", actual " << result->task.primary_reference.transpose();
  EXPECT_NEAR(result->task.primary_reference[2], expected[2], 1.0e-8);
  EXPECT_NEAR(expected[2], 0.0, 1.0e-12);

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
  // Screen-normal feedback and roll retain their separate historical caps;
  // the full 3D orientation vector can therefore exceed one cap in norm.
  EXPECT_NEAR(result->task.primary_reference.head<2>().norm(),
    config.maximum_angular_reference_radps, kTolerance);
  EXPECT_LE(std::abs(result->task.primary_reference[2]),
    config.maximum_angular_reference_radps + kTolerance);
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
  EXPECT_EQ(result->task.primary_matrix.rows(), 2);
  EXPECT_EQ(result->task.secondary_matrix.rows(), 4);
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

TEST(TrackingVelocityTask, VerticalTargetKeepsPhysicalUprightRollObservable)
{
  for (const double x : {-1.0e-9, 0.0, 1.0e-9}) {
    const auto result = makeTrackingVelocityTask(
      Eigen::Isometry3d::Identity(), Eigen::MatrixXd::Identity(6, 6), Eigen::Vector3d::Zero(),
      Eigen::Vector3d(x, 0.0, 1.0), Eigen::Matrix3d::Identity(),
      TrackingVelocityTaskConfig{}, Eigen::Vector3d(0.1, 0.1, 0.0));
    ASSERT_TRUE(result);
    EXPECT_EQ(result->task.primary_matrix.rows(), 3);
    EXPECT_TRUE(result->task.primary_matrix.allFinite());
    EXPECT_LE(result->task.primary_matrix.norm(), 3.0);
    EXPECT_GT(result->task.primary_reference.norm(), 0.1);
    EXPECT_GT(result->task.primary_matrix.row(2).norm(), 0.1);
    EXPECT_NEAR(result->task.primary_reference[2], 0.0, 1.0e-6);
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
  EXPECT_EQ(result->task.primary_matrix.rows(), 3);
  EXPECT_EQ(result->task.secondary_matrix.rows(), 3);
  EXPECT_NEAR(result->pointing_error_rad, 0.0, kTolerance);
  EXPECT_TRUE(result->task.primary_reference.isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_reference.isZero(kTolerance));
  EXPECT_TRUE(result->task.secondary_matrix.isApprox(
      Eigen::MatrixXd::Identity(6, 6).topRows(3), kTolerance));
  const auto upright_roll = makeUprightRollPathTask(
    monitor, Eigen::MatrixXd::Identity(6, 6), Eigen::VectorXd::Zero(6));
  ASSERT_TRUE(upright_roll);
  EXPECT_TRUE(result->task.primary_matrix.row(2).isApprox(
      upright_roll->task.primary_matrix.row(0), kTolerance));
  EXPECT_NEAR(result->roll_error_rad, std::abs(upright_roll->roll_error_rad), kTolerance);
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
    EXPECT_LT((result->task.primary_matrix.topRows(2).col(column) - derivative).norm(),
      1.0e-8) << column;
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

TEST(UprightRollPathTask, TracksProjectedWorldUpAndMatchesFiniteDifference)
{
  const Eigen::Vector3d normal = Eigen::Vector3d(0.6, 0.0, 0.8).normalized();
  const auto upright = pointingRotation(normal);
  ASSERT_TRUE(upright.has_value());
  const double physical_roll = 0.12;
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = Eigen::AngleAxisd(physical_roll, normal).toRotationMatrix() * *upright;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  const Eigen::VectorXd path_velocity = Eigen::VectorXd::LinSpaced(6, -0.3, 0.2);
  TrackingVelocityTaskConfig config = zero_deadbands();

  const auto task = makeUprightRollPathTask(current, jacobian, path_velocity, config);
  ASSERT_TRUE(task.has_value());
  EXPECT_NEAR(task->roll_error_rad, -physical_roll, 1.0e-12);
  EXPECT_EQ(task->task.primary_matrix.rows(), 1);
  EXPECT_EQ(task->task.secondary_matrix.rows(), 6);
  EXPECT_TRUE(task->task.primary_matrix.leftCols(3).isZero(kTolerance));
  EXPECT_TRUE(task->task.secondary_matrix.isIdentity(kTolerance));
  EXPECT_TRUE(task->task.secondary_reference.isApprox(path_velocity, kTolerance));
  EXPECT_NEAR(task->task.primary_reference[0], -config.orientation_gain * physical_roll,
    kTolerance);

  const Eigen::Vector3d angular_velocity(0.17, -0.23, 0.31);
  constexpr double delta = 1.0e-6;
  const auto error_at = [&](const double step) {
      Eigen::Isometry3d moved = current;
      moved.linear() = Eigen::AngleAxisd(
        step * angular_velocity.norm(), angular_velocity.normalized()).toRotationMatrix() *
        current.linear();
      const auto moved_upright = pointingRotation(moved.linear().col(0));
      if (!moved_upright) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      return std::atan2(
        moved.linear().col(0).dot(
          moved.linear().col(2).cross(moved_upright->col(2))),
        moved.linear().col(2).dot(moved_upright->col(2)));
    };
  const double measured_rate = (error_at(delta) - error_at(-delta)) / (2.0 * delta);
  const double predicted_rate = -task->task.primary_matrix.rightCols(3).row(0).dot(
    angular_velocity);
  EXPECT_NEAR(measured_rate, predicted_rate, 2.0e-6);
}

TEST(UprightRollPathTask, WorldYawReorientationPreservesUprightAndAddsNoYawDemand)
{
  const Eigen::Vector3d normal = Eigen::Vector3d(0.5, -0.2, 0.84).normalized();
  const auto upright = pointingRotation(normal);
  ASSERT_TRUE(upright.has_value());
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = Eigen::AngleAxisd(1.1, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    *upright;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  const auto task = makeUprightRollPathTask(
    current, jacobian, Eigen::VectorXd::Zero(6), zero_deadbands());
  ASSERT_TRUE(task.has_value());
  EXPECT_NEAR(task->roll_error_rad, 0.0, kTolerance);
  EXPECT_NEAR(task->task.primary_reference[0], 0.0, kTolerance);
  EXPECT_NEAR(task->task.primary_matrix(0, 5), 0.0, 1.0e-12);
}

TEST(UprightRollPathTask, RollRemainsPrimaryAgainstConflictingPathAndXYZStaysSecondary)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()).toRotationMatrix();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 2);
  jacobian(3, 0) = 1.0;  // q0 is a pure screen-roll joint.
  jacobian(1, 1) = 1.0;  // q1 is pure XYZ translation.
  jacobian(5, 1) = 1.0;  // q1 also changes yaw, but not upright roll here.
  const Eigen::Vector2d conflicting_path_velocity(0.4, -0.7);
  const auto task = makeUprightRollPathTask(
    current, jacobian, conflicting_path_velocity, zero_deadbands());
  ASSERT_TRUE(task.has_value());
  EXPECT_EQ(task->task.primary_matrix.rows(), 1);
  EXPECT_LT(task->task.primary_reference[0], 0.0);
  EXPECT_GT(task->task.secondary_reference[0], 0.0);
  EXPECT_TRUE(task->task.primary_matrix.isApprox(Eigen::Matrix<double, 1, 2>(1.0, 0.0),
      kTolerance));
  EXPECT_TRUE(task->task.secondary_matrix.isIdentity(kTolerance));
  EXPECT_TRUE(task->task.secondary_reference.isApprox(conflicting_path_velocity, kTolerance));
}

TEST(UprightRollPathTask, ExistingQpKeepsRollCorrectionAheadOfConflictingPathVelocity)
{
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitX()).toRotationMatrix();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 2);
  jacobian(3, 0) = 1.0;
  jacobian(1, 1) = 1.0;
  const auto task = makeUprightRollPathTask(
    current, jacobian, Eigen::Vector2d(0.4, 0.5), zero_deadbands());
  ASSERT_TRUE(task.has_value());

  JointMotionState state;
  state.position = Eigen::Vector2d::Zero();
  state.velocity = Eigen::Vector2d::Zero();
  state.acceleration = Eigen::Vector2d::Zero();
  JointMotionLimits limits;
  limits.lower_position = Eigen::Vector2d::Constant(-10.0);
  limits.upper_position = Eigen::Vector2d::Constant(10.0);
  limits.position_margin = Eigen::Vector2d::Zero();
  limits.max_velocity = Eigen::Vector2d::Ones();
  limits.max_acceleration = Eigen::Vector2d::Constant(2.0);
  limits.max_jerk = Eigen::Vector2d::Constant(20.0);
  limits.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-6}};
  HierarchicalVelocityQpConfig qp_config;
  qp_config.primary_preservation_tolerance = 2.0e-4;
  qp_config.solver_absolute_tolerance = 1.0e-5;
  qp_config.solver_relative_tolerance = 1.0e-5;
  const HierarchicalVelocityQp solver{qp_config};

  const QpResult result = solver.solve(task->task, state, limits);

  ASSERT_TRUE(result.command_available());
  ASSERT_TRUE(result.fully_solved());
  EXPECT_LT(result.achieved_primary_velocity[0], 0.0);
  EXPECT_LT(result.joint_velocity[0], 0.0);
  EXPECT_GT(result.joint_velocity[1], 0.0);
  EXPECT_LE(std::abs(result.joint_velocity[0]), 0.0021);
}

TEST(UprightRollPathTask, ExistingQpAllowsYawPathWhileActualScreenRemainsUpright)
{
  const Eigen::Vector3d normal = Eigen::Vector3d(0.5, 0.0, 0.866025403784).normalized();
  const auto upright = pointingRotation(normal);
  ASSERT_TRUE(upright.has_value());
  Eigen::Isometry3d current = Eigen::Isometry3d::Identity();
  current.linear() = *upright;
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 1);
  jacobian(5, 0) = 1.0;  // This joint rotates around world +Z.
  JointMotionState state;
  state.position = Eigen::VectorXd::Zero(1);
  state.velocity = Eigen::VectorXd::Zero(1);
  state.acceleration = Eigen::VectorXd::Zero(1);
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(1, -10.0);
  limits.upper_position = Eigen::VectorXd::Constant(1, 10.0);
  limits.position_margin = Eigen::VectorXd::Zero(1);
  limits.max_velocity = Eigen::VectorXd::Ones(1);
  limits.max_acceleration = Eigen::VectorXd::Constant(1, 2.0);
  limits.max_jerk = Eigen::VectorXd::Constant(1, 20.0);
  limits.numerical_policy = NumericalPolicy{
    Radians{1.0e-12}, RadiansPerSecond{1.0e-12},
    RadiansPerSecondSquared{1.0e-12}, RadiansPerSecondCubed{1.0e-12},
    SolverResidual{1.0e-6}};
  const HierarchicalVelocityQp solver{};
  constexpr double period = 0.01;
  double accumulated_yaw = 0.0;
  double maximum_roll_error = 0.0;

  for (int tick = 0; tick < 100; ++tick) {
    const auto task = makeUprightRollPathTask(
      current, jacobian, Eigen::VectorXd::Constant(1, 0.13), zero_deadbands());
    ASSERT_TRUE(task.has_value()) << "tick " << tick;
    const QpResult result = solver.solve(task->task, state, limits);
    ASSERT_TRUE(result.command_available()) << "tick " << tick;
    accumulated_yaw += result.joint_velocity[0] * period;
    current.linear() = Eigen::AngleAxisd(
      result.joint_velocity[0] * period, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      current.linear();
    state.position[0] += result.joint_velocity[0] * period;
    state.acceleration[0] =
      (result.joint_velocity[0] - state.velocity[0]) / period;
    state.velocity = result.joint_velocity;
    const auto measured = makeUprightRollPathTask(
      current, jacobian, Eigen::VectorXd::Zero(1), zero_deadbands());
    ASSERT_TRUE(measured.has_value()) << "tick " << tick;
    maximum_roll_error = std::max(maximum_roll_error, std::abs(measured->roll_error_rad));
  }

  EXPECT_GT(accumulated_yaw, 0.05);
  EXPECT_LT(maximum_roll_error, 1.0e-8);
}

TEST(UprightRollPathTask, RejectsUndefinedRollAndInvalidInputs)
{
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Identity(6, 6);
  const Eigen::VectorXd path_velocity = Eigen::VectorXd::Zero(6);
  Eigen::Isometry3d vertical = Eigen::Isometry3d::Identity();
  vertical.linear() = Eigen::Quaterniond::FromTwoVectors(
    Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  EXPECT_FALSE(makeUprightRollPathTask(vertical, jacobian, path_velocity));

  Eigen::Isometry3d near_vertical = Eigen::Isometry3d::Identity();
  near_vertical.linear() = Eigen::AngleAxisd(
    std::acos(0.05), Eigen::Vector3d::UnitY()).toRotationMatrix();
  EXPECT_FALSE(makeUprightRollPathTask(near_vertical, jacobian, path_velocity));
  EXPECT_FALSE(makeUprightRollPathTask(
      Eigen::Isometry3d::Identity(), Eigen::MatrixXd::Zero(5, 6), path_velocity));
  EXPECT_FALSE(makeUprightRollPathTask(
      Eigen::Isometry3d::Identity(), jacobian, Eigen::VectorXd::Zero(5)));
  Eigen::VectorXd invalid_path = path_velocity;
  invalid_path[2] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(makeUprightRollPathTask(
      Eigen::Isometry3d::Identity(), jacobian, invalid_path));
  TrackingVelocityTaskConfig invalid_config;
  invalid_config.orientation_gain = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(makeUprightRollPathTask(
    Eigen::Isometry3d::Identity(), jacobian, path_velocity, invalid_config));
}

TEST(HorizontalEdgePathTask, UsesSineRateRowAndKeepsHighPitchObservable)
{
  constexpr double theta = 0.05;
  Eigen::Isometry3d screen_pose = Eigen::Isometry3d::Identity();
  screen_pose.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitX()).toRotationMatrix();
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 1);
  jacobian(3, 0) = 1.0;
  const Eigen::VectorXd path_reference = Eigen::VectorXd::Constant(1, 0.2);
  TrackingVelocityTaskConfig config;
  config.orientation_gain = 4.0;
  config.roll_deadband_rad = std::asin(0.01);
  config.maximum_angular_reference_radps = 0.70;

  const auto task = makeHorizontalEdgePathTask(
    screen_pose, jacobian, path_reference, config);

  ASSERT_TRUE(task);
  EXPECT_NEAR(task->signed_edge_sine, std::sin(theta), 1.0e-12);
  EXPECT_NEAR(task->edge_tilt_error_rad, -theta, 1.0e-12);
  EXPECT_NEAR(task->task.primary_matrix(0, 0), std::cos(theta), 1.0e-12);
  EXPECT_NEAR(task->task.primary_reference[0],
    -config.orientation_gain * (std::sin(theta) - std::sin(config.roll_deadband_rad)),
    1.0e-12);
  EXPECT_TRUE(task->task.secondary_reference.isApprox(path_reference));

  constexpr double step = 1.0e-7;
  Eigen::Isometry3d perturbed = screen_pose;
  perturbed.linear() = Eigen::AngleAxisd(theta + step, Eigen::Vector3d::UnitX())
    .toRotationMatrix();
  const auto perturbed_task = makeHorizontalEdgePathTask(
    perturbed, jacobian, path_reference, config);
  ASSERT_TRUE(perturbed_task);
  EXPECT_NEAR(
    (perturbed_task->signed_edge_sine - task->signed_edge_sine) / step,
    task->task.primary_matrix(0, 0), 1.0e-7);

  Eigen::Isometry3d high_pitch_pose = Eigen::Isometry3d::Identity();
  constexpr double kPi = 3.14159265358979323846;
  high_pitch_pose.linear() = Eigen::AngleAxisd(
    85.0 * kPi / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const auto high_pitch_task = makeHorizontalEdgePathTask(
    high_pitch_pose, jacobian, path_reference, config);
  ASSERT_TRUE(high_pitch_task);
  EXPECT_NEAR(high_pitch_task->edge_tilt_error_rad, 0.0, 1.0e-12);
}

TEST(HorizontalEdgeUpright, EnforcesCapAndRejectsInvalidGeometry)
{
  constexpr double kTilt = 5.1 * 3.14159265358979323846 / 180.0;
  const Eigen::Matrix3d edge_rotation =
    Eigen::AngleAxisd(kTilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
  const double exact_sine = edge_rotation(2, 1);

  EXPECT_TRUE(evaluateHorizontalEdgeUprightWithSineCap(edge_rotation, exact_sine));
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(
      edge_rotation, std::nextafter(exact_sine, 0.0)));
  EXPECT_TRUE(evaluateHorizontalEdgeUprightWithSineCap(
      edge_rotation, std::nextafter(exact_sine, 1.0)));

  const Eigen::Matrix3d flipped_up =
    Eigen::AngleAxisd(3.14159265358979323846, Eigen::Vector3d::UnitY()).toRotationMatrix();
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(flipped_up, 0.0));

  Eigen::Matrix3d reflection = Eigen::Matrix3d::Identity();
  reflection(2, 2) = -1.0;
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(reflection, 1.0));
  Eigen::Matrix3d non_orthogonal = Eigen::Matrix3d::Identity();
  non_orthogonal(0, 1) = 0.01;
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(non_orthogonal, 1.0));
  Eigen::Matrix3d non_finite = Eigen::Matrix3d::Identity();
  non_finite(1, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(non_finite, 1.0));

  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(edge_rotation, -0.01));
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(edge_rotation, 1.01));
  EXPECT_FALSE(evaluateHorizontalEdgeUprightWithSineCap(
      edge_rotation, std::numeric_limits<double>::infinity()));
}

}  // namespace
}  // namespace face_tracking_arm::control
