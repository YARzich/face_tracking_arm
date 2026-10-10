// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "face_tracking_arm/compact_upright.hpp"
#include "face_tracking_arm/pointing_geometry.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

TEST(CompactUprightTest, ReportsSignedPitchAndUprightRoll)
{
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(-0.4,
      Eigen::Vector3d::UnitY()).toRotationMatrix();
  const auto pose = evaluateCompactUprightPose(rotation);

  ASSERT_TRUE(pose);
  EXPECT_NEAR(pose->signed_pitch_rad, 0.4, 1.0e-12);
  EXPECT_NEAR(pose->signed_roll_error_rad, 0.0, 1.0e-12);
  EXPECT_NEAR(pose->horizontal_projection_norm, std::cos(0.4), 1.0e-12);
}

TEST(CompactUprightTest, RollSignMatchesExistingWorldUpCorrectionConvention)
{
  const Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.3,
      Eigen::Vector3d::UnitX()).toRotationMatrix();
  const auto pose = evaluateCompactUprightPose(rotation);

  ASSERT_TRUE(pose);
  EXPECT_NEAR(pose->signed_roll_error_rad, -0.3, 1.0e-12);
}

TEST(CompactUprightTest, PhysicalPitchAndRollStayCorrectAcrossWorldYaw)
{
  constexpr double kPitch = 37.0 * kPi / 180.0;
  constexpr double kRoll = 4.0 * kPi / 180.0;
  for (int yaw_step = 0; yaw_step < 24; ++yaw_step) {
    const double yaw = static_cast<double>(yaw_step) * (2.0 * kPi / 24.0);
    for (const double signed_pitch : {-kPitch, -0.5 * kPitch, 0.0, 0.5 * kPitch, kPitch}) {
      const Eigen::Vector3d normal(
        std::cos(signed_pitch) * std::cos(yaw),
        std::cos(signed_pitch) * std::sin(yaw), std::sin(signed_pitch));
      const auto upright = pointingRotation(normal);
      ASSERT_TRUE(upright);
      for (const double signed_roll : {-kRoll, 0.0, kRoll}) {
        const Eigen::Matrix3d rotation = *upright *
          Eigen::AngleAxisd(signed_roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const auto pose = evaluateCompactUprightPose(rotation);
        ASSERT_TRUE(pose);
        EXPECT_NEAR(pose->signed_pitch_rad, signed_pitch, 1.0e-12);
        EXPECT_NEAR(pose->signed_roll_error_rad, -signed_roll, 1.0e-12);
        EXPECT_LE(std::abs(pose->signed_pitch_rad), kPitch + 1.0e-12);
        EXPECT_LE(std::abs(pose->signed_roll_error_rad), kRoll + 1.0e-12);
      }
    }
  }
}

TEST(CompactUprightTest, PhysicalDomainFitsMoveItXyzGuideForArbitraryYaw)
{
  const double pitch_limit = 37.0 * kPi / 180.0;
  const double roll_limit = 5.1 * kPi / 180.0;
  const auto bounds = compactOrientationGuideBounds(pitch_limit, roll_limit);
  ASSERT_TRUE(bounds);
  const double target_yaw = 0.73;
  const Eigen::Matrix3d target_rotation =
    Eigen::AngleAxisd(target_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  for (int yaw_step = 0; yaw_step < 36; ++yaw_step) {
    const double yaw = target_yaw + static_cast<double>(yaw_step) * (2.0 * kPi / 36.0);
    for (int pitch_step = -10; pitch_step <= 10; ++pitch_step) {
      const double pitch = pitch_limit * static_cast<double>(pitch_step) / 10.0;
      const Eigen::Vector3d normal(
        std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch));
      const auto upright = pointingRotation(normal);
      ASSERT_TRUE(upright);
      for (int roll_step = -5; roll_step <= 5; ++roll_step) {
        const double roll = roll_limit * static_cast<double>(roll_step) / 5.0;
        const Eigen::Matrix3d actual_rotation = *upright *
          Eigen::AngleAxisd(-roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const auto pose = evaluateCompactUprightPose(actual_rotation);
        ASSERT_TRUE(pose);
        ASSERT_LE(std::abs(pose->signed_pitch_rad), pitch_limit + 1.0e-12);
        ASSERT_LE(std::abs(pose->signed_roll_error_rad), roll_limit + 1.0e-12);

        // MoveIt's XYZ parameterization decomposes target^-1 * actual. Its
        // tolerance region is twice the message tolerances on each axis.
        const Eigen::Matrix3d relative = target_rotation.transpose() * actual_rotation;
        const double moveit_x = std::atan2(-relative(1, 2), relative(2, 2));
        const double moveit_y = std::asin(std::clamp(relative(0, 2), -1.0, 1.0));
        const double moveit_z = std::atan2(-relative(0, 1), relative(0, 0));
        EXPECT_LE(std::abs(moveit_x), 2.0 * bounds->absolute_x_axis_tolerance_rad + 1.0e-12);
        EXPECT_LE(std::abs(moveit_y), 2.0 * bounds->absolute_y_axis_tolerance_rad + 1.0e-12);
        EXPECT_LE(std::abs(moveit_z), 2.0 * kPi);
      }
    }
  }
}

TEST(CompactUprightTest, PitchGradientMatchesCentralDifference)
{
  const Eigen::Vector3d world_axis = Eigen::Vector3d(0.3, -0.4, 0.7).normalized();
  const Eigen::Matrix3d initial_rotation = Eigen::AngleAxisd(0.61,
      Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d initial_normal = initial_rotation.col(0);
  Eigen::MatrixXd angular_jacobian(3, 1);
  angular_jacobian.col(0) = world_axis;
  const auto gradient = signedScreenPitchGradient(initial_normal, angular_jacobian);

  ASSERT_TRUE(gradient);
  constexpr double kStep = 1.0e-6;
  const auto pitch_at = [&initial_rotation, &world_axis](const double q) {
      const Eigen::Matrix3d rotation =
        Eigen::AngleAxisd(q, world_axis).toRotationMatrix() * initial_rotation;
      const auto pose = evaluateCompactUprightPose(rotation);
      return pose ? pose->signed_pitch_rad : std::numeric_limits<double>::quiet_NaN();
    };
  const double finite_difference = (pitch_at(kStep) - pitch_at(-kStep)) / (2.0 * kStep);
  EXPECT_NEAR((*gradient)[0], finite_difference, 1.0e-7);
}

TEST(CompactUprightTest, RollErrorGradientMatchesSpatialCentralDifference)
{
  const Eigen::Matrix3d initial_rotation =
    (Eigen::AngleAxisd(0.73, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.42, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.17, Eigen::Vector3d::UnitX())).toRotationMatrix();
  Eigen::Matrix<double, 3, 6> angular_jacobian;
  angular_jacobian <<
    0.3, -0.7, 0.2, 0.5, -0.1, 0.9,
    0.6, 0.4, -0.8, 0.3, 0.2, -0.5,
    -0.2, 0.1, 0.7, -0.6, 0.8, 0.4;
  const auto gradient = signedScreenRollErrorGradient(initial_rotation, angular_jacobian);

  ASSERT_TRUE(gradient);
  constexpr double kStep = 1.0e-6;
  for (Eigen::Index column = 0; column < angular_jacobian.cols(); ++column) {
    const Eigen::Vector3d axis = angular_jacobian.col(column);
    const auto roll_at = [&initial_rotation, &axis](const double q) {
        const Eigen::Matrix3d rotation =
          Eigen::AngleAxisd(q * axis.norm(), axis.normalized()).toRotationMatrix() *
          initial_rotation;
        const auto pose = evaluateCompactUprightPose(rotation);
        return pose ? pose->signed_roll_error_rad :
               std::numeric_limits<double>::quiet_NaN();
      };
    const double plus = roll_at(kStep);
    const double minus = roll_at(-kStep);
    const double finite_difference = std::remainder(plus - minus, 2.0 * kPi) / (2.0 * kStep);
    EXPECT_NEAR((*gradient)[column], finite_difference, 2.0e-7) << "column=" << column;
  }
}

TEST(CompactUprightTest, RejectsPoleAndInvalidInputs)
{
  const Eigen::Matrix3d near_pole = Eigen::AngleAxisd(0.5 * kPi - 0.05,
      Eigen::Vector3d::UnitY()).toRotationMatrix();
  EXPECT_FALSE(evaluateCompactUprightPose(near_pole));
  EXPECT_FALSE(evaluateCompactUprightPose(Eigen::Matrix3d::Constant(
      std::numeric_limits<double>::quiet_NaN())));

  Eigen::MatrixXd angular_jacobian = Eigen::MatrixXd::Identity(3, 3);
  EXPECT_FALSE(signedScreenPitchGradient(near_pole.col(0), angular_jacobian));
  EXPECT_FALSE(signedScreenPitchGradient(Eigen::Vector3d::UnitX(), Eigen::MatrixXd::Zero(2, 3)));
  EXPECT_FALSE(signedScreenRollErrorGradient(near_pole, angular_jacobian));
  EXPECT_FALSE(signedScreenPitchGradient(
      Eigen::Vector3d::UnitX(), angular_jacobian,
      std::numeric_limits<double>::infinity()));
  const auto broad_guide = compactOrientationGuideBounds(0.4, 0.5 * kPi);
  ASSERT_TRUE(broad_guide);
  EXPECT_DOUBLE_EQ(broad_guide->absolute_x_axis_tolerance_rad, kPi);
  EXPECT_DOUBLE_EQ(broad_guide->absolute_y_axis_tolerance_rad, 0.5 * kPi);
}

}  // namespace
}  // namespace face_tracking_arm::control
