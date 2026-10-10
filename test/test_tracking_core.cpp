// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include "face_tracking_arm/tracking_core.hpp"
#include "face_tracking_arm/face_target_age_policy.hpp"

namespace face_tracking_arm::tracking
{
namespace
{

constexpr double kTolerance = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;
constexpr RosTimeNanoseconds kFreshness = 200'000'000;
constexpr RosTimeNanoseconds kReturnDelay = 2'000'000'000;

TrackingConfig make_config()
{
  TrackingConfig config;
  config.freshness_ns = kFreshness;
  config.return_delay_ns = kReturnDelay;
  return config;
}

FaceSample make_sample(
  const Eigen::Vector3d & face,
  const RosTimeNanoseconds stamp_ns,
  const Eigen::Vector3d & base = Eigen::Vector3d::Zero(),
  const Eigen::Vector3d & reach_center = Eigen::Vector3d::Zero())
{
  FaceSample sample;
  sample.face_in_planning_frame = face;
  sample.base_in_planning_frame = base;
  sample.reach_center_in_planning_frame = reach_center;
  sample.measurement_stamp_ns = stamp_ns;
  sample.frame_id = "world";
  return sample;
}

ActualTcpPose make_actual_pose(
  const Eigen::Vector3d & position,
  const Eigen::Quaterniond & orientation,
  const RosTimeNanoseconds stamp_ns)
{
  ActualTcpPose actual;
  actual.pose.position = position;
  actual.pose.orientation = orientation;
  actual.measurement_stamp_ns = stamp_ns;
  return actual;
}

ActualTcpPose make_actual_pose(
  const Eigen::Vector3d & position,
  const double yaw_rad,
  const RosTimeNanoseconds stamp_ns)
{
  return make_actual_pose(
    position,
    Eigen::Quaterniond{Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ())},
    stamp_ns);
}

FaceGeometry geometry(const GeometryResult & result)
{
  if (!std::holds_alternative<FaceGeometry>(result)) {
    ADD_FAILURE() << "Expected accepted face geometry";
    return {};
  }
  return std::get<FaceGeometry>(result);
}

RejectReason rejection(const GeometryResult & result)
{
  if (!std::holds_alternative<RejectReason>(result)) {
    ADD_FAILURE() << "Expected rejected face geometry";
    return RejectReason::kNone;
  }
  return std::get<RejectReason>(result);
}

TargetCommand command(const TickResult & result)
{
  if (!result.command.has_value()) {
    ADD_FAILURE() << "Expected a target command";
    return {};
  }
  return *result.command;
}

TEST(TrackingGeometry, KeepsExactMinimumDistanceInsideReachEnvelope)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d base{0.0, 0.0, 0.75};
  const Eigen::Vector3d reach_center = base;
  const Eigen::Vector3d face{0.55, 0.16, 1.25};
  const Eigen::Vector3d direction = (face - base).normalized();
  const Eigen::Vector3d expected_position =
    face - config.minimum_face_distance_m * direction;

  const FaceGeometry target = geometry(
    compute_face_geometry(face, base, reach_center, config));

  EXPECT_FALSE(target.reach_limited);
  EXPECT_NEAR(target.face_distance_m, config.minimum_face_distance_m, kTolerance);
  EXPECT_NEAR(
    (face - target.monitor_pose.position).norm(),
    config.minimum_face_distance_m,
    kTolerance);
  EXPECT_TRUE(target.monitor_pose.position.isApprox(expected_position, kTolerance));
  EXPECT_LE(
    (target.monitor_pose.position - reach_center).norm(),
    config.safe_reach_radius_m + kTolerance);
}

TEST(TrackingGeometry, BuildsFullLookAtOrientationWithGravityReferencedUp)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d base{0.2, -0.3, 0.75};
  const Eigen::Vector3d face{0.72, 0.08, 1.20};
  const Eigen::Vector3d direction = (face - base).normalized();
  const Eigen::Vector3d expected_up =
    (Eigen::Vector3d::UnitZ() - direction.z() * direction).normalized();

  const FaceGeometry target = geometry(
    compute_face_geometry(face, base, base, config));
  const Eigen::Quaterniond orientation = target.monitor_pose.orientation;

  EXPECT_NEAR(orientation.norm(), 1.0, kTolerance);
  EXPECT_TRUE((orientation * Eigen::Vector3d::UnitX()).isApprox(
    direction, kTolerance));
  EXPECT_TRUE((orientation * Eigen::Vector3d::UnitZ()).isApprox(
    expected_up, kTolerance));
  EXPECT_TRUE((orientation * Eigen::Vector3d::UnitY()).isApprox(
    expected_up.cross(direction), kTolerance));
  EXPECT_LT(target.monitor_pose.position.z(), face.z());
}

TEST(TrackingGeometry, ReachClampPreservesMinimumDistanceAndReportsLimiting)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  const Eigen::Vector3d face{1.0, 0.2, 0.3};
  const Eigen::Vector3d expected_position =
    config.safe_reach_radius_m * face.normalized();

  const FaceGeometry target = geometry(
    compute_face_geometry(face, origin, origin, config));

  EXPECT_TRUE(target.reach_limited);
  EXPECT_NEAR(target.monitor_pose.position.norm(), config.safe_reach_radius_m, kTolerance);
  EXPECT_TRUE(target.monitor_pose.position.isApprox(expected_position, kTolerance));
  EXPECT_GT(target.face_distance_m, config.minimum_face_distance_m);
  EXPECT_NEAR(
    target.face_distance_m,
    (face - target.monitor_pose.position).norm(),
    kTolerance);
}

TEST(TrackingGeometry, ChoosesClosestPointInsideRealOffsetReachSphere)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d base{0.0, 0.0, 0.75};
  const Eigen::Vector3d reach_center{0.0, 0.0, 0.9935};
  const Eigen::Vector3d face{0.58, -0.43, 1.88};
  const Eigen::Vector3d direction = (face - reach_center).normalized();
  const Eigen::Vector3d ideal_position =
    face - config.minimum_face_distance_m * direction;

  ASSERT_LT((base - reach_center).norm(), config.safe_reach_radius_m);
  ASSERT_GT(
    (ideal_position - reach_center).norm(), config.safe_reach_radius_m);

  const FaceGeometry target = geometry(
    compute_face_geometry(face, base, reach_center, config));

  EXPECT_TRUE(target.reach_limited);
  EXPECT_NEAR(
    (target.monitor_pose.position - reach_center).norm(),
    config.safe_reach_radius_m,
    kTolerance);
  EXPECT_GT(target.face_distance_m, config.minimum_face_distance_m);
  EXPECT_TRUE((target.monitor_pose.position - reach_center).normalized().isApprox(
    direction, kTolerance));
  EXPECT_NEAR(
    target.face_distance_m,
    (face - reach_center).norm() - config.safe_reach_radius_m,
    kTolerance);
}

TEST(TrackingGeometry, KeepsCloseAndVerticalFacesButRejectsCoincidentDirection)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();

  EXPECT_EQ(
    rejection(compute_face_geometry(
      Eigen::Vector3d{1.0e-7, 0.0, 0.0}, origin, origin, config)),
    RejectReason::kDegenerateDirection);
  for (const Eigen::Vector3d face : {Eigen::Vector3d(0.30, 0.0, 0.0),
      Eigen::Vector3d(0.0, 0.0, 1.0)})
  {
    const auto target = geometry(compute_face_geometry(face, origin, origin, config));
    EXPECT_TRUE(target.monitor_pose.orientation.coeffs().allFinite());
    EXPECT_TRUE((target.monitor_pose.orientation * Eigen::Vector3d::UnitX()).isApprox(
        (face - target.monitor_pose.position).normalized(), kTolerance));
  }
}

TEST(TrackingGeometry, OffsetSphereDoesNotRequireAnIntersectionWithBaseRay)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d base = Eigen::Vector3d::Zero();
  const Eigen::Vector3d reach_center{0.0, 1.0, 0.0};

  const Eigen::Vector3d face{1.0, 0.0, 0.0};
  const FaceGeometry target = geometry(compute_face_geometry(face, base, reach_center, config));
  const Eigen::Vector3d expected =
    reach_center + config.safe_reach_radius_m * (face - reach_center).normalized();
  EXPECT_TRUE(target.monitor_pose.position.isApprox(expected, kTolerance));
  EXPECT_NEAR(
    target.face_distance_m,
    (face - reach_center).norm() - config.safe_reach_radius_m,
    kTolerance);
}

TEST(TrackingGeometry, KeepsPointingWhenPreferredDistanceIsUnreachable)
{
  TrackingGeometryConfig config;
  config.safe_reach_radius_m = 0.1;
  const Eigen::Vector3d base{-1.0, 0.0, 0.0};
  const Eigen::Vector3d center = Eigen::Vector3d::Zero();
  const auto target = geometry(compute_face_geometry(
      Eigen::Vector3d{0.15, 0.0, 0.0}, base, center, config));
  EXPECT_TRUE(target.reach_limited);
  EXPECT_NEAR(target.monitor_pose.position.x(), -0.1, kTolerance);
  EXPECT_NEAR(target.face_distance_m, 0.25, kTolerance);
  EXPECT_TRUE((target.monitor_pose.orientation * Eigen::Vector3d::UnitX()).isApprox(
      Eigen::Vector3d::UnitX(), kTolerance));
  EXPECT_EQ(
    rejection(compute_face_geometry(center, base, center, config)),
    RejectReason::kDegenerateDirection);
}

TEST(TrackingGeometry, ClosestTargetsAtHumanHeightStayOnTheCorrectSideAroundTheTable)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d base{0.0, 0.0, 0.75};
  const Eigen::Vector3d center{0.0, 0.0, 0.9935};
  for (int side = 0; side < 8; ++side) {
    const double azimuth = side * kPi / 4.0;
    const Eigen::Vector3d face{0.85 * std::cos(azimuth), 0.85 * std::sin(azimuth), 1.68};
    const FaceGeometry target = geometry(compute_face_geometry(face, base, center, config));
    EXPECT_TRUE(target.reach_limited);
    EXPECT_TRUE(target.monitor_pose.position.allFinite());
    EXPECT_TRUE(target.monitor_pose.orientation.coeffs().allFinite());
    EXPECT_GT(target.monitor_pose.position.head(2).dot(face.head(2)), 0.0);
    EXPECT_NEAR((target.monitor_pose.position - center).norm(), 0.42, kTolerance);
    EXPECT_NEAR(target.face_distance_m, 0.6726034275985042, kTolerance);
    EXPECT_NEAR(target.monitor_pose.orientation.norm(), 1.0, kTolerance);
    EXPECT_TRUE((target.monitor_pose.orientation * Eigen::Vector3d::UnitX()).isApprox(
        (face - target.monitor_pose.position).normalized(), kTolerance));
  }
}

TEST(TrackingGeometry, NearFacesNeverPullTheTargetInsideMinimumDistance)
{
  const TrackingGeometryConfig config;
  const Eigen::Vector3d base{-2.0, 0.0, 1.0};
  const Eigen::Vector3d center{0.0, 0.0, 1.0};
  for (const double range : {0.05, 0.25, 0.4, 0.5, 0.82, 1.0}) {
    const Eigen::Vector3d face = center + range * Eigen::Vector3d::UnitX();
    const FaceGeometry target = geometry(compute_face_geometry(face, base, center, config));
    EXPECT_LE((target.monitor_pose.position - center).norm(), 0.42 + kTolerance);
    EXPECT_GE((face - target.monitor_pose.position).norm(), 0.4 - kTolerance);
    EXPECT_NEAR(target.face_distance_m, std::max(0.4, range - 0.42), kTolerance);
  }
}

TEST(TrackingGeometry, RejectsNonFiniteInputAndInvalidConfiguration)
{
  const Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  Eigen::Vector3d non_finite_face{1.0, 0.0, 0.2};
  non_finite_face.x() = std::numeric_limits<double>::quiet_NaN();

  EXPECT_EQ(
    rejection(compute_face_geometry(
      non_finite_face, origin, origin, TrackingGeometryConfig{})),
    RejectReason::kNonFinite);

  TrackingGeometryConfig invalid;
  invalid.minimum_face_distance_m = 0.0;
  EXPECT_EQ(
    rejection(compute_face_geometry(
      Eigen::Vector3d{1.0, 0.0, 0.2}, origin, origin, invalid)),
    RejectReason::kInvalidConfiguration);

  invalid = TrackingGeometryConfig{};
  invalid.safe_reach_radius_m = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
    rejection(compute_face_geometry(
      Eigen::Vector3d{1.0, 0.0, 0.2}, origin, origin, invalid)),
    RejectReason::kInvalidConfiguration);

  invalid = TrackingGeometryConfig{};
  invalid.direction_epsilon_m = -1.0;
  EXPECT_EQ(
    rejection(compute_face_geometry(
      Eigen::Vector3d{1.0, 0.0, 0.2}, origin, origin, invalid)),
    RejectReason::kInvalidConfiguration);

  TrackingGeometryConfig legacy_config;
  legacy_config.up_projection_epsilon = 0.0;
  EXPECT_TRUE(std::holds_alternative<FaceGeometry>(compute_face_geometry(
      Eigen::Vector3d{1.0, 0.0, 0.2}, origin, origin, legacy_config)));
}

TEST(TrackingValidation, RejectsInvalidTimingAndPreservesLastAcceptedStamp)
{
  TrackingCore core{make_config()};
  EXPECT_TRUE(core.ingest_face(make_sample({0.7, 0.0, 0.2}, 100), 100).accepted());

  EXPECT_EQ(
    core.ingest_face(make_sample({0.8, 0.0, 0.2}, 100), 101).reason,
    RejectReason::kNotNewer);
  EXPECT_EQ(
    core.ingest_face(make_sample({0.8, 0.0, 0.2}, 99), 101).reason,
    RejectReason::kNotNewer);
  EXPECT_EQ(
    core.ingest_face(make_sample({0.8, 0.0, 0.2}, 200), 199).reason,
    RejectReason::kFutureStamp);

  FaceSample invalid = make_sample({0.8, 0.0, 0.2}, 200);
  invalid.reach_center_in_planning_frame.x() =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(core.ingest_face(invalid, 200).reason, RejectReason::kNonFinite);
  EXPECT_TRUE(core.ingest_face(make_sample({0.8, 0.0, 0.2}, 200), 200).accepted());
}

TEST(TrackingValidation, RejectsFramesAndStaleSamples)
{
  TrackingCore core{make_config()};

  FaceSample empty_frame = make_sample({0.7, 0.0, 0.2}, 1);
  empty_frame.frame_id.clear();
  EXPECT_EQ(core.ingest_face(empty_frame, 1).reason, RejectReason::kEmptyFrame);

  FaceSample wrong_frame = make_sample({0.7, 0.0, 0.2}, 2);
  wrong_frame.frame_id = "map";
  EXPECT_EQ(
    core.ingest_face(wrong_frame, 2).reason, RejectReason::kUnexpectedFrame);
  EXPECT_EQ(
    core.ingest_face(make_sample({0.7, 0.0, 0.2}, 10), 10 + kFreshness).reason,
    RejectReason::kStale);
}

TEST(TrackingPolicy, RunsFaceHoldRestAndFreshFacePreemption)
{
  TrackingCore core{make_config()};
  const TargetCommand initial = command(core.tick(10));
  EXPECT_EQ(initial.mode, TargetMode::kRest);
  EXPECT_TRUE(initial.pose.position.isApprox(Eigen::Vector3d{0.2, 0.0, 1.05}));
  EXPECT_FALSE(initial.source_face_stamp_ns.has_value());

  constexpr RosTimeNanoseconds first_stamp = 100;
  const IngestResult first_ingest =
    core.ingest_face(make_sample({0.7, 0.1, 0.2}, first_stamp), first_stamp);
  ASSERT_TRUE(first_ingest.accepted());
  const TargetCommand face = command(core.tick(first_stamp));
  EXPECT_EQ(face.mode, TargetMode::kFace);
  EXPECT_EQ(face.source_face_stamp_ns, first_stamp);
  EXPECT_TRUE(face.pose.position.isApprox(
    first_ingest.accepted_geometry->monitor_pose.position, kTolerance));

  const RosTimeNanoseconds hold_time = first_stamp + kFreshness;
  const ActualTcpPose actual = make_actual_pose(
    {0.35, 0.05, 0.95}, 0.25, hold_time);
  const TargetCommand hold = command(core.tick(hold_time, actual));
  EXPECT_EQ(hold.mode, TargetMode::kHold);
  EXPECT_TRUE(hold.pose.position.isApprox(actual.pose.position, kTolerance));

  const RosTimeNanoseconds rest_time = first_stamp + kReturnDelay;
  EXPECT_EQ(command(core.tick(rest_time, actual)).mode, TargetMode::kRest);

  const RosTimeNanoseconds second_stamp = rest_time + 1;
  ASSERT_TRUE(core.ingest_face(
      make_sample({0.6, -0.2, 0.3}, second_stamp), second_stamp).accepted());
  EXPECT_EQ(command(core.tick(second_stamp)).mode, TargetMode::kFace);
}

TEST(TrackingOrientation, RefinesLookAtFromMeasuredTcpOnEveryFaceTick)
{
  TrackingCore core{make_config()};
  const Eigen::Vector3d face{0.72, 0.18, 1.38};
  ASSERT_TRUE(core.ingest_face(make_sample(face, 100), 100).accepted());

  const ActualTcpPose actual = make_actual_pose(
    {0.24, -0.11, 1.02}, Eigen::Quaterniond::Identity(), 100);
  const TargetCommand target = command(core.tick(100, actual));

  const Eigen::Vector3d expected_direction =
    (face - actual.pose.position).normalized();
  EXPECT_TRUE((target.pose.orientation * Eigen::Vector3d::UnitX()).isApprox(
    expected_direction, kTolerance));
}

TEST(TrackingOrientation, RejectsInvalidMeasuredTcpDuringFaceTracking)
{
  TrackingCore core{make_config()};
  ASSERT_TRUE(core.ingest_face(make_sample({0.72, 0.18, 1.38}, 100), 100).accepted());
  ActualTcpPose actual = make_actual_pose(
    {0.24, -0.11, 1.02}, Eigen::Quaterniond::Identity(), 100);
  actual.pose.position.x() = std::numeric_limits<double>::quiet_NaN();

  const TickResult result = core.tick(100, actual);
  EXPECT_EQ(result.status, TickStatus::kInvalidActualPose);
  EXPECT_FALSE(result.command.has_value());
}

TEST(TrackingPolicy, HoldLatchesActualFullPoseOnce)
{
  TrackingCore core{make_config()};
  ASSERT_TRUE(core.ingest_face(make_sample({0.7, 0.0, 0.2}, 0), 0).accepted());

  const Eigen::Quaterniond first_orientation{
    Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitY())};
  const ActualTcpPose first = make_actual_pose(
    {0.35, 0.1, 1.0}, first_orientation, kFreshness);
  const ActualTcpPose second = make_actual_pose(
    {0.2, -0.2, 1.2}, -0.8, kFreshness + 1);
  const TargetCommand first_hold = command(core.tick(kFreshness, first));
  const TargetCommand second_hold = command(core.tick(kFreshness + 1, second));

  EXPECT_TRUE(second_hold.pose.position.isApprox(first_hold.pose.position, kTolerance));
  EXPECT_GT(second_hold.pose.orientation.dot(first_hold.pose.orientation), 1.0 - kTolerance);
}

TEST(TrackingPolicy, MissingOrInvalidActualPoseDoesNotRepeatStaleFace)
{
  TrackingCore core{make_config()};
  ASSERT_TRUE(core.ingest_face(make_sample({0.7, 0.0, 0.2}, 0), 0).accepted());

  const TickResult missing = core.tick(kFreshness);
  EXPECT_EQ(missing.status, TickStatus::kHoldPoseUnavailable);
  EXPECT_FALSE(missing.command.has_value());
  EXPECT_EQ(core.mode(), TargetMode::kHold);

  ActualTcpPose invalid = make_actual_pose({0.3, 0.0, 1.0}, 0.0, kFreshness + 1);
  invalid.pose.orientation.coeffs().setZero();
  const TickResult rejected = core.tick(kFreshness + 1, invalid);
  EXPECT_EQ(rejected.status, TickStatus::kInvalidActualPose);
  EXPECT_FALSE(rejected.command.has_value());

  const TickResult recovered = core.tick(
    kFreshness + 2,
    make_actual_pose({0.3, 0.0, 1.0}, 0.0, kFreshness + 2));
  EXPECT_EQ(command(recovered).mode, TargetMode::kHold);
}

TEST(TrackingPolicy, RestSupplierAcceptsNormalizedFullPose)
{
  TrackingCore core{make_config()};
  Pose3d dynamic_rest;
  dynamic_rest.position = {0.18, -0.04, 1.10};
  dynamic_rest.orientation = Eigen::Quaterniond{
    Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitY())};
  dynamic_rest.orientation.coeffs() *= 2.0;

  EXPECT_TRUE(core.update_rest_target(dynamic_rest));
  const TargetCommand rest = command(core.tick(10));
  EXPECT_EQ(rest.mode, TargetMode::kRest);
  EXPECT_TRUE(rest.pose.position.isApprox(dynamic_rest.position, kTolerance));
  EXPECT_NEAR(rest.pose.orientation.norm(), 1.0, kTolerance);
  EXPECT_NEAR(rest.pose.orientation.angularDistance(
      dynamic_rest.orientation.normalized()), 0.0, kTolerance);

  Pose3d invalid = dynamic_rest;
  invalid.orientation.coeffs().setZero();
  EXPECT_FALSE(core.update_rest_target(invalid));
  invalid = dynamic_rest;
  invalid.position.z() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(core.update_rest_target(invalid));
}

TEST(TrackingOrientation, KeepsQuaternionHemisphereContinuousAcrossYawWrap)
{
  TrackingCore core{make_config()};
  constexpr double degrees_to_radians = kPi / 180.0;

  const Eigen::Vector3d face_before{
    std::cos(179.0 * degrees_to_radians),
    std::sin(179.0 * degrees_to_radians),
    0.3};
  ASSERT_TRUE(core.ingest_face(make_sample(face_before, 0), 0).accepted());
  const TargetCommand before = command(core.tick(0));

  const Eigen::Vector3d face_after{
    std::cos(-179.0 * degrees_to_radians),
    std::sin(-179.0 * degrees_to_radians),
    0.3};
  ASSERT_TRUE(core.ingest_face(make_sample(face_after, 1), 1).accepted());
  const TargetCommand after = command(core.tick(1));

  EXPECT_GE(before.pose.orientation.dot(after.pose.orientation), 0.0);
  EXPECT_LT(
    before.pose.orientation.angularDistance(after.pose.orientation),
    5.0 * degrees_to_radians);
}

TEST(TrackingOrientation, AlignsEquivalentHoldQuaternionSign)
{
  TrackingCore core{make_config()};
  ASSERT_TRUE(core.ingest_face(make_sample({0.7, 0.0, 0.2}, 0), 0).accepted());
  const TargetCommand face = command(core.tick(0));

  ActualTcpPose actual = make_actual_pose(
    {0.3, 0.0, 1.0}, face.pose.orientation, kFreshness);
  actual.pose.orientation.coeffs() *= -1.0;
  const TargetCommand hold = command(core.tick(kFreshness, actual));

  EXPECT_GE(face.pose.orientation.dot(hold.pose.orientation), 0.0);
  EXPECT_NEAR(hold.pose.orientation.norm(), 1.0, kTolerance);
}

TEST(TrackingPolicy, ClockRewindClearsStateAndAcceptsLowerStamps)
{
  TrackingCore core{make_config()};
  ASSERT_TRUE(core.ingest_face(make_sample({0.7, 0.0, 0.2}, 1'000), 1'000).accepted());
  ASSERT_EQ(command(core.tick(1'000)).mode, TargetMode::kFace);

  const TickResult reset = core.tick(100);
  EXPECT_EQ(reset.status, TickStatus::kClockReset);
  EXPECT_EQ(command(reset).mode, TargetMode::kRest);

  EXPECT_TRUE(core.ingest_face(make_sample({0.0, 0.7, 0.2}, 101), 101).accepted());
  EXPECT_EQ(command(core.tick(101)).mode, TargetMode::kFace);
}

TEST(TrackingConfiguration, RejectsInvalidValues)
{
  TrackingConfig config = make_config();
  config.geometry.minimum_face_distance_m = 0.0;
  EXPECT_THROW(TrackingCore{config}, std::invalid_argument);

  config = make_config();
  config.geometry.safe_reach_radius_m = -1.0;
  EXPECT_THROW(TrackingCore{config}, std::invalid_argument);

  config = make_config();
  config.return_delay_ns = config.freshness_ns;
  EXPECT_THROW(TrackingCore{config}, std::invalid_argument);

  config = make_config();
  config.rest_position.z() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(TrackingCore{config}, std::invalid_argument);

  config = make_config();
  config.planning_frame.clear();
  EXPECT_THROW(TrackingCore{config}, std::invalid_argument);
}

TEST(TrackingCore, SearchIsAnIdleIntentAndFacePreemptsItAfterNormalLossDelay)
{
  auto config = make_config();
  config.search_when_idle = true;
  TrackingCore core(config);
  EXPECT_EQ(command(core.tick(1)).mode, TargetMode::kSearch);
  ASSERT_TRUE(core.ingest_face(make_sample({2.0, 0.0, 1.7}, 10), 10).accepted());
  EXPECT_EQ(command(core.tick(10)).mode, TargetMode::kFace);
  const auto actual = make_actual_pose({.2, 0.0, 1.05}, 0.0, kFreshness + 10);
  EXPECT_EQ(command(core.tick(kFreshness + 10, actual)).mode, TargetMode::kHold);
  const auto search = command(core.tick(kReturnDelay + 10));
  EXPECT_EQ(search.mode, TargetMode::kSearch);
  EXPECT_EQ(search.source_face_stamp_ns, 10);
  EXPECT_FALSE(search.face_in_planning_frame);
  ASSERT_TRUE(core.ingest_face(make_sample({2.0, .2, 1.7}, kReturnDelay + 11),
    kReturnDelay + 11).accepted());
  EXPECT_EQ(command(core.tick(kReturnDelay + 11)).mode, TargetMode::kFace);
}

TEST(TrackingPolicy, ForecastUsesRecentMotionOnlyWhenHealthyNoFaceIsExplicit)
{
  TrackingConfig config = make_config();
  config.prediction_timeout_ns = 600'000'000;
  TrackingCore core(config);
  ASSERT_TRUE(core.ingest_face(make_sample({1.0, 0.0, 0.0}, 1'000'000'000),
      1'000'000'000).accepted());
  ASSERT_TRUE(core.ingest_face(make_sample({1.1, 0.0, 0.0}, 1'100'000'000),
      1'100'000'000).accepted());

  const auto actual = make_actual_pose({0.4, 0.0, 0.0}, 0.0, 1'350'000'000);
  const auto no_prediction = command(core.tick(1'350'000'000, actual, false));
  EXPECT_EQ(no_prediction.mode, TargetMode::kHold);

  const auto forecast = command(core.tick(1'350'000'000, actual, true));
  EXPECT_EQ(forecast.mode, TargetMode::kFace);
  ASSERT_TRUE(forecast.face_in_planning_frame);
  EXPECT_GT(forecast.face_in_planning_frame->x(), 1.1);
  ASSERT_TRUE(forecast.source_face_stamp_ns);
  EXPECT_EQ(*forecast.source_face_stamp_ns, 1'100'000'000);
  ASSERT_TRUE(forecast.face_prediction_expiry_ns);
  EXPECT_EQ(*forecast.face_prediction_expiry_ns, 1'700'000'000);
  EXPECT_TRUE(control::validFaceTargetAge(
      *forecast.source_face_stamp_ns, forecast.command_stamp_ns, 1'350'000'000,
      200'000'000, true, *forecast.face_prediction_expiry_ns, 600'000'000));
  EXPECT_FALSE(control::validFaceTargetAge(
      *forecast.source_face_stamp_ns, forecast.command_stamp_ns, 1'350'000'000,
      200'000'000, false, *forecast.face_prediction_expiry_ns, 600'000'000));
  EXPECT_FALSE(control::validFaceTargetAge(
      *forecast.source_face_stamp_ns, forecast.command_stamp_ns, 1'700'000'000,
      200'000'000, true, *forecast.face_prediction_expiry_ns, 600'000'000));

  const auto expiry_pose = make_actual_pose({.4, 0.0, 0.0}, 0.0, 1'700'000'000);
  EXPECT_EQ(command(core.tick(1'700'000'000, expiry_pose, true)).mode, TargetMode::kHold);
}

TEST(TrackingPolicy, HealthyNoFaceForecastUsesSourceHorizonAtThirtyAndFifteenHertz)
{
  constexpr std::int64_t prediction_horizon_ns = 800'000'000;
  constexpr std::int64_t origin_ns = 1'000'000'000;
  for (const double sample_period_sec : {1.0 / 30.0, 1.0 / 15.0}) {
    TrackingConfig config = make_config();
    config.prediction_timeout_ns = prediction_horizon_ns;
    TrackingCore core(config);
    std::int64_t last_stamp_ns = origin_ns;
    double last_face_x = 1.0;
    for (int sample_index = 0; sample_index < 60; ++sample_index) {
      const double elapsed_sec = static_cast<double>(sample_index) * sample_period_sec;
      last_stamp_ns = origin_ns + static_cast<std::int64_t>(std::llround(
          elapsed_sec * 1.0e9));
      last_face_x = 1.0 + 0.2 * elapsed_sec;
      const auto result = core.ingest_face(
        make_sample({last_face_x, 0.0, 0.0}, last_stamp_ns), last_stamp_ns);
      ASSERT_TRUE(result.accepted()) << "sample " << sample_index;
    }

    // Explicit healthy no-face status selects a source-time forecast immediately,
    // without waiting for the unrelated live-target freshness window to expire.
    const auto early_ns = last_stamp_ns + 10'000'000;
    const auto early = command(core.tick(early_ns, std::nullopt, true));
    EXPECT_EQ(early.mode, TargetMode::kFace);
    ASSERT_TRUE(early.face_in_planning_frame);
    EXPECT_GT(early.face_in_planning_frame->x(), last_face_x);
    ASSERT_TRUE(early.face_prediction_expiry_ns);
    EXPECT_EQ(*early.face_prediction_expiry_ns, last_stamp_ns + prediction_horizon_ns);

    const auto half_horizon_ns = last_stamp_ns + 500'000'000;
    EXPECT_EQ(command(core.tick(half_horizon_ns, std::nullopt, true)).mode,
      TargetMode::kFace);
    const auto before_expiry_ns = last_stamp_ns + 799'000'000;
    const auto before_expiry = command(core.tick(before_expiry_ns, std::nullopt, true));
    EXPECT_EQ(before_expiry.mode, TargetMode::kFace);
    EXPECT_EQ(before_expiry.face_prediction_expiry_ns,
      std::optional<RosTimeNanoseconds>(last_stamp_ns + prediction_horizon_ns));

    const auto expiry_ns = last_stamp_ns + prediction_horizon_ns;
    const auto actual = make_actual_pose({0.4, 0.0, 0.0}, 0.0, expiry_ns);
    const auto expired = command(core.tick(expiry_ns, actual, true));
    EXPECT_EQ(expired.mode, TargetMode::kHold);
    EXPECT_FALSE(expired.face_prediction_expiry_ns);
  }
}

TEST(TrackingPolicy, DisabledForecastAndCoreResetNeverEmitPredictedFace)
{
  TrackingCore core(make_config());
  ASSERT_TRUE(core.ingest_face(make_sample({1.0, 0.0, 0.0}, 1'000'000'000),
      1'000'000'000).accepted());
  ASSERT_TRUE(core.ingest_face(make_sample({1.02, 0.0, 0.0}, 1'100'000'000),
      1'100'000'000).accepted());
  const auto now_ns = 1'600'000'000;
  const auto actual = make_actual_pose({0.4, 0.0, 0.0}, 0.0, now_ns);
  const auto disabled = command(core.tick(now_ns, actual, true));
  EXPECT_EQ(disabled.mode, TargetMode::kHold);
  EXPECT_FALSE(disabled.face_in_planning_frame);
  EXPECT_FALSE(disabled.face_prediction_expiry_ns);

  core.reset();
  const auto after_reset = command(core.tick(now_ns + 1, actual, true));
  EXPECT_EQ(after_reset.mode, TargetMode::kRest);
  EXPECT_FALSE(after_reset.face_in_planning_frame);
  EXPECT_FALSE(after_reset.face_prediction_expiry_ns);
}

TEST(TrackingPolicy, RetainedForecastSnapshotExpiresWithoutANewCommand)
{
  // The executor keeps the last command snapshot until its command timeout,
  // which is longer than this forecast horizon. Its control tick must still
  // reject the snapshot as soon as the source-based forecast expires.
  constexpr std::int64_t source_stamp_ns = 1'100'000'000;
  constexpr std::int64_t command_stamp_ns = 1'350'000'000;
  constexpr std::int64_t expiry_ns = 1'700'000'000;
  constexpr std::int64_t live_age_limit_ns = 200'000'000;
  constexpr std::int64_t prediction_horizon_ns = 600'000'000;
  constexpr std::int64_t incoming_command_timeout_ns = 2'000'000'000;

  EXPECT_LT(1'699'000'000 - command_stamp_ns, incoming_command_timeout_ns);
  EXPECT_TRUE(control::validFaceTargetAge(
      source_stamp_ns, command_stamp_ns, 1'699'000'000, live_age_limit_ns,
      true, expiry_ns, prediction_horizon_ns));
  EXPECT_FALSE(control::validFaceTargetAge(
      source_stamp_ns, command_stamp_ns, expiry_ns, live_age_limit_ns,
      true, expiry_ns, prediction_horizon_ns));

  constexpr std::int64_t live_command_stamp_ns = 1'250'000'000;
  EXPECT_TRUE(control::validFaceTargetAge(
      source_stamp_ns, live_command_stamp_ns, 1'290'000'000, live_age_limit_ns,
      false, 0, prediction_horizon_ns));
  EXPECT_FALSE(control::validFaceTargetAge(
      source_stamp_ns, live_command_stamp_ns, 1'301'000'000, live_age_limit_ns,
      false, 0, prediction_horizon_ns));
}

}  // namespace
}  // namespace face_tracking_arm::tracking
