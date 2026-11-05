// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "face_tracking_arm/compact_profile_worker.hpp"

namespace face_tracking_arm::control
{
namespace
{

CompactJointProfile profileAt(const double position)
{
  CompactJointProfile profile;
  JointMotionState sample;
  sample.position = Eigen::VectorXd::Constant(1, position);
  sample.velocity = Eigen::VectorXd::Zero(1);
  sample.acceleration = Eigen::VectorXd::Zero(1);
  profile.samples.push_back(std::move(sample));
  return profile;
}

CompactDynamicsResult passing(const CompactJointProfile & profile)
{
  CompactDynamicsResult result;
  result.status = CompactDynamicsStatus::kPass;
  result.profile = profile;
  return result;
}

template<typename Predicate>
bool waitUntil(Predicate predicate)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {return true;}
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return predicate();
}

CompactProfileWorkRequest request(const double start)
{
  CompactProfileWorkRequest work;
  work.generation = static_cast<std::uint64_t>(start);
  work.scene_revision = 7U;
  work.profile = profileAt(start);
  work.control_period_sec = 0.01;
  work.return_profile_required = false;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 7U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.empty() ? 0U : profile.samples.size() - 1U;
      return certificate;
    };
  return work;
}

JointMotionLimits lineLimits()
{
  JointMotionLimits limits;
  limits.lower_position = Eigen::VectorXd::Constant(1, -2.0);
  limits.upper_position = Eigen::VectorXd::Constant(1, 2.0);
  limits.position_margin = Eigen::VectorXd::Zero(1);
  limits.max_velocity = Eigen::VectorXd::Constant(1, 3.0);
  limits.max_acceleration = Eigen::VectorXd::Constant(1, 10.0);
  limits.max_jerk = Eigen::VectorXd::Constant(1, 100.0);
  const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
    PositionInterval{-2.0, 2.0}, 3.0, 10.0, 100.0, 0.01,
    SolverResidual{1.0e-5});
  if (policy) {
    limits.numerical_policy = *policy;
  }
  return limits;
}

TEST(CompactProfileWorker, LatestGenerationReplacesPendingAndDropsActiveResult)
{
  std::mutex mutex;
  std::condition_variable condition;
  bool first_started = false;
  bool release_first = false;
  CompactProfileWorker worker([&](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      if (profile.samples.front().position[0] == 1.0) {
        std::unique_lock<std::mutex> lock(mutex);
        first_started = true;
        condition.notify_all();
        condition.wait(lock, [&]() {return release_first;});
      }
      return passing(profile);
    });

  static_cast<void>(worker.submit(request(1.0)));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&]() {
        return first_started;
    }));
  }
  const auto latest_id = worker.submit(request(2.0));
  {
    const std::lock_guard<std::mutex> lock(mutex);
    release_first = true;
  }
  condition.notify_all();

  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, latest_id);
  ASSERT_TRUE(result->profile.profile);
  EXPECT_DOUBLE_EQ(result->profile.profile->samples.front().position[0], 2.0);
  EXPECT_TRUE(result->geometry_certificate.passedFor(
      7U, result->profile.profile->samples.size(),
      result->profile.profile->samples.size() - 1U));
}

TEST(CompactProfileWorker, CancelDropsInFlightResult)
{
  std::mutex mutex;
  std::condition_variable condition;
  bool started = false;
  bool release = false;
  CompactProfileWorker worker([&](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      std::unique_lock<std::mutex> lock(mutex);
      started = true;
      condition.notify_all();
      condition.wait(lock, [&]() {return release;});
      return passing(profile);
    });
  static_cast<void>(worker.submit(request(1.0)));
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&]() {return started;}));
  }
  worker.cancel();
  {
    const std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  condition.notify_all();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(worker.takeResult());
}

TEST(CompactProfileWorker, BuildsSourceRouteAndCertifiesRetimedSamples)
{
  CompactProfileWorkRequest work;
  work.generation = 12U;
  work.scene_revision = 21U;
  work.source_route = {Eigen::VectorXd::Constant(1, 0.0),
    Eigen::VectorXd::Constant(1, 0.2)};
  work.segment_speed_scales = {1.0};
  work.start_state = profileAt(0.0).samples.front();
  work.limits = lineLimits();
  work.control_period_sec = 0.01;
  work.return_profile_required = false;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 21U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  const auto request_id = worker.submit(std::move(work));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, request_id);
  ASSERT_TRUE(result->profile.profile);
  EXPECT_DOUBLE_EQ(result->profile.profile->samples.back().position[0], 0.2);
  EXPECT_TRUE(result->geometry_certificate.passedFor(
      21U, result->profile.profile->samples.size(),
      result->profile.profile->samples.size() - 1U));
  EXPECT_TRUE(std::isfinite(result->build_elapsed_ms));
  EXPECT_TRUE(std::isfinite(result->geometry_elapsed_ms));
  EXPECT_TRUE(std::isfinite(result->total_elapsed_ms));
  EXPECT_GE(result->build_elapsed_ms, 0.0);
  EXPECT_GE(result->geometry_elapsed_ms, 0.0);
  EXPECT_GE(result->total_elapsed_ms, 0.0);
}

TEST(CompactProfileWorker, AcceptsStationarySinglePointSourceRoute)
{
  CompactProfileWorkRequest work;
  work.generation = 15U;
  work.scene_revision = 26U;
  work.start_state = profileAt(0.0).samples.front();
  work.source_route = {work.start_state.position};
  work.return_profile_required = false;
  work.limits = lineLimits();
  work.control_period_sec = 0.01;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 26U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  ASSERT_TRUE(result->profile.profile) << result->profile.failure_reason;
  EXPECT_EQ(result->profile.profile->checked_path_vertices.size(), 1U);
  EXPECT_EQ(result->profile.profile->checked_path_segment_scales.size(), 0U);
  EXPECT_TRUE(result->geometry_certificate.passedFor(
      26U, result->profile.profile->samples.size(),
      result->profile.profile->samples.size() - 1U));
}

TEST(CompactProfileWorker, RejectsMovingSinglePointSourceRoute)
{
  std::atomic<unsigned int> retimer_calls{0U};
  CompactProfileWorkRequest work;
  work.scene_revision = 27U;
  work.start_state = profileAt(0.0).samples.front();
  work.start_state.velocity = Eigen::VectorXd::Constant(1, 0.1);
  work.source_route = {work.start_state.position};
  work.return_profile_required = false;
  work.limits = lineLimits();
  work.control_period_sec = 0.01;
  CompactProfileWorker worker([&retimer_calls](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      retimer_calls.fetch_add(1U);
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->profile.status, CompactDynamicsStatus::kInvalidInput);
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("stationary start"), std::string::npos);
  EXPECT_EQ(retimer_calls.load(), 0U);
}

TEST(CompactProfileWorker, PreservesExactPublishedPrefixBeforeBuildingSuffix)
{
  CompactProfileWorkRequest work;
  work.generation = 13U;
  work.scene_revision = 22U;
  JointMotionState first;
  first.position = Eigen::VectorXd::Constant(1, 0.0);
  first.velocity = Eigen::VectorXd::Constant(1, 0.5);
  first.acceleration = Eigen::VectorXd::Zero(1);
  JointMotionState endpoint;
  endpoint.position = Eigen::VectorXd::Constant(1, 0.05);
  endpoint.velocity = Eigen::VectorXd::Zero(1);
  endpoint.acceleration = Eigen::VectorXd::Zero(1);
  work.published_prefix = {first, endpoint};
  work.start_state = endpoint;
  work.prefix_limits = lineLimits();
  work.source_route = {endpoint.position, Eigen::VectorXd::Constant(1, 0.25)};
  work.segment_speed_scales = {1.0};
  work.return_profile_required = false;
  work.limits = lineLimits();
  work.control_period_sec = 0.01;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 22U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  ASSERT_TRUE(result->profile.profile);
  const auto & profile = *result->profile.profile;
  ASSERT_GE(profile.samples.size(), 3U);
  EXPECT_EQ(profile.prefix_endpoint_sample_index, 1U);
  EXPECT_DOUBLE_EQ(profile.samples[0].position[0], first.position[0]);
  EXPECT_DOUBLE_EQ(profile.samples[0].velocity[0], first.velocity[0]);
  EXPECT_DOUBLE_EQ(profile.samples[1].position[0], endpoint.position[0]);
  EXPECT_DOUBLE_EQ(profile.samples[1].velocity[0], endpoint.velocity[0]);
  EXPECT_TRUE(result->geometry_certificate.passedFor(
      22U, profile.samples.size(), profile.samples.size() - 1U));
}

TEST(CompactProfileWorker, AcceptsSingleStationaryPublishedPrefix)
{
  CompactProfileWorkRequest work;
  work.generation = 14U;
  work.scene_revision = 23U;
  const JointMotionState stationary = profileAt(0.0).samples.front();
  work.published_prefix = {stationary};
  work.start_state = stationary;
  work.prefix_limits = lineLimits();
  work.source_route = {stationary.position, Eigen::VectorXd::Constant(1, 0.2)};
  work.segment_speed_scales = {1.0};
  work.return_profile_required = false;
  work.limits = lineLimits();
  work.control_period_sec = 0.01;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 23U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  ASSERT_TRUE(result->profile.profile) << result->profile.failure_reason;
  const auto & profile = *result->profile.profile;
  ASSERT_GE(profile.samples.size(), 2U);
  EXPECT_EQ(profile.prefix_endpoint_sample_index, 0U);
  EXPECT_TRUE(profile.samples.front().position.isApprox(stationary.position, 0.0));
  EXPECT_TRUE(profile.samples.front().velocity.isApprox(stationary.velocity, 0.0));
  EXPECT_TRUE(profile.samples.front().acceleration.isApprox(stationary.acceleration, 0.0));
  EXPECT_TRUE(result->geometry_certificate.passedFor(
      23U, profile.samples.size(), profile.samples.size() - 1U));
}

TEST(CompactProfileWorker, RejectsEmptyPrefixAndContinuesWithNextRequest)
{
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });
  CompactProfileWorkRequest malformed;
  malformed.scene_revision = 25U;
  malformed.source_route = {Eigen::VectorXd::Constant(1, 0.0),
    Eigen::VectorXd::Constant(1, 0.2)};
  malformed.segment_speed_scales = {1.0};
  malformed.published_prefix = {JointMotionState{}};
  malformed.start_state = JointMotionState{};
  malformed.prefix_limits = lineLimits();
  malformed.limits = lineLimits();
  malformed.control_period_sec = 0.01;

  const auto bad_id = worker.submit(std::move(malformed));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, bad_id);
  EXPECT_EQ(result->profile.status, CompactDynamicsStatus::kInvalidInput);
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("published prefix"), std::string::npos);

  const auto good_id = worker.submit(request(1.0));
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, good_id);
  ASSERT_TRUE(result->profile.profile) << result->profile.failure_reason;
}

TEST(CompactProfileWorker, BuilderExceptionReturnsFailureAndWorkerRecovers)
{
  std::atomic<unsigned int> builder_calls{0U};
  CompactProfileWorker worker(
    [](const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    },
    [&builder_calls](
      const std::vector<Eigen::VectorXd> & path, const JointMotionState & start,
      const JointMotionLimits & limits, const double period, const double scale,
      const std::vector<double> & segment_scales, std::string * failure_reason) {
      if (builder_calls.fetch_add(1U) == 0U) {
        throw std::runtime_error("profile builder exploded");
      }
      return buildCompactJointProfile(
        path, start, limits, period, scale, segment_scales, failure_reason);
    });
  CompactProfileWorkRequest work;
  work.scene_revision = 24U;
  work.source_route = {Eigen::VectorXd::Constant(1, 0.0),
    Eigen::VectorXd::Constant(1, 0.2)};
  work.segment_speed_scales = {1.0};
  work.start_state = profileAt(0.0).samples.front();
  work.limits = lineLimits();
  work.control_period_sec = 0.01;
  work.return_profile_required = false;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 24U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };

  const auto first_id = worker.submit(work);
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, first_id);
  EXPECT_EQ(result->profile.status, CompactDynamicsStatus::kSolverFailure);
  EXPECT_NE(result->profile.failure_reason.find("profile builder exploded"), std::string::npos);

  const auto second_id = worker.submit(std::move(work));
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, second_id);
  ASSERT_TRUE(result->profile.profile) << result->profile.failure_reason;
}

TEST(CompactProfileWorker, GeometryRefusalFailsMainAndSkipsReturnWork)
{
  std::atomic<unsigned int> retimer_calls{0U};
  auto work = request(1.0);
  work.return_profile = profileAt(2.0);
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kUnsafe;
      certificate.scene_revision = 7U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      certificate.failure_reason = "radius cap exceeded at an interior sample";
      return certificate;
    };
  CompactProfileWorker worker([&retimer_calls](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      retimer_calls.fetch_add(1U);
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(retimer_calls.load(), 1U);
  EXPECT_EQ(result->geometry_certificate.status, CompactProfileGeometryStatus::kUnsafe);
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("radius cap exceeded"), std::string::npos);
  EXPECT_FALSE(result->return_profile.has_value());
  EXPECT_TRUE(std::isfinite(result->build_elapsed_ms));
  EXPECT_TRUE(std::isfinite(result->geometry_elapsed_ms));
  EXPECT_TRUE(std::isfinite(result->total_elapsed_ms));
}

TEST(CompactProfileWorker, RequiredReturnGeometryRefusalRejectsMainProfile)
{
  std::atomic<unsigned int> main_geometry_calls{0U};
  std::atomic<unsigned int> return_geometry_calls{0U};
  auto work = request(1.0);
  work.return_profile = profileAt(2.0);
  work.return_profile_required = true;
  work.final_geometry_validator = [&main_geometry_calls](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      main_geometry_calls.fetch_add(1U);
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 7U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };
  work.return_final_geometry_validator = [&return_geometry_calls](
    const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      return_geometry_calls.fetch_add(1U);
      certificate.status = CompactProfileGeometryStatus::kUnsafe;
      certificate.scene_revision = 7U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      certificate.failure_reason = "return endpoint is outside the frozen region";
      return certificate;
    };
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("mandatory return profile failed"),
    std::string::npos);
  ASSERT_TRUE(result->return_profile);
  EXPECT_FALSE(result->return_profile->profile);
  EXPECT_EQ(result->return_profile->status, CompactDynamicsStatus::kInvalidInput);
  ASSERT_TRUE(result->return_geometry_certificate);
  EXPECT_EQ(result->return_geometry_certificate->status,
    CompactProfileGeometryStatus::kUnsafe);
  EXPECT_NE(result->return_profile->failure_reason.find(
        "return endpoint is outside the frozen region"),
    std::string::npos);
  EXPECT_EQ(main_geometry_calls.load(), 1U);
  EXPECT_EQ(return_geometry_calls.load(), 1U);
}

TEST(CompactProfileWorker, StaleGeometryCertificateCannotCertifyCurrentScene)
{
  auto work = request(1.0);
  work.scene_revision = 8U;
  work.final_geometry_validator = [](const CompactJointProfile & profile) {
      CompactProfileGeometryCertificate certificate;
      certificate.status = CompactProfileGeometryStatus::kPass;
      certificate.scene_revision = 7U;
      certificate.checked_samples = profile.samples.size();
      certificate.checked_intervals = profile.samples.size() - 1U;
      return certificate;
    };
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->geometry_certificate.status, CompactProfileGeometryStatus::kStaleScene);
  EXPECT_FALSE(result->profile.profile);
  EXPECT_EQ(result->profile.status, CompactDynamicsStatus::kInvalidInput);
  EXPECT_NE(result->profile.failure_reason.find("stale"), std::string::npos);
}

TEST(CompactProfileWorker, MissingGeometryValidatorDoesNotProduceCertificate)
{
  auto work = request(1.0);
  work.final_geometry_validator = {};
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_FALSE(result->geometry_certificate.passedFor(
      result->scene_revision, result->profile.profile->samples.size(), 0U));
  EXPECT_EQ(result->geometry_certificate.status, CompactProfileGeometryStatus::kInvalidInput);
}

TEST(CompactProfileWorker, SolverFailureIsReturnedWithoutProfile)
{
  CompactProfileWorker worker([](
      const CompactJointProfile &, const JointMotionLimits &, double, double)
    -> CompactDynamicsResult {throw std::runtime_error("solver failed");});
  const auto request_id = worker.submit(request(1.0));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, request_id);
  EXPECT_EQ(result->profile.status, CompactDynamicsStatus::kSolverFailure);
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("solver failed"), std::string::npos);
}

TEST(CompactProfileWorker, RequiredReturnFailureRejectsMainProfile)
{
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      if (profile.samples.front().position[0] == 2.0) {
        throw std::runtime_error("return unavailable");
      }
      return passing(profile);
    });
  auto work = request(1.0);
  work.return_profile = profileAt(2.0);
  const auto request_id = worker.submit(std::move(work));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_EQ(result->request_id, request_id);
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("mandatory return profile failed"),
    std::string::npos);
  ASSERT_TRUE(result->return_profile);
  EXPECT_FALSE(result->return_profile->profile);
  EXPECT_EQ(result->return_profile->status, CompactDynamicsStatus::kSolverFailure);
}

TEST(CompactProfileWorker, MissingRequiredReturnRejectsMainProfile)
{
  auto work = request(1.0);
  work.return_profile_required = true;
  CompactProfileWorker worker([](
      const CompactJointProfile & profile, const JointMotionLimits &, double, double) {
      return passing(profile);
    });

  static_cast<void>(worker.submit(std::move(work)));
  std::optional<CompactProfileWorkResult> result;
  ASSERT_TRUE(waitUntil([&]() {
      result = worker.takeResult();
      return result.has_value();
  }));
  EXPECT_FALSE(result->profile.profile);
  EXPECT_NE(result->profile.failure_reason.find("mandatory return profile is missing"),
    std::string::npos);
}

}  // namespace
}  // namespace face_tracking_arm::control
