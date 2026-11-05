// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_profile_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace face_tracking_arm::control
{
namespace
{

CompactProfileWorker::Retimer makeRetimer(
  std::shared_ptr<CompactDynamicsModel> model)
{
  if (!model) {
    throw std::invalid_argument("Compact profile worker requires a dynamics model");
  }
  return [model = std::move(model)](
    const CompactJointProfile & profile, const JointMotionLimits & limits,
    const double period_sec, const double speed_scale) {
           return model->validateAndRetime(profile, limits, period_sec, speed_scale);
         };
}

bool accepted(const CompactDynamicsResult & result)
{
  return result.profile.has_value() &&
         (result.status == CompactDynamicsStatus::kPass ||
         result.status == CompactDynamicsStatus::kRetimed ||
         result.status == CompactDynamicsStatus::kPrefixViolation);
}

bool profileLimitsValid(
  const std::vector<JointMotionState> & states,
  const JointMotionLimits & limits,
  const double period_sec,
  std::string * failure_reason)
{
  const auto fail = [failure_reason](const std::string & reason) {
      if (failure_reason != nullptr) {
        *failure_reason = reason;
      }
      return false;
    };
  if (states.empty() || !std::isfinite(period_sec) || period_sec <= 0.0 ||
    !limits.numerical_policy || !limits.numerical_policy->valid())
  {
    return fail("published prefix or its motion limits are invalid");
  }
  const Eigen::Index count = states.front().position.size();
  if (count <= 0 || limits.lower_position.size() != count ||
    limits.upper_position.size() != count || limits.max_velocity.size() != count ||
    limits.max_acceleration.size() != count || limits.max_jerk.size() != count ||
    !limits.lower_position.allFinite() || !limits.upper_position.allFinite() ||
    !limits.max_velocity.allFinite() || !limits.max_acceleration.allFinite() ||
    !limits.max_jerk.allFinite() ||
    (limits.lower_position.array() > limits.upper_position.array()).any() ||
    (limits.max_velocity.array() <= 0.0).any() ||
    (limits.max_acceleration.array() <= 0.0).any() ||
    (limits.max_jerk.array() <= 0.0).any())
  {
    return fail("published prefix limit dimensions or values are invalid");
  }

  const double position_tolerance = limits.numerical_policy->position_error.value;
  const double velocity_tolerance = limits.numerical_policy->velocity_error.value;
  const double acceleration_tolerance = limits.numerical_policy->acceleration_error.value;
  const double jerk_tolerance = limits.numerical_policy->jerk_error.value;
  const JointMotionState * previous = nullptr;
  for (const auto & state : states) {
    if (state.position.size() != count || state.velocity.size() != count ||
      state.acceleration.size() != count || !state.position.allFinite() ||
      !state.velocity.allFinite() || !state.acceleration.allFinite())
    {
      return fail("published prefix contains a nonfinite or mismatched state");
    }
    if ((state.position.array() < limits.lower_position.array() - position_tolerance).any() ||
      (state.position.array() > limits.upper_position.array() + position_tolerance).any() ||
      (state.velocity.cwiseAbs().array() >
      limits.max_velocity.array() + velocity_tolerance).any() ||
      (state.acceleration.cwiseAbs().array() >
      limits.max_acceleration.array() + acceleration_tolerance).any())
    {
      return fail("published prefix exceeds its immutable joint limits");
    }
    if (previous != nullptr) {
      const Eigen::VectorXd jerk = (state.acceleration - previous->acceleration) / period_sec;
      if (!jerk.allFinite() ||
        (jerk.cwiseAbs().array() > limits.max_jerk.array() + jerk_tolerance).any())
      {
        return fail("published prefix exceeds its immutable jerk limits");
      }
    }
    previous = &state;
  }
  if (!states.back().velocity.isZero(1.0e-12) ||
    !states.back().acceleration.isZero(1.0e-12))
  {
    return fail("published prefix has no exact stationary endpoint");
  }
  return true;
}

CompactDynamicsResult failedResult(
  const CompactDynamicsStatus status, const std::string & reason)
{
  CompactDynamicsResult result;
  result.status = status;
  result.failure_reason = reason;
  return result;
}

CompactProfileGeometryCertificate certifyGeometry(
  const CompactJointProfile & profile,
  const CompactProfileFinalGeometryValidator & validator,
  const std::uint64_t expected_scene_revision)
{
  CompactProfileGeometryCertificate certificate;
  if (!validator) {
    certificate.status = CompactProfileGeometryStatus::kInvalidInput;
    certificate.scene_revision = expected_scene_revision;
    certificate.failure_reason = "final geometry validator is required";
    return certificate;
  }
  if (profile.samples.empty()) {
    certificate.status = CompactProfileGeometryStatus::kInvalidInput;
    certificate.scene_revision = expected_scene_revision;
    certificate.failure_reason = "final profile has no samples";
    return certificate;
  }
  try {
    certificate = validator(profile);
  } catch (const std::exception & error) {
    certificate.status = CompactProfileGeometryStatus::kValidatorFailure;
    certificate.failure_reason = std::string{"final geometry validator threw: "} + error.what();
  } catch (...) {
    certificate.status = CompactProfileGeometryStatus::kValidatorFailure;
    certificate.failure_reason = "final geometry validator threw an unknown exception";
  }
  if (certificate.status == CompactProfileGeometryStatus::kPass &&
    certificate.scene_revision != expected_scene_revision)
  {
    certificate.status = CompactProfileGeometryStatus::kStaleScene;
    certificate.failure_reason = "geometry certificate scene revision is stale";
  }
  if (certificate.status == CompactProfileGeometryStatus::kPass &&
    (certificate.checked_samples != profile.samples.size() ||
    certificate.checked_intervals != profile.samples.size() - 1U))
  {
    certificate.status = CompactProfileGeometryStatus::kInvalidInput;
    certificate.failure_reason = "geometry certificate does not cover the complete profile";
  }
  return certificate;
}

CompactDynamicsResult failedResult(const std::string & reason)
{
  CompactDynamicsResult result;
  result.status = CompactDynamicsStatus::kSolverFailure;
  result.failure_reason = reason;
  return result;
}

}  // namespace

CompactProfileWorker::CompactProfileWorker(std::shared_ptr<CompactDynamicsModel> model)
: CompactProfileWorker(makeRetimer(std::move(model)))
{}

CompactProfileWorker::CompactProfileWorker(Retimer retimer)
: retimer_(std::move(retimer))
{
  if (!retimer_) {
    throw std::invalid_argument("Compact profile worker requires a retimer");
  }
  worker_ = std::thread([this]() {run();});
}

CompactProfileWorker::CompactProfileWorker(
  Retimer retimer, ProfileBuilderForTest builder_for_test)
: retimer_(std::move(retimer)), builder_for_test_(std::move(builder_for_test))
{
  if (!retimer_ || !builder_for_test_) {
    throw std::invalid_argument("Compact profile worker test seam requires retimer and builder");
  }
  worker_ = std::thread([this]() {run();});
}

CompactProfileWorker::~CompactProfileWorker()
{
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    pending_.reset();
    result_.reset();
  }
  condition_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

std::uint64_t CompactProfileWorker::submit(CompactProfileWorkRequest request)
{
  std::uint64_t request_id = 0U;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::logic_error("Compact profile worker is stopping");
    }
    request_id = ++next_request_id_;
    latest_request_id_ = request_id;
    result_.reset();
    pending_ = WorkItem{request_id, std::move(request)};
  }
  condition_.notify_one();
  return request_id;
}

void CompactProfileWorker::cancel()
{
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    latest_request_id_ = ++next_request_id_;
    pending_.reset();
    result_.reset();
  }
  condition_.notify_one();
}

std::optional<CompactProfileWorkResult> CompactProfileWorker::takeResult()
{
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!result_ || result_->request_id != latest_request_id_) {
    return std::nullopt;
  }
  std::optional<CompactProfileWorkResult> result{std::move(result_)};
  result_.reset();
  return result;
}

void CompactProfileWorker::run()
{
  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this]() {return stopping_ || pending_.has_value();});
      if (stopping_) {
        return;
      }
      item = std::move(*pending_);
      pending_.reset();
    }

    CompactProfileWorkResult result;
    const auto total_start = std::chrono::steady_clock::now();
    result.request_id = item.request_id;
    result.generation = item.request.generation;
    result.scene_revision = item.request.scene_revision;
    CompactJointProfile source_profile;
    const auto build_start = std::chrono::steady_clock::now();
    bool build_timer_stopped = false;
    const auto stop_build_timer = [&]() {
        if (!build_timer_stopped) {
          result.build_elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_start).count();
          build_timer_stopped = true;
        }
      };
    try {
      source_profile = item.request.profile;
      if (!item.request.source_route.empty()) {
        std::string failure_reason;
        const auto & request = item.request;
        const JointMotionState * start = &request.start_state;
        if (!request.published_prefix.empty()) {
          const auto & prefix_endpoint = request.published_prefix.back();
          const bool prefix_valid = profileLimitsValid(request.published_prefix,
            request.prefix_limits, request.control_period_sec, &failure_reason);
          const auto & initial = request.start_state;
          const Eigen::Index joint_count = prefix_endpoint.position.size();
          const bool start_valid = initial.position.size() > 0 &&
            initial.velocity.size() == initial.position.size() &&
            initial.acceleration.size() == initial.position.size() &&
            initial.position.allFinite() && initial.velocity.allFinite() &&
            initial.acceleration.allFinite();
          if (!prefix_valid) {
            result.profile = failedResult(CompactDynamicsStatus::kInvalidInput, failure_reason);
          } else {
            if (!start_valid || joint_count <= 0 ||
              request.start_state.position.size() != prefix_endpoint.position.size() ||
              request.start_state.velocity.size() != prefix_endpoint.velocity.size() ||
              request.start_state.acceleration.size() != prefix_endpoint.acceleration.size() ||
              (request.start_state.position - prefix_endpoint.position).
              lpNorm<Eigen::Infinity>() > 1.0e-12 ||
              (request.start_state.velocity - prefix_endpoint.velocity).
              lpNorm<Eigen::Infinity>() > 1.0e-12 ||
              (request.start_state.acceleration - prefix_endpoint.acceleration).
              lpNorm<Eigen::Infinity>() > 1.0e-12)
            {
              result.profile = failedResult(
                CompactDynamicsStatus::kInvalidInput,
                "profile start state differs from the immutable prefix endpoint");
            } else {
              start = &prefix_endpoint;
            }
          }
        }
        if (result.profile.failure_reason.empty()) {
          if (request.source_route.empty() ||
            request.segment_speed_scales.size() + 1U != request.source_route.size() ||
            start->position.size() != request.source_route.front().size() ||
            (start->position - request.source_route.front()).
            lpNorm<Eigen::Infinity>() > kCompactProfilePositionToleranceRad)
          {
            result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
              "source route dimensions or start position are inconsistent");
          } else {
            if (request.source_route.size() == 1U &&
              (!start->velocity.isZero(1.0e-12) || !start->acceleration.isZero(1.0e-12)))
            {
              result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
                "single-point source route requires a stationary start state");
            } else {
              auto built = builder_for_test_ ?
                builder_for_test_(request.source_route, *start, request.limits,
                request.control_period_sec, request.requested_speed_scale,
                request.segment_speed_scales, &failure_reason) :
                buildCompactJointProfile(request.source_route, *start, request.limits,
                request.control_period_sec, request.requested_speed_scale,
                request.segment_speed_scales, &failure_reason);
              if (!built) {
                result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
                  failure_reason.empty() ? "source route profile build failed" : failure_reason);
              } else {
                if (built->samples.empty()) {
                  result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
                    "source profile builder returned no samples");
                } else {
                  if (request.published_prefix.size() > 1U &&
                    !prependValidatedCompactPrefix(*built, request.published_prefix,
                    request.control_period_sec, &failure_reason))
                  {
                    result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
                      "immutable prefix could not be prepended: " + failure_reason);
                  } else {
                    if (request.published_prefix.size() == 1U &&
                      ((built->samples.front().position -
                      request.published_prefix.front().position).
                      lpNorm<Eigen::Infinity>() > 1.0e-12 ||
                      (built->samples.front().velocity -
                      request.published_prefix.front().velocity).
                      lpNorm<Eigen::Infinity>() > 1.0e-12 ||
                      (built->samples.front().acceleration -
                      request.published_prefix.front().acceleration).
                      lpNorm<Eigen::Infinity>() > 1.0e-12))
                    {
                      result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
                        "single-sample immutable prefix differs from profile start");
                    } else {
                      source_profile = std::move(*built);
                    }
                  }
                }
              }
            }
          }
        }
      }
      stop_build_timer();
      if (result.profile.failure_reason.empty()) {
        result.profile = retimer_(source_profile, item.request.limits,
          item.request.control_period_sec, item.request.requested_speed_scale);
      }
    } catch (const std::exception & error) {
      result.profile = failedResult(std::string{"profile build or retimer threw: "} + error.what());
      result.return_profile.reset();
    } catch (...) {
      result.profile = failedResult("profile build or retimer threw an unknown exception");
      result.return_profile.reset();
    }
    stop_build_timer();
    if (accepted(result.profile) && result.profile.profile) {
      const auto geometry_start = std::chrono::steady_clock::now();
      result.geometry_certificate = certifyGeometry(
        *result.profile.profile, item.request.final_geometry_validator,
        item.request.scene_revision);
      result.geometry_elapsed_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - geometry_start).count();
      if (!result.geometry_certificate.passedFor(item.request.scene_revision,
        result.profile.profile->samples.size(),
        result.profile.profile->samples.size() - 1U))
      {
        result.profile = failedResult(CompactDynamicsStatus::kInvalidInput,
          "final geometry rejected profile: " +
            (result.geometry_certificate.failure_reason.empty() ?
            "certificate was incomplete or invalid" : result.geometry_certificate.failure_reason));
      }
    }

    if (accepted(result.profile) && item.request.return_profile_required &&
      !item.request.return_profile)
    {
      result.profile = failedResult(
        CompactDynamicsStatus::kInvalidInput, "mandatory return profile is missing");
    }
    if (accepted(result.profile) && item.request.return_profile) {
      try {
        result.return_profile = retimer_(*item.request.return_profile, item.request.limits,
          item.request.control_period_sec, item.request.return_speed_scale);
      } catch (const std::exception & error) {
        result.return_profile = failedResult(
          std::string{"return profile retimer threw: "} + error.what());
      } catch (...) {
        result.return_profile = failedResult("return profile retimer threw an unknown exception");
      }
      if (result.return_profile && accepted(*result.return_profile) &&
        result.return_profile->profile)
      {
        const auto geometry_start = std::chrono::steady_clock::now();
        result.return_geometry_certificate = certifyGeometry(
          *result.return_profile->profile,
          item.request.return_final_geometry_validator ?
          item.request.return_final_geometry_validator :
          item.request.final_geometry_validator,
          item.request.scene_revision);
        result.geometry_elapsed_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - geometry_start).count();
        if (!result.return_geometry_certificate->passedFor(item.request.scene_revision,
          result.return_profile->profile->samples.size(),
          result.return_profile->profile->samples.size() - 1U))
        {
          result.return_profile = failedResult(CompactDynamicsStatus::kInvalidInput,
            "final geometry rejected return profile: " +
              (result.return_geometry_certificate->failure_reason.empty() ?
              "certificate was incomplete or invalid" :
              result.return_geometry_certificate->failure_reason));
        }
      }
      if (!result.return_profile || !accepted(*result.return_profile)) {
        result.profile = failedResult(
          CompactDynamicsStatus::kInvalidInput,
          "mandatory return profile failed: " +
          (result.return_profile ? result.return_profile->failure_reason :
          std::string{"return result is missing"}));
      }
    }

    result.total_elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - total_start).count();

    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (!stopping_ && item.request_id == latest_request_id_) {
        result_ = std::move(result);
      }
    }
  }
}

}  // namespace face_tracking_arm::control
