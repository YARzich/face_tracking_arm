// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "compact_totg_time_warp.hpp"

#include "compact_path_timing.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <moveit/trajectory_processing/time_optimal_trajectory_generation.hpp>
#include <ruckig/ruckig.hpp>

#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::control::detail
{
namespace
{

constexpr double kPointEpsilon{1.0e-10};
constexpr double kPathTolerance{0.02};
constexpr double kTauRoundoffTolerance{1.0e-12};
// This is MoveIt's internal TOTG integration step, independent of queue dt.

struct TotgArithmeticReserve
{
  double velocity{0.0};
  double acceleration{0.0};
  double jerk{0.0};
};

enum class CandidateFailureKind
{
  kFatal,
  kRetryableLimit,
  kRetryableBrakeTail
};

std::optional<TotgArithmeticReserve> arithmeticReserve(
  const std::vector<Eigen::VectorXd> & path, const JointMotionLimits & limits,
  const double period_sec)
{
  double coordinate_bound = 0.0;
  double route_length = 0.0;
  for (std::size_t index = 0U; index < path.size(); ++index) {
    coordinate_bound = std::max(coordinate_bound, path[index].cwiseAbs().maxCoeff());
    if (index > 0U) {
      route_length += (path[index] - path[index - 1U]).norm();
    }
  }
  coordinate_bound = std::max(coordinate_bound,
    std::max(limits.lower_position.cwiseAbs().maxCoeff(),
      limits.upper_position.cwiseAbs().maxCoeff()));
  const double local_span = coordinate_bound + route_length + kPathTolerance;
  if (!std::isfinite(local_span) || local_span <= 0.0 ||
    !std::isfinite(period_sec) || period_sec <= 0.0)
  {
    return std::nullopt;
  }
  const auto position_error = machineRoundoffBound(2.0 * local_span, 18U);
  if (!position_error) {
    return std::nullopt;
  }
  const double velocity_error = *position_error / period_sec;
  const double acceleration_error = 2.0 * velocity_error / period_sec;
  const double jerk_error = 2.0 * acceleration_error / period_sec;
  if (!std::isfinite(velocity_error) || !std::isfinite(acceleration_error) ||
    !std::isfinite(jerk_error))
  {
    return std::nullopt;
  }
  return TotgArithmeticReserve{velocity_error, acceleration_error, jerk_error};
}

std::string diagnosticNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

bool lowerBoundExceedsSampleBudget(
  const trajectory_processing::Path & path,
  const Eigen::VectorXd & max_velocity,
  const Eigen::VectorXd & max_acceleration,
  const double control_period_sec)
{
  const double path_length = path.getLength();
  const double velocity_duration_lower_bound = path_length / max_velocity.norm();
  const double acceleration_duration_lower_bound =
    2.0 * std::sqrt(path_length / max_acceleration.norm());
  double axis_duration_lower_bound = 0.0;
  Eigen::VectorXd axis_variation = Eigen::VectorXd::Zero(max_velocity.size());
  std::vector<double> path_positions{0.0, path_length};
  for (const auto & switching_point : path.getSwitchingPoints()) {
    if (std::isfinite(switching_point.first) && switching_point.first > 0.0 &&
      switching_point.first < path_length)
    {
      path_positions.push_back(switching_point.first);
    }
  }
  std::sort(path_positions.begin(), path_positions.end());
  path_positions.erase(std::unique(path_positions.begin(), path_positions.end()),
    path_positions.end());
  for (std::size_t index = 1U; index < path_positions.size(); ++index) {
    const Eigen::VectorXd displacement =
      path.getConfig(path_positions[index]) - path.getConfig(path_positions[index - 1U]);
    double interval_duration_lower_bound = 0.0;
    for (Eigen::Index axis = 0; axis < max_velocity.size(); ++axis) {
      axis_variation[axis] += std::abs(displacement[axis]);
      interval_duration_lower_bound = std::max(interval_duration_lower_bound,
        std::abs(displacement[axis]) / max_velocity[axis]);
    }
    axis_duration_lower_bound += interval_duration_lower_bound;
  }
  double axis_acceleration_duration_lower_bound = 0.0;
  for (Eigen::Index axis = 0; axis < max_acceleration.size(); ++axis) {
    axis_acceleration_duration_lower_bound = std::max(
      axis_acceleration_duration_lower_bound,
      2.0 * std::sqrt(axis_variation[axis] / max_acceleration[axis]));
  }

  const double duration_lower_bound = std::max({
        velocity_duration_lower_bound,
        acceleration_duration_lower_bound,
        axis_duration_lower_bound,
        axis_acceleration_duration_lower_bound});
  const double tick_lower_bound = std::ceil(duration_lower_bound / control_period_sec);
  return !std::isfinite(tick_lower_bound) ||
         tick_lower_bound > static_cast<double>(path_timing::kMaximumProfileSamples - 3U);
}

void initializeSampleMetadata(CompactJointProfile & profile)
{
  profile.path_progress_rad_per_sample.assign(profile.samples.size(), 0.0);
  for (std::size_t index = 1U; index < profile.samples.size(); ++index) {
    profile.path_progress_rad_per_sample[index] =
      profile.path_progress_rad_per_sample[index - 1U] +
      (profile.samples[index].position - profile.samples[index - 1U].position).norm();
  }
}

}  // namespace

static std::optional<CompactJointProfile> buildTotgTimeWarpProfileCandidate(
  const std::vector<Eigen::VectorXd> & path,
  const std::vector<double> & relative_phase_scales,
  const JointMotionState & stationary_start,
  const JointMotionLimits & limits,
  const double control_period_sec,
  const double speed_scale,
  const double requested_speed_scale,
  const double jerk_time_acceleration_fraction,
  const bool phase_local_clock,
  const double best_duration_sec,
  std::string * failure_reason)
{
  const auto build_started = std::chrono::steady_clock::now();
  const auto fail = [failure_reason](const std::string & message) {
      if (failure_reason) {
        *failure_reason = message;
      }
      return std::optional<CompactJointProfile>{};
    };
  if (failure_reason) {
    failure_reason->clear();
  }
  const Eigen::Index count = stationary_start.position.size();
  if (path.size() < 2U || relative_phase_scales.size() + 1U != path.size() ||
    count <= 0 || stationary_start.velocity.size() != count ||
    stationary_start.acceleration.size() != count || !stationary_start.position.allFinite() ||
    !stationary_start.velocity.allFinite() || !stationary_start.acceleration.allFinite() ||
    !stationary_start.velocity.isZero(1.0e-12) ||
    !stationary_start.acceleration.isZero(1.0e-12) ||
    !std::isfinite(control_period_sec) || control_period_sec <= 0.0 ||
    !std::isfinite(speed_scale) || speed_scale <= 0.0 ||
    !std::isfinite(requested_speed_scale) || requested_speed_scale <= 0.0)
  {
    return fail("invalid TOTG time-warp input or nonstationary start");
  }
  for (std::size_t index = 0U; index < path.size(); ++index) {
    if (path[index].size() != count || !path[index].allFinite() ||
      (path[index].array() < limits.lower_position.array()).any() ||
      (path[index].array() > limits.upper_position.array()).any())
    {
      return fail("TOTG source path has invalid dimensions, values, or bounds");
    }
    if (index > 0U && (!std::isfinite(relative_phase_scales[index - 1U]) ||
      relative_phase_scales[index - 1U] <= 0.0))
    {
      return fail("TOTG source phase scale is invalid");
    }
  }

  const double position_tolerance = limits.numerical_policy ?
    limits.numerical_policy->position_error.value : 1.0e-9;
  const double velocity_tolerance = limits.numerical_policy ?
    limits.numerical_policy->velocity_error.value : 1.0e-9;
  const double acceleration_tolerance = limits.numerical_policy ?
    limits.numerical_policy->acceleration_error.value : 1.0e-9;
  const double jerk_tolerance = limits.numerical_policy ?
    limits.numerical_policy->jerk_error.value : 1.0e-9;
  const auto geometry_started = std::chrono::steady_clock::now();
  auto smooth_path = trajectory_processing::Path::create(path, kPathTolerance);
  if (!smooth_path) {
    return fail("MoveIt TOTG could not create the smoothed path");
  }
  if (lowerBoundExceedsSampleBudget(
      *smooth_path, limits.max_velocity, limits.max_acceleration, control_period_sec))
  {
    return fail("TOTG lower-bound duration exceeds the existing profile sample bound");
  }
  const auto arithmetic_reserve = arithmeticReserve(path, limits, control_period_sec);
  if (!arithmetic_reserve) {
    return fail("TOTG arithmetic reserve could not be derived from the path bounds");
  }
  Eigen::VectorXd totg_velocity(count);
  Eigen::VectorXd totg_acceleration(count);
  for (Eigen::Index axis = 0; axis < count; ++axis) {
    totg_velocity[axis] = limits.max_velocity[axis] - arithmetic_reserve->velocity;
    // Bound the TOTG acceleration from the sampled queue jerk limit.
    const double usable_acceleration =
      limits.max_acceleration[axis] - arithmetic_reserve->acceleration;
    const double usable_jerk = limits.max_jerk[axis] - arithmetic_reserve->jerk;
    if (totg_velocity[axis] <= 0.0 || usable_acceleration <= 0.0 || usable_jerk <= 0.0) {
      return fail("TOTG arithmetic reserve exhausts a joint motion limit");
    }
    totg_acceleration[axis] = phase_local_clock ? usable_acceleration :
      std::min(usable_acceleration,
        usable_jerk * control_period_sec * jerk_time_acceleration_fraction);
  }
  if (lowerBoundExceedsSampleBudget(
      *smooth_path, totg_velocity, totg_acceleration, control_period_sec))
  {
    return fail("TOTG lower-bound duration exceeds the existing profile sample bound");
  }
  auto totg = trajectory_processing::Trajectory::create(
    *smooth_path, totg_velocity, totg_acceleration, path_timing::kTotgIntegrationStepSec);
  if (!totg || !std::isfinite(totg->getDuration()) || totg->getDuration() <= 0.0) {
    return fail("MoveIt TOTG could not parameterize the smoothed path");
  }
  const double geometry_duration_sec = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - geometry_started).count();
  const double tau_duration = totg->getDuration();
  const double totg_step_count_real = std::ceil(tau_duration / control_period_sec);
  if (!std::isfinite(totg_step_count_real) || totg_step_count_real < 1.0 ||
    totg_step_count_real > static_cast<double>(path_timing::kMaximumProfileSamples - 1U))
  {
    return fail("TOTG duration exceeds the existing profile sample bound");
  }

  std::vector<double> route_prefix(path.size(), 0.0);
  for (std::size_t index = 1U; index < path.size(); ++index) {
    route_prefix[index] = route_prefix[index - 1U] +
      (path[index] - path[index - 1U]).norm();
  }
  const double route_length = route_prefix.back();
  if (!std::isfinite(route_length) || route_length <= kPointEpsilon) {
    return fail("TOTG source path has zero or invalid route length");
  }

  std::vector<double> phase_tau{0.0};
  std::vector<double> source_phase_boundaries;
  std::vector<double> phase_scales;
  double accumulated_length = 0.0;
  for (std::size_t segment = 0U; segment < relative_phase_scales.size(); ++segment) {
    const double relative_scale = relative_phase_scales[segment];
    if (phase_scales.empty()) {
      phase_scales.push_back(relative_scale);
    } else if (std::abs(phase_scales.back() - relative_scale) > 1.0e-12) {
      source_phase_boundaries.push_back(accumulated_length);
      phase_scales.push_back(relative_scale);
    }
    accumulated_length += (path[segment + 1U] - path[segment]).norm();
  }
  if (!source_phase_boundaries.empty()) {
    const double mapping_step_count_real = std::ceil(tau_duration / control_period_sec);
    if (!std::isfinite(mapping_step_count_real) || mapping_step_count_real < 1.0 ||
      mapping_step_count_real > static_cast<double>(path_timing::kMaximumProfileSamples - 1U))
    {
      return fail("TOTG phase mapping exceeds the existing profile sample bound");
    }
    const std::size_t mapping_steps = std::max<std::size_t>(
      1U, static_cast<std::size_t>(mapping_step_count_real));
    std::vector<double> tau_progress(mapping_steps + 1U, 0.0);
    double prior_source_progress = 0.0;
    for (std::size_t step = 0U; step <= mapping_steps; ++step) {
      const double tau = tau_duration * static_cast<double>(step) /
        static_cast<double>(mapping_steps);
      const Eigen::VectorXd q = totg->getPosition(tau);
      double best_distance = std::numeric_limits<double>::infinity();
      double best_progress = prior_source_progress;
      for (std::size_t segment = 0U; segment + 1U < path.size(); ++segment) {
        const Eigen::VectorXd delta = path[segment + 1U] - path[segment];
        const double length = delta.norm();
        if (length <= kPointEpsilon || route_prefix[segment + 1U] < prior_source_progress) {
          continue;
        }
        const double fraction = std::clamp((q - path[segment]).dot(delta) /
          delta.squaredNorm(), 0.0, 1.0);
        const double progress = route_prefix[segment] + fraction * length;
        if (progress + kPointEpsilon < prior_source_progress) {
          continue;
        }
        const double distance = (q - (path[segment] + fraction * delta)).norm();
        if (distance < best_distance) {
          best_distance = distance;
          best_progress = progress;
        }
      }
      prior_source_progress = std::max(prior_source_progress, best_progress);
      tau_progress[step] = prior_source_progress;
    }
    for (const double boundary : source_phase_boundaries) {
      const auto upper = std::lower_bound(tau_progress.begin(), tau_progress.end(), boundary);
      if (upper == tau_progress.end()) {
        return fail("TOTG phase boundary is outside projected source progress");
      }
      const std::size_t upper_index = static_cast<std::size_t>(
        std::distance(tau_progress.begin(), upper));
      if (upper_index == 0U) {
        return fail("TOTG phase boundary projects to the route start");
      }
      const double low_progress = tau_progress[upper_index - 1U];
      const double high_progress = tau_progress[upper_index];
      const double fraction = high_progress > low_progress ?
        (boundary - low_progress) / (high_progress - low_progress) : 0.0;
      const double low_tau = tau_duration * static_cast<double>(upper_index - 1U) /
        static_cast<double>(mapping_steps);
      const double high_tau = tau_duration * static_cast<double>(upper_index) /
        static_cast<double>(mapping_steps);
      phase_tau.push_back(low_tau + std::clamp(fraction, 0.0, 1.0) * (high_tau - low_tau));
    }
  }
  phase_tau.push_back(tau_duration);

  std::string clock_failure;
  auto clock_schedule = path_timing::buildClockSchedule(
    *totg, count, std::move(phase_tau), phase_scales, limits,
    control_period_sec, speed_scale, requested_speed_scale,
    arithmetic_reserve->velocity, arithmetic_reserve->acceleration,
    arithmetic_reserve->jerk, totg_acceleration, phase_local_clock, &clock_failure);
  if (!clock_schedule) {
    return fail(clock_failure);
  }
  auto & clocks = clock_schedule->clocks;
  phase_tau = std::move(clock_schedule->phase_tau);
  const double profile_duration = clock_schedule->duration;

  const double nominal_sample_count_real =
    std::ceil(profile_duration / control_period_sec - 1.0e-12);
  const double maximum_sample_count_real =
    std::ceil(profile_duration * 1.1482440353 / control_period_sec - 1.0e-12);
  constexpr std::size_t kMaximumCumulativeSampleWork =
    7U * path_timing::kMaximumProfileSamples;
  const std::size_t sample_count_limit = path_timing::kMaximumProfileSamples - 3U;
  if (!std::isfinite(nominal_sample_count_real) || nominal_sample_count_real < 1.0 ||
    nominal_sample_count_real > static_cast<double>(sample_count_limit) ||
    !std::isfinite(maximum_sample_count_real))
  {
    return fail("Ruckig clock duration exceeds sample bound");
  }
  const std::size_t nominal_sample_count = static_cast<std::size_t>(nominal_sample_count_real);
  const double bounded_maximum_sample_count = std::min(
    static_cast<double>(sample_count_limit), std::max(1.0, maximum_sample_count_real));
  const std::size_t maximum_sample_count =
    static_cast<std::size_t>(bounded_maximum_sample_count);
  if (maximum_sample_count < nominal_sample_count) {
    return fail("Ruckig clock duration exceeds sample bound");
  }
  const auto tau_roundoff = machineRoundoffBound(2.0 * tau_duration, 18U);
  if (!tau_roundoff) {
    return fail("TOTG endpoint clock tolerance could not be derived");
  }
  const double tau_endpoint_tolerance = std::max(kTauRoundoffTolerance, *tau_roundoff);

  std::size_t cumulative_sample_work = 0U;
  std::string last_sample_failure;
  bool count_valid = true;
  for (std::size_t sample_count = nominal_sample_count;
    sample_count <= maximum_sample_count; ++sample_count)
  {
    const std::size_t attempt_sample_work = sample_count + 3U;
    if (attempt_sample_work > kMaximumCumulativeSampleWork - cumulative_sample_work) {
      break;
    }
    const double sampled_duration = static_cast<double>(sample_count) * control_period_sec;
    if (std::isfinite(best_duration_sec) &&
      sampled_duration + 2.0 * control_period_sec >= best_duration_sec)
    {
      break;
    }
    cumulative_sample_work += attempt_sample_work;
    count_valid = true;
    if (failure_reason) {
      failure_reason->clear();
    }
    const double applied_time_dilation = sampled_duration / profile_duration;
    if (!std::isfinite(applied_time_dilation) || applied_time_dilation < 1.0) {
      return fail("outer uniform clock dilation is invalid");
    }

    CompactJointProfile profile;
    profile.samples.push_back(stationary_start);
    profile.checked_path_vertices = path;
    profile.checked_path_segment_scales = relative_phase_scales;
    Eigen::VectorXd previous_velocity = stationary_start.velocity;
    CandidateFailureKind sample_failure_kind = CandidateFailureKind::kFatal;
    const auto append_sample = [&](Eigen::VectorXd q, Eigen::VectorXd velocity) {
        const Eigen::VectorXd acceleration = (velocity - previous_velocity) / control_period_sec;
        const Eigen::VectorXd jerk = (acceleration - profile.samples.back().acceleration) /
          control_period_sec;
        const bool within_bounds = q.allFinite() && velocity.allFinite() &&
          acceleration.allFinite() && jerk.allFinite() &&
          (q.array() >= limits.lower_position.array() - position_tolerance).all() &&
          (q.array() <= limits.upper_position.array() + position_tolerance).all() &&
          (velocity.cwiseAbs().array() <= limits.max_velocity.array() + velocity_tolerance).all() &&
          (acceleration.cwiseAbs().array() <=
          limits.max_acceleration.array() + acceleration_tolerance).all() &&
          (jerk.cwiseAbs().array() <= limits.max_jerk.array() + jerk_tolerance).all();
        if (!within_bounds) {
          const auto axis_failure = [](const Eigen::Index axis,
            const std::string & kind, const double actual, const double limit) {
              return "queue_limit:" + kind + ":axis=" + std::to_string(axis) +
                     ":actual=" + diagnosticNumber(actual) +
                     ":limit=" + diagnosticNumber(limit);
            };
          for (Eigen::Index axis = 0; axis < count; ++axis) {
            if (!std::isfinite(q[axis]) ||
              q[axis] < limits.lower_position[axis] - position_tolerance ||
              q[axis] > limits.upper_position[axis] + position_tolerance)
            {
              if (failure_reason) {
                *failure_reason = axis_failure(axis, "position", q[axis],
                q[axis] < limits.lower_position[axis] ? limits.lower_position[axis] :
                limits.upper_position[axis]);
              }
              return false;
            }
            if (!std::isfinite(velocity[axis]) || !std::isfinite(acceleration[axis]) ||
              !std::isfinite(jerk[axis]))
            {
              if (failure_reason) {
                *failure_reason = "queue_limit:nonfinite derivative at axis=" +
                  std::to_string(axis);
              }
              return false;
            }
          }
          for (Eigen::Index axis = 0; axis < count; ++axis) {
            if (std::abs(velocity[axis]) > limits.max_velocity[axis] + velocity_tolerance) {
              sample_failure_kind = CandidateFailureKind::kRetryableLimit;
              if (failure_reason) {
                *failure_reason = axis_failure(axis, "velocity", velocity[axis],
                limits.max_velocity[axis]);
              }
              return false;
            }
            if (std::abs(acceleration[axis]) >
              limits.max_acceleration[axis] + acceleration_tolerance)
            {
              sample_failure_kind = CandidateFailureKind::kRetryableLimit;
              if (failure_reason) {
                *failure_reason = axis_failure(axis, "acceleration", acceleration[axis],
                limits.max_acceleration[axis]);
              }
              return false;
            }
            if (std::abs(jerk[axis]) > limits.max_jerk[axis] + jerk_tolerance) {
              sample_failure_kind = CandidateFailureKind::kRetryableLimit;
              if (failure_reason) {
                *failure_reason = axis_failure(axis, "jerk", jerk[axis], limits.max_jerk[axis]);
              }
              return false;
            }
          }
          if (failure_reason) {
            *failure_reason = "queue_limit:finite derivative exceeded its physical bound";
          }
          sample_failure_kind = CandidateFailureKind::kRetryableLimit;
          return false;
        }
        profile.samples.push_back(JointMotionState{
          std::move(q), std::move(velocity), acceleration});
        previous_velocity = profile.samples.back().velocity;
        return true;
      };

    double clock_segment_start_time = 0.0;
    std::size_t clock_index = 0U;
    double previous_tau = 0.0;
    double minimum_tau = 0.0;
    double maximum_tau = 0.0;
    double maximum_backtrack = 0.0;
    for (std::size_t tick = 1U; tick <= sample_count; ++tick) {
      const double time = static_cast<double>(tick) * control_period_sec;
      const double raw_time = time / applied_time_dilation;
      while (clock_index + 1U < clocks.size() &&
        raw_time > clock_segment_start_time + clocks[clock_index].duration + 1.0e-12)
      {
        clock_segment_start_time += clocks[clock_index].duration;
        ++clock_index;
      }
      std::array<double, 1U> tau{}, velocity{}, acceleration{};
      const auto & clock = clocks[clock_index];
      clock.trajectory.at_time(raw_time - clock_segment_start_time,
      tau, velocity, acceleration);
      const double raw_tau = tau[0];
      minimum_tau = std::min(minimum_tau, raw_tau);
      maximum_tau = std::max(maximum_tau, raw_tau);
      maximum_backtrack = std::max(maximum_backtrack, previous_tau - raw_tau);
      const double phase_end = phase_tau[clock_index + 1U];
      if (!std::isfinite(raw_tau) || raw_tau < clock.tau_start - kTauRoundoffTolerance ||
        raw_tau > phase_end + kTauRoundoffTolerance ||
        raw_tau + kTauRoundoffTolerance < previous_tau)
      {
        return fail("sampled Ruckig clock left monotone phase range: tau=" +
        diagnosticNumber(raw_tau) + ", phase=[" + diagnosticNumber(clock.tau_start) + "," +
        diagnosticNumber(phase_end) + "], min=" + diagnosticNumber(minimum_tau) +
        ", max=" + diagnosticNumber(maximum_tau) + ", backtrack=" +
        diagnosticNumber(maximum_backtrack) + ", duration=" +
        diagnosticNumber(profile_duration));
      }
      if (tick == sample_count && std::abs(raw_tau - tau_duration) > tau_endpoint_tolerance) {
        return fail("final Ruckig clock sample missed the exact TOTG endpoint: tau=" +
        diagnosticNumber(raw_tau) + ", expected=" + diagnosticNumber(tau_duration));
      }
      previous_tau = raw_tau;
      Eigen::VectorXd q = tick == sample_count ? path.back() : totg->getPosition(raw_tau);
      Eigen::VectorXd q_velocity = (q - profile.samples.back().position) / control_period_sec;
      if (!append_sample(std::move(q), std::move(q_velocity))) {
        if (sample_failure_kind == CandidateFailureKind::kRetryableLimit) {
          last_sample_failure = failure_reason ? *failure_reason :
            "queue_limit:finite derivative exceeded its physical bound";
          count_valid = false;
          break;
        }
        return fail(failure_reason ? *failure_reason : "TOTG sample rejected");
      }
      profile.speed_scale_per_interval.push_back(clock.relative_scale);
    }
    if (!count_valid) {
      if (failure_reason) {
        failure_reason->clear();
      }
      continue;
    }
    const Eigen::VectorXd stopped_velocity = Eigen::VectorXd::Zero(count);
    for (std::size_t hold = 0U; hold < 2U; ++hold) {
      if (!append_sample(path.back(), stopped_velocity)) {
        if (sample_failure_kind == CandidateFailureKind::kRetryableLimit) {
          last_sample_failure = failure_reason ? *failure_reason :
            "queue_limit:finite derivative exceeded its physical bound";
          count_valid = false;
          break;
        }
        return fail(failure_reason ? *failure_reason : "TOTG endpoint hold rejected");
      }
      profile.speed_scale_per_interval.push_back(relative_phase_scales.back());
    }
    if (!count_valid) {
      if (failure_reason) {
        failure_reason->clear();
      }
      continue;
    }
    profile.duration_sec = sampled_duration + 2.0 * control_period_sec;
    profile.path_length_rad = 0.0;
    for (std::size_t index = 1U; index < profile.samples.size(); ++index) {
      profile.path_length_rad +=
        (profile.samples[index].position - profile.samples[index - 1U].position).norm();
    }
    profile.builder_mode = "totg_time_warp";
    if (sample_count > nominal_sample_count) {
      profile.builder_mode += "_uniform_dilated";
    }
    profile.geometry_prepare_duration_sec = geometry_duration_sec;
    profile.profile_generation_duration_sec = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - build_started).count() - geometry_duration_sec;
    initializeSampleMetadata(profile);

    double retime_factor = 1.0;
    for (std::size_t interval = 0U; interval < profile.speed_scale_per_interval.size();
      ++interval)
    {
      const double local_limit_scale = std::min(
      requested_speed_scale * profile.speed_scale_per_interval[interval], 1.0);
      const auto & previous = profile.samples[interval];
      const auto & current = profile.samples[interval + 1U];
      const Eigen::VectorXd jerk = (current.acceleration - previous.acceleration) /
        control_period_sec;
      for (Eigen::Index axis = 0; axis < count; ++axis) {
        const double velocity_limit = limits.max_velocity[axis] * local_limit_scale;
        const double acceleration_limit = limits.max_acceleration[axis] *
          local_limit_scale * local_limit_scale;
        const double jerk_limit = limits.max_jerk[axis] * local_limit_scale *
          local_limit_scale * local_limit_scale;
        if (std::abs(current.velocity[axis]) > velocity_limit + velocity_tolerance) {
          retime_factor = std::min(retime_factor,
          velocity_limit / std::abs(current.velocity[axis]));
        }
        if (std::abs(current.acceleration[axis]) > acceleration_limit + acceleration_tolerance) {
          retime_factor = std::min(retime_factor,
          std::sqrt(acceleration_limit / std::abs(current.acceleration[axis])));
        }
        if (std::abs(jerk[axis]) > jerk_limit + jerk_tolerance) {
          retime_factor = std::min(retime_factor,
          std::cbrt(jerk_limit / std::abs(jerk[axis])));
        }
      }
    }
    if (retime_factor < 1.0 - 1.0e-10) {
      last_sample_failure = "queue_limit:scaled phase caps require uniform clock dilation=" +
        diagnosticNumber(1.0 / retime_factor);
      if (failure_reason) {
        failure_reason->clear();
      }
      continue;
    }

    JointMotionLimits brake_limits = limits;
    if (!brake_limits.numerical_policy) {
      const auto policy = NumericalPolicy::fromDiscreteMotionLimits(
      PositionInterval{limits.lower_position.minCoeff(), limits.upper_position.maxCoeff()},
      limits.max_velocity.maxCoeff(), limits.max_acceleration.maxCoeff(),
      limits.max_jerk.maxCoeff(), control_period_sec, SolverResidual{1.0e-5});
      if (!policy) {
        return fail("brake_tail:could not derive a policy for an unconfigured profile");
      }
      brake_limits.numerical_policy = *policy;
    }
    bool brake_tail_failed = false;
    for (std::size_t index = 0U; index < profile.samples.size(); ++index) {
      const auto & sample = profile.samples[index];
      Eigen::VectorXd direction = sample.velocity;
      if (direction.isZero(0.0)) {
        direction = sample.acceleration;
      }
      const auto tail = generateCompactProfileBrakeTail(
      sample, direction, brake_limits, EmergencyBrakeTailConfig{control_period_sec});
      if (tail.status != EmergencyBrakeTailStatus::kSuccess) {
        last_sample_failure = "brake_tail:sample=" + std::to_string(index) + " status=" +
          std::to_string(static_cast<int>(tail.status)) + " reason=" + tail.failure_reason;
        brake_tail_failed = true;
        if (failure_reason) {
          failure_reason->clear();
        }
        break;
      }
    }
    if (brake_tail_failed) {
      continue;
    }
    if (sample_count > nominal_sample_count && !last_sample_failure.empty()) {
      profile.builder_fallback_reason = "sample_count_retry: " + last_sample_failure;
    }
    return profile;
  }
  return fail(last_sample_failure.empty() ?
    "TOTG sample-count search exhausted before a valid queue was found" : last_sample_failure);
}

std::optional<CompactJointProfile> buildTotgTimeWarpProfile(
  const std::vector<Eigen::VectorXd> & path,
  const std::vector<double> & relative_phase_scales,
  const JointMotionState & stationary_start,
  const JointMotionLimits & limits,
  const double control_period_sec,
  const double speed_scale,
  const double requested_speed_scale,
  std::string * failure_reason)
{
  const auto candidates_started = std::chrono::steady_clock::now();
  bool fallback_differs = false;
  for (Eigen::Index axis = 0; axis < stationary_start.position.size(); ++axis) {
    const double fast_acceleration = std::min(limits.max_acceleration[axis],
      limits.max_jerk[axis] * control_period_sec);
    const double bounded_acceleration = std::min(limits.max_acceleration[axis],
      limits.max_jerk[axis] * control_period_sec * 0.5);
    fallback_differs = fallback_differs || fast_acceleration > bounded_acceleration;
  }
  struct CandidateSpec
  {
    const char * name;
    double jerk_time_acceleration_fraction;
    bool phase_local;
  };
  const std::array<CandidateSpec, 3U> specs{{
    {"totg_time_warp", 1.0, false},
    {"totg_phase_local_clock", 1.0, true},
    {"totg_time_warp_fallback", 0.5, false}}};
  std::optional<CompactJointProfile> fastest;
  const bool mixed_phase_scales = std::any_of(relative_phase_scales.begin(),
    relative_phase_scales.end(), [first = relative_phase_scales.front()](const double scale) {
        return std::abs(scale - first) > 1.0e-12;
    });
  std::string fast_candidate_failure;
  std::string phase_local_failure;
  std::string bounded_candidate_failure;
  for (const auto & spec : specs) {
    if (spec.phase_local && mixed_phase_scales) {
      phase_local_failure = "phase-local clock disabled for mixed source phase scales";
      continue;
    }
    if (spec.jerk_time_acceleration_fraction < 1.0 && !fallback_differs) {
      continue;
    }
    std::string candidate_failure;
    auto candidate = buildTotgTimeWarpProfileCandidate(path, relative_phase_scales,
      stationary_start, limits, control_period_sec, speed_scale, requested_speed_scale,
      spec.jerk_time_acceleration_fraction, spec.phase_local,
      fastest ? fastest->duration_sec : std::numeric_limits<double>::infinity(),
      &candidate_failure);
    if (candidate) {
      const bool uniform_dilated = candidate->builder_mode.find("_uniform_dilated") !=
        std::string::npos;
      const bool is_fast = std::string(spec.name) == "totg_time_warp";
      std::string selected_mode = spec.name;
      if (!is_fast && !spec.phase_local &&
        spec.jerk_time_acceleration_fraction < 1.0)
      {
        selected_mode = fast_candidate_failure.rfind("queue_limit:jerk:", 0U) == 0U ?
          "totg_time_warp_fallback_jerk" : "totg_time_warp_fallback";
      }
      candidate->builder_mode = selected_mode;
      if (uniform_dilated) {
        candidate->builder_mode += "_uniform_dilated";
      }
      if (!is_fast) {
        const std::string sample_count_retry_reason = candidate->builder_fallback_reason;
        candidate->builder_fallback_reason = !fast_candidate_failure.empty() ?
          fast_candidate_failure : phase_local_failure;
        if (!sample_count_retry_reason.empty()) {
          candidate->builder_fallback_reason +=
            (candidate->builder_fallback_reason.empty() ? "" : "; ") +
            sample_count_retry_reason;
        }
      }
      if (!fastest || candidate->duration_sec < fastest->duration_sec) {
        fastest = std::move(candidate);
      }
    } else if (std::string(spec.name) == "totg_time_warp") {
      fast_candidate_failure = candidate_failure;
    } else if (spec.phase_local) {
      phase_local_failure = candidate_failure;
    } else {
      bounded_candidate_failure = candidate_failure;
    }
  }
  if (!fastest) {
    if (failure_reason) {
      *failure_reason = "TOTG clock candidates failed; fast=" + fast_candidate_failure +
        "; phase_local=" + phase_local_failure + "; bounded=" +
        bounded_candidate_failure;
    }
    return std::nullopt;
  }
  fastest->profile_generation_duration_sec = std::max(0.0,
    std::chrono::duration<double>(std::chrono::steady_clock::now() - candidates_started).count() -
    fastest->geometry_prepare_duration_sec);
  if (failure_reason) {
    failure_reason->clear();
  }
  return fastest;
}

}  // namespace face_tracking_arm::control::detail
