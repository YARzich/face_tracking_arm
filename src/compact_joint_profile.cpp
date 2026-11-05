// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/compact_joint_profile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>

#include <ruckig/ruckig.hpp>
#include "face_tracking_arm/detail/scalar_braking_policy.hpp"
#include "compact_totg_time_warp.hpp"

namespace face_tracking_arm::control
{
namespace
{

constexpr double kPointEpsilon{1.0e-10};
constexpr double kCollinearTolerance{1.0e-10};
constexpr double kInitialMotionTolerance{1.0e-6};
constexpr std::size_t kMaximumProfileSamples{100000U};
thread_local std::size_t compact_profile_build_depth{0U};
thread_local double compact_profile_requested_scale{1.0};

struct CompactProfileBuildDepthGuard
{
  explicit CompactProfileBuildDepthGuard(const double scale)
  : is_root(compact_profile_build_depth == 0U)
  {
    if (is_root) {
      compact_profile_requested_scale = scale;
    }
    ++compact_profile_build_depth;
  }
  ~CompactProfileBuildDepthGuard()
  {
    --compact_profile_build_depth;
    if (is_root) {
      compact_profile_requested_scale = 1.0;
    }
  }
  bool is_root;
};

std::string diagnosticNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}

bool finiteLimits(const JointMotionLimits & limits, const Eigen::Index count)
{
  return limits.lower_position.size() == count && limits.upper_position.size() == count &&
         limits.max_velocity.size() == count && limits.max_acceleration.size() == count &&
         limits.max_jerk.size() == count && limits.lower_position.allFinite() &&
         limits.upper_position.allFinite() && limits.max_velocity.allFinite() &&
         limits.max_acceleration.allFinite() && limits.max_jerk.allFinite() &&
         (limits.upper_position.array() > limits.lower_position.array()).all() &&
         (limits.max_velocity.array() > 0.0).all() &&
         (limits.max_acceleration.array() > 0.0).all() &&
         (limits.max_jerk.array() > 0.0).all();
}

struct ProfileArithmeticReserve
{
  double velocity{0.0};
  double acceleration{0.0};
  double jerk{0.0};
};

std::optional<ProfileArithmeticReserve> profileArithmeticReserve(
  const double local_span, const double period_sec)
{
  if (!std::isfinite(local_span) || local_span <= 0.0 ||
    !std::isfinite(period_sec) || period_sec <= 0.0)
  {
    return std::nullopt;
  }
  // Bound two local Ruckig sample positions, including cubic at_time
  // evaluation, subtraction, direction projection and division. The same
  // local position error is propagated through the queue's finite differences.
  const auto position_difference_error = machineRoundoffBound(2.0 * local_span, 18U);
  if (!position_difference_error) {
    return std::nullopt;
  }
  const double velocity_reserve = *position_difference_error / period_sec;
  const double acceleration_reserve = 2.0 * velocity_reserve / period_sec;
  const double jerk_reserve = 2.0 * acceleration_reserve / period_sec;
  if (!std::isfinite(velocity_reserve) || !std::isfinite(acceleration_reserve) ||
    !std::isfinite(jerk_reserve))
  {
    return std::nullopt;
  }
  return ProfileArithmeticReserve{velocity_reserve, acceleration_reserve, jerk_reserve};
}

struct SimplifiedPath
{
  std::vector<Eigen::VectorXd> points;
  std::vector<double> speed_scales;
};

SimplifiedPath simplifyPath(
  const std::vector<Eigen::VectorXd> & path,
  const std::vector<double> & segment_speed_scales)
{
  SimplifiedPath result;
  result.points.reserve(path.size());
  result.speed_scales.reserve(path.size() - 1U);
  result.points.push_back(path.front());
  for (std::size_t segment = 0U; segment + 1U < path.size(); ++segment) {
    const Eigen::VectorXd delta = path[segment + 1U] - path[segment];
    if (delta.norm() <= kPointEpsilon) {
      continue;
    }
    const double scale = segment_speed_scales.empty() ? 1.0 :
      segment_speed_scales[segment];
    if (result.points.size() > 1U && !result.speed_scales.empty()) {
      const Eigen::VectorXd incoming = result.points.back() -
        result.points[result.points.size() - 2U];
      const double norm_product = incoming.norm() * delta.norm();
      const bool collinear = norm_product > 0.0 &&
        incoming.dot(delta) / norm_product >= 1.0 - kCollinearTolerance;
      if (collinear && std::abs(result.speed_scales.back() - scale) <= 1.0e-12) {
        result.points.back() = path[segment + 1U];
        continue;
      }
    }
    result.points.push_back(path[segment + 1U]);
    result.speed_scales.push_back(scale);
  }
  return result;
}

void initializeProfilePathMetadata(CompactJointProfile & profile)
{
  profile.path_progress_rad_per_sample.assign(profile.samples.size(), 0.0);
  for (std::size_t index = 1U; index < profile.samples.size(); ++index) {
    profile.path_progress_rad_per_sample[index] =
      profile.path_progress_rad_per_sample[index - 1U] +
      (profile.samples[index].position - profile.samples[index - 1U].position).norm();
  }
  const std::size_t interval_count = profile.samples.empty() ? 0U : profile.samples.size() - 1U;
  if (profile.speed_scale_per_interval.size() == interval_count) {
    return;
  }
  profile.speed_scale_per_interval.assign(interval_count, 1.0);
  if (profile.leg_end_sample_indices.size() != profile.checked_path_segment_scales.size()) {
    return;
  }
  std::size_t interval_start = 0U;
  for (std::size_t leg = 0U; leg < profile.leg_end_sample_indices.size(); ++leg) {
    const double scale = profile.checked_path_segment_scales[leg];
    const std::size_t interval_end = std::min(
      profile.leg_end_sample_indices[leg], interval_count);
    for (std::size_t interval = interval_start; interval < interval_end; ++interval) {
      profile.speed_scale_per_interval[interval] = scale;
    }
    interval_start = interval_end;
  }
}

}  // namespace

std::optional<double> projectCompactProfileProgress(
  const CompactJointProfile & profile,
  const Eigen::VectorXd & position,
  const Eigen::VectorXd & velocity,
  const double minimum_progress_rad,
  const double maximum_progress_rad,
  const double residual_limit_rad)
{
  if (profile.samples.size() < 2U || position.size() <= 0 || velocity.size() != position.size() ||
    !position.allFinite() || !velocity.allFinite() || !std::isfinite(minimum_progress_rad) ||
    !std::isfinite(maximum_progress_rad) || minimum_progress_rad < 0.0 ||
    maximum_progress_rad < minimum_progress_rad || !std::isfinite(residual_limit_rad) ||
    residual_limit_rad < 0.0)
  {
    return std::nullopt;
  }
  std::vector<double> fallback_progress;
  const std::vector<double> * progress = &profile.path_progress_rad_per_sample;
  if (progress->size() != profile.samples.size()) {
    fallback_progress.assign(profile.samples.size(), 0.0);
    for (std::size_t index = 1U; index < profile.samples.size(); ++index) {
      fallback_progress[index] = fallback_progress[index - 1U] +
        (profile.samples[index].position - profile.samples[index - 1U].position).norm();
    }
    progress = &fallback_progress;
  }
  constexpr double kTieTolerance{1.0e-9};
  constexpr double kStationaryVelocityTolerance{1.0e-9};
  double best_residual = std::numeric_limits<double>::infinity();
  double best_progress = 0.0;
  bool ambiguous = false;
  for (std::size_t interval = 0U; interval + 1U < profile.samples.size(); ++interval) {
    const double start_progress = (*progress)[interval];
    const double end_progress = (*progress)[interval + 1U];
    if (end_progress < minimum_progress_rad || start_progress > maximum_progress_rad) {
      continue;
    }
    const auto & start = profile.samples[interval].position;
    const auto & end = profile.samples[interval + 1U].position;
    if (start.size() != position.size() || end.size() != position.size()) {
      return std::nullopt;
    }
    const Eigen::VectorXd delta = end - start;
    const double length_squared = delta.squaredNorm();
    if (length_squared <= std::numeric_limits<double>::epsilon()) {
      continue;
    }
    const double low_fraction = std::clamp(
      (minimum_progress_rad - start_progress) / (end_progress - start_progress), 0.0, 1.0);
    const double high_fraction = std::clamp(
      (maximum_progress_rad - start_progress) / (end_progress - start_progress), 0.0, 1.0);
    const double fraction = std::clamp((position - start).dot(delta) / length_squared,
      low_fraction, high_fraction);
    const Eigen::VectorXd tangent = delta.normalized();
    if (velocity.norm() > kStationaryVelocityTolerance && velocity.dot(tangent) < 0.0) {
      continue;
    }
    const double residual = (position - (start + fraction * delta)).norm();
    const double projected_progress = start_progress + fraction * (end_progress - start_progress);
    if (residual + kTieTolerance < best_residual) {
      best_residual = residual;
      best_progress = projected_progress;
      ambiguous = false;
    } else {
      if (std::abs(residual - best_residual) <= kTieTolerance &&
        std::abs(projected_progress - best_progress) > kTieTolerance)
      {
        ambiguous = true;
      }
    }
  }
  if (!std::isfinite(best_residual) || best_residual > residual_limit_rad || ambiguous) {
    return std::nullopt;
  }
  return best_progress;
}

std::optional<std::vector<double>> assignCompactPhaseScales(
  const std::vector<Eigen::VectorXd> & path,
  const std::vector<double> & assembly_radii_m,
  const double radius_floor_m,
  const double fold_speed_scale,
  const double turn_speed_scale)
{
  if (path.empty() || assembly_radii_m.size() != path.size() ||
    !std::isfinite(radius_floor_m) || radius_floor_m < 0.0 ||
    !std::isfinite(fold_speed_scale) || fold_speed_scale <= 0.0 ||
    !std::isfinite(turn_speed_scale) || turn_speed_scale <= 0.0)
  {
    return std::nullopt;
  }
  const Eigen::Index dimension = path.front().size();
  for (std::size_t index = 0U; index < path.size(); ++index) {
    if (path[index].size() != dimension || !path[index].allFinite() ||
      !std::isfinite(assembly_radii_m[index]) || assembly_radii_m[index] < 0.0)
    {
      return std::nullopt;
    }
  }
  std::vector<double> scales(path.size() - 1U);
  if (assembly_radii_m.front() <= radius_floor_m) {
    std::fill(scales.begin(), scales.end(), turn_speed_scale);
    return scales;
  }
  std::size_t floor_index = 1U;
  while (floor_index < assembly_radii_m.size() &&
    assembly_radii_m[floor_index] > radius_floor_m)
  {
    ++floor_index;
  }
  if (floor_index == assembly_radii_m.size()) {
    std::fill(scales.begin(), scales.end(), fold_speed_scale);
    return scales;
  }
  std::size_t leg_start = floor_index - 1U;
  while (leg_start > 0U &&
    (path[leg_start + 1U] - path[leg_start]).norm() <= kPointEpsilon)
  {
    --leg_start;
  }
  if ((path[leg_start + 1U] - path[leg_start]).norm() <= kPointEpsilon) {
    std::fill(scales.begin(), scales.end(), turn_speed_scale);
    return scales;
  }
  std::size_t leg_end = leg_start + 1U;
  std::size_t fold_end = leg_start;
  for (std::size_t segment = leg_end; segment + 1U < path.size(); ++segment) {
    const Eigen::VectorXd delta = path[segment + 1U] - path[segment];
    if (delta.norm() <= kPointEpsilon) {
      continue;
    }
    const Eigen::VectorXd incoming = path[leg_end] - path[leg_start];
    const double norm_product = incoming.norm() * delta.norm();
    const bool collinear = norm_product > 0.0 &&
      incoming.dot(delta) / norm_product >= 1.0 - kCollinearTolerance;
    if (!collinear) {
      break;
    }
    fold_end = segment;
    leg_end = segment + 1U;
  }
  std::fill(scales.begin(), scales.begin() + static_cast<std::ptrdiff_t>(fold_end + 1U),
    fold_speed_scale);
  std::fill(scales.begin() + static_cast<std::ptrdiff_t>(fold_end + 1U), scales.end(),
    turn_speed_scale);
  return scales;
}

std::optional<CompactJointProfile> buildCompactJointProfile(
  const std::vector<Eigen::VectorXd> & path,
  const JointMotionState & start,
  const JointMotionLimits & limits,
  const double control_period_sec,
  const double speed_scale,
  const std::vector<double> & segment_speed_scales,
  std::string * failure_reason)
{
  const CompactProfileBuildDepthGuard build_depth_guard(speed_scale);
  const auto build_started = std::chrono::steady_clock::now();
  const auto fail = [failure_reason](std::string message) {
      if (failure_reason) {
        *failure_reason = std::move(message);
      }
      return std::optional<CompactJointProfile>{};
    };
  if (failure_reason) {
    failure_reason->clear();
  }
  const Eigen::Index count = start.position.size();
  if (count <= 0 || path.empty() || start.velocity.size() != count ||
    start.acceleration.size() != count || !start.position.allFinite() ||
    !start.velocity.allFinite() || !start.acceleration.allFinite() ||
    !finiteLimits(limits, count) || !std::isfinite(control_period_sec) ||
    control_period_sec <= 0.0 || !std::isfinite(speed_scale) || speed_scale <= 0.0)
  {
    return fail("invalid input dimensions, finite values, limits, period, or speed scale");
  }
  if ((start.position.array() < limits.lower_position.array()).any() ||
    (start.position.array() > limits.upper_position.array()).any())
  {
    return fail("start position outside shared position bounds");
  }
  for (const auto & point : path) {
    if (point.size() != count || !point.allFinite()) {
      return fail("path point has invalid dimensions or nonfinite values");
    }
  }
  if ((path.front() - start.position).lpNorm<Eigen::Infinity>() > kPointEpsilon) {
    return fail("path start does not match queue position");
  }
  if (!segment_speed_scales.empty() && segment_speed_scales.size() != path.size() - 1U) {
    return fail("segment phase scale count does not match path");
  }
  const SimplifiedPath simplified = simplifyPath(path, segment_speed_scales);
  auto points = simplified.points;
  auto leg_speed_scales = simplified.speed_scales;
  if (std::any_of(segment_speed_scales.begin(), segment_speed_scales.end(),
    [](const double scale) {return !std::isfinite(scale) || scale <= 0.0;}))
  {
    return fail("path has invalid phase scale");
  }
  if (points.size() == 1U) {
    if (!start.velocity.isZero(0.0) || !start.acceleration.isZero(0.0)) {
      return fail("single-point path cannot hold a moving queue state");
    }
    CompactJointProfile hold;
    hold.samples.push_back(start);
    hold.samples.push_back(JointMotionState{
        start.position, Eigen::VectorXd::Zero(count), Eigen::VectorXd::Zero(count)});
    hold.checked_path_vertices = points;
    hold.duration_sec = control_period_sec;
    initializeProfilePathMetadata(hold);
    return hold;
  }
  if (points.size() < 2U) {
    return fail("path is degenerate");
  }

  CompactJointProfile result;
  result.samples.push_back(start);
  result.checked_path_vertices = points;
  result.checked_path_segment_scales = leg_speed_scales;
  result.leg_end_sample_indices.reserve(points.size() - 1U);
  Eigen::VectorXd previous_velocity = start.velocity;
  double elapsed = 0.0;
  const double position_tolerance = limits.numerical_policy ?
    limits.numerical_policy->position_error.value : 1.0e-9;
  const double velocity_tolerance = limits.numerical_policy ?
    limits.numerical_policy->velocity_error.value : 1.0e-9;
  const double acceleration_tolerance = limits.numerical_policy ?
    limits.numerical_policy->acceleration_error.value : 1.0e-9;
  const double jerk_tolerance = limits.numerical_policy ?
    limits.numerical_policy->jerk_error.value : 1.0e-9;
  const auto effective_scale_for_leg = [&](const std::size_t leg) {
      const double phase_scale = leg_speed_scales[leg];
      return std::min(speed_scale * phase_scale, 1.0);
    };
  const auto append_discrete_sample = [&](Eigen::VectorXd next_position,
    Eigen::VectorXd next_velocity) {
      Eigen::VectorXd next_acceleration = (next_velocity - previous_velocity) /
        control_period_sec;
      const Eigen::VectorXd jerk = (next_acceleration -
        result.samples.back().acceleration) / control_period_sec;
      if (!next_position.allFinite() || !next_velocity.allFinite() ||
        !next_acceleration.allFinite() || !jerk.allFinite() ||
        (next_position.array() < limits.lower_position.array() - position_tolerance).any() ||
        (next_position.array() > limits.upper_position.array() + position_tolerance).any() ||
        (next_velocity.cwiseAbs().array() >
        limits.max_velocity.array() + velocity_tolerance).any() ||
        (next_acceleration.cwiseAbs().array() >
        limits.max_acceleration.array() + acceleration_tolerance).any() ||
        (jerk.cwiseAbs().array() > limits.max_jerk.array() + jerk_tolerance).any())
      {
        if (failure_reason) {
          Eigen::Index axis = 0;
          std::string constraint = "nonfinite";
          double value = std::numeric_limits<double>::quiet_NaN();
          double bound = std::numeric_limits<double>::quiet_NaN();
          double excess = std::numeric_limits<double>::quiet_NaN();
          if (next_position.allFinite() && next_velocity.allFinite() &&
            next_acceleration.allFinite() && jerk.allFinite())
          {
            for (; axis < count; ++axis) {
              if (next_position[axis] < limits.lower_position[axis] - position_tolerance ||
                next_position[axis] > limits.upper_position[axis] + position_tolerance)
              {
                constraint = "position";
                value = next_position[axis];
                bound = value < limits.lower_position[axis] ?
                  limits.lower_position[axis] : limits.upper_position[axis];
                excess = value < bound ? bound - value : value - bound;
                break;
              }
              if (std::abs(next_velocity[axis]) >
                limits.max_velocity[axis] + velocity_tolerance)
              {
                constraint = "velocity";
                value = std::abs(next_velocity[axis]);
                bound = limits.max_velocity[axis];
                excess = value - bound;
                break;
              }
              if (std::abs(next_acceleration[axis]) >
                limits.max_acceleration[axis] + acceleration_tolerance)
              {
                constraint = "acceleration";
                value = std::abs(next_acceleration[axis]);
                bound = limits.max_acceleration[axis];
                excess = value - bound;
                break;
              }
              if (std::abs(jerk[axis]) > limits.max_jerk[axis] + jerk_tolerance) {
                constraint = "jerk";
                value = std::abs(jerk[axis]);
                bound = limits.max_jerk[axis];
                excess = value - bound;
                break;
              }
            }
          }
          *failure_reason = "discrete sample violates " + constraint + " at axis " +
            std::to_string(axis) + ": value=" + diagnosticNumber(value) +
            ", limit=" + diagnosticNumber(bound) +
            ", excess=" + diagnosticNumber(excess);
        }
        return false;
      }
      result.samples.push_back(JointMotionState{
        std::move(next_position), std::move(next_velocity), std::move(next_acceleration)});
      previous_velocity = result.samples.back().velocity;
      return true;
    };

  const Eigen::VectorXd first_direction = (points[1U] - points.front()).normalized();
  const Eigen::VectorXd continuous_start_velocity = start.velocity +
    0.5 * control_period_sec * start.acceleration;
  const double projected_velocity = continuous_start_velocity.dot(first_direction);
  const double projected_acceleration = start.acceleration.dot(first_direction);
  if ((continuous_start_velocity - projected_velocity * first_direction).norm() >
    kInitialMotionTolerance ||
    (start.acceleration - projected_acceleration * first_direction).norm() >
    kInitialMotionTolerance)
  {
    return fail("moving initial state is not aligned with the first path segment; "
      "a validated published braking prefix is required");
  }

  const bool uniform_scale = std::all_of(leg_speed_scales.begin(), leg_speed_scales.end(),
      [&](const double value) {return std::abs(value - leg_speed_scales.front()) <= 1.0e-12;});
  const Eigen::VectorXd route_delta = points.back() - points.front();
  const double route_chord = route_delta.norm();
  bool collinear_route = points.size() == 2U && std::isfinite(route_chord) &&
    route_chord > kPointEpsilon;
  if (collinear_route) {
    const Eigen::VectorXd route_direction = route_delta / route_chord;
    double prior_progress = 0.0;
    for (std::size_t index = 1U; index < points.size(); ++index) {
      const Eigen::VectorXd offset = points[index] - points.front();
      const double route_progress = offset.dot(route_direction);
      if ((offset - route_progress * route_direction).norm() > kCollinearTolerance ||
        route_progress + kPointEpsilon < prior_progress)
      {
        collinear_route = false;
        break;
      }
      prior_progress = route_progress;
    }
  }

  // Keep the established scalar path for a straight route with one uniform
  // phase. Curved paths use MoveIt's checked TOTG geometry and a scalar Ruckig
  // time warp, so sampling never reintroduces a stop at a source waypoint.
  if (!(collinear_route && uniform_scale)) {
    return detail::buildTotgTimeWarpProfile(
      points, leg_speed_scales, start, limits, control_period_sec, speed_scale,
      compact_profile_requested_scale, failure_reason);
  }

  for (std::size_t segment = 0U; segment + 1U < points.size(); ++segment) {
    const Eigen::VectorXd delta = points[segment + 1U] - points[segment];
    const double length = delta.norm();
    if (!std::isfinite(length) || length <= kPointEpsilon) {
      return fail("timed segment " + std::to_string(segment) +
        " has invalid length");
    }
    const Eigen::VectorXd direction = delta / length;
    double scalar_velocity = 0.0;
    double scalar_acceleration = 0.0;
    if (segment == 0U) {
      // The queue stores the endpoint velocity of the previous discrete
      // interval. Ruckig starts with a continuous velocity; offset its initial
      // value by half a period of the known acceleration so the first interval
      // average reproduces the queue's backward-difference acceleration.
      const Eigen::VectorXd continuous_start_velocity = start.velocity +
        0.5 * control_period_sec * start.acceleration;
      scalar_velocity = continuous_start_velocity.dot(direction);
      scalar_acceleration = start.acceleration.dot(direction);
      if ((continuous_start_velocity - scalar_velocity * direction).norm() >
        kInitialMotionTolerance ||
        (start.acceleration - scalar_acceleration * direction).norm() >
        kInitialMotionTolerance)
      {
        return fail("timed segment " + std::to_string(segment) +
          " cannot align queue velocity/acceleration with its direction");
      }
    }

    double maximum_velocity = std::numeric_limits<double>::infinity();
    double maximum_acceleration = std::numeric_limits<double>::infinity();
    double maximum_jerk = std::numeric_limits<double>::infinity();
    for (Eigen::Index axis = 0; axis < count; ++axis) {
      const double coefficient = std::abs(direction[axis]);
      if (coefficient <= kPointEpsilon) {
        continue;
      }
      maximum_velocity = std::min(
        maximum_velocity, limits.max_velocity[axis] / coefficient);
      maximum_acceleration = std::min(
        maximum_acceleration, limits.max_acceleration[axis] / coefficient);
      maximum_jerk = std::min(maximum_jerk, limits.max_jerk[axis] / coefficient);
    }
    const double segment_speed_scale = effective_scale_for_leg(segment);
    maximum_velocity *= segment_speed_scale;
    maximum_acceleration *= segment_speed_scale * segment_speed_scale;
    maximum_jerk *= segment_speed_scale * segment_speed_scale * segment_speed_scale;
    if (!std::isfinite(maximum_velocity) || !std::isfinite(maximum_acceleration) ||
      !std::isfinite(maximum_jerk) || maximum_velocity <= 0.0 ||
      maximum_acceleration <= 0.0 || maximum_jerk <= 0.0)
    {
      return fail("timed segment " + std::to_string(segment) +
        " has invalid effective velocity/acceleration/jerk limits");
    }

    ruckig::Ruckig<1> generator(control_period_sec);
    ruckig::InputParameter<1> input;
    input.current_position[0] = 0.0;
    input.current_velocity[0] = scalar_velocity;
    input.current_acceleration[0] = scalar_acceleration;
    input.target_position[0] = length;
    input.target_velocity[0] = 0.0;
    input.target_acceleration[0] = 0.0;
    input.max_velocity[0] = maximum_velocity;
    input.max_acceleration[0] = maximum_acceleration;
    input.max_jerk[0] = maximum_jerk;
    input.duration_discretization = ruckig::DurationDiscretization::Discrete;
    ruckig::Trajectory<1> trajectory;
    ruckig::Result calculation = generator.calculate(input, trajectory);
    if (calculation != ruckig::Result::Working && calculation != ruckig::Result::Finished) {
      return fail("segment " + std::to_string(segment) +
        " Ruckig calculation failed with result " +
        std::to_string(static_cast<int>(calculation)));
    }
    double trajectory_duration = trajectory.get_duration();
    if (!std::isfinite(trajectory_duration) || trajectory_duration <= 0.0) {
      return fail("segment " + std::to_string(segment) + " Ruckig duration is invalid");
    }
    const double base_velocity = input.max_velocity[0];
    const double base_acceleration = input.max_acceleration[0];
    const double base_jerk = input.max_jerk[0];
    double duration_bound = trajectory_duration + control_period_sec;
    bool reserve_converged = false;
    for (std::size_t attempt = 0U; attempt < 4U; ++attempt) {
      const double duration_squared = duration_bound * duration_bound;
      const double duration_cubed = duration_squared * duration_bound;
      const double span_bound = length + base_velocity * duration_bound +
        0.5 * base_acceleration * duration_squared + base_jerk * duration_cubed / 6.0;
      const auto reserve = profileArithmeticReserve(span_bound, control_period_sec);
      if (!reserve) {
        return fail("segment " + std::to_string(segment) +
          " local arithmetic reserve is invalid");
      }
      input.max_velocity[0] = base_velocity - reserve->velocity;
      input.max_acceleration[0] = base_acceleration - reserve->acceleration;
      input.max_jerk[0] = base_jerk - reserve->jerk;
      if (input.max_velocity[0] <= 0.0 || input.max_acceleration[0] <= 0.0 ||
        input.max_jerk[0] <= 0.0)
      {
        return fail("segment " + std::to_string(segment) +
          " local arithmetic reserve exhausts generator limits");
      }
      calculation = generator.calculate(input, trajectory);
      if (calculation != ruckig::Result::Working && calculation != ruckig::Result::Finished) {
        return fail("segment " + std::to_string(segment) +
          " Ruckig calculation failed after local arithmetic reserve with result " +
          std::to_string(static_cast<int>(calculation)));
      }
      trajectory_duration = trajectory.get_duration();
      if (!std::isfinite(trajectory_duration) || trajectory_duration <= 0.0) {
        return fail("segment " + std::to_string(segment) +
          " Ruckig duration is invalid after local arithmetic reserve");
      }
      if (trajectory_duration <= duration_bound) {
        reserve_converged = true;
        break;
      }
      duration_bound = trajectory_duration + control_period_sec;
    }
    if (!reserve_converged) {
      return fail("segment " + std::to_string(segment) +
        " local arithmetic reserve did not converge on discrete duration");
    }
    const double step_count = trajectory_duration / control_period_sec;
    if (!std::isfinite(step_count) || step_count >
      static_cast<double>(kMaximumProfileSamples - result.samples.size()))
    {
      return fail("segment " + std::to_string(segment) + " duration exceeds sample bound");
    }
    const std::size_t steps = static_cast<std::size_t>(std::llround(step_count));
    if (steps == 0U || steps > kMaximumProfileSamples - result.samples.size()) {
      return fail("segment " + std::to_string(segment) + " has zero or excessive sample count");
    }
    double previous_scalar_position = 0.0;
    for (std::size_t step = 1U; step <= steps; ++step) {
      const double local_time = static_cast<double>(step) * control_period_sec;
      std::array<double, 1U> scalar_position{};
      std::array<double, 1U> scalar_speed{};
      std::array<double, 1U> scalar_accel{};
      trajectory.at_time(local_time, scalar_position, scalar_speed, scalar_accel);
      if (scalar_position[0] > length + kPointEpsilon) {
        return fail("segment " + std::to_string(segment) +
          " scalar position passes segment endpoint: " + std::to_string(scalar_position[0]) +
          " of " + std::to_string(length));
      }
      double local_scalar_position = scalar_position[0];
      if (step == steps) {
        local_scalar_position = length;
      }
      const double scalar_step = local_scalar_position - previous_scalar_position;
      // Multiply the scalar displacement by each small direction coefficient
      // before dividing. This keeps intermediate rounding at the per-joint
      // scale used by the shared discrete arithmetic policy.
      Eigen::VectorXd next_velocity = (scalar_step * direction) / control_period_sec;
      Eigen::VectorXd next_position = result.samples.back().position +
        next_velocity * control_period_sec;
      if (!append_discrete_sample(std::move(next_position), std::move(next_velocity))) {
        return fail("segment " + std::to_string(segment) + ": " +
                 (failure_reason ? *failure_reason : std::string("discrete sample rejected")));
      }
      previous_scalar_position = local_scalar_position;
    }
    const Eigen::VectorXd endpoint = result.samples.back().position;
    const Eigen::VectorXd stopped_velocity = Eigen::VectorXd::Zero(count);
    // The queue derives acceleration from the velocity command. Two stationary
    // samples let that discrete acceleration return to zero before a corner.
    if (!append_discrete_sample(endpoint, stopped_velocity) ||
      !append_discrete_sample(endpoint, stopped_velocity))
    {
      return fail("segment " + std::to_string(segment) +
        " corner stop violates discrete limits: " +
               (failure_reason ? *failure_reason : std::string("sample rejected")));
    }
    result.leg_end_sample_indices.push_back(result.samples.size() - 1U);
    elapsed += trajectory_duration + 2.0 * control_period_sec;
  }
  result.duration_sec = elapsed;
  for (std::size_t index = 1U; index < result.samples.size(); ++index) {
    result.path_length_rad +=
      (result.samples[index].position - result.samples[index - 1U].position).norm();
  }
  initializeProfilePathMetadata(result);
  result.builder_mode = "straight_scalar";
  result.profile_generation_duration_sec = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - build_started).count();
  return result;
}

bool prependValidatedCompactPrefix(
  CompactJointProfile & continuation,
  const std::vector<JointMotionState> & prefix,
  const double control_period_sec,
  std::string * failure_reason)
{
  const auto fail = [failure_reason](const std::string & message) {
      if (failure_reason) {
        *failure_reason = message;
      }
      return false;
    };
  if (failure_reason) {
    failure_reason->clear();
  }
  if (prefix.size() < 2U || continuation.samples.empty() ||
    !std::isfinite(control_period_sec) || control_period_sec <= 0.0)
  {
    return fail("prefix or continuation has invalid size or period");
  }
  const auto & endpoint = prefix.back();
  const auto & continuation_start = continuation.samples.front();
  if (endpoint.position.size() <= 0 ||
    endpoint.velocity.size() != endpoint.position.size() ||
    endpoint.acceleration.size() != endpoint.position.size() ||
    continuation_start.position.size() != endpoint.position.size() ||
    continuation_start.velocity.size() != endpoint.position.size() ||
    continuation_start.acceleration.size() != endpoint.position.size() ||
    !endpoint.position.allFinite() || !endpoint.velocity.allFinite() ||
    !endpoint.acceleration.allFinite() || !continuation_start.position.allFinite() ||
    !continuation_start.velocity.allFinite() || !continuation_start.acceleration.allFinite() ||
    (endpoint.position - continuation_start.position).lpNorm<Eigen::Infinity>() > 1.0e-12 ||
    (endpoint.velocity - continuation_start.velocity).lpNorm<Eigen::Infinity>() > 1.0e-12 ||
    (endpoint.acceleration - continuation_start.acceleration).lpNorm<Eigen::Infinity>() > 1.0e-12 ||
    !endpoint.velocity.isZero(1.0e-12) || !endpoint.acceleration.isZero(1.0e-12))
  {
    return fail("published prefix does not end at the continuation's stationary start");
  }
  const Eigen::Index count = endpoint.position.size();
  for (const auto & sample : prefix) {
    if (sample.position.size() != count || sample.velocity.size() != count ||
      sample.acceleration.size() != count || !sample.position.allFinite() ||
      !sample.velocity.allFinite() || !sample.acceleration.allFinite())
    {
      return fail("published prefix contains invalid state dimensions or values");
    }
  }
  if (prefix.size() + continuation.samples.size() - 1U > kMaximumProfileSamples) {
    return fail("combined profile exceeds sample bound");
  }
  const double prefix_duration = static_cast<double>(prefix.size() - 1U) *
    control_period_sec;
  const double combined_duration = continuation.duration_sec + prefix_duration;
  if (!std::isfinite(prefix_duration) || !std::isfinite(combined_duration)) {
    return fail("combined profile duration is nonfinite");
  }
  double prefix_length = 0.0;
  for (std::size_t index = 1U; index < prefix.size(); ++index) {
    prefix_length += (prefix[index].position - prefix[index - 1U].position).norm();
  }
  if (!std::isfinite(prefix_length)) {
    return fail("published prefix path length is nonfinite");
  }
  if (continuation.path_progress_rad_per_sample.size() != continuation.samples.size() ||
    continuation.speed_scale_per_interval.size() + 1U != continuation.samples.size())
  {
    initializeProfilePathMetadata(continuation);
  }
  std::vector<JointMotionState> combined;
  combined.reserve(prefix.size() + continuation.samples.size() - 1U);
  combined.insert(combined.end(), prefix.begin(), prefix.end());
  combined.insert(combined.end(), std::next(continuation.samples.begin()),
    continuation.samples.end());
  const auto continuation_progress = continuation.path_progress_rad_per_sample;
  const auto continuation_scales = continuation.speed_scale_per_interval;
  continuation.samples = std::move(combined);
  const std::size_t prefix_offset = prefix.size() - 1U;
  std::vector<double> combined_progress;
  combined_progress.reserve(continuation.samples.size());
  double progress = 0.0;
  combined_progress.push_back(progress);
  for (std::size_t index = 1U; index < prefix.size(); ++index) {
    progress += (prefix[index].position - prefix[index - 1U].position).norm();
    combined_progress.push_back(progress);
  }
  const double continuation_progress_offset = progress;
  for (std::size_t index = 1U; index < continuation_progress.size(); ++index) {
    combined_progress.push_back(continuation_progress_offset +
      continuation_progress[index]);
  }
  continuation.path_progress_rad_per_sample = std::move(combined_progress);
  std::vector<double> combined_scales(prefix_offset, 1.0);
  combined_scales.insert(combined_scales.end(), continuation_scales.begin(),
    continuation_scales.end());
  continuation.speed_scale_per_interval = std::move(combined_scales);
  for (auto & leg_end_index : continuation.leg_end_sample_indices) {
    leg_end_index += prefix_offset;
  }
  continuation.duration_sec = combined_duration;
  continuation.path_length_rad += prefix_length;
  continuation.prefix_endpoint_sample_index = prefix.size() - 1U;
  return true;
}

EmergencyBrakeTail generateCompactProfileBrakeTail(
  const JointMotionState & branch_state,
  const Eigen::VectorXd & path_direction,
  const JointMotionLimits & limits,
  const EmergencyBrakeTailConfig & config)
{
  EmergencyBrakeTail result;
  const auto fail = [&result](const EmergencyBrakeTailStatus status,
    const std::string & reason) {
      result.status = status;
      result.failure_reason = reason;
      return result;
    };
  const Eigen::Index count = branch_state.position.size();
  if (count <= 0 || branch_state.velocity.size() != count ||
    branch_state.acceleration.size() != count || path_direction.size() != count ||
    limits.lower_position.size() != count || limits.upper_position.size() != count ||
    limits.max_velocity.size() != count || limits.max_acceleration.size() != count ||
    limits.max_jerk.size() != count || !branch_state.position.allFinite() ||
    !branch_state.velocity.allFinite() || !branch_state.acceleration.allFinite() ||
    !limits.lower_position.allFinite() || !limits.upper_position.allFinite() ||
    !limits.max_velocity.allFinite() || !limits.max_acceleration.allFinite() ||
    !limits.max_jerk.allFinite() || !std::isfinite(config.period_sec) ||
    config.period_sec <= 0.0 || config.maximum_braking_steps == 0U ||
    config.terminal_hold_points < 2U || !limits.numerical_policy ||
    !limits.numerical_policy->valid())
  {
    return fail(EmergencyBrakeTailStatus::kInvalidInput,
      "compact profile brake input, limits, or configuration are invalid");
  }
  if ((limits.upper_position.array() <= limits.lower_position.array()).any() ||
    (limits.max_velocity.array() <= 0.0).any() ||
    (limits.max_acceleration.array() <= 0.0).any() ||
    (limits.max_jerk.array() <= 0.0).any() ||
    (branch_state.position.array() < limits.lower_position.array()).any() ||
    (branch_state.position.array() > limits.upper_position.array()).any())
  {
    return fail(EmergencyBrakeTailStatus::kInvalidInput,
      "compact profile brake branch is outside joint position limits");
  }

  const bool stationary = branch_state.velocity.isZero(0.0) &&
    branch_state.acceleration.isZero(0.0);
  Eigen::VectorXd direction = path_direction;
  if (!direction.allFinite() || (!stationary && direction.norm() <= 0.0)) {
    return fail(EmergencyBrakeTailStatus::kInvalidInput,
      "compact profile brake direction is invalid");
  }
  if (!stationary) {
    direction.normalize();
  }
  double scalar_velocity = 0.0;
  double scalar_acceleration = 0.0;
  if (!stationary) {
    scalar_velocity = direction.dot(branch_state.velocity);
    scalar_acceleration = direction.dot(branch_state.acceleration);
  }

  const auto state_within_limits = [&limits, &config](
    const JointMotionState & current, const JointMotionState * previous) {
      if (!current.position.allFinite() || !current.velocity.allFinite() ||
        !current.acceleration.allFinite() ||
        (current.position.array() < limits.lower_position.array()).any() ||
        (current.position.array() > limits.upper_position.array()).any() ||
        (current.velocity.cwiseAbs().array() > limits.max_velocity.array() +
        limits.numerical_policy->velocity_error.value).any() ||
        (current.acceleration.cwiseAbs().array() > limits.max_acceleration.array() +
        limits.numerical_policy->acceleration_error.value).any())
      {
        return false;
      }
      if (previous != nullptr) {
        const Eigen::VectorXd jerk =
          (current.acceleration - previous->acceleration) / config.period_sec;
        if (!jerk.allFinite() || (jerk.cwiseAbs().array() > limits.max_jerk.array() +
          limits.numerical_policy->jerk_error.value).any())
        {
          return false;
        }
      }
      return true;
    };
  if (!state_within_limits(branch_state, nullptr)) {
    return fail(EmergencyBrakeTailStatus::kInfeasible,
      "compact profile brake branch exceeds joint limits");
  }

  result.points.push_back(branch_state);
  if (stationary) {
    result.first_stationary_point = 0U;
  } else {
    double maximum_velocity = std::numeric_limits<double>::infinity();
    double maximum_acceleration = std::numeric_limits<double>::infinity();
    double maximum_jerk = std::numeric_limits<double>::infinity();
    for (Eigen::Index axis = 0; axis < count; ++axis) {
      const double coefficient = std::abs(direction[axis]);
      if (coefficient == 0.0) {
        continue;
      }
      maximum_velocity = std::min(maximum_velocity, limits.max_velocity[axis] / coefficient);
      maximum_acceleration = std::min(
        maximum_acceleration, limits.max_acceleration[axis] / coefficient);
      maximum_jerk = std::min(maximum_jerk, limits.max_jerk[axis] / coefficient);
    }
    if (!std::isfinite(maximum_velocity) || !std::isfinite(maximum_acceleration) ||
      !std::isfinite(maximum_jerk) || maximum_velocity <= 0.0 ||
      maximum_acceleration <= 0.0 || maximum_jerk <= 0.0)
    {
      return fail(EmergencyBrakeTailStatus::kInvalidInput,
        "compact profile brake projected limits are invalid");
    }
    const Eigen::VectorXd velocity_residual =
      branch_state.velocity - direction * scalar_velocity;
    const Eigen::VectorXd acceleration_residual =
      branch_state.acceleration - direction * scalar_acceleration;
    const Eigen::VectorXd residual_jerk = acceleration_residual / config.period_sec +
      velocity_residual / (config.period_sec * config.period_sec);
    for (Eigen::Index axis = 0; axis < count; ++axis) {
      const double remaining_jerk = limits.max_jerk[axis] - std::abs(residual_jerk[axis]);
      const double coefficient = std::abs(direction[axis]);
      if (remaining_jerk <= 0.0 || (coefficient == 0.0 &&
        std::abs(residual_jerk[axis]) > limits.max_jerk[axis] +
        limits.numerical_policy->jerk_error.value))
      {
        return fail(EmergencyBrakeTailStatus::kInfeasible,
          "compact profile brake residual jerk has no reserve at axis " +
          std::to_string(axis) + ": actual=" + diagnosticNumber(residual_jerk[axis]) +
          ", limit=" + diagnosticNumber(limits.max_jerk[axis]));
      }
      if (coefficient > 0.0) {
        maximum_jerk = std::min(maximum_jerk, remaining_jerk / coefficient);
      }
    }
    if (!std::isfinite(maximum_jerk) || maximum_jerk <= 0.0) {
      return fail(EmergencyBrakeTailStatus::kInfeasible,
        "compact profile brake residual jerk leaves no positive scalar jerk reserve");
    }
    detail::ScalarBrakingOptions options;
    options.period_sec = config.period_sec;
    options.numerical_policy = *limits.numerical_policy;
    options.maximum_braking_steps = config.maximum_braking_steps;
    const detail::ScalarBrakingRollout rollout = detail::rolloutScalarBraking(
      detail::ScalarBrakingState{0.0, scalar_velocity, scalar_acceleration},
      detail::ScalarBrakingLimits{
        maximum_velocity, maximum_acceleration, maximum_jerk}, options, true);
    if (!rollout.commandAvailable()) {
      const auto status = rollout.status == detail::ScalarBrakingStatus::kMaximumStepsExceeded ?
        EmergencyBrakeTailStatus::kMaximumStepsExceeded : EmergencyBrakeTailStatus::kInfeasible;
      return fail(status, "compact profile scalar braking rollout is unavailable");
    }
    for (std::size_t index = 1U; index < rollout.points.size(); ++index) {
      const auto & scalar = rollout.points[index];
      const auto & previous = result.points.back();
      JointMotionState next;
      next.velocity = direction * scalar.velocity;
      // Match the queue convention exactly and validate jerk from the actual
      // previous mapped acceleration, including the original branch sample.
      next.acceleration = (next.velocity - previous.velocity) / config.period_sec;
      next.position = previous.position + next.velocity * config.period_sec;
      if (!state_within_limits(next, &previous)) {
        const Eigen::VectorXd actual_jerk =
          (next.acceleration - previous.acceleration) / config.period_sec;
        for (Eigen::Index axis = 0; axis < count; ++axis) {
          const double limit = limits.max_jerk[axis] +
            limits.numerical_policy->jerk_error.value;
          if (std::abs(actual_jerk[axis]) > limit) {
            return fail(EmergencyBrakeTailStatus::kInfeasible,
              "compact profile scalar brake exceeds mapped jerk at axis " +
              std::to_string(axis) + ": actual=" + diagnosticNumber(actual_jerk[axis]) +
              ", limit=" + diagnosticNumber(limits.max_jerk[axis]));
          }
        }
        return fail(EmergencyBrakeTailStatus::kInfeasible,
          "compact profile scalar brake exceeds mapped joint limits at sample " +
          std::to_string(index));
      }
      result.points.push_back(std::move(next));
    }
    if (result.points.back().velocity.isZero(0.0) &&
      result.points.back().acceleration.isZero(0.0))
    {
      result.first_stationary_point = result.points.size() - 1U;
    } else {
      return fail(EmergencyBrakeTailStatus::kInfeasible,
        "compact profile scalar brake did not end at an exact stationary state");
    }
  }
  for (std::size_t hold = 0U; hold < config.terminal_hold_points; ++hold) {
    result.points.push_back(result.points[result.first_stationary_point]);
  }
  result.status = EmergencyBrakeTailStatus::kSuccess;
  return result;
}

}  // namespace face_tracking_arm::control
