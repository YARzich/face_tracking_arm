// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "compact_path_timing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

namespace face_tracking_arm::control::detail::path_timing
{
namespace
{
constexpr double kTauRoundoffTolerance{1.0e-12};
std::string diagnosticNumber(const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  return stream.str();
}
}  // namespace

std::optional<ClockSchedule> buildClockSchedule(
  const trajectory_processing::Trajectory & totg,
  const Eigen::Index joint_count,
  std::vector<double> phase_tau,
  std::vector<double> phase_scales,
  const JointMotionLimits & limits,
  const double control_period_sec,
  const double speed_scale,
  const double requested_speed_scale,
  const double velocity_reserve,
  const double acceleration_reserve,
  const double jerk_reserve,
  const Eigen::VectorXd & totg_acceleration,
  const bool phase_local_clock,
  std::string * failure_reason)
{
  const auto fail = [failure_reason](const std::string & message) {
      if (failure_reason) {
        *failure_reason = message;
      }
      return std::optional<ClockSchedule>{};
    };
  if (failure_reason) {
    failure_reason->clear();
  }
  struct PhaseClockBounds
  {
    Eigen::VectorXd base_velocity;
    Eigen::VectorXd base_acceleration;
    double tau_start{0.0};
    double tau_end{0.0};
    std::size_t source_phase{0U};
    bool active{false};
    double relative_scale{1.0};
    double source_rate_cap{1.0};
    double rate_cap{0.0};
  };
  std::vector<PhaseClockBounds> phase_clock_bounds;
  std::vector<double> shared_edge_rates;
  if (phase_local_clock) {
    const auto source_rate_cap = [&](const std::size_t source_phase) {
        const double scale = phase_scales[source_phase];
        return std::min({speed_scale * scale, requested_speed_scale * scale, 1.0});
      };
    const auto acceleration_active = [&](const double time) {
        const Eigen::VectorXd acceleration = totg.getAcceleration(time);
        return acceleration.allFinite() &&
               acceleration.cwiseAbs().maxCoeff() > 1.0e-7;
      };
    const auto measure_phase = [&](const double lo, const double hi,
      const std::size_t source_phase, const bool active) ->
      std::optional<PhaseClockBounds> {
        PhaseClockBounds bounds;
        bounds.base_velocity = Eigen::VectorXd::Zero(joint_count);
        bounds.base_acceleration = Eigen::VectorXd::Zero(joint_count);
        bounds.tau_start = lo;
        bounds.tau_end = hi;
        bounds.source_phase = source_phase;
        bounds.active = active;
        bounds.relative_scale = phase_scales[source_phase];
        bounds.source_rate_cap = source_rate_cap(source_phase);
        for (double time = lo; ; ) {
          const Eigen::VectorXd velocity = totg.getVelocity(time);
          const Eigen::VectorXd acceleration = totg.getAcceleration(time);
          if (!velocity.allFinite() || !acceleration.allFinite()) {
            return std::nullopt;
          }
          bounds.base_velocity = bounds.base_velocity.cwiseMax(velocity.cwiseAbs());
          if (active) {
            bounds.base_acceleration =
              bounds.base_acceleration.cwiseMax(acceleration.cwiseAbs());
          }
          const double next = std::min(time + kTotgIntegrationStepSec, hi);
          if (next <= time + kTauRoundoffTolerance) {
            break;
          }
          time = next;
        }
        return bounds;
      };
    for (std::size_t source_phase = 0U; source_phase < phase_scales.size(); ++source_phase) {
      const double phase_start = phase_tau[source_phase];
      const double phase_end = phase_tau[source_phase + 1U];
      double previous_time = phase_start;
      bool previous_active = acceleration_active(phase_start);
      std::vector<double> cuts{phase_start};
      for (double time = std::min(phase_start + kTotgIntegrationStepSec, phase_end);
        time < phase_end; time = std::min(time + kTotgIntegrationStepSec, phase_end))
      {
        const bool active = acceleration_active(time);
        if (active != previous_active) {
          const double boundary = 0.5 * (previous_time + time);
          if (boundary > cuts.back() + kTauRoundoffTolerance &&
            boundary < phase_end - kTauRoundoffTolerance)
          {
            cuts.push_back(boundary);
          }
        }
        previous_time = time;
        previous_active = active;
      }
      cuts.push_back(phase_end);
      // Sub-dt active/coast islands cannot carry an independent command. Merge
      // only local transitions so source phase-scale boundaries remain fixed.
      while (cuts.size() > 2U) {
        std::size_t shortest = cuts.size();
        double shortest_duration = control_period_sec;
        for (std::size_t index = 1U; index < cuts.size(); ++index) {
          const double duration = cuts[index] - cuts[index - 1U];
          if (duration < shortest_duration) {
            shortest = index;
            shortest_duration = duration;
          }
        }
        if (shortest == cuts.size()) {
          break;
        }
        if (shortest == 1U) {
          cuts.erase(cuts.begin() + 1);
        } else if (shortest + 1U == cuts.size()) {
          cuts.erase(cuts.end() - 2);
        } else {
          cuts.erase(cuts.begin() + static_cast<std::ptrdiff_t>(shortest));
        }
      }
      for (std::size_t cut = 1U; cut < cuts.size(); ++cut) {
        const double lo = cuts[cut - 1U];
        const double hi = cuts[cut];
        if (hi <= lo + kTauRoundoffTolerance) {
          continue;
        }
        const auto measured = measure_phase(lo, hi, source_phase,
          acceleration_active(0.5 * (lo + hi)));
        if (!measured) {
          return fail("phase-local TOTG derivatives are nonfinite");
        }
        phase_clock_bounds.push_back(*measured);
      }
    }
    if (phase_clock_bounds.empty()) {
      return fail("phase-local TOTG clock partition is empty");
    }

    const auto edge_cap = [&](const Eigen::VectorXd & left_acceleration,
      const Eigen::VectorXd & right_acceleration, const double left_scale,
      const double right_scale) {
        double cap = 1.0;
        const double left_limit_scale = std::min(requested_speed_scale * left_scale, 1.0);
        const double right_limit_scale = std::min(requested_speed_scale * right_scale, 1.0);
        for (Eigen::Index axis = 0; axis < joint_count; ++axis) {
          const double acceleration_jump =
            left_acceleration[axis] + right_acceleration[axis];
          if (acceleration_jump <= 0.0) {
            continue;
          }
          const double effective_jerk = std::min(
            limits.max_jerk[axis] * left_limit_scale * left_limit_scale * left_limit_scale,
            limits.max_jerk[axis] * right_limit_scale * right_limit_scale * right_limit_scale) -
            jerk_reserve;
          if (effective_jerk <= 0.0) {
            return 0.0;
          }
          cap = std::min(cap,
            std::sqrt(effective_jerk * control_period_sec / acceleration_jump));
        }
        return std::min(cap, 1.0);
      };
    const auto active_phase_cap = [&](const std::size_t index) {
        const auto & phase = phase_clock_bounds[index];
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(joint_count);
        const Eigen::VectorXd left_acceleration = index > 0U &&
          phase_clock_bounds[index - 1U].active &&
          phase_clock_bounds[index - 1U].source_phase == phase.source_phase ?
          phase_clock_bounds[index - 1U].base_acceleration : zero;
        const Eigen::VectorXd right_acceleration = index + 1U < phase_clock_bounds.size() &&
          phase_clock_bounds[index + 1U].active &&
          phase_clock_bounds[index + 1U].source_phase == phase.source_phase ?
          phase_clock_bounds[index + 1U].base_acceleration : zero;
        return std::min({phase.source_rate_cap,
                   edge_cap(left_acceleration, phase.base_acceleration, phase.relative_scale,
            phase.relative_scale),
                   edge_cap(phase.base_acceleration, right_acceleration, phase.relative_scale,
            phase.relative_scale)});
      };
    for (bool changed = true; changed && phase_clock_bounds.size() > 1U; ) {
      changed = false;
      for (std::size_t index = 0U; index < phase_clock_bounds.size(); ++index) {
        const auto & coast = phase_clock_bounds[index];
        if (coast.active) {
          continue;
        }
        const bool has_left = index > 0U && phase_clock_bounds[index - 1U].active &&
          phase_clock_bounds[index - 1U].source_phase == coast.source_phase;
        const bool has_right = index + 1U < phase_clock_bounds.size() &&
          phase_clock_bounds[index + 1U].active &&
          phase_clock_bounds[index + 1U].source_phase == coast.source_phase;
        if (!has_left && !has_right) {
          continue;
        }
        const double left_rate = has_left ? active_phase_cap(index - 1U) : coast.source_rate_cap;
        const double right_rate = has_right ? active_phase_cap(index + 1U) : coast.source_rate_cap;
        const double delta_rate = std::abs(right_rate - left_rate);
        double minimum_time = 0.0;
        for (Eigen::Index axis = 0; axis < joint_count; ++axis) {
          const double base_velocity = coast.base_velocity[axis];
          if (base_velocity <= velocity_reserve) {
            continue;
          }
          const double acceleration_cap =
            (limits.max_acceleration[axis] *
            std::pow(std::min(requested_speed_scale * coast.relative_scale, 1.0), 2.0) -
            acceleration_reserve) / base_velocity;
          const double jerk_cap =
            (limits.max_jerk[axis] *
            std::pow(std::min(requested_speed_scale * coast.relative_scale, 1.0), 3.0) -
            jerk_reserve) / base_velocity;
          if (acceleration_cap <= 0.0 || jerk_cap <= 0.0) {
            minimum_time = std::numeric_limits<double>::infinity();
            break;
          }
          const double switch_delta = acceleration_cap * acceleration_cap / jerk_cap;
          const double axis_time = delta_rate <= switch_delta ?
            2.0 * std::sqrt(delta_rate / jerk_cap) :
            delta_rate / acceleration_cap + acceleration_cap / jerk_cap;
          minimum_time = std::max(minimum_time, axis_time);
        }
        const double minimum_span = 0.5 * (left_rate + right_rate) * minimum_time;
        if (coast.tau_end - coast.tau_start + kTauRoundoffTolerance >= minimum_span) {
          continue;
        }
        const std::size_t target = has_left ? index - 1U : index + 1U;
        auto & active = phase_clock_bounds[target];
        active.tau_start = std::min(active.tau_start, coast.tau_start);
        active.tau_end = std::max(active.tau_end, coast.tau_end);
        const auto merged = measure_phase(
          active.tau_start, active.tau_end, active.source_phase, true);
        if (!merged) {
          return fail("merged phase-local TOTG derivatives are nonfinite");
        }
        active.base_velocity = merged->base_velocity;
        active.base_acceleration = merged->base_acceleration;
        phase_clock_bounds.erase(phase_clock_bounds.begin() +
          static_cast<std::ptrdiff_t>(index));
        changed = true;
        break;
      }
    }

    phase_tau.clear();
    phase_scales.clear();
    phase_tau.push_back(phase_clock_bounds.front().tau_start);
    for (const auto & phase : phase_clock_bounds) {
      phase_tau.push_back(phase.tau_end);
      phase_scales.push_back(phase.relative_scale);
    }
    std::vector<double> edge_caps(phase_clock_bounds.size() + 1U, 1.0);
    for (std::size_t edge = 1U; edge < phase_clock_bounds.size(); ++edge) {
      edge_caps[edge] = edge_cap(phase_clock_bounds[edge - 1U].base_acceleration,
        phase_clock_bounds[edge].base_acceleration,
        phase_clock_bounds[edge - 1U].relative_scale,
        phase_clock_bounds[edge].relative_scale);
    }
    if (phase_clock_bounds.front().active) {
      edge_caps.front() = edge_cap(Eigen::VectorXd::Zero(joint_count),
        phase_clock_bounds.front().base_acceleration,
        phase_clock_bounds.front().relative_scale,
        phase_clock_bounds.front().relative_scale);
    }
    if (phase_clock_bounds.back().active) {
      edge_caps.back() = edge_cap(phase_clock_bounds.back().base_acceleration,
        Eigen::VectorXd::Zero(joint_count), phase_clock_bounds.back().relative_scale,
        phase_clock_bounds.back().relative_scale);
    }
    for (std::size_t phase = 0U; phase < phase_clock_bounds.size(); ++phase) {
      auto & bounds = phase_clock_bounds[phase];
      bounds.rate_cap = bounds.active ?
        std::min({bounds.source_rate_cap, edge_caps[phase], edge_caps[phase + 1U]}) :
        bounds.source_rate_cap;
      if (!std::isfinite(bounds.rate_cap) || bounds.rate_cap <= 0.0) {
        return fail("phase-local TOTG phase has no positive clock rate");
      }
    }
    shared_edge_rates.resize(phase_clock_bounds.size() + 1U);
    shared_edge_rates.front() = phase_clock_bounds.front().rate_cap;
    shared_edge_rates.back() = phase_clock_bounds.back().rate_cap;
    for (std::size_t edge = 1U; edge < phase_clock_bounds.size(); ++edge) {
      shared_edge_rates[edge] = std::min(
        phase_clock_bounds[edge - 1U].rate_cap, phase_clock_bounds[edge].rate_cap);
    }
  }

  std::vector<ClockSegment> clocks;
  double profile_duration = 0.0;
  double clock_rate_factor = 1.0;
  std::string feasibility_failure;
  bool feasible_clock = false;
  constexpr std::size_t kMaximumFeasibilityAttempts{16U};
  for (std::size_t attempt = 0U; attempt < kMaximumFeasibilityAttempts; ++attempt) {
    clocks.clear();
    profile_duration = 0.0;
    bool candidate_feasible = true;
    double previous_tau = phase_tau.front();
    for (std::size_t phase = 0U; phase < phase_scales.size(); ++phase) {
      const double relative_scale = phase_scales[phase];
      const double local_limit_scale = std::min(requested_speed_scale * relative_scale, 1.0);
      const double effective_scale = (phase_local_clock ?
        phase_clock_bounds[phase].rate_cap :
        std::min(speed_scale * relative_scale, 1.0)) * clock_rate_factor;
      const double tau_start = phase_tau[phase];
      const double tau_end = phase_tau[phase + 1U];
      const double start_rate = phase_local_clock ?
        shared_edge_rates[phase] * clock_rate_factor :
        (phase == 0U ? effective_scale : std::min(effective_scale,
        std::min(speed_scale * phase_scales[phase - 1U], 1.0) * clock_rate_factor));
      const double target_rate = phase_local_clock ?
        shared_edge_rates[phase + 1U] * clock_rate_factor :
        (phase + 1U == phase_scales.size() ? effective_scale : std::min(effective_scale,
        std::min(speed_scale * phase_scales[phase + 1U], 1.0) * clock_rate_factor));

      ruckig::InputParameter<1> input;
      input.current_position = {tau_start};
      input.current_velocity = {start_rate};
      input.current_acceleration = {0.0};
      input.target_position = {tau_end};
      input.target_velocity = {target_rate};
      input.target_acceleration = {0.0};
      input.max_velocity = {effective_scale};
      input.min_velocity = {-std::numeric_limits<double>::epsilon()};
      double tau_acceleration_limit = std::numeric_limits<double>::infinity();
      double tau_jerk_limit = std::numeric_limits<double>::infinity();
      if (phase_local_clock) {
        const auto & phase_bounds = phase_clock_bounds[phase];
        for (Eigen::Index axis = 0; axis < joint_count; ++axis) {
          const double base_velocity = phase_bounds.base_velocity[axis];
          const double effective_acceleration = limits.max_acceleration[axis] *
            local_limit_scale * local_limit_scale - acceleration_reserve;
          const double acceleration_residual = effective_acceleration -
            phase_bounds.base_acceleration[axis] * effective_scale * effective_scale;
          if (base_velocity > velocity_reserve) {
            tau_acceleration_limit = std::min(tau_acceleration_limit,
              acceleration_residual / base_velocity);
            const double effective_jerk = limits.max_jerk[axis] *
              local_limit_scale * local_limit_scale * local_limit_scale -
              jerk_reserve;
            tau_jerk_limit = std::min(tau_jerk_limit, effective_jerk / base_velocity);
          }
        }
      } else {
        for (Eigen::Index axis = 0; axis < joint_count; ++axis) {
          const double scaled_acceleration_limit =
            limits.max_acceleration[axis] * local_limit_scale * local_limit_scale -
            acceleration_reserve;
          const double scaled_jerk_limit = limits.max_jerk[axis] *
            local_limit_scale * local_limit_scale * local_limit_scale -
            jerk_reserve;
          const double reserve = std::max(0.0, scaled_acceleration_limit -
            totg_acceleration[axis] * effective_scale * effective_scale);
          tau_acceleration_limit = std::min(tau_acceleration_limit,
            reserve / limits.max_velocity[axis]);
          tau_jerk_limit = std::min(tau_jerk_limit,
            scaled_jerk_limit / limits.max_velocity[axis]);
        }
      }
      if (!std::isfinite(tau_acceleration_limit) ||
        !std::isfinite(tau_jerk_limit) || tau_acceleration_limit <= 0.0 ||
        tau_jerk_limit <= 0.0)
      {
        feasibility_failure = "phase=" + std::to_string(phase) +
          " has no finite positive acceleration reserve";
        candidate_feasible = false;
        break;
      }
      // These bounds apply to the internal clock derivatives themselves.
      // Scaling them again by the clock rate double-counts the time warp and
      // makes short, bounded phases infeasible in Ruckig.
      input.max_acceleration = {tau_acceleration_limit};
      input.max_jerk = {tau_jerk_limit};

      ruckig::Ruckig<1> generator(control_period_sec);
      ruckig::Trajectory<1> trajectory;
      const auto status = generator.calculate(input, trajectory);
      if (status != ruckig::Result::Working && status != ruckig::Result::Finished) {
        feasibility_failure = "phase=" + std::to_string(phase) + " result=" +
          std::to_string(static_cast<int>(status));
        candidate_feasible = false;
        break;
      }
      const double duration = trajectory.get_duration();
      if (!std::isfinite(duration) || duration <= 0.0) {
        feasibility_failure = "phase=" + std::to_string(phase) + " invalid duration";
        candidate_feasible = false;
        break;
      }
      const double phase_step_count_real = std::ceil(duration / control_period_sec);
      const double remaining_sample_budget = static_cast<double>(
        kMaximumProfileSamples - 3U) - profile_duration / control_period_sec;
      if (!std::isfinite(phase_step_count_real) || phase_step_count_real < 1.0 ||
        phase_step_count_real > remaining_sample_budget)
      {
        feasibility_failure = "phase=" + std::to_string(phase) +
          " duration exceeds the existing profile sample bound";
        candidate_feasible = false;
        break;
      }
      const std::size_t steps = static_cast<std::size_t>(
        std::ceil(duration / control_period_sec));
      double phase_previous_tau = tau_start;
      for (std::size_t step = 1U; step <= steps; ++step) {
        std::array<double, 1U> tau{}, velocity{}, acceleration{};
        trajectory.at_time(std::min(static_cast<double>(step) * control_period_sec, duration),
          tau, velocity, acceleration);
        if (!std::isfinite(tau[0]) || tau[0] < tau_start - kTauRoundoffTolerance ||
          tau[0] > tau_end + kTauRoundoffTolerance ||
          tau[0] + kTauRoundoffTolerance < phase_previous_tau ||
          tau[0] + kTauRoundoffTolerance < previous_tau)
        {
          feasibility_failure = "phase=" + std::to_string(phase) + " tau=" +
            diagnosticNumber(tau[0]) + " interval=[" + diagnosticNumber(tau_start) + "," +
            diagnosticNumber(tau_end) + "]";
          candidate_feasible = false;
          break;
        }
        phase_previous_tau = tau[0];
        previous_tau = tau[0];
      }
      if (!candidate_feasible) {
        break;
      }
      profile_duration += duration;
      clocks.push_back(ClockSegment{std::move(trajectory), duration, tau_start, relative_scale});
    }
    if (candidate_feasible && clocks.size() == phase_scales.size()) {
      feasible_clock = true;
      break;
    }
    clock_rate_factor *= 0.5;
  }
  if (!feasible_clock) {
    return fail("Ruckig clock feasibility search failed: " + feasibility_failure);
  }

  return ClockSchedule{std::move(clocks), std::move(phase_tau), profile_duration};
}

}  // namespace face_tracking_arm::control::detail::path_timing
