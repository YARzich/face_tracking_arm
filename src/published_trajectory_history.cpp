// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include "face_tracking_arm/published_trajectory_history.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace face_tracking_arm::control
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000;
constexpr double kStationaryTolerance = 1.0e-12;

bool checkedAdd(
  const std::int64_t left, const std::int64_t right, std::int64_t & result) noexcept
{
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
    (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
  {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedSubtract(
  const std::int64_t left, const std::int64_t right, std::int64_t & result) noexcept
{
  if ((right > 0 && left < std::numeric_limits<std::int64_t>::min() + right) ||
    (right < 0 && left > std::numeric_limits<std::int64_t>::max() + right))
  {
    return false;
  }
  result = left - right;
  return true;
}

std::optional<std::int64_t> durationNanoseconds(
  const builtin_interfaces::msg::Duration & duration) noexcept
{
  if (duration.nanosec >= static_cast<std::uint32_t>(kNanosecondsPerSecond)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(duration.sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(duration.nanosec);
}

std::optional<std::int64_t> stampNanoseconds(
  const builtin_interfaces::msg::Time & stamp) noexcept
{
  if (stamp.sec < 0 || stamp.nanosec >= static_cast<std::uint32_t>(kNanosecondsPerSecond)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(stamp.nanosec);
}

bool finiteState(const moveit_servo::KinematicState & state, const std::size_t joint_count)
{
  return state.joint_names.size() == joint_count &&
         state.positions.size() == static_cast<Eigen::Index>(joint_count) &&
         state.velocities.size() == static_cast<Eigen::Index>(joint_count) &&
         state.accelerations.size() == static_cast<Eigen::Index>(joint_count) &&
         state.positions.allFinite() && state.velocities.allFinite() &&
         state.accelerations.allFinite();
}

bool validRecord(
  const std::deque<moveit_servo::KinematicState> & queue,
  const rclcpp::Time & stationary_time)
{
  if (queue.size() < 3U || queue.front().joint_names.empty()) {
    return false;
  }

  const std::size_t joint_count = queue.front().joint_names.size();
  const rcl_clock_type_t clock_type = queue.front().time_stamp.get_clock_type();
  if (stationary_time.get_clock_type() != clock_type) {
    return false;
  }

  for (auto state = queue.begin(); state != queue.end(); ++state) {
    if (!finiteState(*state, joint_count) ||
      state->joint_names != queue.front().joint_names ||
      state->time_stamp.get_clock_type() != clock_type)
    {
      return false;
    }
    if (state != queue.begin() && state->time_stamp <= std::prev(state)->time_stamp) {
      return false;
    }
  }

  const auto stationary = std::find_if(
    queue.begin(), queue.end(), [&stationary_time](const auto & state) {
      return state.time_stamp == stationary_time;
    });
  const auto transmitted_end = std::prev(queue.end(), 2);
  if (stationary == queue.end() || stationary > transmitted_end) {
    return false;
  }

  return std::all_of(stationary, queue.end(), [&stationary](const auto & state) {
             return state.positions.isApprox(stationary->positions, kStationaryTolerance) &&
                    state.velocities.cwiseAbs().maxCoeff() <= kStationaryTolerance &&
                    state.accelerations.cwiseAbs().maxCoeff() <= kStationaryTolerance;
  });
}

double maximumAbsoluteDifference(
  const std::vector<double> & actual, const Eigen::VectorXd & expected)
{
  if (actual.size() != static_cast<std::size_t>(expected.size())) {
    return std::numeric_limits<double>::infinity();
  }
  double maximum = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index])) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(
      maximum,
      std::abs(actual[index] - expected[static_cast<Eigen::Index>(index)]));
  }
  return maximum;
}

}  // namespace

PublishedTrajectoryHistory::PublishedTrajectoryHistory(const std::size_t capacity)
: capacity_(capacity)
{
  if (capacity == 0U) {
    throw std::invalid_argument("published trajectory history capacity must be positive");
  }
}

std::optional<PublicationId> PublishedTrajectoryHistory::append(
  std::deque<moveit_servo::KinematicState> execution_queue,
  const rclcpp::Time & stationary_time)
{
  if (!validRecord(execution_queue, stationary_time) ||
    next_id_ == std::numeric_limits<PublicationId>::max())
  {
    return std::nullopt;
  }

  const PublicationId id = next_id_++;
  if (records_.size() == capacity_) {
    records_.pop_front();
  }
  records_.push_back({id, std::move(execution_queue), stationary_time});
  return id;
}

void PublishedTrajectoryHistory::clear() noexcept
{
  records_.clear();
}

std::size_t PublishedTrajectoryHistory::size() const noexcept
{
  return records_.size();
}

std::size_t PublishedTrajectoryHistory::capacity() const noexcept
{
  return capacity_;
}

bool PublishedTrajectoryHistory::empty() const noexcept
{
  return records_.empty();
}

const PublishedTrajectoryRecord * PublishedTrajectoryHistory::latest() const noexcept
{
  return records_.empty() ? nullptr : &records_.back();
}

const PublishedTrajectoryRecord * PublishedTrajectoryHistory::find(
  const PublicationId id) const noexcept
{
  const auto record = std::find_if(
    records_.begin(), records_.end(), [id](const auto & candidate) {
      return candidate.id == id;
    });
  return record == records_.end() ? nullptr : &*record;
}

std::optional<ControllerSampleTimes> PublishedTrajectoryHistory::correctedSampleTimes(
  const control_msgs::msg::JointTrajectoryControllerState & state,
  const rcl_clock_type_t clock_type, const rclcpp::Duration & controller_period) noexcept
{
  if (controller_period.nanoseconds() <= 0) {
    return std::nullopt;
  }
  const auto header_ns = stampNanoseconds(state.header.stamp);
  const auto reference_offset_ns = durationNanoseconds(state.reference.time_from_start);
  const auto feedback_offset_ns = durationNanoseconds(state.feedback.time_from_start);
  if (!header_ns.has_value() || !reference_offset_ns.has_value() ||
    !feedback_offset_ns.has_value())
  {
    return std::nullopt;
  }

  std::int64_t trajectory_start_ns = 0;
  std::int64_t reference_ns = 0;
  std::int64_t output_ns = 0;
  if (!checkedSubtract(*header_ns, *feedback_offset_ns, trajectory_start_ns) ||
    !checkedAdd(trajectory_start_ns, *reference_offset_ns, reference_ns) ||
    !checkedAdd(reference_ns, controller_period.nanoseconds(), output_ns) ||
    reference_ns < 0 || output_ns < 0)
  {
    return std::nullopt;
  }

  return ControllerSampleTimes{
    rclcpp::Time(trajectory_start_ns, clock_type),
    rclcpp::Time(reference_ns, clock_type), rclcpp::Time(output_ns, clock_type)};
}

PublicationMatch PublishedTrajectoryHistory::match(
  const control_msgs::msg::JointTrajectoryControllerState & state,
  const rcl_clock_type_t clock_type, const rclcpp::Duration & controller_period,
  const double position_tolerance_rad) const
{
  PublicationMatch result;
  const auto sample_times = correctedSampleTimes(state, clock_type, controller_period);
  if (!sample_times.has_value() || !std::isfinite(position_tolerance_rad) ||
    position_tolerance_rad < 0.0)
  {
    return result;
  }
  result.sample_times = *sample_times;

  for (auto record = records_.rbegin(); record != records_.rend(); ++record) {
    const bool latest_record = record == records_.rbegin();
    const bool trajectory_origin_matches =
      record->execution_queue.front().time_stamp == sample_times->trajectory_origin;
    const auto & queue = record->execution_queue;
    const rclcpp::Time & front_time = queue.front().time_stamp;
    const bool reference_prestart = latest_record && trajectory_origin_matches &&
      sample_times->reference < front_time;
    const bool output_prestart = latest_record && trajectory_origin_matches &&
      sample_times->output < front_time;
    const bool hybrid_prestart = reference_prestart || output_prestart;

    auto expected_reference = expectedStateAt(
      *record, sample_times->reference, trajectory_origin_matches);
    auto expected_output = expectedStateAt(
      *record, sample_times->output, trajectory_origin_matches && !hybrid_prestart);
    if (reference_prestart) {
      expected_reference = expectedPrestartTerminalHold(
        *record, rclcpp::Time(state.header.stamp, clock_type), sample_times->reference);
    }
    if (output_prestart) {
      expected_output = expectedPrestartTerminalHold(
        *record, rclcpp::Time(state.header.stamp, clock_type), sample_times->output);
    }
    if (!expected_reference.has_value() || !expected_output.has_value()) {
      continue;
    }

    const auto exact_zero_finite = [](const std::vector<double> & values) {
        return std::all_of(values.begin(), values.end(), [](const double value) {
                   return std::isfinite(value) && value == 0.0;
        });
      };
    if ((reference_prestart &&
      (state.reference.velocities.size() != expected_reference->velocities.size() ||
      !exact_zero_finite(state.reference.velocities))) ||
      (output_prestart && !state.output.velocities.empty() &&
      (state.output.velocities.size() != expected_output->velocities.size() ||
      !exact_zero_finite(state.output.velocities))))
    {
      continue;
    }

    const double reference_error = maximumAbsoluteDifference(
      state.reference.positions, expected_reference->positions);
    const double output_error = maximumAbsoluteDifference(
      state.output.positions, expected_output->positions);
    if (reference_error > position_tolerance_rad || output_error > position_tolerance_rad) {
      continue;
    }

    result.kind = latest_record ?
      PublicationMatchKind::kLatest : PublicationMatchKind::kOlder;
    result.publication_id = record->id;
    result.expected_reference = *expected_reference;
    result.expected_output = *expected_output;
    result.reference_error_rad = reference_error;
    result.output_error_rad = output_error;
    return result;
  }
  return result;
}

bool PublishedTrajectoryHistory::isPendingTerminalReplacement(
  const PublicationMatch & match, const rclcpp::Time & measurement_time) const
{
  if (match.kind != PublicationMatchKind::kOlder) {
    return false;
  }

  const auto * const older = find(match.publication_id);
  const auto * const replacement = latest();
  if (older == nullptr || replacement == nullptr || older->id == replacement->id ||
    older->execution_queue.size() < 3U || replacement->execution_queue.size() < 3U)
  {
    return false;
  }

  const auto & older_queue = older->execution_queue;
  const auto & replacement_start = replacement->execution_queue.front();
  const auto & older_terminal = *std::prev(older_queue.end(), 2);
  const rcl_clock_type_t clock_type = older_queue.front().time_stamp.get_clock_type();
  if (measurement_time.get_clock_type() != clock_type ||
    match.sample_times.trajectory_origin.get_clock_type() != clock_type ||
    match.sample_times.reference.get_clock_type() != clock_type ||
    match.sample_times.output.get_clock_type() != clock_type ||
    replacement_start.time_stamp.get_clock_type() != clock_type ||
    older->stationary_time.get_clock_type() != clock_type ||
    match.sample_times.trajectory_origin != older_queue.front().time_stamp ||
    measurement_time < older->stationary_time ||
    match.sample_times.reference < older->stationary_time ||
    match.sample_times.output < older->stationary_time ||
    !(match.sample_times.reference < replacement_start.time_stamp) ||
    !(match.sample_times.output < replacement_start.time_stamp) ||
    !(measurement_time < replacement_start.time_stamp))
  {
    return false;
  }

  const auto same_positions = [](const Eigen::VectorXd & lhs, const Eigen::VectorXd & rhs) {
      return lhs.size() == rhs.size() && lhs.allFinite() && rhs.allFinite() &&
             (lhs.array() == rhs.array()).all();
    };
  const auto & expected_reference = match.expected_reference;
  const auto & expected_output = match.expected_output;
  return expected_reference.terminal_hold && expected_output.terminal_hold &&
         same_positions(expected_reference.positions, older_terminal.positions) &&
         same_positions(expected_output.positions, older_terminal.positions) &&
         expected_reference.velocities.size() == older_terminal.positions.size() &&
         expected_output.velocities.size() == older_terminal.positions.size() &&
         expected_reference.velocities.allFinite() &&
         expected_output.velocities.allFinite() &&
         expected_reference.velocities.isZero(0.0) &&
         expected_output.velocities.isZero(0.0) &&
         same_positions(replacement_start.positions, older_terminal.positions) &&
         replacement_start.velocities.allFinite() &&
         replacement_start.accelerations.allFinite() &&
         replacement_start.velocities.isZero(0.0) &&
         replacement_start.accelerations.isZero(0.0);
}

std::optional<ExpectedPositionState> PublishedTrajectoryHistory::expectedAt(
  const PublicationId id, const rclcpp::Time & sample_time) const
{
  const auto * record = find(id);
  if (record == nullptr) {
    return std::nullopt;
  }
  if (record == latest() &&
    sample_time.get_clock_type() == record->execution_queue.front().time_stamp.get_clock_type() &&
    sample_time < record->execution_queue.front().time_stamp)
  {
    return expectedPrestartTerminalHold(*record, sample_time, sample_time);
  }
  return expectedStateAt(*record, sample_time, true);
}

std::optional<ExpectedPositionState> PublishedTrajectoryHistory::expectedPrestartTerminalHold(
  const PublishedTrajectoryRecord & latest_record, const rclcpp::Time & measurement_time,
  const rclcpp::Time & sample_time) const
{
  if (records_.size() < 2U || records_.back().id != latest_record.id ||
    latest_record.execution_queue.size() < 3U)
  {
    return std::nullopt;
  }

  const auto & latest_queue = latest_record.execution_queue;
  const auto & latest_front = latest_queue.front();
  const rcl_clock_type_t clock_type = latest_front.time_stamp.get_clock_type();
  if (measurement_time.get_clock_type() != clock_type ||
    sample_time.get_clock_type() != clock_type ||
    !(measurement_time < latest_front.time_stamp) || !(sample_time < latest_front.time_stamp) ||
    latest_front.joint_names.empty() ||
    latest_front.positions.size() != static_cast<Eigen::Index>(latest_front.joint_names.size()) ||
    latest_front.velocities.size() != latest_front.positions.size() ||
    latest_front.accelerations.size() != latest_front.positions.size() ||
    !latest_front.positions.allFinite() || !latest_front.velocities.allFinite() ||
    !latest_front.accelerations.allFinite() || !latest_front.velocities.isZero(0.0) ||
    !latest_front.accelerations.isZero(0.0))
  {
    return std::nullopt;
  }

  const auto same_front = [&latest_front](const PublishedTrajectoryRecord & record) {
      if (record.execution_queue.size() < 3U) {
        return false;
      }
      const auto & front = record.execution_queue.front();
      return front.time_stamp.get_clock_type() == latest_front.time_stamp.get_clock_type() &&
             front.time_stamp == latest_front.time_stamp &&
             front.joint_names == latest_front.joint_names &&
             front.positions.size() == latest_front.positions.size() &&
             front.positions.allFinite() &&
             (front.positions.array() == latest_front.positions.array()).all() &&
             front.velocities.size() == latest_front.velocities.size() &&
             front.accelerations.size() == latest_front.accelerations.size() &&
             front.velocities.allFinite() && front.accelerations.allFinite() &&
             front.velocities.isZero(0.0) && front.accelerations.isZero(0.0);
    };

  auto root = std::prev(records_.end());
  while (root != records_.begin()) {
    auto previous = std::prev(root);
    if (!same_front(*previous)) {
      break;
    }
    root = previous;
  }
  if (root == records_.begin()) {
    return std::nullopt;
  }

  // The first nonmatching predecessor is the only retained stationary root for this chain.
  const auto & root_record = *std::prev(root);
  const auto & root_queue = root_record.execution_queue;
  if (root_queue.size() < 3U || root_queue.front().joint_names != latest_front.joint_names ||
    root_record.stationary_time.get_clock_type() != clock_type ||
    root_record.stationary_time > measurement_time ||
    root_record.stationary_time > sample_time ||
    !(root_record.stationary_time < latest_front.time_stamp))
  {
    return std::nullopt;
  }

  const auto transmitted_end = std::prev(root_queue.end(), 2);
  const auto stationary = std::find_if(
    root_queue.begin(), root_queue.end(), [&root_record](const auto & state) {
      return state.time_stamp == root_record.stationary_time;
    });
  if (stationary == root_queue.end() || stationary > transmitted_end ||
    transmitted_end->time_stamp < root_record.stationary_time ||
    transmitted_end->joint_names != latest_front.joint_names ||
    transmitted_end->positions.size() != latest_front.positions.size() ||
    !transmitted_end->positions.allFinite() ||
    !(transmitted_end->positions.array() == latest_front.positions.array()).all())
  {
    return std::nullopt;
  }

  const Eigen::Index joint_count = latest_front.positions.size();
  const auto exact_stationary = [&latest_front, joint_count](const auto & state) {
      return state.joint_names == latest_front.joint_names &&
             state.positions.size() == joint_count && state.velocities.size() == joint_count &&
             state.accelerations.size() == joint_count && state.positions.allFinite() &&
             state.velocities.allFinite() && state.accelerations.allFinite() &&
             (state.positions.array() == latest_front.positions.array()).all() &&
             state.velocities.isZero(0.0) && state.accelerations.isZero(0.0);
    };
  if (!std::all_of(stationary, root_queue.end(), exact_stationary)) {
    return std::nullopt;
  }

  ExpectedPositionState expected;
  expected.positions = latest_front.positions;
  expected.velocities = Eigen::VectorXd::Zero(joint_count);
  expected.terminal_hold = true;
  return expected;
}

std::optional<ExpectedPositionState> PublishedTrajectoryHistory::expectedStateAt(
  const PublishedTrajectoryRecord & record, const rclcpp::Time & sample_time,
  const bool allow_terminal_extrapolation)
{
  const auto & queue = record.execution_queue;
  if (queue.size() < 3U ||
    sample_time.get_clock_type() != queue.front().time_stamp.get_clock_type())
  {
    return std::nullopt;
  }

  const auto transmitted_end = std::prev(queue.end(), 2);
  if (sample_time < queue.front().time_stamp) {
    return std::nullopt;
  }
  if (sample_time >= record.stationary_time) {
    if (sample_time <= transmitted_end->time_stamp || allow_terminal_extrapolation) {
      ExpectedPositionState expected;
      expected.positions = transmitted_end->positions;
      expected.velocities = Eigen::VectorXd::Zero(transmitted_end->positions.size());
      expected.terminal_hold = true;
      return expected;
    }
    return std::nullopt;
  }
  if (sample_time > transmitted_end->time_stamp) {
    return std::nullopt;
  }

  auto end = std::find_if(
    queue.begin(), std::next(transmitted_end), [&sample_time](const auto & state) {
      return state.time_stamp >= sample_time;
    });
  if (end == queue.begin()) {
    ++end;
  } else if (end == std::next(transmitted_end)) {
    end = transmitted_end;
  }
  const auto start = std::prev(end);
  const double duration_sec = (end->time_stamp - start->time_stamp).seconds();
  if (!std::isfinite(duration_sec) || duration_sec <= 0.0 ||
    start->positions.size() != end->positions.size())
  {
    return std::nullopt;
  }
  const double elapsed_sec = (sample_time - start->time_stamp).seconds();
  const double ratio = std::clamp(elapsed_sec / duration_sec, 0.0, 1.0);

  ExpectedPositionState expected;
  expected.positions = start->positions + ratio * (end->positions - start->positions);
  expected.velocities = (end->positions - start->positions) / duration_sec;
  if (!expected.positions.allFinite() || !expected.velocities.allFinite()) {
    return std::nullopt;
  }
  return expected;
}

}  // namespace face_tracking_arm::control
