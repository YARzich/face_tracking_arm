// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>

#include "face_tracking_arm/hardware_io_status_policy.hpp"
#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::hardware
{
namespace
{

using SteadyTime = HardwareIoStatusPolicy::SteadyTime;
constexpr std::int64_t kNowRosNs = 10'000'000'000LL;
constexpr std::int64_t kMaxSampleAgeNs = 20'000'000LL;
constexpr std::int64_t kMaxTransportAgeNs = 5'000'000LL;

builtin_interfaces::msg::Time to_stamp(const std::int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(nanoseconds / 1'000'000'000LL);
  stamp.nanosec = static_cast<std::uint32_t>(nanoseconds % 1'000'000'000LL);
  return stamp;
}

msg::HardwareIoStatus fresh_status()
{
  msg::HardwareIoStatus status;
  status.header.stamp = to_stamp(kNowRosNs - 2'000'000LL);
  status.source_kind = msg::HardwareIoStatus::SOURCE_HARDWARE;
  status.runtime_actions_enabled = true;
  status.hardware_active = true;
  status.epoch = 7;
  status.read_epoch = status.epoch;
  status.read_attempt_sequence = 11;
  status.read_sequence = 11;
  status.read_ok = true;
  status.read_stamp = to_stamp(kNowRosNs - 5'000'000LL);
  status.read_age_us = 3'000;
  status.robot_msg_sequence = 4;
  status.robot_msg_stamp = to_stamp(kNowRosNs - 6'000'000LL);
  status.robot_msg_age_us = 4'000;
  status.robot_mode = 1;
  status.robot_state = 0;
  status.robot_mt_able = 0x3f;
  status.robot_mt_brake = 0x3f;
  for (std::size_t index = 0; index < status.q.size(); ++index) {
    status.q[index] = static_cast<double>(index) / 10.0;
    status.dq[index] = 0.01 * static_cast<double>(index);
  }
  return status;
}

HardwareIoStatusPolicy make_policy()
{
  return HardwareIoStatusPolicy(kMaxSampleAgeNs, kMaxTransportAgeNs);
}

TEST(HardwareIoStatusPolicy, AcceptsFreshCoherentHardwareSnapshot)
{
  auto policy = make_policy();
  const SteadyTime received{};
  const auto assessment = policy.observe(fresh_status(), kNowRosNs, received);

  ASSERT_TRUE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kNone);
  EXPECT_EQ(assessment.sample.epoch, 7U);
  EXPECT_EQ(assessment.sample.read_sequence, 11U);
  EXPECT_DOUBLE_EQ(assessment.sample.position[4], 0.4);
  EXPECT_DOUBLE_EQ(assessment.sample.velocity[5], 0.05);
  EXPECT_EQ(assessment.sample.read_stamp_ns, kNowRosNs - 5'000'000LL);
  EXPECT_EQ(assessment.sample.read_age_us, 3'000U);
}

TEST(HardwareIoStatusPolicy, ResetAllowsWriteStampInNewClockEpoch)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.write_sequence = 3;
  status.driver_command_sequence = 3;
  status.write_stamp = to_stamp(kNowRosNs - 4'000'000LL);
  status.write_attempted = true;
  status.write_ok = true;
  ASSERT_TRUE(policy.observe(status, kNowRosNs, SteadyTime{}).valid);

  policy.reset();
  constexpr std::int64_t kNewEpochNowRosNs = 4'000'000'000LL;
  status = fresh_status();
  status.epoch = 8;
  status.read_epoch = status.epoch;
  status.header.stamp = to_stamp(kNewEpochNowRosNs - 2'000'000LL);
  status.read_stamp = to_stamp(kNewEpochNowRosNs - 5'000'000LL);
  status.robot_msg_stamp = to_stamp(kNewEpochNowRosNs - 6'000'000LL);
  status.write_sequence = 1;
  status.driver_command_sequence = 1;
  status.write_stamp = to_stamp(kNewEpochNowRosNs - 4'000'000LL);
  status.write_attempted = true;
  status.write_ok = true;

  const auto assessment = policy.observe(
    status, kNewEpochNowRosNs, SteadyTime{});
  EXPECT_TRUE(assessment.valid) << hardwareIoStatusFailureName(assessment.failure);
}

TEST(HardwareIoStatusPolicy, AddsTransportAgeToSourceSampleAge)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.header.stamp = to_stamp(kNowRosNs - 6'000'000LL);
  status.read_stamp = to_stamp(kNowRosNs - 9'000'000LL);
  status.robot_msg_stamp = to_stamp(kNowRosNs - 10'000'000LL);

  const auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kTransportTooOld);
}

TEST(HardwareIoStatusPolicy, RejectsFutureAndUncorrelatedTimestamps)
{
  auto status = fresh_status();
  status.header.stamp = to_stamp(kNowRosNs + 1);
  auto policy = make_policy();
  auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kFutureOrRewoundTime);

  status = fresh_status();
  status.read_stamp = to_stamp(kNowRosNs - 15'000'000LL);
  policy.reset();
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kInvalidTimestamp);
}

TEST(HardwareIoStatusPolicy, RejectsFailedReadAndMissingRobotMessage)
{
  auto status = fresh_status();
  status.read_ok = false;
  auto policy = make_policy();
  auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kReadFailed);

  status = fresh_status();
  status.robot_msg_sequence = 0;
  policy.reset();
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kRobotMessageUnavailable);
}

TEST(HardwareIoStatusPolicy, RejectsPreActivationSnapshotWithoutLockingEpoch)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.hardware_active = false;
  auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kHardwareInactive);

  status = fresh_status();
  status.epoch++;
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kInvalidEpochOrSequence);

  status.read_epoch = status.epoch;
  status.read_attempt_sequence++;
  status.read_sequence++;
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_TRUE(assessment.valid);
  EXPECT_EQ(assessment.sample.epoch, 8U);
  EXPECT_EQ(assessment.sample.read_epoch, 8U);
}

TEST(HardwareIoStatusPolicy, AcceptsFirstSuccessfulReadAfterPreActivationFailure)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.hardware_active = false;
  status.read_attempt_sequence = 11;
  status.read_sequence = 10;
  status.read_ok = false;
  auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kHardwareInactive);

  // The successful read carries the ID of its attempt; a pre-activation miss
  // does not permanently skew the active provenance sequence.
  status = fresh_status();
  status.epoch++;
  status.read_attempt_sequence = 12;
  status.read_sequence = 12;
  status.read_epoch = status.epoch;
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_TRUE(assessment.valid);
  EXPECT_EQ(assessment.sample.read_sequence, 12U);
}

TEST(HardwareIoStatusPolicy, ActiveReadFailureRemainsLatchedAcrossLaterSuccess)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.read_attempt_sequence = 12;
  status.read_sequence = 11;
  status.read_ok = false;
  status.fault_latched = true;
  status.fault_kind = msg::HardwareIoStatus::FAULT_READ;
  status.fault_code = -37;
  status.fault_sequence = 1;
  status.fault_read_attempt_sequence = 12;
  status.fault_epoch = status.epoch;
  const auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});

  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kDriverFaultLatched);
  EXPECT_EQ(assessment.sample.fault_code, -37);
  EXPECT_EQ(assessment.sample.fault_read_attempt_sequence, 12U);
}

TEST(HardwareIoStatusPolicy, RejectsRegressedSequenceAndChangedEpoch)
{
  auto policy = make_policy();
  auto status = fresh_status();
  ASSERT_TRUE(policy.observe(status, kNowRosNs, SteadyTime{}).valid);

  status.header.stamp = to_stamp(kNowRosNs + 1'000'000LL);
  status.read_attempt_sequence = 12;
  status.read_sequence = 10;
  status.read_stamp = to_stamp(kNowRosNs - 5'000'000LL);
  status.robot_msg_stamp = to_stamp(kNowRosNs - 6'000'000LL);
  auto assessment = policy.observe(status, kNowRosNs + 1'000'000LL,
    SteadyTime{} + std::chrono::milliseconds(1));
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kSequenceRegressed);

  policy.reset();
  status = fresh_status();
  ASSERT_TRUE(policy.observe(status, kNowRosNs, SteadyTime{}).valid);
  status.epoch++;
  status.read_epoch = status.epoch;
  status.header.stamp = to_stamp(kNowRosNs + 1'000'000LL);
  status.read_stamp = to_stamp(kNowRosNs - 4'000'000LL);
  status.robot_msg_stamp = to_stamp(kNowRosNs - 5'000'000LL);
  assessment = policy.observe(status, kNowRosNs + 1'000'000LL,
    SteadyTime{} + std::chrono::milliseconds(1));
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kSequenceRegressed);
}

TEST(HardwareIoStatusPolicy, RejectsRegressedSourceStampsAndContradictorySuccessCodes)
{
  auto policy = make_policy();
  auto status = fresh_status();
  ASSERT_TRUE(policy.observe(status, kNowRosNs, SteadyTime{}).valid);
  status.header.stamp = to_stamp(kNowRosNs - 1'000'000LL);
  status.read_attempt_sequence++;
  status.read_sequence++;
  status.read_stamp = to_stamp(kNowRosNs - 6'000'000LL);
  status.read_age_us = 5'000;
  status.robot_msg_sequence++;
  status.robot_msg_stamp = to_stamp(kNowRosNs - 5'000'000LL);
  status.robot_msg_age_us = 4'000;
  auto assessment = policy.observe(
    status, kNowRosNs + 1'000'000LL, SteadyTime{} + std::chrono::milliseconds(1));
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kSequenceRegressed);

  policy.reset();
  status = fresh_status();
  status.read_code = 3;
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kReadFailed);

  policy.reset();
  status = fresh_status();
  status.write_attempted = true;
  status.write_ok = true;
  status.write_sequence = 1;
  status.write_code = 7;
  status.write_stamp = to_stamp(kNowRosNs - 2'000'000LL);
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kWriteFailed);
}

TEST(HardwareIoStatusPolicy, ExpiresUsingSteadyElapsedAfterReceipt)
{
  auto policy = make_policy();
  ASSERT_TRUE(policy.observe(fresh_status(), kNowRosNs, SteadyTime{}).valid);

  auto assessment = policy.current(
    kNowRosNs + 10'000'000LL, SteadyTime{} + std::chrono::milliseconds(10));
  EXPECT_TRUE(assessment.valid);

  assessment = policy.current(
    kNowRosNs + 16'000'000LL, SteadyTime{} + std::chrono::milliseconds(16));
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kSampleTooOld);
}

TEST(HardwareIoStatusPolicy, ExpiresOlderRobotReportAfterNewerJointRead)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.read_age_us = 1'000;
  status.read_stamp = to_stamp(kNowRosNs - 3'000'000LL);
  status.robot_msg_age_us = 18'000;
  status.robot_msg_stamp = to_stamp(kNowRosNs - 20'000'000LL);
  ASSERT_TRUE(policy.observe(status, kNowRosNs, SteadyTime{}).valid);

  const auto assessment = policy.current(
    kNowRosNs + 3'000'000LL, SteadyTime{} + std::chrono::milliseconds(3));
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kRobotMessageUnavailable);
}

TEST(HardwareIoStatusPolicy, RobotMessageDeadlineIsIndependentOfJointSampleDeadline)
{
  HardwareIoStatusPolicy policy(kMaxSampleAgeNs, kMaxTransportAgeNs, 500'000'000LL);
  auto status = fresh_status();
  status.robot_msg_age_us = 100'000;
  status.robot_msg_stamp = to_stamp(kNowRosNs - 102'000'000LL);
  const auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_TRUE(assessment.valid);
  EXPECT_EQ(assessment.maximum_sample_age_ns, kMaxSampleAgeNs);
  EXPECT_EQ(assessment.maximum_robot_message_age_ns, 500'000'000LL);
}

TEST(HardwareIoStatusPolicy, FreshStatusSeedPassesTheSameStrictWireGuardAsBootstrap)
{
  auto policy = make_policy();
  auto status = fresh_status();
  for (std::size_t index = 0; index < status.q.size(); ++index) {
    status.q[index] = static_cast<double>(static_cast<float>(0.15 + 0.1 * index));
  }
  const auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  ASSERT_TRUE(assessment.valid);

  std::array<control::PositionBoundSet, 6U> bounds{};
  for (std::size_t index = 0; index < bounds.size(); ++index) {
    const auto set = control::makePositionBoundSet(
      control::PositionInterval{-2.0, 2.0},
      control::PositionInterval{-1.0, 1.0},
      control::PositionInterval{-0.8, 0.8}, control::Radians{0.0});
    ASSERT_TRUE(set.has_value());
    bounds[index] = *set;
  }

  // This is the q_hold seed copied from the admitted coherent read. The exact
  // same final guard used before trajectory publication must accept each joint.
  for (std::size_t index = 0; index < bounds.size(); ++index) {
    EXPECT_EQ(control::classifyMeasuredPosition(
        assessment.sample.position[index], bounds[index].hardware),
      control::BoundaryClass::kInterior);
    EXPECT_TRUE(control::isWireCommand(
        assessment.sample.position[index], bounds[index].wire));
  }
}

TEST(HardwareIoStatusPolicy, RejectsChangedPayloadForSameReadSequence)
{
  auto policy = make_policy();
  auto status = fresh_status();
  ASSERT_TRUE(policy.observe(status, kNowRosNs, SteadyTime{}).valid);
  status.header.stamp = to_stamp(kNowRosNs + 1'000'000LL);
  status.read_attempt_sequence++;
  status.q[0] += 0.001;

  const auto assessment = policy.observe(
    status, kNowRosNs + 1'000'000LL, SteadyTime{} + std::chrono::milliseconds(1));
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kSequenceRegressed);
}

TEST(HardwareIoStatusPolicy, RejectsFaultedRobotReport)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.robot_mode = 0;
  const auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kRobotMessageUnavailable);
}

TEST(HardwareIoStatusPolicy, RejectsDisabledActionsLatchedInhibitAndBadWrite)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.runtime_actions_enabled = false;
  auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kRuntimeActionsDisabled);

  status = fresh_status();
  status.inhibit_latched = true;
  policy.reset();
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kInhibitLatched);

  status = fresh_status();
  status.write_attempted = true;
  status.write_ok = false;
  policy.reset();
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kWriteFailed);
}

TEST(HardwareIoStatusPolicy, RejectsWrongSourceAndNonFiniteValues)
{
  auto policy = make_policy();
  auto status = fresh_status();
  status.source_kind = msg::HardwareIoStatus::SOURCE_SIMULATION;
  auto assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kWrongSource);

  status = fresh_status();
  status.q[2] = std::numeric_limits<double>::quiet_NaN();
  policy.reset();
  assessment = policy.observe(status, kNowRosNs, SteadyTime{});
  EXPECT_FALSE(assessment.valid);
  EXPECT_EQ(assessment.failure, HardwareIoStatusFailure::kNonFiniteValue);
}

}  // namespace
}  // namespace face_tracking_arm::hardware
