// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "face_tracking_arm/control_time_grid.hpp"
#include "face_tracking_arm/snapshot_timing.hpp"

namespace face_tracking_arm::control
{
namespace
{

TEST(ControlTimeGrid, RejectsInvalidConfigurationAndInputs)
{
  EXPECT_THROW(ControlTimeGrid{0}, std::invalid_argument);
  EXPECT_THROW(ControlTimeGrid{-1}, std::invalid_argument);

  ControlTimeGrid grid{10};
  EXPECT_THROW(grid.observe(-1), std::invalid_argument);
  EXPECT_THROW(grid.commandStamp(-1, 0, 0), std::invalid_argument);
  EXPECT_THROW(grid.commandStamp(0, -1, 0), std::invalid_argument);
  EXPECT_THROW(grid.commandStamp(0, 0, -1), std::invalid_argument);
  EXPECT_THROW(grid.commandStamp(0, 4, 5), std::invalid_argument);
  EXPECT_THROW(grid.adoptPublishedStamp(-1), std::invalid_argument);
}

TEST(ControlTimeGrid, FastClockRunsNoMoreOftenThanTheFixedPeriod)
{
  ControlTimeGrid grid{10};
  std::uint64_t due_count = 0;

  for (std::int64_t now_ns = 100; now_ns < 200; ++now_ns) {
    const ControlTimeObservation observation = grid.observe(now_ns);
    if (observation.due()) {
      ++due_count;
      ASSERT_TRUE(observation.scheduled_time_ns.has_value());
      EXPECT_EQ(*observation.scheduled_time_ns, now_ns);
    } else {
      EXPECT_EQ(observation.status, ControlTimeStatus::kWait);
      EXPECT_FALSE(observation.scheduled_time_ns.has_value());
    }
  }

  EXPECT_EQ(due_count, 10U);
}

TEST(ControlTimeGrid, SlowIrregularClockDropsMissedPeriods)
{
  ControlTimeGrid grid{10};

  const ControlTimeObservation first = grid.observe(100);
  ASSERT_TRUE(first.due());
  EXPECT_EQ(first.scheduled_time_ns, 100);
  EXPECT_EQ(first.next_due_time_ns, 110);
  EXPECT_EQ(first.skipped_periods, 0U);

  const ControlTimeObservation late = grid.observe(135);
  ASSERT_TRUE(late.due());
  EXPECT_EQ(late.scheduled_time_ns, 130);
  EXPECT_EQ(late.next_due_time_ns, 140);
  EXPECT_EQ(late.skipped_periods, 2U);

  const ControlTimeObservation much_later = grid.observe(171);
  ASSERT_TRUE(much_later.due());
  EXPECT_EQ(much_later.scheduled_time_ns, 170);
  EXPECT_EQ(much_later.next_due_time_ns, 180);
  EXPECT_EQ(much_later.skipped_periods, 3U);
}

TEST(ControlTimeGrid, LargeJumpProducesOneDueObservationAndFutureBoundary)
{
  ControlTimeGrid grid{10};

  ASSERT_TRUE(grid.observe(0).due());
  const ControlTimeObservation jump = grid.observe(1'000'000'007);
  ASSERT_TRUE(jump.due());
  EXPECT_EQ(jump.scheduled_time_ns, 1'000'000'000);
  EXPECT_EQ(jump.skipped_periods, 99'999'999U);
  EXPECT_GT(jump.next_due_time_ns, 1'000'000'007);
  EXPECT_EQ(grid.observe(1'000'000'008).status, ControlTimeStatus::kWait);
}

TEST(ControlTimeGrid, JitterDoesNotMoveTheOriginalGrid)
{
  ControlTimeGrid grid{10};

  EXPECT_EQ(grid.observe(100).status, ControlTimeStatus::kDue);
  EXPECT_EQ(grid.observe(109).status, ControlTimeStatus::kWait);

  const ControlTimeObservation first_late_tick = grid.observe(111);
  ASSERT_TRUE(first_late_tick.due());
  EXPECT_EQ(first_late_tick.scheduled_time_ns, 110);
  EXPECT_EQ(first_late_tick.next_due_time_ns, 120);

  EXPECT_EQ(grid.observe(119).status, ControlTimeStatus::kWait);
  const ControlTimeObservation second_late_tick = grid.observe(121);
  ASSERT_TRUE(second_late_tick.due());
  EXPECT_EQ(second_late_tick.scheduled_time_ns, 120);
  EXPECT_EQ(second_late_tick.next_due_time_ns, 130);
}

TEST(ControlTimeGrid, ReportsStoppedClockWithoutConsumingTheNextSlot)
{
  ControlTimeGrid grid{10};

  ASSERT_TRUE(grid.observe(50).due());
  EXPECT_EQ(grid.observe(55).status, ControlTimeStatus::kWait);

  const ControlTimeObservation stopped = grid.observe(55);
  EXPECT_EQ(stopped.status, ControlTimeStatus::kNotAdvancing);
  EXPECT_EQ(stopped.next_due_time_ns, 60);

  const ControlTimeObservation resumed = grid.observe(60);
  EXPECT_TRUE(resumed.due());
  EXPECT_EQ(resumed.scheduled_time_ns, 60);
}

TEST(ControlTimeGrid, RewindStartsANewObservationEpoch)
{
  ControlTimeGrid grid{10};

  ASSERT_TRUE(grid.observe(100).due());
  const ControlTimeObservation late = grid.observe(125);
  ASSERT_TRUE(late.due());
  EXPECT_EQ(late.next_due_time_ns, 130);

  const ControlTimeObservation rewind = grid.observe(7);
  EXPECT_EQ(rewind.status, ControlTimeStatus::kRewind);
  EXPECT_FALSE(rewind.scheduled_time_ns.has_value());
  EXPECT_EQ(rewind.next_due_time_ns, 17);
  EXPECT_EQ(grid.observe(7).status, ControlTimeStatus::kNotAdvancing);
  EXPECT_EQ(grid.observe(16).status, ControlTimeStatus::kWait);

  const ControlTimeObservation due = grid.observe(17);
  ASSERT_TRUE(due.due());
  EXPECT_EQ(due.scheduled_time_ns, 17);
  EXPECT_EQ(due.next_due_time_ns, 27);
}

TEST(ControlTimeGrid, CommandStampsStayExactlyOnePeriodApartDespiteJitter)
{
  ControlTimeGrid grid{10};

  EXPECT_EQ(grid.commandStamp(100, 50, 20), 150);
  EXPECT_EQ(grid.commandStamp(109, 50, 20), 160);
  EXPECT_EQ(grid.commandStamp(121, 50, 20), 170);
  EXPECT_EQ(grid.commandStamp(129, 50, 20), 180);
}

TEST(ControlTimeGrid, RefusesToRebaseADepletedCommandBuffer)
{
  ControlTimeGrid grid{10};

  EXPECT_EQ(grid.commandStamp(100, 50, 20), 150);
  EXPECT_EQ(grid.commandStamp(120, 50, 20), 160);
  EXPECT_EQ(grid.commandStamp(139, 50, 20), 170);
  EXPECT_FALSE(grid.commandStamp(161, 50, 20).has_value());
  // A refusal must not consume a grid slot.
  EXPECT_EQ(grid.commandStamp(159, 50, 20), 180);
}

TEST(ControlTimeGrid, CountsPublishedPeriodsNeededForTargetLead)
{
  ControlTimeGrid grid{10};
  EXPECT_FALSE(grid.periodsUntilCommandLead(100, 20).has_value());
  ASSERT_EQ(grid.commandStamp(100, 50, 20), 150);

  EXPECT_EQ(grid.periodsUntilCommandLead(121, 20), 0U);
  EXPECT_EQ(grid.periodsUntilCommandLead(141, 20), 1U);
  EXPECT_EQ(grid.periodsUntilCommandLead(171, 20), 4U);
}

TEST(ControlTimeGrid, RecoveredAnchorRetainsComputeLeadAcrossNearBoundaryDelay)
{
  ControlTimeGrid grid{10};
  ASSERT_EQ(grid.commandStamp(100, 10, 0), 110);

  // At 109, the original next stamp has only 11 units of lead. Preserve a
  // 30-unit epoch lead (20 minimum plus one 10-unit compute period).
  ASSERT_EQ(grid.periodsUntilCommandLead(109, 30), 2U);
  ASSERT_TRUE(grid.adoptPublishedStamp(130));

  // Five units of work after the near-boundary recovery still leaves at least
  // the 20-unit physical minimum when the next command is appended.
  EXPECT_EQ(grid.commandStamp(114, 50, 20), 140);
}

TEST(ControlTimeGrid, AdoptsOnlyAFutureStampOnTheActiveCommandGrid)
{
  ControlTimeGrid grid{10};

  EXPECT_FALSE(grid.adoptPublishedStamp(150));
  ASSERT_EQ(grid.commandStamp(100, 50, 20), 150);
  EXPECT_FALSE(grid.adoptPublishedStamp(150));
  EXPECT_FALSE(grid.adoptPublishedStamp(149));
  EXPECT_FALSE(grid.adoptPublishedStamp(171));

  EXPECT_TRUE(grid.adoptPublishedStamp(180));
  EXPECT_EQ(grid.commandStamp(140, 50, 20), 190);
}

TEST(ControlTimeGrid, PublishedStampAdoptionRestoresLeadWithoutChangingPeriod)
{
  ControlTimeGrid grid{10};

  ASSERT_EQ(grid.commandStamp(100, 50, 20), 150);
  ASSERT_EQ(grid.commandStamp(120, 50, 20), 160);
  EXPECT_FALSE(grid.commandStamp(151, 50, 20).has_value());

  // 170 and 180 are assumed to be immutable points from the trajectory that
  // was already sent before the callback became late.
  EXPECT_TRUE(grid.adoptPublishedStamp(180));
  EXPECT_EQ(grid.commandStamp(151, 50, 20), 190);
}

TEST(ControlTimeGrid, CommandEpochCanBeResetAfterAClockRewind)
{
  ControlTimeGrid grid{10};

  EXPECT_EQ(grid.commandStamp(100, 5, 2), 105);
  EXPECT_EQ(grid.commandStamp(110, 5, 2), 115);
  grid.resetCommandEpoch();
  EXPECT_EQ(grid.commandStamp(7, 2, 1), 9);
}

TEST(ControlTimeGrid, DetectsTimestampOverflowWithoutWrapping)
{
  constexpr std::int64_t maximum = std::numeric_limits<std::int64_t>::max();

  ControlTimeGrid observation_grid{10};
  EXPECT_THROW(observation_grid.observe(maximum - 5), std::overflow_error);

  ControlTimeGrid command_grid{10};
  EXPECT_THROW(command_grid.commandStamp(maximum, 1, 0), std::overflow_error);
  EXPECT_EQ(command_grid.commandStamp(maximum - 10, 0, 0), maximum - 10);
  EXPECT_EQ(command_grid.commandStamp(maximum - 10, 0, 0), maximum);
  EXPECT_THROW(command_grid.commandStamp(maximum - 10, 0, 0), std::overflow_error);
}

TEST(SnapshotTiming, TargetArrivingAfterTickStartIsReadBeforeItsValidationTime)
{
  std::int64_t clock_ns = 1'000;
  std::int64_t latest_target_stamp_ns = 0;
  const auto [target_stamp_ns, validation_time_ns] = readSnapshotAtCurrentTime(
    [&]() {
      // Model receive_target replacing latest_pose_ before current_target()
      // acquires its mutex, after control_tick sampled its start time.
      clock_ns = 1'001;
      latest_target_stamp_ns = 1'001;
      return latest_target_stamp_ns;
    },
    [&clock_ns]() {return clock_ns;});
  EXPECT_EQ(target_stamp_ns, 1'001);
  EXPECT_EQ(validation_time_ns, 1'001);
  EXPECT_TRUE(isInSameClockEpoch(1'000, validation_time_ns));
  EXPECT_TRUE(isFreshSnapshotTimestamp(target_stamp_ns, validation_time_ns, 20));
}

TEST(SnapshotTiming, FeedbackUpdateBetweenTickStartAndStateSnapshotIsAccepted)
{
  std::int64_t clock_ns = 1'000;
  std::int64_t feedback_stamp_ns = 0;
  const auto [snapshot_stamp_ns, snapshot_time_ns] = readSnapshotAtCurrentTime(
    [&]() {
      // Model a MoveIt state-monitor update before getCurrentStateAndTime().
      clock_ns = 1'001;
      feedback_stamp_ns = 1'001;
      return feedback_stamp_ns;
    },
    [&clock_ns]() {return clock_ns;});
  EXPECT_EQ(snapshot_stamp_ns, 1'001);
  EXPECT_EQ(snapshot_time_ns, 1'001);
  EXPECT_TRUE(isInSameClockEpoch(1'000, snapshot_time_ns));
  EXPECT_TRUE(isFreshSnapshotTimestamp(snapshot_stamp_ns, snapshot_time_ns, 20));
}

TEST(SnapshotTiming, KeepsFutureStaleAndRewoundFeedbackInvalid)
{
  const auto [ordinary_stamp_ns, ordinary_time_ns] = readSnapshotAtCurrentTime(
    []() {return std::int64_t{900};},
    []() {return std::int64_t{900};});
  EXPECT_TRUE(isFreshSnapshotTimestamp(ordinary_stamp_ns, ordinary_time_ns, 20));
  EXPECT_TRUE(isInSameClockEpoch(ordinary_time_ns, ordinary_time_ns));

  // An actually future sample stays invalid relative to the post-snapshot time.
  EXPECT_FALSE(isFreshSnapshotTimestamp(1'002, 1'001, 20));
  // Freshness remains bounded; taking a later time does not bless stale state.
  EXPECT_FALSE(isFreshSnapshotTimestamp(900, 1'001, 100));

  std::int64_t clock_ns = 2'000;
  const auto [rewound_stamp_ns, rewound_snapshot_time_ns] = readSnapshotAtCurrentTime(
    [&]() {
      // A ROS-time reset between the old tick time and snapshot must use the
      // rewound time, so data from the previous time epoch is still future.
      clock_ns = 1'000;
      return std::int64_t{1'500};
    },
    [&clock_ns]() {return clock_ns;});
  EXPECT_EQ(rewound_stamp_ns, 1'500);
  EXPECT_EQ(rewound_snapshot_time_ns, 1'000);
  EXPECT_FALSE(isInSameClockEpoch(2'000, rewound_snapshot_time_ns));
  EXPECT_FALSE(isFreshSnapshotTimestamp(
      rewound_stamp_ns, rewound_snapshot_time_ns, 20));
}

}  // namespace
}  // namespace face_tracking_arm::control
