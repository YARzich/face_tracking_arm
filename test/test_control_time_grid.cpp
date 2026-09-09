// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "face_tracking_arm/control_time_grid.hpp"

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

}  // namespace
}  // namespace face_tracking_arm::control
