// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "face_tracking_arm/servo_execution_context_ring.hpp"

namespace face_tracking_arm::diagnostics
{
namespace
{

TEST(ServoExecutionContextRing, PreservesOrderAndReportsBoundedOverflow)
{
  ServoExecutionContextRing<2> ring;
  ServoExecutionContextRecord first;
  first.tick_sequence = 10U;
  first.cycle_duration_ns = 1'234'567U;
  ServoExecutionContextRecord second;
  second.tick_sequence = 11U;
  ServoExecutionContextRecord rejected;
  rejected.tick_sequence = 12U;
  ASSERT_TRUE(ring.tryPush(first));
  ASSERT_TRUE(ring.tryPush(second));
  EXPECT_FALSE(ring.tryPush(rejected));
  EXPECT_EQ(ring.droppedCount(), 1U);
  ServoExecutionContextRecord actual;
  ASSERT_TRUE(ring.tryPop(actual));
  EXPECT_EQ(actual.tick_sequence, 10U);
  EXPECT_EQ(actual.cycle_duration_ns, 1'234'567U);
  ASSERT_TRUE(ring.tryPop(actual));
  EXPECT_EQ(actual.tick_sequence, 11U);
  EXPECT_FALSE(ring.tryPop(actual));
}

}  // namespace
}  // namespace face_tracking_arm::diagnostics
