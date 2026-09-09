// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>

#include "face_tracking_arm/rolling_statistics.hpp"

namespace
{

std::atomic<bool> track_allocations{false};
std::atomic<std::size_t> allocation_count{0};

}  // namespace

void * operator new(const std::size_t size)
{
  if (track_allocations.load(std::memory_order_relaxed)) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void * memory = std::malloc(size)) {
    return memory;
  }
  throw std::bad_alloc{};
}

void * operator new[](const std::size_t size)
{
  return ::operator new(size);
}

void operator delete(void * memory) noexcept
{
  std::free(memory);
}

void operator delete[](void * memory) noexcept
{
  std::free(memory);
}

void operator delete(void * memory, std::size_t) noexcept
{
  std::free(memory);
}

void operator delete[](void * memory, std::size_t) noexcept
{
  std::free(memory);
}

namespace face_tracking_arm::telemetry
{
namespace
{

constexpr double kTolerance = 1.0e-12;

TEST(RollingStatistics, RejectsZeroCapacity)
{
  EXPECT_THROW(RollingStatistics{0}, std::invalid_argument);
}

TEST(RollingStatistics, ReportsEmptyWindow)
{
  const RollingStatistics statistics{4};

  EXPECT_EQ(statistics.capacity(), 4U);
  EXPECT_EQ(statistics.count(), 0U);
  EXPECT_FALSE(statistics.full());
  EXPECT_FALSE(statistics.mean().has_value());
  EXPECT_FALSE(statistics.maximum().has_value());
  EXPECT_FALSE(statistics.p99().has_value());
}

TEST(RollingStatistics, RejectsNonFiniteSamplesWithoutChangingTheWindow)
{
  RollingStatistics statistics{3};
  ASSERT_TRUE(statistics.add(2.0));

  EXPECT_FALSE(statistics.add(std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(statistics.add(std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(statistics.add(-std::numeric_limits<double>::infinity()));
  EXPECT_EQ(statistics.count(), 1U);
  ASSERT_TRUE(statistics.mean().has_value());
  EXPECT_DOUBLE_EQ(*statistics.mean(), 2.0);
}

TEST(RollingStatistics, OverwritesOldestSamplesAndRecomputesTheMaximum)
{
  RollingStatistics statistics{3};
  ASSERT_TRUE(statistics.add(5.0));
  ASSERT_TRUE(statistics.add(1.0));
  ASSERT_TRUE(statistics.add(3.0));
  ASSERT_TRUE(statistics.add(2.0));

  EXPECT_TRUE(statistics.full());
  EXPECT_EQ(statistics.count(), 3U);
  ASSERT_TRUE(statistics.mean().has_value());
  ASSERT_TRUE(statistics.maximum().has_value());
  EXPECT_NEAR(*statistics.mean(), 2.0, kTolerance);
  EXPECT_DOUBLE_EQ(*statistics.maximum(), 3.0);

  ASSERT_TRUE(statistics.add(8.0));
  EXPECT_NEAR(*statistics.mean(), 13.0 / 3.0, kTolerance);
  EXPECT_DOUBLE_EQ(*statistics.maximum(), 8.0);
}

TEST(RollingStatistics, UsesDocumentedNearestRankPercentiles)
{
  RollingStatistics statistics{100};
  for (int sample = 100; sample >= 1; --sample) {
    ASSERT_TRUE(statistics.add(static_cast<double>(sample)));
  }

  ASSERT_TRUE(statistics.percentile(0.0).has_value());
  ASSERT_TRUE(statistics.percentile(0.5).has_value());
  ASSERT_TRUE(statistics.p99().has_value());
  ASSERT_TRUE(statistics.percentile(1.0).has_value());
  EXPECT_DOUBLE_EQ(*statistics.percentile(0.0), 1.0);
  EXPECT_DOUBLE_EQ(*statistics.percentile(0.5), 50.0);
  EXPECT_DOUBLE_EQ(*statistics.p99(), 99.0);
  EXPECT_DOUBLE_EQ(*statistics.percentile(1.0), 100.0);
  EXPECT_FALSE(statistics.percentile(-0.1).has_value());
  EXPECT_FALSE(statistics.percentile(1.1).has_value());
  EXPECT_FALSE(statistics.percentile(
      std::numeric_limits<double>::quiet_NaN()).has_value());
}

TEST(RollingStatistics, ClearRetainsCapacityAndResetsAggregates)
{
  RollingStatistics statistics{2};
  ASSERT_TRUE(statistics.add(-2.0));
  ASSERT_TRUE(statistics.add(-1.0));

  statistics.clear();

  EXPECT_EQ(statistics.capacity(), 2U);
  EXPECT_EQ(statistics.count(), 0U);
  EXPECT_FALSE(statistics.mean().has_value());
  EXPECT_FALSE(statistics.maximum().has_value());
  ASSERT_TRUE(statistics.add(4.0));
  EXPECT_DOUBLE_EQ(*statistics.maximum(), 4.0);
}

TEST(RollingStatistics, AddDoesNotAllocateAfterConstruction)
{
  RollingStatistics statistics{128};
  allocation_count.store(0, std::memory_order_relaxed);
  bool all_samples_accepted = true;

  track_allocations.store(true, std::memory_order_relaxed);
  for (int sample = 0; sample < 10000; ++sample) {
    all_samples_accepted = statistics.add(static_cast<double>(sample)) &&
      all_samples_accepted;
  }
  track_allocations.store(false, std::memory_order_relaxed);

  EXPECT_TRUE(all_samples_accepted);
  EXPECT_EQ(allocation_count.load(std::memory_order_relaxed), 0U);
  EXPECT_EQ(statistics.count(), statistics.capacity());
}

}  // namespace
}  // namespace face_tracking_arm::telemetry
