// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <limits>

#include "face_tracking_arm/execution_error_budget.hpp"
#include "face_tracking_arm/joint_boundary_policy.hpp"

namespace face_tracking_arm::control
{
namespace
{

ExecutionBudgetExpectedIdentity expectedIdentity()
{
  return {"xarm6_lab_v1", "xarm6", 1303, "2.7.0", "position_servoj",
    "sha256:test", false};
}

ExecutionErrorBudget profile(const ExecutionBudgetSource source)
{
  ExecutionErrorBudget result;
  result.source = source;
  result.identity = {
    "xarm6_lab_v1", "xarm6", 1303, "2.7.0", "position_servoj", "sha256:test",
    source == ExecutionBudgetSource::kHardwareProfile ? "preflight:2026-10-22" : "",
    true, false};
  result.total_error_rad = 0.03;
  result.reaction_threshold_rad = 0.01;
  result.post_threshold_growth_rad = 0.02;
  return result;
}

TEST(ExecutionErrorBudgetTest, AcceptsMatchedHardwareProfileWithCompleteBudget)
{
  const auto result = validateExecutionErrorBudget(
    profile(ExecutionBudgetSource::kHardwareProfile), expectedIdentity(), false);
  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(result.physical_profile);
  EXPECT_EQ(result.failure, ExecutionBudgetFailure::kNone);
}

TEST(ExecutionErrorBudgetTest, SyntheticProfileIsTestOnly)
{
  const auto budget = profile(ExecutionBudgetSource::kSyntheticTest);
  const auto rejected = validateExecutionErrorBudget(budget, expectedIdentity(), false);
  EXPECT_FALSE(rejected.valid);
  EXPECT_EQ(rejected.failure, ExecutionBudgetFailure::kSyntheticNotAllowed);

  const auto accepted = validateExecutionErrorBudget(budget, expectedIdentity(), true);
  EXPECT_TRUE(accepted.valid);
  EXPECT_FALSE(accepted.physical_profile);
}

TEST(ExecutionErrorBudgetTest, RejectsMissingAndUnknownSource)
{
  auto budget = profile(ExecutionBudgetSource::kUnspecified);
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kMissingSource);
  budget.source = static_cast<ExecutionBudgetSource>(255);
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), true).failure,
    ExecutionBudgetFailure::kUnknownSource);
}

TEST(ExecutionErrorBudgetTest, RejectsMissingIdentityAndEvidence)
{
  auto budget = profile(ExecutionBudgetSource::kHardwareProfile);
  budget.identity.firmware.clear();
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kMissingIdentity);

  budget = profile(ExecutionBudgetSource::kHardwareProfile);
  budget.identity.evidence_reference.clear();
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kMissingEvidenceReference);
}

TEST(ExecutionErrorBudgetTest, RejectsProfileIdentityMismatch)
{
  const auto expected = expectedIdentity();
  const auto change = [&](const auto & modify) {
      auto budget = profile(ExecutionBudgetSource::kHardwareProfile);
      modify(budget.identity);
      EXPECT_EQ(validateExecutionErrorBudget(budget, expected, false).failure,
        ExecutionBudgetFailure::kProfileMismatch);
    };
  change([](auto & identity) {identity.profile_id = "other";});
  change([](auto & identity) {identity.robot_model = "lite6";});
  change([](auto & identity) {identity.robot_model_num = 1305;});
  change([](auto & identity) {identity.firmware = "2.7.1";});
  change([](auto & identity) {identity.controller_mode = "joint";});
  change([](auto & identity) {identity.conditions_fingerprint = "sha256:other";});
  change([](auto & identity) {identity.reduced_mode_enabled = true;});
}

TEST(ExecutionErrorBudgetTest, RejectsInvalidBudgetNumbersAndOrdering)
{
  auto budget = profile(ExecutionBudgetSource::kHardwareProfile);
  budget.total_error_rad = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kInvalidNumber);

  budget = profile(ExecutionBudgetSource::kHardwareProfile);
  budget.reaction_threshold_rad = 0.02;
  budget.post_threshold_growth_rad = 0.02;
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kInvalidOrdering);
}

TEST(ExecutionErrorBudgetTest, AcceptsExactThresholdGrowthBoundary)
{
  auto budget = profile(ExecutionBudgetSource::kHardwareProfile);
  budget.total_error_rad = 0.03;
  budget.reaction_threshold_rad = 0.01;
  budget.post_threshold_growth_rad = 0.02;
  EXPECT_TRUE(validateExecutionErrorBudget(budget, expectedIdentity(), false).valid);

  budget.post_threshold_growth_rad = std::nextafter(0.02, 1.0);
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kInvalidOrdering);
}

TEST(ExecutionErrorBudgetTest, ResponseThresholdAndGrowthUseStrictProfileBoundaries)
{
  const auto budget = profile(ExecutionBudgetSource::kHardwareProfile);
  EXPECT_FALSE(executionResponseGrowthExceeded(0.03, budget));
  EXPECT_TRUE(executionResponseGrowthExceeded(std::nextafter(0.03, 1.0), budget));
}

TEST(ExecutionErrorBudgetTest, TotalErrorBudgetShrinksHardwareWireInterval)
{
  const auto zero = makePositionBoundSet(
    {-2.0, 2.0}, {-1.0, 1.0}, {-1.0, 1.0}, Radians{0.0});
  const auto with_error = makePositionBoundSet(
    {-2.0, 2.0}, {-1.0, 1.0}, {-1.0, 1.0}, Radians{0.03});
  ASSERT_TRUE(zero.has_value());
  ASSERT_TRUE(with_error.has_value());
  EXPECT_GE(with_error->wire.lower_rad, -0.97);
  EXPECT_LE(with_error->wire.upper_rad, 0.97);
  EXPECT_GT(with_error->wire.lower_rad, zero->wire.lower_rad);
  EXPECT_LT(with_error->wire.upper_rad, zero->wire.upper_rad);
}

TEST(ExecutionErrorBudgetTest, RejectsSumOverflow)
{
  auto budget = profile(ExecutionBudgetSource::kHardwareProfile);
  budget.total_error_rad = std::numeric_limits<double>::max();
  budget.reaction_threshold_rad = std::numeric_limits<double>::max();
  budget.post_threshold_growth_rad = std::numeric_limits<double>::max();
  EXPECT_EQ(validateExecutionErrorBudget(budget, expectedIdentity(), false).failure,
    ExecutionBudgetFailure::kInvalidOrdering);
}

}  // namespace
}  // namespace face_tracking_arm::control
