// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FACE_TRACKING_ARM__EXECUTION_ERROR_BUDGET_HPP_
#define FACE_TRACKING_ARM__EXECUTION_ERROR_BUDGET_HPP_

#include <cmath>
#include <string>

namespace face_tracking_arm::control
{

enum class ExecutionBudgetSource
{
  kUnspecified,
  kSyntheticTest,
  kHardwareProfile,
};

enum class ExecutionBudgetFailure
{
  kNone,
  kMissingSource,
  kUnknownSource,
  kSyntheticNotAllowed,
  kMissingIdentity,
  kProfileMismatch,
  kInvalidNumber,
  kInvalidOrdering,
  kMissingEvidenceReference,
};

[[nodiscard]] inline const char * executionBudgetFailureName(
  const ExecutionBudgetFailure failure) noexcept
{
  switch (failure) {
    case ExecutionBudgetFailure::kNone: return "none";
    case ExecutionBudgetFailure::kMissingSource: return "missing_source";
    case ExecutionBudgetFailure::kUnknownSource: return "unknown_source";
    case ExecutionBudgetFailure::kSyntheticNotAllowed: return "synthetic_not_allowed";
    case ExecutionBudgetFailure::kMissingIdentity: return "missing_identity";
    case ExecutionBudgetFailure::kProfileMismatch: return "profile_mismatch";
    case ExecutionBudgetFailure::kInvalidNumber: return "invalid_number";
    case ExecutionBudgetFailure::kInvalidOrdering: return "invalid_ordering";
    case ExecutionBudgetFailure::kMissingEvidenceReference: return "missing_evidence_reference";
  }
  return "unknown_failure";
}

struct ExecutionBudgetIdentity
{
  std::string profile_id;
  std::string robot_model;
  int robot_model_num{0};
  std::string firmware;
  std::string controller_mode;
  std::string conditions_fingerprint;
  std::string evidence_reference;
  bool reduced_mode_known{false};
  bool reduced_mode_enabled{false};
};

/// One profile-wide bound: E covers command error through the complete stop,
/// T is the reaction threshold, and R is the additional growth through stop.
struct ExecutionErrorBudget
{
  ExecutionBudgetSource source{ExecutionBudgetSource::kUnspecified};
  ExecutionBudgetIdentity identity{};
  double total_error_rad{0.0};
  double reaction_threshold_rad{0.0};
  double post_threshold_growth_rad{0.0};
};

struct ExecutionBudgetExpectedIdentity
{
  std::string profile_id;
  std::string robot_model;
  int robot_model_num{0};
  std::string firmware;
  std::string controller_mode;
  std::string conditions_fingerprint;
  bool reduced_mode_enabled{false};
};

struct ExecutionBudgetValidation
{
  bool valid{false};
  bool physical_profile{false};
  ExecutionBudgetFailure failure{ExecutionBudgetFailure::kMissingSource};
};

[[nodiscard]] inline bool executionResponseGrowthExceeded(
  const double position_error_rad, const ExecutionErrorBudget & budget) noexcept
{
  return position_error_rad >
         budget.reaction_threshold_rad + budget.post_threshold_growth_rad;
}

/// Validates the offline profile contract. Synthetic input can be enabled only
/// by an explicit test caller and can never be classified as physical.
[[nodiscard]] inline ExecutionBudgetValidation validateExecutionErrorBudget(
  const ExecutionErrorBudget & budget,
  const ExecutionBudgetExpectedIdentity & expected,
  const bool allow_synthetic_test_profile) noexcept
{
  const auto fail = [](const ExecutionBudgetFailure reason) {
      return ExecutionBudgetValidation{false, false, reason};
    };
  if (budget.source == ExecutionBudgetSource::kUnspecified) {
    return fail(ExecutionBudgetFailure::kMissingSource);
  }
  if (budget.source != ExecutionBudgetSource::kSyntheticTest &&
    budget.source != ExecutionBudgetSource::kHardwareProfile)
  {
    return fail(ExecutionBudgetFailure::kUnknownSource);
  }
  if (budget.source == ExecutionBudgetSource::kSyntheticTest &&
    !allow_synthetic_test_profile)
  {
    return fail(ExecutionBudgetFailure::kSyntheticNotAllowed);
  }
  const auto & identity = budget.identity;
  if (identity.profile_id.empty() || identity.robot_model.empty() ||
    identity.robot_model_num <= 0 ||
    identity.firmware.empty() || identity.controller_mode.empty() ||
    identity.conditions_fingerprint.empty() ||
    !identity.reduced_mode_known)
  {
    return fail(ExecutionBudgetFailure::kMissingIdentity);
  }
  if (budget.source == ExecutionBudgetSource::kHardwareProfile &&
    identity.evidence_reference.empty())
  {
    return fail(ExecutionBudgetFailure::kMissingEvidenceReference);
  }
  if (identity.profile_id != expected.profile_id ||
    identity.robot_model != expected.robot_model ||
    identity.robot_model_num != expected.robot_model_num ||
    identity.firmware != expected.firmware ||
    identity.controller_mode != expected.controller_mode ||
    identity.conditions_fingerprint != expected.conditions_fingerprint ||
    identity.reduced_mode_enabled != expected.reduced_mode_enabled)
  {
    return fail(ExecutionBudgetFailure::kProfileMismatch);
  }
  if (!std::isfinite(budget.total_error_rad) ||
    !std::isfinite(budget.reaction_threshold_rad) ||
    !std::isfinite(budget.post_threshold_growth_rad) ||
    budget.total_error_rad <= 0.0 || budget.reaction_threshold_rad < 0.0 ||
    budget.post_threshold_growth_rad < 0.0)
  {
    return fail(ExecutionBudgetFailure::kInvalidNumber);
  }
  if (budget.reaction_threshold_rad + budget.post_threshold_growth_rad >
    budget.total_error_rad)
  {
    return fail(ExecutionBudgetFailure::kInvalidOrdering);
  }
  return ExecutionBudgetValidation{
    true, budget.source == ExecutionBudgetSource::kHardwareProfile,
    ExecutionBudgetFailure::kNone};
}

}  // namespace face_tracking_arm::control

#endif  // FACE_TRACKING_ARM__EXECUTION_ERROR_BUDGET_HPP_
