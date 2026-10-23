// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

// Standalone CL-15 microbenchmark for OSQP setup/solve/cleanup. This is a synthetic
// maximum-shape kernel, not a robot or end-to-end controller benchmark.

#include <dlfcn.h>

#include <osqp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using AllocControl = void (*)();
using AllocRead = std::uint64_t (*)();

struct AllocatorCounter
{
  AllocControl begin{reinterpret_cast<AllocControl>(dlsym(RTLD_DEFAULT, "alloc_counter_begin"))};
  AllocControl end{reinterpret_cast<AllocControl>(dlsym(RTLD_DEFAULT, "alloc_counter_end"))};
  AllocRead malloc_calls{reinterpret_cast<AllocRead>(dlsym(RTLD_DEFAULT,
        "alloc_counter_malloc_calls"))};
  AllocRead calloc_calls{reinterpret_cast<AllocRead>(dlsym(RTLD_DEFAULT,
        "alloc_counter_calloc_calls"))};
  AllocRead realloc_calls{reinterpret_cast<AllocRead>(dlsym(RTLD_DEFAULT,
        "alloc_counter_realloc_calls"))};
  AllocRead frees{reinterpret_cast<AllocRead>(dlsym(RTLD_DEFAULT, "alloc_counter_frees"))};
  AllocRead bytes{reinterpret_cast<AllocRead>(dlsym(RTLD_DEFAULT, "alloc_counter_bytes"))};
  bool available() const
  {
    return begin && end && malloc_calls && calloc_calls && realloc_calls && frees && bytes;
  }
};
constexpr c_int kVariables = 6;
constexpr c_int kSafetyRows = 137;
constexpr c_int kPrimaryRows = 6;
constexpr c_int kPrimaryConstraintRows = kVariables + kSafetyRows;
constexpr c_int kSecondaryConstraintRows = kPrimaryConstraintRows + kPrimaryRows;
constexpr c_int kSnapshotCapacityRows = 160;
constexpr int kIterations = 4000;

struct Csc
{
  std::vector<c_int> indptr;
  std::vector<c_int> indices;
  std::vector<c_float> values;
};

struct Samples
{
  std::vector<double> setup_us;
  std::vector<double> solve_us;
  std::vector<double> cleanup_us;
  std::vector<double> warm_start_us;
  std::vector<int> solver_iterations;
  int solved{0};
  int failed{0};
  std::uint64_t lifecycle_malloc_calls{0};
  std::uint64_t lifecycle_calloc_calls{0};
  std::uint64_t lifecycle_realloc_calls{0};
  std::uint64_t lifecycle_frees{0};
  std::uint64_t lifecycle_bytes{0};
  bool allocator_counter_available{false};
};

double elapsedUs(const Clock::time_point start, const Clock::time_point end)
{
  return std::chrono::duration<double, std::micro>(end - start).count();
}

Csc makeConstraintMatrix(const c_int row_count)
{
  Csc matrix;
  matrix.indptr.reserve(kVariables + 1);
  matrix.indices.reserve(kVariables * 10);
  matrix.values.reserve(kVariables * 10);
  matrix.indptr.push_back(0);
  for (c_int column = 0; column < kVariables; ++column) {
    for (c_int row = 0; row < row_count; ++row) {
      const bool is_primary_preservation_row = row >= kPrimaryConstraintRows &&
        row < kSecondaryConstraintRows && column == row - kPrimaryConstraintRows;
      if (row < kVariables && row == column) {
        matrix.indices.push_back(row);
        matrix.values.push_back(1.0);
      } else if (row >= kVariables && row < kPrimaryConstraintRows) {
        const c_int pattern = ((row + 1) * (column + 3)) % 13;
        matrix.indices.push_back(row);
        matrix.values.push_back((pattern % 2 == 0 ? 1.0 : -1.0) *
          static_cast<c_float>(pattern + 1) * 0.005);
      } else if (is_primary_preservation_row) {
        matrix.indices.push_back(row);
        matrix.values.push_back(1.0);
      } else if (row >= kSecondaryConstraintRows) {
        // Exercise full row/storage caps with finite synthetic row coefficients.
        const c_int pattern = ((row + 1) * (column + 3)) % 13;
        matrix.indices.push_back(row);
        matrix.values.push_back((pattern % 2 == 0 ? 1.0 : -1.0) *
          static_cast<c_float>(pattern + 1) * 0.005);
      }
    }
    matrix.indptr.push_back(static_cast<c_int>(matrix.values.size()));
  }
  return matrix;
}

void summarize(const char * stage, const std::vector<double> & values)
{
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](const double p) {
      const std::size_t index = static_cast<std::size_t>(p * (sorted.size() - 1U));
      return sorted[index];
    };
  const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) /
    static_cast<double>(sorted.size());
  std::cout << stage << "_us min=" << sorted.front()
            << " p50=" << percentile(0.50)
            << " p95=" << percentile(0.95)
            << " p99=" << percentile(0.99)
            << " max=" << sorted.back()
            << " mean=" << mean << '\n';
}

void runStage(const c_int row_count, const int count, const char * name, Samples & samples)
{
  constexpr c_float hessian_diagonal = 2.0;
  std::vector<c_int> p_indptr(kVariables + 1);
  std::vector<c_int> p_indices(kVariables);
  std::vector<c_float> p_values(kVariables, hessian_diagonal);
  for (c_int i = 0; i <= kVariables; ++i) {
    p_indptr[static_cast<std::size_t>(i)] = i;
  }
  for (c_int i = 0; i < kVariables; ++i) {
    p_indices[static_cast<std::size_t>(i)] = i;
  }
  Csc a = makeConstraintMatrix(row_count);
  std::vector<c_float> q(kVariables, -0.15);
  std::vector<c_float> lower(static_cast<std::size_t>(row_count), -0.25);
  std::vector<c_float> upper(static_cast<std::size_t>(row_count), 0.25);
  std::vector<c_float> warm(kVariables, 0.0);
  csc p_matrix{static_cast<c_int>(p_values.size()), kVariables, kVariables,
    p_indptr.data(), p_indices.data(), p_values.data(), -1};
  csc a_matrix{static_cast<c_int>(a.values.size()), row_count, kVariables,
    a.indptr.data(), a.indices.data(), a.values.data(), -1};
  OSQPData data{};
  data.n = kVariables;
  data.m = row_count;
  data.P = &p_matrix;
  data.A = &a_matrix;
  data.q = q.data();
  data.l = lower.data();
  data.u = upper.data();

  samples.setup_us.reserve(static_cast<std::size_t>(count));
  samples.solve_us.reserve(static_cast<std::size_t>(count));
  samples.cleanup_us.reserve(static_cast<std::size_t>(count));
  samples.warm_start_us.reserve(static_cast<std::size_t>(count));
  samples.solver_iterations.reserve(static_cast<std::size_t>(count));
  const AllocatorCounter allocator;
  samples.allocator_counter_available = allocator.available();
  for (int iteration = 0; iteration < count; ++iteration) {
    OSQPSettings settings{};
    osqp_set_default_settings(&settings);
    settings.verbose = 0;
    settings.polish = 0;
    settings.warm_start = 1;
    settings.adaptive_rho_interval = 10;
    settings.scaled_termination = 1;
    settings.check_termination = 1;
    settings.max_iter = 500;
    settings.eps_prim_inf = 1.0e-10;
    settings.eps_abs = 1.0e-6;
    settings.eps_rel = 1.0e-6;
    settings.time_limit = 0.003;

    OSQPWorkspace * workspace = nullptr;
    if (allocator.available()) {allocator.begin();}
    auto start = Clock::now();
    const c_int setup_code = osqp_setup(&workspace, &data, &settings);
    auto end = Clock::now();
    samples.setup_us.push_back(elapsedUs(start, end));
    if (setup_code != 0 || workspace == nullptr) {
      ++samples.failed;
      samples.solve_us.push_back(0.0);
      samples.warm_start_us.push_back(0.0);
      samples.cleanup_us.push_back(0.0);
      samples.solver_iterations.push_back(0);
      if (allocator.available()) {
        allocator.end();
        samples.lifecycle_malloc_calls += allocator.malloc_calls();
        samples.lifecycle_calloc_calls += allocator.calloc_calls();
        samples.lifecycle_realloc_calls += allocator.realloc_calls();
        samples.lifecycle_frees += allocator.frees();
        samples.lifecycle_bytes += allocator.bytes();
      }
      continue;
    }

    start = Clock::now();
    const c_int warm_code = osqp_warm_start_x(workspace, warm.data());
    end = Clock::now();
    samples.warm_start_us.push_back(elapsedUs(start, end));
    start = Clock::now();
    const c_int solve_code = warm_code == 0 ? osqp_solve(workspace) : warm_code;
    end = Clock::now();
    samples.solve_us.push_back(elapsedUs(start, end));
    const bool solved = solve_code == 0 && workspace->info != nullptr &&
      (workspace->info->status_val == OSQP_SOLVED ||
      workspace->info->status_val == OSQP_SOLVED_INACCURATE);
    if (solved) {
      ++samples.solved;
    } else {
      ++samples.failed;
    }
    samples.solver_iterations.push_back(
      workspace->info == nullptr ? 0 : static_cast<int>(workspace->info->iter));
    start = Clock::now();
    (void)osqp_cleanup(workspace);
    end = Clock::now();
    samples.cleanup_us.push_back(elapsedUs(start, end));
    if (allocator.available()) {
      allocator.end();
      samples.lifecycle_malloc_calls += allocator.malloc_calls();
      samples.lifecycle_calloc_calls += allocator.calloc_calls();
      samples.lifecycle_realloc_calls += allocator.realloc_calls();
      samples.lifecycle_frees += allocator.frees();
      samples.lifecycle_bytes += allocator.bytes();
    }
  }
  std::cout << "stage=" << name << " calls=" << count << " rows=" << row_count
            << " nnz_A=" << a.values.size() << " solved=" << samples.solved
            << " failed=" << samples.failed
            << " lifecycle_malloc_calls=" << samples.lifecycle_malloc_calls
            << " lifecycle_calloc_calls=" << samples.lifecycle_calloc_calls
            << " lifecycle_realloc_calls=" << samples.lifecycle_realloc_calls
            << " lifecycle_frees=" << samples.lifecycle_frees
            << " lifecycle_bytes_requested=" << samples.lifecycle_bytes
            << " allocator_counter=" <<
    (samples.allocator_counter_available ? "enabled" : "absent") << '\n';
}

}  // namespace

int main(int argc, char ** argv)
{
  int count = kIterations;
  if (argc == 2) {
    count = std::atoi(argv[1]);
  }
  if (count < 20 || count > 1000000) {
    std::cerr << "usage: profile_osqp_lifecycle [iterations: 20..1000000]\n";
    return 2;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "OSQP_VERSION=" << OSQP_VERSION
            << " synthetic_controller_joints=" << kVariables
            << " controller_bound_rows=" << kSecondaryConstraintRows
            << " snapshot_capacity_rows=" << kSnapshotCapacityRows
            << " ticks=" << count << " allocator_counter=" << (dlsym(RTLD_DEFAULT,
    "alloc_counter_begin") ? "preloaded" : "not_preloaded") << '\n';
  Samples primary;
  Samples secondary;
  Samples snapshot_capacity;
  runStage(kPrimaryConstraintRows, count, "primary", primary);
  runStage(kSecondaryConstraintRows, count, "secondary", secondary);
  runStage(kSnapshotCapacityRows, count, "snapshot_capacity", snapshot_capacity);
  summarize("primary_setup", primary.setup_us);
  summarize("primary_warm_start", primary.warm_start_us);
  summarize("primary_solve", primary.solve_us);
  summarize("primary_cleanup", primary.cleanup_us);
  summarize("secondary_setup", secondary.setup_us);
  summarize("secondary_warm_start", secondary.warm_start_us);
  summarize("secondary_solve", secondary.solve_us);
  summarize("secondary_cleanup", secondary.cleanup_us);
  summarize("snapshot_capacity_setup", snapshot_capacity.setup_us);
  summarize("snapshot_capacity_warm_start", snapshot_capacity.warm_start_us);
  summarize("snapshot_capacity_solve", snapshot_capacity.solve_us);
  summarize("snapshot_capacity_cleanup", snapshot_capacity.cleanup_us);
  const auto total_malloc_calls = primary.lifecycle_malloc_calls +
    secondary.lifecycle_malloc_calls +
    snapshot_capacity.lifecycle_malloc_calls;
  const auto total_calloc_calls = primary.lifecycle_calloc_calls +
    secondary.lifecycle_calloc_calls +
    snapshot_capacity.lifecycle_calloc_calls;
  const auto total_realloc_calls = primary.lifecycle_realloc_calls +
    secondary.lifecycle_realloc_calls +
    snapshot_capacity.lifecycle_realloc_calls;
  const auto total_frees = primary.lifecycle_frees + secondary.lifecycle_frees +
    snapshot_capacity.lifecycle_frees;
  const auto total_bytes = primary.lifecycle_bytes + secondary.lifecycle_bytes +
    snapshot_capacity.lifecycle_bytes;
  const bool allocation_counted = primary.allocator_counter_available &&
    secondary.allocator_counter_available && snapshot_capacity.allocator_counter_available;
  std::cout << "malloc_calls=" << (allocation_counted ? total_malloc_calls : 0U)
            << " calloc_calls=" << (allocation_counted ? total_calloc_calls : 0U)
            << " realloc_calls=" << (allocation_counted ? total_realloc_calls : 0U)
            << " free_calls=" << (allocation_counted ? total_frees : 0U)
            << " bytes_requested=" << (allocation_counted ? total_bytes : 0U)
            << " allocation_counter=" << (allocation_counted ? "enabled" : "absent")
            << " setup_count=" << 3 * count
            << " solve_count=" <<
    primary.solved + primary.failed + secondary.solved + secondary.failed
            << " max_iter_primary_mean=" << std::accumulate(
    primary.solver_iterations.begin(), primary.solver_iterations.end(), 0.0) /
    static_cast<double>(count)
            << " max_iter_secondary_mean=" << std::accumulate(
    secondary.solver_iterations.begin(), secondary.solver_iterations.end(), 0.0) /
    static_cast<double>(count) << '\n';
  return primary.failed == 0 && secondary.failed == 0 && snapshot_capacity.failed == 0 ? 0 : 1;
}
