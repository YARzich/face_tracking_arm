// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

// Linux/glibc LD_PRELOAD allocator counter for bounded benchmark regions only.
#include <malloc.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

extern "C" void * __libc_malloc(std::size_t) noexcept;
extern "C" void * __libc_calloc(std::size_t, std::size_t) noexcept;
extern "C" void * __libc_realloc(void *, std::size_t) noexcept;
extern "C" void __libc_free(void * ptr) noexcept;
namespace
{
std::atomic<bool> active{false};
std::atomic<std::uint64_t> malloc_calls{0};
std::atomic<std::uint64_t> calloc_calls{0};
std::atomic<std::uint64_t> realloc_calls{0};
std::atomic<std::uint64_t> frees{0};
std::atomic<std::uint64_t> bytes_requested{0};
}
extern "C" void alloc_counter_begin() noexcept
{
  malloc_calls.store(0, std::memory_order_relaxed);
  calloc_calls.store(0, std::memory_order_relaxed);
  realloc_calls.store(0, std::memory_order_relaxed);
  frees.store(0, std::memory_order_relaxed);
  bytes_requested.store(0, std::memory_order_relaxed);
  active.store(true, std::memory_order_release);
}
extern "C" void alloc_counter_end() noexcept {active.store(false, std::memory_order_release);}
extern "C" std::uint64_t alloc_counter_malloc_calls() noexcept {return malloc_calls.load();}
extern "C" std::uint64_t alloc_counter_calloc_calls() noexcept {return calloc_calls.load();}
extern "C" std::uint64_t alloc_counter_realloc_calls() noexcept {return realloc_calls.load();}
extern "C" std::uint64_t alloc_counter_frees() noexcept {return frees.load();}
extern "C" std::uint64_t alloc_counter_bytes() noexcept {return bytes_requested.load();}
extern "C" void * malloc(std::size_t size) noexcept
{
  void *ptr = __libc_malloc(size);
  if (active.load(std::memory_order_acquire) && ptr) {
    malloc_calls.fetch_add(1, std::memory_order_relaxed);
    bytes_requested.fetch_add(size, std::memory_order_relaxed);
  }
  return ptr;
}
extern "C" void * calloc(std::size_t count, std::size_t size) noexcept
{
  void *ptr = __libc_calloc(count, size);
  if (active.load(std::memory_order_acquire) && ptr) {
    calloc_calls.fetch_add(1, std::memory_order_relaxed);
    bytes_requested.fetch_add(count * size, std::memory_order_relaxed);
  }
  return ptr;
}
extern "C" void * realloc(void *old_ptr, std::size_t size) noexcept
{
  void *ptr = __libc_realloc(old_ptr, size);
  if (active.load(std::memory_order_acquire) && ptr) {
    realloc_calls.fetch_add(1, std::memory_order_relaxed);
    bytes_requested.fetch_add(size, std::memory_order_relaxed);
  }
  return ptr;
}
extern "C" void free(void *ptr) noexcept
{
  if (active.load(std::memory_order_acquire) && ptr) {
    frees.fetch_add(1, std::memory_order_relaxed);
  }
  __libc_free(ptr);
}
