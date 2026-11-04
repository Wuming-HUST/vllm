// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <torch/custom_class.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
  #include <dlfcn.h>
#endif

#include "engram_cpu_kernels.h"

#if defined(__linux__) && defined(__aarch64__)
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
#endif

namespace {
using at::Tensor;

bool has_sve2() {
#ifdef ENGRAM_CPU_SVE2
  return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0 &&
         (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0;
#else
  return false;
#endif
}

bool has_neon() {
#ifdef ENGRAM_CPU_NEON
  #if defined(__linux__)
  return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
  #else
  return true;
  #endif
#else
  return false;
#endif
}

int64_t sve_vector_bytes() {
#ifdef ENGRAM_CPU_SVE2
  if (has_sve2()) return engram_sve_vector_bytes();
#endif
  return 0;
}

void check_tensor(const Tensor& t, at::ScalarType dtype, int64_t ndim,
                  const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(t.scalar_type() == dtype && t.dim() == ndim && t.is_contiguous(),
              name, " has invalid dtype, rank, or strides");
}

void check_indices(const Tensor& t, int64_t ndim, const char* name) {
  TORCH_CHECK(t.device().is_cpu() && t.dim() == ndim && t.is_contiguous() &&
                  (t.scalar_type() == at::kInt || t.scalar_type() == at::kLong),
              name, " must be contiguous CPU int32/int64");
}

int64_t index_at(const Tensor& t, int64_t i) {
  return t.scalar_type() == at::kInt ? t.const_data_ptr<int32_t>()[i]
                                     : t.const_data_ptr<int64_t>()[i];
}

void check_output(const Tensor& out, at::IntArrayRef shape,
                  at::TensorList inputs) {
  TORCH_CHECK(out.sizes() == shape, "output shape mismatch");
  at::assert_no_internal_overlap(out);
  for (const auto& input : inputs) at::assert_no_overlap(out, input);
}

// Stateless: the caller supplies the authoritative raw-token lookback for
// every request. No speculative suffix is committed by this operator.
void sources_out(const Tensor& ids, const Tensor& query,
                 const Tensor& positions, const Tensor& lookback,
                 const Tensor& dead, const Tensor& lookback_dead,
                 const Tensor& token_map, int64_t pad_id, Tensor out,
                 const std::optional<Tensor>& cache,
                 const std::optional<Tensor>& slots,
                 const std::optional<Tensor>& blocks, int64_t block_size) {
  check_indices(ids, 1, "ids");
  check_indices(query, 1, "query_start_loc");
  check_indices(positions, 1, "positions");
  check_indices(lookback, 2, "lookback");
  check_tensor(dead, at::kBool, 1, "dead_mask");
  check_tensor(lookback_dead, at::kBool, 2, "lookback_dead_mask");
  check_tensor(token_map, at::kInt, 1, "token_map");
  check_tensor(out, at::kInt, 2, "out");
  const auto tokens = ids.numel(), reqs = lookback.size(0);
  const auto depth = lookback.size(1), width = depth + 1;
  TORCH_CHECK(depth > 0 && query.numel() == reqs + 1 &&
                  positions.numel() == tokens && dead.numel() == tokens &&
                  lookback_dead.sizes() == lookback.sizes(),
              "inconsistent request metadata");
  TORCH_CHECK(pad_id >= 0 && pad_id <= INT32_MAX,
              "pad_id must be a compressed int32 id");
  check_output(
      out, {tokens, width},
      {ids, query, positions, lookback, dead, lookback_dead, token_map});
  TORCH_CHECK(cache.has_value() == slots.has_value() &&
                  cache.has_value() == blocks.has_value(),
              "slot cache metadata must be supplied together");
  if (cache) {
    check_tensor(*cache, at::kInt, 1, "cache");
    check_indices(*slots, 1, "slot_mapping");
    check_indices(*blocks, 2, "block_table");
    TORCH_CHECK(block_size > 0 && slots->numel() == tokens &&
                    blocks->size(0) >= reqs && blocks->size(1) > 0 &&
                    cache->numel() > 0,
                "invalid slot cache metadata");
    check_output(out, {tokens, width}, {*cache, *slots, *blocks});
    check_output(*cache, cache->sizes(),
                 {ids, query, positions, lookback, dead, lookback_dead,
                  token_map, *slots, *blocks});
  }
  TORCH_CHECK(index_at(query, 0) == 0 && index_at(query, reqs) == tokens,
              "query_start_loc must cover exactly the input tokens");
  const auto* lut = token_map.const_data_ptr<int32_t>();
  const auto* mask = dead.const_data_ptr<bool>();
  const auto* old_mask = lookback_dead.const_data_ptr<bool>();
  auto* dst = out.mutable_data_ptr<int32_t>();
  auto* cached = cache ? cache->mutable_data_ptr<int32_t>() : nullptr;
  // Match the device protocol: finish all current-slot writes before any
  // request reads fallback history. Rebinding the KV storage resets this cache.
  if (cache) {
    for (int64_t t = 0; t < tokens; ++t) {
      const auto slot = index_at(*slots, t);
      if (slot < 0) continue;
      TORCH_CHECK(slot < cache->numel(), "slot outside cache");
      const auto raw = index_at(ids, t);
      TORCH_CHECK(mask[t] || (raw >= 0 && raw < token_map.numel()),
                  "token id outside token_map");
      cached[slot] = mask[t] ? -1 : lut[raw];
    }
  }
  at::parallel_for(0, reqs, 1, [&](int64_t first, int64_t last) {
    for (int64_t r = first; r < last; ++r) {
      const auto begin = index_at(query, r), end = index_at(query, r + 1);
      TORCH_CHECK(0 <= begin && begin <= end && end <= tokens,
                  "query_start_loc must be monotonic and in bounds");
      for (int64_t t = begin; t < end; ++t) {
        const auto pos = index_at(positions, t);
        TORCH_CHECK(pos >= 0 && pos - index_at(positions, begin) == t - begin,
                    "positions must be nonnegative and contiguous per request");
        for (int64_t shift = 0; shift < width; ++shift) {
          int32_t value;
          if (shift > pos) {
            value = -1;
          } else {
            const bool current = shift <= t - begin;
            const auto col = shift - (t - begin) - 1;
            const auto i = current ? t - shift : r * depth + col;
            const auto raw = current ? index_at(ids, i) : index_at(lookback, i);
            const bool masked = current ? mask[i] : old_mask[i];
            if (masked) {
              value = -1;
            } else if (!current && raw == -1) {
              value = static_cast<int32_t>(pad_id);
              if (cache) {
                const auto previous = std::clamp(
                    pos - shift, int64_t{0}, blocks->size(1) * block_size - 1);
                const auto block = index_at(
                    *blocks, r * blocks->size(1) + previous / block_size);
                const auto slot =
                    std::clamp(block * block_size + previous % block_size,
                               int64_t{0}, cache->numel() - 1);
                value = cached[slot];
              }
            } else {
              TORCH_CHECK(raw >= 0 && raw < token_map.numel(),
                          "token id outside token_map");
              value = lut[raw];
              TORCH_CHECK(value >= 0, "token_map must contain nonnegative ids");
            }
          }
          dst[t * width + shift] = value;
        }
      }
    }
  });
}

void hash_out(const Tensor& sources, const Tensor& multipliers,
              const Tensor& primes, const Tensor& offsets, int64_t pad_id,
              Tensor out, bool use_sve2) {
  TORCH_CHECK(!use_sve2 || has_sve2(), "SVE2 hash is unavailable on this host");
  check_tensor(sources, at::kInt, 2, "sources");
  check_tensor(multipliers, at::kLong, 2, "multipliers");
  check_tensor(primes, at::kLong, 2, "primes");
  check_tensor(offsets, at::kLong, 2, "offsets");
  check_tensor(out, at::kInt, 3, "out");
  const auto tokens = sources.size(0), width = sources.size(1);
  const auto layers = multipliers.size(0), cols = primes.size(1);
  TORCH_CHECK(width >= 2 && layers > 0 && cols > 0 && cols % (width - 1) == 0 &&
                  multipliers.size(1) == width && primes.size(0) == layers &&
                  offsets.sizes() == primes.sizes(),
              "invalid hash layout");
  TORCH_CHECK(pad_id >= 0 && pad_id <= INT32_MAX, "invalid compressed pad_id");
  check_output(out, {tokens, layers, cols},
               {sources, multipliers, primes, offsets});
  const auto* src = sources.const_data_ptr<int32_t>();
  const auto* mult = multipliers.const_data_ptr<int64_t>();
  const auto* mod = primes.const_data_ptr<int64_t>();
  const auto* off = offsets.const_data_ptr<int64_t>();
  auto* dst = out.mutable_data_ptr<int32_t>();
  const auto heads = cols / (width - 1);
  std::vector<uint64_t> reciprocals(primes.numel());
  for (int64_t j = 0; j < primes.numel(); ++j) {
    TORCH_CHECK(mod[j] > 1 && mod[j] <= INT32_MAX && off[j] >= 0 &&
                    off[j] <= INT32_MAX - (mod[j] - 1),
                "hash bucket range does not fit int32");
    reciprocals[j] =
        std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(mod[j]);
  }
  at::parallel_for(0, tokens, 32, [&](int64_t first, int64_t last) {
    for (auto t = first; t < last; ++t) {
      for (int64_t l = 0; l < layers; ++l) {
        uint64_t rolling = 0;
        bool blocked = false;
        for (int64_t s = 0; s < width; ++s) {
          const auto source = src[t * width + s];
          TORCH_CHECK(source >= -1, "invalid source sentinel");
          blocked |= source == -1;
          const auto value = static_cast<uint64_t>(blocked ? pad_id : source);
          rolling ^= value * static_cast<uint64_t>(mult[l * width + s]);
          if (s == 0) continue;
#ifdef ENGRAM_CPU_SVE2
          if (use_sve2) {
            const auto j = l * cols + (s - 1) * heads;
            engram_mod_sve2(rolling, mod + j, off + j, reciprocals.data() + j,
                            heads,
                            dst + (t * layers + l) * cols + (s - 1) * heads);
            continue;
          }
#endif
          for (int64_t h = 0; h < heads; ++h) {
            const auto col = (s - 1) * heads + h, j = l * cols + col;
            const auto prime = static_cast<uint64_t>(mod[j]);
            const auto high = static_cast<uint64_t>(
                (static_cast<__uint128_t>(rolling) * reciprocals[j]) >> 64);
            auto bucket = rolling - high * prime;
            if (bucket >= prime) bucket -= prime;
            dst[(t * layers + l) * cols + col] =
                static_cast<int32_t>(bucket + static_cast<uint64_t>(off[j]));
          }
        }
      }
    }
  });
}

float decode_fp8(uint8_t code) {
  const auto exp = (code >> 3) & 15, frac = code & 7;
  const auto value = exp == 15 && frac == 7
                         ? std::numeric_limits<float>::quiet_NaN()
                     : exp ? std::ldexp(1.0f + frac / 8.0f, exp - 7)
                           : std::ldexp(static_cast<float>(frac), -9);
  return (code & 128) ? -value : value;
}

const std::array<float, 256> fp8_values = [] {
  std::array<float, 256> values{};
  for (size_t c = 0; c < values.size(); ++c) values[c] = decode_fp8(c);
  return values;
}();

// Output owns only this shard's heads; invalid ids and tail heads are zero.
void gather_out(const Tensor& codes, const Tensor& scales, const Tensor& ids,
                int64_t head_start, int64_t valid_heads, int64_t vocab_start,
                int64_t block_size, Tensor out, const std::string& backend,
                int64_t prefetch_distance, bool legacy_e8m0) {
  TORCH_CHECK(backend == "auto" || backend == "scalar" || backend == "neon" ||
                  backend == "sve2",
              "unknown CPU gather backend: ", backend);
  TORCH_CHECK(backend != "neon" || has_neon(), "NEON gather is unavailable");
  TORCH_CHECK(backend != "sve2" || has_sve2(), "SVE2 gather is unavailable");
  TORCH_CHECK(prefetch_distance >= 0, "prefetch_distance must be nonnegative");
  using Decode =
      void (*)(const uint8_t*, uint16_t*, int64_t, float, bool, const float*);
  Decode decode = nullptr;
#ifdef ENGRAM_CPU_NEON
  if (backend == "neon" || (backend == "auto" && has_neon()))
    decode = &engram_decode_neon;
#endif
#ifdef ENGRAM_CPU_SVE2
  if (backend == "sve2") decode = &engram_decode_sve2;
#endif
  TORCH_CHECK(
      codes.device().is_cpu() && codes.dim() == 2 && codes.is_contiguous(),
      "codes must be a contiguous CPU matrix");
  check_tensor(ids, at::kInt, 2, "ids");
  check_tensor(out, at::kBFloat16, 3, "out");
  const bool fp8 = codes.scalar_type() == at::kFloat8_e4m3fn;
  const bool int8 = codes.scalar_type() == at::kChar;
  const bool bf16 = codes.scalar_type() == at::kBFloat16;
  TORCH_CHECK(fp8 || int8 || bf16, "unsupported table dtype");
  const auto tokens = ids.size(0), cols = ids.size(1);
  const auto rows = codes.size(0), dim = codes.size(1);
  TORCH_CHECK(dim > 0 && block_size > 0 && dim % block_size == 0,
              "invalid block_size or embedding dimension");
  TORCH_CHECK(head_start >= 0 && head_start <= cols && valid_heads >= 0 &&
                  valid_heads <= cols - head_start &&
                  out.size(1) >= valid_heads && vocab_start >= 0 &&
                  vocab_start <= INT32_MAX && rows <= INT32_MAX - vocab_start,
              "invalid shard range");
  if (bf16) {
    TORCH_CHECK(scales.device().is_cpu() && scales.numel() == 0,
                "BF16 tables require empty CPU scales");
  } else {
    check_tensor(scales, fp8 ? at::kByte : at::kFloat, 2, "scales");
    TORCH_CHECK(scales.size(0) == rows && scales.size(1) == dim / block_size,
                "scale shape mismatch");
  }
  check_output(out, {tokens, out.size(1), dim}, {codes, scales, ids});
  const auto* ix = ids.const_data_ptr<int32_t>();
  auto* dst = out.mutable_data_ptr<at::BFloat16>();
  const auto padded_heads = out.size(1);
  const auto* bytes = fp8 || int8
                          ? static_cast<const uint8_t*>(codes.const_data_ptr())
                          : nullptr;
  const auto* values = bf16 ? codes.const_data_ptr<at::BFloat16>() : nullptr;
  const auto* sf8 = fp8 ? scales.const_data_ptr<uint8_t>() : nullptr;
  const auto* sf32 = int8 ? scales.const_data_ptr<float>() : nullptr;
  at::parallel_for(
      0, tokens * padded_heads, 32, [&](int64_t first, int64_t last) {
        for (auto k = first; k < last; ++k) {
          if (prefetch_distance > 0 &&
              prefetch_distance < tokens * padded_heads - k) {
            const auto next = k + prefetch_distance;
            const auto nh = next % padded_heads;
            if (nh < valid_heads) {
              const auto nr =
                  static_cast<int64_t>(
                      ix[(next / padded_heads) * cols + head_start + nh]) -
                  vocab_start;
              if (nr >= 0 && nr < rows) {
                const auto* ptr =
                    bf16 ? reinterpret_cast<const uint8_t*>(values + nr * dim)
                         : bytes + nr * dim;
                const auto row_bytes = dim * (bf16 ? 2 : 1);
                for (int64_t c = 0; c < row_bytes; c += 64)
                  __builtin_prefetch(ptr + c, 0, 1);
                if (fp8)
                  __builtin_prefetch(sf8 + nr * (dim / block_size), 0, 1);
                else if (int8)
                  __builtin_prefetch(sf32 + nr * (dim / block_size), 0, 1);
              }
            }
          }
          const auto t = k / padded_heads, h = k % padded_heads;
          const auto row =
              h < valid_heads
                  ? static_cast<int64_t>(ix[t * cols + head_start + h]) -
                        vocab_start
                  : -1;
          auto* target = dst + k * dim;
          if (row < 0 || row >= rows) {
            std::fill_n(target, dim, at::BFloat16(0.0f));
          } else if (bf16) {
            std::copy_n(values + row * dim, dim, target);
          } else {
            for (int64_t g = 0; g < dim / block_size; ++g) {
              const auto scale_idx = row * (dim / block_size) + g;
              const float scale =
                  fp8 ? legacy_e8m0 && sf8[scale_idx] == 0 ? 0.0f
                        : sf8[scale_idx] == 255
                            ? (legacy_e8m0
                                   ? std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::quiet_NaN())
                            : std::ldexp(1.0f, sf8[scale_idx] - 127)
                      : sf32[scale_idx];
              if (decode != nullptr) {
                decode(bytes + row * dim + g * block_size,
                       reinterpret_cast<uint16_t*>(target + g * block_size),
                       block_size, scale, fp8, fp8_values.data());
                continue;
              }
              for (int64_t c = 0; c < block_size; ++c) {
                const auto i = g * block_size + c;
                const auto code = bytes[row * dim + i];
                const float value =
                    fp8 ? fp8_values[code]
                        : static_cast<float>(static_cast<int8_t>(code));
                target[i] = at::BFloat16(value * scale);
              }
            }
          }
        }
      });
}

class CPUCallback : public torch::CustomClassHolder {
 public:
  explicit CPUCallback(std::function<void()> work) : work_(std::move(work)) {}

  void run() noexcept {
    try {
      work_();
    } catch (...) {
      failure_ = std::current_exception();
    }
    std::lock_guard<std::mutex> guard(mutex_);
    done_ = true;
    ready_.notify_all();
  }

  void wait() {
    std::unique_lock<std::mutex> guard(mutex_);
    ready_.wait(guard, [&] { return done_; });
    if (failure_) std::rethrow_exception(failure_);
  }

 private:
  // Tensor owners are released by the worker after wait(), outside the
  // runtime callback. Pinned allocation/free must never call device APIs here.
  std::function<void()> work_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool done_ = false;
  std::exception_ptr failure_;
};

using LaunchHost = int (*)(void*, void (*)(void*), void*);

LaunchHost callback_launcher(const std::string& device) {
#if defined(__linux__)
  if (device == "npu") {
    static void* library = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    return library ? reinterpret_cast<LaunchHost>(
                         dlsym(library, "aclrtLaunchHostFunc"))
                   : nullptr;
  }
  if (device == "cuda") {
    static void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    return library ? reinterpret_cast<LaunchHost>(
                         dlsym(library, "cuLaunchHostFunc"))
                   : nullptr;
  }
#endif
  return nullptr;
}

bool callback_available(const std::string& device) {
  return callback_launcher(device) != nullptr;
}

c10::intrusive_ptr<CPUCallback> enqueue_gather_out(
    const Tensor& codes, const Tensor& scales, const Tensor& ids,
    int64_t head_start, int64_t valid_heads, int64_t vocab_start,
    int64_t block_size, Tensor out, const std::string& backend,
    int64_t prefetch_distance, bool legacy_e8m0, int64_t stream,
    const std::string& device) {
  TORCH_CHECK(codes.device().is_cpu() && scales.device().is_cpu(),
              "callback table storage must be on CPU");
  check_tensor(ids, at::kInt, 2, "ids");
  check_tensor(out, at::kBFloat16, 3, "out");
  // All tensors are retained by the ticket through H2D retirement. Only CPU
  // operations run in the callback, with no Python/GIL or accelerator APIs.
  auto job = c10::make_intrusive<CPUCallback>([=]() mutable {
    try {
      gather_out(codes, scales, ids, head_start, valid_heads, vocab_start,
                 block_size, out, backend, prefetch_distance, legacy_e8m0);
    } catch (...) {
      out.zero_();
      throw;
    }
  });
  if (device == "cpu") {
    job->run();  // Host-only validation of computation/error propagation.
    return job;
  }
  auto launch = callback_launcher(device);
  TORCH_CHECK(launch != nullptr, "Engram host callback unavailable for ",
              device);
  using Owner = c10::intrusive_ptr<CPUCallback>;
  auto owner = std::make_unique<Owner>(job);
  const auto status = launch(
      reinterpret_cast<void*>(stream),
      [](void* data) {
        std::unique_ptr<Owner> retained(static_cast<Owner*>(data));
        (*retained)->run();
      },
      owner.get());
  TORCH_CHECK(status == 0, "Engram host callback launch failed: ", status);
  owner.release();
  return job;
}
}  // namespace

TORCH_LIBRARY(_engram_cpu, m) {
  m.def("has_sve2() -> bool", &has_sve2);
  m.def("has_neon() -> bool", &has_neon);
  m.def("sve_vector_bytes() -> int", &sve_vector_bytes);
  m.def("callback_available(str device) -> bool", &callback_available);
  m.class_<CPUCallback>("Callback").def("wait", &CPUCallback::wait);
  m.def(
      "sources_out(Tensor ids, Tensor query_start_loc, Tensor positions, "
      "Tensor lookback, Tensor dead_mask, Tensor lookback_dead_mask, "
      "Tensor token_map, int pad_id, Tensor(a!) out, Tensor(b!)? cache=None, "
      "Tensor? slot_mapping=None, Tensor? block_table=None, "
      "int cache_block_size=0) -> ()");
  m.def(
      "hash_out(Tensor sources, Tensor multipliers, Tensor primes, "
      "Tensor offsets, int pad_id, Tensor(a!) out, bool use_sve2=False) -> ()");
  m.def(
      "gather_out(Tensor codes, Tensor scales, Tensor ids, int head_start, "
      "int valid_heads, int vocab_start, int block_size, "
      "Tensor(a!) out, str backend='auto', int prefetch_distance=0, "
      "bool legacy_e8m0=False) -> ()");
  m.def("enqueue_gather_out", &enqueue_gather_out);
}

TORCH_LIBRARY_IMPL(_engram_cpu, CPU, m) {
  m.impl("sources_out", &sources_out);
  m.impl("hash_out", &hash_out);
  m.impl("gather_out", &gather_out);
}

// This is a host-side, pre-graph API. Deliberately no Meta kernel: capturing
// CPU retrieval inside a device graph would replay stale results.
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "engram_cpu_C", nullptr, -1,
                             nullptr};
PyMODINIT_FUNC PyInit_engram_cpu_C() { return PyModule_Create(&module); }
