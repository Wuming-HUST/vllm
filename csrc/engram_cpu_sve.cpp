// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <arm_sve.h>

#include "engram_cpu_kernels.h"

int64_t engram_sve_vector_bytes() { return svcntb(); }

void engram_decode_sve2(const uint8_t* codes, uint16_t* out, int64_t count,
                        float scale, bool fp8, const float* fp8_values) {
  for (uint64_t c = 0; c < static_cast<uint64_t>(count); c += svcntw()) {
    const auto pg = svwhilelt_b32(c, static_cast<uint64_t>(count));
    // FP8 indices are zero-extended bytes, hence always inside [0, 255].
    const auto value =
        fp8 ? svld1_gather_u32index_f32(pg, fp8_values,
                                        svld1ub_u32(pg, codes + c))
            : svcvt_f32_s32_x(
                  pg,
                  svld1sb_s32(pg, reinterpret_cast<const int8_t*>(codes) + c));
    const auto scaled = svmul_n_f32_x(pg, value, scale);
    const auto bits = svreinterpret_u32_f32(scaled);
    const auto bias = svadd_n_u32_x(
        pg, svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1), 0x7fff);
    auto rounded = svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, bias), 16);
    rounded =
        svsel_u32(svcmpne_f32(pg, scaled, scaled), svdup_u32(0x7fc0), rounded);
    svst1h_u32(pg, out + c, rounded);
  }
}

// Compiled separately: merely loading the host extension must not require SVE2.
void engram_mod_sve2(uint64_t rolling, const int64_t* primes,
                     const int64_t* offsets, const uint64_t* reciprocals,
                     int64_t heads, int32_t* out) {
  for (uint64_t h = 0; h < static_cast<uint64_t>(heads); h += svcntd()) {
    const auto pg = svwhilelt_b64(h, static_cast<uint64_t>(heads));
    const auto p = svld1_u64(pg, reinterpret_cast<const uint64_t*>(primes) + h);
    const auto q = svld1_u64(pg, reciprocals + h);
    const auto value = svdup_u64(rolling);
    const auto high = svmulh_u64_x(pg, value, q);
    auto rem = svsub_u64_x(pg, value, svmul_u64_x(pg, high, p));
    const auto correction = svcmpge_u64(pg, rem, p);
    rem = svsub_u64_x(pg, rem, svsel_u64(correction, p, svdup_u64(0)));
    const auto off =
        svld1_u64(pg, reinterpret_cast<const uint64_t*>(offsets) + h);
    svst1w_u64(pg, reinterpret_cast<uint32_t*>(out) + h,
               svadd_u64_x(pg, rem, off));
  }
}
