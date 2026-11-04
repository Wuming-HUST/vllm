// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <arm_neon.h>

#include "engram_cpu_kernels.h"

namespace {
// Match ATen's software round-to-nearest-even, including canonical NaN.
uint16x4_t bf16_bits(float32x4_t value) {
  const auto bits = vreinterpretq_u32_f32(value);
  const auto bias = vaddq_u32(vdupq_n_u32(0x7fff),
                              vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1)));
  auto result = vshrq_n_u32(vaddq_u32(bits, bias), 16);
  result = vbslq_u32(vmvnq_u32(vceqq_f32(value, value)), vdupq_n_u32(0x7fc0),
                     result);
  return vmovn_u32(result);
}

float32x4_t decode_fp8(uint32x4_t code) {
  const auto exponent = vandq_u32(vshrq_n_u32(code, 3), vdupq_n_u32(15));
  const auto fraction = vandq_u32(code, vdupq_n_u32(7));
  auto bits = vorrq_u32(vshlq_n_u32(vaddq_u32(exponent, vdupq_n_u32(120)), 23),
                        vshlq_n_u32(fraction, 20));
  const auto subnormal =
      vreinterpretq_u32_f32(vmulq_n_f32(vcvtq_f32_u32(fraction), 0x1p-9f));
  bits = vbslq_u32(vceqq_u32(exponent, vdupq_n_u32(0)), subnormal, bits);
  bits = vorrq_u32(bits, vshlq_n_u32(vandq_u32(code, vdupq_n_u32(128)), 24));
  const auto nan =
      vceqq_u32(vandq_u32(code, vdupq_n_u32(127)), vdupq_n_u32(127));
  return vreinterpretq_f32_u32(vbslq_u32(nan, vdupq_n_u32(0x7fc00000), bits));
}
}  // namespace

void engram_decode_neon(const uint8_t* codes, uint16_t* out, int64_t count,
                        float scale, bool fp8, const float* fp8_values) {
  int64_t c = 0;
  for (; c + 8 <= count; c += 8) {
    const auto packed = vld1_u8(codes + c);
    float32x4_t low, high;
    if (fp8) {
      const auto widened = vmovl_u8(packed);
      low = decode_fp8(vmovl_u16(vget_low_u16(widened)));
      high = decode_fp8(vmovl_u16(vget_high_u16(widened)));
    } else {
      const auto widened = vmovl_s8(vreinterpret_s8_u8(packed));
      low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(widened)));
      high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(widened)));
    }
    vst1_u16(out + c, bf16_bits(vmulq_n_f32(low, scale)));
    vst1_u16(out + c + 4, bf16_bits(vmulq_n_f32(high, scale)));
  }
  // The tail may be shorter than a vector; never overread a table row.
  for (; c < count; ++c) {
    const float value =
        (fp8 ? fp8_values[codes[c]]
             : static_cast<float>(static_cast<int8_t>(codes[c]))) *
        scale;
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    out[c] =
        value != value ? 0x7fc0 : (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  }
}
