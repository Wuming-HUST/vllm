// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <stdint.h>

// Raw kernels have no Torch dependency, so each ISA can be cross compiled.
// count bytes and count BF16 words must be valid, contiguous and disjoint.
void engram_decode_neon(const uint8_t* codes, uint16_t* out, int64_t count,
                        float scale, bool fp8, const float* fp8_values);
void engram_decode_sve2(const uint8_t* codes, uint16_t* out, int64_t count,
                        float scale, bool fp8, const float* fp8_values);
void engram_mod_sve2(uint64_t rolling, const int64_t* primes,
                     const int64_t* offsets, const uint64_t* reciprocals,
                     int64_t heads, int32_t* out);
int64_t engram_sve_vector_bytes();
