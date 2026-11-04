# Native Engram CPU operators

`lookup_backend=cpu` selects CPU vocabulary conversion, hash and embedding
retrieval in DeepSeek V4.1. CPU preparation stays outside device capture;
existing device graphs consume refreshed buffers and per-layer readiness.
See the [serving guide](engram_cpu_serving.md) for ownership, graph/parallel
contracts, rollback and pending hardware validation. Device remains the default.

## Source and hash contracts

`sources_out` accepts flattened raw IDs and positions `[T]`, ragged offsets
`query_start_loc[R+1]`, and raw lookback IDs `[R,K-1]`, newest first. Index inputs
may be INT32 or INT64. It also accepts BOOL image masks for current and lookback
IDs, an INT32 compression map, and a compressed pad ID. Positions must be
nonnegative and contiguous within each request. The output is INT32 `[T,K]`.

An unknown lookback ID `-1` at a nonnegative position contributes pad. A position
before the sequence or an image mask contributes DEAD `-1`. Hashing replaces
DEAD and all older sources with pad. The API owns no persistent request state;
the caller supplies authoritative history, including accepted speculative tokens.

`hash_out` takes INT32 sources `[T,K]`, INT64 multipliers `[L,K]`, and INT64
primes and offsets `[L,(K-1)*H]`. It writes INT32 global bucket IDs with shape
`[T,L,(K-1)*H]`, ordered by n-gram order then head. Mixing uses uint64 wraparound.
Each bucket range must fit INT32. Select a layer's contiguous slice before
calling `gather_out`; a view across multiple layers generally needs a copy.

The portable hash path is the default. Linux AArch64 builds add a separate SVE2
translation unit when the compiler supports it; runtime HWCAP2 guards execution.
`use_sve2=True` requires `torch.ops._engram_cpu.has_sve2()` to return true.
This feature check does not measure which implementation is faster.

## Table and shard contracts

`gather_out` consumes global INT32 IDs `[T,C]` and a local table `[rows,dim]`.
`head_start` selects input columns; `valid_heads` selects their count.
`vocab_start` is the global row offset of the local table. The BF16 output is
`[T,padded_heads,dim]`, with this shard's heads starting at output head zero.
IDs outside the local vocabulary and padded output heads are zeroed.

Supported storage formats are:

| Codes | Scales | Scale shape |
| --- | --- | --- |
| E4M3FN | UINT8 E8M0 | `[rows,dim/block_size]` |
| INT8 | FP32 | `[rows,dim/block_size]` |
| BF16 | Empty CPU tensor | No scales |

`backend` accepts `auto`, `scalar`, `neon`, and `sve2`. Auto selects NEON when
compiled and usable, otherwise the portable C++ loop. SVE2 gather is explicit
until target measurements select it. Both vector paths implement software BF16
round-to-nearest-even, without requiring BF16 conversion instructions.
`prefetch_distance` counts future output head rows; zero disables prefetch.
Positive values issue bounded read hints for valid local rows only.
`capabilities()` reports compiled-and-usable NEON/SVE2 and the calling thread's
SVE vector length. Unsupported explicit backends raise an error.

The embedding dimension must be divisible by the positive block size. Gather,
dequantization and BF16 conversion write directly to the output. E8M0 byte zero
means `2^-127`; byte 255 means NaN, consistent with PyTorch and the paired Ascend
decoder. The v0.30.0 NVIDIA decoder uses exponent-bit reinterpretation and differs
on those two values. Device parity requires resolving that difference before
enabling a serving backend.

The operators perform no collectives. TP, DP, EDP and sequence parallelism remain
the caller's responsibility. They neither publish device ready events nor manage
buffer retirement. They have no Meta implementations: execute the host producer
outside device graph capture and refresh its output before each replay.

## Building and checking the host target

Use the [incremental build workflow](../incremental_build.md) with the runtime's
PyTorch version, then build and install only `engram_cpu_C`:

```bash
.venv/bin/cmake --build build/engram-cpu --target engram_cpu_C -j 2
.venv/bin/cmake --install build/engram-cpu --component engram_cpu_C
.venv/bin/python -m pytest --noconftest tests/kernels/test_engram_cpu.py -q
```

For an isolated host configuration, set `VLLM_TARGET_DEVICE=empty`,
`VLLM_PYTHON_EXECUTABLE` and `CMAKE_INSTALL_PREFIX` when configuring CMake.
An empty-target setup.py wheel does not build this extension automatically;
install the CMake component explicitly. Existing precompiled release wheels
also lack this new component. The Python API reports a missing-extension error
instead of compiling at runtime.

The isolated tests check native Python import, source/history semantics, unsigned
hash arithmetic, all 256 FP8 codes, special scales, and reconstruction from local
TP1/2/4/8 shards. They do not establish distributed collectives, model accuracy,
device graph correctness or performance. Those require the paired runtime and
target hardware.

See the [Kunpeng offline validation guide](engram_cpu_kunpeng.md) for the
920/950 build boundary, numerical checks and controlled microbenchmark commands.

The gather option `legacy_e8m0=True` preserves the paired NVIDIA tag's
scale-byte 0/255 interpretation (zero/infinity). Its default is native E8M0
(2^-127/NaN), as used by Ascend. Model adapters choose this automatically.
